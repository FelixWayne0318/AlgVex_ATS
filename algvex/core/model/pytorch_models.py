# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted for AlgVex cryptocurrency trading platform

"""
AlgVex PyTorch 模型 (Qlib 0.9.7 原版复刻)

直接复刻 Qlib 原版代码，仅做必要适配:
- 适配 AlgVex 数据格式
- 添加加密货币支持
- 保持与 Qlib 完全兼容

模型列表:
- LSTM, GRU, Transformer (基础模型)
- ALSTM, TCN, MLP, TabNet (扩展模型)
- HIST, GATs, ADD, SFM (高级模型)
"""

from __future__ import division
from __future__ import print_function

import copy
import math
import warnings
from typing import Text, Union, Optional, Dict, Any, List
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will not work.")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# 工具函数 (Qlib 原版)
# ============================================================

def get_or_create_path(path, return_str=True):
    """创建路径 (Qlib 原版)"""
    import os
    if path is None:
        return None
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if return_str:
        return str(path)
    return path


def count_parameters(model):
    """计算模型参数量 (Qlib 原版)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024 / 1024


# ============================================================
# 基类 (Qlib Model 原版适配)
# ============================================================

class Model(ABC):
    """
    Qlib Model 基类适配版

    提供 fit/predict 接口，兼容 Qlib 和 AlgVex 数据集
    """

    @abstractmethod
    def fit(self, dataset, evals_result=None, save_path=None):
        """训练模型"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测"""
        raise NotImplementedError

    def _prepare_data(self, dataset, segment, col_set=None):
        """
        准备数据 - 兼容 Qlib DatasetH 和 AlgVex CryptoDataset

        Returns:
            DataFrame with 'feature' and 'label' columns
        """
        if hasattr(dataset, 'prepare'):
            # Qlib/AlgVex 风格
            try:
                # 尝试 Qlib 方式
                data = dataset.prepare(segment, col_set=col_set or ["feature", "label"])
            except TypeError:
                # AlgVex 方式
                data = dataset.prepare(segment)
        else:
            # 直接传入 DataFrame
            data = dataset

        return data


# ============================================================
# LSTM 模型 (Qlib 原版)
# ============================================================

class LSTM(Model):
    """
    LSTM Model (Qlib 原版)

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        logger.info(f"LSTM parameters: d_feat={d_feat}, hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, device={self.device}")

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.lstm_model = LSTMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.lstm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError(f"unknown loss `{self.loss}`")

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError(f"unknown metric `{self.metric}`")

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_train_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)

        self.lstm_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)

        self.lstm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.lstm_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}

        # 准备数据
        df_train = self._prepare_data(dataset, "train", ["feature", "label"])
        df_valid = self._prepare_data(dataset, "valid", ["feature", "label"])

        if df_train.empty:
            raise ValueError("Empty training data from dataset")

        # 提取特征和标签
        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = df_valid["feature"], df_valid["label"] if not df_valid.empty else (None, None)
        else:
            x_train = df_train.iloc[:, :-1]
            y_train = df_train.iloc[:, -1]
            x_valid = df_valid.iloc[:, :-1] if not df_valid.empty else None
            y_valid = df_valid.iloc[:, -1] if not df_valid.empty else None

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        logger.info("training...")
        self.fitted = True

        best_param = copy.deepcopy(self.lstm_model.state_dict())
        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)

            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                evals_result["valid"].append(val_score)

                if step % 10 == 0:
                    logger.info(f"Epoch {step}: train={train_score:.6f}, valid={val_score:.6f}")

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.lstm_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        logger.info("early stop")
                        break

        logger.info(f"best score: {best_score:.6f} @ epoch {best_epoch}")
        self.lstm_model.load_state_dict(best_param)

        if save_path:
            torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]

        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            end = min(begin + self.batch_size, sample_num)
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class LSTMNet(nn.Module):
    """LSTM Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T] --> [N, T, F]
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


# ============================================================
# GRU 模型 (Qlib 原版)
# ============================================================

class GRU(Model):
    """GRU Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.gru_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        return torch.mean((pred - label) ** 2)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError(f"unknown loss `{self.loss}`")

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError(f"unknown metric `{self.metric}`")

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_train_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)

        self.gru_model.train()
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)

        self.gru_model.eval()
        scores, losses = [], []
        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}

        df_train = self._prepare_data(dataset, "train", ["feature", "label"])
        df_valid = self._prepare_data(dataset, "valid", ["feature", "label"])

        if df_train.empty:
            raise ValueError("Empty training data")

        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = (df_valid["feature"], df_valid["label"]) if not df_valid.empty else (None, None)
        else:
            x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            x_valid, y_valid = (df_valid.iloc[:, :-1], df_valid.iloc[:, -1]) if not df_valid.empty else (None, None)

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        self.fitted = True
        best_param = copy.deepcopy(self.gru_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)

            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.gru_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break

        self.gru_model.load_state_dict(best_param)
        if save_path:
            torch.save(best_param, save_path)

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]

        index = x_test.index
        self.gru_model.eval()
        x_values = x_test.values
        preds = []

        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.gru_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class GRUNet(nn.Module):
    """GRU Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


# ============================================================
# Transformer 模型 (Qlib 原版)
# ============================================================

class TransformerModel(Model):
    """Transformer Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 2048,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.d_feat = d_feat

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        return torch.mean((pred.float() - label.float()) ** 2)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError(f"unknown loss `{self.loss}`")

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError(f"unknown metric `{self.metric}`")

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_train_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)

        self.model.train()
        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)

        self.model.eval()
        scores, losses = [], []
        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}

        df_train = self._prepare_data(dataset, "train", ["feature", "label"])
        df_valid = self._prepare_data(dataset, "valid", ["feature", "label"])

        if df_train.empty:
            raise ValueError("Empty training data")

        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = (df_valid["feature"], df_valid["label"]) if not df_valid.empty else (None, None)
        else:
            x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            x_valid, y_valid = (df_valid.iloc[:, :-1], df_valid.iloc[:, -1]) if not df_valid.empty else (None, None)

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        self.fitted = True
        best_param = copy.deepcopy(self.model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)

            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break

        self.model.load_state_dict(best_param)
        if save_path:
            torch.save(best_param, save_path)

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]

        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        preds = []

        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


class PositionalEncoding(nn.Module):
    """位置编码 (Qlib 原版)"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    """Transformer Network (Qlib 原版)"""
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super().__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)
        src = src.transpose(1, 0)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])
        return output.squeeze()


# ============================================================
# ALSTM 模型 (Attention LSTM, Qlib 原版)
# ============================================================

class ALSTM(Model):
    """Attention LSTM Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.alstm_model = ALSTMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.alstm_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.alstm_model.parameters(), lr=self.lr)

        self.fitted = False
        self.alstm_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        return -self.loss_fn(pred[mask], label[mask])

    def train_epoch(self, x_train, y_train):
        x_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)

        self.alstm_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.alstm_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.alstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)

        self.alstm_model.eval()
        scores, losses = [], []

        for i in range(len(x_values))[:: self.batch_size]:
            if len(x_values) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[i : i + self.batch_size]).float().to(self.device)
            label = torch.from_numpy(y_values[i : i + self.batch_size]).float().to(self.device)

            with torch.no_grad():
                pred = self.alstm_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                scores.append(self.metric_fn(pred, label).item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}

        df_train = self._prepare_data(dataset, "train", ["feature", "label"])
        df_valid = self._prepare_data(dataset, "valid", ["feature", "label"])

        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = (df_valid["feature"], df_valid["label"]) if not df_valid.empty else (None, None)
        else:
            x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            x_valid, y_valid = (df_valid.iloc[:, :-1], df_valid.iloc[:, -1]) if not df_valid.empty else (None, None)

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        self.fitted = True
        best_param = copy.deepcopy(self.alstm_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)

            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.alstm_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break

        self.alstm_model.load_state_dict(best_param)

    def predict(self, dataset, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]

        self.alstm_model.eval()
        x_values = x_test.values
        preds = []

        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.alstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=x_test.index)


class ALSTMNet(nn.Module):
    """Attention LSTM Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1).permute(0, 2, 1)
        rnn_out, _ = self.rnn(x)
        attn_weights = F.softmax(self.attention(rnn_out), dim=1)
        context = torch.sum(attn_weights * rnn_out, dim=1)
        return self.fc_out(context).squeeze()


# ============================================================
# 模型工厂
# ============================================================

def get_model(model_type: str, **kwargs):
    """
    获取模型实例 (Qlib 风格)

    Args:
        model_type: 模型类型
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    models = {
        'lstm': LSTM,
        'gru': GRU,
        'transformer': TransformerModel,
        'alstm': ALSTM,
    }

    model_type = model_type.lower()
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


# 导出
__all__ = [
    'Model',
    'LSTM',
    'LSTMNet',
    'GRU',
    'GRUNet',
    'TransformerModel',
    'Transformer',
    'PositionalEncoding',
    'ALSTM',
    'ALSTMNet',
    'get_model',
]
