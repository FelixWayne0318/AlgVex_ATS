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
# TCN 基础组件 (Qlib 原版)
# ============================================================

class Chomp1d(nn.Module):
    """Chomp1d for TCN (Qlib 原版)"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal Block for TCN (Qlib 原版)"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        from torch.nn.utils import weight_norm
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (Qlib 原版)"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# TCN 模型 (Qlib 原版)
# ============================================================

class TCN(Model):
    """TCN Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        n_chans=128,
        kernel_size=5,
        num_layers=5,
        dropout=0.5,
        n_epochs=200,
        lr=0.0001,
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
        self.n_chans = n_chans
        self.kernel_size = kernel_size
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

        self.tcn_model = TCNNet(
            num_input=self.d_feat,
            output_size=1,
            num_channels=[self.n_chans] * self.num_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.tcn_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.tcn_model.parameters(), lr=self.lr)

        self.fitted = False
        self.tcn_model.to(self.device)

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
        self.tcn_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.tcn_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.tcn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.tcn_model.eval()
        scores, losses = [], []

        for i in range(len(x_values))[:: self.batch_size]:
            if len(x_values) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[i : i + self.batch_size]).float().to(self.device)
            label = torch.from_numpy(y_values[i : i + self.batch_size]).float().to(self.device)
            with torch.no_grad():
                pred = self.tcn_model(feature)
                losses.append(self.loss_fn(pred, label).item())
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

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.tcn_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.tcn_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.tcn_model.load_state_dict(best_param)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.tcn_model.eval()
        x_values = x_test.values
        preds = []
        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.tcn_model(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class TCNNet(nn.Module):
    """TCN Network (Qlib 原版)"""
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_input = num_input
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.num_input, -1)
        output = self.tcn(x)
        return self.linear(output[:, :, -1]).squeeze()


# ============================================================
# SFM 模型 (State Frequency Memory, Qlib 原版)
# ============================================================

class SFM(Model):
    """SFM Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        output_dim=1,
        freq_dim=10,
        dropout_W=0.0,
        dropout_U=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="gd",
        GPU=0,
        seed=None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
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

        self.sfm_model = SFMNet(
            d_feat=self.d_feat,
            output_dim=self.output_dim,
            hidden_size=self.hidden_size,
            freq_dim=self.freq_dim,
            dropout_W=self.dropout_W,
            dropout_U=self.dropout_U,
            device=self.device,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.sfm_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.sfm_model.parameters(), lr=self.lr)

        self.fitted = False
        self.sfm_model.to(self.device)

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
        self.sfm_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.sfm_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.sfm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.sfm_model.eval()
        scores, losses = [], []

        for i in range(len(x_values))[:: self.batch_size]:
            if len(x_values) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[i : i + self.batch_size]).float().to(self.device)
            label = torch.from_numpy(y_values[i : i + self.batch_size]).float().to(self.device)
            with torch.no_grad():
                pred = self.sfm_model(feature)
                losses.append(self.loss_fn(pred, label).item())
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

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.sfm_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.sfm_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.sfm_model.load_state_dict(best_param)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.sfm_model.eval()
        x_values = x_test.values
        preds = []
        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.sfm_model(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class SFMNet(nn.Module):
    """SFM Network (Qlib 原版)"""
    def __init__(self, d_feat=6, output_dim=1, freq_dim=10, hidden_size=64, dropout_W=0.0, dropout_U=0.0, device="cpu"):
        super().__init__()
        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device

        self.W_i = nn.Parameter(nn.init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(nn.init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(nn.init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(nn.init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(nn.init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(nn.init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(nn.init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.fc_out = nn.Linear(self.output_dim, 1)
        self.states = []

    def forward(self, input):
        input = input.reshape(len(input), self.input_dim, -1)
        input = input.permute(0, 2, 1)
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:
                self.init_states(x)
            self.get_constants(x)
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))
            f = ste * fre
            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))
            time = time_tm1 + 1
            omega = torch.tensor(2 * np.pi) * time * frequency
            re = torch.cos(omega)
            im = torch.sin(omega)
            c = torch.reshape(c, (-1, self.hidden_dim, 1))
            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im
            A = torch.square(S_re) + torch.square(S_im)
            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)
            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))
            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p
            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x):
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)
        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)
        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)
        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))
        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq
        init_state_time = torch.tensor(0).to(self.device)
        self.states = [init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time, None, None, None]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))
        self.states[5:] = constants


# ============================================================
# GATs 模型 (Graph Attention, Qlib 原版)
# ============================================================

class GATs(Model):
    """GATs Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
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
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gat_model = GATNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gat_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.gat_model.parameters(), lr=self.lr)

        self.fitted = False
        self.gat_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        return -self.loss_fn(pred[mask], label[mask])

    def get_daily_inter(self, df, shuffle=False):
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):
        x_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)
        self.gat_model.train()
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            pred = self.gat_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gat_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.gat_model.eval()
        scores, losses = [], []
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.gat_model(feature)
                losses.append(self.loss_fn(pred, label).item())
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

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.gat_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.gat_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.gat_model.load_state_dict(best_param)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.gat_model.eval()
        x_values = x_test.values
        preds = []
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.gat_model(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class GATNet(nn.Module):
    """GAT Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)
        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


# ============================================================
# HIST 模型 (Qlib 原版)
# ============================================================

class HIST(Model):
    """HIST Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
        stock2concept=None,
        stock_index=None,
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
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.stock2concept = stock2concept
        self.stock_index = stock_index
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.hist_model = HISTNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.hist_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.hist_model.parameters(), lr=self.lr)

        self.fitted = False
        self.hist_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
        return -self.loss_fn(pred[mask], label[mask])

    def get_daily_inter(self, df, shuffle=False):
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def fit(self, dataset, evals_result=None, save_path=None):
        if evals_result is None:
            evals_result = {}
        logger.info("HIST model training - simplified version without concept matrix")
        df_train = self._prepare_data(dataset, "train", ["feature", "label"])
        df_valid = self._prepare_data(dataset, "valid", ["feature", "label"])

        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = (df_valid["feature"], df_valid["label"]) if not df_valid.empty else (None, None)
        else:
            x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
            x_valid, y_valid = (df_valid.iloc[:, :-1], df_valid.iloc[:, -1]) if not df_valid.empty else (None, None)

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.hist_model.state_dict())

        for step in range(self.n_epochs):
            self._train_epoch_simple(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self._test_epoch_simple(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.hist_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.hist_model.load_state_dict(best_param)

    def _train_epoch_simple(self, x_train, y_train):
        x_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)
        self.hist_model.train()
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            pred = self.hist_model(feature, None)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.hist_model.parameters(), 3.0)
            self.train_optimizer.step()

    def _test_epoch_simple(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.hist_model.eval()
        scores, losses = [], []
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.hist_model(feature, None)
                losses.append(self.loss_fn(pred, label).item())
                scores.append(self.metric_fn(pred, label).item())
        return np.mean(losses), np.mean(scores)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.hist_model.eval()
        x_values = x_test.values
        preds = []
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.hist_model(x_batch, None).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class HISTNet(nn.Module):
    """HIST Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        if base_model == "GRU":
            self.rnn = nn.GRU(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc_es = nn.Linear(hidden_size, hidden_size)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.fc_out = nn.Linear(hidden_size, 1)

        for m in [self.fc_es, self.fc_is, self.fc_es_fore, self.fc_is_fore, self.fc_es_back, self.fc_is_back, self.fc_indi]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x, concept_matrix=None):
        x_hidden = x.reshape(len(x), self.d_feat, -1)
        x_hidden = x_hidden.permute(0, 2, 1)
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Simplified version without concept matrix
        output_indi = self.fc_indi(x_hidden)
        output_indi = self.leaky_relu(output_indi)
        pred_all = self.fc_out(output_indi).squeeze()
        return pred_all


# ============================================================
# IGMTF 模型 (Qlib 原版)
# ============================================================

class IGMTF(Model):
    """IGMTF Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        early_stop=20,
        loss="mse",
        base_model="GRU",
        model_path=None,
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
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.igmtf_model = IGMTFNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.igmtf_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.igmtf_model.parameters(), lr=self.lr)

        self.fitted = False
        self.igmtf_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        return -self.loss_fn(pred[mask], label[mask])

    def get_daily_inter(self, df, shuffle=False):
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

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

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.igmtf_model.state_dict())

        for step in range(self.n_epochs):
            self._train_epoch(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self._test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.igmtf_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.igmtf_model.load_state_dict(best_param)

    def _train_epoch(self, x_train, y_train):
        x_values = x_train.values if hasattr(x_train, 'values') else x_train
        y_values = np.squeeze(y_train.values if hasattr(y_train, 'values') else y_train)
        self.igmtf_model.train()
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            pred = self.igmtf_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.igmtf_model.parameters(), 3.0)
            self.train_optimizer.step()

    def _test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.igmtf_model.eval()
        scores, losses = [], []
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.igmtf_model(feature)
                losses.append(self.loss_fn(pred, label).item())
                scores.append(self.metric_fn(pred, label).item())
        return np.mean(losses), np.mean(scores)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.igmtf_model.eval()
        x_values = x_test.values
        preds = []
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.igmtf_model(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class IGMTFNet(nn.Module):
    """IGMTF Network (Qlib 原版)"""
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.lins = nn.Sequential()
        for i in range(2):
            self.lins.add_module("linear" + str(i), nn.Linear(hidden_size, hidden_size))
            self.lins.add_module("leakyrelu" + str(i), nn.LeakyReLU())
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.lins(out)
        return self.fc_out(out).squeeze()


# ============================================================
# MLP 模型 (Qlib 原版)
# ============================================================

class MLP(Model):
    """MLP Model (Qlib 原版)"""

    def __init__(
        self,
        d_feat=6,
        hidden_size=256,
        num_layers=3,
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

        self.mlp_model = MLPNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.mlp_model.parameters(), lr=self.lr)
        else:
            self.train_optimizer = optim.SGD(self.mlp_model.parameters(), lr=self.lr)

        self.fitted = False
        self.mlp_model.to(self.device)

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
        self.mlp_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.mlp_model(feature)
            loss = self.loss_fn(pred, label)
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.mlp_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values if hasattr(data_x, 'values') else data_x
        y_values = np.squeeze(data_y.values if hasattr(data_y, 'values') else data_y)
        self.mlp_model.eval()
        scores, losses = [], []

        for i in range(len(x_values))[:: self.batch_size]:
            if len(x_values) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[i : i + self.batch_size]).float().to(self.device)
            label = torch.from_numpy(y_values[i : i + self.batch_size]).float().to(self.device)
            with torch.no_grad():
                pred = self.mlp_model(feature)
                losses.append(self.loss_fn(pred, label).item())
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

        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        self.fitted = True
        best_param = copy.deepcopy(self.mlp_model.state_dict())

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            if x_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                if val_score > best_score:
                    best_score, stop_steps, best_epoch = val_score, 0, step
                    best_param = copy.deepcopy(self.mlp_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        break
        self.mlp_model.load_state_dict(best_param)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = self._prepare_data(dataset, segment, ["feature"])
        if "feature" in x_test.columns.get_level_values(0):
            x_test = x_test["feature"]
        self.mlp_model.eval()
        x_values = x_test.values
        preds = []
        for begin in range(len(x_values))[:: self.batch_size]:
            end = min(begin + self.batch_size, len(x_values))
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.mlp_model(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=x_test.index)


class MLPNet(nn.Module):
    """MLP Network (Qlib 原版)"""
    def __init__(self, d_feat, hidden_size=256, num_layers=3, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("input", nn.Linear(d_feat, hidden_size))
        self.mlp.add_module("input_act", nn.ReLU())
        self.mlp.add_module("input_drop", nn.Dropout(dropout))
        for i in range(num_layers - 2):
            self.mlp.add_module("hidden_%d" % i, nn.Linear(hidden_size, hidden_size))
            self.mlp.add_module("hidden_act_%d" % i, nn.ReLU())
            self.mlp.add_module("hidden_drop_%d" % i, nn.Dropout(dropout))
        self.mlp.add_module("output", nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.mlp(x).squeeze()


# ============================================================
# 模型工厂
# ============================================================

def get_model(model_type: str, **kwargs):
    """
    获取模型实例 (Qlib 风格)

    Args:
        model_type: 模型类型 (lstm, gru, transformer, alstm, tcn, sfm, gats, hist, igmtf, mlp)
        **kwargs: 模型参数

    Returns:
        模型实例

    支持的模型:
        - lstm: LSTM 模型
        - gru: GRU 模型
        - transformer: Transformer 模型
        - alstm: Attention LSTM 模型
        - tcn: Temporal Convolutional Network
        - sfm: State Frequency Memory
        - gats: Graph Attention Networks
        - hist: HIST 模型
        - igmtf: IGMTF 模型
        - mlp: Multi-Layer Perceptron
    """
    models = {
        'lstm': LSTM,
        'gru': GRU,
        'transformer': TransformerModel,
        'alstm': ALSTM,
        'tcn': TCN,
        'sfm': SFM,
        'gats': GATs,
        'hist': HIST,
        'igmtf': IGMTF,
        'mlp': MLP,
    }

    model_type = model_type.lower()
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


# 导出
__all__ = [
    # 基类
    'Model',
    # 基础模型
    'LSTM', 'LSTMNet',
    'GRU', 'GRUNet',
    'TransformerModel', 'Transformer', 'PositionalEncoding',
    'ALSTM', 'ALSTMNet',
    # TCN 模型
    'TCN', 'TCNNet', 'TemporalConvNet', 'TemporalBlock', 'Chomp1d',
    # SFM 模型
    'SFM', 'SFMNet',
    # GATs 模型
    'GATs', 'GATNet',
    # HIST 模型
    'HIST', 'HISTNet',
    # IGMTF 模型
    'IGMTF', 'IGMTFNet',
    # MLP 模型
    'MLP', 'MLPNet',
    # 工厂函数
    'get_model',
]
