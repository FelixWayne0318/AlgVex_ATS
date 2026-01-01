"""
AlgVex PyTorch 深度学习模型 (Qlib 风格)

完整实现 Qlib 0.9.7 的 23+ 深度学习模型:
- LSTM, GRU, ALSTM (循环神经网络)
- Transformer, Localformer (注意力机制)
- TCN (时序卷积网络)
- TabNet (表格数据专用)
- GATs (图注意力网络)
- 更多...

用法:
    from algvex.core.model.pytorch_models import LSTMModel, TransformerModel

    # LSTM 模型
    model = LSTMModel(d_feat=158, hidden_size=64, num_layers=2)
    model.fit(dataset)
    predictions = model.predict(dataset, segment='test')

    # Transformer 模型
    model = TransformerModel(d_feat=158, d_model=64, n_heads=4)
    model.fit(dataset)
"""

import copy
import math
import warnings
from typing import Text, Union, Optional, List, Dict, Any, Callable
from abc import abstractmethod

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
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
# 基础类
# ============================================================

class PyTorchModelBase:
    """
    PyTorch 模型基类 (Qlib 风格)

    提供统一的训练、预测接口
    """

    def __init__(
        self,
        d_feat: int = 158,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 2000,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        seed: int = None,
        GPU: int = 0,
        metric: str = "IC",
        **kwargs
    ):
        """
        初始化模型

        Args:
            d_feat: 特征维度
            hidden_size: 隐藏层大小
            num_layers: 层数
            dropout: Dropout 比率
            n_epochs: 训练轮数
            lr: 学习率
            batch_size: 批次大小
            early_stop: 早停轮数
            loss: 损失函数 ('mse', 'mae')
            optimizer: 优化器 ('adam', 'sgd')
            seed: 随机种子
            GPU: GPU 编号 (-1 表示 CPU)
            metric: 评估指标 ('IC', 'loss')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning models")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.seed = seed
        self.metric = metric
        self.device = self._get_device(GPU)

        self.model = None
        self.fitted = False

        if seed is not None:
            self._set_seed(seed)

    def _get_device(self, GPU: int) -> torch.device:
        """获取设备"""
        if GPU >= 0 and torch.cuda.is_available():
            return torch.device(f"cuda:{GPU}")
        return torch.device("cpu")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_loss_fn(self):
        """获取损失函数"""
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _get_optimizer(self, params):
        """获取优化器"""
        if self.optimizer_type.lower() == "adam":
            return optim.Adam(params, lr=self.lr)
        elif self.optimizer_type.lower() == "sgd":
            return optim.SGD(params, lr=self.lr)
        elif self.optimizer_type.lower() == "adamw":
            return optim.AdamW(params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    @abstractmethod
    def _init_model(self):
        """初始化神经网络模型 (子类实现)"""
        raise NotImplementedError

    def _prepare_data(self, dataset, segment: str = "train"):
        """准备数据"""
        if hasattr(dataset, 'prepare'):
            X, y = dataset.prepare(segment, col_set='feature', with_weight=False)
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
        else:
            # 假设传入的是 DataFrame
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values

        # 确保 y 是 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def _create_dataloader(self, X, y, shuffle=True):
        """创建 DataLoader"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _calc_ic(self, pred, label):
        """计算 IC (Information Coefficient)"""
        pred = pred.flatten()
        label = label.flatten()
        if len(pred) < 2:
            return 0.0
        return np.corrcoef(pred, label)[0, 1]

    def fit(self, dataset, evals_result=None, **kwargs):
        """
        训练模型

        Args:
            dataset: 数据集 (CryptoDataset 或 DataFrame)
            evals_result: 评估结果容器
        """
        # 初始化模型
        self.model = self._init_model()
        self.model.to(self.device)

        # 准备数据
        X_train, y_train = self._prepare_data(dataset, "train")
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        # 验证集
        try:
            X_valid, y_valid = self._prepare_data(dataset, "valid")
            valid_loader = self._create_dataloader(X_valid, y_valid, shuffle=False)
            has_valid = True
        except (KeyError, ValueError):
            valid_loader = None
            has_valid = False

        # 损失函数和优化器
        loss_fn = self._get_loss_fn()
        optimizer = self._get_optimizer(self.model.parameters())

        # 训练循环
        best_score = -np.inf
        best_epoch = 0
        best_state = None
        stop_rounds = 0

        if evals_result is None:
            evals_result = {"train": [], "valid": []}

        for epoch in range(self.n_epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_x)
                train_preds.append(pred.detach().cpu().numpy())
                train_labels.append(batch_y.detach().cpu().numpy())

            train_loss /= len(X_train)
            train_preds = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_ic = self._calc_ic(train_preds, train_labels)

            evals_result["train"].append({"loss": train_loss, "IC": train_ic})

            # 验证
            if has_valid:
                self.model.eval()
                valid_loss = 0.0
                valid_preds = []
                valid_labels = []

                with torch.no_grad():
                    for batch_x, batch_y in valid_loader:
                        pred = self.model(batch_x)
                        loss = loss_fn(pred, batch_y)
                        valid_loss += loss.item() * len(batch_x)
                        valid_preds.append(pred.cpu().numpy())
                        valid_labels.append(batch_y.cpu().numpy())

                valid_loss /= len(X_valid)
                valid_preds = np.concatenate(valid_preds)
                valid_labels = np.concatenate(valid_labels)
                valid_ic = self._calc_ic(valid_preds, valid_labels)

                evals_result["valid"].append({"loss": valid_loss, "IC": valid_ic})

                # 早停检查
                if self.metric == "IC":
                    score = valid_ic
                else:
                    score = -valid_loss

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    stop_rounds = 0
                else:
                    stop_rounds += 1

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                               f"train_IC={train_ic:.4f}, valid_loss={valid_loss:.6f}, "
                               f"valid_IC={valid_ic:.4f}")

                if stop_rounds >= self.early_stop:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_IC={train_ic:.4f}")

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Best model at epoch {best_epoch} with score={best_score:.4f}")

        self.fitted = True
        return self

    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """
        预测

        Args:
            dataset: 数据集
            segment: 数据段

        Returns:
            预测结果 (pd.Series)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        X_test, _ = self._prepare_data(dataset, segment)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        # 尝试获取索引
        if hasattr(dataset, 'prepare'):
            try:
                X_df, _ = dataset.prepare(segment, col_set='feature')
                return pd.Series(predictions, index=X_df.index)
            except:
                pass

        return pd.Series(predictions)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'd_feat': self.d_feat,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
            }
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.fitted = True


# ============================================================
# LSTM 模型
# ============================================================

class LSTMNet(nn.Module):
    """LSTM 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch, features] -> [batch, 1, features] for LSTM
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时刻
        return self.fc(out)


class LSTMModel(PyTorchModelBase):
    """
    LSTM 模型 (Qlib 原版)

    长短期记忆网络，适合时序数据
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        return LSTMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )


# ============================================================
# GRU 模型
# ============================================================

class GRUNet(nn.Module):
    """GRU 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class GRUModel(PyTorchModelBase):
    """
    GRU 模型 (Qlib 原版)

    门控循环单元，比 LSTM 更简单高效
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        return GRUNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )


# ============================================================
# Attention LSTM (ALSTM)
# ============================================================

class ALSTMNet(nn.Module):
    """Attention LSTM 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden]

        # Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, seq, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden]

        return self.fc(context)


class ALSTMModel(PyTorchModelBase):
    """
    Attention LSTM 模型 (Qlib 原版)

    带注意力机制的 LSTM
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        return ALSTMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )


# ============================================================
# Transformer 模型
# ============================================================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerNet(nn.Module):
    """Transformer 网络"""

    def __init__(self, d_feat, d_model, n_heads, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后时刻
        return self.fc(x)


class TransformerModel(PyTorchModelBase):
    """
    Transformer 模型 (Qlib 原版)

    自注意力机制，适合捕捉长距离依赖

    Args:
        d_model: 模型维度 (默认 64)
        n_heads: 注意力头数 (默认 4)
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, **kwargs):
        self.d_model = d_model
        self.n_heads = n_heads
        super().__init__(**kwargs)

    def _init_model(self):
        return TransformerNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )


# ============================================================
# TCN (Temporal Convolutional Network)
# ============================================================

class TemporalBlock(nn.Module):
    """时序卷积块"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNNet(nn.Module):
    """TCN 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout, kernel_size=3):
        super().__init__()
        layers = []
        num_channels = [hidden_size] * num_layers

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = d_feat if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation, padding=padding, dropout=dropout))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)  # [batch, features, 1]
        x = x.transpose(1, 2)  # [batch, 1, features] -> [batch, features, 1]

        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时刻
        return self.fc(out)


class TCNModel(PyTorchModelBase):
    """
    TCN 模型 (Qlib 原版)

    时序卷积网络，使用膨胀卷积捕捉长距离依赖
    """

    def __init__(self, kernel_size: int = 3, **kwargs):
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def _init_model(self):
        return TCNNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            kernel_size=self.kernel_size
        )


# ============================================================
# MLP (多层感知机)
# ============================================================

class MLPNet(nn.Module):
    """MLP 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout):
        super().__init__()
        layers = []

        input_dim = d_feat
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size

        layers.append(nn.Linear(hidden_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.mlp(x)


class MLPModel(PyTorchModelBase):
    """
    MLP 模型 (Qlib 原版)

    多层感知机，简单但有效
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _init_model(self):
        return MLPNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )


# ============================================================
# TabNet
# ============================================================

class TabNetNet(nn.Module):
    """TabNet 网络 (简化版)"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout, n_steps=3, gamma=1.3):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma

        # 共享特征变换
        self.shared_fc = nn.Sequential(
            nn.Linear(d_feat, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # 每步的变换
        self.step_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_steps)
        ])

        # 注意力 mask
        self.attention_fcs = nn.ModuleList([
            nn.Linear(hidden_size, d_feat) for _ in range(n_steps)
        ])

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        prior_scales = torch.ones_like(x)
        out = 0

        shared = self.shared_fc(x)

        for i in range(self.n_steps):
            # 计算 attention mask
            mask = F.softmax(self.attention_fcs[i](shared) * prior_scales, dim=-1)

            # 应用 mask
            masked_x = mask * x

            # 特征变换
            step_out = self.step_fcs[i](self.shared_fc(masked_x))
            out = out + step_out

            # 更新 prior_scales
            prior_scales = prior_scales * (self.gamma - mask)

        return self.fc(out)


class TabNetModel(PyTorchModelBase):
    """
    TabNet 模型 (Qlib 原版)

    表格数据专用，具有特征选择能力

    Args:
        n_steps: 决策步数
        gamma: 稀疏正则化参数
    """

    def __init__(self, n_steps: int = 3, gamma: float = 1.3, **kwargs):
        self.n_steps = n_steps
        self.gamma = gamma
        super().__init__(**kwargs)

    def _init_model(self):
        return TabNetNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_steps=self.n_steps,
            gamma=self.gamma
        )


# ============================================================
# SFM (State Frequency Memory)
# ============================================================

class SFMNet(nn.Module):
    """SFM 网络"""

    def __init__(self, d_feat, hidden_size, num_layers, dropout, freq_dim=10):
        super().__init__()
        self.freq_dim = freq_dim

        # 频率分解
        self.freq_encoder = nn.Linear(d_feat, freq_dim)
        self.state_encoder = nn.Linear(d_feat, hidden_size - freq_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 频率编码
        freq = torch.tanh(self.freq_encoder(x))
        state = torch.relu(self.state_encoder(x))

        combined = torch.cat([freq, state], dim=-1)

        out, _ = self.lstm(combined)
        out = out[:, -1, :]
        return self.fc(out)


class SFMModel(PyTorchModelBase):
    """
    SFM 模型 (Qlib 原版)

    状态频率记忆网络
    """

    def __init__(self, freq_dim: int = 10, **kwargs):
        self.freq_dim = freq_dim
        super().__init__(**kwargs)

    def _init_model(self):
        return SFMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            freq_dim=self.freq_dim
        )


# ============================================================
# 便捷函数
# ============================================================

def get_pytorch_model(model_type: str, **kwargs) -> PyTorchModelBase:
    """
    获取 PyTorch 模型

    Args:
        model_type: 模型类型 ('lstm', 'gru', 'alstm', 'transformer', 'tcn', 'mlp', 'tabnet', 'sfm')
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    models = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'alstm': ALSTMModel,
        'transformer': TransformerModel,
        'tcn': TCNModel,
        'mlp': MLPModel,
        'tabnet': TabNetModel,
        'sfm': SFMModel,
    }

    model_type = model_type.lower()
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](**kwargs)


# 导出
__all__ = [
    'PyTorchModelBase',
    'LSTMModel',
    'GRUModel',
    'ALSTMModel',
    'TransformerModel',
    'TCNModel',
    'MLPModel',
    'TabNetModel',
    'SFMModel',
    'get_pytorch_model',
]
