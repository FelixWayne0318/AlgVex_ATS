"""
AlgVex Qlib 模型封装层

完整封装 Qlib 0.9.7 所有模型，包括:
- GBDT: LightGBM, XGBoost, CatBoost
- Linear: 线性模型
- Deep Learning: LSTM, GRU, Transformer, TabNet, ALSTM, etc.
- Ensemble: DoubleEnsemble

版本: 2.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型枚举"""
    # GBDT
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"

    # Linear
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"

    # Deep Learning - Basic
    LSTM = "lstm"
    GRU = "gru"
    MLP = "mlp"
    TCN = "tcn"

    # Deep Learning - Advanced
    TRANSFORMER = "transformer"
    ALSTM = "alstm"
    TABNET = "tabnet"
    GATS = "gats"
    SFM = "sfm"
    HIST = "hist"
    TRA = "tra"
    ADARNN = "adarnn"
    IGMTF = "igmtf"
    KRNN = "krnn"
    LOCALFORMER = "localformer"
    TCTS = "tcts"
    ADD = "add"
    SANDWICH = "sandwich"

    # Ensemble
    DOUBLE_ENSEMBLE = "double_ensemble"


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    params: Dict[str, Any] = field(default_factory=dict)
    gpu: bool = False
    seed: int = 42

    # 训练参数
    n_epochs: int = 100
    batch_size: int = 2048
    early_stop_rounds: int = 20
    learning_rate: float = 0.001

    # 特征参数 (Deep Learning)
    d_feat: int = 158  # Alpha158 特征数
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0


class BaseModelWrapper(ABC):
    """模型包装器基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """创建底层模型"""
        pass

    def fit(self, dataset, evals_result: Optional[Dict] = None):
        """训练模型"""
        if self._model is None:
            self._model = self._create_model()

        self._model.fit(dataset, evals_result=evals_result)
        self._is_fitted = True
        return self

    def predict(self, dataset, segment: str = "test") -> pd.Series:
        """预测"""
        if not self._is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._model.predict(dataset, segment=segment)

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'config': self.config}, f)

    @classmethod
    def load(cls, path: str) -> 'BaseModelWrapper':
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        wrapper = cls(data['config'])
        wrapper._model = data['model']
        wrapper._is_fitted = True
        return wrapper


# ============== GBDT Models ==============

class LightGBMWrapper(BaseModelWrapper):
    """LightGBM 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.gbdt import LGBModel

        default_params = {
            "loss": "mse",
            "num_boost_round": 500,
            "early_stopping_rounds": self.config.early_stop_rounds,
            "learning_rate": self.config.learning_rate,
            "num_leaves": 64,
            "max_depth": 8,
            "lambda_l1": 200,
            "lambda_l2": 200,
            "seed": self.config.seed,
        }
        default_params.update(self.config.params)

        return LGBModel(**default_params)


class XGBoostWrapper(BaseModelWrapper):
    """XGBoost 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.xgboost import XGBModel

        default_params = {
            "n_estimators": 500,
            "early_stopping_rounds": self.config.early_stop_rounds,
            "learning_rate": self.config.learning_rate,
            "max_depth": 8,
            "seed": self.config.seed,
        }
        if self.config.gpu:
            default_params["tree_method"] = "gpu_hist"

        default_params.update(self.config.params)
        return XGBModel(**default_params)


class CatBoostWrapper(BaseModelWrapper):
    """CatBoost 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.catboost_model import CatBoostModel

        default_params = {
            "iterations": 500,
            "early_stopping_rounds": self.config.early_stop_rounds,
            "learning_rate": self.config.learning_rate,
            "depth": 8,
            "random_seed": self.config.seed,
            "verbose": 100,
        }
        if self.config.gpu:
            default_params["task_type"] = "GPU"

        default_params.update(self.config.params)
        return CatBoostModel(**default_params)


# ============== Linear Models ==============

class LinearWrapper(BaseModelWrapper):
    """线性模型封装"""

    def _create_model(self):
        from qlib.contrib.model.linear import LinearModel

        estimator_map = {
            ModelType.LINEAR: "ols",
            ModelType.RIDGE: "ridge",
            ModelType.LASSO: "lasso",
        }

        default_params = {
            "estimator": estimator_map.get(self.config.model_type, "ols"),
            "alpha": self.config.params.get("alpha", 1.0),
        }
        default_params.update(self.config.params)

        return LinearModel(**default_params)


# ============== Deep Learning Models ==============

class LSTMWrapper(BaseModelWrapper):
    """LSTM 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_lstm import LSTM

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return LSTM(**default_params)


class GRUWrapper(BaseModelWrapper):
    """GRU 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_gru import GRU

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return GRU(**default_params)


class MLPWrapper(BaseModelWrapper):
    """MLP (多层感知机) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_nn import DNNModel

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return DNNModel(**default_params)


class TCNWrapper(BaseModelWrapper):
    """TCN (时序卷积网络) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_tcn import TCN

        default_params = {
            "d_feat": self.config.d_feat,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return TCN(**default_params)


class TransformerWrapper(BaseModelWrapper):
    """Transformer 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_transformer import Transformer

        default_params = {
            "d_feat": self.config.d_feat,
            "d_model": self.config.hidden_size,
            "nhead": 4,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return Transformer(**default_params)


class ALSTMWrapper(BaseModelWrapper):
    """ALSTM (注意力 LSTM) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_alstm import ALSTM

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return ALSTM(**default_params)


class TabNetWrapper(BaseModelWrapper):
    """TabNet 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_tabnet import TabNet

        default_params = {
            "d_feat": self.config.d_feat,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return TabNet(**default_params)


class GATsWrapper(BaseModelWrapper):
    """GATs (图注意力网络) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_gats import GATs

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return GATs(**default_params)


class SFMWrapper(BaseModelWrapper):
    """SFM (State Frequency Memory) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_sfm import SFM

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return SFM(**default_params)


class HISTWrapper(BaseModelWrapper):
    """HIST 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_hist import HIST

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return HIST(**default_params)


class TRAWrapper(BaseModelWrapper):
    """TRA (Temporal Routing Adaptor) 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_tra import TRA

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return TRA(**default_params)


class AdaRNNWrapper(BaseModelWrapper):
    """AdaRNN 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_adarnn import ADARNN

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return ADARNN(**default_params)


class IGMTFWrapper(BaseModelWrapper):
    """IGMTF 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_igmtf import IGMTF

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return IGMTF(**default_params)


class KRNNWrapper(BaseModelWrapper):
    """KRNN 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_krnn import KRNN

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return KRNN(**default_params)


class LocalformerWrapper(BaseModelWrapper):
    """Localformer 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_localformer import Localformer

        default_params = {
            "d_feat": self.config.d_feat,
            "d_model": self.config.hidden_size,
            "nhead": 4,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return Localformer(**default_params)


class TCTSWrapper(BaseModelWrapper):
    """TCTS 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_tcts import TCTS

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return TCTS(**default_params)


class ADDWrapper(BaseModelWrapper):
    """ADD 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_add import ADD

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return ADD(**default_params)


class SandwichWrapper(BaseModelWrapper):
    """Sandwich 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.pytorch_sandwich import Sandwich

        default_params = {
            "d_feat": self.config.d_feat,
            "hidden_size": self.config.hidden_size,
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "early_stop": self.config.early_stop_rounds,
            "lr": self.config.learning_rate,
            "seed": self.config.seed,
            "GPU": 0 if self.config.gpu else None,
        }
        default_params.update(self.config.params)

        return Sandwich(**default_params)


# ============== Ensemble Models ==============

class DoubleEnsembleWrapper(BaseModelWrapper):
    """DoubleEnsemble 模型封装"""

    def _create_model(self):
        from qlib.contrib.model.double_ensemble import DEnsembleModel

        default_params = {
            "base_model": "gbm",
            "num_models": 6,
            "enable_sr": True,
            "enable_fs": True,
            "alpha1": 1.0,
            "alpha2": 1.0,
            "bins_sr": 10,
            "bins_fs": 5,
            "decay": 0.5,
            "sample_ratios": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "seed": self.config.seed,
        }
        default_params.update(self.config.params)

        return DEnsembleModel(**default_params)


# ============== Model Factory ==============

class ModelFactory:
    """模型工厂"""

    _registry: Dict[ModelType, Type[BaseModelWrapper]] = {
        # GBDT
        ModelType.LIGHTGBM: LightGBMWrapper,
        ModelType.XGBOOST: XGBoostWrapper,
        ModelType.CATBOOST: CatBoostWrapper,

        # Linear
        ModelType.LINEAR: LinearWrapper,
        ModelType.RIDGE: LinearWrapper,
        ModelType.LASSO: LinearWrapper,

        # Deep Learning - Basic
        ModelType.LSTM: LSTMWrapper,
        ModelType.GRU: GRUWrapper,
        ModelType.MLP: MLPWrapper,
        ModelType.TCN: TCNWrapper,

        # Deep Learning - Advanced
        ModelType.TRANSFORMER: TransformerWrapper,
        ModelType.ALSTM: ALSTMWrapper,
        ModelType.TABNET: TabNetWrapper,
        ModelType.GATS: GATsWrapper,
        ModelType.SFM: SFMWrapper,
        ModelType.HIST: HISTWrapper,
        ModelType.TRA: TRAWrapper,
        ModelType.ADARNN: AdaRNNWrapper,
        ModelType.IGMTF: IGMTFWrapper,
        ModelType.KRNN: KRNNWrapper,
        ModelType.LOCALFORMER: LocalformerWrapper,
        ModelType.TCTS: TCTSWrapper,
        ModelType.ADD: ADDWrapper,
        ModelType.SANDWICH: SandwichWrapper,

        # Ensemble
        ModelType.DOUBLE_ENSEMBLE: DoubleEnsembleWrapper,
    }

    @classmethod
    def create(cls, config: ModelConfig) -> BaseModelWrapper:
        """
        创建模型

        Args:
            config: 模型配置

        Returns:
            模型封装器
        """
        wrapper_cls = cls._registry.get(config.model_type)
        if wrapper_cls is None:
            raise ValueError(f"不支持的模型类型: {config.model_type}")

        return wrapper_cls(config)

    @classmethod
    def create_by_name(
        cls,
        model_name: str,
        **kwargs
    ) -> BaseModelWrapper:
        """
        通过名称创建模型

        Args:
            model_name: 模型名称
            **kwargs: 额外参数

        Returns:
            模型封装器
        """
        try:
            model_type = ModelType(model_name.lower())
        except ValueError:
            raise ValueError(f"不支持的模型名称: {model_name}")

        config = ModelConfig(model_type=model_type, params=kwargs)
        return cls.create(config)

    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有支持的模型"""
        return [t.value for t in ModelType]

    @classmethod
    def get_model_info(cls, model_type: ModelType) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            ModelType.LIGHTGBM: {
                "name": "LightGBM",
                "category": "GBDT",
                "description": "轻量级梯度提升决策树",
                "gpu_support": True,
            },
            ModelType.XGBOOST: {
                "name": "XGBoost",
                "category": "GBDT",
                "description": "极端梯度提升",
                "gpu_support": True,
            },
            ModelType.CATBOOST: {
                "name": "CatBoost",
                "category": "GBDT",
                "description": "类别特征优化的梯度提升",
                "gpu_support": True,
            },
            ModelType.LSTM: {
                "name": "LSTM",
                "category": "Deep Learning",
                "description": "长短期记忆网络",
                "gpu_support": True,
            },
            ModelType.GRU: {
                "name": "GRU",
                "category": "Deep Learning",
                "description": "门控循环单元",
                "gpu_support": True,
            },
            ModelType.TRANSFORMER: {
                "name": "Transformer",
                "category": "Deep Learning",
                "description": "Transformer 编码器",
                "gpu_support": True,
            },
            ModelType.ALSTM: {
                "name": "ALSTM",
                "category": "Deep Learning",
                "description": "注意力增强 LSTM",
                "gpu_support": True,
            },
            ModelType.TABNET: {
                "name": "TabNet",
                "category": "Deep Learning",
                "description": "表格数据注意力网络",
                "gpu_support": True,
            },
            ModelType.DOUBLE_ENSEMBLE: {
                "name": "DoubleEnsemble",
                "category": "Ensemble",
                "description": "双重集成模型",
                "gpu_support": False,
            },
        }
        return info.get(model_type, {"name": model_type.value, "category": "Unknown"})


# ============== 便捷函数 ==============

def create_model(model_type: str, **kwargs) -> BaseModelWrapper:
    """
    便捷函数: 创建模型

    Args:
        model_type: 模型类型名称
        **kwargs: 模型参数

    Returns:
        模型封装器

    Example:
        model = create_model("lightgbm", learning_rate=0.01)
        model.fit(dataset)
        predictions = model.predict(dataset)
    """
    return ModelFactory.create_by_name(model_type, **kwargs)


def list_available_models() -> List[str]:
    """列出所有可用模型"""
    return ModelFactory.list_models()


# 测试代码
if __name__ == "__main__":
    print("AlgVex Qlib Models v2.0.0")
    print("=" * 50)
    print("\n可用模型:")

    for model_name in list_available_models():
        print(f"  - {model_name}")

    print("\n使用示例:")
    print("  from algvex.research.qlib_models import create_model")
    print("  model = create_model('lightgbm', learning_rate=0.01)")
    print("  model.fit(dataset)")
    print("  predictions = model.predict(dataset)")
