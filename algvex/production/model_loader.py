"""
AlgVex 模型加载器

功能:
- 加载从 Qlib 导出的模型权重
- 不依赖 Qlib 运行时
- 支持 LightGBM、XGBoost、CatBoost
- 提供统一的预测接口

使用方式:
    from production.model_loader import ModelLoader

    loader = ModelLoader()
    model = loader.load("models/lightgbm_v1.pkl")
    predictions = model.predict(features)
"""

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_type: str
    version: str
    created_at: str
    weights_hash: str
    features: List[str]
    training_config: Dict[str, Any]
    metrics: Dict[str, float]


class BaseModel:
    """模型基类"""

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
    ):
        self.model = model
        self.metadata = metadata

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        raise NotImplementedError


class LightGBMModel(BaseModel):
    """LightGBM 模型包装器"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(features)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        importance = self.model.feature_importance(importance_type="gain")
        feature_names = self.metadata.features

        if len(feature_names) != len(importance):
            feature_names = [f"f{i}" for i in range(len(importance))]

        return dict(zip(feature_names, importance.tolist()))


class XGBoostModel(BaseModel):
    """XGBoost 模型包装器"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features)
        return self.model.predict(dmatrix)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.model.get_score(importance_type="gain")


class CatBoostModel(BaseModel):
    """CatBoost 模型包装器"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(features)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        importance = self.model.get_feature_importance()
        feature_names = self.metadata.features

        if len(feature_names) != len(importance):
            feature_names = [f"f{i}" for i in range(len(importance))]

        return dict(zip(feature_names, importance.tolist()))


class NumpyModel(BaseModel):
    """NumPy 线性模型 (用于简单场景或测试)"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测: y = X @ weights + bias"""
        weights = self.model.get("weights", np.zeros(features.shape[1]))
        bias = self.model.get("bias", 0.0)
        return features @ weights + bias

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（权重绝对值）"""
        weights = self.model.get("weights", [])
        feature_names = self.metadata.features

        if len(feature_names) != len(weights):
            feature_names = [f"f{i}" for i in range(len(weights))]

        return dict(zip(feature_names, np.abs(weights).tolist()))


class ModelLoader:
    """模型加载器"""

    MODEL_CLASSES = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
        "numpy": NumpyModel,
    }

    def __init__(self, models_dir: str = "models"):
        """
        初始化模型加载器

        Args:
            models_dir: 模型目录
        """
        self.models_dir = Path(models_dir)
        self._loaded_models: Dict[str, BaseModel] = {}

    def load(
        self,
        model_path: str,
        validate_hash: bool = True,
    ) -> BaseModel:
        """
        加载模型

        Args:
            model_path: 模型文件路径
            validate_hash: 是否验证哈希

        Returns:
            模型实例
        """
        model_path = Path(model_path)

        if not model_path.exists():
            # 尝试在 models_dir 中查找
            model_path = self.models_dir / model_path
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型文件
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # 解析模型数据
        if isinstance(data, dict):
            model = data.get("model")
            metadata_dict = data.get("metadata", {})
        else:
            # 直接是模型对象
            model = data
            metadata_dict = {}

        # 构建元数据
        metadata = ModelMetadata(
            model_id=metadata_dict.get("model_id", model_path.stem),
            model_type=metadata_dict.get("model_type", self._detect_model_type(model)),
            version=metadata_dict.get("version", "1.0.0"),
            created_at=metadata_dict.get("created_at", ""),
            weights_hash=metadata_dict.get("weights_hash", ""),
            features=metadata_dict.get("features", []),
            training_config=metadata_dict.get("training_config", {}),
            metrics=metadata_dict.get("metrics", {}),
        )

        # 验证哈希
        if validate_hash and metadata.weights_hash:
            computed_hash = self._compute_model_hash(model)
            if computed_hash != metadata.weights_hash:
                raise ValueError(
                    f"模型哈希不匹配!\n"
                    f"  存储: {metadata.weights_hash}\n"
                    f"  计算: {computed_hash}"
                )

        # 创建模型包装器
        model_class = self.MODEL_CLASSES.get(metadata.model_type, NumpyModel)
        wrapped_model = model_class(model, metadata)

        # 缓存
        self._loaded_models[str(model_path)] = wrapped_model

        return wrapped_model

    def _detect_model_type(self, model: Any) -> str:
        """检测模型类型"""
        model_type_str = str(type(model))

        if "lightgbm" in model_type_str.lower():
            return "lightgbm"
        elif "xgboost" in model_type_str.lower():
            return "xgboost"
        elif "catboost" in model_type_str.lower():
            return "catboost"
        elif isinstance(model, dict):
            return "numpy"
        else:
            return "unknown"

    @staticmethod
    def _compute_model_hash(model: Any) -> str:
        """计算模型哈希"""
        try:
            model_bytes = pickle.dumps(model)
            hash_value = hashlib.sha256(model_bytes).hexdigest()[:16]
            return f"sha256:{hash_value}"
        except Exception:
            return ""

    def save(
        self,
        model: Any,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        保存模型

        Args:
            model: 模型对象
            model_path: 保存路径
            metadata: 元数据
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 计算哈希
        weights_hash = self._compute_model_hash(model)

        # 构建保存数据
        save_data = {
            "model": model,
            "metadata": {
                "model_id": metadata.get("model_id", model_path.stem) if metadata else model_path.stem,
                "model_type": metadata.get("model_type", self._detect_model_type(model)) if metadata else self._detect_model_type(model),
                "version": metadata.get("version", "1.0.0") if metadata else "1.0.0",
                "created_at": datetime.utcnow().isoformat(),
                "weights_hash": weights_hash,
                "features": metadata.get("features", []) if metadata else [],
                "training_config": metadata.get("training_config", {}) if metadata else {},
                "metrics": metadata.get("metrics", {}) if metadata else {},
            },
        }

        with open(model_path, "wb") as f:
            pickle.dump(save_data, f)

        return weights_hash

    def export_from_qlib(
        self,
        qlib_model: Any,
        output_path: str,
        features: List[str],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        从 Qlib 模型导出

        Args:
            qlib_model: Qlib 模型实例
            output_path: 输出路径
            features: 特征列表
            metrics: 评估指标
        """
        # 提取底层模型
        if hasattr(qlib_model, "model"):
            model = qlib_model.model
        else:
            model = qlib_model

        # 检测模型类型
        model_type = self._detect_model_type(model)

        # 保存
        metadata = {
            "model_id": Path(output_path).stem,
            "model_type": model_type,
            "features": features,
            "metrics": metrics or {},
            "training_config": {
                "source": "qlib",
            },
        }

        return self.save(model, output_path, metadata)

    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        if not self.models_dir.exists():
            return []

        models = []
        for ext in ["*.pkl", "*.pickle", "*.model"]:
            models.extend([str(p) for p in self.models_dir.glob(ext)])

        return models

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """获取模型信息（不加载完整模型）"""
        model_path = Path(model_path)

        if not model_path.exists():
            model_path = self.models_dir / model_path

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "metadata" in data:
                return data["metadata"]
            else:
                return {"model_type": self._detect_model_type(data)}
        except Exception as e:
            return {"error": str(e)}


# 测试代码
if __name__ == "__main__":
    # 创建测试模型
    test_model = {
        "weights": np.array([0.1, -0.2, 0.3, 0.15, -0.1]),
        "bias": 0.01,
    }

    # 保存模型
    loader = ModelLoader(models_dir="test_models")
    os.makedirs("test_models", exist_ok=True)

    hash_value = loader.save(
        test_model,
        "test_models/test_linear.pkl",
        metadata={
            "model_id": "test_linear",
            "model_type": "numpy",
            "features": ["f1", "f2", "f3", "f4", "f5"],
            "metrics": {"ic": 0.05, "sharpe": 1.5},
        },
    )
    print(f"模型已保存，哈希: {hash_value}")

    # 加载模型
    model = loader.load("test_models/test_linear.pkl")
    print(f"\n模型信息:")
    print(f"  ID: {model.metadata.model_id}")
    print(f"  类型: {model.metadata.model_type}")
    print(f"  特征: {model.metadata.features}")
    print(f"  指标: {model.metadata.metrics}")

    # 预测
    features = np.random.randn(10, 5)
    predictions = model.predict(features)
    print(f"\n预测结果 (前5个): {predictions[:5]}")

    # 特征重要性
    importance = model.get_feature_importance()
    print(f"\n特征重要性: {importance}")

    # 清理
    import shutil
    shutil.rmtree("test_models", ignore_errors=True)
