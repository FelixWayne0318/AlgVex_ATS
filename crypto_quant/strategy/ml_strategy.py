"""
机器学习策略模块

职责:
1. 模型训练和预测
2. 特征工程
3. 模型持久化
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb


class MLStrategy:
    """
    机器学习策略

    使用 LightGBM 预测价格方向
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        target_horizon: int = 24,  # 预测未来N个周期
        model_dir: str = "~/.cryptoquant/models",
    ):
        self.model_type = model_type
        self.target_horizon = target_horizon
        self.model_dir = Path(model_dir).expanduser()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}

        # 默认模型参数
        self.model_params = {
            "lightgbm": {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "n_estimators": 200,
                "early_stopping_rounds": 20,
            }
        }

    def create_labels(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        threshold: float = 0.01,
    ) -> pd.Series:
        """
        创建标签

        Args:
            df: 数据
            price_col: 价格列
            threshold: 涨跌阈值

        Returns:
            标签 (1=涨, 0=跌)
        """
        # 未来收益率
        future_return = df[price_col].pct_change(self.target_horizon).shift(-self.target_horizon)

        # 三分类 -> 二分类
        labels = (future_return > threshold).astype(int)

        return labels

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        train_ratio: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据

        Args:
            df: 数据
            feature_cols: 特征列
            label_col: 标签列
            train_ratio: 训练集比例

        Returns:
            X_train, X_test, y_train, y_test
        """
        # 删除缺失值
        df = df.dropna(subset=feature_cols + [label_col])

        X = df[feature_cols]
        y = df[label_col]

        # 时间序列分割
        split_idx = int(len(df) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
        symbol: str = "default",
    ) -> Any:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
            symbol: 标的名称

        Returns:
            训练好的模型
        """
        logger.info(f"Training model for {symbol}...")

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers[symbol] = scaler

        if X_valid is not None:
            X_valid_scaled = scaler.transform(X_valid)
        else:
            # 使用最后20%作为验证集
            split = int(len(X_train) * 0.8)
            X_valid_scaled = X_train_scaled[split:]
            y_valid = y_train.iloc[split:]
            X_train_scaled = X_train_scaled[:split]
            y_train = y_train.iloc[:split]

        if self.model_type == "lightgbm":
            model = self._train_lightgbm(
                X_train_scaled, y_train,
                X_valid_scaled, y_valid,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.models[symbol] = model

        # 保存特征重要性
        if hasattr(model, "feature_importances_"):
            self.feature_importance[symbol] = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)

        return model

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_valid: np.ndarray,
        y_valid: pd.Series,
    ) -> lgb.LGBMClassifier:
        """训练 LightGBM 模型"""
        params = self.model_params["lightgbm"].copy()
        n_estimators = params.pop("n_estimators", 200)
        early_stopping = params.pop("early_stopping_rounds", 20)

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )

        return model

    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            X: 特征
            symbol: 标的名称

        Returns:
            预测类别, 预测概率
        """
        if symbol not in self.models:
            raise ValueError(f"Model not found for {symbol}")

        model = self.models[symbol]
        scaler = self.scalers[symbol]

        X_scaled = scaler.transform(X)
        pred_class = model.predict(X_scaled)
        pred_proba = model.predict_proba(X_scaled)[:, 1]

        return pred_class, pred_proba

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        symbol: str = "default",
    ) -> Dict:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签
            symbol: 标的名称

        Returns:
            评估指标
        """
        pred_class, pred_proba = self.predict(X_test, symbol)

        metrics = {
            "accuracy": accuracy_score(y_test, pred_class),
            "precision": precision_score(y_test, pred_class, zero_division=0),
            "recall": recall_score(y_test, pred_class, zero_division=0),
            "f1": f1_score(y_test, pred_class, zero_division=0),
        }

        logger.info(f"Model evaluation for {symbol}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        symbol: str = "default",
    ) -> Dict:
        """
        时间序列交叉验证

        Args:
            X: 特征
            y: 标签
            n_splits: 折数
            symbol: 标的名称

        Returns:
            交叉验证结果
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            self.train(X_train, y_train, X_valid, y_valid, symbol=f"{symbol}_fold{fold}")
            metrics = self.evaluate(X_valid, y_valid, symbol=f"{symbol}_fold{fold}")
            metrics["fold"] = fold
            cv_results.append(metrics)

        cv_df = pd.DataFrame(cv_results)
        mean_metrics = cv_df.drop("fold", axis=1).mean().to_dict()

        logger.info(f"Cross-validation results for {symbol}:")
        for k, v in mean_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return {
            "folds": cv_results,
            "mean": mean_metrics,
        }

    def save_model(self, symbol: str = "default"):
        """保存模型"""
        if symbol not in self.models:
            logger.warning(f"No model to save for {symbol}")
            return

        model_path = self.model_dir / f"{symbol}_model.pkl"
        scaler_path = self.model_dir / f"{symbol}_scaler.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.models[symbol], f)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scalers[symbol], f)

        logger.info(f"Model saved: {model_path}")

    def load_model(self, symbol: str = "default") -> bool:
        """加载模型"""
        model_path = self.model_dir / f"{symbol}_model.pkl"
        scaler_path = self.model_dir / f"{symbol}_scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            logger.warning(f"Model files not found for {symbol}")
            return False

        with open(model_path, "rb") as f:
            self.models[symbol] = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self.scalers[symbol] = pickle.load(f)

        logger.info(f"Model loaded: {model_path}")
        return True

    def get_feature_importance(self, symbol: str = "default", top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if symbol not in self.feature_importance:
            return pd.DataFrame()

        return self.feature_importance[symbol].head(top_n)


class EnsembleStrategy:
    """
    集成策略

    组合多个模型的预测
    """

    def __init__(self, strategies: List[MLStrategy] = None):
        self.strategies = strategies or []
        self.weights: List[float] = []

    def add_strategy(self, strategy: MLStrategy, weight: float = 1.0):
        """添加策略"""
        self.strategies.append(strategy)
        self.weights.append(weight)

    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        集成预测

        Args:
            X: 特征
            symbol: 标的名称

        Returns:
            预测类别, 预测概率
        """
        if not self.strategies:
            raise ValueError("No strategies in ensemble")

        # 加权平均概率
        total_weight = sum(self.weights)
        proba_sum = np.zeros(len(X))

        for strategy, weight in zip(self.strategies, self.weights):
            _, proba = strategy.predict(X, symbol)
            proba_sum += proba * weight

        avg_proba = proba_sum / total_weight
        pred_class = (avg_proba > 0.5).astype(int)

        return pred_class, avg_proba


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 特征
    X = pd.DataFrame({
        f"feature_{i}": np.random.randn(n_samples) for i in range(10)
    })

    # 标签 (简单线性关系 + 噪声)
    y = (X["feature_0"] + X["feature_1"] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    y = pd.Series(y, name="label")

    # 训练模型
    strategy = MLStrategy(target_horizon=1)

    X_train, X_test, y_train, y_test = strategy.prepare_data(
        pd.concat([X, y], axis=1),
        feature_cols=X.columns.tolist(),
        label_col="label",
    )

    strategy.train(X_train, y_train, symbol="test")
    metrics = strategy.evaluate(X_test, y_test, symbol="test")

    print("\n模型评估结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n特征重要性:")
    print(strategy.get_feature_importance("test", top_n=5))

    # 保存模型
    strategy.save_model("test")
