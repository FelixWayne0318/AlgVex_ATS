"""
AlgVex Qlib 风格模型 (Model Wrappers)

实现 Qlib 的模型接口:
- BaseModel: 模型基类
- LinearModel: 线性模型 (OLS, NNLS, Ridge, Lasso)
- LGBModel: LightGBM 模型
- XGBModel: XGBoost 模型

用法:
    from algvex.core.model.qlib_models import LGBModel, LinearModel

    # LightGBM 模型
    model = LGBModel(num_leaves=64, learning_rate=0.05)
    model.fit(dataset)
    predictions = model.predict(dataset, segment='test')

    # 线性模型
    model = LinearModel(estimator='ridge', alpha=0.1)
    model.fit(dataset)
    predictions = model.predict(dataset, segment='test')
"""

from abc import ABC, abstractmethod
from typing import Text, Union, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# 模型基类 (Qlib Model Base)
# ============================================================

class BaseModel(ABC):
    """
    模型基类 (Qlib 风格)

    所有模型必须实现:
    - predict(): 预测
    """

    @abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """进行预测"""
        pass

    def __call__(self, *args, **kwargs) -> object:
        """使模型可调用"""
        return self.predict(*args, **kwargs)

    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class Model(BaseModel):
    """
    可学习模型基类 (Qlib 风格)

    包含 fit 方法
    """

    def fit(self, dataset, reweighter=None):
        """
        训练模型

        Args:
            dataset: 数据集 (CryptoDataset 或 DatasetH)
            reweighter: 样本权重器 (可选)
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """
        预测

        Args:
            dataset: 数据集
            segment: 数据段 ('train', 'valid', 'test' 或 slice)

        Returns:
            预测结果 (pd.Series)
        """
        raise NotImplementedError()


class ModelFT(Model):
    """
    可微调模型基类 (Qlib 风格)

    包含 finetune 方法
    """

    @abstractmethod
    def finetune(self, dataset, **kwargs):
        """
        微调模型

        Args:
            dataset: 数据集
            **kwargs: 微调参数
        """
        raise NotImplementedError()


# ============================================================
# LightGBM 模型 (Qlib LGBModel)
# ============================================================

class LGBModel(ModelFT):
    """
    LightGBM 模型 (Qlib 原版)

    Args:
        loss: 损失函数 ('mse', 'binary')
        early_stopping_rounds: 早停轮数
        num_boost_round: 最大迭代次数
        **kwargs: LightGBM 参数
    """

    def __init__(
        self,
        loss: str = "mse",
        early_stopping_rounds: int = 50,
        num_boost_round: int = 1000,
        **kwargs
    ):
        try:
            import lightgbm as lgb
            self.lgb = lgb
        except ImportError:
            raise ImportError("Please install lightgbm: pip install lightgbm")

        if loss not in {"mse", "binary"}:
            raise NotImplementedError(f"Loss '{loss}' not supported, use 'mse' or 'binary'")

        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset, reweighter=None) -> List:
        """准备 LightGBM 数据集"""
        ds_l = []

        for key in ["train", "valid"]:
            try:
                # 支持 CryptoDataset 和普通 DataFrame
                if hasattr(dataset, 'prepare'):
                    df = dataset.prepare(key, col_set=["feature", "label"])
                else:
                    # 直接传入 DataFrame
                    df = dataset

                if df.empty:
                    if key == "train":
                        raise ValueError("Empty training data")
                    continue

                # 分离特征和标签
                if "feature" in df.columns.get_level_values(0):
                    x = df["feature"]
                    y = df["label"]
                else:
                    # 假设最后一列是标签
                    x = df.iloc[:, :-1]
                    y = df.iloc[:, -1]

                # LightGBM 需要 1D 标签
                if hasattr(y, 'values'):
                    y_values = y.values
                    if y_values.ndim == 2 and y_values.shape[1] == 1:
                        y_values = np.squeeze(y_values)
                else:
                    y_values = y

                # 处理权重
                if reweighter is not None and hasattr(reweighter, 'reweight'):
                    w = reweighter.reweight(df)
                else:
                    w = None

                if hasattr(x, 'values'):
                    x_values = x.values
                else:
                    x_values = x

                ds_l.append((self.lgb.Dataset(x_values, label=y_values, weight=w), key))

            except (KeyError, ValueError) as e:
                if key == "train":
                    raise
                logger.debug(f"Skipping {key} segment: {e}")

        return ds_l

    def fit(
        self,
        dataset,
        num_boost_round: int = None,
        early_stopping_rounds: int = None,
        verbose_eval: int = 20,
        evals_result: Dict = None,
        reweighter=None,
        **kwargs
    ):
        """
        训练模型

        Args:
            dataset: 数据集
            num_boost_round: 迭代次数
            early_stopping_rounds: 早停轮数
            verbose_eval: 日志频率
            evals_result: 评估结果容器
            reweighter: 样本权重器
        """
        if evals_result is None:
            evals_result = {}

        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))

        callbacks = [
            self.lgb.early_stopping(
                early_stopping_rounds or self.early_stopping_rounds
            ),
            self.lgb.log_evaluation(period=verbose_eval),
            self.lgb.record_evaluation(evals_result),
        ]

        self.model = self.lgb.train(
            self.params,
            ds[0],  # 训练集
            num_boost_round=num_boost_round or self.num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=callbacks,
            **kwargs,
        )

        logger.info(f"LGBModel trained with {self.model.num_trees()} trees")
        return self

    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        if hasattr(dataset, 'prepare'):
            x_test = dataset.prepare(segment, col_set="feature")
            if "feature" in x_test.columns.get_level_values(0):
                x_test = x_test["feature"]
        else:
            x_test = dataset

        if hasattr(x_test, 'values'):
            predictions = self.model.predict(x_test.values)
            return pd.Series(predictions, index=x_test.index)
        else:
            return pd.Series(self.model.predict(x_test))

    def finetune(
        self,
        dataset,
        num_boost_round: int = 10,
        verbose_eval: int = 20,
        reweighter=None
    ):
        """
        微调模型

        基于现有模型继续训练更多轮次
        """
        ds_l = self._prepare_data(dataset, reweighter)
        dtrain, _ = ds_l[0]

        callbacks = [
            self.lgb.log_evaluation(period=verbose_eval),
        ]

        self.model = self.lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            callbacks=callbacks,
        )

        return self

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.Series:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return pd.Series(
            self.model.feature_importance(importance_type=importance_type),
            index=self.model.feature_name()
        ).sort_values(ascending=False)


# ============================================================
# XGBoost 模型 (Qlib XGBModel)
# ============================================================

class XGBModel(Model):
    """
    XGBoost 模型 (Qlib 原版)

    Args:
        **kwargs: XGBoost 参数
    """

    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("Please install xgboost: pip install xgboost")

        self._params = kwargs
        self.model = None

    def fit(
        self,
        dataset,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 20,
        evals_result: Dict = None,
        reweighter=None,
        **kwargs
    ):
        """训练模型"""
        if evals_result is None:
            evals_result = {}

        # 准备数据
        if hasattr(dataset, 'prepare'):
            df_train = dataset.prepare("train", col_set=["feature", "label"])
            try:
                df_valid = dataset.prepare("valid", col_set=["feature", "label"])
            except (KeyError, ValueError):
                df_valid = None
        else:
            raise ValueError("Dataset must have prepare method")

        # 分离特征和标签
        if "feature" in df_train.columns.get_level_values(0):
            x_train, y_train = df_train["feature"], df_train["label"]
        else:
            x_train = df_train.iloc[:, :-1]
            y_train = df_train.iloc[:, -1]

        # 1D 标签
        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d = np.squeeze(y_train.values)
        else:
            y_train_1d = y_train.values

        # 处理权重
        if reweighter is not None and hasattr(reweighter, 'reweight'):
            w_train = reweighter.reweight(df_train)
        else:
            w_train = None

        dtrain = self.xgb.DMatrix(x_train.values, label=y_train_1d, weight=w_train)

        evals = [(dtrain, "train")]

        # 验证集
        if df_valid is not None and not df_valid.empty:
            if "feature" in df_valid.columns.get_level_values(0):
                x_valid, y_valid = df_valid["feature"], df_valid["label"]
            else:
                x_valid = df_valid.iloc[:, :-1]
                y_valid = df_valid.iloc[:, -1]

            if y_valid.values.ndim == 2 and y_valid.values.shape[1] == 1:
                y_valid_1d = np.squeeze(y_valid.values)
            else:
                y_valid_1d = y_valid.values

            if reweighter is not None and hasattr(reweighter, 'reweight'):
                w_valid = reweighter.reweight(df_valid)
            else:
                w_valid = None

            dvalid = self.xgb.DMatrix(x_valid.values, label=y_valid_1d, weight=w_valid)
            evals.append((dvalid, "valid"))

        self.model = self.xgb.train(
            self._params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            **kwargs,
        )

        logger.info(f"XGBModel trained")
        return self

    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")

        if hasattr(dataset, 'prepare'):
            x_test = dataset.prepare(segment, col_set="feature")
            if "feature" in x_test.columns.get_level_values(0):
                x_test = x_test["feature"]
        else:
            x_test = dataset

        predictions = self.model.predict(self.xgb.DMatrix(x_test.values))
        return pd.Series(predictions, index=x_test.index)

    def get_feature_importance(self, *args, **kwargs) -> pd.Series:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("Model not fitted yet!")
        return pd.Series(self.model.get_score(*args, **kwargs)).sort_values(ascending=False)


# ============================================================
# 线性模型 (Qlib LinearModel)
# ============================================================

class LinearModel(Model):
    """
    线性模型 (Qlib 原版)

    支持以下估计器:
    - 'ols': 普通最小二乘 min_w |y - Xw|^2_2
    - 'nnls': 非负最小二乘 min_w |y - Xw|^2_2, s.t. w >= 0
    - 'ridge': 岭回归 min_w |y - Xw|^2_2 + α|w|^2_2
    - 'lasso': Lasso min_w |y - Xw|^2_2 + α|w|_1

    Args:
        estimator: 估计器类型
        alpha: 正则化参数 (仅 ridge/lasso)
        fit_intercept: 是否拟合截距
        include_valid: 是否包含验证集进行训练
    """

    OLS = "ols"
    NNLS = "nnls"
    RIDGE = "ridge"
    LASSO = "lasso"

    def __init__(
        self,
        estimator: str = "ols",
        alpha: float = 0.0,
        fit_intercept: bool = False,
        include_valid: bool = False
    ):
        assert estimator in [self.OLS, self.NNLS, self.RIDGE, self.LASSO], \
            f"Unsupported estimator '{estimator}'"
        self.estimator = estimator

        if alpha != 0 and estimator not in [self.RIDGE, self.LASSO]:
            raise ValueError("alpha only supported for ridge/lasso")
        self.alpha = alpha

        self.fit_intercept = fit_intercept
        self.include_valid = include_valid

        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, dataset, reweighter=None):
        """训练模型"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from scipy.optimize import nnls

        # 准备训练数据
        if hasattr(dataset, 'prepare'):
            df_train = dataset.prepare("train", col_set=["feature", "label"])
        else:
            df_train = dataset

        # 可选包含验证集
        if self.include_valid:
            try:
                df_valid = dataset.prepare("valid", col_set=["feature", "label"])
                df_train = pd.concat([df_train, df_valid])
            except (KeyError, ValueError):
                logger.info("include_valid=True but valid does not exist")

        df_train = df_train.dropna()
        if df_train.empty:
            raise ValueError("Empty training data")

        # 分离特征和标签
        if "feature" in df_train.columns.get_level_values(0):
            X = df_train["feature"].values
            y = np.squeeze(df_train["label"].values)
        else:
            X = df_train.iloc[:, :-1].values
            y = df_train.iloc[:, -1].values

        # 处理权重
        if reweighter is not None and hasattr(reweighter, 'reweight'):
            w = reweighter.reweight(df_train).values
        else:
            w = None

        # 训练
        if self.estimator == self.NNLS:
            self._fit_nnls(X, y, w)
        else:
            self._fit_sklearn(X, y, w)

        logger.info(f"LinearModel ({self.estimator}) trained with {len(self.coef_)} features")
        return self

    def _fit_sklearn(self, X, y, w):
        """使用 sklearn 训练"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso

        if self.estimator == self.OLS:
            model = LinearRegression(fit_intercept=self.fit_intercept, copy_X=False)
        elif self.estimator == self.RIDGE:
            model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=False)
        elif self.estimator == self.LASSO:
            model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, copy_X=False)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        model.fit(X, y, sample_weight=w)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_

    def _fit_nnls(self, X, y, w=None):
        """非负最小二乘"""
        from scipy.optimize import nnls

        if w is not None:
            raise NotImplementedError("NNLS with sample weights not supported")

        if self.fit_intercept:
            X = np.c_[X, np.ones(len(X))]

        coef = nnls(X, y)[0]

        if self.fit_intercept:
            self.coef_ = coef[:-1]
            self.intercept_ = coef[-1]
        else:
            self.coef_ = coef
            self.intercept_ = 0.0

    def predict(self, dataset, segment: Union[Text, slice] = "test") -> pd.Series:
        """预测"""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet!")

        if hasattr(dataset, 'prepare'):
            x_test = dataset.prepare(segment, col_set="feature")
            if "feature" in x_test.columns.get_level_values(0):
                x_test = x_test["feature"]
        else:
            x_test = dataset

        predictions = x_test.values @ self.coef_ + self.intercept_
        return pd.Series(predictions, index=x_test.index)


# ============================================================
# 便捷函数
# ============================================================

def get_model(model_type: str, **kwargs) -> Model:
    """
    获取模型实例

    Args:
        model_type: 模型类型 ('lgb', 'xgb', 'linear', 'ridge', 'lasso')
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    model_type = model_type.lower()

    if model_type in ['lgb', 'lightgbm']:
        return LGBModel(**kwargs)
    elif model_type in ['xgb', 'xgboost']:
        return XGBModel(**kwargs)
    elif model_type in ['linear', 'ols']:
        return LinearModel(estimator='ols', **kwargs)
    elif model_type == 'ridge':
        return LinearModel(estimator='ridge', **kwargs)
    elif model_type == 'lasso':
        return LinearModel(estimator='lasso', **kwargs)
    elif model_type == 'nnls':
        return LinearModel(estimator='nnls', **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
