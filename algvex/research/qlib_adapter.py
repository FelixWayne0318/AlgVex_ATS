"""
AlgVex Qlib 适配器

功能:
- 封装 Qlib 接口用于研究
- 提供因子导出功能
- 提供模型导出功能
- 仅用于研究环境

使用方式:
    from research.qlib_adapter import QlibAdapter

    adapter = QlibAdapter()
    adapter.init_qlib(data_path="~/.qlib/qlib_data")

    # 使用 Alpha158
    dataset = adapter.create_dataset(instruments=["BTC", "ETH"])

    # 导出模型
    adapter.export_model(model, "models/exported/model_v1.pkl")
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class QlibAdapter:
    """Qlib 适配器（仅研究环境使用）"""

    def __init__(self):
        """初始化适配器"""
        self._initialized = False
        self._qlib = None

    def init_qlib(
        self,
        data_path: str,
        region: str = "us",
    ) -> bool:
        """
        初始化 Qlib

        Args:
            data_path: 数据路径
            region: 地区 (us/cn)

        Returns:
            是否初始化成功
        """
        try:
            import qlib
            from qlib.constant import REG_US, REG_CN

            # 检查是否已初始化
            try:
                from qlib.data import D
                D.calendar(freq="day")[:1]
                print("Qlib 已初始化，跳过")
                self._initialized = True
                self._qlib = qlib
                return True
            except:
                pass

            # 初始化
            region_map = {"us": REG_US, "cn": REG_CN}
            qlib.init(
                provider_uri=os.path.expanduser(data_path),
                region=region_map.get(region, REG_US),
            )

            self._initialized = True
            self._qlib = qlib
            print(f"Qlib 初始化成功: {data_path}")
            return True

        except ImportError:
            print("错误: Qlib 未安装")
            return False
        except Exception as e:
            print(f"错误: Qlib 初始化失败: {e}")
            return False

    def create_dataset(
        self,
        instruments: List[str],
        start_time: str,
        end_time: str,
        handler_type: str = "Alpha158",
        train_end: Optional[str] = None,
        valid_start: Optional[str] = None,
        valid_end: Optional[str] = None,
        test_start: Optional[str] = None,
    ):
        """
        创建 Qlib 数据集

        Args:
            instruments: 股票/币种列表
            start_time: 开始时间
            end_time: 结束时间
            handler_type: 处理器类型
            train_end: 训练集结束时间
            valid_start: 验证集开始时间
            valid_end: 验证集结束时间
            test_start: 测试集开始时间

        Returns:
            Qlib DatasetH 对象
        """
        if not self._initialized:
            raise RuntimeError("请先调用 init_qlib()")

        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH

        # 创建 Handler
        if handler_type == "Alpha158":
            handler = Alpha158(
                instruments=instruments,
                start_time=start_time,
                end_time=end_time,
                fit_start_time=start_time,
                fit_end_time=train_end or end_time,
            )
        else:
            raise ValueError(f"不支持的 handler 类型: {handler_type}")

        # 设置时间分段
        segments = {
            "train": (start_time, train_end or end_time),
        }

        if valid_start and valid_end:
            segments["valid"] = (valid_start, valid_end)

        if test_start:
            segments["test"] = (test_start, end_time)

        # 创建数据集
        dataset = DatasetH(handler=handler, segments=segments)

        return dataset

    def train_model(
        self,
        dataset,
        model_type: str = "lightgbm",
        model_config: Optional[Dict] = None,
    ):
        """
        训练模型

        Args:
            dataset: Qlib 数据集
            model_type: 模型类型
            model_config: 模型配置

        Returns:
            训练好的模型
        """
        if not self._initialized:
            raise RuntimeError("请先调用 init_qlib()")

        # 默认配置
        default_configs = {
            "lightgbm": {
                "loss": "mse",
                "num_boost_round": 500,
                "early_stopping_rounds": 50,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": 8,
                "lambda_l1": 200,
                "lambda_l2": 200,
            },
            "xgboost": {
                "n_estimators": 500,
                "early_stopping_rounds": 50,
                "learning_rate": 0.05,
                "max_depth": 8,
            },
        }

        config = default_configs.get(model_type, {})
        if model_config:
            config.update(model_config)

        # 创建模型
        if model_type == "lightgbm":
            from qlib.contrib.model.gbdt import LGBModel
            model = LGBModel(**config)
        elif model_type == "xgboost":
            from qlib.contrib.model.xgboost import XGBModel
            model = XGBModel(**config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 训练
        model.fit(dataset)

        return model

    def export_model(
        self,
        model,
        output_path: str,
        features: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        导出模型到生产格式

        Args:
            model: Qlib 模型
            output_path: 输出路径
            features: 特征列表
            metrics: 评估指标

        Returns:
            模型哈希值
        """
        from ..production.model_loader import ModelLoader

        loader = ModelLoader()
        return loader.export_from_qlib(
            qlib_model=model,
            output_path=output_path,
            features=features or [],
            metrics=metrics,
        )

    def get_predictions(
        self,
        model,
        dataset,
        segment: str = "test",
    ) -> pd.Series:
        """
        获取模型预测

        Args:
            model: Qlib 模型
            dataset: 数据集
            segment: 数据段

        Returns:
            预测结果 Series
        """
        return model.predict(dataset, segment=segment)

    def compute_ic(
        self,
        predictions: pd.Series,
        labels: pd.Series,
    ) -> Dict[str, float]:
        """
        计算 IC 指标

        Args:
            predictions: 预测值
            labels: 实际值

        Returns:
            IC 指标字典
        """
        # 对齐索引
        common_idx = predictions.index.intersection(labels.index)
        pred_aligned = predictions.loc[common_idx]
        label_aligned = labels.loc[common_idx]

        # 合并
        df = pd.DataFrame({
            "prediction": pred_aligned,
            "actual": label_aligned,
        })

        # 按日期计算相关性
        daily_corr = df.groupby(level="datetime").apply(
            lambda x: x["prediction"].corr(x["actual"])
        ).dropna()

        ic_mean = daily_corr.mean()
        ic_std = daily_corr.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "ic_positive_ratio": (daily_corr > 0).mean(),
        }


# 测试代码
if __name__ == "__main__":
    adapter = QlibAdapter()

    print("QlibAdapter 已创建")
    print("使用方式:")
    print("  adapter.init_qlib(data_path='~/.qlib/qlib_data/us_data')")
    print("  dataset = adapter.create_dataset(instruments=['AAPL'], ...)")
    print("  model = adapter.train_model(dataset, model_type='lightgbm')")
    print("  adapter.export_model(model, 'models/exported/model_v1.pkl')")
