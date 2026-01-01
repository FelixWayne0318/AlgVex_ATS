"""
AlgVex 数据集类 (Qlib 风格)

实现 Qlib 的 DatasetH 模式:
- 统一的数据划分 (segments)
- prepare() 方法获取指定段数据
- 支持 feature/label 分离
- 集成 Processor 处理链

用法:
    from algvex.core.data.dataset import CryptoDataset

    # 创建数据集
    dataset = CryptoDataset(
        data=training_data,
        segments={
            'train': ("2024-01-01", "2024-06-30"),
            'valid': ("2024-07-01", "2024-08-31"),
            'test':  ("2024-09-01", "2024-12-23"),
        },
        feature_cols=factor_columns,
        label_col='label'
    )

    # 获取数据
    X_train, y_train = dataset.prepare("train")
    X_valid, y_valid = dataset.prepare("valid")
"""

from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置"""
    segments: Dict[str, Tuple[str, str]]
    feature_cols: List[str]
    label_col: str = 'label'
    weight_col: Optional[str] = None


class CryptoDataset:
    """
    加密货币数据集 (Qlib DatasetH 风格)

    特性:
    - 统一的 segments 配置
    - prepare() 方法返回 (X, y) 或 (X, y, w)
    - 支持处理器链
    - 支持样本权重

    与 Qlib 的主要区别:
    - 简化的列结构 (扁平而非 MultiIndex)
    - 针对加密货币单交易对优化
    - 更直观的 API
    """

    def __init__(
        self,
        data: pd.DataFrame,
        segments: Dict[str, Tuple[str, str]],
        feature_cols: List[str],
        label_col: str = 'label',
        processors: List = None,
        reweighter = None,
    ):
        """
        初始化数据集

        Args:
            data: 完整数据 (必须有 DatetimeIndex)
            segments: 数据段配置, 如 {'train': ("2024-01-01", "2024-06-30"), ...}
            feature_cols: 特征列名列表
            label_col: 标签列名
            processors: 处理器列表 (可选)
            reweighter: 重新加权器 (可选)
        """
        # 验证数据
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            data.index = pd.to_datetime(data.index)

        self._data = data.sort_index()
        self.segments = segments
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.processors = processors or []
        self.reweighter = reweighter

        # 缓存处理后的数据
        self._processed_data = None
        self._is_fitted = False

        # 验证列存在
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")

        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        logger.info(f"CryptoDataset initialized: {len(data)} rows, {len(feature_cols)} features")
        logger.info(f"  Segments: {list(segments.keys())}")

    def fit_processors(self, segment: str = 'train'):
        """
        在指定段上拟合处理器

        Args:
            segment: 用于拟合的数据段 (默认 'train')
        """
        if not self.processors:
            self._is_fitted = True
            return

        # 获取训练数据
        train_data = self._get_segment_data(segment)

        # 检查是否是 ProcessorChain 对象
        if hasattr(self.processors, 'fit') and hasattr(self.processors, 'transform'):
            # ProcessorChain 对象，直接调用 fit
            self.processors.fit(train_data)
        else:
            # 列表形式，依次拟合处理器
            current_data = train_data.copy()
            for processor in self.processors:
                processor.fit(current_data)
                current_data = processor.transform(current_data)

        self._is_fitted = True
        logger.info(f"Processors fitted on '{segment}' segment ({len(train_data)} rows)")

    def _get_segment_data(self, segment: str) -> pd.DataFrame:
        """获取指定段的原始数据"""
        if segment not in self.segments:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segments.keys())}")

        start, end = self.segments[segment]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        mask = (self._data.index >= start_dt) & (self._data.index <= end_dt)
        return self._data[mask].copy()

    def _apply_processors(self, data: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
        """应用处理器链"""
        if not self.processors:
            return data

        result = data.copy()

        # 检查是否是 ProcessorChain 对象
        if hasattr(self.processors, 'transform'):
            # ProcessorChain 对象，直接调用 transform
            result = self.processors.transform(result, for_inference=for_inference) if hasattr(self.processors.transform, '__code__') and 'for_inference' in self.processors.transform.__code__.co_varnames else self.processors.transform(result)
        else:
            # 列表形式，依次应用处理器
            for processor in self.processors:
                # 检查是否适用于推理
                if for_inference and hasattr(processor, 'is_for_infer'):
                    if not processor.is_for_infer():
                        continue
                result = processor.transform(result)

        return result

    def prepare(
        self,
        segment: Union[str, List[str]],
        col_set: str = 'all',  # 'all', 'feature', 'label'
        with_weight: bool = False,
        apply_processors: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, np.ndarray]]:
        """
        准备指定段的数据 (Qlib 风格)

        Args:
            segment: 数据段名称或列表
            col_set: 返回的列集合 ('all', 'feature', 'label')
            with_weight: 是否返回样本权重
            apply_processors: 是否应用处理器

        Returns:
            根据参数返回:
            - col_set='all': DataFrame
            - col_set='feature': (X, y) 或 (X, y, w)
        """
        # 处理多段请求
        if isinstance(segment, (list, tuple)):
            return [self.prepare(s, col_set, with_weight, apply_processors) for s in segment]

        # 获取原始数据
        data = self._get_segment_data(segment)

        if len(data) == 0:
            logger.warning(f"Empty data for segment '{segment}'")

        # 应用处理器 (推理段使用 for_inference=True)
        if apply_processors and self.processors:
            if not self._is_fitted:
                logger.warning("Processors not fitted, fitting on train segment...")
                self.fit_processors('train')

            for_inference = segment not in ['train']
            data = self._apply_processors(data, for_inference=for_inference)

        # 返回指定列集
        if col_set == 'all':
            return data

        # 分离 feature 和 label
        available_features = [c for c in self.feature_cols if c in data.columns]
        X = data[available_features]
        y = data[self.label_col]

        if col_set == 'label':
            return y

        # 计算权重
        if with_weight and self.reweighter is not None:
            w = self.reweighter.reweight(data)
            return X, y, w

        return X, y

    def get_segment_info(self) -> Dict[str, Dict]:
        """获取各段的信息"""
        info = {}
        for name, (start, end) in self.segments.items():
            data = self._get_segment_data(name)
            info[name] = {
                'start': start,
                'end': end,
                'rows': len(data),
                'date_range': f"{data.index.min()} ~ {data.index.max()}" if len(data) > 0 else "N/A"
            }
        return info

    def summary(self):
        """打印数据集摘要"""
        print("=" * 60)
        print("📊 CryptoDataset Summary")
        print("=" * 60)
        print(f"Total rows: {len(self._data):,}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Label: {self.label_col}")
        print(f"Date range: {self._data.index.min()} ~ {self._data.index.max()}")
        print(f"\nSegments:")

        for name, info in self.get_segment_info().items():
            print(f"  {name}: {info['rows']:,} rows ({info['start']} ~ {info['end']})")

        if self.processors:
            # 检查是否是 ProcessorChain 对象
            if hasattr(self.processors, 'processors'):
                # ProcessorChain 对象
                print(f"\nProcessors: {len(self.processors.processors)} (ProcessorChain)")
                for p in self.processors.processors:
                    print(f"  - {type(p).__name__}")
            else:
                # 列表形式
                print(f"\nProcessors: {len(self.processors)}")
                for p in self.processors:
                    print(f"  - {type(p).__name__}")

        if self.reweighter:
            print(f"\nReweighter: {type(self.reweighter).__name__}")

        print("=" * 60)

    def __repr__(self):
        return f"CryptoDataset(rows={len(self._data)}, features={len(self.feature_cols)}, segments={list(self.segments.keys())})"


# ============================================================
# 便捷函数
# ============================================================

def create_dataset_from_config(
    data: pd.DataFrame,
    config: dict,
) -> CryptoDataset:
    """
    从配置字典创建数据集

    Args:
        data: 原始数据
        config: 配置字典, 包含:
            - segments: 数据段配置
            - feature_cols: 特征列
            - label_col: 标签列
            - processors: 处理器配置 (可选)

    Returns:
        CryptoDataset 实例
    """
    from algvex.core.factor.processor import ProcessorChain

    processors = None
    if 'processors' in config:
        # 从配置创建处理器
        processors = []
        for p_config in config['processors']:
            # 动态创建处理器实例
            pass  # TODO: 实现处理器工厂

    return CryptoDataset(
        data=data,
        segments=config['segments'],
        feature_cols=config['feature_cols'],
        label_col=config.get('label_col', 'label'),
        processors=processors,
    )
