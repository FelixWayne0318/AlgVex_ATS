"""
AlgVex 信号生成器

根据模型预测生成交易信号。

使用示例:
    config = SignalConfig(
        threshold=0.02,
        holding_period=288,
    )

    generator = SignalGenerator(config)
    signals = generator.generate(predictions, timestamps, prices)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class SignalType(Enum):
    """信号类型"""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass
class Signal:
    """
    交易信号

    Attributes:
        timestamp: 信号时间
        symbol: 交易标的
        signal_type: 信号类型
        strength: 信号强度 (-1 到 1)
        price: 参考价格
        prediction: 原始预测值
        metadata: 额外信息
    """
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float
    price: Optional[float] = None
    prediction: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """是否做多"""
        return self.signal_type == SignalType.LONG

    @property
    def is_short(self) -> bool:
        """是否做空"""
        return self.signal_type == SignalType.SHORT

    @property
    def is_close(self) -> bool:
        """是否平仓"""
        return self.signal_type == SignalType.CLOSE

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "price": self.price,
            "prediction": self.prediction,
            "metadata": self.metadata,
        }


@dataclass
class SignalConfig:
    """
    信号生成配置

    Attributes:
        long_threshold: 做多阈值 (预测值 > 阈值时做多)
        short_threshold: 做空阈值 (预测值 < -阈值时做空)
        holding_period: 持仓周期 (bars)
        min_strength: 最小信号强度
        signal_decay: 信号衰减 (0-1)
        use_zscore: 是否使用 z-score 标准化预测
        zscore_window: z-score 窗口 (建议 20-50)
        zscore_min_periods: z-score 最小数据量 (不足时使用全部可用数据)
        cooldown: 信号冷却期 (bars)
    """
    long_threshold: float = 0.02
    short_threshold: float = -0.02
    holding_period: int = 288  # 1天
    min_strength: float = 0.1
    signal_decay: float = 0.0
    use_zscore: bool = False  # 默认关闭，避免冷启动问题
    zscore_window: int = 20  # 减小窗口，避免数据不足问题
    zscore_min_periods: int = 5  # 最少5个数据点才计算z-score
    cooldown: int = 12  # 1小时

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "holding_period": self.holding_period,
            "min_strength": self.min_strength,
            "signal_decay": self.signal_decay,
            "use_zscore": self.use_zscore,
            "zscore_window": self.zscore_window,
            "zscore_min_periods": self.zscore_min_periods,
            "cooldown": self.cooldown,
        }


class SignalGenerator:
    """
    信号生成器

    使用示例:
        config = SignalConfig(
            long_threshold=0.02,
            short_threshold=-0.02,
        )

        generator = SignalGenerator(config)
        signals = generator.generate(
            predictions=predictions_series,
            prices=prices_df,
            symbol="BTCUSDT",
        )
    """

    def __init__(self, config: SignalConfig):
        """
        初始化信号生成器

        Args:
            config: 信号配置
        """
        self.config = config
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_type: Optional[SignalType] = None

    def generate(
        self,
        predictions: Union[np.ndarray, pd.Series],
        timestamps: Union[List[datetime], pd.DatetimeIndex],
        prices: Optional[Union[np.ndarray, pd.Series]] = None,
        symbol: str = "BTCUSDT",
    ) -> List[Signal]:
        """
        生成交易信号

        Args:
            predictions: 预测值序列
            timestamps: 时间戳序列
            prices: 价格序列 (可选)
            symbol: 交易标的

        Returns:
            信号列表
        """
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamps = timestamps.to_pydatetime().tolist()
        if isinstance(prices, pd.Series):
            prices = prices.values

        n = len(predictions)
        if len(timestamps) != n:
            raise ValueError("Predictions and timestamps must have same length")

        signals = []

        # Z-score 标准化
        if self.config.use_zscore:
            predictions = self._apply_zscore(predictions)

        for i in range(n):
            pred = predictions[i]
            ts = timestamps[i]
            price = prices[i] if prices is not None else None

            signal = self._generate_single(pred, ts, price, symbol, i)
            if signal is not None:
                signals.append(signal)

        return signals

    def _generate_single(
        self,
        prediction: float,
        timestamp: datetime,
        price: Optional[float],
        symbol: str,
        idx: int,
    ) -> Optional[Signal]:
        """生成单个信号"""
        if np.isnan(prediction):
            return None

        # 检查冷却期
        if self._last_signal_time is not None:
            # 简化处理，实际应该用时间差
            pass

        # 确定信号类型
        if prediction > self.config.long_threshold:
            signal_type = SignalType.LONG
            strength = min(prediction / self.config.long_threshold, 1.0)
        elif prediction < self.config.short_threshold:
            signal_type = SignalType.SHORT
            strength = min(abs(prediction) / abs(self.config.short_threshold), 1.0)
        else:
            return None  # 不产生信号

        # 检查最小强度
        if strength < self.config.min_strength:
            return None

        self._last_signal_time = timestamp
        self._last_signal_type = signal_type

        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            price=price,
            prediction=prediction,
            metadata={"idx": idx},
        )

    def _apply_zscore(self, predictions: np.ndarray) -> np.ndarray:
        """
        应用 Z-score 标准化

        改进:
        - 支持自适应窗口，数据不足时使用可用数据
        - 使用 zscore_min_periods 作为最小数据量要求
        """
        window = self.config.zscore_window
        min_periods = self.config.zscore_min_periods
        n = len(predictions)
        result = np.full_like(predictions, np.nan, dtype=float)

        for i in range(n):
            # 计算可用窗口大小
            available_window = min(window, i)

            # 需要至少 min_periods 个数据点
            if available_window < min_periods:
                continue

            window_data = predictions[max(0, i - window):i]

            # 过滤掉 NaN
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) < min_periods:
                continue

            mean = np.mean(valid_data)
            std = np.std(valid_data)

            if std > 1e-10:  # 避免除以接近0的数
                result[i] = (predictions[i] - mean) / std
            else:
                result[i] = 0.0  # 标准差为0时返回0

        return result

    def generate_from_model(
        self,
        model,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> List[Signal]:
        """
        从模型生成信号

        Args:
            model: 训练好的模型 (或 ModelTrainer)
            features: 特征数据
            prices: 价格数据
            symbol: 交易标的

        Returns:
            信号列表
        """
        # 预测
        if hasattr(model, "predict"):
            predictions = model.predict(features)
        else:
            raise ValueError("Model must have predict method")

        # 生成信号
        timestamps = features.index.to_pydatetime().tolist()
        price_values = prices["close"].values if "close" in prices.columns else None

        return self.generate(
            predictions=predictions,
            timestamps=timestamps,
            prices=price_values,
            symbol=symbol,
        )


def signals_to_dataframe(signals: List[Signal]) -> pd.DataFrame:
    """
    将信号列表转换为 DataFrame

    Args:
        signals: 信号列表

    Returns:
        信号 DataFrame
    """
    if not signals:
        return pd.DataFrame()

    data = [s.to_dict() for s in signals]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def filter_conflicting_signals(
    signals: List[Signal],
    cooldown: int = 12,
) -> List[Signal]:
    """
    过滤冲突的信号

    Args:
        signals: 信号列表
        cooldown: 冷却期 (bars)

    Returns:
        过滤后的信号列表
    """
    if not signals:
        return []

    # 按时间排序
    signals = sorted(signals, key=lambda s: s.timestamp)

    filtered = [signals[0]]
    last_signal = signals[0]

    for signal in signals[1:]:
        # 简化处理：只保留不同方向的信号
        if signal.signal_type != last_signal.signal_type:
            filtered.append(signal)
            last_signal = signal

    return filtered
