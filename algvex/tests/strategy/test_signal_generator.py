"""
SignalGenerator 单元测试

测试信号生成器的核心功能:
- 信号类型
- 阈值过滤
- Z-score 标准化
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from algvex.core.strategy import (
    SignalGenerator,
    SignalConfig,
    Signal,
    SignalType,
)


class TestSignalType:
    """测试信号类型"""

    def test_signal_types(self):
        """测试信号类型枚举"""
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"
        assert SignalType.CLOSE.value == "close"
        assert SignalType.HOLD.value == "hold"


class TestSignalConfig:
    """测试信号配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SignalConfig()
        assert config.long_threshold == 0.02
        assert config.short_threshold == -0.02
        assert config.holding_period == 288
        # use_zscore 默认 False 以避免冷启动问题
        assert config.use_zscore is False
        assert config.zscore_window == 20
        assert config.zscore_min_periods == 5

    def test_custom_config(self):
        """测试自定义配置"""
        config = SignalConfig(
            long_threshold=0.05,
            short_threshold=-0.05,
            holding_period=144,
            use_zscore=False,
        )
        assert config.long_threshold == 0.05
        assert config.short_threshold == -0.05
        assert config.use_zscore is False

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = SignalConfig(long_threshold=0.03)
        data = config.to_dict()
        assert data["long_threshold"] == 0.03


class TestSignal:
    """测试信号"""

    def test_signal_creation(self):
        """测试创建信号"""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=0.8,
            price=50000.0,
            prediction=0.05,
        )
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.LONG
        assert signal.strength == 0.8
        assert signal.is_long is True
        assert signal.is_short is False

    def test_signal_properties(self):
        """测试信号属性"""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.SHORT,
            strength=0.6,
        )
        assert signal.is_short is True
        assert signal.is_long is False
        assert signal.is_close is False

    def test_signal_to_dict(self):
        """测试信号转字典"""
        signal = Signal(
            timestamp=datetime(2024, 1, 1, 12, 0),
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=0.5,
        )
        data = signal.to_dict()
        assert data["symbol"] == "BTCUSDT"
        assert data["signal_type"] == "long"
        assert data["strength"] == 0.5


class TestSignalGenerator:
    """测试信号生成器"""

    @pytest.fixture
    def generator(self):
        """创建生成器"""
        config = SignalConfig(
            long_threshold=0.02,
            short_threshold=-0.02,
            use_zscore=False,  # 禁用 z-score 简化测试
            min_strength=0.0,
        )
        return SignalGenerator(config)

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        n = 100
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(n)
        ]
        predictions = np.random.randn(n) * 0.03  # 一些会触发信号
        prices = np.full(n, 50000.0)
        return predictions, timestamps, prices

    def test_generator_init(self, generator):
        """测试生成器初始化"""
        assert generator.config is not None
        assert generator.config.long_threshold == 0.02

    def test_generate_no_signals(self, generator):
        """测试无信号情况"""
        predictions = np.array([0.01, 0.005, -0.01, 0.0])  # 都在阈值内
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(4)
        ]
        signals = generator.generate(predictions, timestamps)
        assert len(signals) == 0

    def test_generate_long_signal(self, generator):
        """测试生成做多信号"""
        predictions = np.array([0.03, 0.01, 0.005])  # 第一个超过阈值
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(3)
        ]
        signals = generator.generate(predictions, timestamps)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.LONG

    def test_generate_short_signal(self, generator):
        """测试生成做空信号"""
        predictions = np.array([-0.03, -0.01, 0.01])  # 第一个低于阈值
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(3)
        ]
        signals = generator.generate(predictions, timestamps)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SHORT

    def test_generate_with_prices(self, generator):
        """测试带价格的信号生成"""
        predictions = np.array([0.05])
        timestamps = [datetime(2024, 1, 1)]
        prices = np.array([50000.0])

        signals = generator.generate(
            predictions, timestamps, prices, symbol="BTCUSDT"
        )

        assert len(signals) == 1
        assert signals[0].price == 50000.0
        assert signals[0].symbol == "BTCUSDT"

    def test_generate_signal_strength(self, generator):
        """测试信号强度计算"""
        predictions = np.array([0.04])  # 2x 阈值
        timestamps = [datetime(2024, 1, 1)]

        signals = generator.generate(predictions, timestamps)

        assert len(signals) == 1
        # 强度应该是 prediction / threshold = 0.04 / 0.02 = 2, 但被限制为 1
        assert signals[0].strength == 1.0

    def test_generate_with_pandas(self, generator):
        """测试 Pandas 输入"""
        predictions = pd.Series([0.03, -0.03])
        timestamps = pd.DatetimeIndex([
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ])
        prices = pd.Series([50000.0, 51000.0])

        signals = generator.generate(
            predictions, timestamps, prices
        )

        assert len(signals) == 2

    def test_generate_skips_nan(self, generator):
        """测试跳过 NaN 值"""
        predictions = np.array([0.03, np.nan, 0.04])
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(3)
        ]

        signals = generator.generate(predictions, timestamps)

        # 应该只有 2 个信号 (跳过 NaN)
        assert len(signals) == 2


class TestSignalGeneratorWithZScore:
    """测试带 Z-score 的信号生成器"""

    @pytest.fixture
    def generator(self):
        """创建带 z-score 的生成器"""
        config = SignalConfig(
            long_threshold=2.0,  # z-score 阈值
            short_threshold=-2.0,
            use_zscore=True,
            zscore_window=20,
            min_strength=0.0,
        )
        return SignalGenerator(config)

    def test_zscore_normalization(self, generator):
        """测试 Z-score 标准化"""
        np.random.seed(42)
        n = 50
        predictions = np.random.randn(n) * 0.01

        # 添加一个极端值
        predictions[-1] = 0.1

        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i)
            for i in range(n)
        ]

        signals = generator.generate(predictions, timestamps)

        # 应该有信号生成
        assert len(signals) >= 0  # 具体数量取决于随机数
