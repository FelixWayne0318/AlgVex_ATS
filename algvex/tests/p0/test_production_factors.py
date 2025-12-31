"""
P0 验收测试: 生产因子计算 (无 Qlib 依赖)

验收标准:
- MVP-11 因子全部可计算
- 因子计算无 Qlib 依赖
- 因子值在合理范围内
- 因子计算性能达标
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.production.factor_engine import FactorEngine


class TestMvp11Factors:
    """测试 MVP-11 因子"""

    @pytest.fixture
    def sample_klines(self):
        """创建测试 K 线数据"""
        # 创建 288 * 30 = 8640 个 5 分钟 bars (30 天数据)
        n_bars = 288 * 30
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

        # 生成模拟价格数据
        np.random.seed(42)
        close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
        high = close + np.abs(np.random.randn(n_bars) * 50)
        low = close - np.abs(np.random.randn(n_bars) * 50)
        open_price = close + np.random.randn(n_bars) * 30

        return pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 10000, n_bars),
        }, index=dates)

    @pytest.fixture
    def sample_oi(self):
        """创建测试持仓量数据"""
        n_bars = 288 * 30
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

        np.random.seed(43)
        oi = 1000000 + np.cumsum(np.random.randn(n_bars) * 10000)

        return pd.DataFrame({
            "open_interest": oi,
        }, index=dates)

    @pytest.fixture
    def sample_funding(self):
        """创建测试资金费率数据"""
        # 每 8 小时一次资金费率
        n_bars = 30 * 3  # 30 天，每天 3 次
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="8h")

        np.random.seed(44)
        funding = np.random.randn(n_bars) * 0.001  # 约 0.1% 的资金费率

        return pd.DataFrame({
            "funding_rate": funding,
        }, index=dates)

    @pytest.fixture
    def factor_engine(self):
        """创建因子引擎"""
        return FactorEngine()

    # ==================== 动量因子测试 ====================

    def test_return_5m(self, factor_engine, sample_klines):
        """测试 return_5m 因子"""
        result = factor_engine.compute_factor("return_5m", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # 收益率应该在合理范围内
        assert result.abs().max() < 0.5  # 单根 K 线收益率不应超过 50%

    def test_return_1h(self, factor_engine, sample_klines):
        """测试 return_1h 因子"""
        result = factor_engine.compute_factor("return_1h", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # 1 小时收益率应该在合理范围内
        assert result.abs().max() < 1.0

    def test_ma_cross(self, factor_engine, sample_klines):
        """测试 ma_cross 因子"""
        result = factor_engine.compute_factor("ma_cross", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # MA 交叉比率应该在合理范围内
        assert result.abs().max() < 1.0

    def test_breakout_20d(self, factor_engine, sample_klines):
        """测试 breakout_20d 因子"""
        result = factor_engine.compute_factor("breakout_20d", klines=sample_klines)

        assert result is not None
        assert len(result) > 0

    def test_trend_strength(self, factor_engine, sample_klines):
        """测试 trend_strength 因子 (ADX)"""
        result = factor_engine.compute_factor("trend_strength", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # ADX 范围是 0-100
        assert result.min() >= 0
        assert result.max() <= 100

    # ==================== 波动率因子测试 ====================

    def test_atr_288(self, factor_engine, sample_klines):
        """测试 atr_288 因子"""
        result = factor_engine.compute_factor("atr_288", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # ATR 应该为正
        assert result.min() >= 0

    def test_realized_vol_1d(self, factor_engine, sample_klines):
        """测试 realized_vol_1d 因子"""
        result = factor_engine.compute_factor("realized_vol_1d", klines=sample_klines)

        assert result is not None
        assert len(result) > 0
        # 波动率应该为正
        assert result.min() >= 0

    def test_vol_regime(self, factor_engine, sample_klines):
        """测试 vol_regime 因子"""
        result = factor_engine.compute_factor("vol_regime", klines=sample_klines)

        assert result is not None
        assert len(result) > 0

    # ==================== 订单流因子测试 ====================

    def test_oi_change_rate(self, factor_engine, sample_oi):
        """测试 oi_change_rate 因子"""
        result = factor_engine.compute_factor("oi_change_rate", oi=sample_oi)

        assert result is not None
        assert len(result) > 0
        # OI 变化率应该在合理范围内
        assert result.abs().max() < 1.0

    def test_funding_momentum(self, factor_engine, sample_funding):
        """测试 funding_momentum 因子"""
        result = factor_engine.compute_factor("funding_momentum", funding=sample_funding)

        assert result is not None
        assert len(result) > 0

    def test_oi_funding_divergence(self, factor_engine, sample_oi, sample_funding):
        """测试 oi_funding_divergence 因子"""
        result = factor_engine.compute_factor(
            "oi_funding_divergence",
            oi=sample_oi,
            funding=sample_funding,
        )

        assert result is not None
        assert len(result) > 0


class TestFactorEnginePerformance:
    """测试因子计算性能"""

    @pytest.fixture
    def large_klines(self):
        """创建大规模 K 线数据"""
        # 1 年数据
        n_bars = 288 * 365
        dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min")

        np.random.seed(42)
        close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)

        return pd.DataFrame({
            "open": close + np.random.randn(n_bars) * 30,
            "high": close + np.abs(np.random.randn(n_bars) * 50),
            "low": close - np.abs(np.random.randn(n_bars) * 50),
            "close": close,
            "volume": np.random.randint(100, 10000, n_bars),
        }, index=dates)

    @pytest.fixture
    def factor_engine(self):
        """创建因子引擎"""
        return FactorEngine()

    def test_momentum_factors_performance(self, factor_engine, large_klines):
        """测试动量因子计算性能"""
        momentum_factors = ["return_5m", "return_1h", "ma_cross", "breakout_20d", "trend_strength"]

        for factor_id in momentum_factors:
            start_time = time.time()
            result = factor_engine.compute_factor(factor_id, klines=large_klines)
            elapsed = time.time() - start_time

            assert result is not None
            # 因子计算应该在 5 秒内完成
            assert elapsed < 5.0, f"{factor_id} 计算耗时 {elapsed:.2f}s 超过阈值"

    def test_volatility_factors_performance(self, factor_engine, large_klines):
        """测试波动率因子计算性能"""
        vol_factors = ["atr_288", "realized_vol_1d", "vol_regime"]

        for factor_id in vol_factors:
            start_time = time.time()
            result = factor_engine.compute_factor(factor_id, klines=large_klines)
            elapsed = time.time() - start_time

            assert result is not None
            assert elapsed < 5.0, f"{factor_id} 计算耗时 {elapsed:.2f}s 超过阈值"


class TestFactorEngineRobustness:
    """测试因子引擎鲁棒性"""

    @pytest.fixture
    def factor_engine(self):
        """创建因子引擎"""
        return FactorEngine()

    def test_empty_data(self, factor_engine):
        """测试空数据处理"""
        empty_klines = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # 应该返回空结果或抛出明确的异常
        try:
            result = factor_engine.compute_factor("return_5m", klines=empty_klines)
            assert result is None or len(result) == 0
        except ValueError:
            pass  # 抛出 ValueError 也是合理的

    def test_nan_handling(self, factor_engine):
        """测试 NaN 值处理"""
        n_bars = 100
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

        klines = pd.DataFrame({
            "open": [50000.0] * n_bars,
            "high": [50100.0] * n_bars,
            "low": [49900.0] * n_bars,
            "close": [50000.0] * n_bars,
            "volume": [1000] * n_bars,
        }, index=dates)

        # 插入一些 NaN
        klines.loc[klines.index[10], "close"] = np.nan
        klines.loc[klines.index[20], "close"] = np.nan

        result = factor_engine.compute_factor("return_5m", klines=klines)

        # 结果应该处理了 NaN
        assert result is not None

    def test_insufficient_data(self, factor_engine):
        """测试数据不足的情况"""
        # 只有 10 个 bars，不足以计算 288 周期的因子
        n_bars = 10
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

        klines = pd.DataFrame({
            "open": [50000.0] * n_bars,
            "high": [50100.0] * n_bars,
            "low": [49900.0] * n_bars,
            "close": [50000.0] * n_bars,
            "volume": [1000] * n_bars,
        }, index=dates)

        # 短周期因子应该可以计算
        result = factor_engine.compute_factor("return_5m", klines=klines)
        assert result is not None

        # 长周期因子可能返回 NaN 或空
        result = factor_engine.compute_factor("atr_288", klines=klines)
        # 不强制要求特定行为，只要不崩溃即可


class TestMvp11FactorList:
    """测试 MVP-11 因子列表完整性"""

    MVP_11_FACTORS = {
        # 动量因子 (5个)
        "return_5m",
        "return_1h",
        "ma_cross",
        "breakout_20d",
        "trend_strength",
        # 波动率因子 (3个)
        "atr_288",
        "realized_vol_1d",
        "vol_regime",
        # 订单流因子 (3个)
        "oi_change_rate",
        "funding_momentum",
        "oi_funding_divergence",
    }

    @pytest.fixture
    def factor_engine(self):
        """创建因子引擎"""
        return FactorEngine()

    def test_all_mvp_factors_registered(self, factor_engine):
        """测试所有 MVP 因子都已注册"""
        registered = factor_engine.get_available_factors()

        for factor_id in self.MVP_11_FACTORS:
            assert factor_id in registered, f"因子 {factor_id} 未注册"

    def test_mvp_factor_count(self, factor_engine):
        """测试 MVP 因子数量"""
        registered = factor_engine.get_available_factors()
        mvp_count = len(set(registered) & self.MVP_11_FACTORS)

        assert mvp_count == 11, f"MVP 因子数量不正确: {mvp_count} != 11"
