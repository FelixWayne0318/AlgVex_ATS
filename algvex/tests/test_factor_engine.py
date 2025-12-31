"""
因子计算引擎测试
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from algvex.production.factor_engine import MVPFactorEngine, FactorValue


class TestMVPFactorEngine:
    """MVP因子引擎测试类"""

    def setup_method(self):
        """测试前准备"""
        self.engine = MVPFactorEngine()

        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=1000, freq="5min")

        self.klines = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(1000) * 0.1),
                "high": 101 + np.cumsum(np.random.randn(1000) * 0.1),
                "low": 99 + np.cumsum(np.random.randn(1000) * 0.1),
                "close": 100.5 + np.cumsum(np.random.randn(1000) * 0.1),
                "volume": 1000 + np.random.randint(0, 500, 1000),
            },
            index=dates,
        )
        # 修正 high/low
        self.klines["high"] = self.klines[["open", "high", "close"]].max(axis=1)
        self.klines["low"] = self.klines[["open", "low", "close"]].min(axis=1)

        self.oi = pd.DataFrame(
            {"open_interest": 50000 + np.cumsum(np.random.randn(1000) * 100)},
            index=dates,
        )

        funding_dates = pd.date_range("2024-01-01", periods=100, freq="8h")
        self.funding = pd.DataFrame(
            {"funding_rate": np.random.randn(100) * 0.0001},
            index=funding_dates,
        )

        self.signal_time = datetime(2024, 1, 4, 10, 5, 0)

    def test_factor_definitions(self):
        """测试因子定义"""
        definitions = self.engine.factor_definitions

        # 应该有11个MVP因子
        assert len(definitions) == 11

        # 检查因子族
        families = {d["family"] for d in definitions.values()}
        assert "momentum" in families
        assert "volatility" in families
        assert "order_flow" in families

    def test_compute_return_5m(self):
        """测试5分钟收益率因子"""
        result = self.engine._compute_return(self.klines, 1, self.signal_time)

        assert isinstance(result, FactorValue)
        assert result.factor_id == "return_1"
        assert result.is_valid
        assert not np.isnan(result.value)

    def test_compute_return_1h(self):
        """测试1小时收益率因子"""
        result = self.engine._compute_return(self.klines, 12, self.signal_time)

        assert result.factor_id == "return_12"
        assert result.is_valid
        assert not np.isnan(result.value)

    def test_compute_ma_cross(self):
        """测试均线交叉因子"""
        result = self.engine._compute_ma_cross(self.klines, self.signal_time)

        assert result.factor_id == "ma_cross"
        assert result.is_valid
        assert not np.isnan(result.value)

    def test_compute_atr(self):
        """测试ATR因子"""
        result = self.engine._compute_atr(self.klines, 288, self.signal_time)

        assert result.factor_id == "atr_288"
        assert result.is_valid
        assert result.value > 0  # ATR应该为正

    def test_compute_realized_vol(self):
        """测试已实现波动率因子"""
        result = self.engine._compute_realized_vol(self.klines, 288, self.signal_time)

        assert result.factor_id == "realized_vol_1d"
        assert result.is_valid
        assert result.value >= 0  # 波动率应该非负

    def test_compute_oi_change(self):
        """测试持仓量变化率因子"""
        result = self.engine._compute_oi_change(self.oi, self.signal_time)

        assert result.factor_id == "oi_change_rate"
        assert result.is_valid
        assert not np.isnan(result.value)

    def test_compute_funding_momentum(self):
        """测试资金费率动量因子"""
        result = self.engine._compute_funding_momentum(self.funding, self.signal_time)

        assert result.factor_id == "funding_momentum"
        assert result.is_valid
        assert not np.isnan(result.value)

    def test_compute_all_factors(self):
        """测试计算所有因子"""
        factors = self.engine.compute_all_factors(
            klines=self.klines,
            oi=self.oi,
            funding=self.funding,
            signal_time=self.signal_time,
        )

        # 应该有11个因子
        assert len(factors) == 11

        # 检查每个因子
        for factor_id, factor_value in factors.items():
            assert isinstance(factor_value, FactorValue)
            assert factor_value.factor_id == factor_id

    def test_insufficient_data(self):
        """测试数据不足的情况"""
        short_klines = self.klines.iloc[:10]  # 只有10条数据

        result = self.engine._compute_ma_cross(short_klines, self.signal_time)

        # 数据不足应返回无效因子
        assert not result.is_valid
        assert np.isnan(result.value)

    def test_factor_value_structure(self):
        """测试因子值结构"""
        result = self.engine._compute_return(self.klines, 1, self.signal_time)

        assert hasattr(result, "factor_id")
        assert hasattr(result, "value")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "data_time")
        assert hasattr(result, "visible_time")
        assert hasattr(result, "is_valid")


class TestFactorConstants:
    """因子常量测试"""

    def test_time_constants(self):
        """测试时间常量"""
        assert MVPFactorEngine.BARS_PER_HOUR == 12
        assert MVPFactorEngine.BARS_PER_DAY == 288
        assert MVPFactorEngine.BARS_PER_WEEK == 2016
        assert MVPFactorEngine.BARS_PER_20D == 5760


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
