"""
可见性检查器测试
"""

from datetime import datetime, timedelta
import pytest

from algvex.shared.visibility_checker import (
    VisibilityChecker,
    VisibilityResult,
    VisibilityViolationError,
    check_visibility,
)


class TestVisibilityChecker:
    """可见性检查器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.checker = VisibilityChecker()

    def test_parse_duration(self):
        """测试时间解析"""
        assert self.checker._parse_duration("5min") == timedelta(minutes=5)
        assert self.checker._parse_duration("1h") == timedelta(hours=1)
        assert self.checker._parse_duration("30s") == timedelta(seconds=30)
        assert self.checker._parse_duration("1d") == timedelta(days=1)

    def test_compute_snapshot_cutoff(self):
        """测试快照截止时间计算"""
        signal_time = datetime(2024, 1, 1, 10, 5, 0)
        cutoff = self.checker.compute_snapshot_cutoff(signal_time)

        # cutoff 应该在 signal_time 之前
        assert cutoff < signal_time

    def test_klines_visibility(self):
        """测试K线数据可见性"""
        # 信号时间需要在bar_close_time + safety_margin之后才能看到该bar
        # 默认 safety_margin = 1s
        bar_close_time = datetime(2024, 1, 1, 10, 5, 0)
        signal_time = datetime(2024, 1, 1, 10, 5, 2)  # 2秒后
        data_time = datetime(2024, 1, 1, 10, 0, 0)

        result = self.checker.check_visibility(
            source_id="klines_5m",
            data_time=data_time,
            signal_time=signal_time,
            bar_close_time=bar_close_time,
        )

        # K线在bar收盘后+safety_margin应该可见
        assert result.is_visible

    def test_oi_delayed_visibility(self):
        """测试OI延迟数据可见性"""
        signal_time = datetime(2024, 1, 1, 10, 5, 0)
        bar_close_time = datetime(2024, 1, 1, 10, 5, 0)
        data_time = datetime(2024, 1, 1, 10, 0, 0)

        result = self.checker.check_visibility(
            source_id="open_interest_5m",
            data_time=data_time,
            signal_time=signal_time,
            bar_close_time=bar_close_time,
        )

        # OI有5分钟延迟，在signal_time应该不可见
        # visible_time = bar_close_time + 5min = 10:10
        # snapshot_cutoff = signal_time - 1s = 10:04:59
        # 10:10 > 10:04:59，所以不可见
        assert not result.is_visible

    def test_get_usable_data_time(self):
        """测试获取可用数据时间"""
        signal_time = datetime(2024, 1, 1, 10, 5, 0)

        # K线数据可用时间
        klines_usable = self.checker.get_usable_data_time("klines_5m", signal_time)
        assert klines_usable < signal_time

        # OI数据可用时间（有5分钟延迟）
        oi_usable = self.checker.get_usable_data_time("open_interest_5m", signal_time)
        assert oi_usable < klines_usable  # OI可用时间更早

    def test_visibility_result_structure(self):
        """测试可见性结果结构"""
        signal_time = datetime(2024, 1, 1, 10, 5, 0)
        data_time = datetime(2024, 1, 1, 10, 0, 0)

        result = self.checker.check_visibility(
            source_id="klines_5m",
            data_time=data_time,
            signal_time=signal_time,
        )

        assert isinstance(result, VisibilityResult)
        assert hasattr(result, "is_visible")
        assert hasattr(result, "data_time")
        assert hasattr(result, "visible_time")
        assert hasattr(result, "snapshot_cutoff")
        assert hasattr(result, "source_id")
        assert hasattr(result, "message")


class TestVisibilityRules:
    """可见性规则测试"""

    def test_no_lookahead(self):
        """测试防止未来信息泄露"""
        checker = VisibilityChecker()

        # 尝试使用未来数据
        signal_time = datetime(2024, 1, 1, 10, 0, 0)
        future_data_time = datetime(2024, 1, 1, 10, 5, 0)  # 5分钟后的数据

        result = checker.check_visibility(
            source_id="klines_5m",
            data_time=future_data_time,
            signal_time=signal_time,
            bar_close_time=datetime(2024, 1, 1, 10, 5, 0),
        )

        # 未来数据应该不可见
        assert not result.is_visible

    def test_funding_rate_visibility(self):
        """测试资金费率可见性"""
        checker = VisibilityChecker()

        # 资金费率在结算时刻可见
        signal_time = datetime(2024, 1, 1, 8, 5, 0)  # 08:05
        funding_time = datetime(2024, 1, 1, 8, 0, 0)  # 08:00 结算

        result = checker.check_visibility(
            source_id="funding_8h",
            data_time=funding_time,
            signal_time=signal_time,
            scheduled_time=funding_time,
        )

        # 结算后的资金费率应该可见
        assert result.is_visible


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
