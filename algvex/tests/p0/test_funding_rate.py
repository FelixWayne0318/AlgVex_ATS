"""
P0-2 验收测试: 资金费率处理

验收标准:
- 结算时间精确对齐 (0/8/16 UTC)
- 持仓跨越结算时间才支付资金费
- 资金费计算正确
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.funding_rate import (
    FundingRateHandler,
    FundingPayment,
    get_funding_handler,
    reset_funding_handler,
)


class TestFundingRateHandler:
    """测试资金费率处理器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_funding_handler()

    def test_settlement_hours(self):
        """测试结算时间配置"""
        handler = FundingRateHandler()
        assert handler.settlement_hours == [0, 8, 16]

    def test_get_next_settlement_time(self):
        """测试获取下一个结算时间"""
        handler = FundingRateHandler()

        # 凌晨 3 点，下一个结算是 8 点
        current = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        next_settlement = handler.get_next_settlement_time(current)
        assert next_settlement == datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        # 上午 10 点，下一个结算是 16 点
        current = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        next_settlement = handler.get_next_settlement_time(current)
        assert next_settlement == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

        # 晚上 20 点，下一个结算是第二天 0 点
        current = datetime(2024, 1, 15, 20, 0, 0, tzinfo=timezone.utc)
        next_settlement = handler.get_next_settlement_time(current)
        assert next_settlement == datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc)

    def test_get_previous_settlement_time(self):
        """测试获取上一个结算时间"""
        handler = FundingRateHandler()

        # 上午 10 点，上一个结算是 8 点
        current = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        prev_settlement = handler.get_previous_settlement_time(current)
        assert prev_settlement == datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        # 凌晨 3 点，上一个结算是当天 0 点
        current = datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc)
        prev_settlement = handler.get_previous_settlement_time(current)
        assert prev_settlement == datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_get_settlement_times_in_range(self):
        """测试获取时间范围内的结算时间"""
        handler = FundingRateHandler()

        start = datetime(2024, 1, 15, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 20, 0, 0, tzinfo=timezone.utc)

        settlements = handler.get_settlement_times_in_range(start, end)

        # 应该有 8:00 和 16:00
        assert len(settlements) == 2
        assert datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc) in settlements
        assert datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc) in settlements

    def test_position_held_through_settlement(self):
        """测试持仓是否跨越结算时间"""
        handler = FundingRateHandler()

        settlement = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)

        # 入场在结算前，出场在结算后 -> 跨越
        entry = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert handler.position_held_through_settlement(entry, exit_time, settlement) is True

        # 入场和出场都在结算前 -> 未跨越
        entry = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        assert handler.position_held_through_settlement(entry, exit_time, settlement) is False

        # 入场在结算后 -> 未跨越
        entry = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert handler.position_held_through_settlement(entry, exit_time, settlement) is False

        # 无出场时间 (仍持有) -> 跨越
        entry = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        assert handler.position_held_through_settlement(entry, None, settlement) is True

    def test_calculate_funding_payment_long(self):
        """测试多头资金费支付"""
        handler = FundingRateHandler()

        entry = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        current = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

        # 正费率，多头支付
        payment = handler.calculate_funding_payment(
            symbol="BTCUSDT",
            position_value=10000,
            side="long",
            entry_time=entry,
            current_time=current,
            funding_rate=0.0001,  # 0.01%
        )

        assert payment is not None
        assert payment.payment == 1.0  # 10000 * 0.0001 = 1
        assert payment.side == "long"
        assert payment.funding_rate == 0.0001

    def test_calculate_funding_payment_short(self):
        """测试空头资金费支付"""
        handler = FundingRateHandler()

        entry = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        current = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

        # 正费率，空头收取
        payment = handler.calculate_funding_payment(
            symbol="BTCUSDT",
            position_value=10000,
            side="short",
            entry_time=entry,
            current_time=current,
            funding_rate=0.0001,
        )

        assert payment is not None
        assert payment.payment == -1.0  # 空头收取
        assert payment.side == "short"

    def test_no_payment_if_not_held_through(self):
        """测试未跨越结算时间不支付"""
        handler = FundingRateHandler()

        # 入场和出场都在同一个结算周期内 (不跨越16:00)
        entry = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)  # 在16:00之前出场

        payment = handler.calculate_funding_payment(
            symbol="BTCUSDT",
            position_value=10000,
            side="long",
            entry_time=entry,
            current_time=exit_time,
            funding_rate=0.0001,
            exit_time=exit_time,  # 明确指定出场时间
        )

        assert payment is None

    def test_calculate_total_funding(self):
        """测试计算总资金费"""
        handler = FundingRateHandler()

        entry = datetime(2024, 1, 15, 1, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 20, 0, 0, tzinfo=timezone.utc)

        # 两个结算时间: 8:00 和 16:00
        funding_rates = {
            datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc): 0.0001,
            datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc): 0.0002,
        }

        total, payments = handler.calculate_total_funding(
            symbol="BTCUSDT",
            position_value=10000,
            side="long",
            entry_time=entry,
            exit_time=exit_time,
            funding_rates=funding_rates,
        )

        # 10000 * 0.0001 + 10000 * 0.0002 = 1 + 2 = 3
        assert len(payments) == 2
        assert total == 3.0

    def test_validate_backtest_funding(self):
        """测试验证回测资金费"""
        handler = FundingRateHandler()

        entry = datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        funding_rates = {
            datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc): 0.0001,
        }

        # 正确的资金费
        trades = [{
            "symbol": "BTCUSDT",
            "entry_time": entry,
            "exit_time": exit_time,
            "position_value": 10000,
            "side": "long",
            "funding_paid": 1.0,  # 正确
            "funding_rates": funding_rates,
        }]

        is_valid, errors = handler.validate_backtest_funding(trades)
        assert is_valid is True
        assert len(errors) == 0

        # 错误的资金费
        trades[0]["funding_paid"] = 2.0  # 错误
        is_valid, errors = handler.validate_backtest_funding(trades)
        assert is_valid is False
        assert len(errors) == 1

    def test_estimate_funding_cost(self):
        """测试估算资金费成本"""
        handler = FundingRateHandler()

        # 持仓 24 小时，平均费率 0.01%
        cost = handler.estimate_funding_cost(
            position_value=10000,
            side="long",
            holding_hours=24,
            avg_funding_rate=0.0001,
        )

        # 24小时 / 8小时 = 3 个周期
        # 10000 * 0.0001 * 3 = 3
        assert cost == 3.0

    def test_global_singleton(self):
        """测试全局单例"""
        reset_funding_handler()

        h1 = get_funding_handler()
        h2 = get_funding_handler()

        assert h1 is h2
