"""
TripleBarrier 单元测试

测试三重屏障风控的各项功能:
- 止损屏障
- 止盈屏障
- 时间屏障
- 移动止损
- 预设配置
"""

import pytest
from datetime import datetime, timedelta

from algvex.core.backtest.triple_barrier import (
    TripleBarrier,
    TripleBarrierConfig,
    BarrierType,
    BarrierCheckResult,
    PositionState,
    TriggerPrice,
    create_conservative_config,
    create_aggressive_config,
    create_scalping_config,
)


class TestTripleBarrierConfig:
    """测试三重屏障配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = TripleBarrierConfig()
        assert config.stop_loss == 0.03
        assert config.take_profit == 0.06
        assert config.time_limit == 86400
        assert config.trailing_stop is None
        assert config.enabled is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = TripleBarrierConfig(
            stop_loss=0.05,
            take_profit=0.10,
            time_limit=172800,
            trailing_stop=0.02,
        )
        assert config.stop_loss == 0.05
        assert config.take_profit == 0.10
        assert config.time_limit == 172800
        assert config.trailing_stop == 0.02

    def test_invalid_stop_loss(self):
        """测试无效止损配置"""
        with pytest.raises(ValueError, match="stop_loss must be positive"):
            TripleBarrierConfig(stop_loss=-0.03)

    def test_invalid_take_profit(self):
        """测试无效止盈配置"""
        with pytest.raises(ValueError, match="take_profit must be positive"):
            TripleBarrierConfig(take_profit=0)

    def test_invalid_time_limit(self):
        """测试无效时间限制配置"""
        with pytest.raises(ValueError, match="time_limit must be positive"):
            TripleBarrierConfig(time_limit=-100)

    def test_to_dict(self):
        """测试转换为字典"""
        config = TripleBarrierConfig()
        data = config.to_dict()
        assert data["stop_loss"] == 0.03
        assert data["take_profit"] == 0.06
        assert data["time_limit"] == 86400
        assert data["enabled"] is True

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "stop_loss": 0.04,
            "take_profit": 0.08,
            "time_limit": 43200,
            "trailing_stop": 0.015,
        }
        config = TripleBarrierConfig.from_dict(data)
        assert config.stop_loss == 0.04
        assert config.take_profit == 0.08
        assert config.time_limit == 43200


class TestPositionState:
    """测试持仓状态"""

    def test_position_state_creation(self):
        """测试创建持仓状态"""
        pos = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        assert pos.symbol == "BTCUSDT"
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.highest_price == 50000.0
        assert pos.lowest_price == 50000.0

    def test_position_state_with_extremes(self):
        """测试带极值的持仓状态"""
        pos = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
            highest_price=52000.0,
            lowest_price=48000.0,
        )
        assert pos.highest_price == 52000.0
        assert pos.lowest_price == 48000.0


class TestTripleBarrierStopLoss:
    """测试止损屏障"""

    @pytest.fixture
    def barrier(self):
        """创建测试用屏障"""
        config = TripleBarrierConfig(
            stop_loss=0.03,  # 3%
            take_profit=0.06,
            time_limit=86400,
        )
        return TripleBarrier(config)

    @pytest.fixture
    def long_position(self):
        """创建多头持仓"""
        return PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )

    @pytest.fixture
    def short_position(self):
        """创建空头持仓"""
        return PositionState(
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )

    def test_long_stop_loss_not_triggered(self, barrier, long_position):
        """测试多头止损未触发"""
        result = barrier.check(
            position=long_position,
            current_price=49000.0,  # -2%, 未到止损
            current_time=datetime.now(),
        )
        assert result.triggered is False
        assert result.barrier_type == BarrierType.NONE

    def test_long_stop_loss_triggered(self, barrier, long_position):
        """测试多头止损触发"""
        result = barrier.check(
            position=long_position,
            current_price=48000.0,  # -4%, 超过3%止损
            current_time=datetime.now(),
        )
        assert result.triggered is True
        assert result.barrier_type == BarrierType.STOP_LOSS

    def test_short_stop_loss_triggered(self, barrier, short_position):
        """测试空头止损触发"""
        result = barrier.check(
            position=short_position,
            current_price=52000.0,  # +4%, 空头亏损超过3%
            current_time=datetime.now(),
        )
        assert result.triggered is True
        assert result.barrier_type == BarrierType.STOP_LOSS


class TestTripleBarrierTakeProfit:
    """测试止盈屏障"""

    @pytest.fixture
    def barrier(self):
        """创建测试用屏障"""
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.06,  # 6%
            time_limit=86400,
        )
        return TripleBarrier(config)

    @pytest.fixture
    def long_position(self):
        """创建多头持仓"""
        return PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )

    def test_take_profit_not_triggered(self, barrier, long_position):
        """测试止盈未触发"""
        result = barrier.check(
            position=long_position,
            current_price=52000.0,  # +4%, 未到止盈
            current_time=datetime.now(),
        )
        assert result.triggered is False

    def test_long_take_profit_triggered(self, barrier, long_position):
        """测试多头止盈触发"""
        result = barrier.check(
            position=long_position,
            current_price=54000.0,  # +8%, 超过6%止盈
            current_time=datetime.now(),
        )
        assert result.triggered is True
        assert result.barrier_type == BarrierType.TAKE_PROFIT

    def test_short_take_profit_triggered(self, barrier):
        """测试空头止盈触发"""
        short_position = PositionState(
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )
        result = barrier.check(
            position=short_position,
            current_price=46000.0,  # -8%, 空头盈利超过6%
            current_time=datetime.now(),
        )
        assert result.triggered is True
        assert result.barrier_type == BarrierType.TAKE_PROFIT


class TestTripleBarrierTimeLimit:
    """测试时间屏障"""

    @pytest.fixture
    def barrier(self):
        """创建测试用屏障"""
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.06,
            time_limit=3600,  # 1小时
        )
        return TripleBarrier(config)

    def test_time_limit_not_triggered(self, barrier):
        """测试时间限制未触发"""
        entry_time = datetime.now()
        position = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
        )
        result = barrier.check(
            position=position,
            current_price=50000.0,
            current_time=entry_time + timedelta(minutes=30),
        )
        assert result.triggered is False

    def test_time_limit_triggered(self, barrier):
        """测试时间限制触发"""
        entry_time = datetime.now()
        position = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
        )
        result = barrier.check(
            position=position,
            current_price=50000.0,
            current_time=entry_time + timedelta(hours=2),
        )
        assert result.triggered is True
        assert result.barrier_type == BarrierType.TIME_LIMIT


class TestTripleBarrierTrailingStop:
    """测试移动止损"""

    @pytest.fixture
    def barrier(self):
        """创建带移动止损的屏障"""
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.10,
            time_limit=86400,
            trailing_stop=0.02,  # 2%
        )
        return TripleBarrier(config)

    def test_trailing_stop_updates(self, barrier):
        """测试移动止损更新"""
        entry_time = datetime.now()
        position = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
        )

        # 价格上涨到 52000
        barrier.check(
            position=position,
            current_price=52000.0,
            current_time=entry_time + timedelta(hours=1),
        )

        # 移动止损应该被设置
        assert position.trailing_stop_price is not None
        assert position.trailing_stop_price == 52000 * (1 - 0.02)  # 50960

    def test_trailing_stop_triggered(self, barrier):
        """测试移动止损触发"""
        entry_time = datetime.now()
        position = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
            highest_price=52000.0,
            trailing_stop_price=50960.0,  # 52000 * 0.98
        )

        # 价格跌破移动止损
        result = barrier.check(
            position=position,
            current_price=50500.0,  # 低于 50960
            current_time=entry_time + timedelta(hours=1),
        )

        assert result.triggered is True
        assert result.barrier_type == BarrierType.TRAILING_STOP

    def test_short_trailing_stop(self, barrier):
        """测试空头移动止损"""
        entry_time = datetime.now()
        position = PositionState(
            symbol="BTCUSDT",
            side="short",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=entry_time,
        )

        # 价格下跌到 48000
        barrier.check(
            position=position,
            current_price=48000.0,
            current_time=entry_time + timedelta(hours=1),
        )

        # 空头移动止损应该在上方
        assert position.trailing_stop_price is not None
        assert position.trailing_stop_price == 48000 * (1 + 0.02)  # 48960


class TestBarrierCheckResult:
    """测试屏障检查结果"""

    def test_result_not_triggered(self):
        """测试未触发结果"""
        result = BarrierCheckResult(
            triggered=False,
            barrier_type=BarrierType.NONE,
        )
        assert result.triggered is False
        assert result.barrier_type == BarrierType.NONE

    def test_result_to_dict(self):
        """测试结果转换为字典"""
        result = BarrierCheckResult(
            triggered=True,
            barrier_type=BarrierType.STOP_LOSS,
            trigger_price=48500.0,
            current_price=48000.0,
            pnl_percentage=-0.04,
            message="Stop loss triggered at -4.00%",
        )
        data = result.to_dict()
        assert data["triggered"] is True
        assert data["barrier_type"] == "stop_loss"
        assert data["trigger_price"] == 48500.0


class TestBarrierDisabled:
    """测试禁用屏障"""

    def test_disabled_barrier(self):
        """测试禁用屏障不触发"""
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.06,
            enabled=False,
        )
        barrier = TripleBarrier(config)

        position = PositionState(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            entry_time=datetime.now(),
        )

        # 即使价格跌破止损，也不应该触发
        result = barrier.check(
            position=position,
            current_price=40000.0,  # -20%
            current_time=datetime.now(),
        )
        assert result.triggered is False


class TestPresetConfigs:
    """测试预设配置"""

    def test_conservative_config(self):
        """测试保守配置"""
        config = create_conservative_config()
        assert config.stop_loss == 0.02
        assert config.take_profit == 0.04
        assert config.time_limit == 43200  # 12h
        assert config.trailing_stop == 0.015

    def test_aggressive_config(self):
        """测试激进配置"""
        config = create_aggressive_config()
        assert config.stop_loss == 0.05
        assert config.take_profit == 0.15
        assert config.time_limit == 172800  # 48h
        assert config.trailing_stop == 0.03

    def test_scalping_config(self):
        """测试剥头皮配置"""
        config = create_scalping_config()
        assert config.stop_loss == 0.01
        assert config.take_profit == 0.02
        assert config.time_limit == 3600  # 1h
        assert config.trailing_stop == 0.005


class TestCalculateBarrierPrices:
    """测试屏障价格计算"""

    @pytest.fixture
    def barrier(self):
        """创建测试用屏障"""
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.06,
        )
        return TripleBarrier(config)

    def test_long_barrier_prices(self, barrier):
        """测试多头屏障价格"""
        prices = barrier.calculate_barrier_prices(
            entry_price=50000.0,
            side="long",
        )
        assert prices["stop_loss_price"] == 50000 * (1 - 0.03)  # 48500
        assert prices["take_profit_price"] == 50000 * (1 + 0.06)  # 53000

    def test_short_barrier_prices(self, barrier):
        """测试空头屏障价格"""
        prices = barrier.calculate_barrier_prices(
            entry_price=50000.0,
            side="short",
        )
        assert prices["stop_loss_price"] == 50000 * (1 + 0.03)  # 51500
        assert prices["take_profit_price"] == 50000 * (1 - 0.06)  # 47000
