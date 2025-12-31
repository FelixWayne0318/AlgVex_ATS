"""
AlgVex Controller 测试

测试 AlgVexController 的信号处理和动作生成
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.execution.controllers.algvex_controller import (
    AlgVexController,
    AlgVexControllerConfig,
    MockSignalGenerator,
    Signal,
    PositionSide,
    ActionType,
    PositionConfig,
)


class TestAlgVexControllerConfig:
    """AlgVexControllerConfig 测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = AlgVexControllerConfig()

        assert config.signal_threshold == 0.5
        assert config.leverage == 1
        assert config.stop_loss_pct == Decimal("0.03")
        assert config.take_profit_pct == Decimal("0.06")
        assert config.cooldown_seconds == 60

    def test_custom_config(self):
        """测试自定义配置"""
        config = AlgVexControllerConfig(
            trading_pairs={"BTCUSDT", "ETHUSDT"},
            signal_threshold=0.3,
            max_position_per_pair=Decimal("0.5"),
            leverage=5,
        )

        assert len(config.trading_pairs) == 2
        assert config.signal_threshold == 0.3
        assert config.leverage == 5

    def test_to_dict(self):
        """测试转换为字典"""
        config = AlgVexControllerConfig(
            trading_pairs={"BTCUSDT"},
            signal_threshold=0.4,
        )

        d = config.to_dict()

        assert "trading_pairs" in d
        assert d["signal_threshold"] == 0.4


class TestSignal:
    """Signal 数据类测试"""

    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            signal_id="sig_001",
            trading_pair="BTCUSDT",
            final_signal=0.8,
            confidence=0.9,
        )

        assert signal.signal_id == "sig_001"
        assert signal.trading_pair == "BTCUSDT"
        assert signal.final_signal == 0.8
        assert signal.confidence == 0.9


class TestMockSignalGenerator:
    """MockSignalGenerator 测试"""

    @pytest.mark.asyncio
    async def test_get_signal(self):
        """测试获取信号"""
        generator = MockSignalGenerator()
        generator.set_signal("BTCUSDT", 0.8)

        signal = await generator.get_signal("BTCUSDT")

        assert signal is not None
        assert signal.trading_pair == "BTCUSDT"
        assert signal.final_signal == 0.8

    @pytest.mark.asyncio
    async def test_get_signal_not_set(self):
        """测试获取未设置的信号"""
        generator = MockSignalGenerator()

        signal = await generator.get_signal("BTCUSDT")

        assert signal is None


class TestPositionConfig:
    """PositionConfig 测试"""

    def test_position_config_creation(self):
        """测试仓位配置创建"""
        config = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
            leverage=5,
        )

        assert config.trading_pair == "BTCUSDT"
        assert config.side == PositionSide.LONG
        assert config.amount == Decimal("0.1")
        assert config.leverage == 5

    def test_to_dict(self):
        """测试转换为字典"""
        config = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.SHORT,
            amount=Decimal("0.2"),
            stop_loss=Decimal("52000"),
            take_profit=Decimal("48000"),
        )

        d = config.to_dict()

        assert d["trading_pair"] == "BTCUSDT"
        assert d["side"] == "SHORT"
        assert d["amount"] == "0.2"
        assert d["stop_loss"] == "52000"
        assert d["take_profit"] == "48000"


class TestAlgVexController:
    """AlgVexController 测试"""

    @pytest.fixture
    def controller(self):
        """创建控制器"""
        config = AlgVexControllerConfig(
            trading_pairs={"BTCUSDT", "ETHUSDT"},
            signal_threshold=0.3,
            max_position_per_pair=Decimal("0.1"),
            cooldown_seconds=0,  # 禁用冷却
        )
        generator = MockSignalGenerator()
        return AlgVexController(config=config, signal_generator=generator)

    @pytest.mark.asyncio
    async def test_update_processed_data(self, controller):
        """测试更新处理数据"""
        controller.signal_generator.set_signal("BTCUSDT", 0.8)

        await controller.update_processed_data()

        signal = controller.get_latest_signal("BTCUSDT")
        assert signal is not None
        assert signal.final_signal == 0.8

    def test_determine_actions_no_signal(self, controller):
        """测试无信号时的动作"""
        actions = controller.determine_executor_actions()

        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_determine_actions_strong_long_signal(self, controller):
        """测试强做多信号"""
        controller.signal_generator.set_signal("BTCUSDT", 0.8)
        await controller.update_processed_data()

        actions = controller.determine_executor_actions()

        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CREATE_POSITION
        assert actions[0].position_config.side == PositionSide.LONG

    @pytest.mark.asyncio
    async def test_determine_actions_strong_short_signal(self, controller):
        """测试强做空信号"""
        controller.signal_generator.set_signal("BTCUSDT", -0.8)
        await controller.update_processed_data()

        actions = controller.determine_executor_actions()

        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CREATE_POSITION
        assert actions[0].position_config.side == PositionSide.SHORT

    @pytest.mark.asyncio
    async def test_determine_actions_weak_signal(self, controller):
        """测试弱信号（低于阈值）"""
        controller.signal_generator.set_signal("BTCUSDT", 0.2)  # 低于 0.3 阈值
        await controller.update_processed_data()

        actions = controller.determine_executor_actions()

        # 弱信号不应该触发动作
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_determine_actions_with_existing_position(self, controller):
        """测试有现有仓位时的动作"""
        # 设置现有多头仓位
        existing_pos = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
        )
        controller.update_position("BTCUSDT", existing_pos)

        # 设置同方向强信号
        controller.signal_generator.set_signal("BTCUSDT", 0.8)
        await controller.update_processed_data()

        actions = controller.determine_executor_actions()

        # 同方向不应该再开仓
        assert len(actions) == 0

    @pytest.mark.asyncio
    async def test_determine_actions_reversal(self, controller):
        """测试信号反转平仓"""
        # 设置现有多头仓位
        existing_pos = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
        )
        controller.update_position("BTCUSDT", existing_pos)

        # 设置反向强信号
        controller.signal_generator.set_signal("BTCUSDT", -0.8)
        await controller.update_processed_data()

        actions = controller.determine_executor_actions()

        # 应该先平仓
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CLOSE_POSITION

    def test_calculate_position_size(self, controller):
        """测试仓位大小计算"""
        signal = Signal(
            signal_id="test",
            trading_pair="BTCUSDT",
            final_signal=0.8,
            confidence=1.0,
        )

        size = controller._calculate_position_size(signal)

        # size = max_position * signal_strength * confidence
        # size = 0.1 * 0.8 * 1.0 = 0.08
        assert size == Decimal("0.08")

    def test_calculate_position_size_with_confidence(self, controller):
        """测试带置信度的仓位大小计算"""
        signal = Signal(
            signal_id="test",
            trading_pair="BTCUSDT",
            final_signal=0.8,
            confidence=0.5,
        )

        size = controller._calculate_position_size(signal)

        # size = 0.1 * 0.8 * 0.5 = 0.04
        assert size == Decimal("0.04")

    def test_update_position(self, controller):
        """测试更新仓位"""
        pos = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
        )

        controller.update_position("BTCUSDT", pos)

        retrieved = controller.get_current_position("BTCUSDT")
        assert retrieved is not None
        assert retrieved.side == PositionSide.LONG

    def test_close_position(self, controller):
        """测试关闭仓位"""
        pos = PositionConfig(
            trading_pair="BTCUSDT",
            side=PositionSide.LONG,
            amount=Decimal("0.1"),
        )

        controller.update_position("BTCUSDT", pos)
        controller.update_position("BTCUSDT", None)

        retrieved = controller.get_current_position("BTCUSDT")
        assert retrieved is None

    def test_get_statistics(self, controller):
        """测试获取统计信息"""
        stats = controller.get_statistics()

        assert "config" in stats
        assert "active_pairs" in stats
        assert "current_positions" in stats
        assert "action_counts" in stats

    def test_cooldown(self):
        """测试冷却时间"""
        import time

        config = AlgVexControllerConfig(
            trading_pairs={"BTCUSDT"},
            signal_threshold=0.3,
            cooldown_seconds=10,
        )
        generator = MockSignalGenerator()
        generator.set_signal("BTCUSDT", 0.8)

        controller = AlgVexController(config=config, signal_generator=generator)

        # 手动设置信号
        controller._latest_signals["BTCUSDT"] = Signal(
            signal_id="test",
            trading_pair="BTCUSDT",
            final_signal=0.8,
        )

        # 第一次应该产生动作
        actions_1 = controller.determine_executor_actions()
        assert len(actions_1) == 1

        # 更新信号
        controller._latest_signals["BTCUSDT"] = Signal(
            signal_id="test2",
            trading_pair="BTCUSDT",
            final_signal=0.9,
        )

        # 第二次应该被冷却阻止
        actions_2 = controller.determine_executor_actions()
        assert len(actions_2) == 0

    def test_action_history(self, controller):
        """测试动作历史"""
        # 生成一些动作
        controller._latest_signals["BTCUSDT"] = Signal(
            signal_id="test",
            trading_pair="BTCUSDT",
            final_signal=0.8,
        )

        controller.determine_executor_actions()

        history = controller.get_action_history()
        assert len(history) == 1
        assert history[0]["action_type"] == "create_position"

    @pytest.mark.asyncio
    async def test_multiple_pairs(self, controller):
        """测试多交易对处理"""
        controller.signal_generator.set_signal("BTCUSDT", 0.8)
        controller.signal_generator.set_signal("ETHUSDT", -0.7)

        await controller.update_processed_data()
        actions = controller.determine_executor_actions()

        # 应该有两个动作
        assert len(actions) == 2

        pairs = {a.trading_pair for a in actions}
        assert "BTCUSDT" in pairs
        assert "ETHUSDT" in pairs
