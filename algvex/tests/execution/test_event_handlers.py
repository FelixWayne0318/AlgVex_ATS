"""
事件处理器测试

测试 AlgVexEventHandler 的事件处理和 trace 写入
"""

import asyncio
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.execution.event_handlers import (
    AlgVexEventHandler,
    EventType,
    OrderEvent,
    FundingPaymentEvent,
    LiquidationEvent,
    TraceWriter,
)
from core.execution.order_tracker import AlgVexOrderTracker, OrderState
from core.execution.state_synchronizer import PositionManager


class TestOrderEvent:
    """OrderEvent 数据类测试"""

    def test_order_event_creation(self):
        """测试订单事件创建"""
        event = OrderEvent(
            event_type=EventType.ORDER_CREATED,
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            timestamp=1704067200.0,
            side="BUY",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert event.event_type == EventType.ORDER_CREATED
        assert event.client_order_id == "test_001"
        assert event.trading_pair == "BTCUSDT"
        assert event.side == "BUY"

    def test_to_trace(self):
        """测试转换为 trace"""
        event = OrderEvent(
            event_type=EventType.ORDER_FILLED,
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            timestamp=1704067200.0,
            side="BUY",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("50000"),
            fee=Decimal("2"),
            fee_asset="USDT",
        )

        trace = event.to_trace()

        assert trace["type"] == "order_filled"
        assert trace["client_order_id"] == "test_001"
        assert trace["trading_pair"] == "BTCUSDT"
        assert trace["side"] == "BUY"
        assert trace["filled_amount"] == "0.1"
        assert trace["average_price"] == "50000"
        assert trace["fee"] == "2"
        assert trace["fee_asset"] == "USDT"
        assert "timestamp" in trace


class TestFundingPaymentEvent:
    """FundingPaymentEvent 测试"""

    def test_funding_event_creation(self):
        """测试资金费率事件创建"""
        event = FundingPaymentEvent(
            trading_pair="BTCUSDT",
            amount=Decimal("10"),
            funding_rate=Decimal("0.0001"),
            timestamp=1704067200.0,
            position_size=Decimal("0.1"),
        )

        assert event.trading_pair == "BTCUSDT"
        assert event.amount == Decimal("10")
        assert event.funding_rate == Decimal("0.0001")

    def test_to_trace(self):
        """测试转换为 trace"""
        event = FundingPaymentEvent(
            trading_pair="BTCUSDT",
            amount=Decimal("10"),
            funding_rate=Decimal("0.0001"),
            timestamp=1704067200.0,
        )

        trace = event.to_trace()

        assert trace["type"] == "funding_payment"
        assert trace["amount"] == "10"
        assert trace["funding_rate"] == "0.0001"


class TestLiquidationEvent:
    """LiquidationEvent 测试"""

    def test_liquidation_event_creation(self):
        """测试清算事件创建"""
        event = LiquidationEvent(
            trading_pair="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            price=Decimal("45000"),
            timestamp=1704067200.0,
            loss=Decimal("500"),
        )

        assert event.trading_pair == "BTCUSDT"
        assert event.side == "LONG"
        assert event.loss == Decimal("500")

    def test_to_trace(self):
        """测试转换为 trace"""
        event = LiquidationEvent(
            trading_pair="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            price=Decimal("45000"),
            timestamp=1704067200.0,
            loss=Decimal("500"),
        )

        trace = event.to_trace()

        assert trace["type"] == "liquidation"
        assert trace["loss"] == "500"


class TestTraceWriter:
    """TraceWriter 测试"""

    def test_write_to_file(self):
        """测试写入文件"""
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            writer = TraceWriter(output_file=filepath, buffer_size=2)

            # 写入两条 trace
            writer.write({"type": "test1", "value": 1})
            writer.write({"type": "test2", "value": 2})

            # 缓冲区满，应该自动 flush
            with open(filepath) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert json.loads(lines[0])["type"] == "test1"
            assert json.loads(lines[1])["type"] == "test2"

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_manual_flush(self):
        """测试手动 flush"""
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            writer = TraceWriter(output_file=filepath, buffer_size=100)

            writer.write({"type": "test", "value": 1})
            writer.flush()

            with open(filepath) as f:
                lines = f.readlines()

            assert len(lines) == 1

        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_close(self):
        """测试关闭"""
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            filepath = f.name

        try:
            writer = TraceWriter(output_file=filepath, buffer_size=100)

            writer.write({"type": "test", "value": 1})
            writer.close()  # 应该 flush 剩余数据

            with open(filepath) as f:
                lines = f.readlines()

            assert len(lines) == 1

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestAlgVexEventHandler:
    """AlgVexEventHandler 测试"""

    @pytest.fixture
    def handler(self):
        """创建事件处理器"""
        return AlgVexEventHandler()

    @pytest.fixture
    def handler_with_components(self):
        """创建带组件的事件处理器"""
        trace_writer = TraceWriter(buffer_size=100)
        order_tracker = AlgVexOrderTracker()
        position_manager = PositionManager()

        return AlgVexEventHandler(
            trace_writer=trace_writer,
            order_tracker=order_tracker,
            position_manager=position_manager,
        )

    @pytest.mark.asyncio
    async def test_handle_order_created(self, handler):
        """测试处理订单创建事件"""
        await handler.handle_order_created(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            order_type="MARKET",
        )

        stats = handler.get_statistics()
        assert stats["event_counts"]["order_created"] == 1

    @pytest.mark.asyncio
    async def test_handle_order_filled(self, handler_with_components):
        """测试处理订单成交事件"""
        handler = handler_with_components

        # 先追踪订单
        handler.order_tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 处理成交事件
        await handler.handle_order_filled(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("50000"),
            fee=Decimal("2"),
        )

        # 订单应该被更新
        order = handler.order_tracker.get_order("test_001")
        assert order.state == OrderState.FILLED
        assert order.filled_amount == Decimal("0.1")

        # 仓位应该被创建
        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "LONG"
        assert pos.amount == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_handle_order_cancelled(self, handler_with_components):
        """测试处理订单取消事件"""
        handler = handler_with_components

        # 先追踪订单
        handler.order_tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 处理取消事件
        await handler.handle_order_cancelled(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            reason="user_request",
        )

        order = handler.order_tracker.get_order("test_001")
        assert order.state == OrderState.CANCELLED

    @pytest.mark.asyncio
    async def test_handle_order_failed(self, handler):
        """测试处理订单失败事件"""
        alerts = []

        def alert_handler(alert_type, details):
            alerts.append((alert_type, details))

        handler._alert_handler = alert_handler

        await handler.handle_order_failed(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            error_message="Insufficient balance",
        )

        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0][0] == "order_failed"
        assert alerts[0][1]["error"] == "Insufficient balance"

    @pytest.mark.asyncio
    async def test_handle_funding_payment(self, handler):
        """测试处理资金费率支付事件"""
        await handler.handle_funding_payment(
            trading_pair="BTCUSDT",
            amount=Decimal("10"),
            funding_rate=Decimal("0.0001"),
            position_size=Decimal("0.1"),
        )

        stats = handler.get_statistics()
        assert stats["event_counts"]["funding_payment"] == 1

    @pytest.mark.asyncio
    async def test_handle_liquidation(self, handler_with_components):
        """测试处理清算事件"""
        handler = handler_with_components
        alerts = []

        def alert_handler(alert_type, details):
            alerts.append((alert_type, details))

        handler._alert_handler = alert_handler

        # 先创建仓位
        handler.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        # 处理清算
        await handler.handle_liquidation(
            trading_pair="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            price=Decimal("45000"),
            loss=Decimal("500"),
        )

        # 仓位应该被清除
        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos is None

        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0][0] == "liquidation"
        assert alerts[0][1]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_position_update_buy(self, handler_with_components):
        """测试买入更新仓位"""
        handler = handler_with_components

        # 第一笔买入 - 开多
        await handler.handle_order_filled(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("50000"),
        )

        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos.side == "LONG"
        assert pos.amount == Decimal("0.1")
        assert pos.entry_price == Decimal("50000")

        # 第二笔买入 - 加仓
        await handler.handle_order_filled(
            client_order_id="test_002",
            trading_pair="BTCUSDT",
            side="BUY",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("51000"),
        )

        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos.amount == Decimal("0.2")
        # 平均入场价 = (50000 * 0.1 + 51000 * 0.1) / 0.2 = 50500
        assert pos.entry_price == Decimal("50500")

    @pytest.mark.asyncio
    async def test_position_update_sell(self, handler_with_components):
        """测试卖出更新仓位"""
        handler = handler_with_components

        # 开多
        handler.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.2"), Decimal("50000")
        )

        # 卖出 - 平多
        await handler.handle_order_filled(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="SELL",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("51000"),
        )

        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos.amount == Decimal("0.1")

        # 全部卖出
        await handler.handle_order_filled(
            client_order_id="test_002",
            trading_pair="BTCUSDT",
            side="SELL",
            filled_amount=Decimal("0.1"),
            average_price=Decimal("52000"),
        )

        pos = handler.position_manager.get_position("BTCUSDT")
        assert pos is None  # 仓位已平

    def test_get_statistics(self, handler):
        """测试获取统计信息"""
        stats = handler.get_statistics()

        assert "event_counts" in stats
        assert "total_events" in stats
        assert "registered_handlers" in stats

    def test_reset_statistics(self, handler):
        """测试重置统计"""
        handler._event_counts["test"] = 100
        handler.reset_statistics()

        stats = handler.get_statistics()
        assert stats["total_events"] == 0
