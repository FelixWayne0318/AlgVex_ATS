"""
订单追踪器测试

测试 AlgVexOrderTracker 的订单生命周期管理
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.execution.order_tracker import (
    AlgVexOrderTracker,
    OrderState,
    TrackedOrder,
)


class TestTrackedOrder:
    """TrackedOrder 数据类测试"""

    def test_order_creation(self):
        """测试订单创建"""
        order = TrackedOrder(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert order.client_order_id == "test_001"
        assert order.trading_pair == "BTCUSDT"
        assert order.side == "BUY"
        assert order.amount == Decimal("0.1")
        assert order.price == Decimal("50000")
        assert order.state == OrderState.PENDING_CREATE

    def test_is_active(self):
        """测试订单活跃状态"""
        order = TrackedOrder(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 初始状态是活跃的
        assert order.is_active is True
        assert order.is_done is False

        # OPEN 状态也是活跃的
        order.state = OrderState.OPEN
        assert order.is_active is True

        # PARTIALLY_FILLED 也是活跃的
        order.state = OrderState.PARTIALLY_FILLED
        assert order.is_active is True

        # FILLED 是完成状态
        order.state = OrderState.FILLED
        assert order.is_active is False
        assert order.is_done is True

    def test_fill_percentage(self):
        """测试成交百分比"""
        order = TrackedOrder(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("1"),
            filled_amount=Decimal("0.5"),
        )

        assert order.fill_percentage == 50.0

        # 测试零数量情况
        order.amount = Decimal("0")
        assert order.fill_percentage == 0.0

    def test_to_dict(self):
        """测试转换为字典"""
        order = TrackedOrder(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            signal_id="sig_001",
        )

        d = order.to_dict()

        assert d["client_order_id"] == "test_001"
        assert d["trading_pair"] == "BTCUSDT"
        assert d["side"] == "BUY"
        assert d["amount"] == "0.1"
        assert d["price"] == "50000"
        assert d["signal_id"] == "sig_001"
        assert d["state"] == "pending_create"
        assert d["is_active"] is True


class TestAlgVexOrderTracker:
    """AlgVexOrderTracker 测试"""

    @pytest.fixture
    def tracker(self):
        """创建追踪器实例"""
        return AlgVexOrderTracker(order_timeout=5.0, max_history_size=100)

    def test_track_order(self, tracker):
        """测试订单追踪"""
        order = tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            signal_id="sig_001",
        )

        assert order.client_order_id == "test_001"
        assert order.signal_id == "sig_001"
        assert tracker.get_order("test_001") is order

    def test_update_order(self, tracker):
        """测试订单更新"""
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 更新订单
        updated = tracker.update_order(
            client_order_id="test_001",
            exchange_order_id="exch_001",
            state=OrderState.OPEN,
        )

        assert updated.exchange_order_id == "exch_001"
        assert updated.state == OrderState.OPEN

    def test_update_order_to_filled(self, tracker):
        """测试订单更新为成交状态"""
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 成交
        tracker.update_order(
            client_order_id="test_001",
            state=OrderState.FILLED,
            filled_amount=Decimal("0.1"),
            average_price=Decimal("50000"),
        )

        # 订单应该移到历史
        assert "test_001" not in tracker._active_orders
        assert len(tracker._completed_orders) == 1

        # 但仍然可以通过 get_order 获取
        order = tracker.get_order("test_001")
        assert order is not None
        assert order.state == OrderState.FILLED

    def test_signal_order_mapping(self, tracker):
        """测试信号-订单映射"""
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
            signal_id="sig_001",
        )

        # 通过信号ID获取订单
        order = tracker.get_order_by_signal("sig_001")
        assert order is not None
        assert order.client_order_id == "test_001"

        # 通过订单ID获取信号ID
        signal_id = tracker.get_signal_by_order("test_001")
        assert signal_id == "sig_001"

    def test_get_active_orders(self, tracker):
        """测试获取活跃订单"""
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )
        tracker.track_order(
            client_order_id="test_002",
            trading_pair="ETHUSDT",
            side="SELL",
            amount=Decimal("1"),
        )

        active = tracker.get_active_orders()
        assert len(active) == 2

        # 获取特定交易对的订单
        btc_orders = tracker.get_active_orders_for_pair("BTCUSDT")
        assert len(btc_orders) == 1
        assert btc_orders[0].client_order_id == "test_001"

    def test_cancel_order(self, tracker):
        """测试取消订单"""
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        result = tracker.cancel_order("test_001", reason="user_request")
        assert result is True

        order = tracker.get_order("test_001")
        assert order.state == OrderState.CANCELLED
        assert order.error_message == "user_request"

    def test_cancel_nonexistent_order(self, tracker):
        """测试取消不存在的订单"""
        result = tracker.cancel_order("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_timeout_orders(self, tracker):
        """测试超时订单检查"""
        import time

        tracker._order_timeout = 0.1  # 100ms 超时

        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 等待超时
        await asyncio.sleep(0.2)

        timeout_orders = await tracker.check_timeout_orders()
        assert "test_001" in timeout_orders

        order = tracker.get_order("test_001")
        assert order.state == OrderState.EXPIRED

    def test_get_statistics(self, tracker):
        """测试统计信息"""
        # 创建并完成一些订单
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )
        tracker.update_order("test_001", state=OrderState.FILLED)

        tracker.track_order(
            client_order_id="test_002",
            trading_pair="ETHUSDT",
            side="SELL",
            amount=Decimal("1"),
        )
        tracker.cancel_order("test_002")

        stats = tracker.get_statistics()

        assert stats["completed_orders"] == 2
        assert stats["active_orders"] == 0
        assert "filled" in stats["state_distribution"]
        assert "cancelled" in stats["state_distribution"]

    def test_order_complete_callback(self):
        """测试订单完成回调"""
        completed_orders = []

        def on_complete(order):
            completed_orders.append(order)

        tracker = AlgVexOrderTracker(on_order_complete=on_complete)

        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )
        tracker.update_order("test_001", state=OrderState.FILLED)

        assert len(completed_orders) == 1
        assert completed_orders[0].client_order_id == "test_001"

    def test_history_size_limit(self, tracker):
        """测试历史大小限制"""
        tracker._max_history_size = 5

        # 创建并完成超过限制的订单
        for i in range(10):
            tracker.track_order(
                client_order_id=f"test_{i:03d}",
                trading_pair="BTCUSDT",
                side="BUY",
                amount=Decimal("0.1"),
            )
            tracker.update_order(f"test_{i:03d}", state=OrderState.FILLED)

        # 历史应该只保留最新的5个
        assert len(tracker._completed_orders) == 5
        assert tracker._completed_orders[-1].client_order_id == "test_009"

    @pytest.mark.asyncio
    async def test_recover_from_exchange(self, tracker):
        """测试从交易所恢复状态"""
        # 创建待恢复的订单
        tracker.track_order(
            client_order_id="test_001",
            trading_pair="BTCUSDT",
            side="BUY",
            amount=Decimal("0.1"),
        )

        # 模拟交易所数据
        exchange_orders = {
            "test_001": {
                "status": "FILLED",
                "executedQty": "0.1",
                "avgPrice": "50000",
            }
        }

        recovered = await tracker.recover_from_exchange(exchange_orders)

        assert recovered == 1

        order = tracker.get_order("test_001")
        assert order.state == OrderState.FILLED
        assert order.filled_amount == Decimal("0.1")
