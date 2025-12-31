"""
订单追踪器

集成 Hummingbot InFlightOrder 进行订单生命周期管理
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """订单状态"""
    PENDING_CREATE = "pending_create"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TrackedOrder:
    """追踪的订单"""
    client_order_id: str
    exchange_order_id: Optional[str] = None
    signal_id: Optional[str] = None
    trading_pair: str = ""
    side: str = ""  # "BUY" or "SELL"
    order_type: str = "MARKET"
    amount: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    filled_amount: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    state: OrderState = OrderState.PENDING_CREATE
    creation_timestamp: float = field(default_factory=time.time)
    last_update_timestamp: float = field(default_factory=time.time)
    trade_fees: Decimal = Decimal("0")
    error_message: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """订单是否活跃"""
        return self.state in (
            OrderState.PENDING_CREATE,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED
        )

    @property
    def is_done(self) -> bool:
        """订单是否完成"""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.FAILED,
            OrderState.EXPIRED
        )

    @property
    def fill_percentage(self) -> float:
        """成交百分比"""
        if self.amount == Decimal("0"):
            return 0.0
        return float(self.filled_amount / self.amount * 100)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "signal_id": self.signal_id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "order_type": self.order_type,
            "amount": str(self.amount),
            "price": str(self.price),
            "filled_amount": str(self.filled_amount),
            "average_price": str(self.average_price),
            "state": self.state.value,
            "creation_timestamp": self.creation_timestamp,
            "last_update_timestamp": self.last_update_timestamp,
            "trade_fees": str(self.trade_fees),
            "fill_percentage": self.fill_percentage,
            "is_active": self.is_active,
            "error_message": self.error_message,
        }


class AlgVexOrderTracker:
    """
    订单追踪器 - 集成 Hummingbot InFlightOrder

    功能:
    1. 订单状态查询
    2. 订单超时检测
    3. 订单历史记录
    4. 信号-订单映射
    5. 断线恢复
    """

    def __init__(
        self,
        order_timeout: float = 60.0,
        max_history_size: int = 10000,
        on_order_complete: Optional[Callable] = None,
        on_order_timeout: Optional[Callable] = None,
    ):
        """
        Args:
            order_timeout: 订单超时时间（秒）
            max_history_size: 历史订单最大保存数量
            on_order_complete: 订单完成回调
            on_order_timeout: 订单超时回调
        """
        self._order_timeout = order_timeout
        self._max_history_size = max_history_size
        self._on_order_complete = on_order_complete
        self._on_order_timeout = on_order_timeout

        # 活跃订单: client_order_id -> TrackedOrder
        self._active_orders: Dict[str, TrackedOrder] = {}

        # 已完成订单历史
        self._completed_orders: List[TrackedOrder] = []

        # 信号到订单映射: signal_id -> client_order_id
        self._signal_to_order: Dict[str, str] = {}

        # 订单到信号映射: client_order_id -> signal_id
        self._order_to_signal: Dict[str, str] = {}

    def track_order(
        self,
        client_order_id: str,
        trading_pair: str,
        side: str,
        amount: Decimal,
        price: Decimal = Decimal("0"),
        order_type: str = "MARKET",
        signal_id: Optional[str] = None,
    ) -> TrackedOrder:
        """
        开始追踪新订单

        Args:
            client_order_id: 客户端订单ID
            trading_pair: 交易对
            side: 买卖方向
            amount: 数量
            price: 价格
            order_type: 订单类型
            signal_id: 关联的信号ID

        Returns:
            TrackedOrder: 追踪的订单对象
        """
        order = TrackedOrder(
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            signal_id=signal_id,
            state=OrderState.PENDING_CREATE,
        )

        self._active_orders[client_order_id] = order

        # 记录信号-订单映射
        if signal_id:
            self._signal_to_order[signal_id] = client_order_id
            self._order_to_signal[client_order_id] = signal_id

        logger.info(
            f"Started tracking order: {client_order_id}, "
            f"pair={trading_pair}, side={side}, amount={amount}"
        )

        return order

    def update_order(
        self,
        client_order_id: str,
        exchange_order_id: Optional[str] = None,
        state: Optional[OrderState] = None,
        filled_amount: Optional[Decimal] = None,
        average_price: Optional[Decimal] = None,
        trade_fees: Optional[Decimal] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TrackedOrder]:
        """
        更新订单状态

        Args:
            client_order_id: 客户端订单ID
            exchange_order_id: 交易所订单ID
            state: 新状态
            filled_amount: 已成交数量
            average_price: 平均成交价
            trade_fees: 手续费
            error_message: 错误信息

        Returns:
            更新后的订单，如果不存在返回None
        """
        order = self._active_orders.get(client_order_id)
        if not order:
            logger.warning(f"Order not found for update: {client_order_id}")
            return None

        # 更新字段
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
        if state:
            order.state = state
        if filled_amount is not None:
            order.filled_amount = filled_amount
        if average_price is not None:
            order.average_price = average_price
        if trade_fees is not None:
            order.trade_fees = trade_fees
        if error_message:
            order.error_message = error_message

        order.last_update_timestamp = time.time()

        # 检查是否完成
        if order.is_done:
            self._move_to_completed(client_order_id)

        logger.debug(
            f"Updated order {client_order_id}: state={order.state.value}, "
            f"filled={order.filled_amount}/{order.amount}"
        )

        return order

    def _move_to_completed(self, client_order_id: str):
        """将订单移动到已完成列表"""
        order = self._active_orders.pop(client_order_id, None)
        if order:
            self._completed_orders.append(order)

            # 限制历史大小
            if len(self._completed_orders) > self._max_history_size:
                self._completed_orders = self._completed_orders[-self._max_history_size:]

            # 触发回调
            if self._on_order_complete:
                try:
                    self._on_order_complete(order)
                except Exception as e:
                    logger.error(f"Error in order complete callback: {e}")

            logger.info(
                f"Order completed: {client_order_id}, state={order.state.value}, "
                f"filled={order.filled_amount}"
            )

    def get_order(self, client_order_id: str) -> Optional[TrackedOrder]:
        """获取订单（活跃或历史）"""
        # 先查活跃订单
        order = self._active_orders.get(client_order_id)
        if order:
            return order

        # 再查历史订单
        for completed in reversed(self._completed_orders):
            if completed.client_order_id == client_order_id:
                return completed

        return None

    def get_order_status(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态字典"""
        order = self.get_order(client_order_id)
        if order:
            return order.to_dict()
        return None

    def get_order_by_signal(self, signal_id: str) -> Optional[TrackedOrder]:
        """通过信号ID获取订单"""
        client_order_id = self._signal_to_order.get(signal_id)
        if client_order_id:
            return self.get_order(client_order_id)
        return None

    def get_signal_by_order(self, client_order_id: str) -> Optional[str]:
        """通过订单ID获取信号ID"""
        return self._order_to_signal.get(client_order_id)

    def get_active_orders(self) -> List[TrackedOrder]:
        """获取所有活跃订单"""
        return list(self._active_orders.values())

    def get_active_orders_for_pair(self, trading_pair: str) -> List[TrackedOrder]:
        """获取指定交易对的活跃订单"""
        return [
            order for order in self._active_orders.values()
            if order.trading_pair == trading_pair
        ]

    async def check_timeout_orders(self) -> List[str]:
        """
        检查超时订单

        Returns:
            超时的订单ID列表
        """
        timeout_orders = []
        current_time = time.time()

        for order_id, order in list(self._active_orders.items()):
            if order.state == OrderState.PENDING_CREATE:
                age = current_time - order.creation_timestamp
                if age > self._order_timeout:
                    timeout_orders.append(order_id)

                    # 标记为超时
                    order.state = OrderState.EXPIRED
                    order.error_message = f"Order timeout after {age:.1f}s"
                    self._move_to_completed(order_id)

                    # 触发超时回调
                    if self._on_order_timeout:
                        try:
                            self._on_order_timeout(order)
                        except Exception as e:
                            logger.error(f"Error in order timeout callback: {e}")

                    logger.warning(f"Order timeout: {order_id}, age={age:.1f}s")

        return timeout_orders

    def cancel_order(self, client_order_id: str, reason: str = "user_cancelled") -> bool:
        """
        取消订单

        Args:
            client_order_id: 订单ID
            reason: 取消原因

        Returns:
            是否成功取消
        """
        order = self._active_orders.get(client_order_id)
        if not order:
            logger.warning(f"Cannot cancel - order not found: {client_order_id}")
            return False

        if order.is_done:
            logger.warning(f"Cannot cancel - order already done: {client_order_id}")
            return False

        order.state = OrderState.CANCELLED
        order.error_message = reason
        self._move_to_completed(client_order_id)

        logger.info(f"Order cancelled: {client_order_id}, reason={reason}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """获取订单统计"""
        active = len(self._active_orders)
        completed = len(self._completed_orders)

        # 统计已完成订单的状态分布
        state_counts = {}
        for order in self._completed_orders:
            state = order.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        # 计算平均成交时间
        fill_times = []
        for order in self._completed_orders:
            if order.state == OrderState.FILLED:
                fill_time = order.last_update_timestamp - order.creation_timestamp
                fill_times.append(fill_time)

        avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0

        return {
            "active_orders": active,
            "completed_orders": completed,
            "total_orders": active + completed,
            "state_distribution": state_counts,
            "avg_fill_time_seconds": avg_fill_time,
            "pending_signals": len(self._signal_to_order),
        }

    def clear_history(self):
        """清空历史订单"""
        self._completed_orders.clear()
        logger.info("Order history cleared")

    def export_history(self) -> List[Dict[str, Any]]:
        """导出订单历史"""
        return [order.to_dict() for order in self._completed_orders]

    async def recover_from_exchange(self, exchange_orders: Dict[str, Any]):
        """
        从交易所恢复订单状态

        用于断线重连后同步状态

        Args:
            exchange_orders: 交易所返回的订单数据
        """
        recovered_count = 0

        for order_id, exchange_data in exchange_orders.items():
            if order_id in self._active_orders:
                order = self._active_orders[order_id]

                # 更新状态
                exchange_state = exchange_data.get("status", "").upper()
                if exchange_state == "FILLED":
                    order.state = OrderState.FILLED
                elif exchange_state == "CANCELED":
                    order.state = OrderState.CANCELLED
                elif exchange_state == "PARTIALLY_FILLED":
                    order.state = OrderState.PARTIALLY_FILLED
                elif exchange_state == "NEW":
                    order.state = OrderState.OPEN

                # 更新成交信息
                if "executedQty" in exchange_data:
                    order.filled_amount = Decimal(str(exchange_data["executedQty"]))
                if "avgPrice" in exchange_data:
                    order.average_price = Decimal(str(exchange_data["avgPrice"]))

                order.last_update_timestamp = time.time()

                # 如果已完成，移动到历史
                if order.is_done:
                    self._move_to_completed(order_id)

                recovered_count += 1

        logger.info(f"Recovered {recovered_count} orders from exchange")
        return recovered_count
