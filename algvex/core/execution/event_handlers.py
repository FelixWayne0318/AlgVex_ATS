"""
事件处理器

处理 Hummingbot 订单事件，写入 trace 追踪
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型"""
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FAILED = "order_failed"
    ORDER_EXPIRED = "order_expired"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    FUNDING_PAYMENT = "funding_payment"
    LIQUIDATION = "liquidation"


@dataclass
class OrderEvent:
    """订单事件基类"""
    event_type: EventType
    client_order_id: str
    trading_pair: str
    timestamp: float
    exchange_order_id: Optional[str] = None
    side: Optional[str] = None  # "BUY" or "SELL"
    order_type: Optional[str] = None  # "MARKET" or "LIMIT"
    amount: Optional[Decimal] = None
    price: Optional[Decimal] = None
    filled_amount: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    fee: Optional[Decimal] = None
    fee_asset: Optional[str] = None
    error_message: Optional[str] = None

    def to_trace(self) -> Dict[str, Any]:
        """转换为 trace 格式"""
        trace = {
            "type": self.event_type.value,
            "client_order_id": self.client_order_id,
            "trading_pair": self.trading_pair,
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }

        if self.exchange_order_id:
            trace["exchange_order_id"] = self.exchange_order_id
        if self.side:
            trace["side"] = self.side
        if self.order_type:
            trace["order_type"] = self.order_type
        if self.amount is not None:
            trace["amount"] = str(self.amount)
        if self.price is not None:
            trace["price"] = str(self.price)
        if self.filled_amount is not None:
            trace["filled_amount"] = str(self.filled_amount)
        if self.average_price is not None:
            trace["average_price"] = str(self.average_price)
        if self.fee is not None:
            trace["fee"] = str(self.fee)
        if self.fee_asset:
            trace["fee_asset"] = self.fee_asset
        if self.error_message:
            trace["error_message"] = self.error_message

        return trace


@dataclass
class FundingPaymentEvent:
    """资金费率支付事件"""
    trading_pair: str
    amount: Decimal
    funding_rate: Decimal
    timestamp: float
    position_size: Optional[Decimal] = None

    def to_trace(self) -> Dict[str, Any]:
        """转换为 trace 格式"""
        trace = {
            "type": EventType.FUNDING_PAYMENT.value,
            "trading_pair": self.trading_pair,
            "amount": str(self.amount),
            "funding_rate": str(self.funding_rate),
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }

        if self.position_size is not None:
            trace["position_size"] = str(self.position_size)

        return trace


@dataclass
class LiquidationEvent:
    """清算事件"""
    trading_pair: str
    side: str
    amount: Decimal
    price: Decimal
    timestamp: float
    loss: Optional[Decimal] = None

    def to_trace(self) -> Dict[str, Any]:
        """转换为 trace 格式"""
        trace = {
            "type": EventType.LIQUIDATION.value,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "amount": str(self.amount),
            "price": str(self.price),
            "timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }

        if self.loss is not None:
            trace["loss"] = str(self.loss)

        return trace


class EventCallback(ABC):
    """事件回调接口"""

    @abstractmethod
    async def on_event(self, event: Any) -> None:
        """处理事件"""
        pass


class AlgVexEventHandler:
    """
    AlgVex 事件处理器

    事件类型映射:
    - BuyOrderCreatedEvent → order_created trace
    - SellOrderCreatedEvent → order_created trace
    - OrderFilledEvent → order_filled trace, 更新仓位
    - OrderCancelledEvent → order_cancelled trace
    - MarketOrderFailureEvent → order_failed trace, 告警
    - FundingPaymentCompletedEvent → funding_payment trace

    设计原则:
    1. 所有事件写入 trace（审计可追溯）
    2. 关键事件触发回调（状态更新、告警）
    3. 支持自定义事件处理器扩展
    """

    def __init__(
        self,
        trace_writer: Optional['TraceWriter'] = None,
        order_tracker: Optional['AlgVexOrderTracker'] = None,
        position_manager: Optional['PositionManager'] = None,
        alert_handler: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Args:
            trace_writer: Trace 写入器
            order_tracker: 订单追踪器
            position_manager: 仓位管理器
            alert_handler: 告警处理函数
        """
        self.trace_writer = trace_writer
        self.order_tracker = order_tracker
        self.position_manager = position_manager
        self._alert_handler = alert_handler

        # 自定义事件处理器
        self._custom_handlers: Dict[EventType, List[EventCallback]] = {}

        # 事件统计
        self._event_counts: Dict[str, int] = {}

    def register_handler(self, event_type: EventType, handler: EventCallback):
        """注册自定义事件处理器"""
        if event_type not in self._custom_handlers:
            self._custom_handlers[event_type] = []
        self._custom_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    async def handle_order_created(
        self,
        client_order_id: str,
        trading_pair: str,
        side: str,
        amount: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "MARKET",
        exchange_order_id: Optional[str] = None,
    ):
        """
        处理订单创建事件

        对应 Hummingbot BuyOrderCreatedEvent / SellOrderCreatedEvent
        """
        event = OrderEvent(
            event_type=EventType.ORDER_CREATED,
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            timestamp=time.time(),
            exchange_order_id=exchange_order_id,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
        )

        await self._process_event(event)

        logger.info(
            f"Order created: {client_order_id}, pair={trading_pair}, "
            f"side={side}, amount={amount}"
        )

    async def handle_order_filled(
        self,
        client_order_id: str,
        trading_pair: str,
        side: str,
        filled_amount: Decimal,
        average_price: Decimal,
        fee: Decimal = Decimal("0"),
        fee_asset: str = "USDT",
        exchange_order_id: Optional[str] = None,
        is_partial: bool = False,
    ):
        """
        处理订单成交事件

        对应 Hummingbot OrderFilledEvent
        """
        event = OrderEvent(
            event_type=EventType.ORDER_FILLED,
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            timestamp=time.time(),
            exchange_order_id=exchange_order_id,
            side=side,
            filled_amount=filled_amount,
            average_price=average_price,
            fee=fee,
            fee_asset=fee_asset,
        )

        await self._process_event(event)

        # 更新订单追踪器
        if self.order_tracker:
            from .order_tracker import OrderState
            self.order_tracker.update_order(
                client_order_id=client_order_id,
                exchange_order_id=exchange_order_id,
                state=OrderState.PARTIALLY_FILLED if is_partial else OrderState.FILLED,
                filled_amount=filled_amount,
                average_price=average_price,
                trade_fees=fee,
            )

        # 更新仓位
        if self.position_manager:
            await self._update_position_from_fill(
                trading_pair, side, filled_amount, average_price
            )

        logger.info(
            f"Order filled: {client_order_id}, pair={trading_pair}, "
            f"filled={filled_amount}@{average_price}, fee={fee} {fee_asset}"
        )

    async def _update_position_from_fill(
        self,
        trading_pair: str,
        side: str,
        filled_amount: Decimal,
        average_price: Decimal,
    ):
        """根据成交更新仓位"""
        current_pos = self.position_manager.get_position(trading_pair)

        if side == "BUY":
            # 买入 = 开多或平空
            if current_pos and current_pos.side == "SHORT":
                # 平空
                new_amount = current_pos.amount - filled_amount
                if new_amount <= Decimal("0"):
                    self.position_manager.close_position(trading_pair)
                else:
                    self.position_manager.update_position(
                        symbol=trading_pair,
                        side="SHORT",
                        amount=new_amount,
                        entry_price=current_pos.entry_price,
                    )
            else:
                # 开多
                if current_pos:
                    # 加仓
                    new_amount = current_pos.amount + filled_amount
                    # 计算新的平均入场价
                    new_entry = (
                        current_pos.entry_price * current_pos.amount +
                        average_price * filled_amount
                    ) / new_amount
                else:
                    new_amount = filled_amount
                    new_entry = average_price

                self.position_manager.update_position(
                    symbol=trading_pair,
                    side="LONG",
                    amount=new_amount,
                    entry_price=new_entry,
                )

        elif side == "SELL":
            # 卖出 = 开空或平多
            if current_pos and current_pos.side == "LONG":
                # 平多
                new_amount = current_pos.amount - filled_amount
                if new_amount <= Decimal("0"):
                    self.position_manager.close_position(trading_pair)
                else:
                    self.position_manager.update_position(
                        symbol=trading_pair,
                        side="LONG",
                        amount=new_amount,
                        entry_price=current_pos.entry_price,
                    )
            else:
                # 开空
                if current_pos:
                    # 加仓
                    new_amount = current_pos.amount + filled_amount
                    new_entry = (
                        current_pos.entry_price * current_pos.amount +
                        average_price * filled_amount
                    ) / new_amount
                else:
                    new_amount = filled_amount
                    new_entry = average_price

                self.position_manager.update_position(
                    symbol=trading_pair,
                    side="SHORT",
                    amount=new_amount,
                    entry_price=new_entry,
                )

    async def handle_order_cancelled(
        self,
        client_order_id: str,
        trading_pair: str,
        exchange_order_id: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """
        处理订单取消事件

        对应 Hummingbot OrderCancelledEvent
        """
        event = OrderEvent(
            event_type=EventType.ORDER_CANCELLED,
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            timestamp=time.time(),
            exchange_order_id=exchange_order_id,
            error_message=reason,
        )

        await self._process_event(event)

        # 更新订单追踪器
        if self.order_tracker:
            self.order_tracker.cancel_order(client_order_id, reason or "cancelled")

        logger.info(f"Order cancelled: {client_order_id}, reason={reason}")

    async def handle_order_failed(
        self,
        client_order_id: str,
        trading_pair: str,
        error_message: str,
        exchange_order_id: Optional[str] = None,
    ):
        """
        处理订单失败事件

        对应 Hummingbot MarketOrderFailureEvent
        需要触发告警
        """
        event = OrderEvent(
            event_type=EventType.ORDER_FAILED,
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            timestamp=time.time(),
            exchange_order_id=exchange_order_id,
            error_message=error_message,
        )

        await self._process_event(event)

        # 更新订单追踪器
        if self.order_tracker:
            from .order_tracker import OrderState
            self.order_tracker.update_order(
                client_order_id=client_order_id,
                state=OrderState.FAILED,
                error_message=error_message,
            )

        # 触发告警
        if self._alert_handler:
            self._alert_handler("order_failed", {
                "client_order_id": client_order_id,
                "trading_pair": trading_pair,
                "error": error_message,
            })

        logger.error(f"Order failed: {client_order_id}, error={error_message}")

    async def handle_funding_payment(
        self,
        trading_pair: str,
        amount: Decimal,
        funding_rate: Decimal,
        position_size: Optional[Decimal] = None,
    ):
        """
        处理资金费率支付事件

        对应 Hummingbot FundingPaymentCompletedEvent
        """
        event = FundingPaymentEvent(
            trading_pair=trading_pair,
            amount=amount,
            funding_rate=funding_rate,
            timestamp=time.time(),
            position_size=position_size,
        )

        # 写入 trace
        if self.trace_writer:
            self.trace_writer.write(event.to_trace())

        # 更新统计
        self._increment_count(EventType.FUNDING_PAYMENT.value)

        # 调用自定义处理器
        await self._call_custom_handlers(EventType.FUNDING_PAYMENT, event)

        logger.info(
            f"Funding payment: {trading_pair}, amount={amount}, rate={funding_rate}"
        )

    async def handle_liquidation(
        self,
        trading_pair: str,
        side: str,
        amount: Decimal,
        price: Decimal,
        loss: Optional[Decimal] = None,
    ):
        """
        处理清算事件

        紧急告警！
        """
        event = LiquidationEvent(
            trading_pair=trading_pair,
            side=side,
            amount=amount,
            price=price,
            timestamp=time.time(),
            loss=loss,
        )

        # 写入 trace
        if self.trace_writer:
            self.trace_writer.write(event.to_trace())

        # 更新统计
        self._increment_count(EventType.LIQUIDATION.value)

        # 清除仓位
        if self.position_manager:
            self.position_manager.close_position(trading_pair)

        # 紧急告警
        if self._alert_handler:
            self._alert_handler("liquidation", {
                "trading_pair": trading_pair,
                "side": side,
                "amount": str(amount),
                "price": str(price),
                "loss": str(loss) if loss else None,
                "severity": "critical",
            })

        logger.critical(
            f"LIQUIDATION: {trading_pair} {side} {amount}@{price}, loss={loss}"
        )

    async def _process_event(self, event: OrderEvent):
        """处理订单事件"""
        # 写入 trace
        if self.trace_writer:
            self.trace_writer.write(event.to_trace())

        # 更新统计
        self._increment_count(event.event_type.value)

        # 调用自定义处理器
        await self._call_custom_handlers(event.event_type, event)

    async def _call_custom_handlers(self, event_type: EventType, event: Any):
        """调用自定义处理器"""
        handlers = self._custom_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler.on_event(event)
            except Exception as e:
                logger.error(f"Error in custom handler for {event_type.value}: {e}")

    def _increment_count(self, event_type: str):
        """增加事件计数"""
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        return {
            "event_counts": dict(self._event_counts),
            "total_events": sum(self._event_counts.values()),
            "registered_handlers": {
                et.value: len(handlers)
                for et, handlers in self._custom_handlers.items()
            },
        }

    def reset_statistics(self):
        """重置统计"""
        self._event_counts.clear()


class TraceWriter:
    """
    Trace 写入器

    将事件 trace 写入文件/数据库
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        buffer_size: int = 100,
    ):
        """
        Args:
            output_file: 输出文件路径
            buffer_size: 缓冲区大小
        """
        self.output_file = output_file
        self.buffer_size = buffer_size
        self._buffer: List[Dict] = []

    def write(self, trace: Dict):
        """写入 trace"""
        import json

        self._buffer.append(trace)

        # 达到缓冲区大小时刷新
        if len(self._buffer) >= self.buffer_size:
            self.flush()

        # 如果没有输出文件，记录到日志
        if not self.output_file:
            logger.debug(f"Trace: {json.dumps(trace)}")

    def flush(self):
        """刷新缓冲区"""
        if not self._buffer or not self.output_file:
            self._buffer.clear()
            return

        import json

        try:
            with open(self.output_file, 'a') as f:
                for trace in self._buffer:
                    f.write(json.dumps(trace) + "\n")
            self._buffer.clear()
            logger.debug(f"Flushed {len(self._buffer)} traces to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to flush traces: {e}")

    def close(self):
        """关闭并刷新剩余数据"""
        self.flush()
