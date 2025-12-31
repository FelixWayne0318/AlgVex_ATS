# Hummingbot 执行层集成设计

> **版本**: 1.0.0
> **日期**: 2025-12-22
> **目的**: 补充 AlgVex 与 Hummingbot 的执行层集成细节

---

## 1. 集成架构概述

### 1.1 设计原则

1. **最小侵入**: 不修改 Hummingbot 核心代码，通过适配层对接
2. **状态同步**: AlgVex 和 Hummingbot 保持订单/仓位状态一致
3. **幂等性**: 信号到订单的转换必须幂等
4. **可观测**: 完整的订单生命周期追踪

### 1.2 模块架构

```
algvex/execution/
├── __init__.py
├── hummingbot_bridge.py       # 核心桥接层
├── order_converter.py         # AlgVex Signal -> Hummingbot Order
├── state_synchronizer.py      # 状态同步器
├── event_handlers.py          # 事件处理器
└── controllers/
    └── algvex_controller.py   # Hummingbot V2 Controller 实现
```

---

## 2. 核心组件设计

### 2.1 HummingbotBridge (桥接层)

```python
# algvex/execution/hummingbot_bridge.py
"""
Hummingbot 桥接层

职责:
1. 连接 AlgVex SignalGenerator 和 Hummingbot Connector
2. 管理订单生命周期
3. 状态同步
"""

from decimal import Decimal
from typing import Dict, Optional, List
from datetime import datetime
import asyncio

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate

from algvex.production.signal.signal_generator import Signal
from algvex.shared.trace_serializer import DeterministicTraceSerializer


class HummingbotBridge:
    """AlgVex 与 Hummingbot 的桥接层"""

    def __init__(
        self,
        connector: ConnectorBase,
        trace_writer: Optional['TraceWriter'] = None,
    ):
        self.connector = connector
        self.trace_writer = trace_writer
        self.serializer = DeterministicTraceSerializer()

        # 订单跟踪
        self._signal_to_order: Dict[str, str] = {}  # signal_id -> client_order_id
        self._order_to_signal: Dict[str, str] = {}  # client_order_id -> signal_id
        self._pending_signals: Dict[str, Signal] = {}

    async def execute_signal(self, signal: Signal) -> Dict:
        """
        执行 AlgVex 信号

        Args:
            signal: AlgVex 信号对象

        Returns:
            执行结果 dict
        """
        # 1. 生成幂等的 client_order_id
        client_order_id = self._generate_idempotent_order_id(signal)

        # 2. 检查是否已处理过 (幂等)
        if client_order_id in self.connector.in_flight_orders:
            existing_order = self.connector.in_flight_orders[client_order_id]
            return {
                "status": "duplicate",
                "client_order_id": client_order_id,
                "order_state": existing_order.current_state.name,
            }

        # 3. 转换为 Hummingbot OrderCandidate
        order_candidate = self._signal_to_order_candidate(signal, client_order_id)

        # 4. 预算检查
        adjusted_candidate = self.connector.budget_checker.adjust_candidate(
            order_candidate, all_or_none=False
        )
        if adjusted_candidate.amount == Decimal("0"):
            return {
                "status": "rejected",
                "reason": "insufficient_funds",
                "signal_id": signal.signal_id,
            }

        # 5. 下单
        try:
            if signal.direction > 0:
                order_id = self.connector.buy(
                    trading_pair=signal.symbol,
                    amount=adjusted_candidate.amount,
                    order_type=OrderType.MARKET,
                    price=Decimal("0"),  # Market order
                )
            else:
                order_id = self.connector.sell(
                    trading_pair=signal.symbol,
                    amount=adjusted_candidate.amount,
                    order_type=OrderType.MARKET,
                    price=Decimal("0"),
                )

            # 6. 记录映射
            self._signal_to_order[signal.signal_id] = order_id
            self._order_to_signal[order_id] = signal.signal_id
            self._pending_signals[order_id] = signal

            # 7. 写入 trace
            if self.trace_writer:
                self._write_order_trace(signal, order_id, "submitted")

            return {
                "status": "submitted",
                "client_order_id": order_id,
                "signal_id": signal.signal_id,
                "amount": str(adjusted_candidate.amount),
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "signal_id": signal.signal_id,
            }

    def _generate_idempotent_order_id(self, signal: Signal) -> str:
        """
        生成幂等的订单 ID

        基于信号内容 hash，相同信号生成相同 ID
        """
        content = {
            "symbol": signal.symbol,
            "direction": signal.direction,
            "bar_close_time": signal.bar_close_time.isoformat(),
            "final_signal": signal.final_signal,
        }
        hash_str = self.serializer.compute_hash(content)
        return f"algvex_{hash_str}"

    def _signal_to_order_candidate(
        self, signal: Signal, client_order_id: str
    ) -> OrderCandidate:
        """将 AlgVex Signal 转换为 Hummingbot OrderCandidate"""
        return OrderCandidate(
            trading_pair=signal.symbol,
            is_maker=False,  # Market order = taker
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY if signal.direction > 0 else TradeType.SELL,
            amount=Decimal(str(signal.quantity)),
            price=Decimal("0"),  # Market order
        )

    def _write_order_trace(self, signal: Signal, order_id: str, status: str):
        """写入订单追踪"""
        trace = {
            "type": "order_event",
            "signal_id": signal.signal_id,
            "client_order_id": order_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.trace_writer.write(trace)

    # ==================== 事件处理 ====================

    async def on_order_filled(self, event: 'OrderFilledEvent'):
        """处理订单成交事件"""
        order_id = event.order_id
        signal_id = self._order_to_signal.get(order_id)

        if signal_id:
            signal = self._pending_signals.get(order_id)

            # 写入执行 trace
            if self.trace_writer:
                trace = {
                    "type": "order_filled",
                    "signal_id": signal_id,
                    "client_order_id": order_id,
                    "exchange_order_id": event.exchange_order_id,
                    "price": str(event.price),
                    "amount": str(event.amount),
                    "trade_fee": str(event.trade_fee.flat_fees),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                self.trace_writer.write(trace)

            # 清理
            del self._pending_signals[order_id]

    async def on_order_cancelled(self, event: 'OrderCancelledEvent'):
        """处理订单取消事件"""
        order_id = event.order_id
        signal_id = self._order_to_signal.get(order_id)

        if signal_id and self.trace_writer:
            trace = {
                "type": "order_cancelled",
                "signal_id": signal_id,
                "client_order_id": order_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self.trace_writer.write(trace)

    async def on_order_failure(self, event: 'MarketOrderFailureEvent'):
        """处理订单失败事件"""
        order_id = event.order_id
        signal_id = self._order_to_signal.get(order_id)

        if signal_id and self.trace_writer:
            trace = {
                "type": "order_failed",
                "signal_id": signal_id,
                "client_order_id": order_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self.trace_writer.write(trace)

    # ==================== 状态同步 ====================

    async def sync_positions(self) -> Dict:
        """与交易所同步仓位"""
        exchange_positions = {}

        # 从 Hummingbot Connector 获取仓位
        for trading_pair in self.connector.trading_pairs:
            position = self.connector.get_position(trading_pair)
            if position:
                exchange_positions[trading_pair] = {
                    "amount": str(position.amount),
                    "entry_price": str(position.entry_price),
                    "leverage": position.leverage,
                    "unrealized_pnl": str(position.unrealized_pnl),
                }

        return exchange_positions

    async def reconcile(self) -> Dict:
        """
        对账: 比较本地状态和交易所状态

        Returns:
            对账结果
        """
        exchange_positions = await self.sync_positions()
        local_positions = self._get_local_positions()

        discrepancies = []
        for symbol in set(exchange_positions.keys()) | set(local_positions.keys()):
            exchange_amt = Decimal(exchange_positions.get(symbol, {}).get("amount", "0"))
            local_amt = Decimal(local_positions.get(symbol, {}).get("amount", "0"))

            if abs(exchange_amt - local_amt) > Decimal("0.00001"):
                discrepancies.append({
                    "symbol": symbol,
                    "exchange_amount": str(exchange_amt),
                    "local_amount": str(local_amt),
                    "diff": str(exchange_amt - local_amt),
                })

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "exchange_positions": len(exchange_positions),
            "local_positions": len(local_positions),
            "discrepancies": discrepancies,
            "aligned": len(discrepancies) == 0,
        }

    def _get_local_positions(self) -> Dict:
        """获取本地仓位状态"""
        # 从 AlgVex PositionManager 获取
        # TODO: 实现
        return {}
```

### 2.2 AlgVexController (V2 Controller)

```python
# algvex/execution/controllers/algvex_controller.py
"""
Hummingbot V2 Controller 实现

集成 AlgVex 信号生成到 Hummingbot Strategy V2 框架
"""

from decimal import Decimal
from typing import List, Optional, Set
import asyncio

from hummingbot.smart_components.controllers.controller_base import ControllerBase
from hummingbot.smart_components.executors.position_executor.position_executor import PositionExecutor
from hummingbot.smart_components.models.executors import PositionConfig

from algvex.production.signal.signal_generator import SignalGenerator


class AlgVexControllerConfig:
    """AlgVex Controller 配置"""
    trading_pairs: Set[str]
    signal_threshold: float = 0.5  # 信号阈值
    max_position_per_pair: Decimal = Decimal("0.1")  # 单品种最大仓位 (BTC)
    leverage: int = 1


class AlgVexController(ControllerBase):
    """
    AlgVex 信号控制器

    将 AlgVex SignalGenerator 集成到 Hummingbot V2 架构
    """

    def __init__(
        self,
        config: AlgVexControllerConfig,
        signal_generator: SignalGenerator,
    ):
        super().__init__(config)
        self.config = config
        self.signal_generator = signal_generator
        self._latest_signals = {}

    async def update_processed_data(self):
        """
        更新处理后的数据

        每个 tick 调用，获取最新的 AlgVex 信号
        """
        for trading_pair in self.config.trading_pairs:
            try:
                signal = await self.signal_generator.get_signal(trading_pair)
                self._latest_signals[trading_pair] = signal
            except Exception as e:
                self.logger().warning(f"Failed to get signal for {trading_pair}: {e}")

    def determine_executor_actions(self) -> List:
        """
        确定执行器动作

        基于 AlgVex 信号决定是否开仓/平仓
        """
        actions = []

        for trading_pair, signal in self._latest_signals.items():
            if signal is None:
                continue

            # 检查信号强度
            if abs(signal.final_signal) < self.config.signal_threshold:
                continue

            # 确定方向
            is_long = signal.final_signal > 0

            # 创建仓位配置
            position_config = PositionConfig(
                trading_pair=trading_pair,
                side="LONG" if is_long else "SHORT",
                amount=self._calculate_position_size(signal),
                leverage=self.config.leverage,
                # 止损止盈
                stop_loss=Decimal("0.02"),  # 2%
                take_profit=Decimal("0.04"),  # 4%
            )

            actions.append({
                "type": "create_position",
                "config": position_config,
                "signal": signal,
            })

        return actions

    def _calculate_position_size(self, signal) -> Decimal:
        """计算仓位大小"""
        # 基于信号强度调整仓位
        base_size = self.config.max_position_per_pair
        signal_weight = Decimal(str(abs(signal.final_signal)))
        return base_size * signal_weight

    async def on_executor_complete(self, executor_id: str, result: dict):
        """
        执行器完成回调

        记录执行结果用于归因分析
        """
        self.logger().info(f"Executor {executor_id} completed: {result}")
        # TODO: 写入 trace


# ==================== 策略集成示例 ====================

class AlgVexStrategy:
    """
    AlgVex 策略 (Hummingbot V2 风格)

    集成示例:
    1. 配置 Controller 和 Executors
    2. 运行主循环
    """

    def __init__(self, connectors: dict, signal_generator: SignalGenerator):
        self.connectors = connectors
        self.signal_generator = signal_generator

        # 初始化 Controller
        self.controller = AlgVexController(
            config=AlgVexControllerConfig(
                trading_pairs={"BTCUSDT", "ETHUSDT"},
                signal_threshold=0.5,
                max_position_per_pair=Decimal("0.1"),
            ),
            signal_generator=signal_generator,
        )

        # 初始化 Executors
        self.executors = {}

    async def tick(self, current_time: float):
        """主循环 tick"""
        # 1. 更新信号
        await self.controller.update_processed_data()

        # 2. 获取动作
        actions = self.controller.determine_executor_actions()

        # 3. 执行动作
        for action in actions:
            if action["type"] == "create_position":
                await self._create_position_executor(action["config"])

        # 4. 更新现有 Executors
        for executor in list(self.executors.values()):
            if executor.is_active:
                await executor.tick()
            else:
                await self.controller.on_executor_complete(
                    executor.executor_id,
                    executor.get_result()
                )
                del self.executors[executor.executor_id]

    async def _create_position_executor(self, config: PositionConfig):
        """创建仓位执行器"""
        connector = self.connectors[config.trading_pair.split("-")[0]]
        executor = PositionExecutor(
            connector=connector,
            config=config,
        )
        self.executors[executor.executor_id] = executor
        await executor.start()
```

### 2.3 状态同步器

```python
# algvex/execution/state_synchronizer.py
"""
状态同步器

确保 AlgVex 和 Hummingbot 状态一致
"""

from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class StateSynchronizer:
    """
    状态同步器

    职责:
    1. 定期同步仓位状态
    2. 检测并处理状态不一致
    3. 处理断线重连
    """

    def __init__(
        self,
        bridge: 'HummingbotBridge',
        position_manager: 'PositionManager',
        sync_interval: float = 60.0,  # 60秒同步一次
    ):
        self.bridge = bridge
        self.position_manager = position_manager
        self.sync_interval = sync_interval
        self._running = False
        self._last_sync = None

    async def start(self):
        """启动同步循环"""
        self._running = True
        while self._running:
            try:
                await self.sync()
            except Exception as e:
                logger.error(f"Sync failed: {e}")

            await asyncio.sleep(self.sync_interval)

    async def stop(self):
        """停止同步"""
        self._running = False

    async def sync(self):
        """执行一次同步"""
        logger.info("Starting position sync...")

        # 1. 获取交易所仓位
        exchange_positions = await self.bridge.sync_positions()

        # 2. 获取本地仓位
        local_positions = self.position_manager.get_all_positions()

        # 3. 对比
        for symbol in set(exchange_positions.keys()) | set(local_positions.keys()):
            await self._sync_symbol(symbol, exchange_positions, local_positions)

        self._last_sync = datetime.utcnow()
        logger.info(f"Position sync completed at {self._last_sync}")

    async def _sync_symbol(
        self,
        symbol: str,
        exchange_positions: Dict,
        local_positions: Dict,
    ):
        """同步单个品种"""
        exchange_pos = exchange_positions.get(symbol)
        local_pos = local_positions.get(symbol)

        if exchange_pos is None and local_pos is None:
            return

        # 交易所有仓位，本地没有 -> 更新本地
        if exchange_pos and not local_pos:
            logger.warning(f"Missing local position for {symbol}, syncing from exchange")
            self.position_manager.update_position(symbol, {
                "amount": Decimal(exchange_pos["amount"]),
                "entry_price": Decimal(exchange_pos["entry_price"]),
                "source": "exchange_sync",
            })
            return

        # 本地有仓位，交易所没有 -> 可能已平仓
        if local_pos and not exchange_pos:
            logger.warning(f"Position {symbol} closed on exchange, updating local")
            self.position_manager.close_position(symbol)
            return

        # 两边都有，检查数量是否一致
        exchange_amt = Decimal(exchange_pos["amount"])
        local_amt = local_pos["amount"]

        if abs(exchange_amt - local_amt) > Decimal("0.00001"):
            logger.error(
                f"Position mismatch for {symbol}: "
                f"exchange={exchange_amt}, local={local_amt}"
            )
            # 以交易所为准
            self.position_manager.update_position(symbol, {
                "amount": exchange_amt,
                "entry_price": Decimal(exchange_pos["entry_price"]),
                "source": "exchange_sync_correction",
            })

    # ==================== 断线处理 ====================

    async def on_disconnect(self):
        """处理断线"""
        logger.warning("Connector disconnected, entering protection mode")
        # 暂停新信号处理
        self.position_manager.enter_protection_mode()

    async def on_reconnect(self):
        """处理重连"""
        logger.info("Connector reconnected, performing full sync")
        # 强制全量同步
        await self.sync()
        # 恢复正常模式
        self.position_manager.exit_protection_mode()
```

---

## 3. 事件处理流程

### 3.1 事件类型映射

| Hummingbot 事件 | AlgVex 处理 |
|-----------------|-------------|
| `BuyOrderCreatedEvent` | 记录 order_created trace |
| `SellOrderCreatedEvent` | 记录 order_created trace |
| `OrderFilledEvent` | 记录 order_filled trace, 更新仓位 |
| `OrderCancelledEvent` | 记录 order_cancelled trace |
| `MarketOrderFailureEvent` | 记录 order_failed trace, 告警 |
| `FundingPaymentCompletedEvent` | 记录 funding_payment trace |

### 3.2 事件处理器

```python
# algvex/execution/event_handlers.py
"""
Hummingbot 事件处理器
"""

from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    SellOrderCreatedEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    MarketOrderFailureEvent,
    FundingPaymentCompletedEvent,
)


class AlgVexEventHandler:
    """AlgVex 事件处理器"""

    def __init__(self, bridge: 'HummingbotBridge', trace_writer: 'TraceWriter'):
        self.bridge = bridge
        self.trace_writer = trace_writer

    def register_events(self, connector):
        """注册事件监听"""
        connector.add_listener(
            BuyOrderCreatedEvent, self.on_buy_order_created
        )
        connector.add_listener(
            SellOrderCreatedEvent, self.on_sell_order_created
        )
        connector.add_listener(
            OrderFilledEvent, self.on_order_filled
        )
        connector.add_listener(
            OrderCancelledEvent, self.on_order_cancelled
        )
        connector.add_listener(
            MarketOrderFailureEvent, self.on_order_failure
        )
        connector.add_listener(
            FundingPaymentCompletedEvent, self.on_funding_payment
        )

    async def on_buy_order_created(self, event: BuyOrderCreatedEvent):
        """买单创建"""
        await self._handle_order_created(event, "BUY")

    async def on_sell_order_created(self, event: SellOrderCreatedEvent):
        """卖单创建"""
        await self._handle_order_created(event, "SELL")

    async def _handle_order_created(self, event, side: str):
        """处理订单创建"""
        # 写入 trace
        trace = {
            "type": "order_created",
            "client_order_id": event.order_id,
            "trading_pair": event.trading_pair,
            "side": side,
            "amount": str(event.amount),
            "price": str(event.price) if event.price else None,
            "order_type": event.type.name,
            "timestamp": event.timestamp,
        }
        self.trace_writer.write(trace)

    async def on_order_filled(self, event: OrderFilledEvent):
        """订单成交"""
        await self.bridge.on_order_filled(event)

    async def on_order_cancelled(self, event: OrderCancelledEvent):
        """订单取消"""
        await self.bridge.on_order_cancelled(event)

    async def on_order_failure(self, event: MarketOrderFailureEvent):
        """订单失败"""
        await self.bridge.on_order_failure(event)

    async def on_funding_payment(self, event: FundingPaymentCompletedEvent):
        """资金费率支付"""
        trace = {
            "type": "funding_payment",
            "trading_pair": event.trading_pair,
            "amount": str(event.amount),
            "funding_rate": str(event.funding_rate),
            "timestamp": event.timestamp,
        }
        self.trace_writer.write(trace)
```

---

## 4. 配置与部署

### 4.1 Hummingbot Connector 配置

```yaml
# config/hummingbot_connector.yaml
connector:
  exchange: binance_perpetual
  api_key: ${BINANCE_API_KEY}
  api_secret: ${BINANCE_API_SECRET}

  # 交易对配置
  trading_pairs:
    - BTCUSDT
    - ETHUSDT

  # 杠杆配置
  leverage:
    default: 1
    max: 10

  # 仓位模式
  position_mode: one_way  # one_way | hedge

# 同步配置
sync:
  position_sync_interval: 60  # 秒
  order_book_depth: 20
  enable_funding_tracking: true

# 风控配置
risk:
  max_position_value_usd: 10000
  max_daily_loss_usd: 500
  enable_auto_reduce: true
```

### 4.2 启动脚本

```python
# scripts/start_live_trading.py
"""
启动实盘交易

集成 AlgVex + Hummingbot
"""

import asyncio
from algvex.execution.hummingbot_bridge import HummingbotBridge
from algvex.execution.state_synchronizer import StateSynchronizer
from algvex.execution.event_handlers import AlgVexEventHandler
from algvex.execution.controllers.algvex_controller import AlgVexStrategy
from algvex.production.signal.signal_generator import SignalGenerator
from algvex.core.trace_serializer import TraceWriter

# Hummingbot imports
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative import BinancePerpetualDerivative


async def main():
    # 1. 初始化 Hummingbot Connector
    connector = BinancePerpetualDerivative(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        trading_pairs=["BTCUSDT", "ETHUSDT"],
    )

    # 2. 初始化 TraceWriter
    trace_writer = TraceWriter(output_path=f"logs/live_output_{date.today()}.jsonl")

    # 3. 初始化 HummingbotBridge
    bridge = HummingbotBridge(
        connector=connector,
        trace_writer=trace_writer,
    )

    # 4. 注册事件处理
    event_handler = AlgVexEventHandler(bridge, trace_writer)
    event_handler.register_events(connector)

    # 5. 初始化 SignalGenerator
    signal_generator = SignalGenerator()

    # 6. 初始化 Strategy
    strategy = AlgVexStrategy(
        connectors={"BINANCE": connector},
        signal_generator=signal_generator,
    )

    # 7. 初始化状态同步器
    synchronizer = StateSynchronizer(
        bridge=bridge,
        position_manager=strategy.position_manager,
        sync_interval=60.0,
    )

    # 8. 启动
    await connector.start()
    asyncio.create_task(synchronizer.start())

    # 9. 主循环
    while True:
        current_time = time.time()
        await strategy.tick(current_time)
        await asyncio.sleep(1)  # 1秒 tick


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. 测试计划

### 5.1 单元测试

```python
# tests/execution/test_hummingbot_bridge.py

class TestHummingbotBridge:
    """HummingbotBridge 测试"""

    def test_idempotent_order_id(self):
        """测试订单 ID 幂等性"""
        signal = create_test_signal()
        bridge = HummingbotBridge(mock_connector)

        id1 = bridge._generate_idempotent_order_id(signal)
        id2 = bridge._generate_idempotent_order_id(signal)

        assert id1 == id2, "相同信号应生成相同订单 ID"

    def test_duplicate_signal_handling(self):
        """测试重复信号处理"""
        signal = create_test_signal()
        bridge = HummingbotBridge(mock_connector)

        result1 = await bridge.execute_signal(signal)
        result2 = await bridge.execute_signal(signal)

        assert result1["status"] == "submitted"
        assert result2["status"] == "duplicate"

    def test_position_reconciliation(self):
        """测试仓位对账"""
        bridge = HummingbotBridge(mock_connector)
        mock_connector.set_position("BTCUSDT", amount=Decimal("1.0"))

        result = await bridge.reconcile()

        assert result["aligned"] == True
```

### 5.2 集成测试

```python
# tests/integration/test_hummingbot_integration.py

class TestHummingbotIntegration:
    """Hummingbot 集成测试"""

    @pytest.fixture
    def paper_trading_connector(self):
        """Paper Trading 连接器"""
        return MockBinanceConnector(mode="paper")

    async def test_full_signal_to_order_flow(self, paper_trading_connector):
        """测试完整信号 -> 订单流程"""
        bridge = HummingbotBridge(paper_trading_connector)
        signal_generator = SignalGenerator()

        # 生成信号
        signal = await signal_generator.get_signal("BTCUSDT")

        # 执行
        result = await bridge.execute_signal(signal)

        assert result["status"] == "submitted"

        # 等待成交
        await asyncio.sleep(1)

        # 验证 trace
        traces = read_trace_file()
        assert any(t["type"] == "order_filled" for t in traces)
```

---

## 6. 后续规划

### 6.1 Phase 1 (MVP)

- [x] HummingbotBridge 基础实现
- [x] 订单幂等性
- [x] 事件处理
- [ ] Paper Trading 验证

### 6.2 Phase 2

- [ ] AlgVexController 完整实现
- [ ] 多策略支持
- [ ] 高级订单类型 (限价单, 止损单)

### 6.3 Phase 3

- [ ] 多交易所支持 (Bybit, OKX)
- [ ] 智能订单路由
- [ ] 执行算法 (TWAP, VWAP)
