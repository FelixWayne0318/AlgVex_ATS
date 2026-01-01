"""
Hummingbot 桥接模块 v2.0.0

将 Qlib 信号转换为 Hummingbot 可执行的交易指令。
完全集成多交易所连接器和高级执行策略。

功能:
1. 信号转换: Qlib 信号 → 交易订单
2. 多交易所: Binance, Bybit, OKX, Gate.io
3. 执行策略: TWAP, VWAP, Grid, DCA, Iceberg
4. 风控集成: 预交易检查, 仓位管理
5. 幂等执行: 重放安全

版本: 2.0.0
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .exchange_connectors import (
    BaseExchangeConnector,
    ConnectorFactory,
    ExchangeConfig,
    ExchangeType,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Ticker,
)
from .executors import (
    BaseExecutor,
    DCAExecutor,
    ExecutorFactory,
    ExecutorResult,
    GridExecutor,
    IcebergExecutor,
    TWAPExecutor,
    VWAPExecutor,
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    DIRECT = "direct"  # 直接下单
    TWAP = "twap"  # 时间加权
    VWAP = "vwap"  # 成交量加权
    ICEBERG = "iceberg"  # 冰山订单


@dataclass
class Signal:
    """Qlib 信号"""
    signal_id: str  # 唯一标识
    symbol: str
    score: float  # -1 到 1，负数做空，正数做多
    timestamp: datetime
    confidence: float = 0.0
    features: Optional[Dict] = None
    model_version: str = ""

    def __post_init__(self):
        if not self.signal_id:
            # 自动生成 signal_id
            hash_input = f"{self.symbol}:{self.score}:{self.timestamp.isoformat()}"
            self.signal_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


@dataclass
class ExecutionConfig:
    """执行配置"""
    # 订单参数
    min_order_value: Decimal = Decimal("10.0")
    max_order_value: Decimal = Decimal("100000.0")
    default_leverage: int = 3
    max_leverage: int = 10

    # 信号过滤
    min_signal_strength: float = 0.3
    min_confidence: float = 0.0

    # 执行策略
    execution_mode: ExecutionMode = ExecutionMode.DIRECT
    twap_duration_minutes: int = 30
    twap_slices: int = 6

    # 风控
    max_positions: int = 10
    max_position_value: Decimal = Decimal("50000.0")
    daily_loss_limit: Decimal = Decimal("5000.0")

    # 重试
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class Order:
    """交易订单"""
    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    leverage: int = 1
    reduce_only: bool = False
    time_in_force: str = "GTC"
    status: str = "pending"
    exchange_order_id: Optional[str] = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class HummingbotBridge:
    """
    Hummingbot 桥接器 v2.0.0

    完全重写，支持:
    - 多交易所连接 (Binance, Bybit, OKX, etc.)
    - 高级执行策略 (TWAP, VWAP, Grid, DCA)
    - 幂等执行 (重放安全)
    - 风控集成
    """

    def __init__(
        self,
        exchange: str = "binance_perpetual",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = True,
        config: Optional[ExecutionConfig] = None,
    ):
        self.exchange_type = ExchangeType(exchange)
        self.testnet = testnet
        self.config = config or ExecutionConfig()

        # 交易所配置
        self._exchange_config = ExchangeConfig(
            exchange_type=self.exchange_type,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=testnet,
        )

        # 连接器
        self._connector: Optional[BaseExchangeConnector] = None

        # 状态
        self._is_connected = False
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._signal_order_map: Dict[str, str] = {}  # signal_id -> order_id

        # 执行器
        self._active_executors: Dict[str, BaseExecutor] = {}

        # 统计
        self._stats = {
            "total_signals": 0,
            "executed_signals": 0,
            "rejected_signals": 0,
            "total_orders": 0,
            "filled_orders": 0,
            "failed_orders": 0,
            "total_pnl": Decimal("0"),
            "daily_pnl": Decimal("0"),
        }

        # 回调
        self._on_order_update: Optional[Callable] = None
        self._on_position_update: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        logger.info(
            f"HummingbotBridge v2.0.0 initialized: "
            f"{exchange} testnet={testnet}"
        )

    # ============== 连接管理 ==============

    async def connect(self) -> bool:
        """连接到交易所"""
        try:
            self._connector = ConnectorFactory.create(self._exchange_config)
            success = await self._connector.connect()

            if success:
                self._is_connected = True
                # 同步持仓
                await self._sync_positions()
                logger.info(f"Connected to {self.exchange_type.value}")

            return success

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if self._on_error:
                self._on_error(e)
            return False

    async def disconnect(self):
        """断开连接"""
        # 取消所有活动执行器
        for executor in self._active_executors.values():
            await executor.cancel()

        if self._connector:
            await self._connector.disconnect()

        self._is_connected = False
        logger.info("Disconnected")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ============== 信号处理 ==============

    def _generate_idempotent_order_id(self, signal_id: str) -> str:
        """
        生成幂等的订单ID

        相同的 signal_id 总是生成相同的 order_id，
        确保重放相同信号不会产生重复订单。
        """
        hash_input = f"algvex:v2:{signal_id}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"algvex_{hash_value}"

    def _check_duplicate_signal(self, signal: Signal) -> bool:
        """检查重复信号"""
        order_id = self._generate_idempotent_order_id(signal.signal_id)
        return order_id in self._orders or signal.signal_id in self._signal_order_map

    def signal_to_order(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float = 0.02,
        leverage: Optional[int] = None,
    ) -> Optional[Order]:
        """
        将 Qlib 信号转换为订单

        Args:
            signal: Qlib 信号 (-1 到 1)
            capital: 可用资金
            risk_per_trade: 单笔风险比例 (默认2%)
            leverage: 杠杆倍数 (默认使用配置)

        Returns:
            Order 或 None (信号太弱或被过滤)
        """
        self._stats["total_signals"] += 1

        # 检查重复
        if self._check_duplicate_signal(signal):
            logger.debug(f"Duplicate signal ignored: {signal.signal_id}")
            return None

        # 过滤弱信号
        if abs(signal.score) < self.config.min_signal_strength:
            logger.debug(f"Signal too weak: {signal.score:.2f}")
            self._stats["rejected_signals"] += 1
            return None

        # 检查置信度
        if signal.confidence < self.config.min_confidence:
            logger.debug(f"Signal confidence too low: {signal.confidence:.2f}")
            self._stats["rejected_signals"] += 1
            return None

        # 确定方向
        side = OrderSide.BUY if signal.score > 0 else OrderSide.SELL

        # 计算仓位大小 (根据信号强度调整)
        lev = leverage or self.config.default_leverage
        lev = min(lev, self.config.max_leverage)

        signal_strength = abs(signal.score)
        position_value = Decimal(str(capital * risk_per_trade * signal_strength * lev))

        # 订单值检查
        if position_value < self.config.min_order_value:
            logger.debug(f"Order value too small: {position_value:.2f}")
            self._stats["rejected_signals"] += 1
            return None

        if position_value > self.config.max_order_value:
            position_value = self.config.max_order_value
            logger.warning(f"Order value capped at {position_value:.2f}")

        # 创建订单
        order_id = self._generate_idempotent_order_id(signal.signal_id)

        order = Order(
            order_id=order_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=position_value,  # 这里是价值，需要根据价格转换
            leverage=lev,
        )

        self._signal_order_map[signal.signal_id] = order_id

        logger.info(
            f"Signal converted: {signal.symbol} {side.value} "
            f"score={signal.score:.2f} confidence={signal.confidence:.2f} "
            f"value=${position_value:.2f}"
        )

        return order

    # ============== 订单执行 ==============

    async def execute_order(
        self,
        order: Order,
        execution_mode: Optional[ExecutionMode] = None,
    ) -> Dict[str, Any]:
        """
        执行订单

        Args:
            order: 订单对象
            execution_mode: 执行模式 (覆盖默认配置)

        Returns:
            执行结果
        """
        if not self._is_connected:
            return {"error": "Not connected", "order_id": order.order_id}

        mode = execution_mode or self.config.execution_mode
        self._stats["total_orders"] += 1

        try:
            # 获取当前价格
            ticker = await self._connector.get_ticker(order.symbol)
            current_price = ticker.last_price

            # 将价值转换为数量
            quantity = order.quantity / current_price

            logger.info(
                f"Executing order: {order.order_id} "
                f"mode={mode.value} qty={quantity:.6f}"
            )

            # 设置杠杆
            await self._connector.set_leverage(order.symbol, order.leverage)

            # 根据执行模式选择策略
            if mode == ExecutionMode.DIRECT:
                result = await self._execute_direct(order, quantity)
            elif mode == ExecutionMode.TWAP:
                result = await self._execute_twap(order, quantity)
            elif mode == ExecutionMode.VWAP:
                result = await self._execute_vwap(order, quantity)
            elif mode == ExecutionMode.ICEBERG:
                result = await self._execute_iceberg(order, quantity, current_price)
            else:
                result = await self._execute_direct(order, quantity)

            # 更新统计
            if result.get("status") == "filled":
                self._stats["filled_orders"] += 1
                self._stats["executed_signals"] += 1
            elif result.get("error"):
                self._stats["failed_orders"] += 1

            # 更新订单状态
            order.status = result.get("status", "unknown")
            order.exchange_order_id = result.get("exchange_order_id")
            order.filled_quantity = Decimal(str(result.get("filled_quantity", 0)))
            order.average_price = Decimal(str(result.get("average_price", 0)))
            order.fee = Decimal(str(result.get("fee", 0)))
            order.updated_at = datetime.now()

            self._orders[order.order_id] = order

            if self._on_order_update:
                self._on_order_update(order)

            # 同步持仓
            await self._sync_positions()

            return result

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            self._stats["failed_orders"] += 1
            order.status = "failed"
            self._orders[order.order_id] = order

            if self._on_error:
                self._on_error(e)

            return {"error": str(e), "order_id": order.order_id}

    async def _execute_direct(
        self,
        order: Order,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """直接执行"""
        request = OrderRequest(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=quantity,
            price=order.price,
            leverage=order.leverage,
            reduce_only=order.reduce_only,
            client_order_id=order.order_id,
        )

        response = await self._connector.create_order(request)

        return {
            "order_id": order.order_id,
            "exchange_order_id": response.order_id,
            "status": response.status.value,
            "filled_quantity": float(response.filled_quantity),
            "average_price": float(response.average_price or 0),
            "fee": float(response.fee),
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_twap(
        self,
        order: Order,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """TWAP 执行"""
        executor = TWAPExecutor(
            connector=self._connector,
            symbol=order.symbol,
            side=order.side,
            total_quantity=quantity,
            duration_minutes=self.config.twap_duration_minutes,
            num_slices=self.config.twap_slices,
        )

        self._active_executors[order.order_id] = executor
        result = await executor.execute()
        del self._active_executors[order.order_id]

        return {
            "order_id": order.order_id,
            "status": "filled" if result.fill_rate > 0.95 else "partial",
            "filled_quantity": float(result.filled_quantity),
            "average_price": float(result.average_price),
            "fee": float(result.total_fee),
            "execution_time": str(result.duration),
            "slices": len(result.orders),
        }

    async def _execute_vwap(
        self,
        order: Order,
        quantity: Decimal,
    ) -> Dict[str, Any]:
        """VWAP 执行"""
        executor = VWAPExecutor(
            connector=self._connector,
            symbol=order.symbol,
            side=order.side,
            total_quantity=quantity,
            duration_minutes=self.config.twap_duration_minutes,
            num_slices=self.config.twap_slices,
        )

        self._active_executors[order.order_id] = executor
        result = await executor.execute()
        del self._active_executors[order.order_id]

        return {
            "order_id": order.order_id,
            "status": "filled" if result.fill_rate > 0.95 else "partial",
            "filled_quantity": float(result.filled_quantity),
            "average_price": float(result.average_price),
            "fee": float(result.total_fee),
        }

    async def _execute_iceberg(
        self,
        order: Order,
        quantity: Decimal,
        current_price: Decimal,
    ) -> Dict[str, Any]:
        """冰山订单执行"""
        visible_qty = quantity / 10  # 每次显示 10%

        executor = IcebergExecutor(
            connector=self._connector,
            symbol=order.symbol,
            side=order.side,
            total_quantity=quantity,
            visible_quantity=visible_qty,
            price=current_price,
        )

        self._active_executors[order.order_id] = executor
        result = await executor.execute()
        del self._active_executors[order.order_id]

        return {
            "order_id": order.order_id,
            "status": "filled" if result.fill_rate > 0.95 else "partial",
            "filled_quantity": float(result.filled_quantity),
            "average_price": float(result.average_price),
            "fee": float(result.total_fee),
        }

    # ============== 批量处理 ==============

    async def process_signals(
        self,
        signals: List[Signal],
        capital: float,
        max_positions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量处理 Qlib 信号

        Args:
            signals: 信号列表
            capital: 总资金
            max_positions: 最大持仓数
        """
        results = []
        max_pos = max_positions or self.config.max_positions

        # 按信号强度排序
        sorted_signals = sorted(signals, key=lambda s: abs(s.score), reverse=True)

        # 计算每个仓位的资金
        capital_per_position = capital / max_pos

        for signal in sorted_signals[:max_pos]:
            order = self.signal_to_order(
                signal=signal,
                capital=capital_per_position,
            )

            if order:
                result = await self.execute_order(order)
                results.append(result)

        return results

    # ============== 仓位管理 ==============

    async def _sync_positions(self):
        """同步持仓"""
        if not self._is_connected:
            return

        try:
            positions = await self._connector.get_positions()
            self._positions = {p.symbol: p for p in positions if p.quantity > 0}

            if self._on_position_update:
                self._on_position_update(self._positions)

        except Exception as e:
            logger.error(f"Sync positions failed: {e}")

    async def get_positions(self) -> Dict[str, Position]:
        """获取当前持仓"""
        await self._sync_positions()
        return self._positions

    async def close_position(
        self,
        symbol: str,
        reduce_percent: float = 1.0,
    ) -> Dict[str, Any]:
        """
        平仓

        Args:
            symbol: 交易对
            reduce_percent: 平仓比例 (0-1)
        """
        position = self._positions.get(symbol)
        if not position:
            return {"error": f"No position for {symbol}"}

        close_quantity = position.quantity * Decimal(str(reduce_percent))
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        request = OrderRequest(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            reduce_only=True,
        )

        response = await self._connector.create_order(request)

        await self._sync_positions()

        return {
            "order_id": response.order_id,
            "status": response.status.value,
            "closed_quantity": float(close_quantity),
        }

    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """平掉所有仓位"""
        results = []
        for symbol in list(self._positions.keys()):
            result = await self.close_position(symbol)
            results.append(result)
        return results

    # ============== 止盈止损 ==============

    async def set_stop_loss(
        self,
        symbol: str,
        stop_price: float,
    ) -> Dict[str, Any]:
        """设置止损"""
        position = self._positions.get(symbol)
        if not position:
            return {"error": f"No position for {symbol}"}

        request = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            order_type=OrderType.STOP_MARKET,
            quantity=position.quantity,
            stop_price=Decimal(str(stop_price)),
            reduce_only=True,
        )

        response = await self._connector.create_order(request)

        return {
            "order_id": response.order_id,
            "stop_price": stop_price,
        }

    async def set_take_profit(
        self,
        symbol: str,
        take_profit_price: float,
    ) -> Dict[str, Any]:
        """设置止盈"""
        position = self._positions.get(symbol)
        if not position:
            return {"error": f"No position for {symbol}"}

        request = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            quantity=position.quantity,
            stop_price=Decimal(str(take_profit_price)),
            reduce_only=True,
        )

        response = await self._connector.create_order(request)

        return {
            "order_id": response.order_id,
            "take_profit_price": take_profit_price,
        }

    # ============== 网格交易 ==============

    async def start_grid_trading(
        self,
        symbol: str,
        total_amount: float,
        lower_price: float,
        upper_price: float,
        num_grids: int = 10,
    ) -> str:
        """
        启动网格交易

        Returns:
            executor_id
        """
        executor = GridExecutor(
            connector=self._connector,
            symbol=symbol,
            total_quantity=Decimal(str(total_amount)),
            lower_price=Decimal(str(lower_price)),
            upper_price=Decimal(str(upper_price)),
            num_grids=num_grids,
        )

        self._active_executors[executor.executor_id] = executor
        asyncio.create_task(executor.execute())

        return executor.executor_id

    async def stop_grid_trading(self, executor_id: str) -> bool:
        """停止网格交易"""
        executor = self._active_executors.get(executor_id)
        if executor:
            await executor.cancel()
            del self._active_executors[executor_id]
            return True
        return False

    # ============== DCA 定投 ==============

    async def start_dca(
        self,
        symbol: str,
        side: str,
        amount_per_order: float,
        num_orders: int = 10,
        interval_hours: float = 24.0,
    ) -> str:
        """
        启动 DCA 定投

        Returns:
            executor_id
        """
        executor = DCAExecutor(
            connector=self._connector,
            symbol=symbol,
            side=OrderSide(side.lower()),
            amount_per_order=Decimal(str(amount_per_order)),
            num_orders=num_orders,
            interval_hours=interval_hours,
        )

        self._active_executors[executor.executor_id] = executor
        asyncio.create_task(executor.execute())

        return executor.executor_id

    async def stop_dca(self, executor_id: str) -> bool:
        """停止 DCA"""
        executor = self._active_executors.get(executor_id)
        if executor:
            await executor.cancel()
            del self._active_executors[executor_id]
            return True
        return False

    # ============== 回调注册 ==============

    def on_order_update(self, callback: Callable):
        """注册订单更新回调"""
        self._on_order_update = callback

    def on_position_update(self, callback: Callable):
        """注册持仓更新回调"""
        self._on_position_update = callback

    def on_error(self, callback: Callable):
        """注册错误回调"""
        self._on_error = callback

    # ============== 统计信息 ==============

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self._stats,
            "active_positions": len(self._positions),
            "active_executors": len(self._active_executors),
            "orders_in_memory": len(self._orders),
            "fill_rate": (
                self._stats["filled_orders"] / self._stats["total_orders"]
                if self._stats["total_orders"] > 0 else 0
            ),
            "signal_conversion_rate": (
                self._stats["executed_signals"] / self._stats["total_signals"]
                if self._stats["total_signals"] > 0 else 0
            ),
        }

    def reset_daily_stats(self):
        """重置每日统计"""
        self._stats["daily_pnl"] = Decimal("0")


# 便捷函数
async def create_bridge(
    exchange: str = "binance_perpetual",
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = True,
) -> HummingbotBridge:
    """
    创建并连接桥接器

    Example:
        bridge = await create_bridge("binance_perpetual", api_key, api_secret)
        result = await bridge.execute_order(order)
    """
    bridge = HummingbotBridge(
        exchange=exchange,
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    await bridge.connect()
    return bridge


# 测试代码
if __name__ == "__main__":
    print("AlgVex HummingbotBridge v2.0.0")
    print("=" * 50)
    print("\n功能:")
    print("  - 多交易所支持 (Binance, Bybit, OKX)")
    print("  - 执行策略 (TWAP, VWAP, Grid, DCA, Iceberg)")
    print("  - 幂等执行 (重放安全)")
    print("  - 风控集成")
    print("\n使用示例:")
    print("  bridge = await create_bridge('binance_perpetual', api_key, api_secret)")
    print("  order = bridge.signal_to_order(signal, capital=10000)")
    print("  result = await bridge.execute_order(order)")
