"""
AlgVex 策略执行器

实现高级订单执行策略:
- TWAP (时间加权平均价格)
- VWAP (成交量加权平均价格)
- Grid (网格交易)
- DCA (定投策略)
- Iceberg (冰山订单)

版本: 2.0.0
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .exchange_connectors import (
    BaseExchangeConnector,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    OrderStatus,
    Ticker,
)

logger = logging.getLogger(__name__)


class ExecutorStatus(Enum):
    """执行器状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutorResult:
    """执行结果"""
    executor_id: str
    status: ExecutorStatus
    total_quantity: Decimal
    filled_quantity: Decimal
    average_price: Decimal
    total_fee: Decimal
    orders: List[OrderResponse] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def fill_rate(self) -> float:
        if self.total_quantity == 0:
            return 0.0
        return float(self.filled_quantity / self.total_quantity)

    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time


class BaseExecutor(ABC):
    """执行器基类"""

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
    ):
        self.connector = connector
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity

        self._executor_id = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self._status = ExecutorStatus.PENDING
        self._filled_quantity = Decimal("0")
        self._total_cost = Decimal("0")
        self._total_fee = Decimal("0")
        self._orders: List[OrderResponse] = []
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._error: Optional[str] = None

        # 回调
        self._on_fill: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None

    @property
    def executor_id(self) -> str:
        return self._executor_id

    @property
    def status(self) -> ExecutorStatus:
        return self._status

    @property
    def remaining_quantity(self) -> Decimal:
        return self.total_quantity - self._filled_quantity

    @property
    def average_price(self) -> Decimal:
        if self._filled_quantity == 0:
            return Decimal("0")
        return self._total_cost / self._filled_quantity

    def on_fill(self, callback: Callable):
        """注册成交回调"""
        self._on_fill = callback

    def on_complete(self, callback: Callable):
        """注册完成回调"""
        self._on_complete = callback

    @abstractmethod
    async def execute(self) -> ExecutorResult:
        """执行策略"""
        pass

    async def cancel(self) -> bool:
        """取消执行"""
        self._status = ExecutorStatus.CANCELLED
        self._end_time = datetime.now()
        return True

    def get_result(self) -> ExecutorResult:
        """获取执行结果"""
        return ExecutorResult(
            executor_id=self._executor_id,
            status=self._status,
            total_quantity=self.total_quantity,
            filled_quantity=self._filled_quantity,
            average_price=self.average_price,
            total_fee=self._total_fee,
            orders=self._orders,
            start_time=self._start_time or datetime.now(),
            end_time=self._end_time,
            error=self._error,
        )

    async def _place_order(
        self,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
    ) -> Optional[OrderResponse]:
        """下单辅助方法"""
        try:
            request = OrderRequest(
                symbol=self.symbol,
                side=self.side,
                order_type=order_type,
                quantity=quantity,
                price=price,
            )
            response = await self.connector.create_order(request)
            self._orders.append(response)

            if response.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                self._filled_quantity += response.filled_quantity
                if response.average_price:
                    self._total_cost += response.filled_quantity * response.average_price
                self._total_fee += response.fee

                if self._on_fill:
                    self._on_fill(response)

            return response

        except Exception as e:
            logger.error(f"Order failed: {e}")
            self._error = str(e)
            return None


class TWAPExecutor(BaseExecutor):
    """
    TWAP (时间加权平均价格) 执行器

    将大订单拆分成多个小订单，在指定时间内均匀执行。
    """

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        duration_minutes: int = 60,
        num_slices: int = 12,
        randomize: bool = True,
        max_deviation: float = 0.1,  # 随机偏差范围
    ):
        super().__init__(connector, symbol, side, total_quantity)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.randomize = randomize
        self.max_deviation = max_deviation

    async def execute(self) -> ExecutorResult:
        """执行 TWAP 策略"""
        self._status = ExecutorStatus.RUNNING
        self._start_time = datetime.now()

        # 计算每次下单数量和间隔
        slice_quantity = self.total_quantity / self.num_slices
        interval_seconds = (self.duration_minutes * 60) / self.num_slices

        logger.info(
            f"TWAP started: {self.symbol} {self.side.value} "
            f"qty={self.total_quantity} slices={self.num_slices} "
            f"interval={interval_seconds:.1f}s"
        )

        for i in range(self.num_slices):
            if self._status == ExecutorStatus.CANCELLED:
                break

            if self.remaining_quantity <= 0:
                break

            # 计算本次下单数量 (最后一次下剩余量)
            qty = min(slice_quantity, self.remaining_quantity)

            # 添加随机性
            if self.randomize and i < self.num_slices - 1:
                import random
                deviation = random.uniform(-self.max_deviation, self.max_deviation)
                qty = qty * Decimal(str(1 + deviation))
                qty = min(qty, self.remaining_quantity)

            # 下单
            response = await self._place_order(qty, OrderType.MARKET)
            if response:
                logger.info(
                    f"TWAP slice {i+1}/{self.num_slices}: "
                    f"qty={qty:.4f} filled={response.filled_quantity:.4f}"
                )

            # 等待下一次
            if i < self.num_slices - 1 and self.remaining_quantity > 0:
                # 添加随机延迟
                delay = interval_seconds
                if self.randomize:
                    import random
                    delay *= random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)

        self._status = ExecutorStatus.COMPLETED
        self._end_time = datetime.now()

        if self._on_complete:
            self._on_complete(self.get_result())

        logger.info(
            f"TWAP completed: filled={self._filled_quantity}/{self.total_quantity} "
            f"avg_price={self.average_price:.4f}"
        )

        return self.get_result()


class VWAPExecutor(BaseExecutor):
    """
    VWAP (成交量加权平均价格) 执行器

    根据历史成交量分布调整下单节奏。
    """

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        duration_minutes: int = 60,
        num_slices: int = 12,
        volume_profile: Optional[List[float]] = None,  # 成交量分布
    ):
        super().__init__(connector, symbol, side, total_quantity)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices

        # 默认成交量分布 (早盘和尾盘更活跃)
        self.volume_profile = volume_profile or self._default_volume_profile()

    def _default_volume_profile(self) -> List[float]:
        """默认成交量分布"""
        # 加密货币市场通常 UTC 时间 8-10, 14-16 更活跃
        profile = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        # 归一化
        total = sum(profile)
        return [p / total * self.num_slices for p in profile]

    async def execute(self) -> ExecutorResult:
        """执行 VWAP 策略"""
        self._status = ExecutorStatus.RUNNING
        self._start_time = datetime.now()

        interval_seconds = (self.duration_minutes * 60) / self.num_slices

        logger.info(
            f"VWAP started: {self.symbol} {self.side.value} "
            f"qty={self.total_quantity}"
        )

        for i in range(self.num_slices):
            if self._status == ExecutorStatus.CANCELLED:
                break

            if self.remaining_quantity <= 0:
                break

            # 根据成交量分布计算下单量
            volume_weight = self.volume_profile[i % len(self.volume_profile)]
            base_qty = self.total_quantity / self.num_slices
            qty = base_qty * Decimal(str(volume_weight))
            qty = min(qty, self.remaining_quantity)

            # 下单
            response = await self._place_order(qty, OrderType.MARKET)
            if response:
                logger.info(
                    f"VWAP slice {i+1}/{self.num_slices}: "
                    f"weight={volume_weight:.2f} qty={qty:.4f}"
                )

            if i < self.num_slices - 1 and self.remaining_quantity > 0:
                await asyncio.sleep(interval_seconds)

        self._status = ExecutorStatus.COMPLETED
        self._end_time = datetime.now()

        if self._on_complete:
            self._on_complete(self.get_result())

        return self.get_result()


class GridExecutor(BaseExecutor):
    """
    网格交易执行器

    在价格区间内设置多个买卖订单，低买高卖获利。
    """

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        total_quantity: Decimal,
        lower_price: Decimal,
        upper_price: Decimal,
        num_grids: int = 10,
        grid_type: str = "arithmetic",  # arithmetic or geometric
    ):
        # Grid 是双向的，side 设为 BUY 作为默认
        super().__init__(connector, symbol, OrderSide.BUY, total_quantity)
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.num_grids = num_grids
        self.grid_type = grid_type

        self._grid_prices: List[Decimal] = []
        self._grid_orders: Dict[str, str] = {}  # price -> order_id
        self._running = False

    def _calculate_grid_prices(self) -> List[Decimal]:
        """计算网格价格"""
        prices = []

        if self.grid_type == "arithmetic":
            # 等差网格
            step = (self.upper_price - self.lower_price) / self.num_grids
            for i in range(self.num_grids + 1):
                prices.append(self.lower_price + step * i)
        else:
            # 等比网格
            import math
            ratio = (float(self.upper_price) / float(self.lower_price)) ** (1 / self.num_grids)
            for i in range(self.num_grids + 1):
                prices.append(self.lower_price * Decimal(str(ratio ** i)))

        return prices

    async def execute(self) -> ExecutorResult:
        """执行网格策略"""
        self._status = ExecutorStatus.RUNNING
        self._start_time = datetime.now()
        self._running = True

        # 计算网格价格
        self._grid_prices = self._calculate_grid_prices()

        # 获取当前价格
        ticker = await self.connector.get_ticker(self.symbol)
        current_price = ticker.last_price

        # 每格数量
        qty_per_grid = self.total_quantity / self.num_grids

        logger.info(
            f"Grid started: {self.symbol} "
            f"range=[{self.lower_price}, {self.upper_price}] "
            f"grids={self.num_grids} current={current_price}"
        )

        # 在当前价格下方挂买单，上方挂卖单
        for i, price in enumerate(self._grid_prices):
            if not self._running:
                break

            if price < current_price:
                # 买单
                request = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=qty_per_grid,
                    price=price,
                )
            else:
                # 卖单
                request = OrderRequest(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=qty_per_grid,
                    price=price,
                )

            try:
                response = await self.connector.create_order(request)
                self._orders.append(response)
                self._grid_orders[str(price)] = response.order_id
                logger.debug(f"Grid order placed: {request.side.value} @ {price}")
            except Exception as e:
                logger.error(f"Grid order failed: {e}")

        # 网格需要持续运行，这里只是初始化
        # 实际运行需要监听成交并补单
        self._status = ExecutorStatus.RUNNING

        return self.get_result()

    async def cancel(self) -> bool:
        """取消网格"""
        self._running = False

        # 取消所有挂单
        for order_id in self._grid_orders.values():
            try:
                await self.connector.cancel_order(self.symbol, order_id)
            except Exception as e:
                logger.error(f"Cancel grid order failed: {e}")

        self._status = ExecutorStatus.CANCELLED
        self._end_time = datetime.now()
        return True


class DCAExecutor(BaseExecutor):
    """
    DCA (Dollar Cost Averaging) 定投执行器

    定期定额买入，平滑成本。
    """

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        side: OrderSide,
        amount_per_order: Decimal,  # 每次买入金额
        num_orders: int = 10,
        interval_hours: float = 24.0,  # 间隔小时
        price_limit: Optional[Decimal] = None,  # 价格上限
    ):
        # DCA 的 total_quantity 是动态的
        super().__init__(connector, symbol, side, Decimal("0"))
        self.amount_per_order = amount_per_order
        self.num_orders = num_orders
        self.interval_hours = interval_hours
        self.price_limit = price_limit

        self._executed_orders = 0

    async def execute(self) -> ExecutorResult:
        """执行 DCA 策略"""
        self._status = ExecutorStatus.RUNNING
        self._start_time = datetime.now()

        logger.info(
            f"DCA started: {self.symbol} {self.side.value} "
            f"amount=${self.amount_per_order} orders={self.num_orders} "
            f"interval={self.interval_hours}h"
        )

        for i in range(self.num_orders):
            if self._status == ExecutorStatus.CANCELLED:
                break

            # 获取当前价格
            ticker = await self.connector.get_ticker(self.symbol)
            current_price = ticker.last_price

            # 检查价格限制
            if self.price_limit:
                if self.side == OrderSide.BUY and current_price > self.price_limit:
                    logger.info(f"DCA skipped: price {current_price} > limit {self.price_limit}")
                    await asyncio.sleep(self.interval_hours * 3600)
                    continue
                elif self.side == OrderSide.SELL and current_price < self.price_limit:
                    logger.info(f"DCA skipped: price {current_price} < limit {self.price_limit}")
                    await asyncio.sleep(self.interval_hours * 3600)
                    continue

            # 计算数量 (金额 / 价格)
            quantity = self.amount_per_order / current_price

            # 下单
            response = await self._place_order(quantity, OrderType.MARKET)
            if response:
                self._executed_orders += 1
                self.total_quantity += quantity  # 更新总量
                logger.info(
                    f"DCA order {i+1}/{self.num_orders}: "
                    f"price={current_price} qty={quantity:.6f}"
                )

            # 等待下一次
            if i < self.num_orders - 1:
                await asyncio.sleep(self.interval_hours * 3600)

        self._status = ExecutorStatus.COMPLETED
        self._end_time = datetime.now()

        if self._on_complete:
            self._on_complete(self.get_result())

        logger.info(
            f"DCA completed: {self._executed_orders}/{self.num_orders} orders "
            f"total_qty={self._filled_quantity} avg_price={self.average_price:.4f}"
        )

        return self.get_result()


class IcebergExecutor(BaseExecutor):
    """
    冰山订单执行器

    将大单拆分成小单，每次只显示一部分。
    """

    def __init__(
        self,
        connector: BaseExchangeConnector,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        visible_quantity: Decimal,  # 每次可见数量
        price: Decimal,
        price_variance: Decimal = Decimal("0"),  # 价格浮动范围
    ):
        super().__init__(connector, symbol, side, total_quantity)
        self.visible_quantity = visible_quantity
        self.price = price
        self.price_variance = price_variance

    async def execute(self) -> ExecutorResult:
        """执行冰山策略"""
        self._status = ExecutorStatus.RUNNING
        self._start_time = datetime.now()

        logger.info(
            f"Iceberg started: {self.symbol} {self.side.value} "
            f"total={self.total_quantity} visible={self.visible_quantity}"
        )

        order_count = 0
        while self.remaining_quantity > 0 and self._status == ExecutorStatus.RUNNING:
            # 计算本次下单量
            qty = min(self.visible_quantity, self.remaining_quantity)

            # 价格浮动
            order_price = self.price
            if self.price_variance > 0:
                import random
                variance = Decimal(str(random.uniform(-1, 1))) * self.price_variance
                order_price = self.price + variance

            # 下限价单
            response = await self._place_order(qty, OrderType.LIMIT, order_price)

            if response:
                order_count += 1
                logger.debug(
                    f"Iceberg order {order_count}: qty={qty} price={order_price}"
                )

                # 等待成交
                max_wait = 60  # 最大等待时间
                waited = 0
                while waited < max_wait:
                    order = await self.connector.get_order(self.symbol, response.order_id)
                    if order.status == OrderStatus.FILLED:
                        break
                    elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                        break
                    await asyncio.sleep(1)
                    waited += 1

                # 未完全成交则取消
                if order.status not in (OrderStatus.FILLED,):
                    await self.connector.cancel_order(self.symbol, response.order_id)

        self._status = ExecutorStatus.COMPLETED
        self._end_time = datetime.now()

        if self._on_complete:
            self._on_complete(self.get_result())

        return self.get_result()


# ============== 执行器工厂 ==============

class ExecutorFactory:
    """执行器工厂"""

    @staticmethod
    def create_twap(
        connector: BaseExchangeConnector,
        symbol: str,
        side: str,
        quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 12,
    ) -> TWAPExecutor:
        """创建 TWAP 执行器"""
        return TWAPExecutor(
            connector=connector,
            symbol=symbol,
            side=OrderSide(side.lower()),
            total_quantity=Decimal(str(quantity)),
            duration_minutes=duration_minutes,
            num_slices=num_slices,
        )

    @staticmethod
    def create_vwap(
        connector: BaseExchangeConnector,
        symbol: str,
        side: str,
        quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 12,
    ) -> VWAPExecutor:
        """创建 VWAP 执行器"""
        return VWAPExecutor(
            connector=connector,
            symbol=symbol,
            side=OrderSide(side.lower()),
            total_quantity=Decimal(str(quantity)),
            duration_minutes=duration_minutes,
            num_slices=num_slices,
        )

    @staticmethod
    def create_grid(
        connector: BaseExchangeConnector,
        symbol: str,
        quantity: float,
        lower_price: float,
        upper_price: float,
        num_grids: int = 10,
    ) -> GridExecutor:
        """创建网格执行器"""
        return GridExecutor(
            connector=connector,
            symbol=symbol,
            total_quantity=Decimal(str(quantity)),
            lower_price=Decimal(str(lower_price)),
            upper_price=Decimal(str(upper_price)),
            num_grids=num_grids,
        )

    @staticmethod
    def create_dca(
        connector: BaseExchangeConnector,
        symbol: str,
        side: str,
        amount_per_order: float,
        num_orders: int = 10,
        interval_hours: float = 24.0,
    ) -> DCAExecutor:
        """创建 DCA 执行器"""
        return DCAExecutor(
            connector=connector,
            symbol=symbol,
            side=OrderSide(side.lower()),
            amount_per_order=Decimal(str(amount_per_order)),
            num_orders=num_orders,
            interval_hours=interval_hours,
        )

    @staticmethod
    def create_iceberg(
        connector: BaseExchangeConnector,
        symbol: str,
        side: str,
        total_quantity: float,
        visible_quantity: float,
        price: float,
    ) -> IcebergExecutor:
        """创建冰山执行器"""
        return IcebergExecutor(
            connector=connector,
            symbol=symbol,
            side=OrderSide(side.lower()),
            total_quantity=Decimal(str(total_quantity)),
            visible_quantity=Decimal(str(visible_quantity)),
            price=Decimal(str(price)),
        )


# 测试代码
if __name__ == "__main__":
    print("AlgVex Executors v2.0.0")
    print("=" * 50)
    print("\n可用执行器:")
    print("  - TWAP (时间加权平均价格)")
    print("  - VWAP (成交量加权平均价格)")
    print("  - Grid (网格交易)")
    print("  - DCA  (定投策略)")
    print("  - Iceberg (冰山订单)")
