"""
AlgVex 统一成交模型

设计文档参考: Section 12.5 回测-实盘成交对齐

核心功能:
- 统一回测和实盘的成交逻辑
- 成交价格计算 (考虑滑点)
- 手续费计算 (VIP等级)
- 部分成交模拟

使用方式:
    # 回测引擎使用
    class CryptoPerpetualBacktest:
        def __init__(self, config):
            self.execution_model = ExecutionModel(config.execution_config)

    # 实盘桥接器使用
    class HummingbotBridge:
        def __init__(self, config):
            self.execution_model = ExecutionModel(config.execution_config)
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from shared.execution_models import DynamicSlippageModel, FeeModel, VIPLevel
from shared.price_semantics import PriceSemantics, PriceScenario, PriceData


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


@dataclass
class ExecutionConfig:
    """执行配置"""
    # 交易所
    exchange: str = "binance"
    vip_level: VIPLevel = VIPLevel.VIP0

    # 费率 (可自定义覆盖)
    maker_fee: Optional[float] = None
    taker_fee: Optional[float] = None

    # 滑点
    slippage_model: str = "dynamic"  # static, dynamic
    base_slippage: float = 0.0001    # 0.01%
    max_slippage: float = 0.01       # 1%

    # 成交价格
    fill_price_type: str = "close_price"  # close_price, last_price, mark_price

    # 部分成交
    partial_fill_enabled: bool = True
    min_fill_ratio: float = 0.1      # 最小成交比例 10%


@dataclass
class FillResult:
    """成交结果"""
    filled: bool
    fill_price: float
    fill_quantity: float
    fee: float
    slippage: float
    slippage_cost: float
    total_cost: float
    partial: bool = False
    fill_ratio: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExecutionModel:
    """
    统一成交模型 - 回测和实盘共用

    确保回测的 fill_price, fee_model, slippage_model 与实盘一致。

    使用示例:
        config = ExecutionConfig(
            exchange="binance",
            vip_level=VIPLevel.VIP0,
        )
        model = ExecutionModel(config)

        # 计算成交价格
        fill_price = model.calculate_fill_price(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            market_data={"last_price": 50000, "symbol": "BTCUSDT"},
        )

        # 计算手续费
        fee = model.calculate_fee(notional=10000, is_maker=False)
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        初始化执行模型

        Args:
            config: 执行配置
        """
        self.config = config or ExecutionConfig()

        # 初始化组件
        self.price_semantics = PriceSemantics()
        self.slippage_model = DynamicSlippageModel(
            base_slippage=self.config.base_slippage,
            max_slippage=self.config.max_slippage,
        )
        self.fee_model = FeeModel(
            exchange=self.config.exchange,
            vip_level=self.config.vip_level,
            custom_fees={
                "maker": self.config.maker_fee,
                "taker": self.config.taker_fee,
            } if self.config.maker_fee or self.config.taker_fee else None,
        )

    def calculate_fill_price(
        self,
        side: OrderSide,
        order_type: OrderType,
        market_data: Dict[str, Any],
        limit_price: Optional[float] = None,
    ) -> float:
        """
        计算成交价格

        规则 (回测和实盘必须一致):
        - MARKET 单: base_price + slippage
        - LIMIT 单: limit_price (假设完全成交)
        - STOP/TP 单: 触发后按市价成交

        Args:
            side: 订单方向 (BUY/SELL)
            order_type: 订单类型
            market_data: 市场数据 (需包含 last_price, symbol 等)
            limit_price: 限价单价格

        Returns:
            成交价格
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        order_size_usd = market_data.get("order_size_usd", 10000)

        # 获取基准价格
        if order_type == OrderType.LIMIT and limit_price is not None:
            # 限价单直接使用限价
            return limit_price

        # 市价单或触发单使用市场价格
        base_price = self._get_base_price(market_data)

        # 计算滑点
        slippage = self.slippage_model.get_slippage_for_backtest(
            symbol=symbol,
            order_size_usd=order_size_usd,
            avg_daily_volume=market_data.get("avg_daily_volume", 10_000_000),
        )

        # 应用滑点
        if side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        return fill_price

    def calculate_fee(
        self,
        notional: float,
        is_maker: bool = False,
    ) -> float:
        """
        计算手续费

        Args:
            notional: 名义价值
            is_maker: 是否为 Maker

        Returns:
            手续费金额
        """
        return self.fee_model.calculate_fee(notional, is_maker)

    def calculate_slippage(
        self,
        symbol: str,
        order_size_usd: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        计算滑点

        Args:
            symbol: 交易对
            order_size_usd: 订单金额
            market_data: 市场数据

        Returns:
            滑点比例
        """
        avg_volume = 10_000_000
        if market_data:
            avg_volume = market_data.get("avg_daily_volume", avg_volume)

        return self.slippage_model.get_slippage_for_backtest(
            symbol=symbol,
            order_size_usd=order_size_usd,
            avg_daily_volume=avg_volume,
        )

    def execute_order(
        self,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        market_data: Dict[str, Any],
        limit_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> FillResult:
        """
        执行订单 (模拟成交)

        Args:
            side: 订单方向
            order_type: 订单类型
            quantity: 订单数量
            market_data: 市场数据
            limit_price: 限价
            timestamp: 时间戳

        Returns:
            FillResult 成交结果
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        base_price = self._get_base_price(market_data)

        # 计算成交价格
        fill_price = self.calculate_fill_price(
            side=side,
            order_type=order_type,
            market_data=market_data,
            limit_price=limit_price,
        )

        # 计算成交数量 (考虑部分成交)
        fill_quantity, partial, fill_ratio = self._calculate_fill_quantity(
            quantity=quantity,
            order_type=order_type,
            market_data=market_data,
        )

        # 计算名义价值
        notional = fill_quantity * fill_price

        # 计算滑点成本
        slippage = abs(fill_price - base_price) / base_price if base_price > 0 else 0
        slippage_cost = notional * slippage

        # 计算手续费 (市价单为 taker)
        is_maker = order_type == OrderType.LIMIT
        fee = self.calculate_fee(notional, is_maker)

        # 总成本
        total_cost = fee + slippage_cost

        return FillResult(
            filled=fill_quantity > 0,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fee=fee,
            slippage=slippage,
            slippage_cost=slippage_cost,
            total_cost=total_cost,
            partial=partial,
            fill_ratio=fill_ratio,
            timestamp=timestamp,
            metadata={
                "symbol": symbol,
                "side": side.value,
                "order_type": order_type.value,
                "base_price": base_price,
            },
        )

    def _get_base_price(self, market_data: Dict[str, Any]) -> float:
        """获取基准价格"""
        # 优先使用配置的价格类型
        price_type = self.config.fill_price_type

        if price_type == "last_price" and "last_price" in market_data:
            return market_data["last_price"]
        elif price_type == "mark_price" and "mark_price" in market_data:
            return market_data["mark_price"]
        elif price_type == "close_price" and "close_price" in market_data:
            return market_data["close_price"]

        # 回退顺序
        for key in ["close_price", "close", "last_price", "mark_price", "price"]:
            if key in market_data and market_data[key] is not None:
                return float(market_data[key])

        raise ValueError("No valid price found in market_data")

    def _calculate_fill_quantity(
        self,
        quantity: float,
        order_type: OrderType,
        market_data: Dict[str, Any],
    ) -> Tuple[float, bool, float]:
        """
        计算成交数量

        Args:
            quantity: 请求数量
            order_type: 订单类型
            market_data: 市场数据

        Returns:
            (成交数量, 是否部分成交, 成交比例)
        """
        if not self.config.partial_fill_enabled:
            return quantity, False, 1.0

        # 市价单通常完全成交
        if order_type == OrderType.MARKET:
            return quantity, False, 1.0

        # 限价单可能部分成交 (简化模型)
        # 实际应基于深度数据，这里使用随机模型
        import random
        fill_ratio = random.uniform(self.config.min_fill_ratio, 1.0)
        fill_quantity = quantity * fill_ratio
        partial = fill_ratio < 1.0

        return fill_quantity, partial, fill_ratio

    # ===== 对齐验证接口 =====

    def get_fill_price_impl(self) -> str:
        """获取成交价格实现"""
        return self.config.fill_price_type

    def get_partial_fill_impl(self) -> bool:
        """获取部分成交实现"""
        return self.config.partial_fill_enabled

    def get_fee_model_impl(self) -> Dict[str, Any]:
        """获取费率模型实现"""
        return self.fee_model.get_fee_summary()

    def get_slippage_model_impl(self) -> Dict[str, Any]:
        """获取滑点模型实现"""
        return {
            "type": self.config.slippage_model,
            "base": self.config.base_slippage,
            "max": self.config.max_slippage,
        }

    def validate_alignment(self, other: "ExecutionModel") -> Dict[str, Any]:
        """
        验证与另一个执行模型的对齐

        Args:
            other: 另一个执行模型 (例如实盘)

        Returns:
            对齐验证结果
        """
        results = {}

        # 检查各项配置
        checks = [
            ("fill_price", self.get_fill_price_impl(), other.get_fill_price_impl()),
            ("partial_fill", self.get_partial_fill_impl(), other.get_partial_fill_impl()),
            ("fee_model", self.get_fee_model_impl(), other.get_fee_model_impl()),
            ("slippage_model", self.get_slippage_model_impl(), other.get_slippage_model_impl()),
        ]

        all_aligned = True
        for name, self_impl, other_impl in checks:
            aligned = self_impl == other_impl
            results[name] = {
                "aligned": aligned,
                "self": self_impl,
                "other": other_impl,
            }
            if not aligned:
                all_aligned = False

        results["all_aligned"] = all_aligned
        return results


# 便捷函数
def create_backtest_execution_model(
    exchange: str = "binance",
    vip_level: VIPLevel = VIPLevel.VIP0,
    taker_fee: float = 0.0004,
    maker_fee: float = 0.0002,
    slippage: float = 0.0001,
) -> ExecutionModel:
    """创建回测用执行模型"""
    config = ExecutionConfig(
        exchange=exchange,
        vip_level=vip_level,
        taker_fee=taker_fee,
        maker_fee=maker_fee,
        base_slippage=slippage,
        fill_price_type="close_price",  # 回测使用收盘价
    )
    return ExecutionModel(config)


def create_live_execution_model(
    exchange: str = "binance",
    vip_level: VIPLevel = VIPLevel.VIP0,
) -> ExecutionModel:
    """创建实盘用执行模型"""
    config = ExecutionConfig(
        exchange=exchange,
        vip_level=vip_level,
        fill_price_type="last_price",  # 实盘使用最新价
    )
    return ExecutionModel(config)
