"""
AlgVex 回测数据模型

定义回测中的核心数据结构:
- Position: 持仓
- Trade: 交易记录
- Signal: 交易信号
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"


class TradeType(Enum):
    """交易类型"""
    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    CLOSE_SHORT = "close_short"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TradeStatus(Enum):
    """交易状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    signal_type: str  # "long", "short", "close"
    strength: float   # 信号强度 [-1, 1]
    timestamp: datetime
    price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.signal_type == "long"

    @property
    def is_short(self) -> bool:
        return self.signal_type == "short"

    @property
    def is_close(self) -> bool:
        return self.signal_type == "close"


@dataclass
class Trade:
    """
    交易记录

    记录每一笔成交的详细信息，用于:
    - 交易日志
    - 绩效归因
    - 回测验证
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    trade_type: TradeType = TradeType.OPEN_LONG
    side: PositionSide = PositionSide.LONG
    quantity: float = 0.0           # 数量
    price: float = 0.0              # 成交价格
    value: float = 0.0              # 成交金额 (quantity * price)
    fee: float = 0.0                # 手续费
    slippage: float = 0.0           # 滑点成本
    timestamp: Optional[datetime] = None
    order_type: OrderType = OrderType.MARKET
    status: TradeStatus = TradeStatus.FILLED

    # 盈亏信息 (平仓时填写)
    pnl: float = 0.0                # 已实现盈亏
    pnl_percentage: float = 0.0     # 盈亏百分比
    funding_paid: float = 0.0       # 资金费支出

    # 关联信息
    signal_id: Optional[str] = None
    position_id: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.value == 0.0 and self.quantity > 0 and self.price > 0:
            self.value = self.quantity * self.price

    @property
    def is_open(self) -> bool:
        """是否为开仓交易"""
        return self.trade_type in [TradeType.OPEN_LONG, TradeType.OPEN_SHORT]

    @property
    def is_close(self) -> bool:
        """是否为平仓交易"""
        return self.trade_type in [TradeType.CLOSE_LONG, TradeType.CLOSE_SHORT]

    @property
    def total_cost(self) -> float:
        """总成本 (手续费 + 滑点)"""
        return self.fee + self.slippage

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "trade_type": self.trade_type.value,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "fee": self.fee,
            "slippage": self.slippage,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "pnl": self.pnl,
            "pnl_percentage": self.pnl_percentage,
            "funding_paid": self.funding_paid,
            "signal_id": self.signal_id,
            "position_id": self.position_id,
            "metadata": self.metadata,
        }


@dataclass
class Position:
    """
    持仓

    跟踪永续合约持仓状态，包括:
    - 持仓数量和方向
    - 入场价格和时间
    - 未实现盈亏
    - 资金费累计
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    side: PositionSide = PositionSide.LONG
    quantity: float = 0.0           # 持仓数量 (合约张数或币数)
    entry_price: float = 0.0        # 平均入场价格
    current_price: float = 0.0      # 当前价格 (mark_price)
    leverage: float = 1.0           # 杠杆倍数

    # 时间戳
    entry_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None

    # 资金
    margin: float = 0.0             # 保证金
    initial_margin: float = 0.0     # 初始保证金

    # 盈亏
    unrealized_pnl: float = 0.0     # 未实现盈亏
    realized_pnl: float = 0.0       # 已实现盈亏
    total_funding_paid: float = 0.0 # 累计资金费

    # 成本
    total_fees: float = 0.0         # 累计手续费
    total_slippage: float = 0.0     # 累计滑点

    # 状态
    is_open: bool = True
    liquidated: bool = False

    # 交易历史
    trades: List[Trade] = field(default_factory=list)

    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price

    @property
    def position_value(self) -> float:
        """持仓价值 (当前价格)"""
        return self.quantity * self.current_price

    @property
    def notional_value(self) -> float:
        """名义价值"""
        return self.position_value

    @property
    def entry_value(self) -> float:
        """入场价值"""
        return self.quantity * self.entry_price

    @property
    def margin_ratio(self) -> float:
        """保证金率"""
        if self.position_value == 0:
            return 1.0
        return self.margin / self.position_value

    @property
    def effective_leverage(self) -> float:
        """有效杠杆"""
        if self.margin == 0:
            return 0
        return self.position_value / self.margin

    def update_price(self, price: float, timestamp: Optional[datetime] = None):
        """更新价格并重新计算未实现盈亏"""
        self.current_price = price
        self.last_update_time = timestamp

        # 计算未实现盈亏
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

    def add_margin(self, amount: float):
        """增加保证金"""
        self.margin += amount

    def reduce_margin(self, amount: float) -> bool:
        """减少保证金"""
        if amount > self.margin:
            return False
        self.margin -= amount
        return True

    def add_funding(self, amount: float):
        """记录资金费"""
        self.total_funding_paid += amount
        # 资金费从保证金中扣除
        self.margin -= amount

    def check_liquidation(self, maintenance_margin_rate: float = 0.005) -> bool:
        """
        检查是否触发强平

        Args:
            maintenance_margin_rate: 维持保证金率

        Returns:
            是否应该被强平
        """
        if self.position_value == 0:
            return False

        maintenance_margin = self.position_value * maintenance_margin_rate

        # 计算可用保证金 (保证金 + 未实现盈亏)
        available_margin = self.margin + self.unrealized_pnl

        return available_margin < maintenance_margin

    def calculate_liquidation_price(self, maintenance_margin_rate: float = 0.005) -> float:
        """
        计算强平价格

        Args:
            maintenance_margin_rate: 维持保证金率

        Returns:
            强平价格
        """
        if self.quantity == 0:
            return 0.0

        # 强平条件: margin + unrealized_pnl < position_value * maintenance_margin_rate
        # 对于多头: margin + (price - entry_price) * quantity < price * quantity * maintenance_margin_rate
        # 对于空头: margin + (entry_price - price) * quantity < price * quantity * maintenance_margin_rate

        if self.side == PositionSide.LONG:
            # price * (1 - maintenance_margin_rate) * quantity - entry_price * quantity < -margin
            # price < (entry_price * quantity - margin) / ((1 - maintenance_margin_rate) * quantity)
            denominator = (1 - maintenance_margin_rate) * self.quantity
            if denominator <= 0:
                return 0.0
            liq_price = (self.entry_price * self.quantity - self.margin) / denominator
        else:
            # margin - (entry_price - price) * quantity < price * quantity * maintenance_margin_rate
            # price * (1 + maintenance_margin_rate) * quantity > margin + entry_price * quantity
            denominator = (1 + maintenance_margin_rate) * self.quantity
            if denominator <= 0:
                return float('inf')
            liq_price = (self.margin + self.entry_price * self.quantity) / denominator

        return max(0.0, liq_price)

    def close(
        self,
        close_price: float,
        close_time: datetime,
        fee: float = 0.0,
        slippage: float = 0.0,
    ) -> Trade:
        """
        平仓

        Args:
            close_price: 平仓价格
            close_time: 平仓时间
            fee: 手续费
            slippage: 滑点

        Returns:
            平仓交易记录
        """
        # 计算最终盈亏
        if self.side == PositionSide.LONG:
            pnl = (close_price - self.entry_price) * self.quantity
            trade_type = TradeType.CLOSE_LONG
        else:
            pnl = (self.entry_price - close_price) * self.quantity
            trade_type = TradeType.CLOSE_SHORT

        # 扣除所有成本
        net_pnl = pnl - self.total_fees - self.total_slippage - fee - slippage - self.total_funding_paid

        # 创建平仓交易
        trade = Trade(
            symbol=self.symbol,
            trade_type=trade_type,
            side=self.side,
            quantity=self.quantity,
            price=close_price,
            value=self.quantity * close_price,
            fee=fee,
            slippage=slippage,
            timestamp=close_time,
            pnl=net_pnl,
            pnl_percentage=net_pnl / self.initial_margin if self.initial_margin > 0 else 0,
            funding_paid=self.total_funding_paid,
            position_id=self.id,
        )

        # 更新持仓状态
        self.realized_pnl = net_pnl
        self.is_open = False
        self.quantity = 0
        self.trades.append(trade)

        return trade

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "leverage": self.leverage,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "margin": self.margin,
            "initial_margin": self.initial_margin,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_funding_paid": self.total_funding_paid,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "is_open": self.is_open,
            "liquidated": self.liquidated,
            "position_value": self.position_value,
            "margin_ratio": self.margin_ratio,
            "num_trades": len(self.trades),
        }


@dataclass
class Account:
    """
    账户状态

    跟踪账户余额、保证金使用、总敞口等
    """
    balance: float = 0.0            # 可用余额
    initial_balance: float = 0.0    # 初始余额
    used_margin: float = 0.0        # 已用保证金
    unrealized_pnl: float = 0.0     # 总未实现盈亏
    realized_pnl: float = 0.0       # 总已实现盈亏
    total_fees: float = 0.0         # 累计手续费
    total_funding: float = 0.0      # 累计资金费
    total_slippage: float = 0.0     # 累计滑点

    # 持仓
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)

    # 交易历史
    trades: List[Trade] = field(default_factory=list)

    # 统计
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0

    @property
    def equity(self) -> float:
        """总权益 = 余额 + 未实现盈亏"""
        return self.balance + self.unrealized_pnl

    @property
    def total_exposure(self) -> float:
        """总敞口"""
        return sum(pos.position_value for pos in self.positions.values())

    @property
    def free_margin(self) -> float:
        """可用保证金"""
        return self.balance - self.used_margin + self.unrealized_pnl

    @property
    def margin_level(self) -> float:
        """保证金水平"""
        if self.used_margin == 0:
            return float('inf')
        return self.equity / self.used_margin

    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.num_trades == 0:
            return 0.0
        return self.num_wins / self.num_trades

    @property
    def total_return(self) -> float:
        """总收益率"""
        if self.initial_balance == 0:
            return 0.0
        return (self.equity - self.initial_balance) / self.initial_balance

    def update_equity_stats(self):
        """更新权益统计"""
        current_equity = self.equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "equity": self.equity,
            "used_margin": self.used_margin,
            "free_margin": self.free_margin,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_slippage": self.total_slippage,
            "total_exposure": self.total_exposure,
            "margin_level": self.margin_level,
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "num_open_positions": len(self.positions),
        }
