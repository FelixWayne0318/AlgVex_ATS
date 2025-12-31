"""
AlgVex 回测配置

定义回测引擎的配置参数，包括:
- 资金配置 (初始资金、杠杆)
- 费率配置 (maker/taker费率、滑点)
- 风控配置 (强平阈值、最大持仓)
- 执行配置 (成交模型)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from shared.execution_models import VIPLevel


class PositionMode(Enum):
    """持仓模式"""
    ONE_WAY = "one_way"      # 单向持仓
    HEDGE = "hedge"          # 对冲模式 (双向持仓)


class MarginMode(Enum):
    """保证金模式"""
    CROSS = "cross"          # 全仓
    ISOLATED = "isolated"    # 逐仓


@dataclass
class ExecutionConfig:
    """执行配置"""
    # 成交价格类型
    fill_price: str = "close_price"  # close_price, mark_price, last_price

    # 部分成交
    partial_fill: bool = True

    # 滑点配置
    slippage_model: str = "dynamic"  # static, dynamic
    base_slippage: float = 0.0001    # 0.01%
    max_slippage: float = 0.01       # 1%

    # 费率配置
    exchange: str = "binance"
    vip_level: VIPLevel = VIPLevel.VIP0
    maker_fee: Optional[float] = None  # 自定义费率覆盖
    taker_fee: Optional[float] = None

    # 仓位模式
    position_mode: PositionMode = PositionMode.ONE_WAY
    reduce_only: bool = True

    # 触发单逻辑
    stop_loss_trigger: str = "mark_price"
    take_profit_trigger: str = "last_price"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "fill_price": self.fill_price,
            "partial_fill": self.partial_fill,
            "slippage_model": self.slippage_model,
            "base_slippage": self.base_slippage,
            "max_slippage": self.max_slippage,
            "exchange": self.exchange,
            "vip_level": self.vip_level.value,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "position_mode": self.position_mode.value,
            "reduce_only": self.reduce_only,
            "stop_loss_trigger": self.stop_loss_trigger,
            "take_profit_trigger": self.take_profit_trigger,
        }


@dataclass
class BacktestConfig:
    """
    回测配置

    设计文档参考: Section 5.1 CryptoPerpetualBacktest

    使用示例:
        config = BacktestConfig(
            initial_capital=100000.0,
            leverage=3.0,
            taker_fee=0.0004,
            maker_fee=0.0002,
        )
    """

    # === 资金配置 ===
    initial_capital: float = 100000.0     # 初始资金 (USDT)
    leverage: float = 3.0                  # 默认杠杆
    max_leverage: float = 10.0             # 最大杠杆

    # === 费率配置 ===
    taker_fee: float = 0.0004              # Taker费率 0.04%
    maker_fee: float = 0.0002              # Maker费率 0.02%
    slippage: float = 0.0001               # 滑点 0.01%

    # === 资金费率配置 ===
    funding_rate_interval: int = 8         # 资金费率间隔 (小时)
    enable_funding: bool = True            # 是否启用资金费率

    # === 风控配置 ===
    liquidation_threshold: float = 0.8     # 爆仓阈值 (维持保证金率)
    maintenance_margin_rate: float = 0.005 # 维持保证金率 0.5%
    max_position_size: Optional[float] = None  # 最大单笔持仓 (USDT)
    max_total_exposure: Optional[float] = None # 最大总敞口

    # === 执行配置 ===
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)

    # === 保证金模式 ===
    margin_mode: MarginMode = MarginMode.CROSS

    # === 交易限制 ===
    symbols: List[str] = field(default_factory=list)  # 交易标的列表
    min_order_value: float = 10.0          # 最小订单价值 (USDT)

    # === 时间配置 ===
    frequency: str = "5m"                  # K线频率
    timezone: str = "UTC"

    # === 调试配置 ===
    verbose: bool = False
    trace_enabled: bool = False

    def __post_init__(self):
        """验证配置"""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.leverage <= 0 or self.leverage > self.max_leverage:
            raise ValueError(f"leverage must be between 0 and {self.max_leverage}")
        if self.taker_fee < 0 or self.maker_fee < 0:
            raise ValueError("fees must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")
        if self.liquidation_threshold <= 0 or self.liquidation_threshold > 1:
            raise ValueError("liquidation_threshold must be between 0 and 1")

        # 如果提供了自定义费率，更新 execution_config
        if self.execution_config.maker_fee is None:
            self.execution_config.maker_fee = self.maker_fee
        if self.execution_config.taker_fee is None:
            self.execution_config.taker_fee = self.taker_fee

    def get_effective_leverage(self, requested_leverage: Optional[float] = None) -> float:
        """获取有效杠杆 (考虑最大限制)"""
        leverage = requested_leverage or self.leverage
        return min(leverage, self.max_leverage)

    def calculate_required_margin(self, position_value: float) -> float:
        """计算所需保证金"""
        return position_value / self.leverage

    def calculate_maintenance_margin(self, position_value: float) -> float:
        """计算维持保证金"""
        return position_value * self.maintenance_margin_rate

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "initial_capital": self.initial_capital,
            "leverage": self.leverage,
            "max_leverage": self.max_leverage,
            "taker_fee": self.taker_fee,
            "maker_fee": self.maker_fee,
            "slippage": self.slippage,
            "funding_rate_interval": self.funding_rate_interval,
            "enable_funding": self.enable_funding,
            "liquidation_threshold": self.liquidation_threshold,
            "maintenance_margin_rate": self.maintenance_margin_rate,
            "max_position_size": self.max_position_size,
            "max_total_exposure": self.max_total_exposure,
            "margin_mode": self.margin_mode.value,
            "min_order_value": self.min_order_value,
            "frequency": self.frequency,
            "timezone": self.timezone,
            "execution_config": self.execution_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """从字典创建配置"""
        execution_data = data.pop("execution_config", {})

        # 处理枚举类型
        if "margin_mode" in data and isinstance(data["margin_mode"], str):
            data["margin_mode"] = MarginMode(data["margin_mode"])

        if "vip_level" in execution_data and isinstance(execution_data["vip_level"], str):
            execution_data["vip_level"] = VIPLevel(execution_data["vip_level"])
        if "position_mode" in execution_data and isinstance(execution_data["position_mode"], str):
            execution_data["position_mode"] = PositionMode(execution_data["position_mode"])

        execution_config = ExecutionConfig(**execution_data) if execution_data else ExecutionConfig()

        return cls(execution_config=execution_config, **data)

    def create_mvp_config(self) -> "BacktestConfig":
        """创建 MVP 配置 (最小可行配置)"""
        return BacktestConfig(
            initial_capital=self.initial_capital,
            leverage=3.0,
            max_leverage=10.0,
            taker_fee=0.0004,
            maker_fee=0.0002,
            slippage=0.0001,
            enable_funding=True,
            frequency="5m",
        )
