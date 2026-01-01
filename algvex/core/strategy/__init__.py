"""
AlgVex 策略层

提供交易策略生成和管理:
- SignalGenerator: 信号生成器
- PositionSizer: 仓位管理
- RiskManager: 风险管理

使用示例:
    from algvex.core.strategy import SignalGenerator, SignalConfig

    config = SignalConfig(
        threshold=0.02,
        holding_period=288,  # 1天
    )

    generator = SignalGenerator(config)
    signals = generator.generate(predictions, prices)
"""

from .signal_generator import (
    SignalGenerator,
    SignalConfig,
    Signal,
    SignalType,
)

from .position_sizer import (
    PositionSizer,
    PositionSizeConfig,
    PositionSize,
    SizingMethod,
    create_conservative_sizer,
    create_moderate_sizer,
    create_aggressive_sizer,
)

from .risk_manager import (
    RiskManager,
    RiskConfig,
    RiskMetrics,
    RiskLevel,
    RiskAction,
    create_conservative_risk_manager,
    create_moderate_risk_manager,
    create_aggressive_risk_manager,
)

# 投资组合优化 (Qlib 风格)
from .portfolio_optimizer import (
    OptMethod,
    PortfolioOptimizer,
    RiskModel,
    TopkDropoutStrategy,
    optimize_portfolio,
    calculate_portfolio_metrics,
)

__all__ = [
    # Signal Generator
    "SignalGenerator",
    "SignalConfig",
    "Signal",
    "SignalType",
    # Position Sizer
    "PositionSizer",
    "PositionSizeConfig",
    "PositionSize",
    "SizingMethod",
    "create_conservative_sizer",
    "create_moderate_sizer",
    "create_aggressive_sizer",
    # Risk Manager
    "RiskManager",
    "RiskConfig",
    "RiskMetrics",
    "RiskLevel",
    "RiskAction",
    "create_conservative_risk_manager",
    "create_moderate_risk_manager",
    "create_aggressive_risk_manager",
    # 投资组合优化
    "OptMethod",
    "PortfolioOptimizer",
    "RiskModel",
    "TopkDropoutStrategy",
    "optimize_portfolio",
    "calculate_portfolio_metrics",
]
