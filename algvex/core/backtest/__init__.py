"""
AlgVex 回测引擎模块

核心组件:
- BacktestConfig: 回测配置
- CryptoPerpetualBacktest: 永续合约回测引擎
- BacktestResult: 回测结果
- BacktestMetrics: 回测指标计算
- ExecutionModel: 统一成交模型 (回测-实盘对齐)
- TripleBarrier: 三重屏障风控

使用示例:
    from algvex.core.backtest import BacktestConfig, CryptoPerpetualBacktest

    config = BacktestConfig(
        initial_capital=100000.0,
        leverage=3.0,
        taker_fee=0.0004,
        maker_fee=0.0002,
    )

    engine = CryptoPerpetualBacktest(config)
    results = engine.run(signals, prices, funding_rates)

    # 使用三重屏障
    from algvex.core.backtest import TripleBarrierConfig, TripleBarrier
    barrier_config = TripleBarrierConfig(stop_loss=0.03, take_profit=0.06)
    barrier = TripleBarrier(barrier_config)
"""

from .config import BacktestConfig, ExecutionConfig as BacktestExecutionConfig
from .models import Position, Trade, TradeType, PositionSide
from .metrics import BacktestMetrics, BacktestResult
from .engine import CryptoPerpetualBacktest, BacktestEngine
from .execution_model import (
    ExecutionModel,
    ExecutionConfig,
    FillResult,
    OrderSide,
    OrderType,
    create_backtest_execution_model,
    create_live_execution_model,
)
from .triple_barrier import (
    TripleBarrier,
    TripleBarrierConfig,
    BarrierType,
    BarrierCheckResult,
    PositionState,
    TriggerPrice,
    create_conservative_config,
    create_aggressive_config,
    create_scalping_config,
)

__all__ = [
    # Config
    "BacktestConfig",
    "BacktestExecutionConfig",
    "ExecutionConfig",
    # Models
    "Position",
    "Trade",
    "TradeType",
    "PositionSide",
    # Metrics
    "BacktestMetrics",
    "BacktestResult",
    # Engine
    "CryptoPerpetualBacktest",
    "BacktestEngine",  # Backward compatibility alias
    # Execution Model
    "ExecutionModel",
    "FillResult",
    "OrderSide",
    "OrderType",
    "create_backtest_execution_model",
    "create_live_execution_model",
    # Triple Barrier
    "TripleBarrier",
    "TripleBarrierConfig",
    "BarrierType",
    "BarrierCheckResult",
    "PositionState",
    "TriggerPrice",
    "create_conservative_config",
    "create_aggressive_config",
    "create_scalping_config",
]
