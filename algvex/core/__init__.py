"""
AlgVex Core - 量化引擎核心

融合 Microsoft Qlib 和 Hummingbot 的最佳特性:
- Qlib: 因子库、机器学习框架、回测基础设施
- Hummingbot: 企业级执行引擎、完善风控、多策略支持

使用方式:
    from core import AlgVex

    # 初始化引擎
    engine = AlgVex()

    # 采集数据
    engine.collect_data(symbols=['btcusdt'], days=90)

    # 训练模型
    engine.train_model()

    # 回测
    result = engine.backtest(start='2023-01-01', end='2024-01-01')

    # 模拟交易
    await engine.start_paper_trading()

v2.0 更新:
- 执行层从 ValueCell 升级为 Hummingbot
- 新增企业级风控系统
- 新增智能仓位管理器
"""

__version__ = "2.0.0"
__author__ = "AlgVex Team"
__email__ = "admin@algvex.com"

# 延迟导入 - engine 模块可能不存在（Phase 2）
try:
    from .engine import AlgVexEngine as AlgVex
    from .engine import AlgVexConfig
except ImportError:
    AlgVex = None
    AlgVexConfig = None

# 延迟导入子模块
def __getattr__(name):
    if name == "DataCollector":
        from .data.collector import DataCollector
        return DataCollector
    elif name == "DataHandler":
        from .data.handler import DataHandler
        return DataHandler
    elif name == "FactorEngine":
        from .factor.engine import FactorEngine
        return FactorEngine
    elif name == "ModelTrainer":
        from .model.trainer import ModelTrainer
        return ModelTrainer
    elif name == "BacktestEngine":
        from .backtest.engine import BacktestEngine
        return BacktestEngine
    # Hummingbot 执行层
    elif name == "HummingbotBridge":
        from .execution.hummingbot_bridge import HummingbotBridge
        return HummingbotBridge
    elif name == "RiskManager":
        from .execution.risk_manager import RiskManager
        return RiskManager
    elif name == "PositionManager":
        from .execution.position_manager import PositionManager
        return PositionManager
    elif name == "SignalGenerator":
        from .strategy.signal import SignalGenerator
        return SignalGenerator
    # 评估模块 (Qlib 风格)
    elif name == "risk_analysis":
        from .evaluate import risk_analysis
        return risk_analysis
    elif name == "calc_ic":
        from .evaluate import calc_ic
        return calc_ic
    elif name == "calc_long_short_return":
        from .evaluate import calc_long_short_return
        return calc_long_short_return
    elif name == "calc_long_short_prec":
        from .evaluate import calc_long_short_prec
        return calc_long_short_prec
    elif name == "generate_report":
        from .evaluate import generate_report
        return generate_report
    # 强化学习模块 (Qlib 风格)
    elif name == "TradingEnv":
        from .rl.env import TradingEnv
        return TradingEnv
    elif name == "CryptoTradingEnv":
        from .rl.env import CryptoTradingEnv
        return CryptoTradingEnv
    elif name == "PPOPolicy":
        from .rl.policy import PPOPolicy
        return PPOPolicy
    elif name == "DQNPolicy":
        from .rl.policy import DQNPolicy
        return DQNPolicy
    elif name == "Trainer":
        from .rl.trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AlgVex",
    "AlgVexConfig",
    "DataCollector",
    "DataHandler",
    "FactorEngine",
    "ModelTrainer",
    "BacktestEngine",
    # Hummingbot 执行层
    "HummingbotBridge",
    "RiskManager",
    "PositionManager",
    "SignalGenerator",
    # 评估函数 (Qlib 风格)
    "risk_analysis",
    "calc_ic",
    "calc_long_short_return",
    "calc_long_short_prec",
    "generate_report",
    # 强化学习模块 (Qlib 风格)
    "TradingEnv",
    "CryptoTradingEnv",
    "PPOPolicy",
    "DQNPolicy",
    "Trainer",
]
