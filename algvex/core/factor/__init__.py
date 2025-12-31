"""
AlgVex 因子层

核心因子库，包含 180 个研究因子 + 21 个 P1 扩展因子。

因子分类:
- 基础价量因子 (50个): 动量、波动率、成交量
- 永续合约专用因子 (45个): 资金费率、持仓量、订单流
- 期权/波动率因子 (20个): 隐含波动率、期权持仓
- 衍生品结构因子 (15个): 基差、市场结构
- 链上因子 (10个): 稳定币、DeFi TVL
- 情绪因子 (10个): Fear&Greed、Google Trends
- 宏观关联因子 (15个): 美元/利率、风险资产
- 复合/ML因子 (15个)

使用示例:
    from algvex.core.factor import FactorRegistry

    # 获取因子注册表
    registry = FactorRegistry.get_instance()

    # 获取单个因子
    factor = registry.get_factor("return_1h")

    # 计算因子值
    result = factor.compute(data, signal_time)

    # 获取MVP因子
    mvp_factors = registry.get_mvp_factors()

    # 按因子族获取
    momentum_factors = registry.get_factors_by_family(FactorFamily.MOMENTUM)
"""

# 基础类
from .base import (
    BaseFactor,
    FactorFamily,
    FactorMetadata,
    FactorResult,
    DataDependency,
    HistoryTier,
    # 工具函数
    ema,
    sma,
    rolling_std,
    rolling_max,
    rolling_min,
    zscore,
    rank,
    returns,
    log_returns,
)

# 因子模块
from .momentum import (
    get_momentum_factors,
    MOMENTUM_FACTORS,
    Return1H,
    Return4H,
    Return24H,
    Return7D,
    TrendStrength,
    Breakout20D,
)

from .volatility import (
    get_volatility_factors,
    VOLATILITY_FACTORS,
    Volatility24H,
    ATR24H,
    RealizedVol1D,
    VolRegime,
)

from .perpetual import (
    get_perpetual_factors,
    PERPETUAL_FACTORS,
    FundingRate,
    FundingMomentum,
    OIChangeRate,
    OIFundingDivergence,
    LongShortRatio,
    CVD,
)

# 注册表
from .registry import (
    FactorRegistry,
    FACTOR_CATALOG,
    get_factor_catalog_summary,
)

# Alpha158 因子 (Qlib 完整实现)
from .alpha158 import (
    Alpha158Calculator,
    Alpha158Config,
    Operators,
    get_alpha158_calculator,
    compute_alpha158,
)

# 处理器 (Qlib 风格)
from .processor import (
    BaseProcessor,
    ProcessorChain,
    DropnaProcessor,
    DropnaLabel,
    FillnaProcessor,
    ZScoreNormalizer,
    MinMaxNormalizer,
    RobustScaler,
    WinsorizeProcessor,
    LagProcessor,
    DiffProcessor,
    RollingProcessor,
    # Qlib 跨截面处理器
    CSZScoreNorm,
    CSRankNorm,
    CSFillna,
    TanhProcess,
    ProcessInf,
    FilterCol,
    DropCol,
    # 预定义处理器链
    get_default_learn_processors,
    get_default_infer_processors,
    get_crypto_processors,
)


__all__ = [
    # 基础类
    "BaseFactor",
    "FactorFamily",
    "FactorMetadata",
    "FactorResult",
    "DataDependency",
    "HistoryTier",
    # 工具函数
    "ema",
    "sma",
    "rolling_std",
    "rolling_max",
    "rolling_min",
    "zscore",
    "rank",
    "returns",
    "log_returns",
    # 因子获取
    "get_momentum_factors",
    "get_volatility_factors",
    "get_perpetual_factors",
    # 常用因子
    "Return1H",
    "Return4H",
    "Return24H",
    "Return7D",
    "TrendStrength",
    "Breakout20D",
    "Volatility24H",
    "ATR24H",
    "RealizedVol1D",
    "VolRegime",
    "FundingRate",
    "FundingMomentum",
    "OIChangeRate",
    "OIFundingDivergence",
    "LongShortRatio",
    "CVD",
    # 注册表
    "FactorRegistry",
    "FACTOR_CATALOG",
    "get_factor_catalog_summary",
    # Alpha158 因子
    "Alpha158Calculator",
    "Alpha158Config",
    "Operators",
    "get_alpha158_calculator",
    "compute_alpha158",
    # 处理器
    "BaseProcessor",
    "ProcessorChain",
    "DropnaProcessor",
    "DropnaLabel",
    "FillnaProcessor",
    "ZScoreNormalizer",
    "MinMaxNormalizer",
    "RobustScaler",
    "WinsorizeProcessor",
    "LagProcessor",
    "DiffProcessor",
    "RollingProcessor",
    # Qlib 跨截面处理器
    "CSZScoreNorm",
    "CSRankNorm",
    "CSFillna",
    "TanhProcess",
    "ProcessInf",
    "FilterCol",
    "DropCol",
    # 预定义处理器链
    "get_default_learn_processors",
    "get_default_infer_processors",
    "get_crypto_processors",
]


def get_registry() -> FactorRegistry:
    """便捷函数获取因子注册表"""
    return FactorRegistry.get_instance()


def get_all_factors():
    """获取所有已注册因子"""
    return get_registry().get_all_factors()


def get_factor(factor_id: str):
    """获取单个因子"""
    return get_registry().get_factor(factor_id)
