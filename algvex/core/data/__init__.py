"""
AlgVex 数据层

数据流:
1. MultiSourceDataManager: 统一管理多数据源
2. DataCollector: 采集币安永续合约数据
3. DataValidator: 数据质量检查
4. DataHandler: 转换为 Qlib 格式
5. RealtimeStream: WebSocket 实时推送

支持的数据源:
- 币安永续合约 (K线、资金费率、持仓量、多空比)
- Deribit (期权 IV、Greeks、DVOL)
- DeFiLlama (TVL、稳定币、桥接)
- 链上数据 (Glassnode、CryptoQuant)
- 情绪数据 (恐慌贪婪指数)
- 宏观数据 (DXY、利率、标普500)
"""

__all__ = []

# 核心数据采集器 (仅需 requests, pandas)
try:
    from .collector import BinanceDataCollector
    __all__.append("BinanceDataCollector")
except ImportError:
    BinanceDataCollector = None

try:
    from .deribit_collector import DeribitDataCollector
    __all__.append("DeribitDataCollector")
except ImportError:
    DeribitDataCollector = None

try:
    from .defillama_collector import DeFiLlamaCollector
    __all__.append("DeFiLlamaCollector")
except ImportError:
    DeFiLlamaCollector = None

# 数据处理 (可能需要 qlib)
try:
    from .handler import CryptoDataHandler
    __all__.append("CryptoDataHandler")
except ImportError:
    CryptoDataHandler = None

# 数据验证
try:
    from .validator import DataValidator
    __all__.append("DataValidator")
except ImportError:
    DataValidator = None

# 实时流 (需要 websockets)
try:
    from .realtime import RealtimeStream
    __all__.append("RealtimeStream")
except ImportError:
    RealtimeStream = None

# 多源管理器
try:
    from .multi_source_manager import (
        MultiSourceDataManager,
        BinanceDataSource,
        OnChainDataSource,
        SentimentDataSource,
        MacroDataSource,
    )
    __all__.extend([
        "MultiSourceDataManager",
        "BinanceDataSource",
        "OnChainDataSource",
        "SentimentDataSource",
        "MacroDataSource",
    ])
except ImportError:
    MultiSourceDataManager = None
    BinanceDataSource = None
    OnChainDataSource = None
    SentimentDataSource = None
    MacroDataSource = None

# Fear & Greed Index
try:
    from .fear_greed_collector import FearGreedCollector
    __all__.append("FearGreedCollector")
except ImportError:
    FearGreedCollector = None

# Yahoo Finance (宏观数据)
try:
    from .yahoo_collector import YahooMacroCollector
    __all__.append("YahooMacroCollector")
except ImportError:
    YahooMacroCollector = None

# Google Trends
try:
    from .google_trends_collector import GoogleTrendsCollector
    __all__.append("GoogleTrendsCollector")
except ImportError:
    GoogleTrendsCollector = None

# Bybit (多交易所Basis - Step 11)
try:
    from .bybit_collector import BybitDataCollector
    __all__.append("BybitDataCollector")
except ImportError:
    BybitDataCollector = None

# OKX (多交易所Basis - Step 11)
try:
    from .okx_collector import OKXDataCollector
    __all__.append("OKXDataCollector")
except ImportError:
    OKXDataCollector = None

# CryptoDataset (Qlib DatasetH 风格)
try:
    from .dataset import CryptoDataset, DatasetConfig, create_dataset_from_config
    __all__.extend(["CryptoDataset", "DatasetConfig", "create_dataset_from_config"])
except ImportError:
    CryptoDataset = None
    DatasetConfig = None
    create_dataset_from_config = None

# Reweighter (Qlib 风格样本权重)
try:
    from .reweighter import (
        Reweighter,
        TimeDecayReweighter,
        VolatilityReweighter,
        CombinedReweighter,
        CustomReweighter,
        get_default_reweighter,
        get_crypto_reweighter,
    )
    __all__.extend([
        "Reweighter",
        "TimeDecayReweighter",
        "VolatilityReweighter",
        "CombinedReweighter",
        "CustomReweighter",
        "get_default_reweighter",
        "get_crypto_reweighter",
    ])
except ImportError:
    Reweighter = None
    TimeDecayReweighter = None
    VolatilityReweighter = None
    CombinedReweighter = None
    CustomReweighter = None
    get_default_reweighter = None
    get_crypto_reweighter = None

# 缓存模块 (Qlib 风格)
try:
    from .cache import (
        MemCache,
        MemCacheExpire,
        DiskCache,
        RedisCache,
        MultiLevelCache,
        DatasetCache,
        get_mem_cache,
        clear_mem_cache,
        REDIS_AVAILABLE,
    )
    __all__.extend([
        "MemCache",
        "MemCacheExpire",
        "DiskCache",
        "RedisCache",
        "MultiLevelCache",
        "DatasetCache",
        "get_mem_cache",
        "clear_mem_cache",
        "REDIS_AVAILABLE",
    ])
except ImportError:
    MemCache = None
    MemCacheExpire = None
    DiskCache = None
    RedisCache = None
    MultiLevelCache = None
    DatasetCache = None
    get_mem_cache = None
    clear_mem_cache = None
    REDIS_AVAILABLE = False
