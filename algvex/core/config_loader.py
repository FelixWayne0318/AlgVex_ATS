"""
AlgVex 配置加载器

功能:
- 加载 YAML 配置文件
- 支持环境变量替换
- 支持路径展开
- 提供类型安全的访问

用法:
    from algvex.core.config_loader import load_config, TutorialConfig

    # 加载配置
    config = load_config('tutorial_config.yaml')

    # 访问配置
    print(config.trading.symbols)
    print(config.capital.initial_capital)
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


def _expand_path(path: str) -> str:
    """展开路径中的 ~ 和环境变量"""
    if path is None:
        return None
    return os.path.expandvars(os.path.expanduser(path))


def _parse_date(date_str: str) -> datetime:
    """解析日期字符串"""
    if isinstance(date_str, datetime):
        return date_str
    return datetime.strptime(date_str, "%Y-%m-%d")


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class QlibInitConfig:
    """Qlib 初始化配置"""
    provider_uri: str = "~/.algvex/qlib_data"
    region: str = "crypto"

    def __post_init__(self):
        self.provider_uri = _expand_path(self.provider_uri)


@dataclass
class ProjectConfig:
    """项目配置"""
    name: str = "AlgVex Tutorial"
    version: str = "3.12"
    data_dir: str = "./data/crypto"

    def __post_init__(self):
        self.data_dir = _expand_path(self.data_dir)


@dataclass
class TradingConfig:
    """交易配置"""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    extended_symbols: List[str] = field(default_factory=list)
    timeframe: str = "1h"
    quote_currency: str = "USDT"

    @property
    def all_symbols(self) -> List[str]:
        """获取所有交易对"""
        return self.symbols + self.extended_symbols


@dataclass
class TimeRangeConfig:
    """时间范围配置"""
    data_start: str = "2024-01-01"
    data_end: str = "2024-12-23"
    train_start: str = "2024-01-01"
    train_end: str = "2024-08-31"
    valid_start: str = "2024-09-01"
    valid_end: str = "2024-10-31"
    test_start: str = "2024-11-01"
    test_end: str = "2024-12-23"

    @property
    def data_start_dt(self) -> datetime:
        return _parse_date(self.data_start)

    @property
    def data_end_dt(self) -> datetime:
        return _parse_date(self.data_end)

    @property
    def train_range(self) -> tuple:
        return (_parse_date(self.train_start), _parse_date(self.train_end))

    @property
    def valid_range(self) -> tuple:
        return (_parse_date(self.valid_start), _parse_date(self.valid_end))

    @property
    def test_range(self) -> tuple:
        return (_parse_date(self.test_start), _parse_date(self.test_end))


@dataclass
class CapitalConfig:
    """资金配置"""
    initial_capital: float = 10000.0
    quote_currency: str = "USDT"
    leverage: float = 3.0
    max_leverage: float = 10.0
    position_size: float = 0.1
    max_positions: int = 5

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.leverage < 1:
            raise ValueError("leverage must be >= 1")
        if self.leverage > self.max_leverage:
            raise ValueError("leverage cannot exceed max_leverage")


@dataclass
class FeesConfig:
    """费率配置"""
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    slippage: float = 0.0001


@dataclass
class RiskConfig:
    """风控配置"""
    liquidation_threshold: float = 0.8
    maintenance_margin_rate: float = 0.005
    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_drawdown: float = 0.20


@dataclass
class DataCollectorConfig:
    """数据采集配置"""
    class_name: str = "BinanceDataCollector"
    module_path: str = "algvex.core.data.collector"
    data_types: List[str] = field(default_factory=lambda: [
        "klines", "funding_rate", "open_interest",
        "long_short_ratio", "taker_buy_sell"
    ])
    max_collector_count: int = 3
    retry_delay: float = 2.0
    max_workers: int = 4
    check_data_length: int = 100
    rate_limit_delay: float = 0.1


@dataclass
class FactorConfig:
    """单个因子配置"""
    name: str
    class_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessorConfig:
    """处理器配置"""
    class_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorHandlerConfig:
    """因子处理器配置"""
    class_name: str = "CryptoFactorHandler"
    module_path: str = "algvex.core.factor.handler"
    factors: List[FactorConfig] = field(default_factory=list)
    learn_processors: List[ProcessorConfig] = field(default_factory=list)
    infer_processors: List[ProcessorConfig] = field(default_factory=list)


@dataclass
class ModelConfig:
    """模型配置"""
    class_name: str = "LGBModel"
    module_path: str = "qlib.contrib.model.gbdt"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """回测配置"""
    class_name: str = "CryptoBacktestEngine"
    module_path: str = "algvex.core.backtest.engine"
    fill_price: str = "close_price"
    slippage_model: str = "static"
    position_mode: str = "one_way"
    enable_funding: bool = True
    funding_interval: str = "8h"
    record_trades: bool = True
    record_positions: bool = True
    record_equity_curve: bool = True


@dataclass
class ExperimentConfig:
    """实验追踪配置"""
    tracking_uri: str = "./mlruns"
    experiment_name: str = "algvex_crypto_tutorial"
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    artifacts: List[str] = field(default_factory=lambda: [
        "equity_curve", "trade_history", "factor_importance", "model_checkpoint"
    ])

    def __post_init__(self):
        self.tracking_uri = _expand_path(self.tracking_uri)


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(message)s"
    file: str = "./logs/tutorial.log"
    rotation: str = "10 MB"

    def __post_init__(self):
        self.file = _expand_path(self.file)


@dataclass
class TutorialConfig:
    """教程总配置"""
    qlib_init: QlibInitConfig = field(default_factory=QlibInitConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    time_range: TimeRangeConfig = field(default_factory=TimeRangeConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    fees: FeesConfig = field(default_factory=FeesConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data_collector: DataCollectorConfig = field(default_factory=DataCollectorConfig)
    factor_handler: FactorHandlerConfig = field(default_factory=FactorHandlerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str):
        """保存配置到 YAML 文件"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


# ============================================================
# 配置加载函数
# ============================================================

def _dict_to_dataclass(data: Dict, cls):
    """将字典转换为数据类"""
    if data is None:
        return cls()

    # 获取数据类的字段
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for key, value in data.items():
        # 处理 class 关键字冲突
        if key == 'class':
            key = 'class_name'

        if key in field_types:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: str) -> TutorialConfig:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        TutorialConfig 对象
    """
    config_path = Path(_expand_path(config_path))

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    # 解析各个部分
    config = TutorialConfig(
        qlib_init=_dict_to_dataclass(raw_config.get('qlib_init'), QlibInitConfig),
        project=_dict_to_dataclass(raw_config.get('project'), ProjectConfig),
        trading=_dict_to_dataclass(raw_config.get('trading'), TradingConfig),
        time_range=_dict_to_dataclass(raw_config.get('time_range'), TimeRangeConfig),
        capital=_dict_to_dataclass(raw_config.get('capital'), CapitalConfig),
        fees=_dict_to_dataclass(raw_config.get('fees'), FeesConfig),
        risk=_dict_to_dataclass(raw_config.get('risk'), RiskConfig),
        data_collector=_parse_data_collector_config(raw_config.get('data_collector')),
        factor_handler=_parse_factor_handler_config(raw_config.get('factor_handler')),
        model=_parse_model_config(raw_config.get('model')),
        backtest=_parse_backtest_config(raw_config.get('backtest')),
        experiment=_dict_to_dataclass(raw_config.get('experiment'), ExperimentConfig),
        logging=_dict_to_dataclass(raw_config.get('logging'), LoggingConfig),
    )

    return config


def _parse_data_collector_config(data: Dict) -> DataCollectorConfig:
    """解析数据采集配置"""
    if data is None:
        return DataCollectorConfig()

    kwargs = data.get('kwargs', {})
    return DataCollectorConfig(
        class_name=data.get('class', 'BinanceDataCollector'),
        module_path=data.get('module_path', 'algvex.core.data.collector'),
        data_types=kwargs.get('data_types', []),
        max_collector_count=kwargs.get('max_collector_count', 3),
        retry_delay=kwargs.get('retry_delay', 2.0),
        max_workers=kwargs.get('max_workers', 4),
        check_data_length=kwargs.get('check_data_length', 100),
        rate_limit_delay=kwargs.get('rate_limit_delay', 0.1),
    )


def _parse_factor_handler_config(data: Dict) -> FactorHandlerConfig:
    """解析因子处理器配置"""
    if data is None:
        return FactorHandlerConfig()

    kwargs = data.get('kwargs', {})

    # 解析因子列表
    factors = []
    for f in kwargs.get('factors', []):
        factors.append(FactorConfig(
            name=f.get('name', ''),
            class_name=f.get('class', ''),
            kwargs=f.get('kwargs', {})
        ))

    # 解析处理器
    learn_processors = []
    for p in kwargs.get('learn_processors', []):
        learn_processors.append(ProcessorConfig(
            class_name=p.get('class', ''),
            kwargs=p.get('kwargs', {})
        ))

    infer_processors = []
    for p in kwargs.get('infer_processors', []):
        infer_processors.append(ProcessorConfig(
            class_name=p.get('class', ''),
            kwargs=p.get('kwargs', {})
        ))

    return FactorHandlerConfig(
        class_name=data.get('class', 'CryptoFactorHandler'),
        module_path=data.get('module_path', 'algvex.core.factor.handler'),
        factors=factors,
        learn_processors=learn_processors,
        infer_processors=infer_processors,
    )


def _parse_model_config(data: Dict) -> ModelConfig:
    """解析模型配置"""
    if data is None:
        return ModelConfig()

    return ModelConfig(
        class_name=data.get('class', 'LGBModel'),
        module_path=data.get('module_path', 'qlib.contrib.model.gbdt'),
        kwargs=data.get('kwargs', {}),
    )


def _parse_backtest_config(data: Dict) -> BacktestConfig:
    """解析回测配置"""
    if data is None:
        return BacktestConfig()

    kwargs = data.get('kwargs', {})
    return BacktestConfig(
        class_name=data.get('class', 'CryptoBacktestEngine'),
        module_path=data.get('module_path', 'algvex.core.backtest.engine'),
        fill_price=kwargs.get('fill_price', 'close_price'),
        slippage_model=kwargs.get('slippage_model', 'static'),
        position_mode=kwargs.get('position_mode', 'one_way'),
        enable_funding=kwargs.get('enable_funding', True),
        funding_interval=kwargs.get('funding_interval', '8h'),
        record_trades=kwargs.get('record_trades', True),
        record_positions=kwargs.get('record_positions', True),
        record_equity_curve=kwargs.get('record_equity_curve', True),
    )


# ============================================================
# 快捷函数
# ============================================================

def get_default_config() -> TutorialConfig:
    """获取默认配置"""
    return TutorialConfig()


def create_config_template(output_path: str = "tutorial_config.yaml"):
    """创建配置模板文件"""
    import shutil
    template_path = Path(__file__).parent.parent / "config" / "tutorial_config.yaml"
    if template_path.exists():
        shutil.copy(template_path, output_path)
        print(f"Config template created: {output_path}")
    else:
        # 使用默认配置
        config = get_default_config()
        config.save(output_path)
        print(f"Default config created: {output_path}")
