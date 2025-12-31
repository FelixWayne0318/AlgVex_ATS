"""
AlgVex Shared - 共享模块

包含:
- 配置验证器
- 可见性检查器
- Trace日志
- 数据服务接口
- 时间提供者
- 随机数生成器
"""

from .config_validator import ConfigValidator, validate_config_hash
from .visibility_checker import VisibilityChecker, check_visibility
from .trace_logger import TraceLogger, create_trace
from .data_service import DataService
from .time_provider import TimeProvider, get_current_time
from .seeded_random import SeededRandom

__all__ = [
    "ConfigValidator",
    "validate_config_hash",
    "VisibilityChecker",
    "check_visibility",
    "TraceLogger",
    "create_trace",
    "DataService",
    "TimeProvider",
    "get_current_time",
    "SeededRandom",
]
