"""
MVP Scope 强制检查器

关键检查点:
1. 启动时: 检查 symbols/factors/data_sources 是否在 MVP 范围内
2. 信号生成时: 检查使用的因子是否在 MVP-11 白名单
3. 数据请求时: 检查数据源是否在 MVP 允许列表

违规处理:
- reject: 抛出异常，拒绝执行 (生产环境默认)
- warn: 记录警告但继续执行 (研究环境)
- log: 仅记录日志

配置文件: config/mvp_scope.yaml
"""

import fnmatch
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


logger = logging.getLogger(__name__)


class ViolationAction(Enum):
    """违规处理方式"""
    REJECT = "reject"
    WARN = "warn"
    LOG = "log"


@dataclass
class MvpViolation:
    """MVP违规记录"""
    category: str  # symbol / factor / data_source / frequency
    value: str
    message: str
    timestamp: Optional[str] = None


class MvpScopeViolationError(Exception):
    """MVP范围违规异常 - 生产环境必须拒绝"""
    pass


class MvpScopeEnforcer:
    """
    MVP Scope 强制检查器

    使用方法:
        enforcer = MvpScopeEnforcer()

        # 启动时检查
        enforcer.check_startup(active_symbols, active_factors)

        # 每次信号生成前检查
        enforcer.check_signal(frequency, symbol, factors_used)

        # 数据请求前检查
        enforcer.check_data_source(data_source)
    """

    # MVP-11 因子白名单 (硬编码作为默认值)
    DEFAULT_MVP_FACTORS = {
        # 动量族 (5个)
        "return_5m",
        "return_1h",
        "ma_cross",
        "breakout_20d",
        "trend_strength",
        # 波动率族 (3个)
        "atr_288",
        "realized_vol_1d",
        "vol_regime",
        # 订单流族 (3个)
        "oi_change_rate",
        "funding_momentum",
        "oi_funding_divergence",
    }

    # MVP 数据源白名单
    DEFAULT_MVP_DATA_SOURCES = {
        "klines_5m",
        "open_interest_5m",
        "funding_8h",
    }

    # MVP 允许的时间框架
    DEFAULT_ALLOWED_FREQUENCIES = {"5m"}

    def __init__(self, config_path: str = "config/mvp_scope.yaml"):
        """
        初始化 MVP Scope 检查器

        Args:
            config_path: MVP 范围配置文件路径
        """
        self.config_path = config_path
        self.violations: List[MvpViolation] = []

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载 MVP 范围配置"""
        config_file = Path(self.config_path)

        if config_file.exists():
            with open(config_file, encoding='utf-8') as f:
                config = yaml.safe_load(f)
            constraints = config.get("mvp_constraints", {})
        else:
            logger.warning(f"MVP配置文件不存在: {self.config_path}, 使用默认配置")
            constraints = {}

        # 时间框架
        self.allowed_frequencies: Set[str] = set(
            constraints.get("allowed_frequencies", list(self.DEFAULT_ALLOWED_FREQUENCIES))
        )

        # 标的配置
        universe = constraints.get("universe", {})
        self.max_symbols: int = universe.get("max_symbols", 50)
        self.allowed_symbols: Set[str] = set(universe.get("allowed_symbols", []))
        self.default_symbols: List[str] = universe.get("default_symbols", ["BTCUSDT", "ETHUSDT"])

        # 因子配置 - MVP-11 白名单优先
        self.allowed_factors: Set[str] = set(
            constraints.get("allowed_factors", list(self.DEFAULT_MVP_FACTORS))
        )
        self.forbidden_factors: List[str] = constraints.get("forbidden_factors", [])

        # 数据源配置 - 白名单是权威
        self.allowed_data_sources: Set[str] = set(
            constraints.get("allowed_data_sources", list(self.DEFAULT_MVP_DATA_SOURCES))
        )
        self.forbidden_data_sources: List[str] = constraints.get("forbidden_data_sources", [])

        # 执行配置
        enforcement = constraints.get("enforcement", {})
        self.on_violation = ViolationAction(enforcement.get("on_violation", "reject"))
        self.check_at_startup = enforcement.get("check_at_startup", True)
        self.check_on_signal = enforcement.get("check_on_signal", True)

    def check_startup(
        self,
        active_symbols: List[str],
        active_factors: List[str]
    ) -> bool:
        """
        启动时检查

        必须在 SignalGenerator.__init__ 或 main() 中调用

        Args:
            active_symbols: 活跃标的列表
            active_factors: 活跃因子列表

        Returns:
            True if passed, False if violations (when on_violation != reject)

        Raises:
            MvpScopeViolationError: 当 on_violation == reject 且有违规时
        """
        if not self.check_at_startup:
            return True

        self.violations.clear()

        # 检查标的数量
        if len(active_symbols) > self.max_symbols:
            self.violations.append(MvpViolation(
                category="symbol",
                value=str(len(active_symbols)),
                message=f"标的数量 {len(active_symbols)} 超过MVP限制 {self.max_symbols}"
            ))

        # 检查标的是否在白名单 (如果配置了白名单)
        if self.allowed_symbols:
            for symbol in active_symbols:
                if symbol not in self.allowed_symbols:
                    self.violations.append(MvpViolation(
                        category="symbol",
                        value=symbol,
                        message=f"标的 {symbol} 不在MVP允许列表中"
                    ))

        # 检查因子 - 白名单优先，不在白名单的一律拒绝
        for factor in active_factors:
            if factor not in self.allowed_factors:
                self.violations.append(MvpViolation(
                    category="factor",
                    value=factor,
                    message=f"因子 {factor} 不在MVP-11允许列表中"
                ))

        return self._handle_violations("startup")

    def check_signal(
        self,
        frequency: str,
        symbol: str,
        factors_used: List[str]
    ) -> bool:
        """
        每次信号生成前检查

        必须在 SignalGenerator.generate() 入口调用

        Args:
            frequency: 时间框架 (如 "5m")
            symbol: 交易对
            factors_used: 使用的因子列表

        Returns:
            True if passed

        Raises:
            MvpScopeViolationError: 当 on_violation == reject 且有违规时
        """
        if not self.check_on_signal:
            return True

        self.violations.clear()

        # 检查时间框架
        if frequency not in self.allowed_frequencies:
            self.violations.append(MvpViolation(
                category="frequency",
                value=frequency,
                message=f"时间框架 {frequency} 不在MVP允许列表 {self.allowed_frequencies}"
            ))

        # 检查标的 (如果配置了白名单)
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            self.violations.append(MvpViolation(
                category="symbol",
                value=symbol,
                message=f"标的 {symbol} 不在MVP允许列表中"
            ))

        # 检查因子 - 白名单优先
        for factor in factors_used:
            if factor not in self.allowed_factors:
                self.violations.append(MvpViolation(
                    category="factor",
                    value=factor,
                    message=f"信号使用了非MVP因子 {factor}"
                ))

        return self._handle_violations(f"signal:{symbol}")

    def check_data_source(self, data_source: str) -> bool:
        """
        数据请求前检查

        必须在 DataService.get_* 方法入口调用

        关键逻辑: 只要不在 allowed_data_sources 就拒绝!
        forbidden_data_sources 仅用于提供更清晰的错误信息

        Args:
            data_source: 数据源ID

        Returns:
            True if passed

        Raises:
            MvpScopeViolationError: 当 on_violation == reject 且有违规时
        """
        self.violations.clear()

        if data_source not in self.allowed_data_sources:
            # 不在白名单 = 一律拒绝 (这是MVP边界的核心!)
            if self._matches_pattern(data_source, self.forbidden_data_sources):
                message = f"数据源 {data_source} 在MVP禁止列表中"
            else:
                message = f"数据源 {data_source} 不在MVP允许列表中 (需审批后加入)"

            self.violations.append(MvpViolation(
                category="data_source",
                value=data_source,
                message=message
            ))
            return self._handle_violations(f"data:{data_source}")

        return True

    def _matches_pattern(self, value: str, patterns: List[str]) -> bool:
        """检查是否匹配禁止模式 (支持通配符)"""
        for pattern in patterns:
            if fnmatch.fnmatch(value, pattern):
                return True
        return False

    def _handle_violations(self, context: str) -> bool:
        """处理违规"""
        if not self.violations:
            return True

        for v in self.violations:
            msg = f"[MVP违规] {context} - {v.category}: {v.message}"

            if self.on_violation == ViolationAction.REJECT:
                logger.error(msg)
                raise MvpScopeViolationError(msg)
            elif self.on_violation == ViolationAction.WARN:
                warnings.warn(msg, UserWarning)
                logger.warning(msg)
            else:  # LOG
                logger.info(msg)

        return self.on_violation != ViolationAction.REJECT

    def get_allowed_factors(self) -> Set[str]:
        """获取允许的因子列表"""
        return self.allowed_factors.copy()

    def get_allowed_data_sources(self) -> Set[str]:
        """获取允许的数据源列表"""
        return self.allowed_data_sources.copy()

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "allowed_frequencies": list(self.allowed_frequencies),
            "max_symbols": self.max_symbols,
            "allowed_factors_count": len(self.allowed_factors),
            "allowed_factors": sorted(self.allowed_factors),
            "allowed_data_sources": sorted(self.allowed_data_sources),
            "on_violation": self.on_violation.value,
        }


# ============== 全局单例 ==============

_enforcer: Optional[MvpScopeEnforcer] = None


def get_enforcer(config_path: str = "config/mvp_scope.yaml") -> MvpScopeEnforcer:
    """获取全局 MVP Scope 检查器实例"""
    global _enforcer
    if _enforcer is None:
        _enforcer = MvpScopeEnforcer(config_path)
    return _enforcer


def reset_enforcer():
    """重置全局检查器 (用于测试)"""
    global _enforcer
    _enforcer = None
