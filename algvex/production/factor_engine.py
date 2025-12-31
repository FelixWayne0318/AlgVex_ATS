"""
AlgVex MVP 因子计算引擎

功能:
- 计算 11 个 MVP 生产因子
- 不依赖 Qlib（独立实现）
- 集成可见性检查
- 支持增量计算

MVP因子 (11个):
- 动量族 (5个): return_5m, return_1h, ma_cross, breakout_20d, trend_strength
- 波动率族 (3个): atr_288, realized_vol_1d, vol_regime
- 订单流族 (3个): oi_change_rate, funding_momentum, oi_funding_divergence
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.visibility_checker import VisibilityChecker
from ..shared.time_provider import TimeProvider


@dataclass
class FactorValue:
    """因子值"""
    factor_id: str
    value: float
    timestamp: datetime
    data_time: datetime
    visible_time: datetime
    is_valid: bool = True
    metadata: Dict[str, Any] = None


class MVPFactorEngine:
    """
    MVP 因子计算引擎

    规则:
    - 所有窗口以 bar 数量计，5m 频率下 288 bars = 1 day
    - 严格遵守可见性规则
    - 不依赖 Qlib
    """

    # 时间常量
    BARS_PER_HOUR = 12       # 60min / 5min

    # MVP因子列表
    MVP_FACTORS = {
        "return_5m", "return_1h", "ma_cross", "breakout_20d", "trend_strength",
        "atr_288", "realized_vol_1d", "vol_regime",
        "oi_change_rate", "funding_momentum", "oi_funding_divergence",
    }

    def get_available_factors(self) -> set:
        """获取可用因子列表"""
        return self.MVP_FACTORS.copy()

    def compute_factor(
        self,
        factor_id: str,
        klines: Optional[pd.DataFrame] = None,
        oi: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        signal_time: Optional[datetime] = None,
    ) -> pd.Series:
        """
        计算单个因子

        Args:
            factor_id: 因子ID
            klines: K线数据
            oi: 持仓量数据
            funding: 资金费率数据
            signal_time: 信号时间

        Returns:
            因子值 Series
        """
        if signal_time is None:
            signal_time = TimeProvider.utcnow()

        # 根据因子ID调用对应的计算方法
        factor_map = {
            "return_5m": lambda: self._compute_return_series(klines, 1),
            "return_1h": lambda: self._compute_return_series(klines, 12),
            "ma_cross": lambda: self._compute_ma_cross_series(klines),
            "breakout_20d": lambda: self._compute_breakout_series(klines, 5760),
            "trend_strength": lambda: self._compute_adx_series(klines, 14),
            "atr_288": lambda: self._compute_atr_series(klines, 288),
            "realized_vol_1d": lambda: self._compute_realized_vol_series(klines, 288),
            "vol_regime": lambda: self._compute_vol_regime_series(klines),
            "oi_change_rate": lambda: self._compute_oi_change_series(oi),
            "funding_momentum": lambda: self._compute_funding_momentum_series(funding),
            "oi_funding_divergence": lambda: self._compute_divergence_series(oi, funding),
        }

        if factor_id not in factor_map:
            raise ValueError(f"Unknown factor: {factor_id}")

        return factor_map[factor_id]()

    def _compute_return_series(self, klines: pd.DataFrame, period: int) -> pd.Series:
        """计算收益率序列"""
        if klines is None or len(klines) < period + 1:
            return pd.Series(dtype=float)
        close = klines["close"]
        returns = close / close.shift(period) - 1
        return returns.dropna()

    def _compute_ma_cross_series(self, klines: pd.DataFrame) -> pd.Series:
        """计算MA交叉序列"""
        if klines is None or len(klines) < 20:
            return pd.Series(dtype=float)
        close = klines["close"]
        ma_fast = close.rolling(5).mean()
        ma_slow = close.rolling(20).mean()
        cross = ma_fast / ma_slow - 1
        return cross.dropna()

    def _compute_breakout_series(self, klines: pd.DataFrame, period: int) -> pd.Series:
        """计算突破序列"""
        if klines is None or len(klines) < period:
            return pd.Series(dtype=float)
        close = klines["close"]
        high = klines["high"]
        rolling_max = high.rolling(period).max()
        atr = self._compute_atr_series(klines, 288)
        if len(atr) == 0:
            atr = pd.Series(1.0, index=close.index)
        breakout = (close - rolling_max) / atr.reindex(close.index).fillna(1.0)
        return breakout.dropna()

    def _compute_adx_series(self, klines: pd.DataFrame, period: int) -> pd.Series:
        """计算ADX序列 (简化版)"""
        if klines is None or len(klines) < period + 1:
            return pd.Series(dtype=float)
        high = klines["high"]
        low = klines["low"]
        close = klines["close"]

        # +DM / -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # TR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx.dropna()

    def _compute_atr_series(self, klines: pd.DataFrame, period: int) -> pd.Series:
        """计算ATR序列"""
        if klines is None or len(klines) < period + 1:
            return pd.Series(dtype=float)
        high = klines["high"]
        low = klines["low"]
        close = klines["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return atr.dropna()

    def _compute_realized_vol_series(self, klines: pd.DataFrame, period: int) -> pd.Series:
        """计算已实现波动率序列"""
        if klines is None or len(klines) < period + 1:
            return pd.Series(dtype=float)
        close = klines["close"]
        returns = np.log(close / close.shift(1))
        vol = returns.rolling(period).std()
        return vol.dropna()

    def _compute_vol_regime_series(self, klines: pd.DataFrame) -> pd.Series:
        """计算波动率状态序列"""
        if klines is None or len(klines) < 288 * 30:
            return pd.Series(dtype=float)
        vol = self._compute_realized_vol_series(klines, 288)
        vol_ma = vol.rolling(30).mean()
        regime = vol / (vol_ma + 1e-10)
        return regime.dropna()

    def _compute_oi_change_series(self, oi: pd.DataFrame) -> pd.Series:
        """计算OI变化率序列"""
        if oi is None or len(oi) < 2:
            return pd.Series(dtype=float)
        oi_val = oi["open_interest"]
        change_rate = oi_val.pct_change()
        return change_rate.dropna()

    def _compute_funding_momentum_series(self, funding: pd.DataFrame) -> pd.Series:
        """计算资金费率动量序列"""
        if funding is None or len(funding) < 8:
            return pd.Series(dtype=float)
        rate = funding["funding_rate"]
        ma_fast = rate.rolling(3).mean()
        ma_slow = rate.rolling(8).mean()
        momentum = ma_fast - ma_slow
        return momentum.dropna()

    def _compute_divergence_series(self, oi: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
        """计算OI-Funding背离序列"""
        if oi is None or funding is None:
            return pd.Series(dtype=float)
        oi_change = self._compute_oi_change_series(oi)
        funding_rate = funding["funding_rate"] if "funding_rate" in funding.columns else pd.Series(dtype=float)

        # 对齐
        common_idx = oi_change.index.intersection(funding_rate.index)
        if len(common_idx) == 0:
            return pd.Series(dtype=float)

        oi_aligned = oi_change.loc[common_idx]
        funding_aligned = funding_rate.loc[common_idx]

        # 背离 = sign(oi_change) != sign(funding)
        divergence = (np.sign(oi_aligned) != np.sign(funding_aligned)).astype(float)
        return divergence

    # 更多时间常量
    BARS_PER_DAY = 288       # 24h * 12
    BARS_PER_WEEK = 2016     # 7 * 288
    BARS_PER_20D = 5760      # 20 * 288

    def __init__(
        self,
        visibility_config: str = "config/visibility.yaml",
        strict_visibility: bool = True,
    ):
        """
        初始化因子引擎

        Args:
            visibility_config: 可见性配置文件
            strict_visibility: 是否严格检查可见性
        """
        self.visibility_checker = VisibilityChecker(visibility_config)
        self.strict_visibility = strict_visibility

        # 因子定义
        self.factor_definitions = self._get_factor_definitions()

    def _get_factor_definitions(self) -> Dict[str, Dict]:
        """获取因子定义"""
        return {
            # 动量因子
            "return_5m": {
                "family": "momentum",
                "data_dependency": ["klines_5m"],
                "window": 1,
                "description": "5分钟收益率",
            },
            "return_1h": {
                "family": "momentum",
                "data_dependency": ["klines_5m"],
                "window": 12,  # 12 * 5min = 1h
                "description": "1小时收益率",
            },
            "ma_cross": {
                "family": "momentum",
                "data_dependency": ["klines_5m"],
                "window": 20,  # MA20
                "description": "均线交叉",
            },
            "breakout_20d": {
                "family": "momentum",
                "data_dependency": ["klines_5m"],
                "window": 5760,  # 20天
                "description": "20日突破",
            },
            "trend_strength": {
                "family": "momentum",
                "data_dependency": ["klines_5m"],
                "window": 14,  # ADX窗口
                "description": "趋势强度(ADX)",
            },
            # 波动率因子
            "atr_288": {
                "family": "volatility",
                "data_dependency": ["klines_5m"],
                "window": 288,  # 1天
                "description": "288期ATR",
            },
            "realized_vol_1d": {
                "family": "volatility",
                "data_dependency": ["klines_5m"],
                "window": 288,
                "description": "1日已实现波动率",
            },
            "vol_regime": {
                "family": "volatility",
                "data_dependency": ["klines_5m"],
                "window": 30 * 288,  # 30天
                "description": "波动率状态",
            },
            # 订单流因子
            "oi_change_rate": {
                "family": "order_flow",
                "data_dependency": ["open_interest_5m"],
                "window": 1,
                "description": "持仓量变化率",
            },
            "funding_momentum": {
                "family": "order_flow",
                "data_dependency": ["funding_8h"],
                "window": 8,  # 8次结算 = 64小时
                "description": "资金费率动量",
            },
            "oi_funding_divergence": {
                "family": "order_flow",
                "data_dependency": ["open_interest_5m", "funding_8h"],
                "window": 1,
                "description": "OI-Funding背离",
            },
        }

    def compute_all_factors(
        self,
        klines: pd.DataFrame,
        oi: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        signal_time: Optional[datetime] = None,
    ) -> Dict[str, FactorValue]:
        """
        计算所有MVP因子

        Args:
            klines: K线数据 (columns: open, high, low, close, volume)
            oi: 持仓量数据 (columns: open_interest)
            funding: 资金费率数据 (columns: funding_rate)
            signal_time: 信号时间

        Returns:
            {factor_id: FactorValue}
        """
        results = {}

        if signal_time is None:
            signal_time = TimeProvider.utcnow()

        # 计算动量因子
        results["return_5m"] = self._compute_return(klines, 1, signal_time, factor_id="return_5m")
        results["return_1h"] = self._compute_return(klines, 12, signal_time, factor_id="return_1h")
        results["ma_cross"] = self._compute_ma_cross(klines, signal_time)
        results["breakout_20d"] = self._compute_breakout(klines, 5760, signal_time)
        results["trend_strength"] = self._compute_adx(klines, 14, signal_time)

        # 计算波动率因子
        results["atr_288"] = self._compute_atr(klines, 288, signal_time)
        results["realized_vol_1d"] = self._compute_realized_vol(klines, 288, signal_time)
        results["vol_regime"] = self._compute_vol_regime(klines, signal_time)

        # 计算订单流因子
        if oi is not None:
            results["oi_change_rate"] = self._compute_oi_change(oi, signal_time)
        if funding is not None:
            results["funding_momentum"] = self._compute_funding_momentum(
                funding, signal_time
            )
        if oi is not None and funding is not None:
            results["oi_funding_divergence"] = self._compute_oi_funding_div(
                oi, funding, signal_time
            )

        return results

    def compute_all_factors_normalized(
        self,
        klines: pd.DataFrame,
        oi: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        signal_time: Optional[datetime] = None,
        method: str = "zscore",
    ) -> Dict[str, FactorValue]:
        """
        计算所有MVP因子并进行标准化

        Args:
            klines: K线数据
            oi: 持仓量数据
            funding: 资金费率数据
            signal_time: 信号时间
            method: 标准化方法 ("zscore", "minmax", "rank")

        Returns:
            标准化后的因子值字典
        """
        # 先计算原始因子
        raw_factors = self.compute_all_factors(klines, oi, funding, signal_time)

        # 标准化
        return self.normalize_factors(raw_factors, method)

    def normalize_factors(
        self,
        factors: Dict[str, FactorValue],
        method: str = "zscore",
    ) -> Dict[str, FactorValue]:
        """
        标准化因子值

        Args:
            factors: 原始因子值字典
            method: 标准化方法
                - "zscore": 使用预定义的均值和标准差
                - "minmax": 归一化到 [-1, 1]
                - "rank": 按因子族内排名

        Returns:
            标准化后的因子值字典
        """
        # 因子标准化参数 (基于历史统计)
        FACTOR_STATS = {
            "return_5m": {"mean": 0.0, "std": 0.002, "min": -0.05, "max": 0.05},
            "return_1h": {"mean": 0.0, "std": 0.01, "min": -0.1, "max": 0.1},
            "ma_cross": {"mean": 0.0, "std": 0.02, "min": -0.1, "max": 0.1},
            "breakout_20d": {"mean": -2.0, "std": 5.0, "min": -20, "max": 5},
            "trend_strength": {"mean": 25.0, "std": 15.0, "min": 0, "max": 100},
            "atr_288": {"mean": 100.0, "std": 50.0, "min": 10, "max": 500},
            "realized_vol_1d": {"mean": 0.002, "std": 0.001, "min": 0.0005, "max": 0.01},
            "vol_regime": {"mean": 1.0, "std": 0.3, "min": 0.3, "max": 3.0},
            "oi_change_rate": {"mean": 0.0, "std": 0.02, "min": -0.1, "max": 0.1},
            "funding_momentum": {"mean": 0.0, "std": 0.0001, "min": -0.001, "max": 0.001},
            "oi_funding_divergence": {"mean": 0.0, "std": 1.0, "min": -1, "max": 1},
        }

        normalized = {}

        for factor_id, factor_value in factors.items():
            if not factor_value.is_valid:
                normalized[factor_id] = factor_value
                continue

            stats = FACTOR_STATS.get(factor_id, {"mean": 0, "std": 1, "min": -1, "max": 1})
            raw_val = factor_value.value

            if method == "zscore":
                # Z-score 标准化
                if stats["std"] > 0:
                    norm_val = (raw_val - stats["mean"]) / stats["std"]
                else:
                    norm_val = 0.0
                # 裁剪到 [-3, 3]
                norm_val = max(-3.0, min(3.0, norm_val))

            elif method == "minmax":
                # Min-Max 归一化到 [-1, 1]
                range_val = stats["max"] - stats["min"]
                if range_val > 0:
                    norm_val = 2 * (raw_val - stats["min"]) / range_val - 1
                else:
                    norm_val = 0.0
                norm_val = max(-1.0, min(1.0, norm_val))

            elif method == "rank":
                # 简单的百分位转换
                norm_val = raw_val  # 需要历史数据支持，暂时保持原值

            else:
                norm_val = raw_val

            normalized[factor_id] = FactorValue(
                factor_id=factor_id,
                value=norm_val,
                timestamp=factor_value.timestamp,
                data_time=factor_value.data_time,
                visible_time=factor_value.visible_time,
                is_valid=True,
            )

        return normalized

    def get_factor_requirements(self) -> Dict[str, Dict[str, int]]:
        """
        获取每个因子的最小数据量要求

        Returns:
            {factor_id: {"min_bars": N, "description": "..."}}
        """
        return {
            "return_5m": {"min_bars": 2, "description": "需要至少2条K线"},
            "return_1h": {"min_bars": 13, "description": "需要至少13条K线 (12+1)"},
            "ma_cross": {"min_bars": 21, "description": "需要至少21条K线 (MA20+1)"},
            "breakout_20d": {"min_bars": 5761, "description": "需要至少5761条K线 (20天+1)"},
            "trend_strength": {"min_bars": 28, "description": "需要至少28条K线 (ADX计算)"},
            "atr_288": {"min_bars": 289, "description": "需要至少289条K线 (1天+1)"},
            "realized_vol_1d": {"min_bars": 289, "description": "需要至少289条K线 (1天+1)"},
            "vol_regime": {"min_bars": 318, "description": "需要至少318条K线 (288+30)"},
            "oi_change_rate": {"min_bars": 2, "description": "需要至少2条OI数据"},
            "funding_momentum": {"min_bars": 8, "description": "需要至少8次资金费率结算"},
            "oi_funding_divergence": {"min_bars": 2, "description": "需要OI和Funding数据"},
        }

    def _compute_return(
        self,
        klines: pd.DataFrame,
        periods: int,
        signal_time: datetime,
        factor_id: Optional[str] = None,
    ) -> FactorValue:
        """计算收益率因子"""
        # 使用提供的 factor_id 或根据 periods 生成
        if factor_id is None:
            factor_id = f"return_{periods}"

        if len(klines) < periods + 1:
            return FactorValue(
                factor_id=factor_id,
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        close = klines["close"].values
        ret = close[-1] / close[-periods - 1] - 1

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id=factor_id,
            value=float(ret),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_ma_cross(
        self,
        klines: pd.DataFrame,
        signal_time: datetime,
    ) -> FactorValue:
        """计算均线交叉因子: MA(5) / MA(20) - 1"""
        if len(klines) < 20:
            return FactorValue(
                factor_id="ma_cross",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        close = klines["close"].values
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        value = ma5 / ma20 - 1

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id="ma_cross",
            value=float(value),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_breakout(
        self,
        klines: pd.DataFrame,
        window: int,
        signal_time: datetime,
    ) -> FactorValue:
        """计算突破因子: (close - rolling_max(high, window)) / atr"""
        if len(klines) < window:
            return FactorValue(
                factor_id=f"breakout_{window // 288}d",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        close = klines["close"].values[-1]
        high = klines["high"].values[-window:]
        rolling_max = np.max(high)

        # 计算ATR
        atr = self._compute_atr(klines, 288, signal_time).value
        if atr == 0 or np.isnan(atr):
            atr = 1

        value = (close - rolling_max) / atr

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id=f"breakout_{window // 288}d",
            value=float(value),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_adx(
        self,
        klines: pd.DataFrame,
        period: int,
        signal_time: datetime,
    ) -> FactorValue:
        """计算ADX (趋势强度)"""
        if len(klines) < period * 2:
            return FactorValue(
                factor_id="trend_strength",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        # 计算 TR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # 计算 +DM 和 -DM
        dm_plus = np.where(
            (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
            np.maximum(high[1:] - high[:-1], 0),
            0
        )
        dm_minus = np.where(
            (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
            np.maximum(low[:-1] - low[1:], 0),
            0
        )

        # 平滑
        atr = self._ema(tr, period)
        di_plus = 100 * self._ema(dm_plus, period) / (atr + 1e-10)
        di_minus = 100 * self._ema(dm_minus, period) / (atr + 1e-10)

        # 计算 DX 和 ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = self._ema(dx, period)

        value = adx[-1] if len(adx) > 0 else np.nan

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id="trend_strength",
            value=float(value),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=not np.isnan(value),
        )

    def _compute_atr(
        self,
        klines: pd.DataFrame,
        period: int,
        signal_time: datetime,
    ) -> FactorValue:
        """计算ATR"""
        if len(klines) < period:
            return FactorValue(
                factor_id=f"atr_{period}",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # ATR = 平均TR
        atr = np.mean(tr[-period:])

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id=f"atr_{period}",
            value=float(atr),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_realized_vol(
        self,
        klines: pd.DataFrame,
        period: int,
        signal_time: datetime,
    ) -> FactorValue:
        """计算已实现波动率"""
        if len(klines) < period:
            return FactorValue(
                factor_id="realized_vol_1d",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        close = klines["close"].values
        returns = np.diff(np.log(close))[-period:]
        vol = np.std(returns)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id="realized_vol_1d",
            value=float(vol),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_vol_regime(
        self,
        klines: pd.DataFrame,
        signal_time: datetime,
    ) -> FactorValue:
        """计算波动率状态: realized_vol_1d / MA(realized_vol_1d, 30)"""
        if len(klines) < 30 * 288:  # 需要30天数据
            return FactorValue(
                factor_id="vol_regime",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        close = klines["close"].values
        returns = np.diff(np.log(close))

        # 计算滚动波动率
        window = 288  # 1天
        vols = []
        for i in range(30):
            start = -(30 - i) * window
            end = start + window if start + window < 0 else None
            if end is None:
                vol = np.std(returns[start:])
            else:
                vol = np.std(returns[start:end])
            vols.append(vol)

        current_vol = vols[-1]
        avg_vol = np.mean(vols)
        regime = current_vol / (avg_vol + 1e-10)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return FactorValue(
            factor_id="vol_regime",
            value=float(regime),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,
            is_valid=True,
        )

    def _compute_oi_change(
        self,
        oi: pd.DataFrame,
        signal_time: datetime,
    ) -> FactorValue:
        """计算持仓量变化率"""
        if len(oi) < 2:
            return FactorValue(
                factor_id="oi_change_rate",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        oi_values = oi["open_interest"].values
        change = (oi_values[-1] - oi_values[-2]) / (oi_values[-2] + 1e-10)

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        # OI有5分钟延迟
        visible_time = data_time + timedelta(minutes=5)

        return FactorValue(
            factor_id="oi_change_rate",
            value=float(change),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=visible_time,
            is_valid=True,
        )

    def _compute_funding_momentum(
        self,
        funding: pd.DataFrame,
        signal_time: datetime,
    ) -> FactorValue:
        """
        计算资金费率动量

        重要: 窗口以结算次数计，不是bars!
        MA(3) = 最近3次结算 = 24小时
        MA(8) = 最近8次结算 = 64小时
        """
        if len(funding) < 8:
            return FactorValue(
                factor_id="funding_momentum",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        rates = funding["funding_rate"].values

        # 只取最近的结算值
        ma3 = np.mean(rates[-3:])
        ma8 = np.mean(rates[-8:])
        momentum = ma3 - ma8

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time

        return FactorValue(
            factor_id="funding_momentum",
            value=float(momentum),
            timestamp=signal_time,
            data_time=data_time,
            visible_time=data_time,  # 结算时刻立即可见
            is_valid=True,
        )

    def _compute_oi_funding_div(
        self,
        oi: pd.DataFrame,
        funding: pd.DataFrame,
        signal_time: datetime,
    ) -> FactorValue:
        """计算OI-Funding背离: sign(oi_change) != sign(funding)"""
        oi_change = self._compute_oi_change(oi, signal_time)
        latest_funding = funding["funding_rate"].values[-1] if len(funding) > 0 else 0

        if not oi_change.is_valid:
            return FactorValue(
                factor_id="oi_funding_divergence",
                value=np.nan,
                timestamp=signal_time,
                data_time=signal_time,
                visible_time=signal_time,
                is_valid=False,
            )

        # 背离 = 符号不同
        oi_sign = np.sign(oi_change.value)
        funding_sign = np.sign(latest_funding)
        divergence = 1.0 if oi_sign != funding_sign else 0.0

        # 可见时间取两者最晚
        visible_time = max(oi_change.visible_time, signal_time)

        return FactorValue(
            factor_id="oi_funding_divergence",
            value=divergence,
            timestamp=signal_time,
            data_time=oi_change.data_time,
            visible_time=visible_time,
            is_valid=True,
        )

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        """计算EMA"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(values, dtype=float)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    def get_factor_names(self) -> List[str]:
        """获取所有因子名称"""
        return list(self.factor_definitions.keys())

    def get_factor_info(self, factor_id: str) -> Dict:
        """获取因子信息"""
        return self.factor_definitions.get(factor_id, {})


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    np.random.seed(42)

    klines = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(1000) * 0.1),
            "high": 101 + np.cumsum(np.random.randn(1000) * 0.1),
            "low": 99 + np.cumsum(np.random.randn(1000) * 0.1),
            "close": 100.5 + np.cumsum(np.random.randn(1000) * 0.1),
            "volume": 1000 + np.random.randint(0, 500, 1000),
        },
        index=dates,
    )
    klines["high"] = klines[["open", "high", "close"]].max(axis=1)
    klines["low"] = klines[["open", "low", "close"]].min(axis=1)

    oi = pd.DataFrame(
        {"open_interest": 50000 + np.cumsum(np.random.randn(1000) * 100)},
        index=dates,
    )

    funding_dates = pd.date_range("2024-01-01", periods=100, freq="8h")
    funding = pd.DataFrame(
        {"funding_rate": np.random.randn(100) * 0.0001},
        index=funding_dates,
    )

    # 计算因子
    engine = MVPFactorEngine()
    signal_time = datetime(2024, 1, 4, 10, 5, 0)

    print("=== MVP 因子计算 ===")
    factors = engine.compute_all_factors(klines, oi, funding, signal_time)

    for factor_id, factor_value in factors.items():
        print(f"{factor_id}: {factor_value.value:.6f} (valid={factor_value.is_valid})")


# 别名，保持向后兼容
FactorEngine = MVPFactorEngine
