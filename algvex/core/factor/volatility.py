"""
AlgVex 波动率因子 (15个)

包含:
- 波动率: volatility_12h, volatility_24h, volatility_7d
- ATR: atr_24h, atr_7d
- 偏度/峰度: skewness, kurtosis
- 波动率比率: volatility_ratio
- 已实现波动率: realized_vol, rv_ratio
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
    ema, rolling_std, log_returns,
)


BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BARS_PER_WEEK = 2016


class Volatility12H(BaseFactor):
    """12小时波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_12h",
            name="12小时波动率",
            family=FactorFamily.VOLATILITY,
            description="过去12小时对数收益率标准差 (年化)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR * 12,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = BARS_PER_HOUR * 12
        if klines is None or len(klines) < period + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-period:]
        vol = np.std(rets) * np.sqrt(BARS_PER_DAY * 365)  # 年化

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(vol), signal_time, data_time)


class Volatility24H(BaseFactor):
    """24小时波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_24h",
            name="24小时波动率",
            family=FactorFamily.VOLATILITY,
            description="过去24小时对数收益率标准差 (年化)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_DAY:]
        vol = np.std(rets) * np.sqrt(BARS_PER_DAY * 365)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(vol), signal_time, data_time)


class Volatility7D(BaseFactor):
    """7天波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_7d",
            name="7天波动率",
            family=FactorFamily.VOLATILITY,
            description="过去7天对数收益率标准差 (年化)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_WEEK:]
        vol = np.std(rets) * np.sqrt(BARS_PER_DAY * 365)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(vol), signal_time, data_time)


class ATR24H(BaseFactor):
    """24小时ATR"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="atr_24h",
            name="24小时ATR",
            family=FactorFamily.VOLATILITY,
            description="24小时平均真实范围",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-BARS_PER_DAY:])

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(atr), signal_time, data_time)


class ATR7D(BaseFactor):
    """7天ATR"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="atr_7d",
            name="7天ATR",
            family=FactorFamily.VOLATILITY,
            description="7天平均真实范围",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-BARS_PER_WEEK:])

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(atr), signal_time, data_time)


class Skewness(BaseFactor):
    """收益率偏度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="skewness",
            name="收益率偏度",
            family=FactorFamily.VOLATILITY,
            description="24小时收益率偏度",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_DAY:]

        if len(rets) < 3:
            return self.create_invalid_result(signal_time, "Insufficient returns")

        # 使用 pandas 计算偏度
        skew = pd.Series(rets).skew()

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(skew), signal_time, data_time)


class Kurtosis(BaseFactor):
    """收益率峰度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="kurtosis",
            name="收益率峰度",
            family=FactorFamily.VOLATILITY,
            description="24小时收益率峰度 (超额)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_DAY:]

        if len(rets) < 4:
            return self.create_invalid_result(signal_time, "Insufficient returns")

        # 使用 pandas 计算峰度 (超额)
        kurt = pd.Series(rets).kurtosis()

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(kurt), signal_time, data_time)


class VolatilityRatio(BaseFactor):
    """波动率比率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_ratio",
            name="波动率比率",
            family=FactorFamily.VOLATILITY,
            description="短期波动率 / 长期波动率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))

        vol_short = np.std(rets[-BARS_PER_DAY:])
        vol_long = np.std(rets[-BARS_PER_WEEK:])

        if vol_long == 0:
            value = 1.0
        else:
            value = vol_short / vol_long

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class RealizedVol1D(BaseFactor):
    """1日已实现波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="realized_vol_1d",
            name="1日已实现波动率",
            family=FactorFamily.VOLATILITY,
            description="使用5分钟收益率计算的已实现波动率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_DAY:]

        # 已实现波动率 = sqrt(sum(r^2))
        rv = np.sqrt(np.sum(rets ** 2))

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(rv), signal_time, data_time)


class RVRatio(BaseFactor):
    """已实现波动率比率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="rv_ratio",
            name="已实现波动率比率",
            family=FactorFamily.VOLATILITY,
            description="当日RV / 7日平均RV",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))

        # 计算每日 RV
        daily_rvs = []
        for i in range(7):
            start = -(7 - i) * BARS_PER_DAY
            end = start + BARS_PER_DAY if start + BARS_PER_DAY < 0 else None
            if end is None:
                day_rets = rets[start:]
            else:
                day_rets = rets[start:end]
            daily_rvs.append(np.sqrt(np.sum(day_rets ** 2)))

        current_rv = daily_rvs[-1]
        avg_rv = np.mean(daily_rvs)

        if avg_rv == 0:
            value = 1.0
        else:
            value = current_rv / avg_rv

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class VolRegime(BaseFactor):
    """波动率状态"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vol_regime",
            name="波动率状态",
            family=FactorFamily.VOLATILITY,
            description="当前波动率相对30日均值的位置",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY * 30 + 1:
            return self.create_invalid_result(signal_time, "Insufficient data for 30 days")

        close = klines["close"].values
        rets = np.diff(np.log(close))

        # 计算滚动波动率
        vols = []
        for i in range(30):
            start = -(30 - i) * BARS_PER_DAY
            end = start + BARS_PER_DAY if start + BARS_PER_DAY < 0 else None
            if end is None:
                vol = np.std(rets[start:])
            else:
                vol = np.std(rets[start:end])
            vols.append(vol)

        current_vol = vols[-1]
        avg_vol = np.mean(vols)
        regime = current_vol / (avg_vol + 1e-10)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(regime), signal_time, data_time)


class GarmanKlass(BaseFactor):
    """Garman-Klass波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="garman_klass",
            name="Garman-Klass波动率",
            family=FactorFamily.VOLATILITY,
            description="使用OHLC的波动率估计",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        open_p = klines["open"].values[-BARS_PER_DAY:]
        high = klines["high"].values[-BARS_PER_DAY:]
        low = klines["low"].values[-BARS_PER_DAY:]
        close = klines["close"].values[-BARS_PER_DAY:]

        # Garman-Klass 估计
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_p) ** 2

        gk = np.sqrt(np.mean(0.5 * log_hl - (2 * np.log(2) - 1) * log_co))
        # 年化
        gk_annual = gk * np.sqrt(BARS_PER_DAY * 365)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(gk_annual), signal_time, data_time)


class Parkinson(BaseFactor):
    """Parkinson波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="parkinson",
            name="Parkinson波动率",
            family=FactorFamily.VOLATILITY,
            description="使用最高最低价的波动率估计",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values[-BARS_PER_DAY:]
        low = klines["low"].values[-BARS_PER_DAY:]

        # Parkinson 估计
        log_hl = np.log(high / low) ** 2
        pk = np.sqrt(np.mean(log_hl) / (4 * np.log(2)))
        # 年化
        pk_annual = pk * np.sqrt(BARS_PER_DAY * 365)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(pk_annual), signal_time, data_time)


class VolZScore(BaseFactor):
    """波动率Z-Score"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vol_zscore",
            name="波动率Z-Score",
            family=FactorFamily.VOLATILITY,
            description="当前波动率的标准化值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY * 30 + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))

        # 计算滚动波动率
        vols = []
        for i in range(30):
            start = -(30 - i) * BARS_PER_DAY
            end = start + BARS_PER_DAY if start + BARS_PER_DAY < 0 else None
            if end is None:
                vol = np.std(rets[start:])
            else:
                vol = np.std(rets[start:end])
            vols.append(vol)

        current_vol = vols[-1]
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)

        if std_vol == 0:
            zscore = 0
        else:
            zscore = (current_vol - mean_vol) / std_vol

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(zscore), signal_time, data_time)


# ===== 追加因子 (1个) =====

class VolatilityTrend(BaseFactor):
    """波动率趋势"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_trend",
            name="波动率趋势",
            family=FactorFamily.VOLATILITY,
            description="波动率上升(+1)或下降(-1)趋势",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))

        # 近期vs远期波动率
        vol_recent = np.std(rets[-BARS_PER_DAY:])
        vol_past = np.std(rets[-BARS_PER_DAY * 7:-BARS_PER_DAY])

        if vol_past == 0:
            trend = 0
        else:
            ratio = vol_recent / vol_past
            if ratio > 1.2:
                trend = 1.0   # 波动率上升
            elif ratio < 0.8:
                trend = -1.0  # 波动率下降
            else:
                trend = 0.0   # 波动率稳定

        return self.create_result(float(trend), signal_time)


# 因子注册表
VOLATILITY_FACTORS = [
    Volatility12H,
    Volatility24H,
    Volatility7D,
    ATR24H,
    ATR7D,
    Skewness,
    Kurtosis,
    VolatilityRatio,
    RealizedVol1D,
    RVRatio,
    VolRegime,
    GarmanKlass,
    Parkinson,
    VolZScore,
    VolatilityTrend,
]


def get_volatility_factors() -> List[BaseFactor]:
    """获取所有波动率因子实例"""
    return [f() for f in VOLATILITY_FACTORS]
