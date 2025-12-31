"""
成交量因子 (15个)

基于成交量的技术指标和衍生因子。

因子列表:
1. VolumeRatio12H - 12小时成交量比
2. VolumeRatio24H - 24小时成交量比
3. VolumeRatio7D - 7天成交量比
4. VolumeTrend - 成交量趋势
5. PriceVolumeCorr - 价量相关性
6. OBV - 能量潮指标
7. OBVChange - OBV变化率
8. VolumeBreakout - 成交量突破
9. RelativeVolume - 相对成交量
10. VolumeMA20 - 20周期成交量均线
11. VolumeMomentum - 成交量动量
12. VolumeZScore - 成交量Z分数
13. VWAP - 成交量加权平均价
14. VWAPDeviation - VWAP偏离度
15. AccumulationDistribution - 累积/派发指标
"""

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
)


BARS_PER_HOUR = 12
BARS_PER_DAY = 288


class VolumeRatio12H(BaseFactor):
    """12小时成交量比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_ratio_12h",
            name="12小时成交量比",
            family=FactorFamily.VOLUME,
            description="当前12小时成交量与过去30天平均12小时成交量的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR * 12 * 30,
            history_tier=HistoryTier.A,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_HOUR * 12:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        current_vol = np.sum(volume[-BARS_PER_HOUR * 12:])
        avg_vol = np.mean([
            np.sum(volume[i:i+BARS_PER_HOUR*12])
            for i in range(0, len(volume)-BARS_PER_HOUR*12, BARS_PER_HOUR*12)
        ][-30:])

        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        return self.create_result(float(ratio), signal_time)


class VolumeRatio24H(BaseFactor):
    """24小时成交量比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_ratio_24h",
            name="24小时成交量比",
            family=FactorFamily.VOLUME,
            description="当前24小时成交量与过去30天平均的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        current_vol = np.sum(volume[-BARS_PER_DAY:])
        avg_vol = np.mean([
            np.sum(volume[i:i+BARS_PER_DAY])
            for i in range(0, len(volume)-BARS_PER_DAY, BARS_PER_DAY)
        ][-30:])

        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        return self.create_result(float(ratio), signal_time)


class VolumeRatio7D(BaseFactor):
    """7天成交量比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_ratio_7d",
            name="7天成交量比",
            family=FactorFamily.VOLUME,
            description="当前7天成交量与过去4周平均的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 28,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        week_bars = BARS_PER_DAY * 7
        current_vol = np.sum(volume[-week_bars:])
        avg_vol = np.mean([
            np.sum(volume[i:i+week_bars])
            for i in range(0, len(volume)-week_bars, week_bars)
        ][-4:])

        ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        return self.create_result(float(ratio), signal_time)


class VolumeTrend(BaseFactor):
    """成交量趋势"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_trend",
            name="成交量趋势",
            family=FactorFamily.VOLUME,
            description="过去24小时成交量的线性回归斜率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values[-BARS_PER_DAY:]
        x = np.arange(len(volume))
        slope, _ = np.polyfit(x, volume, 1)

        normalized_slope = slope / np.mean(volume) if np.mean(volume) > 0 else 0
        return self.create_result(float(normalized_slope), signal_time)


class PriceVolumeCorr(BaseFactor):
    """价量相关性"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="price_volume_corr",
            name="价量相关性",
            family=FactorFamily.VOLUME,
            description="价格变化与成交量的相关系数 (24小时)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values[-BARS_PER_DAY:]
        volume = klines["volume"].values[-BARS_PER_DAY:]

        returns = np.diff(close) / close[:-1]
        corr = np.corrcoef(returns, volume[1:])[0, 1]

        return self.create_result(float(corr) if not np.isnan(corr) else 0.0, signal_time)


class OBV(BaseFactor):
    """On-Balance Volume"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="obv",
            name="能量潮指标",
            family=FactorFamily.VOLUME,
            description="累计成交量指标，上涨加量，下跌减量",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < 50:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        volume = klines["volume"].values

        obv = np.zeros(len(close))
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        obv_zscore = (obv[-1] - np.mean(obv[-BARS_PER_DAY:])) / np.std(obv[-BARS_PER_DAY:])
        return self.create_result(float(obv_zscore) if not np.isnan(obv_zscore) else 0.0, signal_time)


class OBVChange(BaseFactor):
    """OBV变化率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="obv_change",
            name="OBV变化率",
            family=FactorFamily.VOLUME,
            description="OBV的24小时变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 2,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY * 2:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        volume = klines["volume"].values

        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        change = (obv[-1] - obv[-BARS_PER_DAY-1]) / abs(obv[-BARS_PER_DAY-1]) if obv[-BARS_PER_DAY-1] != 0 else 0
        return self.create_result(float(change), signal_time)


class VolumeBreakout(BaseFactor):
    """成交量突破"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_breakout",
            name="成交量突破",
            family=FactorFamily.VOLUME,
            description="当前成交量相对20日均量的突破程度",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY * 20:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        current_vol = volume[-1]
        ma20 = np.mean(volume[-BARS_PER_DAY*20:])
        std20 = np.std(volume[-BARS_PER_DAY*20:])

        breakout = (current_vol - ma20) / std20 if std20 > 0 else 0
        return self.create_result(float(breakout), signal_time)


class RelativeVolume(BaseFactor):
    """相对成交量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="relative_volume",
            name="相对成交量",
            family=FactorFamily.VOLUME,
            description="当前成交量与同一时段历史均值的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        current_vol = volume[-1]

        same_time_vols = [volume[-(BARS_PER_DAY*i+1)] for i in range(1, 8) if len(volume) > BARS_PER_DAY*i]
        avg_vol = np.mean(same_time_vols) if same_time_vols else current_vol

        rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
        return self.create_result(float(rvol), signal_time)


class VolumeMA20(BaseFactor):
    """20周期成交量均线比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_ma20",
            name="成交量20均线比",
            family=FactorFamily.VOLUME,
            description="当前成交量与20周期均量的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < 20:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        ma20 = np.mean(volume[-20:])
        ratio = volume[-1] / ma20 if ma20 > 0 else 1.0

        return self.create_result(float(ratio), signal_time)


class VolumeMomentum(BaseFactor):
    """成交量动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_momentum",
            name="成交量动量",
            family=FactorFamily.VOLUME,
            description="成交量短期均线与长期均线的比值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        ma_short = np.mean(volume[-BARS_PER_HOUR:])
        ma_long = np.mean(volume[-BARS_PER_DAY:])

        momentum = (ma_short / ma_long - 1) if ma_long > 0 else 0
        return self.create_result(float(momentum), signal_time)


class VolumeZScore(BaseFactor):
    """成交量Z分数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_zscore",
            name="成交量Z分数",
            family=FactorFamily.VOLUME,
            description="当前成交量的Z分数 (基于20日)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY * 20:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values
        mean_vol = np.mean(volume[-BARS_PER_DAY*20:])
        std_vol = np.std(volume[-BARS_PER_DAY*20:])

        zscore = (volume[-1] - mean_vol) / std_vol if std_vol > 0 else 0
        return self.create_result(float(zscore), signal_time)


class VWAP(BaseFactor):
    """成交量加权平均价"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vwap",
            name="成交量加权平均价",
            family=FactorFamily.VOLUME,
            description="日内成交量加权平均价",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values[-BARS_PER_DAY:]
        low = klines["low"].values[-BARS_PER_DAY:]
        close = klines["close"].values[-BARS_PER_DAY:]
        volume = klines["volume"].values[-BARS_PER_DAY:]

        typical_price = (high + low + close) / 3
        vwap = np.sum(typical_price * volume) / np.sum(volume)

        return self.create_result(float(vwap), signal_time)


class VWAPDeviation(BaseFactor):
    """VWAP偏离度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vwap_deviation",
            name="VWAP偏离度",
            family=FactorFamily.VOLUME,
            description="当前价格与VWAP的偏离程度",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values[-BARS_PER_DAY:]
        low = klines["low"].values[-BARS_PER_DAY:]
        close = klines["close"].values[-BARS_PER_DAY:]
        volume = klines["volume"].values[-BARS_PER_DAY:]

        typical_price = (high + low + close) / 3
        vwap = np.sum(typical_price * volume) / np.sum(volume)

        current_price = close[-1]
        deviation = (current_price - vwap) / vwap if vwap > 0 else 0

        return self.create_result(float(deviation), signal_time)


class AccumulationDistribution(BaseFactor):
    """累积/派发指标"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="accumulation_distribution",
            name="累积派发指标",
            family=FactorFamily.VOLUME,
            description="累积派发线的Z-score",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m") or data.get("klines")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values
        volume = klines["volume"].values

        mfm = np.zeros(len(close))
        for i in range(len(close)):
            hl_range = high[i] - low[i]
            if hl_range > 0:
                mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            else:
                mfm[i] = 0

        mfv = mfm * volume
        ad = np.cumsum(mfv)

        ad_recent = ad[-BARS_PER_DAY:]
        std_val = np.std(ad_recent)
        zscore = (ad[-1] - np.mean(ad_recent)) / std_val if std_val > 0 else 0

        return self.create_result(float(zscore) if not np.isnan(zscore) else 0.0, signal_time)


# 导出所有因子
VOLUME_FACTORS = [
    VolumeRatio12H,
    VolumeRatio24H,
    VolumeRatio7D,
    VolumeTrend,
    PriceVolumeCorr,
    OBV,
    OBVChange,
    VolumeBreakout,
    RelativeVolume,
    VolumeMA20,
    VolumeMomentum,
    VolumeZScore,
    VWAP,
    VWAPDeviation,
    AccumulationDistribution,
]
