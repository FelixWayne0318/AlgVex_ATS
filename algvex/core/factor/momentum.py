"""
AlgVex 动量因子 (20个)

包含:
- 收益率因子: return_1h, return_4h, return_24h, return_7d
- 动量因子: mom_12h, mom_24h, mom_72h
- 均线交叉: ma_cross_12_24, ma_cross_24_72
- 价格位置: price_position_52w
- 突破: breakout_20d, breakout_60d
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
    ema, sma, rolling_max, rolling_min, returns,
)


# 时间常量 (5分钟频率)
BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BARS_PER_WEEK = 2016


class Return1H(BaseFactor):
    """1小时收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="return_1h",
            name="1小时收益率",
            family=FactorFamily.MOMENTUM,
            description="过去1小时的价格变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_HOUR + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret = close[-1] / close[-BARS_PER_HOUR - 1] - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(ret), signal_time, data_time)


class Return4H(BaseFactor):
    """4小时收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="return_4h",
            name="4小时收益率",
            family=FactorFamily.MOMENTUM,
            description="过去4小时的价格变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = BARS_PER_HOUR * 4
        if klines is None or len(klines) < period + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret = close[-1] / close[-period - 1] - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(ret), signal_time, data_time)


class Return24H(BaseFactor):
    """24小时收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="return_24h",
            name="24小时收益率",
            family=FactorFamily.MOMENTUM,
            description="过去24小时的价格变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret = close[-1] / close[-BARS_PER_DAY - 1] - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(ret), signal_time, data_time)


class Return7D(BaseFactor):
    """7天收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="return_7d",
            name="7天收益率",
            family=FactorFamily.MOMENTUM,
            description="过去7天的价格变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret = close[-1] / close[-BARS_PER_WEEK - 1] - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(ret), signal_time, data_time)


class Momentum12H(BaseFactor):
    """12小时动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="mom_12h",
            name="12小时动量",
            family=FactorFamily.MOMENTUM,
            description="12小时价格动量 (收益率加速度)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR * 12 * 2,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = BARS_PER_HOUR * 12
        if klines is None or len(klines) < period * 2 + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret_current = close[-1] / close[-period - 1] - 1
        ret_previous = close[-period - 1] / close[-period * 2 - 1] - 1
        momentum = ret_current - ret_previous
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(momentum), signal_time, data_time)


class Momentum24H(BaseFactor):
    """24小时动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="mom_24h",
            name="24小时动量",
            family=FactorFamily.MOMENTUM,
            description="24小时价格动量",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 2,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = BARS_PER_DAY
        if klines is None or len(klines) < period * 2 + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret_current = close[-1] / close[-period - 1] - 1
        ret_previous = close[-period - 1] / close[-period * 2 - 1] - 1
        momentum = ret_current - ret_previous
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(momentum), signal_time, data_time)


class Momentum72H(BaseFactor):
    """72小时动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="mom_72h",
            name="72小时动量",
            family=FactorFamily.MOMENTUM,
            description="72小时价格动量",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 3 * 2,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = BARS_PER_DAY * 3
        if klines is None or len(klines) < period * 2 + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ret_current = close[-1] / close[-period - 1] - 1
        ret_previous = close[-period - 1] / close[-period * 2 - 1] - 1
        momentum = ret_current - ret_previous
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(momentum), signal_time, data_time)


class MACross12_24(BaseFactor):
    """MA12/MA24 交叉"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ma_cross_12_24",
            name="MA12/MA24交叉",
            family=FactorFamily.MOMENTUM,
            description="12期均线与24期均线的比值 - 1",
            data_dependencies=[DataDependency.KLINES_5M],
            window=24,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < 24:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ma12 = np.mean(close[-12:])
        ma24 = np.mean(close[-24:])
        value = ma12 / ma24 - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class MACross24_72(BaseFactor):
    """MA24/MA72 交叉"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ma_cross_24_72",
            name="MA24/MA72交叉",
            family=FactorFamily.MOMENTUM,
            description="24期均线与72期均线的比值 - 1",
            data_dependencies=[DataDependency.KLINES_5M],
            window=72,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < 72:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        ma24 = np.mean(close[-24:])
        ma72 = np.mean(close[-72:])
        value = ma24 / ma72 - 1
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class PricePosition52W(BaseFactor):
    """52周价格位置"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="price_position_52w",
            name="52周价格位置",
            family=FactorFamily.MOMENTUM,
            description="当前价格在52周高低范围内的位置 (0-1)",
            data_dependencies=[DataDependency.KLINES_1D],
            window=365,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_1d") or data.get("klines_5m")
        if klines is None:
            return self.create_invalid_result(signal_time, "No price data")

        # 尝试使用日线，否则用5分钟线计算
        if "klines_1d" in data and len(data["klines_1d"]) >= 365:
            high = data["klines_1d"]["high"].values[-365:]
            low = data["klines_1d"]["low"].values[-365:]
            close = data["klines_1d"]["close"].values[-1]
        elif len(klines) >= BARS_PER_DAY * 365:
            high = klines["high"].values[-(BARS_PER_DAY * 365):]
            low = klines["low"].values[-(BARS_PER_DAY * 365):]
            close = klines["close"].values[-1]
        else:
            return self.create_invalid_result(signal_time, "Insufficient data for 52w")

        high_52w = np.max(high)
        low_52w = np.min(low)
        if high_52w == low_52w:
            value = 0.5
        else:
            value = (close - low_52w) / (high_52w - low_52w)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class Breakout20D(BaseFactor):
    """20日突破"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="breakout_20d",
            name="20日突破",
            family=FactorFamily.MOMENTUM,
            description="(当前价格 - 20日最高价) / ATR",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        window = BARS_PER_DAY * 20
        if klines is None or len(klines) < window:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values[-1]
        high = klines["high"].values[-window:]
        rolling_high = np.max(high)

        # 计算 ATR
        atr = self._compute_atr(klines, BARS_PER_DAY)
        if atr == 0 or np.isnan(atr):
            atr = 1

        value = (close - rolling_high) / atr
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)

    def _compute_atr(self, klines: pd.DataFrame, period: int) -> float:
        """计算ATR"""
        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        if len(high) < period + 1:
            return np.nan

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return np.mean(tr[-period:])


class Breakout60D(BaseFactor):
    """60日突破"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="breakout_60d",
            name="60日突破",
            family=FactorFamily.MOMENTUM,
            description="(当前价格 - 60日最高价) / ATR",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 60,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        window = BARS_PER_DAY * 60
        if klines is None or len(klines) < window:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values[-1]
        high = klines["high"].values[-window:]
        rolling_high = np.max(high)

        # 计算 ATR
        atr = self._compute_atr(klines, BARS_PER_DAY)
        if atr == 0 or np.isnan(atr):
            atr = 1

        value = (close - rolling_high) / atr
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)

    def _compute_atr(self, klines: pd.DataFrame, period: int) -> float:
        """计算ATR"""
        high = klines["high"].values
        low = klines["low"].values
        close = klines["close"].values

        if len(high) < period + 1:
            return np.nan

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return np.mean(tr[-period:])


class TrendStrength(BaseFactor):
    """趋势强度 (ADX)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="trend_strength",
            name="趋势强度",
            family=FactorFamily.MOMENTUM,
            description="ADX指标，衡量趋势强度 (0-100)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=28,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = 14
        if klines is None or len(klines) < period * 2:
            return self.create_invalid_result(signal_time, "Insufficient data")

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
        atr = ema(tr, period)
        di_plus = 100 * ema(dm_plus, period) / (atr + 1e-10)
        di_minus = 100 * ema(dm_minus, period) / (atr + 1e-10)

        # 计算 DX 和 ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = ema(dx, period)

        value = adx[-1] if len(adx) > 0 else np.nan
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class RSI14(BaseFactor):
    """14期RSI"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="rsi_14",
            name="14期RSI",
            family=FactorFamily.MOMENTUM,
            description="相对强弱指标 (0-100)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=14,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        period = 14
        if klines is None or len(klines) < period + 1:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        delta = np.diff(close)

        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = ema(gains, period)[-1]
        avg_loss = ema(losses, period)[-1]

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(rsi), signal_time, data_time)


class MACD(BaseFactor):
    """MACD"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="macd",
            name="MACD",
            family=FactorFamily.MOMENTUM,
            description="MACD柱状图值",
            data_dependencies=[DataDependency.KLINES_5M],
            window=26,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < 35:  # 26 + 9
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        ema12 = ema(close, 12)
        ema26 = ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        histogram = macd_line - signal_line

        value = histogram[-1]
        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class MomentumQuality(BaseFactor):
    """动量质量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="momentum_quality",
            name="动量质量",
            family=FactorFamily.MOMENTUM,
            description="动量的一致性 (收益率 / 波动率)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        rets = np.diff(np.log(close))[-BARS_PER_DAY:]

        ret = close[-1] / close[-BARS_PER_DAY] - 1
        vol = np.std(rets) * np.sqrt(BARS_PER_DAY)

        if vol == 0:
            value = 0
        else:
            value = ret / vol

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class HighLowRatio(BaseFactor):
    """高低比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="high_low_ratio",
            name="高低比",
            family=FactorFamily.MOMENTUM,
            description="(收盘-最低) / (最高-最低)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values[-1]
        high = np.max(klines["high"].values[-BARS_PER_DAY:])
        low = np.min(klines["low"].values[-BARS_PER_DAY:])

        if high == low:
            value = 0.5
        else:
            value = (close - low) / (high - low)

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class PriceMomentum(BaseFactor):
    """价格动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="price_momentum",
            name="价格动量",
            family=FactorFamily.MOMENTUM,
            description="多周期动量加权平均",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_WEEK,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_WEEK:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 多周期动量
        ret_1d = close[-1] / close[-BARS_PER_DAY] - 1 if len(close) >= BARS_PER_DAY else 0
        ret_3d = close[-1] / close[-BARS_PER_DAY * 3] - 1 if len(close) >= BARS_PER_DAY * 3 else 0
        ret_7d = close[-1] / close[-BARS_PER_WEEK] - 1 if len(close) >= BARS_PER_WEEK else 0

        # 加权平均 (近期权重更高)
        value = 0.5 * ret_1d + 0.3 * ret_3d + 0.2 * ret_7d

        data_time = klines.index[-1] if hasattr(klines.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


# ===== 追加因子 (2个) =====

class Return5M(BaseFactor):
    """5分钟收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="return_5m",
            name="5分钟收益率",
            family=FactorFamily.MOMENTUM,
            description="最近5分钟的收益率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=2,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < 2:
            return self.create_invalid_result(signal_time, "Insufficient data")
        close = klines["close"].values
        value = close[-1] / close[-2] - 1
        return self.create_result(float(value), signal_time)


class MACross5_20(BaseFactor):
    """MA交叉 5/20"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ma_cross_5_20",
            name="MA交叉 5/20",
            family=FactorFamily.MOMENTUM,
            description="5周期均线相对20周期均线",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.A,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")
        close = klines["close"].values
        ma5 = np.mean(close[-5:])
        ma20 = np.mean(close[-20:])
        value = ma5 / ma20 - 1
        return self.create_result(float(value), signal_time)


# 因子注册表
MOMENTUM_FACTORS = [
    Return5M,
    Return1H,
    Return4H,
    Return24H,
    Return7D,
    Momentum12H,
    Momentum24H,
    Momentum72H,
    MACross5_20,
    MACross12_24,
    MACross24_72,
    PricePosition52W,
    Breakout20D,
    Breakout60D,
    TrendStrength,
    RSI14,
    MACD,
    MomentumQuality,
    HighLowRatio,
    PriceMomentum,
]


def get_momentum_factors() -> List[BaseFactor]:
    """获取所有动量因子实例"""
    return [f() for f in MOMENTUM_FACTORS]
