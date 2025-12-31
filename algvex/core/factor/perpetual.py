"""
AlgVex 永续合约专用因子 (45个)

包含:
- 资金费率因子 (12个): funding_rate, funding_rate_ma_8h, funding_momentum, etc.
- 持仓量因子 (12个): oi_change_1h, oi_volume_ratio, oi_price_divergence, etc.
- 多空博弈+CVD因子 (21个): long_short_ratio, taker_buy_sell_ratio, cvd, etc.
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
    ema, sma, zscore,
)


BARS_PER_HOUR = 12
BARS_PER_DAY = 288


# ===== 资金费率因子 (12个) =====

class FundingRate(BaseFactor):
    """当前资金费率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_rate",
            name="资金费率",
            family=FactorFamily.FUNDING,
            description="当前8小时资金费率",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")

        value = funding["funding_rate"].values[-1]
        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class FundingRateMA8H(BaseFactor):
    """8小时资金费率均值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_rate_ma_8h",
            name="8小时资金费率均值",
            family=FactorFamily.FUNDING,
            description="最近1次结算的资金费率 (即当前)",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")

        # 实际上8h就是一次结算
        value = funding["funding_rate"].values[-1]
        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class FundingRateMA24H(BaseFactor):
    """24小时资金费率均值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_rate_ma_24h",
            name="24小时资金费率均值",
            family=FactorFamily.FUNDING,
            description="最近3次结算的资金费率均值",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=3,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 3:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values[-3:]
        value = np.mean(rates)
        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class FundingPremium(BaseFactor):
    """资金费率溢价"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_premium",
            name="资金费率溢价",
            family=FactorFamily.FUNDING,
            description="当前资金费率 - 历史均值",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 30:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values
        current = rates[-1]
        historical_mean = np.mean(rates[-30:])
        value = current - historical_mean

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingMomentum(BaseFactor):
    """资金费率动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_momentum",
            name="资金费率动量",
            family=FactorFamily.FUNDING,
            description="MA(3) - MA(8) 以结算次数计",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=8,
            history_tier=HistoryTier.B,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 8:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values
        ma3 = np.mean(rates[-3:])
        ma8 = np.mean(rates[-8:])
        value = ma3 - ma8

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingZScore(BaseFactor):
    """资金费率Z-Score"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_zscore",
            name="资金费率Z-Score",
            family=FactorFamily.FUNDING,
            description="当前资金费率的标准化值",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 30:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values[-30:]
        current = rates[-1]
        mean = np.mean(rates)
        std = np.std(rates)

        if std == 0:
            value = 0
        else:
            value = (current - mean) / std

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingExtreme(BaseFactor):
    """资金费率极值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_extreme",
            name="资金费率极值",
            family=FactorFamily.FUNDING,
            description=">0.1% 返回1, <-0.05% 返回-1, 否则0",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")

        rate = funding["funding_rate"].values[-1]

        if rate > 0.001:  # >0.1%
            value = 1.0
        elif rate < -0.0005:  # <-0.05%
            value = -1.0
        else:
            value = 0.0

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingCumsum24H(BaseFactor):
    """24小时累计资金费率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_cumsum_24h",
            name="24小时累计资金费率",
            family=FactorFamily.FUNDING,
            description="最近3次结算的资金费率累计",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=3,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 3:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values[-3:]
        value = np.sum(rates)

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingCumsum7D(BaseFactor):
    """7天累计资金费率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_cumsum_7d",
            name="7天累计资金费率",
            family=FactorFamily.FUNDING,
            description="最近21次结算的资金费率累计",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=21,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 21:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values[-21:]
        value = np.sum(rates)

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class FundingReversalSignal(BaseFactor):
    """资金费率反转信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_reversal_signal",
            name="资金费率反转信号",
            family=FactorFamily.FUNDING,
            description="极端资金费率后的反转概率",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 30:
            return self.create_invalid_result(signal_time, "Insufficient funding data")

        rates = funding["funding_rate"].values[-30:]
        current = rates[-1]
        mean = np.mean(rates)
        std = np.std(rates)

        if std == 0:
            value = 0
        else:
            zscore = (current - mean) / std
            # 极端值反转信号
            if zscore > 2:
                value = -1.0  # 做空信号
            elif zscore < -2:
                value = 1.0   # 做多信号
            else:
                value = 0.0

        data_time = funding.index[-1] if hasattr(funding.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


# ===== 持仓量因子 (12个) =====

class OIChange1H(BaseFactor):
    """1小时OI变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_change_1h",
            name="1小时OI变化",
            family=FactorFamily.OPEN_INTEREST,
            description="过去1小时持仓量变化率",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
            visibility_delay=timedelta(minutes=5),
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_HOUR + 1:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values
        current = oi_values[-1]
        prev = oi_values[-BARS_PER_HOUR - 1]
        value = (current - prev) / (prev + 1e-10)

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIChange4H(BaseFactor):
    """4小时OI变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_change_4h",
            name="4小时OI变化",
            family=FactorFamily.OPEN_INTEREST,
            description="过去4小时持仓量变化率",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.B,
            visibility_delay=timedelta(minutes=5),
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        period = BARS_PER_HOUR * 4
        if oi is None or len(oi) < period + 1:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values
        current = oi_values[-1]
        prev = oi_values[-period - 1]
        value = (current - prev) / (prev + 1e-10)

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIChange24H(BaseFactor):
    """24小时OI变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_change_24h",
            name="24小时OI变化",
            family=FactorFamily.OPEN_INTEREST,
            description="过去24小时持仓量变化率",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
            visibility_delay=timedelta(minutes=5),
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values
        current = oi_values[-1]
        prev = oi_values[-BARS_PER_DAY - 1]
        value = (current - prev) / (prev + 1e-10)

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIVolumeRatio(BaseFactor):
    """OI/成交量比率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_volume_ratio",
            name="OI/成交量比率",
            family=FactorFamily.OPEN_INTEREST,
            description="持仓量 / 24小时成交量",
            data_dependencies=[DataDependency.OPEN_INTEREST, DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        klines = data.get("klines_5m")

        if oi is None or len(oi) == 0:
            return self.create_invalid_result(signal_time, "No OI data")
        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient klines data")

        current_oi = oi["open_interest"].values[-1]
        volume_24h = np.sum(klines["volume"].values[-BARS_PER_DAY:])

        if volume_24h == 0:
            value = 0
        else:
            value = current_oi / volume_24h

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIPriceDivergence(BaseFactor):
    """OI-价格背离"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_price_divergence",
            name="OI-价格背离",
            family=FactorFamily.OPEN_INTEREST,
            description="OI变化与价格变化方向不同时为1",
            data_dependencies=[DataDependency.OPEN_INTEREST, DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        klines = data.get("klines_5m")

        if oi is None or len(oi) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient OI data")
        if klines is None or len(klines) < BARS_PER_DAY + 1:
            return self.create_invalid_result(signal_time, "Insufficient klines data")

        oi_change = oi["open_interest"].values[-1] / oi["open_interest"].values[-BARS_PER_DAY - 1] - 1
        price_change = klines["close"].values[-1] / klines["close"].values[-BARS_PER_DAY - 1] - 1

        # 背离 = 符号不同
        if np.sign(oi_change) != np.sign(price_change):
            value = 1.0
        else:
            value = 0.0

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIMomentum(BaseFactor):
    """OI动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_momentum",
            name="OI动量",
            family=FactorFamily.OPEN_INTEREST,
            description="OI变化的加速度",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY * 2,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_DAY * 2 + 1:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values

        # 当期变化
        change_current = oi_values[-1] / oi_values[-BARS_PER_DAY - 1] - 1
        # 前期变化
        change_prev = oi_values[-BARS_PER_DAY - 1] / oi_values[-BARS_PER_DAY * 2 - 1] - 1

        value = change_current - change_prev

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIZScore(BaseFactor):
    """OI Z-Score"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_zscore",
            name="OI Z-Score",
            family=FactorFamily.OPEN_INTEREST,
            description="当前OI的标准化值",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_DAY * 30:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values[-(BARS_PER_DAY * 30):]
        current = oi_values[-1]
        mean = np.mean(oi_values)
        std = np.std(oi_values)

        if std == 0:
            value = 0
        else:
            value = (current - mean) / std

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIChangeRate(BaseFactor):
    """OI变化率 (MVP)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_change_rate",
            name="OI变化率",
            family=FactorFamily.OPEN_INTEREST,
            description="5分钟OI变化率",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=1,
            history_tier=HistoryTier.B,
            is_mvp=True,
            visibility_delay=timedelta(minutes=5),
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < 2:
            return self.create_invalid_result(signal_time, "Insufficient OI data")

        oi_values = oi["open_interest"].values
        value = (oi_values[-1] - oi_values[-2]) / (oi_values[-2] + 1e-10)

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class OIFundingDivergence(BaseFactor):
    """OI-Funding背离"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_funding_divergence",
            name="OI-Funding背离",
            family=FactorFamily.OPEN_INTEREST,
            description="OI变化与资金费率方向不同时为1",
            data_dependencies=[DataDependency.OPEN_INTEREST, DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
            is_mvp=True,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        funding = data.get("funding_rate")

        if oi is None or len(oi) < 2:
            return self.create_invalid_result(signal_time, "Insufficient OI data")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")

        oi_change = oi["open_interest"].values[-1] / oi["open_interest"].values[-2] - 1
        funding_rate = funding["funding_rate"].values[-1]

        # 背离 = 符号不同
        if np.sign(oi_change) != np.sign(funding_rate):
            value = 1.0
        else:
            value = 0.0

        data_time = oi.index[-1] if hasattr(oi.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


# ===== 多空博弈+CVD因子 (21个) =====

class LongShortRatio(BaseFactor):
    """多空比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="long_short_ratio",
            name="多空比",
            family=FactorFamily.ORDER_FLOW,
            description="多头账户数 / 空头账户数",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls_data = data.get("long_short_ratio")
        if ls_data is None or len(ls_data) == 0:
            return self.create_invalid_result(signal_time, "No long/short data")

        value = ls_data["long_short_ratio"].values[-1]
        data_time = ls_data.index[-1] if hasattr(ls_data.index, '__iter__') else signal_time

        return self.create_result(float(value), signal_time, data_time)


class TopTraderLongShortRatio(BaseFactor):
    """大户多空比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="top_trader_long_short_ratio",
            name="大户多空比",
            family=FactorFamily.ORDER_FLOW,
            description="大户多头 / 空头账户比",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls_data = data.get("long_short_ratio")
        if ls_data is None or len(ls_data) == 0:
            return self.create_invalid_result(signal_time, "No long/short data")

        if "top_trader_ls_ratio" in ls_data.columns:
            value = ls_data["top_trader_ls_ratio"].values[-1]
        else:
            value = ls_data["long_short_ratio"].values[-1]

        data_time = ls_data.index[-1] if hasattr(ls_data.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class TakerBuySellRatio(BaseFactor):
    """主动买卖比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_buy_sell_ratio",
            name="主动买卖比",
            family=FactorFamily.ORDER_FLOW,
            description="主动买入量 / 主动卖出量",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")

        buy_volume = np.sum(taker["taker_buy_volume"].values[-BARS_PER_HOUR:])
        sell_volume = np.sum(taker["taker_sell_volume"].values[-BARS_PER_HOUR:])

        if sell_volume == 0:
            value = 1.0 if buy_volume > 0 else 0.0
        else:
            value = buy_volume / sell_volume

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class TakerDelta(BaseFactor):
    """Taker Delta"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_delta",
            name="Taker Delta",
            family=FactorFamily.ORDER_FLOW,
            description="(主动买入 - 主动卖出) / 总成交量",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")

        buy_volume = np.sum(taker["taker_buy_volume"].values[-BARS_PER_HOUR:])
        sell_volume = np.sum(taker["taker_sell_volume"].values[-BARS_PER_HOUR:])
        total = buy_volume + sell_volume

        if total == 0:
            value = 0
        else:
            value = (buy_volume - sell_volume) / total

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class CVD(BaseFactor):
    """累计成交量差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd",
            name="累计成交量差",
            family=FactorFamily.ORDER_FLOW,
            description="Cumulative Volume Delta",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient taker data")

        buy = taker["taker_buy_volume"].values[-BARS_PER_DAY:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_DAY:]
        delta = buy - sell
        cvd = np.cumsum(delta)[-1]

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(cvd), signal_time, data_time)


class CVDChange1H(BaseFactor):
    """1小时CVD变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_change_1h",
            name="1小时CVD变化",
            family=FactorFamily.ORDER_FLOW,
            description="过去1小时的CVD变化",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")

        buy = taker["taker_buy_volume"].values[-BARS_PER_HOUR:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_HOUR:]
        delta = buy - sell
        cvd_change = np.sum(delta)

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(cvd_change), signal_time, data_time)


class CVDPriceDivergence(BaseFactor):
    """CVD-价格背离"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_price_divergence",
            name="CVD-价格背离",
            family=FactorFamily.ORDER_FLOW,
            description="CVD变化与价格变化方向不同时为1",
            data_dependencies=[DataDependency.TAKER_BUY_SELL, DataDependency.KLINES_5M],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        klines = data.get("klines_5m")

        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        if klines is None or len(klines) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient klines data")

        # CVD变化
        buy = taker["taker_buy_volume"].values[-BARS_PER_HOUR:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_HOUR:]
        cvd_change = np.sum(buy - sell)

        # 价格变化
        price_change = klines["close"].values[-1] / klines["close"].values[-BARS_PER_HOUR] - 1

        # 背离
        if np.sign(cvd_change) != np.sign(price_change):
            value = 1.0
        else:
            value = 0.0

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


class NetTakerFlow(BaseFactor):
    """净主动流"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="net_taker_flow",
            name="净主动流",
            family=FactorFamily.ORDER_FLOW,
            description="标准化的净主动买入",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient taker data")

        buy = taker["taker_buy_volume"].values[-BARS_PER_DAY:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_DAY:]
        net_flow = buy - sell

        # 标准化
        mean = np.mean(net_flow)
        std = np.std(net_flow)
        if std == 0:
            value = 0
        else:
            value = (net_flow[-1] - mean) / std

        data_time = taker.index[-1] if hasattr(taker.index, '__iter__') else signal_time
        return self.create_result(float(value), signal_time, data_time)


# ===== 追加因子: 资金费率 (2个) =====

class FundingCumsum1H(BaseFactor):
    """1小时资金费率累计 (短期)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_cumsum_1h",
            name="1小时资金费率累计",
            family=FactorFamily.FUNDING,
            description="短期资金费率累计效应",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")
        # 当前资金费率 / 1小时等效
        value = funding["funding_rate"].values[-1] / 8  # 8h费率分摊到1h
        return self.create_result(float(value), signal_time)


class FundingVolatility(BaseFactor):
    """资金费率波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_volatility",
            name="资金费率波动率",
            family=FactorFamily.FUNDING,
            description="7天资金费率标准差",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=21,  # 7天 * 3次/天
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) < 21:
            return self.create_invalid_result(signal_time, "Insufficient funding data")
        rates = funding["funding_rate"].values[-21:]
        value = np.std(rates)
        return self.create_result(float(value), signal_time)


# ===== 追加因子: 持仓量 (3个) =====

class OIConcentration(BaseFactor):
    """持仓量集中度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_concentration",
            name="持仓量集中度",
            family=FactorFamily.OPEN_INTEREST,
            description="大户持仓占比变化",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient OI data")
        oi_values = oi["open_interest"].values[-BARS_PER_DAY:]
        # 计算OI相对变化的集中度
        changes = np.diff(oi_values)
        if np.std(changes) == 0:
            value = 0
        else:
            value = np.max(np.abs(changes)) / np.std(changes)
        return self.create_result(float(value), signal_time)


class OIBreakout(BaseFactor):
    """持仓量突破"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_breakout",
            name="持仓量突破",
            family=FactorFamily.OPEN_INTEREST,
            description="持仓量创新高/新低信号",
            data_dependencies=[DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        oi = data.get("open_interest")
        if oi is None or len(oi) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient OI data")
        oi_values = oi["open_interest"].values
        current = oi_values[-1]
        max_7d = np.max(oi_values[-BARS_PER_DAY * 7:])
        min_7d = np.min(oi_values[-BARS_PER_DAY * 7:])
        if current >= max_7d:
            value = 1.0
        elif current <= min_7d:
            value = -1.0
        else:
            value = 0.0
        return self.create_result(float(value), signal_time)


class OILongShortRatio(BaseFactor):
    """持仓量多空比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="oi_long_short_ratio",
            name="持仓量多空比",
            family=FactorFamily.OPEN_INTEREST,
            description="大户多头持仓/空头持仓",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls = data.get("long_short_ratio")
        if ls is None or len(ls) == 0:
            return self.create_invalid_result(signal_time, "No L/S data")
        # 使用大户账户多空比
        if "long_account" in ls.columns and "short_account" in ls.columns:
            long_acc = ls["long_account"].values[-1]
            short_acc = ls["short_account"].values[-1]
            value = long_acc / short_acc if short_acc > 0 else 1.0
        else:
            value = ls["long_short_ratio"].values[-1] if "long_short_ratio" in ls.columns else 1.0
        return self.create_result(float(value), signal_time)


# ===== 追加因子: 订单流 (13个) =====

class TopTraderPositionRatio(BaseFactor):
    """顶级交易者持仓比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="top_trader_position_ratio",
            name="顶级交易者持仓比",
            family=FactorFamily.ORDER_FLOW,
            description="大户持仓量占总持仓比例",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls = data.get("long_short_ratio")
        if ls is None or len(ls) == 0:
            return self.create_invalid_result(signal_time, "No L/S data")
        if "top_position_ratio" in ls.columns:
            value = ls["top_position_ratio"].values[-1]
        else:
            value = 0.5  # 默认50%
        return self.create_result(float(value), signal_time)


class LSMomentum(BaseFactor):
    """多空比动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ls_momentum",
            name="多空比动量",
            family=FactorFamily.ORDER_FLOW,
            description="多空比的变化率",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls = data.get("long_short_ratio")
        if ls is None or len(ls) < 2:
            return self.create_invalid_result(signal_time, "Insufficient L/S data")
        ratios = ls["long_short_ratio"].values if "long_short_ratio" in ls.columns else ls.iloc[:, 0].values
        if len(ratios) < 2:
            return self.create_invalid_result(signal_time, "Insufficient L/S data")
        momentum = ratios[-1] / ratios[0] - 1 if ratios[0] != 0 else 0
        return self.create_result(float(momentum), signal_time)


class LSExtreme(BaseFactor):
    """多空比极端"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ls_extreme",
            name="多空比极端",
            family=FactorFamily.ORDER_FLOW,
            description="多空比是否处于极端水平",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=BARS_PER_DAY * 7,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls = data.get("long_short_ratio")
        if ls is None or len(ls) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient L/S data")
        ratios = ls["long_short_ratio"].values if "long_short_ratio" in ls.columns else ls.iloc[:, 0].values
        current = ratios[-1]
        mean = np.mean(ratios)
        std = np.std(ratios)
        if std == 0:
            value = 0
        else:
            z = (current - mean) / std
            value = 1.0 if abs(z) > 2 else 0.0
        return self.create_result(float(value), signal_time)


class LSReversalSignal(BaseFactor):
    """多空比反转信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="ls_reversal_signal",
            name="多空比反转信号",
            family=FactorFamily.ORDER_FLOW,
            description="极端多空比后的反转信号",
            data_dependencies=[DataDependency.LONG_SHORT_RATIO],
            window=BARS_PER_DAY * 7,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        ls = data.get("long_short_ratio")
        if ls is None or len(ls) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient L/S data")
        ratios = ls["long_short_ratio"].values if "long_short_ratio" in ls.columns else ls.iloc[:, 0].values
        current = ratios[-1]
        mean = np.mean(ratios)
        std = np.std(ratios)
        if std == 0:
            value = 0
        else:
            z = (current - mean) / std
            # 极端看多时看空，极端看空时看多
            if z > 2:
                value = -1.0
            elif z < -2:
                value = 1.0
            else:
                value = 0.0
        return self.create_result(float(value), signal_time)


class TakerBuyVolume(BaseFactor):
    """主动买入量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_buy_volume",
            name="主动买入量",
            family=FactorFamily.ORDER_FLOW,
            description="1小时主动买入量",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        value = np.sum(taker["taker_buy_volume"].values[-BARS_PER_HOUR:])
        return self.create_result(float(value), signal_time)


class TakerSellVolume(BaseFactor):
    """主动卖出量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_sell_volume",
            name="主动卖出量",
            family=FactorFamily.ORDER_FLOW,
            description="1小时主动卖出量",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        value = np.sum(taker["taker_sell_volume"].values[-BARS_PER_HOUR:])
        return self.create_result(float(value), signal_time)


class CVDChange4H(BaseFactor):
    """4小时CVD变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_change_4h",
            name="4小时CVD变化",
            family=FactorFamily.ORDER_FLOW,
            description="过去4小时的CVD变化",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR * 4:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_HOUR * 4:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_HOUR * 4:]
        cvd_change = np.sum(buy - sell)
        return self.create_result(float(cvd_change), signal_time)


class CVDChange24H(BaseFactor):
    """24小时CVD变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_change_24h",
            name="24小时CVD变化",
            family=FactorFamily.ORDER_FLOW,
            description="过去24小时的CVD变化",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_DAY:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_DAY:]
        cvd_change = np.sum(buy - sell)
        return self.create_result(float(cvd_change), signal_time)


class CVDMomentum(BaseFactor):
    """CVD动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_momentum",
            name="CVD动量",
            family=FactorFamily.ORDER_FLOW,
            description="CVD变化速率",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR * 4:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_HOUR * 4:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_HOUR * 4:]
        delta = buy - sell
        cvd = np.cumsum(delta)
        # 动量 = 最近1小时CVD变化 / 前3小时CVD变化
        recent = cvd[-BARS_PER_HOUR:].sum() if len(cvd) >= BARS_PER_HOUR else cvd.sum()
        past = cvd[:-BARS_PER_HOUR].sum() if len(cvd) > BARS_PER_HOUR else 1
        momentum = recent / past if past != 0 else 0
        return self.create_result(float(momentum), signal_time)


class CVDZScore(BaseFactor):
    """CVD Z-Score"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cvd_zscore",
            name="CVD Z-Score",
            family=FactorFamily.ORDER_FLOW,
            description="CVD的标准化得分",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_DAY * 7,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_DAY * 7:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_DAY * 7:]
        delta = buy - sell
        # 计算每小时CVD变化
        hourly_cvd = []
        for i in range(0, len(delta), BARS_PER_HOUR):
            hourly_cvd.append(np.sum(delta[i:i+BARS_PER_HOUR]))
        if len(hourly_cvd) < 2:
            return self.create_invalid_result(signal_time, "Insufficient data for zscore")
        current = hourly_cvd[-1]
        mean = np.mean(hourly_cvd)
        std = np.std(hourly_cvd)
        if std == 0:
            value = 0
        else:
            value = (current - mean) / std
        return self.create_result(float(value), signal_time)


class TakerFlowImbalance(BaseFactor):
    """主动流不平衡"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_flow_imbalance",
            name="主动流不平衡",
            family=FactorFamily.ORDER_FLOW,
            description="买卖主动流的不平衡程度",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = np.sum(taker["taker_buy_volume"].values[-BARS_PER_HOUR:])
        sell = np.sum(taker["taker_sell_volume"].values[-BARS_PER_HOUR:])
        total = buy + sell
        if total == 0:
            value = 0
        else:
            value = abs(buy - sell) / total  # 0-1范围
        return self.create_result(float(value), signal_time)


class AggressiveBuyRatio(BaseFactor):
    """激进买入比例"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="aggressive_buy_ratio",
            name="激进买入比例",
            family=FactorFamily.ORDER_FLOW,
            description="大额主动买入占比",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_HOUR:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_HOUR:]
        # 大额定义为超过均值1.5倍的bar
        mean_buy = np.mean(buy)
        aggressive = buy[buy > mean_buy * 1.5].sum()
        total = buy.sum()
        value = aggressive / total if total > 0 else 0
        return self.create_result(float(value), signal_time)


class TakerVolumeSpike(BaseFactor):
    """主动成交量激增"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="taker_volume_spike",
            name="主动成交量激增",
            family=FactorFamily.ORDER_FLOW,
            description="主动成交量是否激增",
            data_dependencies=[DataDependency.TAKER_BUY_SELL],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        taker = data.get("taker_buy_sell")
        if taker is None or len(taker) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient taker data")
        buy = taker["taker_buy_volume"].values[-BARS_PER_DAY:]
        sell = taker["taker_sell_volume"].values[-BARS_PER_DAY:]
        total = buy + sell
        current = total[-BARS_PER_HOUR:].sum()
        avg = np.mean([total[i:i+BARS_PER_HOUR].sum() for i in range(0, len(total)-BARS_PER_HOUR, BARS_PER_HOUR)])
        value = 1.0 if current > avg * 2 else 0.0
        return self.create_result(float(value), signal_time)


class FundingAnnualized(BaseFactor):
    """年化资金费率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="funding_annualized",
            name="年化资金费率",
            family=FactorFamily.FUNDING,
            description="资金费率年化 (8h费率 * 3 * 365)",
            data_dependencies=[DataDependency.FUNDING_RATE],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        funding = data.get("funding_rate")
        if funding is None or len(funding) == 0:
            return self.create_invalid_result(signal_time, "No funding data")
        rate = funding["funding_rate"].values[-1]
        # 年化: 8小时费率 * 每天3次 * 365天
        annualized = rate * 3 * 365 * 100  # 百分比
        return self.create_result(float(annualized), signal_time)


# 因子注册表
PERPETUAL_FACTORS = [
    # 资金费率 (13个)
    FundingRate,
    FundingRateMA8H,
    FundingRateMA24H,
    FundingPremium,
    FundingMomentum,
    FundingZScore,
    FundingExtreme,
    FundingCumsum24H,
    FundingCumsum7D,
    FundingReversalSignal,
    FundingCumsum1H,
    FundingVolatility,
    FundingAnnualized,
    # 持仓量 (12个)
    OIChange1H,
    OIChange4H,
    OIChange24H,
    OIVolumeRatio,
    OIPriceDivergence,
    OIMomentum,
    OIZScore,
    OIChangeRate,
    OIFundingDivergence,
    OIConcentration,
    OIBreakout,
    OILongShortRatio,
    # 订单流 (21个)
    LongShortRatio,
    TopTraderLongShortRatio,
    TopTraderPositionRatio,
    LSMomentum,
    LSExtreme,
    LSReversalSignal,
    TakerBuyVolume,
    TakerSellVolume,
    TakerBuySellRatio,
    TakerDelta,
    TakerFlowImbalance,
    AggressiveBuyRatio,
    TakerVolumeSpike,
    CVD,
    CVDChange1H,
    CVDChange4H,
    CVDChange24H,
    CVDMomentum,
    CVDZScore,
    CVDPriceDivergence,
    NetTakerFlow,
]


def get_perpetual_factors() -> List[BaseFactor]:
    """获取所有永续合约因子实例"""
    return [f() for f in PERPETUAL_FACTORS]
