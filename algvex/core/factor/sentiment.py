"""
情绪因子 (10个)

基于Fear & Greed指数和Google Trends的情绪因子。

Fear & Greed因子 (5个):
1. FearGreedIndex - 恐惧贪婪指数
2. FearGreedMA7D - 7天均线
3. FearGreedMomentum - 指数动量
4. FearGreedExtreme - 极端值指标
5. FearGreedReversal - 反转信号

Google Trends因子 (5个):
6. BTCSearchTrend - BTC搜索趋势
7. CryptoSearchTrend - Crypto搜索趋势
8. SearchTrendMomentum - 搜索动量
9. SearchTrendSpike - 搜索激增
10. SearchTrendReversal - 搜索反转
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
)


class FearGreedIndex(BaseFactor):
    """恐惧贪婪指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="fear_greed_index",
            name="恐惧贪婪指数",
            family=FactorFamily.SENTIMENT,
            description="Alternative.me Fear & Greed指数 (0-100)",
            data_dependencies=[DataDependency.FEAR_GREED],
            window=1,
            history_tier=HistoryTier.A,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        fg_data = data.get("fear_greed")

        if fg_data is None or fg_data.empty:
            return self.create_invalid_result(signal_time, "Missing Fear & Greed data")

        if "value" in fg_data.columns:
            value = fg_data["value"].iloc[-1]
            return self.create_result(float(value), signal_time)

        if "fear_greed" in fg_data.columns:
            value = fg_data["fear_greed"].iloc[-1]
            return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing value column")


class FearGreedMA7D(BaseFactor):
    """恐惧贪婪7天均线"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="fear_greed_ma_7d",
            name="恐惧贪婪7天均线",
            family=FactorFamily.SENTIMENT,
            description="Fear & Greed指数的7天移动平均",
            data_dependencies=[DataDependency.FEAR_GREED],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        fg_data = data.get("fear_greed")

        if fg_data is None or len(fg_data) < 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = "value" if "value" in fg_data.columns else "fear_greed"
        if col not in fg_data.columns:
            return self.create_invalid_result(signal_time, "Missing column")

        ma7 = np.mean(fg_data[col].values[-7:])
        return self.create_result(float(ma7), signal_time)


class FearGreedMomentum(BaseFactor):
    """恐惧贪婪动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="fear_greed_momentum",
            name="恐惧贪婪动量",
            family=FactorFamily.SENTIMENT,
            description="Fear & Greed指数的短期变化趋势",
            data_dependencies=[DataDependency.FEAR_GREED],
            window=14,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        fg_data = data.get("fear_greed")

        if fg_data is None or len(fg_data) < 14:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = "value" if "value" in fg_data.columns else "fear_greed"
        if col not in fg_data.columns:
            return self.create_invalid_result(signal_time, "Missing column")

        values = fg_data[col].values
        ma_short = np.mean(values[-3:])
        ma_long = np.mean(values[-14:])
        momentum = (ma_short / ma_long - 1) if ma_long > 0 else 0

        return self.create_result(float(momentum), signal_time)


class FearGreedExtreme(BaseFactor):
    """恐惧贪婪极端值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="fear_greed_extreme",
            name="恐惧贪婪极端值",
            family=FactorFamily.SENTIMENT,
            description="是否处于极端恐惧(<20)或极端贪婪(>80)",
            data_dependencies=[DataDependency.FEAR_GREED],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        fg_data = data.get("fear_greed")

        if fg_data is None or fg_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        col = "value" if "value" in fg_data.columns else "fear_greed"
        if col not in fg_data.columns:
            return self.create_invalid_result(signal_time, "Missing column")

        value = fg_data[col].iloc[-1]

        if value < 20:
            extreme = -1.0  # 极端恐惧
        elif value > 80:
            extreme = 1.0   # 极端贪婪
        else:
            extreme = 0.0   # 中性

        return self.create_result(float(extreme), signal_time)


class FearGreedReversal(BaseFactor):
    """恐惧贪婪反转信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="fear_greed_reversal",
            name="恐惧贪婪反转",
            family=FactorFamily.SENTIMENT,
            description="从极端值反转的信号 (-1=从恐惧反转, +1=从贪婪反转)",
            data_dependencies=[DataDependency.FEAR_GREED],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        fg_data = data.get("fear_greed")

        if fg_data is None or len(fg_data) < 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = "value" if "value" in fg_data.columns else "fear_greed"
        if col not in fg_data.columns:
            return self.create_invalid_result(signal_time, "Missing column")

        values = fg_data[col].values
        current = values[-1]
        min_7d = np.min(values[-7:])
        max_7d = np.max(values[-7:])

        # 从极端恐惧反弹
        if min_7d < 20 and current > min_7d + 10:
            reversal = -1.0  # 买入信号
        # 从极端贪婪回落
        elif max_7d > 80 and current < max_7d - 10:
            reversal = 1.0   # 卖出信号
        else:
            reversal = 0.0

        return self.create_result(float(reversal), signal_time)


class BTCSearchTrend(BaseFactor):
    """BTC搜索趋势"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="btc_search_trend",
            name="BTC搜索趋势",
            family=FactorFamily.SENTIMENT,
            description="Google Trends 'Bitcoin'搜索热度 (0-100)",
            data_dependencies=[DataDependency.GOOGLE_TRENDS],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        trends_data = data.get("google_trends")

        if trends_data is None or trends_data.empty:
            return self.create_invalid_result(signal_time, "Missing Google Trends data")

        if "bitcoin" in trends_data.columns:
            value = trends_data["bitcoin"].iloc[-1]
            return self.create_result(float(value), signal_time)

        if "btc" in trends_data.columns:
            value = trends_data["btc"].iloc[-1]
            return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing bitcoin column")


class CryptoSearchTrend(BaseFactor):
    """Crypto搜索趋势"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="crypto_search_trend",
            name="Crypto搜索趋势",
            family=FactorFamily.SENTIMENT,
            description="Google Trends 'Cryptocurrency'搜索热度",
            data_dependencies=[DataDependency.GOOGLE_TRENDS],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        trends_data = data.get("google_trends")

        if trends_data is None or trends_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        for col in ["cryptocurrency", "crypto", "加密货币"]:
            if col in trends_data.columns:
                value = trends_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing crypto column")


class SearchTrendMomentum(BaseFactor):
    """搜索趋势动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="search_trend_momentum",
            name="搜索趋势动量",
            family=FactorFamily.SENTIMENT,
            description="BTC搜索热度的周变化率",
            data_dependencies=[DataDependency.GOOGLE_TRENDS],
            window=14,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        trends_data = data.get("google_trends")

        if trends_data is None or len(trends_data) < 14:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = None
        for c in ["bitcoin", "btc", "crypto"]:
            if c in trends_data.columns:
                col = c
                break

        if col is None:
            return self.create_invalid_result(signal_time, "Missing column")

        values = trends_data[col].values
        ma_short = np.mean(values[-7:])
        ma_long = np.mean(values[-14:])
        momentum = (ma_short / ma_long - 1) if ma_long > 0 else 0

        return self.create_result(float(momentum), signal_time)


class SearchTrendSpike(BaseFactor):
    """搜索趋势激增"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="search_trend_spike",
            name="搜索趋势激增",
            family=FactorFamily.SENTIMENT,
            description="搜索热度相对均值的Z分数",
            data_dependencies=[DataDependency.GOOGLE_TRENDS],
            window=30,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        trends_data = data.get("google_trends")

        if trends_data is None or len(trends_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = None
        for c in ["bitcoin", "btc", "crypto"]:
            if c in trends_data.columns:
                col = c
                break

        if col is None:
            return self.create_invalid_result(signal_time, "Missing column")

        values = trends_data[col].values
        current = values[-1]
        mean = np.mean(values[-30:])
        std = np.std(values[-30:])
        zscore = (current - mean) / std if std > 0 else 0

        return self.create_result(float(zscore), signal_time)


class SearchTrendReversal(BaseFactor):
    """搜索趋势反转"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="search_trend_reversal",
            name="搜索趋势反转",
            family=FactorFamily.SENTIMENT,
            description="搜索热度从极值反转的信号",
            data_dependencies=[DataDependency.GOOGLE_TRENDS],
            window=30,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        trends_data = data.get("google_trends")

        if trends_data is None or len(trends_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = None
        for c in ["bitcoin", "btc", "crypto"]:
            if c in trends_data.columns:
                col = c
                break

        if col is None:
            return self.create_invalid_result(signal_time, "Missing column")

        values = trends_data[col].values
        current = values[-1]
        p10 = np.percentile(values[-30:], 10)
        p90 = np.percentile(values[-30:], 90)

        # 从低点反弹
        if current > p10 and np.min(values[-7:]) <= p10:
            reversal = 1.0  # 可能见底
        # 从高点回落
        elif current < p90 and np.max(values[-7:]) >= p90:
            reversal = -1.0  # 可能见顶
        else:
            reversal = 0.0

        return self.create_result(float(reversal), signal_time)


# 导出所有因子
SENTIMENT_FACTORS = [
    FearGreedIndex,
    FearGreedMA7D,
    FearGreedMomentum,
    FearGreedExtreme,
    FearGreedReversal,
    BTCSearchTrend,
    CryptoSearchTrend,
    SearchTrendMomentum,
    SearchTrendSpike,
    SearchTrendReversal,
]
