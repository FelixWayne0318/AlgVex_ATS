"""
复合/ML因子 (15个)

基于多因子组合和统计模型的复合因子。

动量组合因子 (5个):
1. AlphaMomentum - 综合动量因子
2. CrossAssetMomentum - 跨资产动量
3. MomentumQuality - 动量质量
4. TrendFollowing - 趋势跟踪信号
5. MeanReversion - 均值回归信号

风险/市场状态因子 (5个):
6. RegimeIndicator - 市场状态指标
7. VolatilityRegime - 波动率状态
8. LiquidityScore - 流动性评分
9. CrowdingFactor - 拥挤度因子
10. MarketStress - 市场压力指数

复合信号因子 (5个):
11. CompositeSignal - 综合信号
12. FactorMomentum - 因子动量
13. AlphaDecay - Alpha衰减
14. SignalStrength - 信号强度
15. ConfidenceScore - 置信度评分
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier, sma, rolling_std, zscore,
)


BARS_PER_HOUR = 12
BARS_PER_DAY = 288


class AlphaMomentum(BaseFactor):
    """综合动量因子"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="alpha_momentum",
            name="综合动量",
            family=FactorFamily.COMPOSITE,
            description="多时间尺度动量的加权组合",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 多时间尺度动量
        mom_1d = close[-1] / close[-BARS_PER_DAY] - 1 if len(close) > BARS_PER_DAY else 0
        mom_3d = close[-1] / close[-BARS_PER_DAY*3] - 1 if len(close) > BARS_PER_DAY*3 else 0
        mom_7d = close[-1] / close[-BARS_PER_DAY*7] - 1 if len(close) > BARS_PER_DAY*7 else 0

        # 波动率调整权重
        vol = np.std(np.diff(np.log(close[-BARS_PER_DAY:])))
        vol = max(vol, 0.001)  # 避免除零

        # 加权组合 (短期权重更高)
        alpha = (mom_1d * 0.5 + mom_3d * 0.3 + mom_7d * 0.2) / vol

        return self.create_result(float(alpha), signal_time)


class CrossAssetMomentum(BaseFactor):
    """跨资产动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cross_asset_momentum",
            name="跨资产动量",
            family=FactorFamily.COMPOSITE,
            description="BTC相对于其他资产的相对动量",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.SPX],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        btc_data = data.get("klines_5m")
        macro_data = data.get("macro")

        if btc_data is None or len(btc_data) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient BTC data")

        btc_close = btc_data["close"].values
        btc_mom = btc_close[-1] / btc_close[-BARS_PER_DAY*7] - 1

        # 与其他资产比较
        cross_mom = btc_mom  # 默认

        if macro_data is not None and len(macro_data) >= 7:
            for col in ["spx", "SPX"]:
                if col in macro_data.columns:
                    spx = macro_data[col].values
                    spx_mom = spx[-1] / spx[-7] - 1 if len(spx) >= 7 else 0
                    cross_mom = btc_mom - spx_mom
                    break

        return self.create_result(float(cross_mom), signal_time)


class TrendConsistency(BaseFactor):
    """趋势一致性 (原MomentumQuality重命名以避免冲突)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="trend_consistency",
            name="趋势一致性",
            family=FactorFamily.COMPOSITE,
            description="多时间尺度动量方向的一致性评分",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 计算多个时间尺度的动量方向
        mom_1h = np.sign(close[-1] - close[-BARS_PER_HOUR])
        mom_4h = np.sign(close[-1] - close[-BARS_PER_HOUR*4])
        mom_1d = np.sign(close[-1] - close[-BARS_PER_DAY])
        mom_3d = np.sign(close[-1] - close[-BARS_PER_DAY*3])
        mom_7d = np.sign(close[-1] - close[-BARS_PER_DAY*7])

        # 质量 = 方向一致性
        directions = [mom_1h, mom_4h, mom_1d, mom_3d, mom_7d]
        quality = abs(sum(directions)) / len(directions)

        return self.create_result(float(quality), signal_time)


class TrendFollowing(BaseFactor):
    """趋势跟踪信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="trend_following",
            name="趋势跟踪",
            family=FactorFamily.COMPOSITE,
            description="基于均线交叉的趋势信号",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 50,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 50:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 多重均线系统
        ma_short = np.mean(close[-BARS_PER_DAY*5:])
        ma_medium = np.mean(close[-BARS_PER_DAY*20:])
        ma_long = np.mean(close[-BARS_PER_DAY*50:])

        # 趋势信号
        short_trend = 1 if ma_short > ma_medium else -1
        long_trend = 1 if ma_medium > ma_long else -1
        price_above_ma = 1 if close[-1] > ma_short else -1

        trend_signal = (short_trend + long_trend + price_above_ma) / 3

        return self.create_result(float(trend_signal), signal_time)


class MeanReversion(BaseFactor):
    """均值回归信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="mean_reversion",
            name="均值回归",
            family=FactorFamily.COMPOSITE,
            description="基于布林带的均值回归信号",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 20:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 布林带
        ma = np.mean(close[-BARS_PER_DAY*20:])
        std = np.std(close[-BARS_PER_DAY*20:])

        upper = ma + 2 * std
        lower = ma - 2 * std

        current = close[-1]

        # 均值回归信号
        if current > upper:
            signal = -1.0  # 超买，预期回落
        elif current < lower:
            signal = 1.0   # 超卖，预期反弹
        else:
            # 在通道内，信号强度与偏离程度成正比
            signal = -(current - ma) / (2 * std) if std > 0 else 0

        return self.create_result(float(signal), signal_time)


class RegimeIndicator(BaseFactor):
    """市场状态指标"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="regime_indicator",
            name="市场状态",
            family=FactorFamily.COMPOSITE,
            description="趋势(+1)/震荡(-1)/转换(0)状态判断",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 14:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        high = klines["high"].values
        low = klines["low"].values

        # ADX判断趋势强度
        tr = np.maximum(high[-14:] - low[-14:],
                       np.abs(high[-14:] - close[-15:-1]),
                       np.abs(low[-14:] - close[-15:-1]))
        atr = np.mean(tr)

        # 方向性运动
        price_range = np.max(close[-14:]) - np.min(close[-14:])
        direction = abs(close[-1] - close[-14]) / atr if atr > 0 else 0

        # 判断状态
        if direction > 1.5:
            regime = 1.0   # 趋势
        elif direction < 0.5:
            regime = -1.0  # 震荡
        else:
            regime = 0.0   # 转换期

        return self.create_result(float(regime), signal_time)


class VolatilityRegime(BaseFactor):
    """波动率状态"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volatility_regime",
            name="波动率状态",
            family=FactorFamily.COMPOSITE,
            description="高波动(+1)/低波动(-1)状态",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 90,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        returns = np.diff(np.log(close))

        # 短期vs长期波动率
        vol_short = np.std(returns[-BARS_PER_DAY*7:]) * np.sqrt(BARS_PER_DAY * 365)
        vol_long = np.std(returns[-BARS_PER_DAY*30:]) * np.sqrt(BARS_PER_DAY * 365)

        ratio = vol_short / vol_long if vol_long > 0 else 1.0

        # 状态判断
        if ratio > 1.3:
            regime = 1.0   # 高波动
        elif ratio < 0.7:
            regime = -1.0  # 低波动
        else:
            regime = (ratio - 1.0) / 0.3  # 线性插值

        return self.create_result(float(regime), signal_time)


class LiquidityScore(BaseFactor):
    """流动性评分"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidity_score",
            name="流动性评分",
            family=FactorFamily.COMPOSITE,
            description="基于成交量和价差的流动性评分 (0-1)",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.ORDER_BOOK],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient data")

        volume = klines["volume"].values

        # 成交量相对水平
        current_vol = volume[-1]
        avg_vol = np.mean(volume[-BARS_PER_DAY:])
        vol_score = min(current_vol / avg_vol, 2.0) / 2.0 if avg_vol > 0 else 0.5

        # 如果有订单簿数据，加入价差评分
        orderbook = data.get("order_book")
        if orderbook is not None:
            if "spread" in orderbook.columns:
                spread = orderbook["spread"].iloc[-1]
                spread_score = max(0, 1 - spread * 1000)  # 假设spread以百分比表示
                score = (vol_score + spread_score) / 2
            else:
                score = vol_score
        else:
            score = vol_score

        return self.create_result(float(score), signal_time)


class CrowdingFactor(BaseFactor):
    """拥挤度因子"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="crowding_factor",
            name="拥挤度",
            family=FactorFamily.COMPOSITE,
            description="交易拥挤程度 (基于持仓量和成交量)",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.OPEN_INTEREST],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        oi_data = data.get("open_interest")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient klines")

        volume = klines["volume"].values

        # 成交量异常
        vol_ma = np.mean(volume[-BARS_PER_DAY*7:])
        vol_std = np.std(volume[-BARS_PER_DAY*7:])
        vol_zscore = (volume[-1] - vol_ma) / vol_std if vol_std > 0 else 0

        # 持仓量异常
        oi_zscore = 0
        if oi_data is not None and len(oi_data) >= 7:
            if "open_interest" in oi_data.columns:
                oi = oi_data["open_interest"].values
                oi_ma = np.mean(oi[-7:])
                oi_std = np.std(oi[-7:])
                oi_zscore = (oi[-1] - oi_ma) / oi_std if oi_std > 0 else 0

        # 拥挤度 = 综合异常程度
        crowding = (abs(vol_zscore) + abs(oi_zscore)) / 2

        return self.create_result(float(crowding), signal_time)


class MarketStress(BaseFactor):
    """市场压力指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="market_stress",
            name="市场压力",
            family=FactorFamily.COMPOSITE,
            description="综合市场压力指数 (0=正常, 1=极端)",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.FUNDING_RATE],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        volume = klines["volume"].values

        # 波动率压力
        returns = np.diff(np.log(close[-BARS_PER_DAY*7:]))
        vol = np.std(returns) * np.sqrt(BARS_PER_DAY * 365)
        vol_hist = np.std(np.diff(np.log(close))) * np.sqrt(BARS_PER_DAY * 365)
        vol_stress = min(vol / max(vol_hist, 0.1), 3) / 3

        # 成交量压力
        vol_ma = np.mean(volume[-BARS_PER_DAY*7:])
        vol_spike = min(volume[-1] / max(vol_ma, 1), 5) / 5

        # 价格压力 (跌幅)
        drawdown = 1 - close[-1] / np.max(close[-BARS_PER_DAY*7:])
        price_stress = min(drawdown * 10, 1)

        # 综合压力
        stress = (vol_stress + vol_spike + price_stress) / 3

        return self.create_result(float(stress), signal_time)


class CompositeSignal(BaseFactor):
    """综合信号"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="composite_signal",
            name="综合信号",
            family=FactorFamily.COMPOSITE,
            description="多因子加权综合信号 (-1到+1)",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.FUNDING_RATE],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        volume = klines["volume"].values

        # 动量信号
        mom = close[-1] / close[-BARS_PER_DAY] - 1
        mom_signal = np.tanh(mom * 20)  # 归一化

        # 均值回归信号
        ma = np.mean(close[-BARS_PER_DAY*20:]) if len(close) >= BARS_PER_DAY*20 else np.mean(close)
        deviation = (close[-1] - ma) / ma
        mr_signal = -np.tanh(deviation * 10)

        # 成交量确认
        vol_ratio = volume[-1] / np.mean(volume[-BARS_PER_DAY:])
        vol_confirm = min(vol_ratio, 2) / 2

        # 综合
        signal = (mom_signal * 0.4 + mr_signal * 0.3) * (0.5 + vol_confirm * 0.5)

        return self.create_result(float(signal), signal_time)


class FactorMomentum(BaseFactor):
    """因子动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="factor_momentum",
            name="因子动量",
            family=FactorFamily.COMPOSITE,
            description="因子表现的动量 (基于历史因子IC)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 60,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")
        factor_perf = data.get("factor_performance")

        if klines is None:
            return self.create_invalid_result(signal_time, "Missing klines")

        # 如果有因子表现数据
        if factor_perf is not None and "ic" in factor_perf.columns:
            recent_ic = factor_perf["ic"].iloc[-30:].mean()
            past_ic = factor_perf["ic"].iloc[-60:-30].mean()
            momentum = recent_ic - past_ic
            return self.create_result(float(momentum), signal_time)

        # 否则使用价格动量作为代理
        if len(klines) >= BARS_PER_DAY * 60:
            close = klines["close"].values
            recent_ret = close[-1] / close[-BARS_PER_DAY*30] - 1
            past_ret = close[-BARS_PER_DAY*30] / close[-BARS_PER_DAY*60] - 1
            momentum = recent_ret - past_ret
            return self.create_result(float(momentum), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient data")


class AlphaDecay(BaseFactor):
    """Alpha衰减"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="alpha_decay",
            name="Alpha衰减",
            family=FactorFamily.COMPOSITE,
            description="信号有效性的衰减速度指标",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values

        # 自相关作为衰减代理
        returns = np.diff(np.log(close[-BARS_PER_DAY*7:]))

        if len(returns) < 10:
            return self.create_invalid_result(signal_time, "Too few returns")

        # 计算滞后1-5的自相关
        autocorrs = []
        for lag in range(1, 6):
            if len(returns) > lag:
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))

        decay = np.mean(autocorrs) if autocorrs else 0

        return self.create_result(float(decay), signal_time)


class SignalStrength(BaseFactor):
    """信号强度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="signal_strength",
            name="信号强度",
            family=FactorFamily.COMPOSITE,
            description="当前交易信号的强度 (0-1)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 20,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None or len(klines) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        close = klines["close"].values
        volume = klines["volume"].values

        # 价格动量强度
        mom = abs(close[-1] / close[-BARS_PER_DAY] - 1)
        mom_strength = min(mom * 20, 1)

        # 成交量确认
        vol_ratio = volume[-1] / np.mean(volume[-BARS_PER_DAY:])
        vol_strength = min(vol_ratio / 2, 1)

        # 趋势一致性
        short_trend = np.sign(close[-1] - close[-BARS_PER_HOUR])
        mid_trend = np.sign(close[-1] - close[-BARS_PER_DAY])
        consistency = 1 if short_trend == mid_trend else 0.5

        strength = (mom_strength * 0.4 + vol_strength * 0.3 + consistency * 0.3)

        return self.create_result(float(strength), signal_time)


class ConfidenceScore(BaseFactor):
    """置信度评分"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="confidence_score",
            name="置信度评分",
            family=FactorFamily.COMPOSITE,
            description="综合置信度 (数据质量+信号一致性)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        klines = data.get("klines_5m")

        if klines is None:
            return self.create_invalid_result(signal_time, "Missing data")

        # 数据质量评分
        data_quality = 1.0
        if len(klines) < BARS_PER_DAY * 7:
            data_quality *= 0.5

        # 检查缺失值
        if klines.isnull().any().any():
            missing_ratio = klines.isnull().sum().sum() / klines.size
            data_quality *= (1 - missing_ratio)

        # 信号一致性
        close = klines["close"].values
        consistency = 0.5  # 默认

        if len(close) >= BARS_PER_DAY * 7:
            trends = [
                np.sign(close[-1] - close[-BARS_PER_HOUR]),
                np.sign(close[-1] - close[-BARS_PER_HOUR*4]),
                np.sign(close[-1] - close[-BARS_PER_DAY]),
                np.sign(close[-1] - close[-BARS_PER_DAY*3]),
                np.sign(close[-1] - close[-BARS_PER_DAY*7]),
            ]
            consistency = abs(sum(trends)) / len(trends)

        confidence = data_quality * 0.5 + consistency * 0.5

        return self.create_result(float(confidence), signal_time)


# 导出所有因子
COMPOSITE_FACTORS = [
    AlphaMomentum,
    CrossAssetMomentum,
    TrendConsistency,
    TrendFollowing,
    MeanReversion,
    RegimeIndicator,
    VolatilityRegime,
    LiquidityScore,
    CrowdingFactor,
    MarketStress,
    CompositeSignal,
    FactorMomentum,
    AlphaDecay,
    SignalStrength,
    ConfidenceScore,
]
