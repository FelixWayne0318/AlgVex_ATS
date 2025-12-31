"""
宏观因子 (15个)

基于宏观经济指标的因子。

美元/利率因子 (5个):
1. DXY - 美元指数
2. DXYChange - 美元指数变化
3. US10Y - 10年期国债收益率
4. US2Y10YSpread - 2年-10年利差
5. RealYield - 实际收益率

股市/风险因子 (5个):
6. SPX - 标普500
7. SPXChange - 标普500变化
8. VIX - 波动率指数
9. VIXChange - VIX变化
10. RiskOnOff - 风险偏好指标

商品/相关因子 (5个):
11. Gold - 黄金价格
12. GoldBTCRatio - 黄金/BTC比值
13. BTCSPXCorr - BTC与SPX相关性
14. BTCDXYCorr - BTC与DXY相关性
15. MacroMomentum - 宏观动量
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
)


BARS_PER_DAY = 288


class DXY(BaseFactor):
    """美元指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="dxy",
            name="美元指数",
            family=FactorFamily.MACRO,
            description="美元指数 (DXY)",
            data_dependencies=[DataDependency.DXY],
            window=1,
            history_tier=HistoryTier.A,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        macro_data = data.get("macro") or data.get("dxy")

        if macro_data is None or macro_data.empty:
            return self.create_invalid_result(signal_time, "Missing macro data")

        if "dxy" in macro_data.columns:
            value = macro_data["dxy"].iloc[-1]
            return self.create_result(float(value), signal_time)

        if "close" in macro_data.columns:
            value = macro_data["close"].iloc[-1]
            return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing DXY column")


class DXYChange(BaseFactor):
    """美元指数变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="dxy_change",
            name="美元指数变化",
            family=FactorFamily.MACRO,
            description="美元指数的周变化率",
            data_dependencies=[DataDependency.DXY],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        macro_data = data.get("macro") or data.get("dxy")

        if macro_data is None or len(macro_data) < 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = "dxy" if "dxy" in macro_data.columns else "close"
        if col not in macro_data.columns:
            return self.create_invalid_result(signal_time, "Missing column")

        current = macro_data[col].iloc[-1]
        prev = macro_data[col].iloc[-7]
        change = (current - prev) / prev if prev > 0 else 0

        return self.create_result(float(change), signal_time)


class US10Y(BaseFactor):
    """10年期国债收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="us10y",
            name="10年期国债收益率",
            family=FactorFamily.MACRO,
            description="美国10年期国债收益率",
            data_dependencies=[DataDependency.TREASURY],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        treasury_data = data.get("treasury") or data.get("macro")

        if treasury_data is None or treasury_data.empty:
            return self.create_invalid_result(signal_time, "Missing treasury data")

        for col in ["us10y", "US10Y", "10y", "yield_10y"]:
            if col in treasury_data.columns:
                value = treasury_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing US10Y column")


class US2Y10YSpread(BaseFactor):
    """2年-10年利差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="us2y10y_spread",
            name="2年-10年利差",
            family=FactorFamily.MACRO,
            description="美国10年期与2年期国债利差 (倒挂指标)",
            data_dependencies=[DataDependency.TREASURY],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        treasury_data = data.get("treasury") or data.get("macro")

        if treasury_data is None or treasury_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        us10y = None
        us2y = None

        for col in ["us10y", "US10Y", "10y", "yield_10y"]:
            if col in treasury_data.columns:
                us10y = treasury_data[col].iloc[-1]
                break

        for col in ["us2y", "US2Y", "2y", "yield_2y"]:
            if col in treasury_data.columns:
                us2y = treasury_data[col].iloc[-1]
                break

        if us10y is not None and us2y is not None:
            spread = us10y - us2y
            return self.create_result(float(spread), signal_time)

        return self.create_invalid_result(signal_time, "Missing treasury data")


class RealYield(BaseFactor):
    """实际收益率 (名义利率 - 通胀预期)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="real_yield",
            name="实际收益率",
            family=FactorFamily.MACRO,
            description="10年期实际收益率 (TIPS)",
            data_dependencies=[DataDependency.TREASURY],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        treasury_data = data.get("treasury") or data.get("macro")

        if treasury_data is None or treasury_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        # 直接使用TIPS收益率
        for col in ["real_yield", "tips_10y", "TIPS10Y"]:
            if col in treasury_data.columns:
                value = treasury_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        # 或者用名义利率减去预期通胀
        us10y = None
        inflation = None

        for col in ["us10y", "US10Y", "10y"]:
            if col in treasury_data.columns:
                us10y = treasury_data[col].iloc[-1]
                break

        for col in ["breakeven", "inflation_expectation", "bei"]:
            if col in treasury_data.columns:
                inflation = treasury_data[col].iloc[-1]
                break

        if us10y is not None and inflation is not None:
            real = us10y - inflation
            return self.create_result(float(real), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


class SPX(BaseFactor):
    """标普500指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="spx",
            name="标普500",
            family=FactorFamily.MACRO,
            description="标普500指数水平",
            data_dependencies=[DataDependency.SPX],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        spx_data = data.get("spx") or data.get("macro")

        if spx_data is None or spx_data.empty:
            return self.create_invalid_result(signal_time, "Missing SPX data")

        for col in ["spx", "SPX", "close", "sp500"]:
            if col in spx_data.columns:
                value = spx_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing SPX column")


class SPXChange(BaseFactor):
    """标普500变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="spx_change",
            name="标普500变化",
            family=FactorFamily.MACRO,
            description="标普500的周变化率",
            data_dependencies=[DataDependency.SPX],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        spx_data = data.get("spx") or data.get("macro")

        if spx_data is None or len(spx_data) < 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = None
        for c in ["spx", "SPX", "close", "sp500"]:
            if c in spx_data.columns:
                col = c
                break

        if col is None:
            return self.create_invalid_result(signal_time, "Missing column")

        current = spx_data[col].iloc[-1]
        prev = spx_data[col].iloc[-7]
        change = (current - prev) / prev if prev > 0 else 0

        return self.create_result(float(change), signal_time)


class VIX(BaseFactor):
    """VIX波动率指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vix",
            name="VIX指数",
            family=FactorFamily.MACRO,
            description="CBOE VIX波动率指数",
            data_dependencies=[DataDependency.SPX],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vix_data = data.get("vix") or data.get("macro")

        if vix_data is None or vix_data.empty:
            return self.create_invalid_result(signal_time, "Missing VIX data")

        for col in ["vix", "VIX", "close"]:
            if col in vix_data.columns:
                value = vix_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing VIX column")


class VIXChange(BaseFactor):
    """VIX变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vix_change",
            name="VIX变化",
            family=FactorFamily.MACRO,
            description="VIX的周变化率",
            data_dependencies=[DataDependency.SPX],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vix_data = data.get("vix") or data.get("macro")

        if vix_data is None or len(vix_data) < 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        col = None
        for c in ["vix", "VIX", "close"]:
            if c in vix_data.columns:
                col = c
                break

        if col is None:
            return self.create_invalid_result(signal_time, "Missing column")

        current = vix_data[col].iloc[-1]
        prev = vix_data[col].iloc[-7]
        change = (current - prev) / prev if prev > 0 else 0

        return self.create_result(float(change), signal_time)


class RiskOnOff(BaseFactor):
    """风险偏好指标"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="risk_on_off",
            name="风险偏好",
            family=FactorFamily.MACRO,
            description="综合风险偏好指标 (VIX + 利差)",
            data_dependencies=[DataDependency.SPX, DataDependency.TREASURY],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vix_data = data.get("vix") or data.get("macro")

        if vix_data is None or vix_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        vix = None
        for col in ["vix", "VIX", "close"]:
            if col in vix_data.columns:
                vix = vix_data[col].iloc[-1]
                break

        if vix is None:
            return self.create_invalid_result(signal_time, "Missing VIX")

        # 简化版: 基于VIX判断
        # VIX < 15: Risk On (+1)
        # VIX > 25: Risk Off (-1)
        # 中间: 中性 (0)
        if vix < 15:
            risk = 1.0
        elif vix > 25:
            risk = -1.0
        else:
            risk = (20 - vix) / 5  # 线性插值

        return self.create_result(float(risk), signal_time)


class Gold(BaseFactor):
    """黄金价格"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="gold",
            name="黄金价格",
            family=FactorFamily.MACRO,
            description="黄金现货价格 (USD/oz)",
            data_dependencies=[DataDependency.DXY],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        gold_data = data.get("gold") or data.get("macro")

        if gold_data is None or gold_data.empty:
            return self.create_invalid_result(signal_time, "Missing gold data")

        for col in ["gold", "GOLD", "xauusd", "close"]:
            if col in gold_data.columns:
                value = gold_data[col].iloc[-1]
                return self.create_result(float(value), signal_time)

        return self.create_invalid_result(signal_time, "Missing gold column")


class GoldBTCRatio(BaseFactor):
    """黄金/BTC比值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="gold_btc_ratio",
            name="黄金/BTC比值",
            family=FactorFamily.MACRO,
            description="黄金价格除以BTC价格 (相对估值)",
            data_dependencies=[DataDependency.DXY, DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        gold_data = data.get("gold") or data.get("macro")
        btc_data = data.get("klines_5m")

        if gold_data is None or btc_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        gold = None
        for col in ["gold", "GOLD", "xauusd"]:
            if col in gold_data.columns:
                gold = gold_data[col].iloc[-1]
                break

        btc = btc_data["close"].iloc[-1] if "close" in btc_data.columns else None

        if gold is not None and btc is not None and btc > 0:
            ratio = gold / btc
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing price data")


class BTCSPXCorr(BaseFactor):
    """BTC与SPX相关性"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="btc_spx_corr",
            name="BTC-SPX相关性",
            family=FactorFamily.MACRO,
            description="BTC与标普500的30天滚动相关系数",
            data_dependencies=[DataDependency.SPX, DataDependency.KLINES_5M],
            window=30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        spx_data = data.get("spx") or data.get("macro")
        btc_data = data.get("klines_5m")

        if spx_data is None or btc_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if len(spx_data) < 30 or len(btc_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        spx_col = None
        for c in ["spx", "SPX", "close"]:
            if c in spx_data.columns:
                spx_col = c
                break

        if spx_col is None:
            return self.create_invalid_result(signal_time, "Missing SPX column")

        # 计算收益率
        spx_rets = np.diff(np.log(spx_data[spx_col].values[-31:]))
        btc_rets = np.diff(np.log(btc_data["close"].values[-31:]))

        # 取较短的长度对齐
        min_len = min(len(spx_rets), len(btc_rets))
        if min_len < 20:
            return self.create_invalid_result(signal_time, "Insufficient returns")

        corr = np.corrcoef(spx_rets[-min_len:], btc_rets[-min_len:])[0, 1]
        return self.create_result(float(corr) if not np.isnan(corr) else 0, signal_time)


class BTCDXYCorr(BaseFactor):
    """BTC与DXY相关性"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="btc_dxy_corr",
            name="BTC-DXY相关性",
            family=FactorFamily.MACRO,
            description="BTC与美元指数的30天滚动相关系数",
            data_dependencies=[DataDependency.DXY, DataDependency.KLINES_5M],
            window=30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        dxy_data = data.get("dxy") or data.get("macro")
        btc_data = data.get("klines_5m")

        if dxy_data is None or btc_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if len(dxy_data) < 30 or len(btc_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        dxy_col = None
        for c in ["dxy", "DXY", "close"]:
            if c in dxy_data.columns:
                dxy_col = c
                break

        if dxy_col is None:
            return self.create_invalid_result(signal_time, "Missing DXY column")

        dxy_rets = np.diff(np.log(dxy_data[dxy_col].values[-31:]))
        btc_rets = np.diff(np.log(btc_data["close"].values[-31:]))

        min_len = min(len(dxy_rets), len(btc_rets))
        if min_len < 20:
            return self.create_invalid_result(signal_time, "Insufficient returns")

        corr = np.corrcoef(dxy_rets[-min_len:], btc_rets[-min_len:])[0, 1]
        return self.create_result(float(corr) if not np.isnan(corr) else 0, signal_time)


class MacroMomentum(BaseFactor):
    """宏观动量综合指标"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="macro_momentum",
            name="宏观动量",
            family=FactorFamily.MACRO,
            description="综合宏观动量 (SPX+Gold-DXY-VIX)",
            data_dependencies=[DataDependency.SPX, DataDependency.DXY],
            window=14,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        macro_data = data.get("macro")

        if macro_data is None:
            return self.create_invalid_result(signal_time, "Missing macro data")

        components = []

        # SPX动量 (正)
        for col in ["spx", "SPX"]:
            if col in macro_data.columns and len(macro_data) >= 14:
                current = macro_data[col].iloc[-1]
                prev = macro_data[col].iloc[-14]
                mom = (current / prev - 1) if prev > 0 else 0
                components.append(mom)
                break

        # Gold动量 (正)
        for col in ["gold", "GOLD"]:
            if col in macro_data.columns and len(macro_data) >= 14:
                current = macro_data[col].iloc[-1]
                prev = macro_data[col].iloc[-14]
                mom = (current / prev - 1) if prev > 0 else 0
                components.append(mom)
                break

        # DXY动量 (负)
        for col in ["dxy", "DXY"]:
            if col in macro_data.columns and len(macro_data) >= 14:
                current = macro_data[col].iloc[-1]
                prev = macro_data[col].iloc[-14]
                mom = -(current / prev - 1) if prev > 0 else 0
                components.append(mom)
                break

        # VIX变化 (负)
        for col in ["vix", "VIX"]:
            if col in macro_data.columns and len(macro_data) >= 14:
                current = macro_data[col].iloc[-1]
                prev = macro_data[col].iloc[-14]
                mom = -(current / prev - 1) if prev > 0 else 0
                components.append(mom)
                break

        if not components:
            return self.create_invalid_result(signal_time, "No components available")

        macro_mom = np.mean(components)
        return self.create_result(float(macro_mom), signal_time)


# 导出所有因子
MACRO_FACTORS = [
    DXY,
    DXYChange,
    US10Y,
    US2Y10YSpread,
    RealYield,
    SPX,
    SPXChange,
    VIX,
    VIXChange,
    RiskOnOff,
    Gold,
    GoldBTCRatio,
    BTCSPXCorr,
    BTCDXYCorr,
    MacroMomentum,
]
