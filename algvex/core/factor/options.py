"""
期权/波动率因子 (20个)

基于Deribit期权数据的因子。

隐含波动率因子 (10个):
1. DVOL_BTC - BTC DVOL指数
2. DVOL_ETH - ETH DVOL指数
3. DVOLChange24H - DVOL 24小时变化
4. IVATM - ATM隐含波动率
5. IVSkew - 看跌/看涨波动率偏斜
6. IVTermStructure - 期限结构斜率
7. IVRVSpread - IV-RV价差
8. IVPercentile - IV历史百分位
9. VolRiskPremium - 波动率风险溢价
10. IVChange - IV变化率

期权持仓因子 (10个):
11. PutCallRatio - 看跌/看涨成交量比
12. PutCallOIRatio - 看跌/看涨持仓量比
13. MaxPain - 最大痛点价格
14. MaxPainDistance - 距最大痛点距离
15. GammaExposure - Gamma敞口
16. OptionVolumeSpike - 期权成交量激增
17. LargePutOI - 大额看跌持仓
18. LargeCallOI - 大额看涨持仓
19. OptionSkewChange - 偏斜变化
20. VannaExposure - Vanna敞口
"""

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from .base import (
    BaseFactor, FactorFamily, FactorMetadata, FactorResult,
    DataDependency, HistoryTier,
)


BARS_PER_DAY = 288


class DVOL_BTC(BaseFactor):
    """BTC DVOL波动率指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="dvol_btc",
            name="BTC DVOL指数",
            family=FactorFamily.OPTIONS,
            description="Deribit BTC 30天隐含波动率指数",
            data_dependencies=[DataDependency.DVOL],
            window=1,
            history_tier=HistoryTier.B,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        if vol_data is None or vol_data.empty:
            return self.create_invalid_result(signal_time, "Missing DVOL data")

        if "currency" in vol_data.columns:
            btc_data = vol_data[vol_data["currency"] == "BTC"]
            if btc_data.empty:
                return self.create_invalid_result(signal_time, "No BTC data")
            dvol = btc_data["dvol"].iloc[-1]
        else:
            dvol = vol_data["dvol"].iloc[-1] if "dvol" in vol_data.columns else np.nan

        return self.create_result(float(dvol), signal_time)


class DVOL_ETH(BaseFactor):
    """ETH DVOL波动率指数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="dvol_eth",
            name="ETH DVOL指数",
            family=FactorFamily.OPTIONS,
            description="Deribit ETH 30天隐含波动率指数",
            data_dependencies=[DataDependency.DVOL],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        if vol_data is None or vol_data.empty:
            return self.create_invalid_result(signal_time, "Missing DVOL data")

        if "currency" in vol_data.columns:
            eth_data = vol_data[vol_data["currency"] == "ETH"]
            if eth_data.empty:
                return self.create_invalid_result(signal_time, "No ETH data")
            dvol = eth_data["dvol"].iloc[-1]
        else:
            return self.create_invalid_result(signal_time, "No currency column")

        return self.create_result(float(dvol), signal_time)


class DVOLChange24H(BaseFactor):
    """DVOL 24小时变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="dvol_change_24h",
            name="DVOL 24小时变化",
            family=FactorFamily.OPTIONS,
            description="DVOL指数的24小时变化率",
            data_dependencies=[DataDependency.DVOL],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        if vol_data is None or len(vol_data) < 2:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "currency" in vol_data.columns:
            btc_data = vol_data[vol_data["currency"] == "BTC"].sort_values("datetime")
        else:
            btc_data = vol_data.sort_values("datetime") if "datetime" in vol_data.columns else vol_data

        if len(btc_data) < 2:
            return self.create_invalid_result(signal_time, "Insufficient BTC data")

        current = btc_data["dvol"].iloc[-1]
        prev = btc_data["dvol"].iloc[0]
        change = (current - prev) / prev if prev > 0 else 0

        return self.create_result(float(change), signal_time)


class IVATM(BaseFactor):
    """ATM隐含波动率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_atm",
            name="ATM隐含波动率",
            family=FactorFamily.OPTIONS,
            description="平值期权隐含波动率",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing IV surface")

        if "underlying_price" in iv_surface.columns and "strike" in iv_surface.columns:
            price = iv_surface["underlying_price"].iloc[-1]
            iv_surface = iv_surface.copy()
            iv_surface["moneyness"] = abs(iv_surface["strike"] - price) / price
            atm = iv_surface.loc[iv_surface["moneyness"].idxmin()]
            iv = atm.get("mark_iv", np.nan)
            return self.create_result(float(iv) if not pd.isna(iv) else np.nan, signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class IVSkew(BaseFactor):
    """IV偏斜"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_skew",
            name="IV偏斜",
            family=FactorFamily.OPTIONS,
            description="25-delta put IV - 25-delta call IV",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing IV surface")

        if "option_type" in iv_surface.columns and "mark_iv" in iv_surface.columns:
            puts = iv_surface[iv_surface["option_type"] == "P"]
            calls = iv_surface[iv_surface["option_type"] == "C"]

            if not puts.empty and not calls.empty:
                put_iv = puts["mark_iv"].mean()
                call_iv = calls["mark_iv"].mean()
                skew = put_iv - call_iv
                return self.create_result(float(skew), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


class IVTermStructure(BaseFactor):
    """IV期限结构斜率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_term_structure",
            name="IV期限结构",
            family=FactorFamily.OPTIONS,
            description="短期IV vs 长期IV的比值",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "expiry" in iv_surface.columns and "mark_iv" in iv_surface.columns:
            expiries = iv_surface.groupby("expiry")["mark_iv"].mean().sort_index()
            if len(expiries) >= 2:
                short_term = expiries.iloc[0]
                long_term = expiries.iloc[-1]
                ratio = short_term / long_term if long_term > 0 else 1.0
                return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient expiries")


class IVRVSpread(BaseFactor):
    """IV-RV价差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_rv_spread",
            name="IV-RV价差",
            family=FactorFamily.OPTIONS,
            description="隐含波动率与已实现波动率之差",
            data_dependencies=[DataDependency.DVOL, DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        klines = data.get("klines_5m") or data.get("klines")

        if vol_data is None or klines is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if "currency" in vol_data.columns:
            btc_data = vol_data[vol_data["currency"] == "BTC"]
        else:
            btc_data = vol_data

        if btc_data.empty or "dvol" not in btc_data.columns:
            return self.create_invalid_result(signal_time, "Missing DVOL")

        iv = btc_data["dvol"].iloc[-1]

        close = klines["close"].values
        if len(close) < BARS_PER_DAY * 30:
            return self.create_invalid_result(signal_time, "Insufficient klines")

        returns = np.diff(np.log(close[-BARS_PER_DAY*30:]))
        rv = np.std(returns) * np.sqrt(BARS_PER_DAY * 365) * 100

        spread = iv - rv
        return self.create_result(float(spread), signal_time)


class IVPercentile(BaseFactor):
    """IV历史百分位"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_percentile",
            name="IV历史百分位",
            family=FactorFamily.OPTIONS,
            description="当前IV在过去一年的百分位",
            data_dependencies=[DataDependency.DVOL],
            window=BARS_PER_DAY * 365,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        if vol_data is None or len(vol_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "currency" in vol_data.columns:
            btc_data = vol_data[vol_data["currency"] == "BTC"]
        else:
            btc_data = vol_data

        if len(btc_data) < 30 or "dvol" not in btc_data.columns:
            return self.create_invalid_result(signal_time, "Missing data")

        dvol_history = btc_data["dvol"].values
        current = dvol_history[-1]
        percentile = np.sum(dvol_history < current) / len(dvol_history)

        return self.create_result(float(percentile), signal_time)


class VolRiskPremium(BaseFactor):
    """波动率风险溢价"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vol_risk_premium",
            name="波动率风险溢价",
            family=FactorFamily.OPTIONS,
            description="IV/RV比值",
            data_dependencies=[DataDependency.DVOL, DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        vol_data = data.get("deribit_vol_index") or data.get("dvol")
        klines = data.get("klines_5m") or data.get("klines")

        if vol_data is None or klines is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if "currency" in vol_data.columns:
            btc_data = vol_data[vol_data["currency"] == "BTC"]
        else:
            btc_data = vol_data

        if btc_data.empty or "dvol" not in btc_data.columns:
            return self.create_invalid_result(signal_time, "Missing DVOL")

        iv = btc_data["dvol"].iloc[-1]

        close = klines["close"].values
        if len(close) < BARS_PER_DAY * 30:
            return self.create_invalid_result(signal_time, "Insufficient klines")

        returns = np.diff(np.log(close[-BARS_PER_DAY*30:]))
        rv = np.std(returns) * np.sqrt(BARS_PER_DAY * 365) * 100

        ratio = iv / rv if rv > 0 else 1.0
        return self.create_result(float(ratio), signal_time)


class IVChange(BaseFactor):
    """IV变化率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="iv_change",
            name="IV变化率",
            family=FactorFamily.OPTIONS,
            description="ATM IV的日变化率",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or len(iv_surface) < 2:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "mark_iv" in iv_surface.columns:
            ivs = iv_surface["mark_iv"].dropna()
            if len(ivs) >= 2:
                change = (ivs.iloc[-1] - ivs.iloc[0]) / ivs.iloc[0] if ivs.iloc[0] > 0 else 0
                return self.create_result(float(change), signal_time)

        return self.create_invalid_result(signal_time, "Missing IV data")


class PutCallRatio(BaseFactor):
    """看跌/看涨成交量比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="put_call_ratio",
            name="看跌看涨比",
            family=FactorFamily.OPTIONS,
            description="看跌期权成交量 / 看涨期权成交量",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "option_type" in iv_surface.columns and "volume_usd" in iv_surface.columns:
            puts = iv_surface[iv_surface["option_type"] == "P"]["volume_usd"].sum()
            calls = iv_surface[iv_surface["option_type"] == "C"]["volume_usd"].sum()
            ratio = puts / calls if calls > 0 else 1.0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class PutCallOIRatio(BaseFactor):
    """看跌/看涨持仓量比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="put_call_oi_ratio",
            name="看跌看涨持仓比",
            family=FactorFamily.OPTIONS,
            description="看跌期权OI / 看涨期权OI",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "option_type" in iv_surface.columns and "open_interest" in iv_surface.columns:
            puts = iv_surface[iv_surface["option_type"] == "P"]["open_interest"].sum()
            calls = iv_surface[iv_surface["option_type"] == "C"]["open_interest"].sum()
            ratio = puts / calls if calls > 0 else 1.0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class MaxPain(BaseFactor):
    """最大痛点价格"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="max_pain",
            name="最大痛点",
            family=FactorFamily.OPTIONS,
            description="期权最大痛点价格",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "strike" not in iv_surface.columns or "open_interest" not in iv_surface.columns:
            return self.create_invalid_result(signal_time, "Missing columns")

        strikes = iv_surface["strike"].unique()
        if len(strikes) == 0:
            return self.create_invalid_result(signal_time, "No strikes")

        # 简化计算: 使用OI加权的行权价
        weighted_strike = np.average(
            iv_surface["strike"],
            weights=iv_surface["open_interest"].fillna(0) + 1
        )
        return self.create_result(float(weighted_strike), signal_time)


class MaxPainDistance(BaseFactor):
    """距最大痛点距离"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="max_pain_distance",
            name="距最大痛点距离",
            family=FactorFamily.OPTIONS,
            description="当前价格距最大痛点的百分比距离",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "underlying_price" not in iv_surface.columns:
            return self.create_invalid_result(signal_time, "Missing price")

        current_price = iv_surface["underlying_price"].iloc[-1]

        if "strike" in iv_surface.columns and "open_interest" in iv_surface.columns:
            weighted_strike = np.average(
                iv_surface["strike"],
                weights=iv_surface["open_interest"].fillna(0) + 1
            )
            distance = (current_price - weighted_strike) / weighted_strike
            return self.create_result(float(distance), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class GammaExposure(BaseFactor):
    """Gamma敞口"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="gamma_exposure",
            name="Gamma敞口",
            family=FactorFamily.OPTIONS,
            description="市场总Gamma敞口",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        greeks = data.get("deribit_greeks") or data.get("greeks")
        if greeks is None or greeks.empty:
            return self.create_invalid_result(signal_time, "Missing Greeks")

        if "gamma" in greeks.columns:
            total_gamma = greeks["gamma"].sum()
            return self.create_result(float(total_gamma), signal_time)

        return self.create_invalid_result(signal_time, "Missing gamma")


class OptionVolumeSpike(BaseFactor):
    """期权成交量激增"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="option_volume_spike",
            name="期权成交量激增",
            family=FactorFamily.OPTIONS,
            description="期权成交量相对均值的倍数",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=BARS_PER_DAY * 7,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "volume_usd" in iv_surface.columns:
            total_vol = iv_surface["volume_usd"].sum()
            avg_vol = iv_surface["volume_usd"].mean() * len(iv_surface)
            spike = total_vol / avg_vol if avg_vol > 0 else 1.0
            return self.create_result(float(spike), signal_time)

        return self.create_invalid_result(signal_time, "Missing volume")


class LargePutOI(BaseFactor):
    """大额看跌持仓"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="large_put_oi",
            name="大额看跌持仓",
            family=FactorFamily.OPTIONS,
            description="大额看跌期权持仓占比",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "option_type" in iv_surface.columns and "open_interest" in iv_surface.columns:
            puts = iv_surface[iv_surface["option_type"] == "P"]
            if puts.empty:
                return self.create_result(0.0, signal_time)

            total_oi = puts["open_interest"].sum()
            threshold = puts["open_interest"].quantile(0.75)
            large_oi = puts[puts["open_interest"] > threshold]["open_interest"].sum()

            ratio = large_oi / total_oi if total_oi > 0 else 0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class LargeCallOI(BaseFactor):
    """大额看涨持仓"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="large_call_oi",
            name="大额看涨持仓",
            family=FactorFamily.OPTIONS,
            description="大额看涨期权持仓占比",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "option_type" in iv_surface.columns and "open_interest" in iv_surface.columns:
            calls = iv_surface[iv_surface["option_type"] == "C"]
            if calls.empty:
                return self.create_result(0.0, signal_time)

            total_oi = calls["open_interest"].sum()
            threshold = calls["open_interest"].quantile(0.75)
            large_oi = calls[calls["open_interest"] > threshold]["open_interest"].sum()

            ratio = large_oi / total_oi if total_oi > 0 else 0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class OptionSkewChange(BaseFactor):
    """偏斜变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="option_skew_change",
            name="偏斜变化",
            family=FactorFamily.OPTIONS,
            description="IV偏斜的日变化",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        iv_surface = data.get("deribit_iv_surface") or data.get("option_chain")
        if iv_surface is None or iv_surface.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "option_type" in iv_surface.columns and "mark_iv" in iv_surface.columns:
            puts = iv_surface[iv_surface["option_type"] == "P"]["mark_iv"].mean()
            calls = iv_surface[iv_surface["option_type"] == "C"]["mark_iv"].mean()
            skew = puts - calls if not (np.isnan(puts) or np.isnan(calls)) else 0
            return self.create_result(float(skew), signal_time)

        return self.create_invalid_result(signal_time, "Missing columns")


class VannaExposure(BaseFactor):
    """Vanna敞口"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="vanna_exposure",
            name="Vanna敞口",
            family=FactorFamily.OPTIONS,
            description="市场Vanna敞口",
            data_dependencies=[DataDependency.OPTION_CHAIN],
            window=1,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        greeks = data.get("deribit_greeks") or data.get("greeks")
        if greeks is None or greeks.empty:
            return self.create_invalid_result(signal_time, "Missing Greeks")

        if "delta" in greeks.columns and "vega" in greeks.columns:
            vanna = (greeks["delta"] * greeks["vega"]).sum()
            return self.create_result(float(vanna), signal_time)

        return self.create_invalid_result(signal_time, "Missing delta/vega")


# 导出所有因子
OPTIONS_FACTORS = [
    DVOL_BTC,
    DVOL_ETH,
    DVOLChange24H,
    IVATM,
    IVSkew,
    IVTermStructure,
    IVRVSpread,
    IVPercentile,
    VolRiskPremium,
    IVChange,
    PutCallRatio,
    PutCallOIRatio,
    MaxPain,
    MaxPainDistance,
    GammaExposure,
    OptionVolumeSpike,
    LargePutOI,
    LargeCallOI,
    OptionSkewChange,
    VannaExposure,
]
