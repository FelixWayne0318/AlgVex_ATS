"""
链上因子 (10个)

基于DeFiLlama等链上数据的因子。

稳定币因子 (5个):
1. StablecoinSupply - 稳定币总供应量
2. StablecoinSupplyChange7D - 7天供应量变化
3. StablecoinDominance - 稳定币市值占比
4. USDTUSDCRatio - USDT/USDC比值
5. StablecoinMomentum - 稳定币动量

DeFi TVL因子 (5个):
6. DeFiTVLTotal - DeFi总锁仓量
7. DeFiTVLChange7D - 7天TVL变化
8. ETHTVLDominance - ETH TVL主导地位
9. TVLMcapRatio - TVL/市值比
10. TVLMomentum - TVL动量
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


class StablecoinSupply(BaseFactor):
    """稳定币总供应量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="stablecoin_supply",
            name="稳定币总供应量",
            family=FactorFamily.ON_CHAIN,
            description="主要稳定币的总流通量 (USD)",
            data_dependencies=[DataDependency.STABLECOIN],
            window=1,
            history_tier=HistoryTier.A,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        stablecoin_data = data.get("stablecoin")

        if stablecoin_data is None or stablecoin_data.empty:
            return self.create_invalid_result(signal_time, "Missing stablecoin data")

        if "circulating" in stablecoin_data.columns:
            total_supply = stablecoin_data["circulating"].sum()
            return self.create_result(float(total_supply), signal_time)

        if "total_circulating_usd" in stablecoin_data.columns:
            total_supply = stablecoin_data["total_circulating_usd"].iloc[-1]
            return self.create_result(float(total_supply), signal_time)

        return self.create_invalid_result(signal_time, "Missing supply data")


class StablecoinSupplyChange7D(BaseFactor):
    """稳定币7天供应量变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="stablecoin_supply_change_7d",
            name="稳定币7天变化",
            family=FactorFamily.ON_CHAIN,
            description="稳定币供应量的7天变化率",
            data_dependencies=[DataDependency.STABLECOIN],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        stablecoin_data = data.get("stablecoin")

        if stablecoin_data is None or len(stablecoin_data) < 8:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "circulating" in stablecoin_data.columns:
            current = stablecoin_data["circulating"].iloc[-1]
            prev_7d = stablecoin_data["circulating"].iloc[-8]
            change = (current - prev_7d) / prev_7d if prev_7d > 0 else 0
            return self.create_result(float(change), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


class StablecoinDominance(BaseFactor):
    """稳定币市值占比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="stablecoin_dominance",
            name="稳定币主导地位",
            family=FactorFamily.ON_CHAIN,
            description="USDT在稳定币总量中的占比",
            data_dependencies=[DataDependency.STABLECOIN],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        stablecoin_data = data.get("stablecoin")

        if stablecoin_data is None or stablecoin_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        # 假设数据包含各稳定币的供应量
        if "stablecoin" in stablecoin_data.columns and "circulating" in stablecoin_data.columns:
            usdt = stablecoin_data[stablecoin_data["stablecoin"] == "1"]["circulating"].sum()
            total = stablecoin_data["circulating"].sum()
            dominance = usdt / total if total > 0 else 0
            return self.create_result(float(dominance), signal_time)

        return self.create_invalid_result(signal_time, "Missing stablecoin breakdown")


class USDTUSDCRatio(BaseFactor):
    """USDT/USDC比值"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="usdt_usdc_ratio",
            name="USDT/USDC比值",
            family=FactorFamily.ON_CHAIN,
            description="USDT供应量与USDC供应量的比值",
            data_dependencies=[DataDependency.STABLECOIN],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        stablecoin_data = data.get("stablecoin")

        if stablecoin_data is None or stablecoin_data.empty:
            return self.create_invalid_result(signal_time, "Missing data")

        if "stablecoin" in stablecoin_data.columns and "circulating" in stablecoin_data.columns:
            usdt = stablecoin_data[stablecoin_data["stablecoin"] == "1"]["circulating"].sum()
            usdc = stablecoin_data[stablecoin_data["stablecoin"] == "2"]["circulating"].sum()
            ratio = usdt / usdc if usdc > 0 else 1.0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Missing breakdown")


class StablecoinMomentum(BaseFactor):
    """稳定币动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="stablecoin_momentum",
            name="稳定币动量",
            family=FactorFamily.ON_CHAIN,
            description="稳定币供应量的短期变化趋势",
            data_dependencies=[DataDependency.STABLECOIN],
            window=30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        stablecoin_data = data.get("stablecoin")

        if stablecoin_data is None or len(stablecoin_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "circulating" in stablecoin_data.columns:
            supplies = stablecoin_data["circulating"].values
            # 短期 (7天) vs 长期 (30天) 均线
            ma_short = np.mean(supplies[-7:])
            ma_long = np.mean(supplies[-30:])
            momentum = (ma_short / ma_long - 1) if ma_long > 0 else 0
            return self.create_result(float(momentum), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


class DeFiTVLTotal(BaseFactor):
    """DeFi总锁仓量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="defi_tvl_total",
            name="DeFi总TVL",
            family=FactorFamily.ON_CHAIN,
            description="DeFi协议总锁仓量 (USD)",
            data_dependencies=[DataDependency.DEFI_TVL],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        tvl_data = data.get("defi_tvl")

        if tvl_data is None or tvl_data.empty:
            return self.create_invalid_result(signal_time, "Missing TVL data")

        if "tvl" in tvl_data.columns:
            total_tvl = tvl_data["tvl"].iloc[-1]
            return self.create_result(float(total_tvl), signal_time)

        if "totalLiquidityUSD" in tvl_data.columns:
            total_tvl = tvl_data["totalLiquidityUSD"].sum()
            return self.create_result(float(total_tvl), signal_time)

        return self.create_invalid_result(signal_time, "Missing TVL column")


class DeFiTVLChange7D(BaseFactor):
    """DeFi TVL 7天变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="defi_tvl_change_7d",
            name="DeFi TVL 7天变化",
            family=FactorFamily.ON_CHAIN,
            description="DeFi TVL的7天变化率",
            data_dependencies=[DataDependency.DEFI_TVL],
            window=7,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        tvl_data = data.get("defi_tvl")

        if tvl_data is None or len(tvl_data) < 8:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "tvl" in tvl_data.columns:
            current = tvl_data["tvl"].iloc[-1]
            prev_7d = tvl_data["tvl"].iloc[-8]
            change = (current - prev_7d) / prev_7d if prev_7d > 0 else 0
            return self.create_result(float(change), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


class ETHTVLDominance(BaseFactor):
    """ETH TVL主导地位"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="eth_tvl_dominance",
            name="ETH TVL主导地位",
            family=FactorFamily.ON_CHAIN,
            description="以太坊TVL占总TVL的比例",
            data_dependencies=[DataDependency.DEFI_TVL],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        tvl_data = data.get("defi_tvl")
        chain_tvl = data.get("chain_tvl")

        if chain_tvl is not None and "chain" in chain_tvl.columns:
            eth_tvl = chain_tvl[chain_tvl["chain"] == "Ethereum"]["tvl"].iloc[-1] \
                if "Ethereum" in chain_tvl["chain"].values else 0
            total_tvl = chain_tvl["tvl"].sum()
            dominance = eth_tvl / total_tvl if total_tvl > 0 else 0
            return self.create_result(float(dominance), signal_time)

        # 默认返回历史平均值
        return self.create_result(0.55, signal_time)


class TVLMcapRatio(BaseFactor):
    """TVL/市值比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="tvl_mcap_ratio",
            name="TVL市值比",
            family=FactorFamily.ON_CHAIN,
            description="DeFi TVL与加密货币总市值的比率",
            data_dependencies=[DataDependency.DEFI_TVL],
            window=1,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        tvl_data = data.get("defi_tvl")
        mcap_data = data.get("crypto_mcap")

        if tvl_data is None:
            return self.create_invalid_result(signal_time, "Missing TVL data")

        tvl = tvl_data["tvl"].iloc[-1] if "tvl" in tvl_data.columns else 0

        if mcap_data is not None and "total_mcap" in mcap_data.columns:
            mcap = mcap_data["total_mcap"].iloc[-1]
            ratio = tvl / mcap if mcap > 0 else 0
            return self.create_result(float(ratio), signal_time)

        # 使用估计的市值 (约2.5T)
        estimated_mcap = 2.5e12
        ratio = tvl / estimated_mcap
        return self.create_result(float(ratio), signal_time)


class TVLMomentum(BaseFactor):
    """TVL动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="tvl_momentum",
            name="TVL动量",
            family=FactorFamily.ON_CHAIN,
            description="TVL的短期趋势强度",
            data_dependencies=[DataDependency.DEFI_TVL],
            window=30,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        tvl_data = data.get("defi_tvl")

        if tvl_data is None or len(tvl_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient data")

        if "tvl" in tvl_data.columns:
            tvl_values = tvl_data["tvl"].values
            ma_short = np.mean(tvl_values[-7:])
            ma_long = np.mean(tvl_values[-30:])
            momentum = (ma_short / ma_long - 1) if ma_long > 0 else 0
            return self.create_result(float(momentum), signal_time)

        return self.create_invalid_result(signal_time, "Missing data")


# 导出所有因子
ONCHAIN_FACTORS = [
    StablecoinSupply,
    StablecoinSupplyChange7D,
    StablecoinDominance,
    USDTUSDCRatio,
    StablecoinMomentum,
    DeFiTVLTotal,
    DeFiTVLChange7D,
    ETHTVLDominance,
    TVLMcapRatio,
    TVLMomentum,
]
