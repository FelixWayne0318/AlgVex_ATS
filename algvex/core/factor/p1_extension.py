"""
P1 扩展因子 (21个)

需要自建数据落盘的高级因子:
- L2深度因子 (8个): 订单簿深度分析
- 清算因子 (5个): 强平数据分析
- 多交易所Basis (8个): 跨交易所价差分析

数据可得性: B/C档 (需自建WebSocket采集)
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


# =====================================================
# L2 深度因子 (8个) - Step 9
# =====================================================

class BidAskSpread(BaseFactor):
    """买卖价差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="bid_ask_spread",
            name="买卖价差",
            family=FactorFamily.DERIVATIVES,
            description="最优买卖价差 (bps)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "spread" in depth.columns:
            value = depth["spread"].values[-1]
        elif "best_ask" in depth.columns and "best_bid" in depth.columns:
            ask = depth["best_ask"].values[-1]
            bid = depth["best_bid"].values[-1]
            mid = (ask + bid) / 2
            value = (ask - bid) / mid * 10000  # bps
        else:
            return self.create_invalid_result(signal_time, "Missing spread columns")

        return self.create_result(float(value), signal_time)


class OrderBookImbalance(BaseFactor):
    """订单簿不平衡"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="order_book_imbalance",
            name="订单簿不平衡",
            family=FactorFamily.DERIVATIVES,
            description="(买单量-卖单量)/(买单量+卖单量)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "imbalance" in depth.columns:
            value = depth["imbalance"].values[-1]
        elif "bid_volume" in depth.columns and "ask_volume" in depth.columns:
            bid_vol = depth["bid_volume"].values[-1]
            ask_vol = depth["ask_volume"].values[-1]
            total = bid_vol + ask_vol
            value = (bid_vol - ask_vol) / total if total > 0 else 0
        else:
            return self.create_invalid_result(signal_time, "Missing volume columns")

        return self.create_result(float(value), signal_time)


class Depth1PctBid(BaseFactor):
    """1%深度买单"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="depth_1pct_bid",
            name="1%深度买单",
            family=FactorFamily.DERIVATIVES,
            description="价格下跌1%内的买单总量 (USD)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "depth_1pct_bid" in depth.columns:
            value = depth["depth_1pct_bid"].values[-1]
        else:
            value = 0  # 需要聚合数据

        return self.create_result(float(value), signal_time)


class Depth1PctAsk(BaseFactor):
    """1%深度卖单"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="depth_1pct_ask",
            name="1%深度卖单",
            family=FactorFamily.DERIVATIVES,
            description="价格上涨1%内的卖单总量 (USD)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "depth_1pct_ask" in depth.columns:
            value = depth["depth_1pct_ask"].values[-1]
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class DepthSlopeBid(BaseFactor):
    """买单深度斜率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="depth_slope_bid",
            name="买单深度斜率",
            family=FactorFamily.DERIVATIVES,
            description="买单累积量随价格变化的斜率",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "depth_slope_bid" in depth.columns:
            value = depth["depth_slope_bid"].values[-1]
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class DepthSlopeAsk(BaseFactor):
    """卖单深度斜率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="depth_slope_ask",
            name="卖单深度斜率",
            family=FactorFamily.DERIVATIVES,
            description="卖单累积量随价格变化的斜率",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "depth_slope_ask" in depth.columns:
            value = depth["depth_slope_ask"].values[-1]
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class ImpactCostBuy(BaseFactor):
    """买入冲击成本"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="impact_cost_buy",
            name="买入冲击成本",
            family=FactorFamily.DERIVATIVES,
            description="买入$100K的预期滑点 (bps)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "impact_cost_buy" in depth.columns:
            value = depth["impact_cost_buy"].values[-1]
        elif "impact_cost_100k_buy" in depth.columns:
            value = depth["impact_cost_100k_buy"].values[-1]
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class ImpactCostSell(BaseFactor):
    """卖出冲击成本"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="impact_cost_sell",
            name="卖出冲击成本",
            family=FactorFamily.DERIVATIVES,
            description="卖出$100K的预期滑点 (bps)",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        depth = data.get("order_book") or data.get("depth")
        if depth is None or len(depth) == 0:
            return self.create_invalid_result(signal_time, "No depth data")

        if "impact_cost_sell" in depth.columns:
            value = depth["impact_cost_sell"].values[-1]
        elif "impact_cost_100k_sell" in depth.columns:
            value = depth["impact_cost_100k_sell"].values[-1]
        else:
            value = 0

        return self.create_result(float(value), signal_time)


# =====================================================
# 清算因子 (5个) - Step 10
# =====================================================

class LiquidationVolumeLong(BaseFactor):
    """多头清算量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidation_volume_long",
            name="多头清算量",
            family=FactorFamily.ORDER_FLOW,
            description="1小时多头清算金额 (USD)",
            data_dependencies=[DataDependency.LIQUIDATION],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        liq = data.get("liquidations")
        if liq is None or len(liq) == 0:
            return self.create_invalid_result(signal_time, "No liquidation data")

        if "volume_long" in liq.columns:
            value = liq["volume_long"].values[-BARS_PER_HOUR:].sum() if len(liq) >= BARS_PER_HOUR else liq["volume_long"].sum()
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class LiquidationVolumeShort(BaseFactor):
    """空头清算量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidation_volume_short",
            name="空头清算量",
            family=FactorFamily.ORDER_FLOW,
            description="1小时空头清算金额 (USD)",
            data_dependencies=[DataDependency.LIQUIDATION],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        liq = data.get("liquidations")
        if liq is None or len(liq) == 0:
            return self.create_invalid_result(signal_time, "No liquidation data")

        if "volume_short" in liq.columns:
            value = liq["volume_short"].values[-BARS_PER_HOUR:].sum() if len(liq) >= BARS_PER_HOUR else liq["volume_short"].sum()
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class LiquidationImbalance(BaseFactor):
    """清算不平衡"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidation_imbalance",
            name="清算不平衡",
            family=FactorFamily.ORDER_FLOW,
            description="(多头清算-空头清算)/(总清算)",
            data_dependencies=[DataDependency.LIQUIDATION],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        liq = data.get("liquidations")
        if liq is None or len(liq) == 0:
            return self.create_invalid_result(signal_time, "No liquidation data")

        if "imbalance" in liq.columns:
            value = liq["imbalance"].values[-1]
        elif "volume_long" in liq.columns and "volume_short" in liq.columns:
            long_vol = liq["volume_long"].values[-BARS_PER_HOUR:].sum() if len(liq) >= BARS_PER_HOUR else liq["volume_long"].sum()
            short_vol = liq["volume_short"].values[-BARS_PER_HOUR:].sum() if len(liq) >= BARS_PER_HOUR else liq["volume_short"].sum()
            total = long_vol + short_vol
            value = (long_vol - short_vol) / total if total > 0 else 0
        else:
            value = 0

        return self.create_result(float(value), signal_time)


class LiquidationSpike(BaseFactor):
    """清算激增"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidation_spike",
            name="清算激增",
            family=FactorFamily.ORDER_FLOW,
            description="清算量是否激增 (>3x均值)",
            data_dependencies=[DataDependency.LIQUIDATION],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        liq = data.get("liquidations")
        if liq is None or len(liq) < BARS_PER_DAY:
            return self.create_invalid_result(signal_time, "Insufficient liquidation data")

        if "total_volume" in liq.columns:
            volumes = liq["total_volume"].values[-BARS_PER_DAY:]
        elif "volume_long" in liq.columns and "volume_short" in liq.columns:
            volumes = (liq["volume_long"] + liq["volume_short"]).values[-BARS_PER_DAY:]
        else:
            return self.create_invalid_result(signal_time, "Missing volume columns")

        current_hour = volumes[-BARS_PER_HOUR:].sum()
        avg_hour = np.mean([volumes[i:i+BARS_PER_HOUR].sum() for i in range(0, len(volumes)-BARS_PER_HOUR, BARS_PER_HOUR)])

        value = 1.0 if current_hour > avg_hour * 3 else 0.0

        return self.create_result(float(value), signal_time)


class LiquidationMomentum(BaseFactor):
    """清算动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="liquidation_momentum",
            name="清算动量",
            family=FactorFamily.ORDER_FLOW,
            description="清算量变化趋势",
            data_dependencies=[DataDependency.LIQUIDATION],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.B,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        liq = data.get("liquidations")
        if liq is None or len(liq) < BARS_PER_HOUR * 4:
            return self.create_invalid_result(signal_time, "Insufficient liquidation data")

        if "total_volume" in liq.columns:
            volumes = liq["total_volume"].values[-BARS_PER_HOUR * 4:]
        elif "volume_long" in liq.columns and "volume_short" in liq.columns:
            volumes = (liq["volume_long"] + liq["volume_short"]).values[-BARS_PER_HOUR * 4:]
        else:
            return self.create_invalid_result(signal_time, "Missing volume columns")

        recent = volumes[-BARS_PER_HOUR:].sum()
        past = volumes[:-BARS_PER_HOUR].sum()

        value = recent / past - 1 if past > 0 else 0

        return self.create_result(float(value), signal_time)


# =====================================================
# 多交易所Basis因子 (8个) - Step 11
# =====================================================

class BasisBinance(BaseFactor):
    """Binance基差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_binance",
            name="Binance基差",
            family=FactorFamily.DERIVATIVES,
            description="Binance永续-现货价差",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is not None and "binance" in multi_basis.columns:
            value = multi_basis["binance"].values[-1]
        else:
            # Fallback: 使用默认基差
            basis = data.get("basis")
            if basis is not None and len(basis) > 0:
                value = basis["basis"].values[-1] if "basis" in basis.columns else 0
            else:
                return self.create_invalid_result(signal_time, "No basis data")

        return self.create_result(float(value), signal_time)


class BasisBybit(BaseFactor):
    """Bybit基差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_bybit",
            name="Bybit基差",
            family=FactorFamily.DERIVATIVES,
            description="Bybit永续-现货价差",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is not None and "bybit" in multi_basis.columns:
            value = multi_basis["bybit"].values[-1]
        else:
            return self.create_invalid_result(signal_time, "No Bybit basis data")

        return self.create_result(float(value), signal_time)


class BasisOKX(BaseFactor):
    """OKX基差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_okx",
            name="OKX基差",
            family=FactorFamily.DERIVATIVES,
            description="OKX永续-现货价差",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is not None and "okx" in multi_basis.columns:
            value = multi_basis["okx"].values[-1]
        else:
            return self.create_invalid_result(signal_time, "No OKX basis data")

        return self.create_result(float(value), signal_time)


class BasisConsensus(BaseFactor):
    """基差共识"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_consensus",
            name="基差共识",
            family=FactorFamily.DERIVATIVES,
            description="多交易所基差中位数",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is None or len(multi_basis) == 0:
            return self.create_invalid_result(signal_time, "No multi-exchange basis data")

        # 取各交易所基差的中位数
        exchanges = ["binance", "bybit", "okx"]
        values = []
        for ex in exchanges:
            if ex in multi_basis.columns:
                values.append(multi_basis[ex].values[-1])

        if len(values) == 0:
            return self.create_invalid_result(signal_time, "No exchange basis found")

        value = np.median(values)
        return self.create_result(float(value), signal_time)


class BasisDispersion(BaseFactor):
    """基差离散度"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_dispersion",
            name="基差离散度",
            family=FactorFamily.DERIVATIVES,
            description="多交易所基差标准差",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is None or len(multi_basis) == 0:
            return self.create_invalid_result(signal_time, "No multi-exchange basis data")

        exchanges = ["binance", "bybit", "okx"]
        values = []
        for ex in exchanges:
            if ex in multi_basis.columns:
                values.append(multi_basis[ex].values[-1])

        if len(values) < 2:
            return self.create_invalid_result(signal_time, "Need at least 2 exchanges")

        value = np.std(values)
        return self.create_result(float(value), signal_time)


class CrossExchangeSpreadP1(BaseFactor):
    """跨交易所价差 (P1)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cross_exchange_spread_p1",
            name="跨交易所价差",
            family=FactorFamily.DERIVATIVES,
            description="交易所间最大价差",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is None or len(multi_basis) == 0:
            return self.create_invalid_result(signal_time, "No multi-exchange basis data")

        exchanges = ["binance", "bybit", "okx"]
        values = []
        for ex in exchanges:
            if ex in multi_basis.columns:
                values.append(multi_basis[ex].values[-1])

        if len(values) < 2:
            return self.create_invalid_result(signal_time, "Need at least 2 exchanges")

        value = max(values) - min(values)
        return self.create_result(float(value), signal_time)


class PriceDiscoveryLeaderP1(BaseFactor):
    """价格发现领导者"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="price_discovery_leader",
            name="价格发现领导者",
            family=FactorFamily.DERIVATIVES,
            description="哪个交易所领先价格发现 (1=Binance, 2=Bybit, 3=OKX)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        # 需要多交易所价格时序数据来计算领先滞后关系
        multi_prices = data.get("multi_exchange_prices")
        if multi_prices is None or len(multi_prices) < BARS_PER_HOUR:
            # 默认返回Binance (1)
            return self.create_result(1.0, signal_time)

        # 简化版本: 返回成交量最大的交易所
        exchanges = ["binance", "bybit", "okx"]
        volumes = {}
        for ex in exchanges:
            vol_col = f"{ex}_volume"
            if vol_col in multi_prices.columns:
                volumes[ex] = multi_prices[vol_col].values[-BARS_PER_HOUR:].sum()

        if not volumes:
            return self.create_result(1.0, signal_time)

        leader = max(volumes, key=volumes.get)
        leader_map = {"binance": 1, "bybit": 2, "okx": 3}
        value = leader_map.get(leader, 1)

        return self.create_result(float(value), signal_time)


class ArbitragePressure(BaseFactor):
    """套利压力"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="arbitrage_pressure",
            name="套利压力",
            family=FactorFamily.DERIVATIVES,
            description="跨交易所套利机会强度",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_basis = data.get("multi_exchange_basis")
        if multi_basis is None or len(multi_basis) == 0:
            return self.create_invalid_result(signal_time, "No multi-exchange basis data")

        exchanges = ["binance", "bybit", "okx"]
        values = []
        for ex in exchanges:
            if ex in multi_basis.columns:
                values.append(multi_basis[ex].values[-1])

        if len(values) < 2:
            return self.create_invalid_result(signal_time, "Need at least 2 exchanges")

        # 套利压力 = 价差 / 交易成本 (假设交易成本约5bps)
        spread = max(values) - min(values)
        trading_cost = 5  # bps
        pressure = spread / trading_cost if trading_cost > 0 else 0

        return self.create_result(float(pressure), signal_time)


# 导出所有P1因子
P1_EXTENSION_FACTORS = [
    # L2 深度 (8个)
    BidAskSpread,
    OrderBookImbalance,
    Depth1PctBid,
    Depth1PctAsk,
    DepthSlopeBid,
    DepthSlopeAsk,
    ImpactCostBuy,
    ImpactCostSell,
    # 清算 (5个)
    LiquidationVolumeLong,
    LiquidationVolumeShort,
    LiquidationImbalance,
    LiquidationSpike,
    LiquidationMomentum,
    # 多交易所Basis (8个)
    BasisBinance,
    BasisBybit,
    BasisOKX,
    BasisConsensus,
    BasisDispersion,
    CrossExchangeSpreadP1,
    PriceDiscoveryLeaderP1,
    ArbitragePressure,
]
