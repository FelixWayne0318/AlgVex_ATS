"""
衍生品结构因子 (15个)

基于跨交易所基差和市场结构的因子。

基差因子 (8个):
1. Basis - 现货-永续基差
2. BasisPercentage - 基差百分比
3. BasisMA24H - 24小时基差均线
4. BasisZScore - 基差Z分数
5. BasisMomentum - 基差动量
6. AnnualizedBasis - 年化基差
7. BasisFundingCorr - 基差与资金费率相关性
8. BasisExtreme - 基差极端值

市场结构因子 (7个):
9. CrossExchangeSpread - 跨交易所价差
10. BinancePremium - 币安溢价
11. InsuranceFundChange - 保险基金变化
12. MarketDepthRatio - 市场深度比
13. ExchangeDominance - 交易所主导地位
14. VolumeConcentration - 成交量集中度
15. PriceDiscoveryLeader - 价格发现领导者
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


class Basis(BaseFactor):
    """现货-永续基差 (bps)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis",
            name="现货永续基差",
            family=FactorFamily.DERIVATIVES,
            description="现货价格与永续合约价格之差 (bps)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
            is_mvp=False,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        # 需要现货和永续价格数据
        spot_data = data.get("spot_klines")
        perp_data = data.get("klines_5m")

        if spot_data is None or perp_data is None:
            # 如果没有现货数据，使用简化版本（标记价格与最新价格差）
            if perp_data is not None and "mark_price" in perp_data.columns:
                mark = perp_data["mark_price"].iloc[-1]
                last = perp_data["close"].iloc[-1]
                basis_bps = (mark - last) / last * 10000
                return self.create_result(float(basis_bps), signal_time)
            return self.create_invalid_result(signal_time, "Missing spot/perp data")

        spot_price = spot_data["close"].iloc[-1]
        perp_price = perp_data["close"].iloc[-1]

        basis_bps = (spot_price - perp_price) / spot_price * 10000
        return self.create_result(float(basis_bps), signal_time)


class BasisPercentage(BaseFactor):
    """基差百分比"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_percentage",
            name="基差百分比",
            family=FactorFamily.DERIVATIVES,
            description="基差占现货价格的百分比",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        spot_data = data.get("spot_klines")
        perp_data = data.get("klines_5m")

        if perp_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        # 简化版本：使用永续合约close和mark_price
        if "mark_price" in perp_data.columns:
            mark = perp_data["mark_price"].iloc[-1]
            close = perp_data["close"].iloc[-1]
            basis_pct = (mark - close) / close * 100
            return self.create_result(float(basis_pct), signal_time)

        if spot_data is not None:
            spot_price = spot_data["close"].iloc[-1]
            perp_price = perp_data["close"].iloc[-1]
            basis_pct = (spot_price - perp_price) / spot_price * 100
            return self.create_result(float(basis_pct), signal_time)

        return self.create_invalid_result(signal_time, "Missing spot data")


class BasisMA24H(BaseFactor):
    """24小时基差均线"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_ma_24h",
            name="基差24小时均线",
            family=FactorFamily.DERIVATIVES,
            description="基差的24小时移动平均",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        basis_data = data.get("basis")

        if basis_data is not None and len(basis_data) >= BARS_PER_DAY:
            basis_values = basis_data["basis"].values
            ma_24h = np.mean(basis_values[-BARS_PER_DAY:])
            return self.create_result(float(ma_24h), signal_time)

        # 如果没有预计算的basis数据，尝试计算
        perp_data = data.get("klines_5m")
        if perp_data is not None and "mark_price" in perp_data.columns:
            if len(perp_data) >= BARS_PER_DAY:
                marks = perp_data["mark_price"].values[-BARS_PER_DAY:]
                closes = perp_data["close"].values[-BARS_PER_DAY:]
                basis = (marks - closes) / closes * 10000
                return self.create_result(float(np.mean(basis)), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient data")


class BasisZScore(BaseFactor):
    """基差Z分数"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_zscore",
            name="基差Z分数",
            family=FactorFamily.DERIVATIVES,
            description="基差相对于历史的标准化分数",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        basis_data = data.get("basis")

        if basis_data is not None and len(basis_data) >= BARS_PER_DAY * 7:
            basis_values = basis_data["basis"].values
            current = basis_values[-1]
            mean = np.mean(basis_values)
            std = np.std(basis_values)
            z = (current - mean) / std if std > 0 else 0
            return self.create_result(float(z), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient basis history")


class BasisMomentum(BaseFactor):
    """基差动量"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_momentum",
            name="基差动量",
            family=FactorFamily.DERIVATIVES,
            description="基差的4小时变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR * 4,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        basis_data = data.get("basis")

        if basis_data is not None and len(basis_data) >= BARS_PER_HOUR * 4:
            basis_values = basis_data["basis"].values
            current = basis_values[-1]
            prev = basis_values[-BARS_PER_HOUR * 4]
            momentum = current - prev
            return self.create_result(float(momentum), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient data")


class AnnualizedBasis(BaseFactor):
    """年化基差收益率"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="annualized_basis",
            name="年化基差",
            family=FactorFamily.DERIVATIVES,
            description="基差年化收益率 (APR)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        perp_data = data.get("klines_5m")

        if perp_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if "mark_price" in perp_data.columns:
            mark = perp_data["mark_price"].iloc[-1]
            close = perp_data["close"].iloc[-1]
            basis_pct = (mark - close) / close
            # 假设资金费率结算周期为8小时，年化 = 基差 * 365 * 3
            annualized = basis_pct * 365 * 3 * 100
            return self.create_result(float(annualized), signal_time)

        return self.create_invalid_result(signal_time, "Missing mark_price")


class BasisFundingCorr(BaseFactor):
    """基差与资金费率相关性"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_funding_corr",
            name="基差资金费率相关性",
            family=FactorFamily.DERIVATIVES,
            description="基差与资金费率的30天相关系数",
            data_dependencies=[DataDependency.KLINES_5M, DataDependency.FUNDING_RATE],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        basis_data = data.get("basis")
        funding_data = data.get("funding_rate")

        if basis_data is None or funding_data is None:
            return self.create_invalid_result(signal_time, "Missing data")

        if len(basis_data) < 30 or len(funding_data) < 30:
            return self.create_invalid_result(signal_time, "Insufficient history")

        # 对齐时间序列并计算相关性
        basis_values = basis_data["basis"].values[-30:]
        funding_values = funding_data["funding_rate"].values[-30:]

        if len(basis_values) != len(funding_values):
            return self.create_invalid_result(signal_time, "Misaligned data")

        corr = np.corrcoef(basis_values, funding_values)[0, 1]
        return self.create_result(float(corr) if not np.isnan(corr) else 0, signal_time)


class BasisExtreme(BaseFactor):
    """基差极端值指标"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="basis_extreme",
            name="基差极端值",
            family=FactorFamily.DERIVATIVES,
            description="基差是否处于历史极端水平 (-1/0/+1)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 30,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        basis_data = data.get("basis")

        if basis_data is None or len(basis_data) < BARS_PER_DAY * 7:
            return self.create_invalid_result(signal_time, "Insufficient data")

        basis_values = basis_data["basis"].values
        current = basis_values[-1]
        p5 = np.percentile(basis_values, 5)
        p95 = np.percentile(basis_values, 95)

        if current < p5:
            extreme = -1.0  # 极端负基差
        elif current > p95:
            extreme = 1.0   # 极端正基差
        else:
            extreme = 0.0   # 正常范围

        return self.create_result(float(extreme), signal_time)


class CrossExchangeSpread(BaseFactor):
    """跨交易所价差"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="cross_exchange_spread",
            name="跨交易所价差",
            family=FactorFamily.DERIVATIVES,
            description="不同交易所间的永续合约价差 (bps)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_exchange = data.get("multi_exchange_prices")

        if multi_exchange is None:
            return self.create_invalid_result(signal_time, "Missing multi-exchange data")

        if "cross_exchange_spread_perp" in multi_exchange.columns:
            spread = multi_exchange["cross_exchange_spread_perp"].iloc[-1]
            return self.create_result(float(spread), signal_time)

        # 如果有各交易所价格，计算最大价差
        price_cols = [c for c in multi_exchange.columns if "price" in c.lower()]
        if len(price_cols) >= 2:
            prices = multi_exchange[price_cols].iloc[-1].values
            spread_bps = (max(prices) - min(prices)) / min(prices) * 10000
            return self.create_result(float(spread_bps), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient exchange data")


class BinancePremium(BaseFactor):
    """币安溢价"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="binance_premium",
            name="币安溢价",
            family=FactorFamily.DERIVATIVES,
            description="币安相对于其他交易所的溢价 (bps)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_exchange = data.get("multi_exchange_prices")

        if multi_exchange is None:
            return self.create_invalid_result(signal_time, "Missing data")

        # 假设有 binance_price 和 consensus_price 列
        if "binance_price" in multi_exchange.columns:
            binance = multi_exchange["binance_price"].iloc[-1]
            consensus = multi_exchange.get("consensus_price", multi_exchange.iloc[-1].mean())
            if hasattr(consensus, 'iloc'):
                consensus = consensus.iloc[-1]
            premium_bps = (binance - consensus) / consensus * 10000
            return self.create_result(float(premium_bps), signal_time)

        return self.create_invalid_result(signal_time, "Missing binance price")


class InsuranceFundChange(BaseFactor):
    """保险基金变化"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="insurance_fund_change",
            name="保险基金变化",
            family=FactorFamily.DERIVATIVES,
            description="交易所保险基金的日变化率",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY * 2,
            history_tier=HistoryTier.A,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        insurance_data = data.get("insurance_fund")

        if insurance_data is None or len(insurance_data) < 2:
            return self.create_invalid_result(signal_time, "Missing insurance fund data")

        current = insurance_data["fund_balance"].iloc[-1]
        prev = insurance_data["fund_balance"].iloc[-2]

        change = (current - prev) / prev if prev > 0 else 0
        return self.create_result(float(change), signal_time)


class MarketDepthRatio(BaseFactor):
    """市场深度比 (买/卖)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="market_depth_ratio",
            name="市场深度比",
            family=FactorFamily.DERIVATIVES,
            description="买盘深度与卖盘深度的比值",
            data_dependencies=[DataDependency.ORDER_BOOK],
            window=1,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        orderbook = data.get("order_book")

        if orderbook is None:
            return self.create_invalid_result(signal_time, "Missing orderbook data")

        # 假设有 bid_depth 和 ask_depth 列
        if "bid_depth" in orderbook.columns and "ask_depth" in orderbook.columns:
            bid = orderbook["bid_depth"].iloc[-1]
            ask = orderbook["ask_depth"].iloc[-1]
            ratio = bid / ask if ask > 0 else 1.0
            return self.create_result(float(ratio), signal_time)

        # 或者计算前N档深度
        bid_cols = [c for c in orderbook.columns if c.startswith("bsize")]
        ask_cols = [c for c in orderbook.columns if c.startswith("asize")]

        if bid_cols and ask_cols:
            bid_depth = orderbook[bid_cols].iloc[-1].sum()
            ask_depth = orderbook[ask_cols].iloc[-1].sum()
            ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0
            return self.create_result(float(ratio), signal_time)

        return self.create_invalid_result(signal_time, "Insufficient depth data")


class ExchangeDominance(BaseFactor):
    """交易所主导地位"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="exchange_dominance",
            name="交易所主导地位",
            family=FactorFamily.DERIVATIVES,
            description="主导交易所的成交量占比",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_exchange = data.get("multi_exchange_volume")

        if multi_exchange is None:
            # 默认返回1.0（单一交易所）
            return self.create_result(1.0, signal_time)

        volume_cols = [c for c in multi_exchange.columns if "volume" in c.lower()]
        if len(volume_cols) >= 2:
            volumes = multi_exchange[volume_cols].iloc[-1].values
            total = sum(volumes)
            dominance = max(volumes) / total if total > 0 else 1.0
            return self.create_result(float(dominance), signal_time)

        return self.create_result(1.0, signal_time)


class VolumeConcentration(BaseFactor):
    """成交量集中度 (Herfindahl Index)"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="volume_concentration",
            name="成交量集中度",
            family=FactorFamily.DERIVATIVES,
            description="跨交易所成交量的Herfindahl指数",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_DAY,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_exchange = data.get("multi_exchange_volume")

        if multi_exchange is None:
            return self.create_result(1.0, signal_time)  # 单一交易所

        volume_cols = [c for c in multi_exchange.columns if "volume" in c.lower()]
        if len(volume_cols) >= 2:
            volumes = multi_exchange[volume_cols].iloc[-1].values
            total = sum(volumes)
            if total > 0:
                shares = volumes / total
                hhi = sum(s ** 2 for s in shares)
                return self.create_result(float(hhi), signal_time)

        return self.create_result(1.0, signal_time)


class PriceDiscoveryLeader(BaseFactor):
    """价格发现领导者"""

    def get_metadata(self) -> FactorMetadata:
        return FactorMetadata(
            factor_id="price_discovery_leader",
            name="价格发现领导者",
            family=FactorFamily.DERIVATIVES,
            description="价格变化领先的交易所 (1=Binance, 2=Bybit, 3=OKX)",
            data_dependencies=[DataDependency.KLINES_5M],
            window=BARS_PER_HOUR,
            history_tier=HistoryTier.C,
        )

    def compute(self, data: Dict[str, pd.DataFrame], signal_time: datetime) -> FactorResult:
        multi_exchange = data.get("multi_exchange_prices")

        if multi_exchange is None:
            return self.create_result(1.0, signal_time)  # 默认Binance

        # 计算各交易所价格变化幅度
        leader_map = {"binance": 1, "bybit": 2, "okx": 3}
        max_change = 0
        leader = 1

        for exchange, code in leader_map.items():
            col = f"{exchange}_price"
            if col in multi_exchange.columns and len(multi_exchange) >= 2:
                change = abs(multi_exchange[col].iloc[-1] - multi_exchange[col].iloc[-2])
                if change > max_change:
                    max_change = change
                    leader = code

        return self.create_result(float(leader), signal_time)


# 导出所有因子
DERIVATIVES_FACTORS = [
    Basis,
    BasisPercentage,
    BasisMA24H,
    BasisZScore,
    BasisMomentum,
    AnnualizedBasis,
    BasisFundingCorr,
    BasisExtreme,
    CrossExchangeSpread,
    BinancePremium,
    InsuranceFundChange,
    MarketDepthRatio,
    ExchangeDominance,
    VolumeConcentration,
    PriceDiscoveryLeader,
]
