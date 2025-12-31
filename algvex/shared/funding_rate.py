"""
AlgVex 资金费率处理器 (P0-2)

功能:
- 处理永续合约的资金费率计算
- 精确对齐结算时间 (0/8/16 UTC)
- 计算持仓的资金费支付
- 验证回测资金费率计算

资金费率机制:
- 每8小时结算一次 (UTC 0:00, 8:00, 16:00)
- 持仓必须跨越结算时间才需支付资金费
- 正费率: 多头支付给空头
- 负费率: 空头支付给多头
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FundingPayment:
    """资金费支付记录"""
    symbol: str
    settlement_time: datetime
    funding_rate: float
    position_value: float
    payment: float  # 正数表示支付，负数表示收取
    side: str  # "long" or "short"


@dataclass
class FundingRateRecord:
    """资金费率记录"""
    symbol: str
    funding_rate: float
    funding_time: datetime  # 结算时间
    mark_price: float
    index_price: Optional[float] = None
    next_funding_time: Optional[datetime] = None


class FundingRateHandler:
    """
    资金费率处理器 - 必须精确对齐结算时间

    使用方法:
        handler = FundingRateHandler()

        # 检查是否需要支付资金费
        payment = handler.calculate_funding_payment(
            symbol="BTCUSDT",
            position_value=10000,
            side="long",
            entry_time=entry_time,
            current_time=current_time,
            funding_rate=0.0001,
        )

        # 验证回测资金费率
        is_valid = handler.validate_backtest_funding(trades)
    """

    # 标准结算时间 (UTC)
    SETTLEMENT_HOURS = [0, 8, 16]

    # 结算周期 (小时)
    SETTLEMENT_INTERVAL = 8

    def __init__(
        self,
        settlement_hours: Optional[List[int]] = None,
        max_funding_rate: float = 0.01,  # 1% 上限
    ):
        """
        初始化资金费率处理器

        Args:
            settlement_hours: 结算时间 (UTC小时列表)
            max_funding_rate: 最大资金费率
        """
        self.settlement_hours = settlement_hours or self.SETTLEMENT_HOURS
        self.max_funding_rate = max_funding_rate

        # 资金费率缓存
        self._rate_cache: Dict[str, List[FundingRateRecord]] = {}

        logger.info(f"FundingRateHandler initialized with settlement hours: {self.settlement_hours}")

    def get_next_settlement_time(self, current_time: datetime) -> datetime:
        """
        获取下一个结算时间

        Args:
            current_time: 当前时间 (必须是 UTC)

        Returns:
            下一个结算时间 (UTC)
        """
        # 确保是 UTC 时间
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        current_hour = current_time.hour
        current_date = current_time.date()

        # 找到下一个结算小时
        for hour in sorted(self.settlement_hours):
            if hour > current_hour:
                return datetime(
                    current_date.year,
                    current_date.month,
                    current_date.day,
                    hour,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
            elif hour == current_hour and current_time.minute == 0 and current_time.second == 0:
                # 正好在结算时间点
                return current_time

        # 下一个结算时间是明天的第一个结算小时
        next_date = current_date + timedelta(days=1)
        first_hour = sorted(self.settlement_hours)[0]
        return datetime(
            next_date.year,
            next_date.month,
            next_date.day,
            first_hour,
            0,
            0,
            tzinfo=timezone.utc,
        )

    def get_previous_settlement_time(self, current_time: datetime) -> datetime:
        """
        获取上一个结算时间

        Args:
            current_time: 当前时间 (必须是 UTC)

        Returns:
            上一个结算时间 (UTC)
        """
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        current_hour = current_time.hour
        current_date = current_time.date()

        # 找到上一个结算小时
        for hour in sorted(self.settlement_hours, reverse=True):
            if hour < current_hour:
                return datetime(
                    current_date.year,
                    current_date.month,
                    current_date.day,
                    hour,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
            elif hour == current_hour and current_time.minute == 0 and current_time.second == 0:
                # 正好在结算时间点，返回上一个
                continue

        # 上一个结算时间是昨天的最后一个结算小时
        prev_date = current_date - timedelta(days=1)
        last_hour = sorted(self.settlement_hours)[-1]
        return datetime(
            prev_date.year,
            prev_date.month,
            prev_date.day,
            last_hour,
            0,
            0,
            tzinfo=timezone.utc,
        )

    def get_settlement_times_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[datetime]:
        """
        获取时间范围内的所有结算时间

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            结算时间列表
        """
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        settlements = []
        current = self.get_next_settlement_time(start_time)

        while current <= end_time:
            if current > start_time:  # 不包含开始时间点
                settlements.append(current)
            current = self.get_next_settlement_time(current + timedelta(seconds=1))

        return settlements

    def position_held_through_settlement(
        self,
        entry_time: datetime,
        exit_time: Optional[datetime],
        settlement_time: datetime,
    ) -> bool:
        """
        检查持仓是否跨越了结算时间

        Args:
            entry_time: 入场时间
            exit_time: 出场时间 (None 表示仍持有)
            settlement_time: 结算时间

        Returns:
            是否跨越结算时间
        """
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        if settlement_time.tzinfo is None:
            settlement_time = settlement_time.replace(tzinfo=timezone.utc)

        # 入场必须在结算时间之前
        if entry_time >= settlement_time:
            return False

        # 如果没有出场，视为跨越
        if exit_time is None:
            return True

        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)

        # 出场必须在结算时间之后
        return exit_time > settlement_time

    def calculate_funding_payment(
        self,
        symbol: str,
        position_value: float,
        side: str,
        entry_time: datetime,
        current_time: datetime,
        funding_rate: float,
        exit_time: Optional[datetime] = None,
    ) -> Optional[FundingPayment]:
        """
        计算资金费支付

        Args:
            symbol: 交易对
            position_value: 持仓价值 (正数)
            side: 持仓方向 ("long" or "short")
            entry_time: 入场时间
            current_time: 当前时间
            funding_rate: 资金费率
            exit_time: 出场时间 (可选)

        Returns:
            资金费支付记录 (如果需要支付)
        """
        # 验证资金费率
        if abs(funding_rate) > self.max_funding_rate:
            logger.warning(
                f"Funding rate {funding_rate} exceeds max {self.max_funding_rate}"
            )

        # 获取下一个结算时间
        next_settlement = self.get_next_settlement_time(entry_time)

        # 检查是否跨越结算时间
        if not self.position_held_through_settlement(entry_time, exit_time, next_settlement):
            return None

        # 计算支付金额
        # 正费率: 多头支付，空头收取
        # 负费率: 空头支付，多头收取
        if side.lower() == "long":
            payment = position_value * funding_rate
        else:
            payment = -position_value * funding_rate

        return FundingPayment(
            symbol=symbol,
            settlement_time=next_settlement,
            funding_rate=funding_rate,
            position_value=position_value,
            payment=payment,
            side=side,
        )

    def calculate_total_funding(
        self,
        symbol: str,
        position_value: float,
        side: str,
        entry_time: datetime,
        exit_time: datetime,
        funding_rates: Dict[datetime, float],
    ) -> Tuple[float, List[FundingPayment]]:
        """
        计算持仓期间的总资金费

        Args:
            symbol: 交易对
            position_value: 持仓价值
            side: 持仓方向
            entry_time: 入场时间
            exit_time: 出场时间
            funding_rates: 各结算时间的资金费率

        Returns:
            (总资金费, 支付记录列表)
        """
        # 获取持仓期间的所有结算时间
        settlements = self.get_settlement_times_in_range(entry_time, exit_time)

        total_payment = 0.0
        payments = []

        for settlement in settlements:
            # 查找该结算时间的资金费率
            rate = funding_rates.get(settlement)
            if rate is None:
                # 尝试查找最接近的费率
                rate = self._find_closest_rate(settlement, funding_rates)
                if rate is None:
                    logger.warning(f"No funding rate found for {settlement}")
                    continue

            payment = self.calculate_funding_payment(
                symbol=symbol,
                position_value=position_value,
                side=side,
                entry_time=entry_time,
                current_time=settlement,
                funding_rate=rate,
                exit_time=exit_time,
            )

            if payment:
                total_payment += payment.payment
                payments.append(payment)

        return total_payment, payments

    def _find_closest_rate(
        self,
        target_time: datetime,
        funding_rates: Dict[datetime, float],
    ) -> Optional[float]:
        """查找最接近目标时间的资金费率"""
        if not funding_rates:
            return None

        closest_time = min(
            funding_rates.keys(),
            key=lambda t: abs((t - target_time).total_seconds()),
        )

        # 只接受8小时内的费率
        if abs((closest_time - target_time).total_seconds()) < 8 * 3600:
            return funding_rates[closest_time]

        return None

    def validate_backtest_funding(
        self,
        trades: List[Dict],
        tolerance: float = 1e-6,
    ) -> Tuple[bool, List[Dict]]:
        """
        验证回测资金费率计算是否正确

        Args:
            trades: 交易列表，每个交易需要包含:
                - entry_time, exit_time
                - position_value, side
                - funding_paid (回测计算的资金费)
                - funding_rates (各结算时间的费率)
            tolerance: 误差容忍度

        Returns:
            (是否全部正确, 错误列表)
        """
        errors = []

        for i, trade in enumerate(trades):
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            position_value = trade.get("position_value", 0)
            side = trade.get("side", "long")
            reported_funding = trade.get("funding_paid", 0)
            funding_rates = trade.get("funding_rates", {})
            symbol = trade.get("symbol", "UNKNOWN")

            if not entry_time or not exit_time:
                continue

            # 计算预期资金费
            expected_funding, _ = self.calculate_total_funding(
                symbol=symbol,
                position_value=position_value,
                side=side,
                entry_time=entry_time,
                exit_time=exit_time,
                funding_rates=funding_rates,
            )

            # 比较
            diff = abs(expected_funding - reported_funding)
            if diff > tolerance:
                errors.append({
                    "trade_index": i,
                    "symbol": symbol,
                    "expected": expected_funding,
                    "reported": reported_funding,
                    "difference": diff,
                    "entry_time": entry_time.isoformat() if entry_time else None,
                    "exit_time": exit_time.isoformat() if exit_time else None,
                })

        is_valid = len(errors) == 0
        if not is_valid:
            logger.error(f"Funding rate validation failed: {len(errors)} errors")

        return is_valid, errors

    def get_applicable_funding_rate(
        self,
        position_time: datetime,
        funding_rates: Dict[datetime, float],
    ) -> Optional[float]:
        """
        获取适用于当前持仓的资金费率

        Args:
            position_time: 持仓时间
            funding_rates: 历史资金费率

        Returns:
            适用的资金费率
        """
        next_settlement = self.get_next_settlement_time(position_time)
        return funding_rates.get(next_settlement)

    def estimate_funding_cost(
        self,
        position_value: float,
        side: str,
        holding_hours: int,
        avg_funding_rate: float = 0.0001,  # 默认 0.01%
    ) -> float:
        """
        估算持仓的资金费成本

        Args:
            position_value: 持仓价值
            side: 持仓方向
            holding_hours: 预计持仓小时数
            avg_funding_rate: 平均资金费率

        Returns:
            预计资金费成本
        """
        # 计算会经过多少个结算周期
        num_settlements = holding_hours // self.SETTLEMENT_INTERVAL

        # 计算总成本
        if side.lower() == "long":
            cost = position_value * avg_funding_rate * num_settlements
        else:
            cost = -position_value * avg_funding_rate * num_settlements

        return cost


# 全局单例
_funding_handler: Optional[FundingRateHandler] = None


def get_funding_handler() -> FundingRateHandler:
    """获取全局 FundingRateHandler 实例"""
    global _funding_handler
    if _funding_handler is None:
        _funding_handler = FundingRateHandler()
    return _funding_handler


def reset_funding_handler():
    """重置全局实例 (用于测试)"""
    global _funding_handler
    _funding_handler = None
