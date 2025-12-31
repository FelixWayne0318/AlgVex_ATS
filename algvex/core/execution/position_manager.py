"""
AlgVex 仓位管理器
智能仓位分配和再平衡
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RebalanceMethod(Enum):
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    RISK_PARITY = "risk_parity"  # 风险平价
    SIGNAL_WEIGHT = "signal_weight"  # 信号加权
    VOLATILITY_TARGET = "volatility_target"  # 波动率目标


@dataclass
class PositionTarget:
    """目标仓位"""
    symbol: str
    target_weight: float  # 目标权重 (0-1)
    target_value: float  # 目标价值
    current_value: float  # 当前价值
    delta: float  # 需要调整的价值
    side: str  # 方向 (long/short)
    signal_score: float  # 信号分数


class PositionManager:
    """
    仓位管理器

    功能:
    1. 根据信号计算目标仓位
    2. 生成再平衡订单
    3. 资金效率优化
    4. 多策略仓位协调
    """

    def __init__(
        self,
        total_capital: float,
        max_positions: int = 10,
        min_position_weight: float = 0.05,  # 最小仓位权重 5%
        max_position_weight: float = 0.25,  # 最大仓位权重 25%
        rebalance_threshold: float = 0.05,  # 再平衡阈值 5%
    ):
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.min_position_weight = min_position_weight
        self.max_position_weight = max_position_weight
        self.rebalance_threshold = rebalance_threshold

        self._current_positions: Dict[str, Dict] = {}
        self._target_positions: Dict[str, PositionTarget] = {}

        logger.info(f"PositionManager initialized with ${total_capital:,.0f}")

    def calculate_targets(
        self,
        signals: List[Dict],
        method: RebalanceMethod = RebalanceMethod.SIGNAL_WEIGHT,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionTarget]:
        """
        根据信号计算目标仓位

        Args:
            signals: 信号列表 [{"symbol": "BTCUSDT", "score": 0.8}, ...]
            method: 权重分配方法
            volatilities: 波动率字典 (用于风险平价)
        """
        # 过滤弱信号
        valid_signals = [s for s in signals if abs(s.get("score", 0)) >= 0.3]

        if not valid_signals:
            return {}

        # 按信号强度排序
        sorted_signals = sorted(
            valid_signals,
            key=lambda x: abs(x.get("score", 0)),
            reverse=True
        )[:self.max_positions]

        # 计算权重
        if method == RebalanceMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(sorted_signals)
        elif method == RebalanceMethod.RISK_PARITY:
            weights = self._risk_parity(sorted_signals, volatilities or {})
        elif method == RebalanceMethod.SIGNAL_WEIGHT:
            weights = self._signal_weight(sorted_signals)
        elif method == RebalanceMethod.VOLATILITY_TARGET:
            weights = self._volatility_target(sorted_signals, volatilities or {})
        else:
            weights = self._equal_weight(sorted_signals)

        # 生成目标仓位
        targets = {}
        for signal, weight in zip(sorted_signals, weights):
            symbol = signal["symbol"]
            score = signal.get("score", 0)
            current = self._current_positions.get(symbol, {})
            current_value = current.get("value", 0)

            target_value = self.total_capital * weight
            delta = target_value - current_value

            targets[symbol] = PositionTarget(
                symbol=symbol,
                target_weight=weight,
                target_value=target_value,
                current_value=current_value,
                delta=delta,
                side="long" if score > 0 else "short",
                signal_score=score,
            )

        self._target_positions = targets
        return targets

    def _equal_weight(self, signals: List[Dict]) -> List[float]:
        """等权重分配"""
        n = len(signals)
        weight = min(1.0 / n, self.max_position_weight)
        return [weight] * n

    def _signal_weight(self, signals: List[Dict]) -> List[float]:
        """信号强度加权"""
        scores = [abs(s.get("score", 0)) for s in signals]
        total_score = sum(scores)

        if total_score == 0:
            return self._equal_weight(signals)

        weights = []
        for score in scores:
            raw_weight = score / total_score
            # 限制在 min 和 max 之间
            clamped = max(self.min_position_weight,
                         min(raw_weight, self.max_position_weight))
            weights.append(clamped)

        # 归一化
        total = sum(weights)
        return [w / total for w in weights]

    def _risk_parity(
        self,
        signals: List[Dict],
        volatilities: Dict[str, float],
    ) -> List[float]:
        """风险平价"""
        # 获取波动率
        vols = []
        for s in signals:
            symbol = s["symbol"]
            vol = volatilities.get(symbol, 0.02)  # 默认 2% 日波动
            vols.append(vol)

        # 波动率倒数作为权重
        inv_vols = [1 / v if v > 0 else 1 for v in vols]
        total = sum(inv_vols)

        weights = []
        for inv_vol in inv_vols:
            raw_weight = inv_vol / total
            clamped = max(self.min_position_weight,
                         min(raw_weight, self.max_position_weight))
            weights.append(clamped)

        # 归一化
        total = sum(weights)
        return [w / total for w in weights]

    def _volatility_target(
        self,
        signals: List[Dict],
        volatilities: Dict[str, float],
        target_vol: float = 0.15,  # 目标年化波动率 15%
    ) -> List[float]:
        """波动率目标"""
        # 先用等权重，然后根据波动率调整杠杆
        base_weights = self._equal_weight(signals)

        weights = []
        for signal, base_weight in zip(signals, base_weights):
            symbol = signal["symbol"]
            vol = volatilities.get(symbol, 0.02)
            annual_vol = vol * (365 ** 0.5)  # 年化

            # 波动率目标调整
            vol_scalar = target_vol / annual_vol if annual_vol > 0 else 1
            adjusted = base_weight * vol_scalar

            clamped = max(self.min_position_weight,
                         min(adjusted, self.max_position_weight))
            weights.append(clamped)

        total = sum(weights)
        return [w / total for w in weights] if total > 0 else base_weights

    def generate_rebalance_orders(
        self,
        current_prices: Dict[str, float],
    ) -> List[Dict]:
        """
        生成再平衡订单

        Args:
            current_prices: 当前价格

        Returns:
            订单列表
        """
        orders = []

        for symbol, target in self._target_positions.items():
            price = current_prices.get(symbol)
            if not price:
                continue

            # 计算需要调整的金额
            delta_value = target.delta

            # 检查是否超过再平衡阈值
            if abs(delta_value) < self.total_capital * self.rebalance_threshold:
                continue

            # 计算数量
            quantity = abs(delta_value) / price

            order = {
                "symbol": symbol,
                "side": "buy" if delta_value > 0 else "sell",
                "quantity": quantity,
                "value": abs(delta_value),
                "target_weight": target.target_weight,
                "signal_score": target.signal_score,
                "position_side": target.side,
            }

            orders.append(order)

        # 按优先级排序：先平仓再开仓
        orders.sort(key=lambda x: (x["side"] == "buy", -x["value"]))

        logger.info(f"Generated {len(orders)} rebalance orders")
        return orders

    def update_position(
        self,
        symbol: str,
        quantity: float,
        value: float,
        side: str,
        entry_price: float,
    ) -> None:
        """更新当前持仓"""
        self._current_positions[symbol] = {
            "quantity": quantity,
            "value": value,
            "side": side,
            "entry_price": entry_price,
            "updated_at": datetime.now(),
        }

    def remove_position(self, symbol: str) -> None:
        """移除持仓"""
        if symbol in self._current_positions:
            del self._current_positions[symbol]
        if symbol in self._target_positions:
            del self._target_positions[symbol]

    def update_capital(self, new_capital: float) -> None:
        """更新总资金"""
        self.total_capital = new_capital
        logger.info(f"Capital updated to ${new_capital:,.0f}")

    def get_utilization(self) -> float:
        """获取资金利用率"""
        total_position_value = sum(
            p.get("value", 0) for p in self._current_positions.values()
        )
        return total_position_value / self.total_capital if self.total_capital > 0 else 0

    def get_long_short_ratio(self) -> Tuple[float, float]:
        """获取多空比例"""
        long_value = sum(
            p.get("value", 0)
            for p in self._current_positions.values()
            if p.get("side") == "long"
        )
        short_value = sum(
            p.get("value", 0)
            for p in self._current_positions.values()
            if p.get("side") == "short"
        )

        total = long_value + short_value
        if total == 0:
            return 0.5, 0.5

        return long_value / total, short_value / total

    def get_status(self) -> Dict:
        """获取仓位状态"""
        long_ratio, short_ratio = self.get_long_short_ratio()

        return {
            "total_capital": self.total_capital,
            "utilization": self.get_utilization(),
            "positions_count": len(self._current_positions),
            "max_positions": self.max_positions,
            "long_ratio": long_ratio,
            "short_ratio": short_ratio,
            "current_positions": list(self._current_positions.keys()),
            "target_positions": list(self._target_positions.keys()),
        }

    def get_position_detail(self, symbol: str) -> Optional[Dict]:
        """获取持仓详情"""
        current = self._current_positions.get(symbol, {})
        target = self._target_positions.get(symbol)

        if not current and not target:
            return None

        return {
            "symbol": symbol,
            "current": current,
            "target": {
                "weight": target.target_weight,
                "value": target.target_value,
                "delta": target.delta,
            } if target else None,
        }
