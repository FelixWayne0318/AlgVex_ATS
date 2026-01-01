# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 奖励模块 - Qlib 0.9.7 原版复刻

"""
奖励计算模块 (Qlib 原版)

Reward calculation component that takes a single argument: state of simulator.
Returns a real number: reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import numpy as np

SimulatorState = TypeVar("SimulatorState")


class Reward(Generic[SimulatorState], ABC):
    """
    奖励基类 (Qlib 原版)

    Reward calculation component that takes a single argument: state of simulator.
    Returns a real number: reward.

    Subclass should implement ``reward(simulator_state)`` to implement their own reward calculation recipe.
    """

    env: Optional[Any] = None

    def __call__(self, simulator_state: SimulatorState) -> float:
        """调用奖励函数"""
        return self.reward(simulator_state)

    @abstractmethod
    def reward(self, simulator_state: SimulatorState) -> float:
        """
        计算奖励

        Implement this method for your own reward.

        Parameters
        ----------
        simulator_state
            模拟器状态

        Returns
        -------
        float
            奖励值
        """
        raise NotImplementedError("Implement reward calculation recipe in `reward()`.")

    def log(self, name: str, value: Any) -> None:
        """记录日志"""
        if self.env is not None and hasattr(self.env, 'logger'):
            self.env.logger.add_scalar(name, value)


class RewardCombination(Reward):
    """
    组合奖励 (Qlib 原版)

    Combination of multiple reward.
    """

    def __init__(self, rewards: Dict[str, Tuple[Reward, float]]) -> None:
        """
        初始化组合奖励

        Parameters
        ----------
        rewards
            奖励字典，格式: {name: (reward_fn, weight)}
        """
        self.rewards = rewards

    def reward(self, simulator_state: Any) -> float:
        """计算组合奖励"""
        total_reward = 0.0
        for name, (reward_fn, weight) in self.rewards.items():
            rew = reward_fn(simulator_state) * weight
            total_reward += rew
            self.log(name, rew)
        return total_reward


# ============================================================
# 预定义奖励函数 (AlgVex 扩展)
# ============================================================

class PnLReward(Reward[dict]):
    """
    盈亏奖励

    基于收益率计算奖励
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._last_capital = None

    def reward(self, simulator_state: dict) -> float:
        """计算盈亏奖励"""
        capital = simulator_state.get("capital", 0)
        initial_capital = simulator_state.get("initial_capital", capital)

        if self._last_capital is None:
            self._last_capital = initial_capital

        # 计算单步收益率
        if self._last_capital > 0:
            step_return = (capital - self._last_capital) / self._last_capital
        else:
            step_return = 0

        self._last_capital = capital

        return step_return * self.scale

    def reset(self):
        """重置状态"""
        self._last_capital = None


class SharpeReward(Reward[dict]):
    """
    夏普比率奖励

    基于风险调整收益计算奖励
    """

    def __init__(self, window: int = 20, risk_free_rate: float = 0.0):
        self.window = window
        self.risk_free_rate = risk_free_rate
        self._returns = []

    def reward(self, simulator_state: dict) -> float:
        """计算夏普奖励"""
        capital = simulator_state.get("capital", 0)
        pnl = simulator_state.get("pnl", 0)
        initial_capital = simulator_state.get("initial_capital", capital)

        # 计算收益率
        if len(self._returns) > 0:
            last_capital = self._returns[-1][1]
            if last_capital > 0:
                ret = (capital - last_capital) / last_capital
            else:
                ret = 0
        else:
            ret = pnl / max(initial_capital, 1)

        self._returns.append((ret, capital))

        # 保持窗口大小
        if len(self._returns) > self.window:
            self._returns.pop(0)

        # 计算夏普比率
        if len(self._returns) >= 2:
            returns = [r[0] for r in self._returns]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                sharpe = (mean_ret - self.risk_free_rate) / std_ret
            else:
                sharpe = 0
            return sharpe
        else:
            return 0

    def reset(self):
        """重置状态"""
        self._returns = []


class DrawdownPenalty(Reward[dict]):
    """
    回撤惩罚

    对回撤给予惩罚
    """

    def __init__(self, penalty_scale: float = 1.0):
        self.penalty_scale = penalty_scale
        self._max_capital = None

    def reward(self, simulator_state: dict) -> float:
        """计算回撤惩罚"""
        capital = simulator_state.get("capital", 0)

        if self._max_capital is None:
            self._max_capital = capital

        # 更新最大资本
        if capital > self._max_capital:
            self._max_capital = capital

        # 计算回撤
        if self._max_capital > 0:
            drawdown = (self._max_capital - capital) / self._max_capital
        else:
            drawdown = 0

        # 回撤惩罚 (负奖励)
        return -drawdown * self.penalty_scale

    def reset(self):
        """重置状态"""
        self._max_capital = None


class TradeCostPenalty(Reward[dict]):
    """
    交易成本惩罚

    对频繁交易给予惩罚
    """

    def __init__(self, penalty_per_trade: float = 0.001):
        self.penalty_per_trade = penalty_per_trade
        self._last_trade_count = 0

    def reward(self, simulator_state: dict) -> float:
        """计算交易成本惩罚"""
        total_trades = simulator_state.get("total_trades", 0)

        # 新交易数量
        new_trades = total_trades - self._last_trade_count
        self._last_trade_count = total_trades

        # 交易成本惩罚 (负奖励)
        return -new_trades * self.penalty_per_trade

    def reset(self):
        """重置状态"""
        self._last_trade_count = 0


class PositionSizeReward(Reward[dict]):
    """
    持仓规模奖励

    鼓励适度持仓
    """

    def __init__(self, target_ratio: float = 0.5, tolerance: float = 0.2):
        self.target_ratio = target_ratio
        self.tolerance = tolerance

    def reward(self, simulator_state: dict) -> float:
        """计算持仓规模奖励"""
        capital = simulator_state.get("capital", 0)
        positions = simulator_state.get("positions", {})
        current_price = simulator_state.get("current_price", 1)

        # 计算持仓价值
        position_value = sum(v * current_price for v in positions.values())
        total_value = capital + position_value

        if total_value > 0:
            position_ratio = position_value / total_value
        else:
            position_ratio = 0

        # 计算与目标的偏差
        deviation = abs(position_ratio - self.target_ratio)

        if deviation <= self.tolerance:
            return 0.1  # 在容忍范围内给予小奖励
        else:
            return -deviation  # 偏离过大给予惩罚


class CompositeReward(RewardCombination):
    """
    复合奖励

    预定义的标准交易奖励组合
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        sharpe_weight: float = 0.5,
        drawdown_weight: float = 0.3,
        trade_cost_weight: float = 0.1,
    ):
        rewards = {
            "pnl": (PnLReward(), pnl_weight),
            "sharpe": (SharpeReward(), sharpe_weight),
            "drawdown": (DrawdownPenalty(), drawdown_weight),
            "trade_cost": (TradeCostPenalty(), trade_cost_weight),
        }
        super().__init__(rewards)

    def reset(self):
        """重置所有奖励函数"""
        for name, (reward_fn, _) in self.rewards.items():
            if hasattr(reward_fn, 'reset'):
                reward_fn.reset()


__all__ = [
    "Reward",
    "RewardCombination",
    "PnLReward",
    "SharpeReward",
    "DrawdownPenalty",
    "TradeCostPenalty",
    "PositionSizeReward",
    "CompositeReward",
]
