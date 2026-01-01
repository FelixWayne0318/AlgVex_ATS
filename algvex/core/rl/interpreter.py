# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 解释器 - Qlib 0.9.7 原版复刻

"""
解释器模块 (Qlib 原版)

Interpreter is a media between states produced by simulators and states needed by RL policies.
Interpreters are two-way:

1. From simulator state to policy state (aka observation), see StateInterpreter.
2. From policy action to action accepted by simulator, see ActionInterpreter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

# 类型定义
StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")
PolicyActType = TypeVar("PolicyActType")
ActType = TypeVar("ActType")


class Interpreter(ABC):
    """
    解释器基类 (Qlib 原版)

    Interpreter is a media between states produced by simulators and states needed by RL policies.
    Interpreters are two-way:

    1. From simulator state to policy state (aka observation), see StateInterpreter.
    2. From policy action to action accepted by simulator, see ActionInterpreter.

    Inherit one of the two sub-classes to define your own interpreter.
    This super-class is only used for isinstance check.

    Interpreters are recommended to be stateless, meaning that storing temporary information with ``self.xxx``
    in interpreter is anti-pattern.
    """
    pass


class StateInterpreter(Generic[StateType, ObsType], Interpreter):
    """
    状态解释器 (Qlib 原版)

    State Interpreter that interpret execution result of simulator into rl env state
    """

    @property
    def observation_space(self) -> Any:
        """
        定义观察空间

        Returns
        -------
        observation space (可以是 gym.Space 或自定义)
        """
        raise NotImplementedError()

    def __call__(self, simulator_state: StateType) -> ObsType:
        """调用解释器"""
        obs = self.interpret(simulator_state)
        return obs

    @abstractmethod
    def interpret(self, simulator_state: StateType) -> ObsType:
        """
        解释模拟器状态

        Interpret the state of simulator.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.

        Returns
        -------
        State needed by policy. Should conform with the state space defined in ``observation_space``.
        """
        raise NotImplementedError("interpret is not implemented!")


class ActionInterpreter(Generic[StateType, PolicyActType, ActType], Interpreter):
    """
    动作解释器 (Qlib 原版)

    Action Interpreter that interpret rl agent action into simulator actions
    """

    @property
    def action_space(self) -> Any:
        """
        定义动作空间

        Returns
        -------
        action space (可以是 gym.Space 或自定义)
        """
        raise NotImplementedError()

    def __call__(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        """调用解释器"""
        return self.interpret(simulator_state, action)

    @abstractmethod
    def interpret(self, simulator_state: StateType, action: PolicyActType) -> ActType:
        """
        转换策略动作

        Convert the policy action to simulator action.

        Parameters
        ----------
        simulator_state
            Retrieved with ``simulator.get_state()``.
        action
            Raw action given by policy.

        Returns
        -------
        The action needed by simulator.
        """
        raise NotImplementedError("interpret is not implemented!")


# ============================================================
# 预定义解释器 (AlgVex 扩展)
# ============================================================

class TradingStateInterpreter(StateInterpreter[dict, np.ndarray]):
    """
    交易状态解释器

    将交易状态转换为神经网络输入
    """

    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim

    @property
    def observation_space(self) -> dict:
        return {
            "shape": (self.feature_dim,),
            "dtype": np.float32,
            "low": -np.inf,
            "high": np.inf,
        }

    def interpret(self, simulator_state: dict) -> np.ndarray:
        """将模拟器状态转换为观察向量"""
        # 提取特征
        features = []

        # 资本状态
        capital = simulator_state.get("capital", 0)
        initial_capital = simulator_state.get("initial_capital", capital)
        features.append(capital / max(initial_capital, 1))  # 归一化资本

        # 收益状态
        pnl = simulator_state.get("pnl", 0)
        features.append(pnl / max(initial_capital, 1))  # 归一化收益

        # 持仓状态
        positions = simulator_state.get("positions", {})
        position_value = sum(positions.values()) if positions else 0
        features.append(position_value)

        # 填充剩余特征
        while len(features) < self.feature_dim:
            features.append(0)

        return np.array(features[:self.feature_dim], dtype=np.float32)


class TradingActionInterpreter(ActionInterpreter[dict, int, dict]):
    """
    交易动作解释器

    将离散动作转换为交易指令
    """

    def __init__(
        self,
        symbols: list = None,
        action_space_size: int = 3,
    ):
        self.symbols = symbols or ["BTC"]
        self.action_space_size = action_space_size  # 0: hold, 1: buy, 2: sell

    @property
    def action_space(self) -> dict:
        return {
            "type": "discrete",
            "n": self.action_space_size,
        }

    def interpret(self, simulator_state: dict, action: int) -> dict:
        """将离散动作转换为交易指令"""
        symbol = self.symbols[0] if self.symbols else "BTC"
        price = simulator_state.get("current_price", 0)
        capital = simulator_state.get("capital", 0)

        if action == 0:  # 持有
            return {"symbol": symbol, "direction": 0, "amount": 0, "price": price}
        elif action == 1:  # 买入
            amount = (capital * 0.1) / max(price, 1)  # 用 10% 资本买入
            return {"symbol": symbol, "direction": 1, "amount": amount, "price": price}
        elif action == 2:  # 卖出
            positions = simulator_state.get("positions", {})
            amount = positions.get(symbol, 0) * 0.5  # 卖出 50% 持仓
            return {"symbol": symbol, "direction": -1, "amount": amount, "price": price}
        else:
            return {"symbol": symbol, "direction": 0, "amount": 0, "price": price}


class ContinuousActionInterpreter(ActionInterpreter[dict, np.ndarray, dict]):
    """
    连续动作解释器

    将连续动作向量转换为交易指令
    """

    def __init__(self, symbols: list = None):
        self.symbols = symbols or ["BTC"]

    @property
    def action_space(self) -> dict:
        return {
            "type": "continuous",
            "shape": (2,),  # [direction, amount_ratio]
            "low": np.array([-1.0, 0.0]),
            "high": np.array([1.0, 1.0]),
        }

    def interpret(self, simulator_state: dict, action: np.ndarray) -> dict:
        """将连续动作转换为交易指令"""
        symbol = self.symbols[0] if self.symbols else "BTC"
        price = simulator_state.get("current_price", 0)
        capital = simulator_state.get("capital", 0)
        positions = simulator_state.get("positions", {})

        direction_raw = action[0]  # -1 to 1
        amount_ratio = np.clip(action[1], 0, 1)  # 0 to 1

        if direction_raw > 0.3:  # 买入阈值
            direction = 1
            amount = (capital * amount_ratio) / max(price, 1)
        elif direction_raw < -0.3:  # 卖出阈值
            direction = -1
            amount = positions.get(symbol, 0) * amount_ratio
        else:  # 持有
            direction = 0
            amount = 0

        return {
            "symbol": symbol,
            "direction": direction,
            "amount": amount,
            "price": price,
        }


__all__ = [
    "Interpreter",
    "StateInterpreter",
    "ActionInterpreter",
    "TradingStateInterpreter",
    "TradingActionInterpreter",
    "ContinuousActionInterpreter",
]
