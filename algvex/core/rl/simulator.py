# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 模拟器 - Qlib 0.9.7 原版复刻

"""
模拟器基类 (Qlib 原版)

Simulator 是 RL 环境的核心组件，负责：
1. 接收动作并更新状态
2. 提供当前状态
3. 判断是否结束
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

# 类型定义
InitialStateType = TypeVar("InitialStateType")
StateType = TypeVar("StateType")
ActType = TypeVar("ActType")


class Simulator(Generic[InitialStateType, StateType, ActType], ABC):
    """
    模拟器基类 (Qlib 原版)

    Simulator that resets with ``__init__``, and transits with ``step(action)``.

    To make the data-flow clear, we make the following restrictions to Simulator:

    1. The only way to modify the inner status of a simulator is by using ``step(action)``.
    2. External modules can *read* the status of a simulator by using ``simulator.get_state()``,
       and check whether the simulator is in the ending state by calling ``simulator.done()``.

    A simulator is defined to be bounded with three types:

    - *InitialStateType* that is the type of the data used to create the simulator.
    - *StateType* that is the type of the **status** (state) of the simulator.
    - *ActType* that is the type of the **action**, which is the input received in each step.

    Different simulators might share the same StateType. For example, when they are dealing with the same task,
    but with different simulation implementation. With the same type, they can safely share other components in the MDP.

    Simulators are ephemeral. The lifecycle of a simulator starts with an initial state, and ends with the trajectory.
    In another word, when the trajectory ends, simulator is recycled.

    Attributes
    ----------
    env
        A reference of env-wrapper, which could be useful in some corner cases.
    """

    env: Optional[Any] = None

    def __init__(self, initial: InitialStateType, **kwargs: Any) -> None:
        """
        初始化模拟器

        Parameters
        ----------
        initial
            初始状态数据
        **kwargs
            其他参数
        """
        pass

    @abstractmethod
    def step(self, action: ActType) -> None:
        """
        执行一步动作

        Receives an action of ActType.
        Simulator should update its internal state, and return None.
        The updated state can be retrieved with ``simulator.get_state()``.

        Parameters
        ----------
        action
            要执行的动作
        """
        raise NotImplementedError()

    @abstractmethod
    def get_state(self) -> StateType:
        """
        获取当前状态

        Returns
        -------
        StateType
            当前状态
        """
        raise NotImplementedError()

    @abstractmethod
    def done(self) -> bool:
        """
        检查是否结束

        Check whether the simulator is in a "done" state.
        When simulator is in a "done" state,
        it should no longer receives any ``step`` request.

        Returns
        -------
        bool
            是否结束
        """
        raise NotImplementedError()


class TradingSimulator(Simulator):
    """
    交易模拟器 (AlgVex 扩展)

    用于模拟加密货币交易环境
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        **kwargs,
    ):
        super().__init__(initial_capital, **kwargs)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # 持仓
        self.positions = {}
        self.trades = []

        # 状态
        self._done = False
        self._step = 0

    def step(self, action: dict) -> None:
        """
        执行交易动作

        Parameters
        ----------
        action
            交易动作，包含:
            - symbol: 交易标的
            - direction: 买入(1)/卖出(-1)/持有(0)
            - amount: 交易数量或比例
        """
        if self._done:
            return

        symbol = action.get("symbol", "BTC")
        direction = action.get("direction", 0)
        amount = action.get("amount", 0)
        price = action.get("price", 0)

        if direction != 0 and amount > 0 and price > 0:
            # 计算交易成本
            cost = price * amount * (1 + self.slippage)
            commission = cost * self.commission

            if direction > 0:  # 买入
                if self.capital >= cost + commission:
                    self.capital -= (cost + commission)
                    self.positions[symbol] = self.positions.get(symbol, 0) + amount
                    self.trades.append({
                        "step": self._step,
                        "symbol": symbol,
                        "direction": "buy",
                        "amount": amount,
                        "price": price,
                        "cost": cost + commission,
                    })
            else:  # 卖出
                if self.positions.get(symbol, 0) >= amount:
                    self.positions[symbol] -= amount
                    revenue = price * amount * (1 - self.slippage)
                    self.capital += (revenue - revenue * self.commission)
                    self.trades.append({
                        "step": self._step,
                        "symbol": symbol,
                        "direction": "sell",
                        "amount": amount,
                        "price": price,
                        "revenue": revenue - revenue * self.commission,
                    })

        self._step += 1

    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            "step": self._step,
            "capital": self.capital,
            "positions": self.positions.copy(),
            "total_trades": len(self.trades),
            "pnl": self.capital - self.initial_capital,
            "return_pct": (self.capital - self.initial_capital) / self.initial_capital,
        }

    def done(self) -> bool:
        """检查是否结束"""
        return self._done

    def set_done(self, done: bool = True) -> None:
        """设置结束状态"""
        self._done = done

    def reset(self, initial_capital: Optional[float] = None) -> dict:
        """重置模拟器"""
        self.capital = initial_capital or self.initial_capital
        self.positions = {}
        self.trades = []
        self._done = False
        self._step = 0
        return self.get_state()


__all__ = [
    "Simulator",
    "TradingSimulator",
    "InitialStateType",
    "StateType",
    "ActType",
]
