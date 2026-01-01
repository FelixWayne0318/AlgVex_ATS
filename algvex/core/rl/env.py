# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 交易环境 - 基于 Qlib RL 扩展

"""
强化学习交易环境

提供标准 Gym 风格的交易环境
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .simulator import TradingSimulator
from .interpreter import (
    StateInterpreter,
    ActionInterpreter,
    TradingStateInterpreter,
    TradingActionInterpreter,
)
from .reward import Reward, CompositeReward


class TradingEnv:
    """
    交易环境

    标准 Gym 风格的交易环境，用于强化学习

    Parameters
    ----------
    data
        价格数据 (DataFrame or ndarray)
    initial_capital
        初始资本
    commission
        手续费率
    slippage
        滑点
    state_interpreter
        状态解释器
    action_interpreter
        动作解释器
    reward_fn
        奖励函数
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        state_interpreter: Optional[StateInterpreter] = None,
        action_interpreter: Optional[ActionInterpreter] = None,
        reward_fn: Optional[Reward] = None,
        max_steps: Optional[int] = None,
        window_size: int = 20,
    ):
        # 数据
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.columns = data.columns.tolist()
        else:
            self.data = data
            self.columns = None

        self.n_steps = len(self.data)
        self.max_steps = max_steps or self.n_steps
        self.window_size = window_size

        # 模拟器
        self.simulator = TradingSimulator(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )

        # 解释器
        self.state_interpreter = state_interpreter or TradingStateInterpreter(
            feature_dim=window_size * (self.data.shape[1] if len(self.data.shape) > 1 else 1) + 10
        )
        self.action_interpreter = action_interpreter or TradingActionInterpreter()

        # 奖励函数
        self.reward_fn = reward_fn or CompositeReward()

        # 状态
        self._current_step = 0
        self._done = False

    @property
    def observation_space(self) -> Dict[str, Any]:
        """观察空间"""
        return self.state_interpreter.observation_space

    @property
    def action_space(self) -> Dict[str, Any]:
        """动作空间"""
        return self.action_interpreter.action_space

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns
        -------
        np.ndarray
            初始观察
        """
        self._current_step = self.window_size
        self._done = False

        # 重置模拟器
        self.simulator.reset()

        # 重置奖励函数
        if hasattr(self.reward_fn, 'reset'):
            self.reward_fn.reset()

        return self._get_observation()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步

        Parameters
        ----------
        action
            动作

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # 获取当前价格
        current_price = self._get_current_price()

        # 获取模拟器状态
        sim_state = self.simulator.get_state()
        sim_state["current_price"] = current_price
        sim_state["initial_capital"] = self.simulator.initial_capital

        # 解释动作
        sim_action = self.action_interpreter(sim_state, action)
        sim_action["price"] = current_price

        # 执行动作
        self.simulator.step(sim_action)

        # 更新状态
        self._current_step += 1

        # 检查是否结束
        if self._current_step >= self.n_steps or self._current_step >= self.max_steps + self.window_size:
            self._done = True
            self.simulator.set_done(True)

        # 获取新状态
        new_sim_state = self.simulator.get_state()
        new_sim_state["current_price"] = self._get_current_price() if not self._done else current_price
        new_sim_state["initial_capital"] = self.simulator.initial_capital

        # 计算奖励
        reward = self.reward_fn(new_sim_state)

        # 获取观察
        observation = self._get_observation()

        # 信息
        info = {
            "step": self._current_step,
            "capital": new_sim_state["capital"],
            "pnl": new_sim_state["pnl"],
            "return_pct": new_sim_state["return_pct"],
            "total_trades": new_sim_state["total_trades"],
        }

        return observation, reward, self._done, info

    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        # 获取历史数据窗口
        start_idx = max(0, self._current_step - self.window_size)
        end_idx = self._current_step
        window_data = self.data[start_idx:end_idx]

        # 扁平化数据
        if len(window_data.shape) > 1:
            flat_data = window_data.flatten()
        else:
            flat_data = window_data

        # 获取模拟器状态
        sim_state = self.simulator.get_state()
        sim_state["initial_capital"] = self.simulator.initial_capital
        sim_state["current_price"] = self._get_current_price()

        # 使用状态解释器
        obs = self.state_interpreter(sim_state)

        # 如果需要，将价格数据添加到观察中
        if isinstance(obs, np.ndarray):
            # 确保观察向量大小一致
            expected_size = self.state_interpreter.observation_space.get("shape", (10,))[0]
            if len(obs) < expected_size:
                padding = np.zeros(expected_size - len(obs))
                obs = np.concatenate([flat_data[:expected_size - len(obs)], obs, padding])[:expected_size]

        return obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """获取当前价格"""
        if self._current_step >= self.n_steps:
            return self.data[-1, 0] if len(self.data.shape) > 1 else self.data[-1]

        if len(self.data.shape) > 1:
            # 假设第一列是收盘价
            return float(self.data[self._current_step, 0])
        else:
            return float(self.data[self._current_step])

    def render(self, mode: str = "human") -> None:
        """渲染环境"""
        state = self.simulator.get_state()
        print(f"Step: {self._current_step}, Capital: {state['capital']:.2f}, PnL: {state['pnl']:.2f}")

    def close(self) -> None:
        """关闭环境"""
        pass


class CryptoTradingEnv(TradingEnv):
    """
    加密货币交易环境

    专门为加密货币交易优化的环境

    Parameters
    ----------
    data
        OHLCV 数据
    symbols
        交易对列表
    initial_capital
        初始资本
    commission
        手续费率
    slippage
        滑点
    leverage
        杠杆倍数
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        symbols: List[str] = None,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: float = 1.0,
        window_size: int = 20,
        **kwargs,
    ):
        self.symbols = symbols or ["BTC"]
        self.leverage = leverage

        super().__init__(
            data=data,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            window_size=window_size,
            **kwargs,
        )

        # 更新动作解释器
        self.action_interpreter = TradingActionInterpreter(symbols=self.symbols)

    def _add_technical_features(self, data: np.ndarray) -> np.ndarray:
        """添加技术指标特征"""
        # 这里可以添加更多技术指标
        # 简单示例：计算简单移动平均
        features = [data]

        if len(data.shape) > 1 and data.shape[0] >= 5:
            # 5 周期 SMA
            sma_5 = np.mean(data[-5:, 0]) if data.shape[0] >= 5 else data[-1, 0]
            # 10 周期 SMA
            sma_10 = np.mean(data[-10:, 0]) if data.shape[0] >= 10 else data[-1, 0]
            # RSI 简化计算
            if data.shape[0] >= 14:
                changes = np.diff(data[-15:, 0])
                gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
                losses = -np.mean(changes[changes < 0]) if np.any(changes < 0) else 0
                rsi = 100 - (100 / (1 + gains / (losses + 1e-10)))
            else:
                rsi = 50

            # 添加特征
            extra_features = np.array([sma_5, sma_10, rsi])
            return np.concatenate([data.flatten(), extra_features])

        return data.flatten()


class MultiAssetTradingEnv(TradingEnv):
    """
    多资产交易环境

    支持同时交易多个资产

    Parameters
    ----------
    data
        多资产价格数据 {symbol: DataFrame}
    initial_capital
        初始资本
    commission
        手续费率
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        window_size: int = 20,
        **kwargs,
    ):
        self.symbol_data = data
        self.symbols = list(data.keys())
        self.n_assets = len(self.symbols)

        # 合并数据
        combined_data = self._combine_data(data)

        super().__init__(
            data=combined_data,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            window_size=window_size,
            **kwargs,
        )

        # 多资产动作解释器
        self.action_interpreter = MultiAssetActionInterpreter(symbols=self.symbols)

    def _combine_data(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """合并多资产数据"""
        combined = []
        for symbol in self.symbols:
            df = data[symbol]
            if isinstance(df, pd.DataFrame):
                combined.append(df.values)
            else:
                combined.append(df)

        return np.concatenate(combined, axis=1)

    def _get_asset_prices(self) -> Dict[str, float]:
        """获取各资产当前价格"""
        prices = {}
        cols_per_asset = self.data.shape[1] // self.n_assets

        for i, symbol in enumerate(self.symbols):
            idx = i * cols_per_asset
            prices[symbol] = float(self.data[self._current_step, idx])

        return prices


class MultiAssetActionInterpreter(ActionInterpreter):
    """多资产动作解释器"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.n_assets = len(symbols)

    @property
    def action_space(self) -> dict:
        return {
            "type": "continuous",
            "shape": (self.n_assets * 2,),  # 每个资产 [direction, amount]
            "low": -1.0,
            "high": 1.0,
        }

    def interpret(self, simulator_state: dict, action: np.ndarray) -> List[dict]:
        """解释多资产动作"""
        actions = []
        capital = simulator_state.get("capital", 0)
        capital_per_asset = capital / self.n_assets

        for i, symbol in enumerate(self.symbols):
            direction_raw = action[i * 2]
            amount_ratio = np.clip(action[i * 2 + 1], 0, 1)
            price = simulator_state.get("prices", {}).get(symbol, 0)

            if direction_raw > 0.3:
                direction = 1
                amount = (capital_per_asset * amount_ratio) / max(price, 1)
            elif direction_raw < -0.3:
                direction = -1
                positions = simulator_state.get("positions", {})
                amount = positions.get(symbol, 0) * amount_ratio
            else:
                direction = 0
                amount = 0

            actions.append({
                "symbol": symbol,
                "direction": direction,
                "amount": amount,
                "price": price,
            })

        return actions


__all__ = [
    "TradingEnv",
    "CryptoTradingEnv",
    "MultiAssetTradingEnv",
    "MultiAssetActionInterpreter",
]
