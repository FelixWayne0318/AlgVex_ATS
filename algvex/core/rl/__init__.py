# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 强化学习模块 - Qlib 0.9.7 原版复刻

"""
AlgVex 强化学习模块 (Qlib 原版)

核心组件:
- Simulator: 模拟器基类
- Interpreter: 状态/动作解释器
- Reward: 奖励计算
- Trainer: 训练器

使用示例:
    from algvex.core.rl import (
        Simulator,
        StateInterpreter,
        ActionInterpreter,
        Reward,
        Trainer,
    )

    # 创建自定义模拟器
    class MySimulator(Simulator):
        def step(self, action):
            ...
        def get_state(self):
            ...
        def done(self):
            ...

    # 创建奖励函数
    class MyReward(Reward):
        def reward(self, state):
            return calculate_reward(state)
"""

from .simulator import Simulator
from .interpreter import (
    Interpreter,
    StateInterpreter,
    ActionInterpreter,
)
from .reward import Reward, RewardCombination
from .trainer import Trainer, TrainerConfig
from .policy import Policy, PPOPolicy, DQNPolicy
from .env import TradingEnv, CryptoTradingEnv

__all__ = [
    # 核心组件
    "Simulator",
    "Interpreter",
    "StateInterpreter",
    "ActionInterpreter",
    "Reward",
    "RewardCombination",
    "Trainer",
    "TrainerConfig",
    # 策略
    "Policy",
    "PPOPolicy",
    "DQNPolicy",
    # 环境
    "TradingEnv",
    "CryptoTradingEnv",
]
