# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 策略模块 - 基于 Qlib RL 扩展

"""
强化学习策略模块

提供 PPO 和 DQN 策略实现
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Policy(ABC):
    """
    策略基类

    所有 RL 策略的基类
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Any:
        """选择动作"""
        raise NotImplementedError()

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """更新策略"""
        raise NotImplementedError()

    def save(self, path: str) -> None:
        """保存策略"""
        pass

    def load(self, path: str) -> None:
        """加载策略"""
        pass


if TORCH_AVAILABLE:

    class ActorCritic(nn.Module):
        """
        Actor-Critic 网络

        用于 PPO 策略
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            continuous: bool = False,
        ):
            super().__init__()
            self.continuous = continuous

            # 共享特征提取器
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            # Actor (策略网络)
            if continuous:
                self.actor_mean = nn.Linear(hidden_dim, action_dim)
                self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
            else:
                self.actor = nn.Linear(hidden_dim, action_dim)

            # Critic (价值网络)
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """前向传播"""
            features = self.feature(state)

            if self.continuous:
                action_mean = self.actor_mean(features)
                action_std = self.actor_log_std.exp().expand_as(action_mean)
                return action_mean, action_std
            else:
                action_probs = F.softmax(self.actor(features), dim=-1)
                return action_probs, None

        def get_value(self, state: torch.Tensor) -> torch.Tensor:
            """获取状态价值"""
            features = self.feature(state)
            return self.critic(features)

        def evaluate(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """评估动作"""
            features = self.feature(state)
            value = self.critic(features)

            if self.continuous:
                action_mean = self.actor_mean(features)
                action_std = self.actor_log_std.exp().expand_as(action_mean)
                dist = Normal(action_mean, action_std)
                action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
            else:
                action_probs = F.softmax(self.actor(features), dim=-1)
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action.squeeze()).unsqueeze(-1)
                entropy = dist.entropy().unsqueeze(-1)

            return action_log_prob, value, entropy


    class PPOPolicy(Policy):
        """
        PPO (Proximal Policy Optimization) 策略

        实现 PPO-Clip 算法
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            lr: float = 3e-4,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            n_epochs: int = 10,
            batch_size: int = 64,
            continuous: bool = False,
            device: str = "cpu",
        ):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for PPOPolicy")

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.max_grad_norm = max_grad_norm
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.continuous = continuous
            self.device = torch.device(device)

            # 网络
            self.actor_critic = ActorCritic(
                state_dim, action_dim, hidden_dim, continuous
            ).to(self.device)

            # 优化器
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

            # 经验缓冲
            self.buffer = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
                "log_probs": [],
                "values": [],
            }

        def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[Any, float, float]:
            """选择动作"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.continuous:
                    action_mean, action_std = self.actor_critic(state_tensor)
                    if deterministic:
                        action = action_mean
                    else:
                        dist = Normal(action_mean, action_std)
                        action = dist.sample()
                    log_prob = Normal(action_mean, action_std).log_prob(action).sum()
                    action = action.cpu().numpy().flatten()
                else:
                    action_probs, _ = self.actor_critic(state_tensor)
                    if deterministic:
                        action = action_probs.argmax(dim=-1)
                    else:
                        dist = Categorical(action_probs)
                        action = dist.sample()
                    log_prob = Categorical(action_probs).log_prob(action)
                    action = action.item()

                value = self.actor_critic.get_value(state_tensor)

            return action, log_prob.item(), value.item()

        def store_transition(
            self,
            state: np.ndarray,
            action: Any,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            log_prob: float,
            value: float,
        ):
            """存储经验"""
            self.buffer["states"].append(state)
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["next_states"].append(next_state)
            self.buffer["dones"].append(done)
            self.buffer["log_probs"].append(log_prob)
            self.buffer["values"].append(value)

        def compute_gae(
            self,
            rewards: List[float],
            values: List[float],
            dones: List[bool],
            next_value: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """计算 GAE (Generalized Advantage Estimation)"""
            advantages = []
            returns = []
            gae = 0

            values = values + [next_value]

            for t in reversed(range(len(rewards))):
                if dones[t]:
                    delta = rewards[t] - values[t]
                    gae = delta
                else:
                    delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                    gae = delta + self.gamma * self.gae_lambda * gae

                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])

            return np.array(advantages), np.array(returns)

        def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
            """更新策略"""
            if len(self.buffer["states"]) == 0:
                return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

            # 准备数据
            states = torch.FloatTensor(np.array(self.buffer["states"])).to(self.device)
            if self.continuous:
                actions = torch.FloatTensor(np.array(self.buffer["actions"])).to(self.device)
            else:
                actions = torch.LongTensor(np.array(self.buffer["actions"])).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.buffer["log_probs"])).to(self.device)

            # 计算 GAE
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    torch.FloatTensor(self.buffer["next_states"][-1]).unsqueeze(0).to(self.device)
                ).item()

            advantages, returns = self.compute_gae(
                self.buffer["rewards"],
                self.buffer["values"],
                self.buffer["dones"],
                next_value,
            )

            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO 更新
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0

            for _ in range(self.n_epochs):
                # Mini-batch 更新
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # 评估动作
                    log_probs, values, entropy = self.actor_critic.evaluate(
                        batch_states, batch_actions
                    )

                    # 计算比率
                    ratio = torch.exp(log_probs.squeeze() - batch_old_log_probs)

                    # PPO-Clip 损失
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 价值损失
                    value_loss = F.mse_loss(values.squeeze(), batch_returns)

                    # 熵损失
                    entropy_loss = -entropy.mean()

                    # 总损失
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                    )

                    # 优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()

            # 清空缓冲
            self.buffer = {k: [] for k in self.buffer}

            n_updates = self.n_epochs * (len(states) // self.batch_size + 1)
            return {
                "policy_loss": total_policy_loss / max(n_updates, 1),
                "value_loss": total_value_loss / max(n_updates, 1),
                "entropy": total_entropy / max(n_updates, 1),
            }

        def save(self, path: str) -> None:
            """保存策略"""
            torch.save({
                "actor_critic": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, path)

        def load(self, path: str) -> None:
            """加载策略"""
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint["actor_critic"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])


    class DQNNetwork(nn.Module):
        """
        DQN 网络
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            dueling: bool = True,
        ):
            super().__init__()
            self.dueling = dueling

            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

            if dueling:
                # Dueling DQN
                self.value_stream = nn.Linear(hidden_dim, 1)
                self.advantage_stream = nn.Linear(hidden_dim, action_dim)
            else:
                self.q_head = nn.Linear(hidden_dim, action_dim)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """前向传播"""
            features = self.feature(state)

            if self.dueling:
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            else:
                q_values = self.q_head(features)

            return q_values


    class DQNPolicy(Policy):
        """
        DQN (Deep Q-Network) 策略

        实现带有经验回放和目标网络的 DQN
        """

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            lr: float = 1e-4,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_size: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 100,
            dueling: bool = True,
            double_dqn: bool = True,
            device: str = "cpu",
        ):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for DQNPolicy")

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self.double_dqn = double_dqn
            self.device = torch.device(device)

            # 网络
            self.q_network = DQNNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
            self.target_network = DQNNetwork(state_dim, action_dim, hidden_dim, dueling).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            # 优化器
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

            # 经验回放缓冲
            self.buffer_size = buffer_size
            self.buffer = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "dones": [],
            }

            self.update_count = 0

        def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
            """选择动作 (epsilon-greedy)"""
            if not deterministic and np.random.random() < self.epsilon:
                return np.random.randint(self.action_dim)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=-1).item()

        def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
        ):
            """存储经验"""
            self.buffer["states"].append(state)
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["next_states"].append(next_state)
            self.buffer["dones"].append(done)

            # 保持缓冲区大小
            if len(self.buffer["states"]) > self.buffer_size:
                for key in self.buffer:
                    self.buffer[key].pop(0)

        def update(self, batch: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
            """更新策略"""
            if len(self.buffer["states"]) < self.batch_size:
                return {"q_loss": 0}

            # 采样 mini-batch
            indices = np.random.choice(len(self.buffer["states"]), self.batch_size, replace=False)

            states = torch.FloatTensor(np.array([self.buffer["states"][i] for i in indices])).to(self.device)
            actions = torch.LongTensor(np.array([self.buffer["actions"][i] for i in indices])).to(self.device)
            rewards = torch.FloatTensor(np.array([self.buffer["rewards"][i] for i in indices])).to(self.device)
            next_states = torch.FloatTensor(np.array([self.buffer["next_states"][i] for i in indices])).to(self.device)
            dones = torch.FloatTensor(np.array([self.buffer["dones"][i] for i in indices])).to(self.device)

            # 计算当前 Q 值
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

            # 计算目标 Q 值
            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN
                    next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
                    next_q = self.target_network(next_states).gather(1, next_actions)
                else:
                    next_q = self.target_network(next_states).max(dim=-1, keepdim=True)[0]

                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

            # 计算损失
            loss = F.smooth_l1_loss(current_q, target_q)

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            self.optimizer.step()

            # 更新目标网络
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # 衰减 epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            return {"q_loss": loss.item(), "epsilon": self.epsilon}

        def save(self, path: str) -> None:
            """保存策略"""
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            }, path)

        def load(self, path: str) -> None:
            """加载策略"""
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]


else:
    # PyTorch 不可用时的占位符
    class PPOPolicy(Policy):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PPOPolicy")

        def select_action(self, state):
            pass

        def update(self, batch):
            pass


    class DQNPolicy(Policy):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQNPolicy")

        def select_action(self, state):
            pass

        def update(self, batch):
            pass


__all__ = [
    "Policy",
    "PPOPolicy",
    "DQNPolicy",
]
