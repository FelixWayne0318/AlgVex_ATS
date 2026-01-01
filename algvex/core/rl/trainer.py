# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 训练器 - 基于 Qlib RL 原版

"""
强化学习训练器模块

提供 RL 策略训练功能
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from loguru import logger


@dataclass
class TrainerConfig:
    """
    训练器配置

    Parameters
    ----------
    max_episodes
        最大训练回合数
    max_steps_per_episode
        每回合最大步数
    eval_interval
        评估间隔 (回合数)
    save_interval
        保存间隔 (回合数)
    log_interval
        日志间隔 (回合数)
    save_dir
        模型保存目录
    seed
        随机种子
    """

    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_interval: int = 10
    save_interval: int = 100
    log_interval: int = 1
    save_dir: str = "./checkpoints"
    seed: Optional[int] = None
    early_stop_patience: int = 50
    early_stop_min_delta: float = 0.001


class Trainer:
    """
    强化学习训练器 (Qlib 风格)

    Utility to train a policy on a particular task.

    Different from traditional DL trainer, the iteration of this trainer is "episode",
    rather than "epoch", or "mini-batch".

    Parameters
    ----------
    config
        训练器配置
    """

    def __init__(self, config: TrainerConfig = None):
        self.config = config or TrainerConfig()

        # 训练状态
        self.current_episode = 0
        self.current_step = 0
        self.should_stop = False

        # 指标记录
        self.metrics: Dict[str, List[float]] = {
            "episode_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
        }

        # 最佳模型跟踪
        self.best_reward = float("-inf")
        self.best_episode = 0
        self.no_improve_count = 0

        # 设置随机种子
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def fit(
        self,
        env: Any,
        policy: Any,
        eval_env: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        训练策略

        Parameters
        ----------
        env
            训练环境
        policy
            要训练的策略
        eval_env
            评估环境 (可选)
        callbacks
            回调函数列表

        Returns
        -------
        Dict[str, Any]
            训练结果
        """
        logger.info(f"Starting training for {self.config.max_episodes} episodes")

        # 创建保存目录
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.should_stop = False
        callbacks = callbacks or []

        for episode in range(self.config.max_episodes):
            if self.should_stop:
                break

            self.current_episode = episode

            # 训练一个回合
            episode_metrics = self._train_episode(env, policy)

            # 记录指标
            for key, value in episode_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)

            # 日志
            if (episode + 1) % self.config.log_interval == 0:
                self._log_metrics(episode, episode_metrics)

            # 评估
            if eval_env is not None and (episode + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(eval_env, policy)
                logger.info(f"Evaluation at episode {episode + 1}: {eval_metrics}")

                # 早停检查
                if eval_metrics.get("mean_reward", 0) > self.best_reward + self.config.early_stop_min_delta:
                    self.best_reward = eval_metrics["mean_reward"]
                    self.best_episode = episode
                    self.no_improve_count = 0

                    # 保存最佳模型
                    best_path = save_dir / "best_model.pt"
                    policy.save(str(best_path))
                    logger.info(f"Saved best model with reward {self.best_reward:.4f}")
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.config.early_stop_patience:
                        logger.info(f"Early stopping at episode {episode + 1}")
                        self.should_stop = True

            # 保存检查点
            if (episode + 1) % self.config.save_interval == 0:
                ckpt_path = save_dir / f"checkpoint_{episode + 1}.pt"
                policy.save(str(ckpt_path))

            # 回调
            for callback in callbacks:
                callback(self, episode, episode_metrics)

        # 训练完成
        logger.info(f"Training completed. Best reward: {self.best_reward:.4f} at episode {self.best_episode}")

        return {
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "total_episodes": self.current_episode + 1,
            "metrics": self.metrics,
        }

    def _train_episode(self, env: Any, policy: Any) -> Dict[str, float]:
        """
        训练一个回合

        Parameters
        ----------
        env
            训练环境
        policy
            策略

        Returns
        -------
        Dict[str, float]
            回合指标
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(self.config.max_steps_per_episode):
            self.current_step = step

            # 选择动作
            if hasattr(policy, 'store_transition'):
                # PPO style
                action, log_prob, value = policy.select_action(state)
            else:
                action = policy.select_action(state)
                log_prob, value = None, None

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储经验
            if hasattr(policy, 'store_transition'):
                if log_prob is not None:
                    policy.store_transition(state, action, reward, next_state, done, log_prob, value)
                else:
                    policy.store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        # 更新策略
        update_metrics = policy.update()

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **update_metrics,
        }

    def evaluate(
        self,
        env: Any,
        policy: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估策略

        Parameters
        ----------
        env
            评估环境
        policy
            策略
        n_episodes
            评估回合数
        deterministic
            是否使用确定性策略

        Returns
        -------
        Dict[str, float]
            评估指标
        """
        rewards = []
        lengths = []

        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(self.config.max_steps_per_episode):
                action = policy.select_action(state, deterministic=deterministic)
                if isinstance(action, tuple):
                    action = action[0]

                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
        }

    def test(self, env: Any, policy: Any, n_episodes: int = 100) -> Dict[str, Any]:
        """
        测试策略

        Parameters
        ----------
        env
            测试环境
        policy
            策略
        n_episodes
            测试回合数

        Returns
        -------
        Dict[str, Any]
            测试结果
        """
        logger.info(f"Testing policy for {n_episodes} episodes")

        results = self.evaluate(env, policy, n_episodes, deterministic=True)

        logger.info(f"Test results: Mean reward = {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")

        return results

    def _log_metrics(self, episode: int, metrics: Dict[str, float]) -> None:
        """记录指标"""
        msg = f"Episode {episode + 1}: "
        msg += ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(msg)

    def state_dict(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            "current_episode": self.current_episode,
            "current_step": self.current_step,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "no_improve_count": self.no_improve_count,
            "metrics": self.metrics,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载训练状态"""
        self.current_episode = state_dict["current_episode"]
        self.current_step = state_dict["current_step"]
        self.best_reward = state_dict["best_reward"]
        self.best_episode = state_dict["best_episode"]
        self.no_improve_count = state_dict["no_improve_count"]
        self.metrics = state_dict["metrics"]


__all__ = [
    "Trainer",
    "TrainerConfig",
]
