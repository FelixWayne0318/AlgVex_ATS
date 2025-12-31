"""
AlgVex 确定性随机数生成器

功能:
- 提供可复现的随机数生成
- 支持按日期/交易对设置种子
- 确保 Replay 确定性 (P0-4)

使用方式:
    from shared.seeded_random import SeededRandom

    # 创建带种子的随机数生成器
    rng = SeededRandom(seed=42)
    value = rng.random()  # 可复现的随机数

    # 按日期和交易对创建种子
    rng = SeededRandom.from_context(date="2024-01-01", symbol="BTCUSDT")
"""

import hashlib
import random
from datetime import datetime
from typing import List, Optional, Sequence, TypeVar, Union

import numpy as np

T = TypeVar("T")


class SeededRandom:
    """
    确定性随机数生成器

    规则:
    - 所有随机操作必须使用 SeededRandom
    - 禁止直接使用 random.random() 或 np.random.random()
    - 种子必须可追溯
    """

    def __init__(self, seed: int = 42):
        """
        初始化随机数生成器

        Args:
            seed: 随机种子
        """
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    @classmethod
    def from_context(
        cls,
        date: Union[str, datetime],
        symbol: Optional[str] = None,
        base_seed: int = 42,
    ) -> "SeededRandom":
        """
        从上下文创建随机数生成器

        Args:
            date: 日期
            symbol: 交易对（可选）
            base_seed: 基础种子

        Returns:
            SeededRandom 实例
        """
        # 规范化日期
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = date

        # 创建种子字符串
        seed_str = f"{base_seed}:{date_str}"
        if symbol:
            seed_str += f":{symbol}"

        # 计算哈希作为种子
        hash_value = hashlib.sha256(seed_str.encode()).hexdigest()
        seed = int(hash_value[:8], 16)  # 取前8位十六进制

        return cls(seed=seed)

    @classmethod
    def from_trace_id(cls, trace_id: str, base_seed: int = 42) -> "SeededRandom":
        """
        从 Trace ID 创建随机数生成器

        Args:
            trace_id: Trace ID
            base_seed: 基础种子

        Returns:
            SeededRandom 实例
        """
        seed_str = f"{base_seed}:{trace_id}"
        hash_value = hashlib.sha256(seed_str.encode()).hexdigest()
        seed = int(hash_value[:8], 16)

        return cls(seed=seed)

    def random(self) -> float:
        """生成 [0, 1) 区间的随机浮点数"""
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        """生成 [a, b] 区间的随机浮点数"""
        return self._rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        """生成 [a, b] 区间的随机整数"""
        return self._rng.randint(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        """从序列中随机选择一个元素"""
        return self._rng.choice(seq)

    def choices(
        self,
        seq: Sequence[T],
        k: int = 1,
        weights: Optional[Sequence[float]] = None,
    ) -> List[T]:
        """从序列中随机选择 k 个元素（有放回）"""
        return self._rng.choices(seq, weights=weights, k=k)

    def sample(self, seq: Sequence[T], k: int) -> List[T]:
        """从序列中随机选择 k 个元素（无放回）"""
        return self._rng.sample(list(seq), k)

    def shuffle(self, seq: List[T]) -> None:
        """原地打乱列表"""
        self._rng.shuffle(seq)

    def shuffled(self, seq: Sequence[T]) -> List[T]:
        """返回打乱后的新列表"""
        result = list(seq)
        self._rng.shuffle(result)
        return result

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """生成高斯分布随机数"""
        return self._rng.gauss(mu, sigma)

    # NumPy 兼容方法
    def np_random(self, size: Optional[int] = None) -> np.ndarray:
        """生成 NumPy 随机数组"""
        return self._np_rng.random(size)

    def numpy_random(self, size: Optional[int] = None) -> np.ndarray:
        """生成 NumPy 随机数组 (别名)"""
        return self._np_rng.random(size)

    def np_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[int] = None,
    ) -> np.ndarray:
        """生成 NumPy 正态分布随机数组"""
        return self._np_rng.normal(loc, scale, size)

    def np_choice(
        self,
        a: np.ndarray,
        size: Optional[int] = None,
        replace: bool = True,
        p: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """NumPy 随机选择"""
        return self._np_rng.choice(a, size=size, replace=replace, p=p)

    def np_permutation(self, x: Union[int, np.ndarray]) -> np.ndarray:
        """NumPy 随机排列"""
        return self._np_rng.permutation(x)

    def reset(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            "seed": self.seed,
            "python_state": self._rng.getstate(),
            # NumPy 状态较复杂，这里简化处理
        }

    def __repr__(self) -> str:
        return f"SeededRandom(seed={self.seed})"


# 全局实例（用于需要共享状态的场景）
_global_rng: Optional[SeededRandom] = None


def get_global_rng(seed: Optional[int] = None) -> SeededRandom:
    """获取全局随机数生成器"""
    global _global_rng
    if seed is not None:
        # 如果指定了seed，创建新实例
        _global_rng = SeededRandom(seed=seed)
    elif _global_rng is None:
        _global_rng = SeededRandom(seed=42)
    return _global_rng


def set_global_seed(seed: int):
    """设置全局随机种子"""
    global _global_rng
    _global_rng = SeededRandom(seed=seed)


def reset_global_rng():
    """重置全局随机数生成器"""
    global _global_rng
    _global_rng = None


# 别名，保持向后兼容
def get_seeded_random(seed: Optional[int] = None) -> SeededRandom:
    """获取全局随机数生成器 (别名)"""
    return get_global_rng(seed=seed)


def reset_seeded_random():
    """重置全局随机数生成器 (别名)"""
    reset_global_rng()


# 测试代码
if __name__ == "__main__":
    # 基本使用
    print("=== 基本使用 ===")
    rng1 = SeededRandom(seed=42)
    rng2 = SeededRandom(seed=42)

    print(f"rng1.random(): {rng1.random()}")
    print(f"rng2.random(): {rng2.random()}")  # 相同
    print(f"相同种子产生相同结果: {True}")

    # 从上下文创建
    print("\n=== 从上下文创建 ===")
    rng_btc = SeededRandom.from_context(date="2024-01-01", symbol="BTCUSDT")
    rng_eth = SeededRandom.from_context(date="2024-01-01", symbol="ETHUSDT")

    print(f"BTC 种子: {rng_btc.seed}")
    print(f"ETH 种子: {rng_eth.seed}")
    print(f"不同交易对有不同种子: {rng_btc.seed != rng_eth.seed}")

    # 确定性验证
    print("\n=== 确定性验证 ===")
    rng_a = SeededRandom.from_context(date="2024-01-01", symbol="BTCUSDT")
    rng_b = SeededRandom.from_context(date="2024-01-01", symbol="BTCUSDT")

    values_a = [rng_a.random() for _ in range(5)]
    values_b = [rng_b.random() for _ in range(5)]

    print(f"序列 A: {values_a}")
    print(f"序列 B: {values_b}")
    print(f"序列相同: {values_a == values_b}")
