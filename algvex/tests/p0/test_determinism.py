"""
P0 验收测试: 确定性保证

验收标准:
- TimeProvider 提供一致的时间
- SeededRandom 产生可重现的随机数
- 相同输入产生相同输出
- 跨运行结果一致
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.shared.time_provider import TimeProvider, get_time_provider, reset_time_provider
from algvex.shared.seeded_random import SeededRandom, get_seeded_random, reset_seeded_random


class TestTimeProvider:
    """测试时间提供器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_time_provider()

    def test_live_mode(self):
        """测试 Live 模式"""
        provider = TimeProvider(mode="live")

        now = provider.now()
        assert isinstance(now, datetime)

        # Live 模式时间应该接近真实时间
        real_now = datetime.utcnow()
        diff = abs((now - real_now).total_seconds())
        assert diff < 1.0  # 误差应该小于 1 秒

    def test_replay_mode(self):
        """测试 Replay 模式"""
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        provider = TimeProvider(mode="replay", fixed_time=fixed_time)

        now = provider.now()
        assert now == fixed_time

        # 多次调用应该返回相同时间
        assert provider.now() == fixed_time
        assert provider.now() == fixed_time

    def test_replay_time_advance(self):
        """测试 Replay 模式时间推进"""
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        provider = TimeProvider(mode="replay", fixed_time=fixed_time)

        # 推进时间
        provider.advance_time(timedelta(minutes=5))
        assert provider.now() == fixed_time + timedelta(minutes=5)

        provider.advance_time(timedelta(hours=1))
        assert provider.now() == fixed_time + timedelta(minutes=5, hours=1)

    def test_replay_set_time(self):
        """测试 Replay 模式设置时间"""
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        provider = TimeProvider(mode="replay", fixed_time=fixed_time)

        new_time = datetime(2024, 1, 16, 8, 0, 0)
        provider.set_time(new_time)
        assert provider.now() == new_time

    def test_global_singleton(self):
        """测试全局单例"""
        reset_time_provider()

        provider1 = get_time_provider()
        provider2 = get_time_provider()

        assert provider1 is provider2

    def test_mode_transition(self):
        """测试模式切换"""
        provider = TimeProvider(mode="live")
        assert provider.mode == "live"

        # 切换到 Replay 模式
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        provider.switch_to_replay(fixed_time)

        assert provider.mode == "replay"
        assert provider.now() == fixed_time


class TestSeededRandom:
    """测试确定性随机数"""

    def setup_method(self):
        """每个测试前重置"""
        reset_seeded_random()

    def test_reproducibility(self):
        """测试可重现性"""
        seed = 42

        # 第一次运行
        rng1 = SeededRandom(seed=seed)
        values1 = [rng1.random() for _ in range(10)]

        # 第二次运行 (相同种子)
        rng2 = SeededRandom(seed=seed)
        values2 = [rng2.random() for _ in range(10)]

        assert values1 == values2

    def test_different_seeds(self):
        """测试不同种子产生不同结果"""
        rng1 = SeededRandom(seed=42)
        rng2 = SeededRandom(seed=123)

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]

        assert values1 != values2

    def test_randint(self):
        """测试整数随机数"""
        seed = 42
        rng1 = SeededRandom(seed=seed)
        rng2 = SeededRandom(seed=seed)

        values1 = [rng1.randint(0, 100) for _ in range(10)]
        values2 = [rng2.randint(0, 100) for _ in range(10)]

        assert values1 == values2

    def test_choice(self):
        """测试随机选择"""
        seed = 42
        options = ["A", "B", "C", "D", "E"]

        rng1 = SeededRandom(seed=seed)
        rng2 = SeededRandom(seed=seed)

        choices1 = [rng1.choice(options) for _ in range(10)]
        choices2 = [rng2.choice(options) for _ in range(10)]

        assert choices1 == choices2

    def test_shuffle(self):
        """测试洗牌"""
        seed = 42
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        rng1 = SeededRandom(seed=seed)
        rng2 = SeededRandom(seed=seed)

        list1 = items.copy()
        list2 = items.copy()

        rng1.shuffle(list1)
        rng2.shuffle(list2)

        assert list1 == list2

    def test_numpy_random(self):
        """测试 NumPy 随机数"""
        seed = 42

        rng1 = SeededRandom(seed=seed)
        rng2 = SeededRandom(seed=seed)

        arr1 = rng1.numpy_random(size=10)
        arr2 = rng2.numpy_random(size=10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_global_singleton(self):
        """测试全局单例"""
        reset_seeded_random()

        rng1 = get_seeded_random(seed=42)
        rng2 = get_seeded_random()

        assert rng1 is rng2


class TestDeterminismIntegration:
    """测试确定性集成"""

    def setup_method(self):
        """每个测试前重置"""
        reset_time_provider()
        reset_seeded_random()

    def test_combined_determinism(self):
        """测试时间+随机数组合确定性"""
        seed = 42
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)

        def simulate_run(seed, start_time):
            """模拟一次运行"""
            time_provider = TimeProvider(mode="replay", fixed_time=start_time)
            rng = SeededRandom(seed=seed)

            results = []
            for i in range(5):
                current_time = time_provider.now()
                random_value = rng.random()
                results.append({
                    "step": i,
                    "time": current_time.isoformat(),
                    "value": random_value,
                })
                time_provider.advance_time(timedelta(minutes=5))

            return results

        # 运行两次
        run1 = simulate_run(seed, fixed_time)
        run2 = simulate_run(seed, fixed_time)

        # 结果应该完全相同
        assert run1 == run2

    def test_different_runs_produce_different_results(self):
        """测试不同参数产生不同结果"""
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)

        rng1 = SeededRandom(seed=42)
        rng2 = SeededRandom(seed=123)

        values1 = [rng1.random() for _ in range(5)]
        values2 = [rng2.random() for _ in range(5)]

        assert values1 != values2

    def test_state_isolation(self):
        """测试状态隔离"""
        seed = 42

        # 创建两个独立的随机数生成器
        rng1 = SeededRandom(seed=seed)
        rng2 = SeededRandom(seed=seed)

        # 消耗 rng1 的一些状态
        for _ in range(100):
            rng1.random()

        # rng2 应该仍然从初始状态开始
        rng3 = SeededRandom(seed=seed)

        # rng2 和 rng3 应该产生相同的序列
        values2 = [rng2.random() for _ in range(10)]
        values3 = [rng3.random() for _ in range(10)]

        assert values2 == values3
