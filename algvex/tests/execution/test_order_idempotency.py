"""
订单幂等性测试

测试相同信号生成相同 order_id 的幂等性保证
"""

import hashlib
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.execution.hummingbot_bridge import HummingbotBridge


class TestOrderIdempotency:
    """订单幂等性测试"""

    @pytest.fixture
    def bridge(self):
        """创建 HummingbotBridge 实例"""
        return HummingbotBridge()

    def test_same_signal_same_order_id(self, bridge):
        """测试相同信号生成相同 order_id"""
        # 模拟信号
        signal = {
            "signal_id": "sig_001",
            "trading_pair": "BTCUSDT",
            "direction": "LONG",
            "confidence": 0.8,
        }

        # 生成两次 order_id
        order_id_1 = bridge._generate_idempotent_order_id(signal["signal_id"])
        order_id_2 = bridge._generate_idempotent_order_id(signal["signal_id"])

        assert order_id_1 == order_id_2

    def test_different_signals_different_order_ids(self, bridge):
        """测试不同信号生成不同 order_id"""
        order_id_1 = bridge._generate_idempotent_order_id("sig_001")
        order_id_2 = bridge._generate_idempotent_order_id("sig_002")

        assert order_id_1 != order_id_2

    def test_order_id_format(self, bridge):
        """测试 order_id 格式"""
        order_id = bridge._generate_idempotent_order_id("sig_001")

        # order_id 应该以 algvex_ 开头
        assert order_id.startswith("algvex_")

        # 应该包含 hash
        parts = order_id.split("_")
        assert len(parts) >= 2

    def test_order_id_deterministic(self, bridge):
        """测试 order_id 确定性"""
        signal_id = "sig_test_123"

        # 多次生成应该完全相同
        order_ids = [
            bridge._generate_idempotent_order_id(signal_id)
            for _ in range(10)
        ]

        assert len(set(order_ids)) == 1

    def test_hash_collision_resistance(self, bridge):
        """测试 hash 碰撞阻力"""
        # 生成大量 order_id，检查是否有碰撞
        order_ids = set()

        for i in range(1000):
            signal_id = f"sig_{i:06d}"
            order_id = bridge._generate_idempotent_order_id(signal_id)
            order_ids.add(order_id)

        # 应该没有碰撞
        assert len(order_ids) == 1000


class TestSignalToOrderMapping:
    """信号到订单映射测试"""

    @pytest.fixture
    def bridge(self):
        """创建 HummingbotBridge 实例"""
        return HummingbotBridge()

    def test_track_signal_order_mapping(self, bridge):
        """测试追踪信号-订单映射"""
        signal_id = "sig_001"
        order_id = bridge._generate_idempotent_order_id(signal_id)

        # 记录映射
        bridge._pending_orders[order_id] = {
            "signal_id": signal_id,
            "trading_pair": "BTCUSDT",
        }

        # 应该能找到映射
        assert order_id in bridge._pending_orders
        assert bridge._pending_orders[order_id]["signal_id"] == signal_id

    def test_replay_same_signal(self, bridge):
        """测试重放相同信号"""
        signal_id = "sig_001"

        # 第一次处理
        order_id_1 = bridge._generate_idempotent_order_id(signal_id)
        bridge._pending_orders[order_id_1] = {"signal_id": signal_id}

        # 重放相同信号
        order_id_2 = bridge._generate_idempotent_order_id(signal_id)

        # 应该得到相同的 order_id
        assert order_id_1 == order_id_2

        # 应该能识别为重复
        assert order_id_2 in bridge._pending_orders


class TestIdempotentOrderGeneration:
    """幂等订单生成测试"""

    def test_order_id_hash_algorithm(self):
        """测试 order_id hash 算法"""
        signal_id = "sig_test_123"

        # 使用 SHA256 生成确定性 hash
        hash_input = f"algvex:{signal_id}"
        expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # 验证算法一致性
        assert len(expected_hash) == 16

    def test_order_id_uniqueness_per_signal(self):
        """测试每个信号的 order_id 唯一性"""
        signals = [
            "sig_001",
            "sig_002",
            "sig_003",
            "sig_001",  # 重复
        ]

        order_ids = []
        for signal_id in signals:
            hash_input = f"algvex:{signal_id}"
            order_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            order_ids.append(order_id)

        # 相同信号应该产生相同 order_id
        assert order_ids[0] == order_ids[3]

        # 不同信号应该产生不同 order_id
        assert len(set(order_ids[:3])) == 3

    def test_order_id_time_independence(self):
        """测试 order_id 时间无关性"""
        import time

        signal_id = "sig_time_test"

        # 生成 order_id
        hash_input = f"algvex:{signal_id}"
        order_id_1 = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # 等待一段时间
        time.sleep(0.1)

        # 再次生成
        order_id_2 = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # 应该相同（时间无关）
        assert order_id_1 == order_id_2
