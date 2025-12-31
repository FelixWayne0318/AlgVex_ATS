"""
P0 验收测试: 快照存储/恢复

验收标准:
- 快照可以正确创建和保存
- 快照可以正确加载和恢复
- 快照内容完整性校验通过
- 快照元数据正确记录
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.core.data.snapshot_manager import SnapshotManager


class TestSnapshotCreation:
    """测试快照创建"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def snapshot_manager(self, temp_dir):
        """创建快照管理器"""
        return SnapshotManager(base_dir=temp_dir)

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        return pd.DataFrame({
            "open": np.random.randn(100) * 100 + 50000,
            "high": np.random.randn(100) * 100 + 50100,
            "low": np.random.randn(100) * 100 + 49900,
            "close": np.random.randn(100) * 100 + 50000,
            "volume": np.random.randint(100, 10000, 100),
        }, index=dates)

    def test_create_snapshot(self, snapshot_manager, sample_data, temp_dir):
        """测试创建快照"""
        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        snapshot_id = snapshot_manager.create_snapshot(
            data={"BTCUSDT": sample_data},
            cutoff_time=cutoff_time,
            metadata={"symbols": ["BTCUSDT"]},
        )

        assert snapshot_id is not None
        assert snapshot_id.startswith("snap_")

    def test_snapshot_metadata(self, snapshot_manager, sample_data, temp_dir):
        """测试快照元数据"""
        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        snapshot_id = snapshot_manager.create_snapshot(
            data={"BTCUSDT": sample_data},
            cutoff_time=cutoff_time,
            metadata={
                "symbols": ["BTCUSDT"],
                "config_hash": "sha256:test_config",
            },
        )

        # 获取快照信息
        info = snapshot_manager.get_snapshot_info(snapshot_id)

        assert info is not None
        assert "cutoff_time" in info
        assert "symbols" in info.get("metadata", {})

    def test_snapshot_list(self, snapshot_manager, sample_data, temp_dir):
        """测试列出快照"""
        # 创建多个快照
        for i in range(3):
            cutoff_time = datetime(2024, 1, 15 + i, 12, 0, 0)
            snapshot_manager.create_snapshot(
                data={"BTCUSDT": sample_data},
                cutoff_time=cutoff_time,
                metadata={"index": i},
            )

        snapshots = snapshot_manager.list_snapshots()
        assert len(snapshots) >= 3


class TestSnapshotLoading:
    """测试快照加载"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def snapshot_manager(self, temp_dir):
        """创建快照管理器"""
        return SnapshotManager(base_dir=temp_dir)

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        return pd.DataFrame({
            "open": np.random.randn(100) * 100 + 50000,
            "high": np.random.randn(100) * 100 + 50100,
            "low": np.random.randn(100) * 100 + 49900,
            "close": np.random.randn(100) * 100 + 50000,
            "volume": np.random.randint(100, 10000, 100),
        }, index=dates)

    def test_load_snapshot(self, snapshot_manager, sample_data, temp_dir):
        """测试加载快照"""
        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        # 创建快照
        snapshot_id = snapshot_manager.create_snapshot(
            data={"BTCUSDT": sample_data},
            cutoff_time=cutoff_time,
            metadata={"symbols": ["BTCUSDT"]},
        )

        # 加载快照
        loaded_data = snapshot_manager.load_snapshot(snapshot_id)

        assert loaded_data is not None
        assert "BTCUSDT" in loaded_data
        assert len(loaded_data["BTCUSDT"]) == len(sample_data)

    def test_load_nonexistent_snapshot(self, snapshot_manager):
        """测试加载不存在的快照"""
        with pytest.raises(FileNotFoundError):
            snapshot_manager.load_snapshot("nonexistent_snapshot")

    def test_data_integrity(self, snapshot_manager, sample_data, temp_dir):
        """测试数据完整性"""
        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        # 创建快照
        snapshot_id = snapshot_manager.create_snapshot(
            data={"BTCUSDT": sample_data},
            cutoff_time=cutoff_time,
            metadata={},
        )

        # 加载快照
        loaded_data = snapshot_manager.load_snapshot(snapshot_id)

        # 验证数据完整性
        original = sample_data
        loaded = loaded_data["BTCUSDT"]

        # 检查列
        assert set(original.columns) == set(loaded.columns)

        # 检查数值 (允许浮点误差)
        for col in original.columns:
            np.testing.assert_array_almost_equal(
                original[col].values,
                loaded[col].values,
                decimal=5,
            )


class TestSnapshotIntegrity:
    """测试快照完整性"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def snapshot_manager(self, temp_dir):
        """创建快照管理器"""
        return SnapshotManager(base_dir=temp_dir)

    def test_multiple_symbols(self, snapshot_manager, temp_dir):
        """测试多标的快照"""
        dates = pd.date_range("2024-01-01", periods=50, freq="5min")

        data = {}
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            data[symbol] = pd.DataFrame({
                "close": np.random.randn(50) * 100 + 1000,
                "volume": np.random.randint(100, 1000, 50),
            }, index=dates)

        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)
        snapshot_id = snapshot_manager.create_snapshot(
            data=data,
            cutoff_time=cutoff_time,
            metadata={"symbols": list(data.keys())},
        )

        loaded_data = snapshot_manager.load_snapshot(snapshot_id)

        assert len(loaded_data) == 3
        assert all(s in loaded_data for s in ["BTCUSDT", "ETHUSDT", "BNBUSDT"])

    def test_empty_data(self, snapshot_manager, temp_dir):
        """测试空数据快照"""
        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        snapshot_id = snapshot_manager.create_snapshot(
            data={},
            cutoff_time=cutoff_time,
            metadata={"empty": True},
        )

        loaded_data = snapshot_manager.load_snapshot(snapshot_id)
        assert len(loaded_data) == 0

    def test_snapshot_hash_consistency(self, snapshot_manager, temp_dir):
        """测试快照哈希一致性"""
        dates = pd.date_range("2024-01-01", periods=50, freq="5min")
        data = {
            "BTCUSDT": pd.DataFrame({
                "close": [50000.0] * 50,
                "volume": [1000] * 50,
            }, index=dates)
        }

        cutoff_time = datetime(2024, 1, 15, 12, 0, 0)

        # 创建两次相同数据的快照
        snapshot_id_1 = snapshot_manager.create_snapshot(
            data=data,
            cutoff_time=cutoff_time,
            metadata={},
        )

        snapshot_id_2 = snapshot_manager.create_snapshot(
            data=data,
            cutoff_time=cutoff_time,
            metadata={},
        )

        # 两个快照的内容应该相同 (但 ID 可能不同)
        loaded_1 = snapshot_manager.load_snapshot(snapshot_id_1)
        loaded_2 = snapshot_manager.load_snapshot(snapshot_id_2)

        np.testing.assert_array_equal(
            loaded_1["BTCUSDT"]["close"].values,
            loaded_2["BTCUSDT"]["close"].values,
        )
