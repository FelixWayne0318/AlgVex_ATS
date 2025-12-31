"""
每日任务集成测试

测试每日对齐检查和重放任务
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDailyAlignmentJob:
    """每日对齐任务测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def create_test_traces(self, filepath: str, traces: list):
        """创建测试 trace 文件"""
        with open(filepath, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

    def test_daily_alignment_success(self, temp_dir):
        """测试每日对齐成功场景"""
        from algvex.core.alignment_checker import AlignmentChecker

        date = "2024-01-15"

        # 创建相同的 live 和 replay traces
        traces = [
            {
                "signal_id": "sig_001",
                "data_hash": "sha256:abc123",
                "features_hash": "sha256:def456",
                "raw_prediction": 0.5,
                "final_signal": 1,
                "config_hash": "sha256:config",
                "snapshot_id": "snap_20240115",
            },
            {
                "signal_id": "sig_002",
                "data_hash": "sha256:abc123",
                "features_hash": "sha256:def456",
                "raw_prediction": -0.3,
                "final_signal": -1,
                "config_hash": "sha256:config",
                "snapshot_id": "snap_20240115",
            },
        ]

        live_path = f"{temp_dir}/live_output_{date}.jsonl"
        replay_path = f"{temp_dir}/replay_output_{date}.jsonl"

        self.create_test_traces(live_path, traces)
        self.create_test_traces(replay_path, traces)

        # 创建检查器
        checker = AlignmentChecker()
        checker.live_output_pattern = f"{temp_dir}/live_output_{{date}}.jsonl"
        checker.replay_output_pattern = f"{temp_dir}/replay_output_{{date}}.jsonl"
        checker.report_pattern = f"{temp_dir}/alignment_report_{{date}}.json"

        # 执行检查
        report = checker.check_daily_alignment(date)

        assert report.passed is True
        assert report.matched == 2
        assert len(report.missing_in_replay) == 0
        assert len(report.mismatched) == 0

    def test_daily_alignment_mismatch(self, temp_dir):
        """测试每日对齐不匹配场景"""
        from algvex.core.alignment_checker import AlignmentChecker

        date = "2024-01-16"

        live_traces = [
            {
                "signal_id": "sig_001",
                "data_hash": "sha256:live_hash",
                "final_signal": 1,
            },
        ]

        replay_traces = [
            {
                "signal_id": "sig_001",
                "data_hash": "sha256:replay_hash",  # 不同的哈希
                "final_signal": 1,
            },
        ]

        live_path = f"{temp_dir}/live_output_{date}.jsonl"
        replay_path = f"{temp_dir}/replay_output_{date}.jsonl"

        self.create_test_traces(live_path, live_traces)
        self.create_test_traces(replay_path, replay_traces)

        checker = AlignmentChecker()
        checker.live_output_pattern = f"{temp_dir}/live_output_{{date}}.jsonl"
        checker.replay_output_pattern = f"{temp_dir}/replay_output_{{date}}.jsonl"
        checker.report_pattern = f"{temp_dir}/alignment_report_{{date}}.json"

        report = checker.check_daily_alignment(date)

        assert report.passed is False
        assert len(report.mismatched) == 1


class TestDailyReplayJob:
    """每日重放任务测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_replay_uses_fixed_time(self, temp_dir):
        """测试重放使用固定时间"""
        from algvex.shared.time_provider import TimeProvider, reset_time_provider

        reset_time_provider()

        # 模拟重放模式
        fixed_time = datetime(2024, 1, 15, 12, 0, 0)
        provider = TimeProvider(mode="replay", fixed_time=fixed_time)

        # 验证时间不变
        assert provider.now() == fixed_time
        assert provider.now() == fixed_time

        # 推进时间
        from datetime import timedelta
        provider.advance_time(timedelta(minutes=5))
        assert provider.now() == fixed_time + timedelta(minutes=5)

    def test_replay_uses_seeded_random(self):
        """测试重放使用固定随机种子"""
        from algvex.shared.seeded_random import SeededRandom, reset_seeded_random

        reset_seeded_random()

        # 两次使用相同种子
        rng1 = SeededRandom(seed=12345)
        values1 = [rng1.random() for _ in range(10)]

        rng2 = SeededRandom(seed=12345)
        values2 = [rng2.random() for _ in range(10)]

        assert values1 == values2


class TestReportGeneration:
    """报告生成测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_alignment_report_saved(self, temp_dir):
        """测试对齐报告保存"""
        from algvex.core.alignment_checker import AlignmentChecker

        date = "2024-01-15"

        # 创建空的 trace 文件
        live_path = Path(temp_dir) / f"live_output_{date}.jsonl"
        replay_path = Path(temp_dir) / f"replay_output_{date}.jsonl"
        live_path.touch()
        replay_path.touch()

        checker = AlignmentChecker()
        checker.live_output_pattern = f"{temp_dir}/live_output_{{date}}.jsonl"
        checker.replay_output_pattern = f"{temp_dir}/replay_output_{{date}}.jsonl"
        checker.report_pattern = f"{temp_dir}/alignment_report_{{date}}.json"

        # 执行检查
        report = checker.check_daily_alignment(date)

        # 验证报告文件已创建
        report_path = Path(temp_dir) / f"alignment_report_{date}.json"
        assert report_path.exists()

        # 验证报告内容
        with open(report_path) as f:
            saved_report = json.load(f)

        assert saved_report["date"] == date
        assert "passed" in saved_report
        assert "summary" in saved_report
