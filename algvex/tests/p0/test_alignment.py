"""
P0 验收测试: Live vs Replay 对齐

验收标准:
- Live 和 Replay 的 trace 输出必须一致
- 信号差异必须在允许阈值内
- 缺失信号数量必须为 0
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.core.alignment_checker import AlignmentChecker, AlignmentReport, AlignmentResult


class TestAlignmentChecker:
    """测试对齐检查器"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def checker(self, temp_dir):
        """创建对齐检查器"""
        # 创建测试配置
        config_path = Path(temp_dir) / "alignment.yaml"
        config_path.write_text("""
daily_alignment:
  paths:
    live_output_pattern: "{temp_dir}/live_output_{{date}}.jsonl"
    replay_output_pattern: "{temp_dir}/replay_output_{{date}}.jsonl"
    alignment_report_pattern: "{temp_dir}/alignment_report_{{date}}.json"
  acceptance_criteria:
    data_hash:
      tolerance: 0
    features_hash:
      tolerance: 0
    raw_prediction:
      absolute_tolerance: 1e-8
      relative_tolerance: 1e-6
      max_diff_threshold: 1e-4
    final_signal:
      tolerance: 0
  alert_thresholds:
    max_missing_signals: 0
    max_signal_diff: 0.001
    max_mismatched_signals: 0
""".format(temp_dir=temp_dir))

        checker = AlignmentChecker(str(config_path))
        checker.live_output_pattern = f"{temp_dir}/live_output_{{date}}.jsonl"
        checker.replay_output_pattern = f"{temp_dir}/replay_output_{{date}}.jsonl"
        checker.report_pattern = f"{temp_dir}/alignment_report_{{date}}.json"
        return checker

    def create_trace(
        self,
        signal_id: str,
        data_hash: str = "sha256:abc123",
        features_hash: str = "sha256:def456",
        raw_prediction: float = 0.5,
        final_signal: int = 1,
        config_hash: str = "sha256:config",
        snapshot_id: str = "snap_20240115_abc",
    ) -> Dict[str, Any]:
        """创建测试 trace"""
        return {
            "signal_id": signal_id,
            "data_hash": data_hash,
            "features_hash": features_hash,
            "raw_prediction": raw_prediction,
            "final_signal": final_signal,
            "config_hash": config_hash,
            "snapshot_id": snapshot_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def write_traces(self, filepath: str, traces: List[Dict[str, Any]]):
        """写入 traces 到文件"""
        with open(filepath, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")

    def test_perfect_alignment(self, checker, temp_dir):
        """测试完美对齐情况"""
        date = "2024-01-15"

        # 创建相同的 live 和 replay traces
        traces = [
            self.create_trace("sig_001", raw_prediction=0.5, final_signal=1),
            self.create_trace("sig_002", raw_prediction=-0.3, final_signal=-1),
            self.create_trace("sig_003", raw_prediction=0.0, final_signal=0),
        ]

        live_path = f"{temp_dir}/live_output_{date}.jsonl"
        replay_path = f"{temp_dir}/replay_output_{date}.jsonl"

        self.write_traces(live_path, traces)
        self.write_traces(replay_path, traces)

        # 执行对齐检查
        report = checker.check_daily_alignment(date)

        # 验证
        assert report.passed is True
        assert report.total_live == 3
        assert report.total_replay == 3
        assert report.matched == 3
        assert len(report.missing_in_replay) == 0
        assert len(report.missing_in_live) == 0
        assert len(report.mismatched) == 0

    def test_data_hash_mismatch(self, checker, temp_dir):
        """测试 L1 数据哈希不匹配"""
        date = "2024-01-16"

        live_traces = [
            self.create_trace("sig_001", data_hash="sha256:live_hash"),
        ]
        replay_traces = [
            self.create_trace("sig_001", data_hash="sha256:replay_hash"),
        ]

        self.write_traces(f"{temp_dir}/live_output_{date}.jsonl", live_traces)
        self.write_traces(f"{temp_dir}/replay_output_{date}.jsonl", replay_traces)

        report = checker.check_daily_alignment(date)

        assert report.passed is False
        assert len(report.mismatched) == 1
        assert report.mismatched[0]["level"] == "L1"

    def test_signal_mismatch(self, checker, temp_dir):
        """测试 L4 最终信号不匹配"""
        date = "2024-01-17"

        live_traces = [
            self.create_trace("sig_001", final_signal=1),
        ]
        replay_traces = [
            self.create_trace("sig_001", final_signal=-1),
        ]

        self.write_traces(f"{temp_dir}/live_output_{date}.jsonl", live_traces)
        self.write_traces(f"{temp_dir}/replay_output_{date}.jsonl", replay_traces)

        report = checker.check_daily_alignment(date)

        assert report.passed is False
        assert len(report.mismatched) == 1
        assert report.mismatched[0]["level"] == "L4"

    def test_missing_signals(self, checker, temp_dir):
        """测试缺失信号"""
        date = "2024-01-18"

        live_traces = [
            self.create_trace("sig_001"),
            self.create_trace("sig_002"),
            self.create_trace("sig_003"),
        ]
        replay_traces = [
            self.create_trace("sig_001"),
            # sig_002 缺失
            self.create_trace("sig_003"),
        ]

        self.write_traces(f"{temp_dir}/live_output_{date}.jsonl", live_traces)
        self.write_traces(f"{temp_dir}/replay_output_{date}.jsonl", replay_traces)

        report = checker.check_daily_alignment(date)

        assert report.passed is False
        assert len(report.missing_in_replay) == 1
        assert "sig_002" in report.missing_in_replay

    def test_extra_signals(self, checker, temp_dir):
        """测试多余信号"""
        date = "2024-01-19"

        live_traces = [
            self.create_trace("sig_001"),
        ]
        replay_traces = [
            self.create_trace("sig_001"),
            self.create_trace("sig_002"),  # 多余
        ]

        self.write_traces(f"{temp_dir}/live_output_{date}.jsonl", live_traces)
        self.write_traces(f"{temp_dir}/replay_output_{date}.jsonl", replay_traces)

        report = checker.check_daily_alignment(date)

        assert report.passed is False
        assert len(report.missing_in_live) == 1
        assert "sig_002" in report.missing_in_live


class TestAlignmentResult:
    """测试对齐结果"""

    def test_passed_result(self):
        """测试通过的结果"""
        result = AlignmentResult(passed=True, level="all", reason="OK")
        assert result.passed is True
        assert result.level == "all"

    def test_failed_result(self):
        """测试失败的结果"""
        result = AlignmentResult(
            passed=False,
            level="L1",
            reason="data_hash mismatch",
            diff_value=0.0,
        )
        assert result.passed is False
        assert result.level == "L1"


class TestVerifyAlignment:
    """测试 verify_alignment 方法的各层级"""

    @pytest.fixture
    def checker(self):
        """创建默认检查器"""
        return AlignmentChecker()

    def test_l1_data_hash(self, checker):
        """L1: data_hash 必须 100% 一致"""
        live = {"data_hash": "sha256:aaa", "final_signal": 1}
        replay = {"data_hash": "sha256:bbb", "final_signal": 1}

        result = checker.verify_alignment(live, replay)
        assert result.passed is False
        assert result.level == "L1"

    def test_l2_features_hash(self, checker):
        """L2: features_hash 必须 100% 一致"""
        live = {
            "data_hash": "sha256:same",
            "features_hash": "sha256:feat_a",
            "final_signal": 1,
        }
        replay = {
            "data_hash": "sha256:same",
            "features_hash": "sha256:feat_b",
            "final_signal": 1,
        }

        result = checker.verify_alignment(live, replay)
        assert result.passed is False
        assert result.level == "L2"

    def test_l3_raw_prediction(self, checker):
        """L3: raw_prediction 允许微小差异"""
        live = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.5,
            "final_signal": 1,
        }
        replay = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.5 + 1e-10,  # 微小差异
            "final_signal": 1,
        }

        result = checker.verify_alignment(live, replay)
        assert result.passed is True  # 微小差异应该通过

    def test_l3_raw_prediction_large_diff(self, checker):
        """L3: raw_prediction 大差异应该失败"""
        live = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.5,
            "final_signal": 1,
        }
        replay = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.6,  # 大差异
            "final_signal": 1,
        }

        result = checker.verify_alignment(live, replay)
        assert result.passed is False
        assert result.level == "L3"

    def test_l4_final_signal(self, checker):
        """L4: final_signal 必须 100% 一致"""
        live = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.5,
            "final_signal": 1,
        }
        replay = {
            "data_hash": "sha256:same",
            "raw_prediction": 0.5,
            "final_signal": -1,
        }

        result = checker.verify_alignment(live, replay)
        assert result.passed is False
        assert result.level == "L4"

    def test_all_pass(self, checker):
        """测试全部通过"""
        live = {
            "data_hash": "sha256:same",
            "features_hash": "sha256:feat",
            "raw_prediction": 0.5,
            "final_signal": 1,
        }
        replay = {
            "data_hash": "sha256:same",
            "features_hash": "sha256:feat",
            "raw_prediction": 0.5,
            "final_signal": 1,
        }

        result = checker.verify_alignment(live, replay)
        assert result.passed is True
        assert result.level == "all"
