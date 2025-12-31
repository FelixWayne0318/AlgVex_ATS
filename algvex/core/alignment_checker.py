"""
对齐检查器 (Alignment Checker)

功能:
- 比对 Live 和 Replay 的 trace 输出
- 验证信号一致性
- 生成对齐报告
- 触发告警

依赖配置: config/alignment.yaml
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import yaml


logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """对齐结果"""
    passed: bool
    level: str  # L1/L2/L3/L4/L5 or "all"
    reason: str
    diff_value: Optional[float] = None


@dataclass
class TraceComparison:
    """单条 trace 的比对结果"""
    signal_id: str
    live_trace: Optional[Dict[str, Any]]
    replay_trace: Optional[Dict[str, Any]]
    alignment_result: Optional[AlignmentResult] = None
    diffs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentReport:
    """对齐报告"""
    date: str
    total_live: int
    total_replay: int
    matched: int
    missing_in_replay: List[str]
    missing_in_live: List[str]
    mismatched: List[Dict[str, Any]]
    max_signal_diff: float
    config_hash: str
    snapshot_id: str
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class AlignmentChecker:
    """
    对齐检查器

    使用方法:
        checker = AlignmentChecker()
        report = checker.check_daily_alignment("2024-01-15")
        if not report.passed:
            checker.send_alert(report)
    """

    # 默认阈值
    DEFAULT_SIGNAL_DIFF_THRESHOLD = 0.001  # 0.1%
    DEFAULT_FLOAT_ATOL = 1e-8
    DEFAULT_FLOAT_RTOL = 1e-6

    def __init__(self, config_path: str = "config/alignment.yaml"):
        """
        初始化对齐检查器

        Args:
            config_path: 对齐配置文件路径
        """
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """加载配置"""
        config_file = Path(self.config_path)

        if config_file.exists():
            with open(config_file, encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"对齐配置文件不存在: {self.config_path}, 使用默认配置")
            config = {}

        # 路径配置
        paths = config.get("daily_alignment", {}).get("paths", {})
        self.live_output_pattern = paths.get("live_output_pattern", "logs/live_output_{date}.jsonl")
        self.replay_output_pattern = paths.get("replay_output_pattern", "logs/replay_output_{date}.jsonl")
        self.report_pattern = paths.get("alignment_report_pattern", "logs/alignment_report_{date}.json")

        # 验收标准
        criteria = config.get("daily_alignment", {}).get("acceptance_criteria", {})

        self.data_hash_tolerance = criteria.get("data_hash", {}).get("tolerance", 0)
        self.features_hash_tolerance = criteria.get("features_hash", {}).get("tolerance", 0)

        raw_pred = criteria.get("raw_prediction", {})
        self.prediction_atol = raw_pred.get("absolute_tolerance", self.DEFAULT_FLOAT_ATOL)
        self.prediction_rtol = raw_pred.get("relative_tolerance", self.DEFAULT_FLOAT_RTOL)
        self.prediction_max_diff = raw_pred.get("max_diff_threshold", 1e-4)

        self.final_signal_tolerance = criteria.get("final_signal", {}).get("tolerance", 0)

        # 告警阈值
        thresholds = config.get("daily_alignment", {}).get("alert_thresholds", {})
        self.max_missing_signals = thresholds.get("max_missing_signals", 0)
        self.max_signal_diff = thresholds.get("max_signal_diff", self.DEFAULT_SIGNAL_DIFF_THRESHOLD)
        self.max_mismatched = thresholds.get("max_mismatched_signals", 0)

    def load_traces(self, filepath: str) -> List[Dict[str, Any]]:
        """加载 JSONL 格式的 trace 文件"""
        traces = []
        path = Path(filepath)

        if not path.exists():
            logger.warning(f"Trace 文件不存在: {filepath}")
            return traces

        with open(path, encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                    traces.append(trace)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析错误 {filepath}:{line_no}: {e}")

        return traces

    def verify_alignment(
        self,
        live: Dict[str, Any],
        replay: Dict[str, Any]
    ) -> AlignmentResult:
        """
        分层验证 Live vs Replay 对齐

        层级:
        - L1: data_hash 必须 100% 一致
        - L2: features_hash 必须 100% 一致
        - L3: raw_prediction 允许微小浮点差异
        - L4: final_signal 必须 100% 一致
        """
        # L1: data_hash
        live_data_hash = live.get("data_hash", "")
        replay_data_hash = replay.get("data_hash", "")
        if live_data_hash != replay_data_hash:
            return AlignmentResult(
                passed=False,
                level="L1",
                reason=f"data_hash mismatch: {live_data_hash} vs {replay_data_hash}"
            )

        # L2: features_hash (如果有)
        live_features_hash = live.get("features_hash", "")
        replay_features_hash = replay.get("features_hash", "")
        if live_features_hash and replay_features_hash:
            if live_features_hash != replay_features_hash:
                return AlignmentResult(
                    passed=False,
                    level="L2",
                    reason=f"features_hash mismatch"
                )

        # L3: raw_prediction
        live_pred = live.get("raw_prediction")
        replay_pred = replay.get("raw_prediction")
        if live_pred is not None and replay_pred is not None:
            try:
                if not np.allclose(live_pred, replay_pred,
                                   atol=self.prediction_atol,
                                   rtol=self.prediction_rtol):
                    diff = abs(float(live_pred) - float(replay_pred))
                    if diff > self.prediction_max_diff:
                        return AlignmentResult(
                            passed=False,
                            level="L3",
                            reason=f"raw_prediction diff={diff}",
                            diff_value=diff
                        )
            except (TypeError, ValueError):
                pass  # 非数值类型，跳过

        # L4: final_signal
        live_signal = live.get("final_signal")
        replay_signal = replay.get("final_signal")
        if live_signal != replay_signal:
            diff = abs(float(live_signal or 0) - float(replay_signal or 0))
            return AlignmentResult(
                passed=False,
                level="L4",
                reason=f"final_signal mismatch: {live_signal} vs {replay_signal}",
                diff_value=diff
            )

        return AlignmentResult(passed=True, level="all", reason="OK")

    def check_daily_alignment(self, date: str) -> AlignmentReport:
        """
        执行每日对齐检查

        Args:
            date: 日期字符串 (YYYY-MM-DD)

        Returns:
            AlignmentReport 对齐报告
        """
        # 加载 trace 文件
        live_path = self.live_output_pattern.format(date=date)
        replay_path = self.replay_output_pattern.format(date=date)

        live_traces = self.load_traces(live_path)
        replay_traces = self.load_traces(replay_path)

        # 建立索引 (按 signal_id 去重，保留最后一条)
        live_by_id = {}
        for trace in live_traces:
            signal_id = trace.get("signal_id", "")
            if signal_id:
                live_by_id[signal_id] = trace

        replay_by_id = {}
        for trace in replay_traces:
            signal_id = trace.get("signal_id", "")
            if signal_id:
                replay_by_id[signal_id] = trace

        # 获取配置信息
        config_hash = ""
        snapshot_id = ""
        if live_traces:
            first_trace = live_traces[0]
            config_hash = first_trace.get("config_hash", "")
            snapshot_id = first_trace.get("snapshot_id", "")

        # 比对
        all_signal_ids = set(live_by_id.keys()) | set(replay_by_id.keys())

        missing_in_replay = []
        missing_in_live = []
        mismatched = []
        max_diff = 0.0

        for sid in sorted(all_signal_ids):
            live_t = live_by_id.get(sid)
            replay_t = replay_by_id.get(sid)

            if live_t and not replay_t:
                missing_in_replay.append(sid)
            elif replay_t and not live_t:
                missing_in_live.append(sid)
            else:
                result = self.verify_alignment(live_t, replay_t)
                if not result.passed:
                    mismatched.append({
                        "signal_id": sid,
                        "level": result.level,
                        "reason": result.reason,
                        "live_signal": live_t.get("final_signal"),
                        "replay_signal": replay_t.get("final_signal"),
                        "diff": result.diff_value,
                    })
                    if result.diff_value:
                        max_diff = max(max_diff, result.diff_value)

        # 判断是否通过
        failure_reasons = []

        if len(missing_in_replay) > self.max_missing_signals:
            failure_reasons.append(
                f"缺失信号过多: {len(missing_in_replay)} > {self.max_missing_signals}"
            )

        if len(missing_in_live) > self.max_missing_signals:
            failure_reasons.append(
                f"多余信号过多: {len(missing_in_live)} > {self.max_missing_signals}"
            )

        if len(mismatched) > self.max_mismatched:
            failure_reasons.append(
                f"不匹配信号过多: {len(mismatched)} > {self.max_mismatched}"
            )

        if max_diff > self.max_signal_diff:
            failure_reasons.append(
                f"信号差异过大: {max_diff} > {self.max_signal_diff}"
            )

        passed = len(failure_reasons) == 0

        # 生成报告
        report = AlignmentReport(
            date=date,
            total_live=len(live_by_id),
            total_replay=len(replay_by_id),
            matched=len(all_signal_ids) - len(missing_in_replay) - len(missing_in_live) - len(mismatched),
            missing_in_replay=missing_in_replay,
            missing_in_live=missing_in_live,
            mismatched=mismatched,
            max_signal_diff=max_diff,
            config_hash=config_hash,
            snapshot_id=snapshot_id,
            passed=passed,
            failure_reasons=failure_reasons,
        )

        # 保存报告
        self._save_report(report, date)

        return report

    def _save_report(self, report: AlignmentReport, date: str):
        """保存对齐报告"""
        report_path = Path(self.report_pattern.format(date=date))
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = {
            "date": report.date,
            "generated_at": report.generated_at,
            "passed": report.passed,
            "summary": {
                "total_live": report.total_live,
                "total_replay": report.total_replay,
                "matched": report.matched,
                "missing_in_replay": len(report.missing_in_replay),
                "missing_in_live": len(report.missing_in_live),
                "mismatched": len(report.mismatched),
                "max_signal_diff": report.max_signal_diff,
            },
            "config_hash": report.config_hash,
            "snapshot_id": report.snapshot_id,
            "failure_reasons": report.failure_reasons,
            "details": {
                "missing_in_replay": report.missing_in_replay[:100],  # 限制数量
                "missing_in_live": report.missing_in_live[:100],
                "mismatched": report.mismatched[:100],
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"对齐报告已保存: {report_path}")

    def send_alert(self, report: AlignmentReport):
        """发送告警"""
        if report.passed:
            logger.info(f"[{report.date}] 对齐检查通过")
            return

        # 构建告警消息
        message = f"""
[AlgVex 对齐告警] {report.date}

状态: ❌ 失败
原因:
{chr(10).join('  - ' + r for r in report.failure_reasons)}

摘要:
  - Live 信号数: {report.total_live}
  - Replay 信号数: {report.total_replay}
  - 匹配: {report.matched}
  - 缺失: {len(report.missing_in_replay)}
  - 多余: {len(report.missing_in_live)}
  - 不匹配: {len(report.mismatched)}
  - 最大差异: {report.max_signal_diff:.6f}

配置: {report.config_hash}
快照: {report.snapshot_id}
"""
        logger.error(message)

        # TODO: 发送到 Slack/邮件/短信
        # notify_oncall(message)

    def get_summary(self, report: AlignmentReport) -> str:
        """获取报告摘要"""
        status = "✅ 通过" if report.passed else "❌ 失败"
        return f"""
对齐检查报告 - {report.date}
状态: {status}
Live: {report.total_live} | Replay: {report.total_replay} | 匹配: {report.matched}
缺失: {len(report.missing_in_replay)} | 多余: {len(report.missing_in_live)} | 不匹配: {len(report.mismatched)}
最大差异: {report.max_signal_diff:.6f}
"""
