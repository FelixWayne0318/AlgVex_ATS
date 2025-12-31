"""
对齐检查服务
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ...core.alignment_checker import AlignmentChecker, AlignmentReport


logger = logging.getLogger(__name__)


class AlignmentService:
    """对齐检查服务"""

    def __init__(self, db: Optional[Session] = None, config_path: str = "config/alignment.yaml"):
        self.db = db
        self.checker = AlignmentChecker(config_path)

    def check_daily_alignment(self, date: str) -> Dict[str, Any]:
        """
        执行每日对齐检查

        Args:
            date: 日期字符串 YYYY-MM-DD

        Returns:
            对齐检查结果
        """
        try:
            report = self.checker.check_daily_alignment(date)
            return self._report_to_dict(report)
        except Exception as e:
            logger.error(f"对齐检查失败: {e}")
            return {
                "date": date,
                "passed": False,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }

    def get_alignment_report(self, date: str) -> Optional[Dict[str, Any]]:
        """
        获取已保存的对齐报告

        Args:
            date: 日期字符串

        Returns:
            对齐报告字典
        """
        report_path = Path(self.checker.report_pattern.format(date=date))

        if not report_path.exists():
            return None

        try:
            with open(report_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取对齐报告失败: {e}")
            return None

    def get_recent_reports(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取最近几天的对齐报告

        Args:
            days: 天数

        Returns:
            报告列表
        """
        reports = []
        from datetime import timedelta

        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            report = self.get_alignment_report(date)
            if report:
                reports.append(report)

        return reports

    def get_alignment_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        获取对齐检查摘要

        Args:
            days: 统计天数

        Returns:
            摘要信息
        """
        reports = self.get_recent_reports(days)

        if not reports:
            return {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_match_rate": 0.0,
                "max_signal_diff": 0.0,
            }

        passed_count = sum(1 for r in reports if r.get("passed", False))
        total_count = len(reports)

        # 计算平均匹配率
        match_rates = []
        max_diffs = []
        for r in reports:
            summary = r.get("summary", {})
            total = summary.get("total_live", 0) + summary.get("total_replay", 0)
            matched = summary.get("matched", 0)
            if total > 0:
                match_rates.append(matched / total * 2)  # *2 因为 total 是两边之和
            max_diffs.append(summary.get("max_signal_diff", 0))

        avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 0
        max_signal_diff = max(max_diffs) if max_diffs else 0

        return {
            "total_checks": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "avg_match_rate": avg_match_rate,
            "max_signal_diff": max_signal_diff,
        }

    def send_alert_if_failed(self, report: Dict[str, Any]) -> bool:
        """
        如果检查失败则发送告警

        Args:
            report: 检查报告

        Returns:
            是否发送了告警
        """
        if report.get("passed", True):
            return False

        # 构建告警报告对象
        from ...core.alignment_checker import AlignmentReport

        alert_report = AlignmentReport(
            date=report.get("date", ""),
            total_live=report.get("summary", {}).get("total_live", 0),
            total_replay=report.get("summary", {}).get("total_replay", 0),
            matched=report.get("summary", {}).get("matched", 0),
            missing_in_replay=[],
            missing_in_live=[],
            mismatched=[],
            max_signal_diff=report.get("summary", {}).get("max_signal_diff", 0),
            config_hash=report.get("config_hash", ""),
            snapshot_id=report.get("snapshot_id", ""),
            passed=False,
            failure_reasons=report.get("failure_reasons", []),
        )

        self.checker.send_alert(alert_report)
        return True

    def _report_to_dict(self, report: AlignmentReport) -> Dict[str, Any]:
        """将 AlignmentReport 转换为字典"""
        return {
            "date": report.date,
            "passed": report.passed,
            "generated_at": report.generated_at,
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
                "missing_in_replay": report.missing_in_replay[:20],
                "missing_in_live": report.missing_in_live[:20],
                "mismatched": report.mismatched[:20],
            },
        }
