"""
对齐检查任务
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="algvex.api.tasks.alignment_tasks.run_daily_alignment_task")
def run_daily_alignment_task(date: Optional[str] = None) -> Dict[str, Any]:
    """
    执行每日对齐检查

    Args:
        date: 日期字符串 YYYY-MM-DD (默认为昨天)

    Returns:
        对齐检查结果
    """
    if date is None:
        date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"开始对齐检查: date={date}")

    try:
        from ...core.alignment_checker import AlignmentChecker

        checker = AlignmentChecker()
        report = checker.check_daily_alignment(date)

        # 如果检查失败，发送告警
        if not report.passed:
            checker.send_alert(report)
            logger.warning(f"对齐检查失败: date={date}, reasons={report.failure_reasons}")
        else:
            logger.info(f"对齐检查通过: date={date}")

        return {
            "status": "completed",
            "date": date,
            "passed": report.passed,
            "summary": {
                "total_live": report.total_live,
                "total_replay": report.total_replay,
                "matched": report.matched,
                "missing_in_replay": len(report.missing_in_replay),
                "missing_in_live": len(report.missing_in_live),
                "mismatched": len(report.mismatched),
            },
            "failure_reasons": report.failure_reasons,
        }

    except Exception as e:
        logger.error(f"对齐检查失败: date={date}, error={e}")
        return {
            "status": "failed",
            "date": date,
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.alignment_tasks.run_replay_task")
def run_replay_task(
    date: str,
    snapshot_id: str,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行重放任务

    Args:
        date: 重放日期
        snapshot_id: 快照ID
        config_hash: 配置哈希 (可选)

    Returns:
        重放结果
    """
    logger.info(f"开始重放: date={date}, snapshot_id={snapshot_id}")

    try:
        from ...core.replay.replay_runner import ReplayRunner

        runner = ReplayRunner()
        result = runner.run_replay(
            date=date,
            snapshot_id=snapshot_id,
            config_hash=config_hash,
        )

        logger.info(f"重放完成: date={date}")

        return {
            "status": "completed",
            "date": date,
            "snapshot_id": snapshot_id,
            "signals_generated": result.get("signals_count", 0),
            "output_path": result.get("output_path"),
        }

    except Exception as e:
        logger.error(f"重放失败: date={date}, error={e}")
        return {
            "status": "failed",
            "date": date,
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.alignment_tasks.cleanup_old_signals_task")
def cleanup_old_signals_task(days: int = 90) -> Dict[str, Any]:
    """
    清理旧信号

    Args:
        days: 保留天数

    Returns:
        清理结果
    """
    logger.info(f"开始清理旧信号: 保留 {days} 天")

    try:
        from ..database import SessionLocal
        from ..services.signal_service import SignalService

        db = SessionLocal()
        try:
            service = SignalService(db)
            deleted_count = service.delete_old_signals(days)

            logger.info(f"清理完成: 删除了 {deleted_count} 条信号")

            return {
                "status": "completed",
                "deleted_count": deleted_count,
            }
        finally:
            db.close()

    except Exception as e:
        logger.error(f"清理失败: error={e}")
        return {
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.alignment_tasks.batch_alignment_check_task")
def batch_alignment_check_task(dates: list) -> Dict[str, Any]:
    """
    批量对齐检查

    Args:
        dates: 日期列表

    Returns:
        批量检查结果
    """
    logger.info(f"开始批量对齐检查: {len(dates)} 天")

    results = []
    for date in dates:
        result = run_daily_alignment_task.apply(args=(date,)).get()
        results.append(result)

    passed_count = sum(1 for r in results if r.get("passed", False))

    return {
        "status": "completed",
        "total": len(dates),
        "passed": passed_count,
        "failed": len(dates) - passed_count,
        "results": results,
    }
