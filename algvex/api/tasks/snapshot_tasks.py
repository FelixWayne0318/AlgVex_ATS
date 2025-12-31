"""
快照任务
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="algvex.api.tasks.snapshot_tasks.create_snapshot_task")
def create_snapshot_task(
    symbols: List[str],
    cutoff_time: Optional[str] = None,
    data_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    创建数据快照

    Args:
        symbols: 标的列表
        cutoff_time: 截止时间 (ISO格式)
        data_sources: 数据源列表

    Returns:
        快照创建结果
    """
    if cutoff_time:
        cutoff = datetime.fromisoformat(cutoff_time.replace("Z", "+00:00"))
    else:
        cutoff = datetime.utcnow()

    logger.info(f"开始创建快照: symbols={symbols}, cutoff={cutoff}")

    try:
        from ...core.data.snapshot_manager import SnapshotManager
        from ...shared.data_service import DataService

        # 获取数据
        data_service = DataService()
        data = {}

        for symbol in symbols:
            try:
                # 获取 K 线数据
                klines = data_service.get_bars(
                    symbol=symbol,
                    start=None,  # 获取所有可用数据
                    end=cutoff,
                    freq="5m",
                )
                if klines is not None and len(klines) > 0:
                    data[symbol] = klines
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")

        if not data:
            return {
                "status": "failed",
                "error": "没有可用数据",
            }

        # 创建快照
        snapshot_manager = SnapshotManager()
        snapshot_id = snapshot_manager.create_snapshot(
            data=data,
            cutoff_time=cutoff,
            metadata={
                "symbols": symbols,
                "data_sources": data_sources or ["klines_5m"],
                "created_by": "task",
            },
        )

        logger.info(f"快照创建完成: snapshot_id={snapshot_id}")

        return {
            "status": "completed",
            "snapshot_id": snapshot_id,
            "symbols": symbols,
            "cutoff_time": cutoff.isoformat() + "Z",
        }

    except Exception as e:
        logger.error(f"快照创建失败: error={e}")
        return {
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.snapshot_tasks.validate_snapshot_task")
def validate_snapshot_task(snapshot_id: str) -> Dict[str, Any]:
    """
    验证快照完整性

    Args:
        snapshot_id: 快照ID

    Returns:
        验证结果
    """
    logger.info(f"开始验证快照: snapshot_id={snapshot_id}")

    try:
        from ...core.data.snapshot_manager import SnapshotManager

        snapshot_manager = SnapshotManager()

        # 获取快照信息
        info = snapshot_manager.get_snapshot_info(snapshot_id)
        if not info:
            return {
                "status": "failed",
                "error": f"快照不存在: {snapshot_id}",
            }

        # 加载并验证数据
        data = snapshot_manager.load_snapshot(snapshot_id)

        # 检查数据完整性
        symbols = list(data.keys())
        total_rows = sum(len(df) for df in data.values())

        return {
            "status": "completed",
            "snapshot_id": snapshot_id,
            "valid": True,
            "symbols": symbols,
            "total_rows": total_rows,
        }

    except Exception as e:
        logger.error(f"快照验证失败: snapshot_id={snapshot_id}, error={e}")
        return {
            "status": "failed",
            "snapshot_id": snapshot_id,
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.snapshot_tasks.cleanup_old_snapshots_task")
def cleanup_old_snapshots_task(days: int = 30) -> Dict[str, Any]:
    """
    清理旧快照

    Args:
        days: 保留天数

    Returns:
        清理结果
    """
    logger.info(f"开始清理旧快照: 保留 {days} 天")

    try:
        from datetime import timedelta
        from ...core.data.snapshot_manager import SnapshotManager

        snapshot_manager = SnapshotManager()
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # 获取所有快照
        snapshots = snapshot_manager.list_snapshots()

        deleted_count = 0
        for snapshot in snapshots:
            try:
                info = snapshot_manager.get_snapshot_info(snapshot)
                if info and info.get("cutoff_time"):
                    snapshot_time = datetime.fromisoformat(info["cutoff_time"].replace("Z", "+00:00"))
                    if snapshot_time < cutoff_time:
                        snapshot_manager.delete_snapshot(snapshot)
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"删除快照失败 {snapshot}: {e}")

        logger.info(f"快照清理完成: 删除了 {deleted_count} 个快照")

        return {
            "status": "completed",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"快照清理失败: error={e}")
        return {
            "status": "failed",
            "error": str(e),
        }
