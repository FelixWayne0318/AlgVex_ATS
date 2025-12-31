"""
回测任务
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from .celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="algvex.api.tasks.backtest_tasks.run_backtest_task")
def run_backtest_task(
    self,
    backtest_id: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    执行回测任务

    Args:
        backtest_id: 回测ID
        config: 回测配置

    Returns:
        回测结果
    """
    logger.info(f"开始回测任务: backtest_id={backtest_id}")

    try:
        # 更新状态为运行中
        self.update_state(state="RUNNING", meta={"progress": 0})

        # 导入回测引擎
        from ...scripts.run_backtest import BacktestRunner

        # 提取配置
        symbols = config.get("symbols", ["BTCUSDT"])
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        initial_capital = config.get("initial_capital", 100000)
        snapshot_id = config.get("snapshot_id")

        # 创建回测运行器
        runner = BacktestRunner(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            snapshot_id=snapshot_id,
        )

        # 运行回测 (带进度回调)
        def progress_callback(progress: float):
            self.update_state(state="RUNNING", meta={"progress": progress})

        result = runner.run(progress_callback=progress_callback)

        logger.info(f"回测完成: backtest_id={backtest_id}")

        return {
            "status": "completed",
            "backtest_id": backtest_id,
            "result": result,
        }

    except Exception as e:
        logger.error(f"回测失败: backtest_id={backtest_id}, error={e}")
        logger.error(traceback.format_exc())

        return {
            "status": "failed",
            "backtest_id": backtest_id,
            "error": str(e),
        }


@celery_app.task(name="algvex.api.tasks.backtest_tasks.cancel_backtest_task")
def cancel_backtest_task(backtest_id: int) -> Dict[str, Any]:
    """
    取消回测任务

    Args:
        backtest_id: 回测ID

    Returns:
        取消结果
    """
    logger.info(f"取消回测任务: backtest_id={backtest_id}")

    # 在实际实现中，需要找到正在运行的任务并终止它
    # 这里只是返回取消状态

    return {
        "status": "cancelled",
        "backtest_id": backtest_id,
    }


@celery_app.task(name="algvex.api.tasks.backtest_tasks.batch_backtest_task")
def batch_backtest_task(
    backtest_configs: list,
) -> Dict[str, Any]:
    """
    批量回测任务

    Args:
        backtest_configs: 回测配置列表

    Returns:
        批量回测结果
    """
    logger.info(f"开始批量回测: {len(backtest_configs)} 个任务")

    results = []
    for config in backtest_configs:
        try:
            result = run_backtest_task.apply(
                args=(config.get("backtest_id"), config)
            ).get()
            results.append(result)
        except Exception as e:
            results.append({
                "status": "failed",
                "error": str(e),
            })

    return {
        "status": "completed",
        "total": len(backtest_configs),
        "results": results,
    }
