"""
Celery 应用配置
"""

import os
from celery import Celery


# 从环境变量获取 Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 创建 Celery 应用
celery_app = Celery(
    "algvex",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "algvex.api.tasks.backtest_tasks",
        "algvex.api.tasks.alignment_tasks",
        "algvex.api.tasks.snapshot_tasks",
    ],
)

# Celery 配置
celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # 时区
    timezone="UTC",
    enable_utc=True,

    # 任务结果
    result_expires=3600 * 24,  # 结果保留 24 小时

    # 任务执行
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # 并发
    worker_prefetch_multiplier=1,
    worker_concurrency=4,

    # 任务路由
    task_routes={
        "algvex.api.tasks.backtest_tasks.*": {"queue": "backtest"},
        "algvex.api.tasks.alignment_tasks.*": {"queue": "alignment"},
        "algvex.api.tasks.snapshot_tasks.*": {"queue": "snapshot"},
    },

    # 定时任务
    beat_schedule={
        "daily-alignment-check": {
            "task": "algvex.api.tasks.alignment_tasks.run_daily_alignment_task",
            "schedule": 3600 * 6,  # 每 6 小时
            "args": (),
        },
        "cleanup-old-signals": {
            "task": "algvex.api.tasks.alignment_tasks.cleanup_old_signals_task",
            "schedule": 3600 * 24,  # 每天
            "args": (90,),  # 保留 90 天
        },
    },
)


# 配置别名
app = celery_app
