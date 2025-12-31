"""
数据采集调度器

职责:
1. 定时调度数据采集任务
2. 增量更新数据
3. 数据质量检查
4. 失败重试
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import yaml
from loguru import logger

from .binance_collector import BinanceDataCollector
from .sentiment_collector import SentimentDataCollector


@dataclass
class TaskConfig:
    """任务配置"""
    name: str
    interval: int  # 秒
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    last_run: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    enabled: bool = True

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class DataScheduler:
    """
    数据采集调度器

    支持多任务并发调度
    """

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.tasks: Dict[str, TaskConfig] = {}
        self.running = False

        # 初始化采集器
        self.binance_collector = BinanceDataCollector()
        self.sentiment_collector = SentimentDataCollector()

        # 数据目录
        self.data_dir = Path(self.config.get("data", {}).get("data_dir", "~/.cryptoquant/data")).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 加载标的配置
        self.symbols = self._load_symbols()

        # 初始化任务
        self._init_tasks()

    def _load_config(self, config_path: str = None) -> dict:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _load_symbols(self) -> List[str]:
        """加载交易标的"""
        symbols_path = Path(__file__).parent.parent / "config" / "symbols.yaml"

        if symbols_path.exists():
            with open(symbols_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            symbols = []
            for item in config.get("primary", []):
                if item.get("enabled", True):
                    symbols.append(item["symbol"])
            return symbols

        # 默认标的
        return ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]

    def _init_tasks(self):
        """初始化采集任务"""
        collection_config = self.config.get("data", {}).get("collection", {})

        # K线数据采集 (每分钟)
        self.add_task(TaskConfig(
            name="kline_1m",
            interval=collection_config.get("kline_interval", 60),
            func=self._collect_klines,
            kwargs={"interval": "1m"},
        ))

        # 小时K线 (每小时)
        self.add_task(TaskConfig(
            name="kline_1h",
            interval=3600,
            func=self._collect_klines,
            kwargs={"interval": "1h"},
        ))

        # 资金费率 (每8小时，但每小时检查一次)
        self.add_task(TaskConfig(
            name="funding_rate",
            interval=collection_config.get("funding_interval", 3600),
            func=self._collect_funding_rates,
        ))

        # 持仓量 (每小时)
        self.add_task(TaskConfig(
            name="open_interest",
            interval=3600,
            func=self._collect_open_interest,
        ))

        # 多空比 (每小时)
        self.add_task(TaskConfig(
            name="long_short_ratio",
            interval=3600,
            func=self._collect_long_short_ratio,
        ))

        # 恐惧贪婪指数 (每天)
        self.add_task(TaskConfig(
            name="fear_greed",
            interval=collection_config.get("sentiment_interval", 86400),
            func=self._collect_fear_greed,
        ))

        # TVL (每天)
        self.add_task(TaskConfig(
            name="tvl",
            interval=86400,
            func=self._collect_tvl,
        ))

        logger.info(f"Initialized {len(self.tasks)} collection tasks")

    def add_task(self, task: TaskConfig):
        """添加任务"""
        self.tasks[task.name] = task
        logger.debug(f"Added task: {task.name}, interval={task.interval}s")

    def remove_task(self, name: str):
        """移除任务"""
        if name in self.tasks:
            del self.tasks[name]
            logger.debug(f"Removed task: {name}")

    async def _collect_klines(self, interval: str = "1h"):
        """采集K线数据"""
        for symbol in self.symbols:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)  # 增量24小时

                data = self.binance_collector.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000),
                )

                if not data.empty:
                    self._save_data(data, f"klines/{interval}", symbol)
                    logger.debug(f"Collected {len(data)} {interval} klines for {symbol}")

            except Exception as e:
                logger.error(f"Failed to collect klines for {symbol}: {e}")

    async def _collect_funding_rates(self):
        """采集资金费率"""
        for symbol in self.symbols:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)

                data = self.binance_collector.fetch_funding_rate(
                    symbol=symbol,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000),
                )

                if not data.empty:
                    self._save_data(data, "funding_rate", symbol)
                    logger.debug(f"Collected {len(data)} funding rates for {symbol}")

            except Exception as e:
                logger.error(f"Failed to collect funding rate for {symbol}: {e}")

    async def _collect_open_interest(self):
        """采集持仓量"""
        for symbol in self.symbols:
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)

                data = self.binance_collector.fetch_open_interest_history(
                    symbol=symbol,
                    period="1h",
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000),
                )

                if not data.empty:
                    self._save_data(data, "open_interest", symbol)
                    logger.debug(f"Collected {len(data)} OI records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to collect open interest for {symbol}: {e}")

    async def _collect_long_short_ratio(self):
        """采集多空比"""
        for symbol in self.symbols:
            try:
                data = self.binance_collector.fetch_long_short_ratio(
                    symbol=symbol,
                    period="1h",
                    limit=24,
                )

                if not data.empty:
                    self._save_data(data, "long_short_ratio", symbol)
                    logger.debug(f"Collected {len(data)} LS ratio records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to collect long/short ratio for {symbol}: {e}")

    async def _collect_fear_greed(self):
        """采集恐惧贪婪指数"""
        try:
            data = self.sentiment_collector.fetch_fear_greed_index(limit=30)

            if not data.empty:
                self._save_data(data, "sentiment", "fear_greed")
                logger.debug(f"Collected {len(data)} fear/greed records")

        except Exception as e:
            logger.error(f"Failed to collect fear/greed index: {e}")

    async def _collect_tvl(self):
        """采集TVL数据"""
        try:
            data = self.sentiment_collector.fetch_tvl_history()

            if not data.empty:
                self._save_data(data, "defi", "tvl")
                logger.debug(f"Collected {len(data)} TVL records")

        except Exception as e:
            logger.error(f"Failed to collect TVL: {e}")

    def _save_data(self, data, category: str, symbol: str):
        """保存数据"""
        save_dir = self.data_dir / category
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"{symbol}.parquet"

        if file_path.exists():
            # 增量更新
            existing = pd.read_parquet(file_path)
            combined = pd.concat([existing, data]).drop_duplicates()
            combined.to_parquet(file_path)
        else:
            data.to_parquet(file_path)

    async def _run_task(self, task: TaskConfig):
        """执行单个任务"""
        try:
            logger.info(f"Running task: {task.name}")

            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                task.func(*task.args, **task.kwargs)

            task.last_run = datetime.now()
            task.retry_count = 0
            logger.info(f"Task {task.name} completed")

        except Exception as e:
            task.retry_count += 1
            logger.error(f"Task {task.name} failed (attempt {task.retry_count}): {e}")

            if task.retry_count >= task.max_retries:
                logger.error(f"Task {task.name} exceeded max retries, skipping")
                task.retry_count = 0
                task.last_run = datetime.now()

    def _should_run(self, task: TaskConfig) -> bool:
        """判断任务是否应该执行"""
        if not task.enabled:
            return False

        if task.last_run is None:
            return True

        elapsed = (datetime.now() - task.last_run).total_seconds()
        return elapsed >= task.interval

    async def run_once(self):
        """执行一次所有待执行任务"""
        tasks_to_run = [
            self._run_task(task)
            for task in self.tasks.values()
            if self._should_run(task)
        ]

        if tasks_to_run:
            await asyncio.gather(*tasks_to_run, return_exceptions=True)

    async def run_forever(self, check_interval: int = 10):
        """持续运行调度器"""
        self.running = True
        logger.info("Data scheduler started")

        try:
            while self.running:
                await self.run_once()
                await asyncio.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted")
        finally:
            self.running = False
            logger.info("Scheduler stopped")

    def stop(self):
        """停止调度器"""
        self.running = False

    def get_status(self) -> Dict:
        """获取调度器状态"""
        return {
            "running": self.running,
            "tasks": {
                name: {
                    "enabled": task.enabled,
                    "interval": task.interval,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "retry_count": task.retry_count,
                }
                for name, task in self.tasks.items()
            }
        }


# ==================== 独立运行 ====================
import pandas as pd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CryptoQuant Data Scheduler")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--config", type=str, help="Config file path")
    args = parser.parse_args()

    scheduler = DataScheduler(config_path=args.config)

    if args.once:
        asyncio.run(scheduler.run_once())
    else:
        asyncio.run(scheduler.run_forever())
