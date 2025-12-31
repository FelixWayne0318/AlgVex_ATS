#!/usr/bin/env python3
"""
AlgVex 数据自动落盘脚本

功能:
1. 定期从币安采集所有免费数据
2. 增量保存到本地 Parquet 文件
3. 自动去重和数据合并
4. 支持多种运行模式 (一次性/定时/守护进程)

使用方法:
    # 一次性采集
    python data_archiver.py --once

    # 定时采集 (每小时)
    python data_archiver.py --interval 3600

    # 守护进程模式
    python data_archiver.py --daemon

数据存储结构:
    ~/.algvex/data/
    ├── klines/
    │   ├── BTCUSDT_1h.parquet
    │   └── ETHUSDT_1h.parquet
    ├── funding/
    │   └── funding_rate.parquet
    ├── oi/
    │   └── open_interest.parquet
    ├── ls_ratio/
    │   └── long_short_ratio.parquet
    └── taker/
        └── taker_buy_sell.parquet
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.core.data.collector import BinanceDataCollector

try:
    from loguru import logger
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
    logger = logging.getLogger(__name__)


class DataArchiver:
    """
    数据落盘管理器

    特性:
    - 增量采集: 只采集新数据
    - 自动去重: 基于时间戳去重
    - 故障恢复: 断点续采
    - 数据压缩: Parquet格式高效存储
    """

    # 默认配置
    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    DEFAULT_DATA_DIR = "~/.algvex/data"
    DEFAULT_INTERVAL = "1h"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = None,
        interval: str = None,
        rate_limit_delay: float = 0.1,
    ):
        """
        初始化落盘管理器

        Args:
            symbols: 交易对列表
            data_dir: 数据存储目录
            interval: K线周期
            rate_limit_delay: API调用间隔
        """
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.data_dir = Path(data_dir or self.DEFAULT_DATA_DIR).expanduser()
        self.interval = interval or self.DEFAULT_INTERVAL
        self.rate_limit_delay = rate_limit_delay

        # 创建目录结构
        self._create_directories()

        # 初始化采集器
        self.collector = BinanceDataCollector(
            symbols=self.symbols,
            data_dir=str(self.data_dir),
            rate_limit_delay=rate_limit_delay,
        )

        # 运行状态
        self._running = True
        self._setup_signal_handlers()

        logger.info(f"DataArchiver initialized")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Interval: {self.interval}")

    def _create_directories(self):
        """创建数据目录结构"""
        dirs = ["klines", "funding", "oi", "ls_ratio", "taker", "logs"]
        for d in dirs:
            (self.data_dir / d).mkdir(parents=True, exist_ok=True)

    def _setup_signal_handlers(self):
        """设置信号处理"""
        def handler(signum, frame):
            logger.info("Received shutdown signal, stopping...")
            self._running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _get_last_timestamp(self, data_type: str, symbol: str = None) -> Optional[int]:
        """获取最后一条数据的时间戳"""
        if data_type == "klines":
            file_path = self.data_dir / "klines" / f"{symbol}_{self.interval}.parquet"
        else:
            file_path = self.data_dir / data_type / f"{data_type}.parquet"

        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return None

            if symbol and "symbol" in df.columns:
                df = df[df["symbol"] == symbol]

            if df.empty:
                return None

            last_time = df["datetime"].max()
            if pd.isna(last_time):
                return None

            return int(pd.Timestamp(last_time).timestamp() * 1000)
        except Exception as e:
            logger.warning(f"Failed to read last timestamp from {file_path}: {e}")
            return None

    def _save_incremental(self, data_type: str, df: pd.DataFrame, symbol: str = None):
        """增量保存数据"""
        if df.empty:
            return

        if data_type == "klines":
            file_path = self.data_dir / "klines" / f"{symbol}_{self.interval}.parquet"
        else:
            file_path = self.data_dir / data_type / f"{data_type}.parquet"

        # 读取现有数据
        if file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                # 合并并去重
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(
                    subset=["datetime", "symbol"] if "symbol" in combined_df.columns else ["datetime"],
                    keep="last"
                )
                combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Failed to read existing data, overwriting: {e}")
                combined_df = df
        else:
            combined_df = df

        # 保存
        combined_df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(df)} new rows to {file_path.name} (total: {len(combined_df)})")

    def collect_klines(self, symbol: str) -> pd.DataFrame:
        """采集K线数据 (增量)"""
        # 获取上次采集的最后时间
        last_ts = self._get_last_timestamp("klines", symbol)

        if last_ts:
            # 从上次结束时间开始采集
            start_ts = last_ts + 1
            logger.debug(f"Incremental klines for {symbol} from {datetime.fromtimestamp(start_ts/1000)}")
        else:
            # 首次采集，获取最近30天
            start_ts = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            logger.debug(f"Initial klines for {symbol}")

        end_ts = int(datetime.now().timestamp() * 1000)

        # 分页采集
        all_data = []
        current_start = start_ts

        while current_start < end_ts:
            df = self.collector.fetch_klines(
                symbol=symbol,
                interval=self.interval,
                start_time=current_start,
                end_time=end_ts,
                limit=1500
            )

            if df.empty:
                break

            all_data.append(df)

            # 更新起始时间
            last_time = df["datetime"].max()
            current_start = int(pd.Timestamp(last_time).timestamp() * 1000) + 1

            # 如果返回数量少于limit，说明已经到达终点
            if len(df) < 1500:
                break

        if all_data:
            result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["datetime"])
            return result
        return pd.DataFrame()

    def collect_funding_rate(self, symbol: str) -> pd.DataFrame:
        """采集资金费率 (增量)"""
        last_ts = self._get_last_timestamp("funding", symbol)

        if last_ts:
            start_ts = last_ts + 1
        else:
            # 资金费率历史较长，尝试获取更多
            start_ts = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

        end_ts = int(datetime.now().timestamp() * 1000)

        # 分页采集
        all_data = []
        current_start = start_ts

        while current_start < end_ts:
            df = self.collector.fetch_funding_rate(
                symbol=symbol,
                start_time=current_start,
                end_time=end_ts,
                limit=1000
            )

            if df.empty:
                break

            all_data.append(df)

            last_time = df["datetime"].max()
            current_start = int(pd.Timestamp(last_time).timestamp() * 1000) + 1

            if len(df) < 1000:
                break

        if all_data:
            result = pd.concat(all_data, ignore_index=True).drop_duplicates(subset=["datetime", "symbol"])
            return result
        return pd.DataFrame()

    def collect_open_interest(self, symbol: str) -> pd.DataFrame:
        """采集持仓量 (仅最近30天可用)"""
        # 持仓量历史API只有约30天数据
        df = self.collector.fetch_open_interest_history(
            symbol=symbol,
            period=self.interval,
            limit=500
        )
        return df

    def collect_long_short_ratio(self, symbol: str) -> pd.DataFrame:
        """采集多空比 (仅最近30天可用)"""
        df = self.collector.fetch_long_short_ratio(
            symbol=symbol,
            period=self.interval,
            limit=500
        )
        return df

    def collect_taker_ratio(self, symbol: str) -> pd.DataFrame:
        """采集主动买卖比 (仅最近30天可用)"""
        df = self.collector.fetch_taker_long_short_ratio(
            symbol=symbol,
            period=self.interval,
            limit=500
        )
        return df

    def run_once(self) -> Dict[str, int]:
        """
        执行一次完整采集

        Returns:
            各数据类型采集的记录数
        """
        logger.info("=" * 50)
        logger.info(f"Starting data collection at {datetime.now()}")
        logger.info("=" * 50)

        stats = {
            "klines": 0,
            "funding": 0,
            "oi": 0,
            "ls_ratio": 0,
            "taker": 0,
        }

        for symbol in self.symbols:
            if not self._running:
                break

            logger.info(f"Collecting data for {symbol}...")

            # 1. K线 (有完整历史)
            try:
                klines = self.collect_klines(symbol)
                if not klines.empty:
                    self._save_incremental("klines", klines, symbol)
                    stats["klines"] += len(klines)
            except Exception as e:
                logger.error(f"Failed to collect klines for {symbol}: {e}")

            # 2. 资金费率 (有完整历史)
            try:
                funding = self.collect_funding_rate(symbol)
                if not funding.empty:
                    self._save_incremental("funding", funding)
                    stats["funding"] += len(funding)
            except Exception as e:
                logger.error(f"Failed to collect funding rate for {symbol}: {e}")

            # 3. 持仓量 (仅30天)
            try:
                oi = self.collect_open_interest(symbol)
                if not oi.empty:
                    self._save_incremental("oi", oi)
                    stats["oi"] += len(oi)
            except Exception as e:
                logger.error(f"Failed to collect open interest for {symbol}: {e}")

            # 4. 多空比 (仅30天)
            try:
                ls_ratio = self.collect_long_short_ratio(symbol)
                if not ls_ratio.empty:
                    self._save_incremental("ls_ratio", ls_ratio)
                    stats["ls_ratio"] += len(ls_ratio)
            except Exception as e:
                logger.error(f"Failed to collect long/short ratio for {symbol}: {e}")

            # 5. 主动买卖比 (仅30天)
            try:
                taker = self.collect_taker_ratio(symbol)
                if not taker.empty:
                    self._save_incremental("taker", taker)
                    stats["taker"] += len(taker)
            except Exception as e:
                logger.error(f"Failed to collect taker ratio for {symbol}: {e}")

        logger.info("-" * 50)
        logger.info("Collection completed:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v} records")
        logger.info("-" * 50)

        return stats

    def run_scheduled(self, interval_seconds: int = 3600):
        """
        定时运行采集

        Args:
            interval_seconds: 采集间隔(秒), 默认1小时
        """
        logger.info(f"Starting scheduled collection (interval: {interval_seconds}s)")

        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Collection failed: {e}")

            if not self._running:
                break

            # 等待下次采集
            logger.info(f"Next collection in {interval_seconds} seconds...")
            for _ in range(interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("Scheduled collection stopped")

    def get_data_stats(self) -> Dict[str, Dict]:
        """获取数据统计信息"""
        stats = {}

        # K线统计
        klines_dir = self.data_dir / "klines"
        if klines_dir.exists():
            klines_stats = {}
            for f in klines_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f)
                    klines_stats[f.stem] = {
                        "rows": len(df),
                        "start": str(df["datetime"].min()) if not df.empty else None,
                        "end": str(df["datetime"].max()) if not df.empty else None,
                    }
                except Exception:
                    pass
            stats["klines"] = klines_stats

        # 其他数据统计
        for data_type in ["funding", "oi", "ls_ratio", "taker"]:
            file_path = self.data_dir / data_type / f"{data_type}.parquet"
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    stats[data_type] = {
                        "rows": len(df),
                        "start": str(df["datetime"].min()) if not df.empty else None,
                        "end": str(df["datetime"].max()) if not df.empty else None,
                        "symbols": list(df["symbol"].unique()) if "symbol" in df.columns else [],
                    }
                except Exception:
                    pass

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="AlgVex 数据自动落盘脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 一次性采集
  python data_archiver.py --once

  # 每小时采集一次
  python data_archiver.py --interval 3600

  # 查看数据统计
  python data_archiver.py --stats

  # 自定义交易对和目录
  python data_archiver.py --once --symbols BTCUSDT,ETHUSDT --data-dir /data/crypto
        """
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="执行一次采集后退出"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="定时采集间隔(秒), 默认3600"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="守护进程模式运行"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示数据统计信息"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="交易对列表(逗号分隔), 例如: BTCUSDT,ETHUSDT"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据存储目录, 默认: ~/.algvex/data"
    )
    parser.add_argument(
        "--kline-interval",
        type=str,
        default="1h",
        help="K线周期, 默认: 1h"
    )

    args = parser.parse_args()

    # 解析交易对
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # 初始化
    archiver = DataArchiver(
        symbols=symbols,
        data_dir=args.data_dir,
        interval=args.kline_interval,
    )

    # 执行相应操作
    if args.stats:
        stats = archiver.get_data_stats()
        print("\n📊 数据统计信息:")
        print("=" * 60)
        for data_type, info in stats.items():
            print(f"\n📁 {data_type}:")
            if isinstance(info, dict) and "rows" in info:
                print(f"   记录数: {info['rows']}")
                print(f"   开始时间: {info['start']}")
                print(f"   结束时间: {info['end']}")
                if info.get("symbols"):
                    print(f"   交易对: {', '.join(info['symbols'])}")
            elif isinstance(info, dict):
                for name, details in info.items():
                    print(f"   {name}: {details['rows']} 条 ({details['start']} ~ {details['end']})")
        print("\n" + "=" * 60)
        return

    if args.once:
        archiver.run_once()
    elif args.daemon or args.interval:
        archiver.run_scheduled(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
