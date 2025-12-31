#!/usr/bin/env python
"""
数据采集脚本

用法:
    python scripts/collect_data.py --days 365
    python scripts/collect_data.py --start 2023-01-01 --end 2024-01-01
    python scripts/collect_data.py --daemon  # 后台持续运行
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from data_collector.binance_collector import BinanceDataCollector
from data_collector.sentiment_collector import SentimentDataCollector
from data_collector.scheduler import DataScheduler


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level,
    )
    logger.add(
        Path("~/.cryptoquant/logs/collect_{time:YYYY-MM-DD}.log").expanduser(),
        rotation="1 day",
        retention="30 days",
        level=log_level,
    )


def collect_historical(
    symbols: list,
    start_date: str,
    end_date: str,
    interval: str = "1h",
    data_dir: str = None,
):
    """
    采集历史数据

    Args:
        symbols: 标的列表
        start_date: 开始日期
        end_date: 结束日期
        interval: K线周期
        data_dir: 数据目录
    """
    if data_dir is None:
        data_dir = Path("~/.cryptoquant/data").expanduser()
    else:
        data_dir = Path(data_dir).expanduser()

    data_dir.mkdir(parents=True, exist_ok=True)

    # 初始化采集器
    binance = BinanceDataCollector()
    sentiment = SentimentDataCollector()

    logger.info(f"Collecting data from {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Interval: {interval}")

    # 转换日期
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    # 采集K线数据
    logger.info("Collecting klines...")
    for symbol in symbols:
        try:
            data = binance.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_ts,
                end_time=end_ts,
                limit=1500,
            )
            if not data.empty:
                save_path = data_dir / f"klines/{interval}/{symbol}.parquet"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(save_path)
                logger.info(f"  {symbol}: {len(data)} records")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")

    # 采集资金费率
    logger.info("Collecting funding rates...")
    for symbol in symbols:
        try:
            data = binance.fetch_funding_rate(
                symbol=symbol,
                start_time=start_ts,
                end_time=end_ts,
            )
            if not data.empty:
                save_path = data_dir / f"funding_rate/{symbol}.parquet"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(save_path)
                logger.info(f"  {symbol}: {len(data)} records")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")

    # 采集持仓量
    logger.info("Collecting open interest...")
    for symbol in symbols:
        try:
            data = binance.fetch_open_interest_history(
                symbol=symbol,
                period="1h",
                start_time=start_ts,
                end_time=end_ts,
            )
            if not data.empty:
                save_path = data_dir / f"open_interest/{symbol}.parquet"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(save_path)
                logger.info(f"  {symbol}: {len(data)} records")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")

    # 采集多空比
    logger.info("Collecting long/short ratio...")
    for symbol in symbols:
        try:
            data = binance.fetch_long_short_ratio(
                symbol=symbol,
                period="1h",
                limit=500,
            )
            if not data.empty:
                save_path = data_dir / f"long_short_ratio/{symbol}.parquet"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(save_path)
                logger.info(f"  {symbol}: {len(data)} records")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")

    # 采集大户持仓比
    logger.info("Collecting taker buy/sell ratio...")
    for symbol in symbols:
        try:
            data = binance.fetch_taker_buy_sell_ratio(
                symbol=symbol,
                period="1h",
                limit=500,
            )
            if not data.empty:
                save_path = data_dir / f"taker_ratio/{symbol}.parquet"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(save_path)
                logger.info(f"  {symbol}: {len(data)} records")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")

    # 采集情绪数据
    logger.info("Collecting sentiment data...")
    try:
        days = (end_dt - start_dt).days
        fear_greed = sentiment.fetch_fear_greed_index(limit=min(days, 365))
        if not fear_greed.empty:
            save_path = data_dir / "sentiment/fear_greed.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fear_greed.to_parquet(save_path)
            logger.info(f"  Fear/Greed: {len(fear_greed)} records")
    except Exception as e:
        logger.error(f"  Fear/Greed: Failed - {e}")

    try:
        tvl = sentiment.fetch_tvl_history()
        if not tvl.empty:
            save_path = data_dir / "defi/tvl.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tvl.to_parquet(save_path)
            logger.info(f"  TVL: {len(tvl)} records")
    except Exception as e:
        logger.error(f"  TVL: Failed - {e}")

    try:
        stablecoins = sentiment.fetch_stablecoin_supply()
        if not stablecoins.empty:
            save_path = data_dir / "defi/stablecoins.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            stablecoins.to_parquet(save_path)
            logger.info(f"  Stablecoins: {len(stablecoins)} records")
    except Exception as e:
        logger.error(f"  Stablecoins: Failed - {e}")

    logger.info("Data collection completed!")


def run_daemon():
    """后台持续运行"""
    scheduler = DataScheduler()
    asyncio.run(scheduler.run_forever())


def main():
    parser = argparse.ArgumentParser(description="CryptoQuant Data Collector")

    parser.add_argument(
        "--symbols",
        type=str,
        default="btcusdt,ethusdt,bnbusdt,solusdt,xrpusdt",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to collect",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Kline interval (1m, 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.daemon:
        logger.info("Starting data collector daemon...")
        run_daemon()
    else:
        # 确定时间范围
        if args.days:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        elif args.start and args.end:
            start_date = args.start
            end_date = args.end
        else:
            # 默认采集最近90天
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        symbols = [s.strip().lower() for s in args.symbols.split(",")]

        collect_historical(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
