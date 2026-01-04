"""
加密货币数据准备脚本 (v10.0.5)

从 Binance 获取历史 K 线数据，输出为 Parquet 格式。

输出目录: ~/.algvex/data/{freq}/
输出文件: {instrument}.parquet

用法:
    python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --interval 1h

Windows 兼容性:
    - 自动检测 Windows 并设置正确的事件循环策略
    - 如果 aiohttp 失败，会自动回退到同步模式 (requests)
"""

import json
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd

# Windows 兼容性修复
if sys.platform == "win32":
    # Windows 上使用 SelectorEventLoop 以兼容 aiohttp
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def fetch_binance_klines_sync(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据 (同步版本，Windows 备用方案)

    Parameters
    ----------
    trading_pair : str
        交易对，如 "BTC-USDT"
    interval : str
        K 线间隔，如 "1h", "1d"
    start_time : int
        开始时间戳 (毫秒)
    end_time : int
        结束时间戳 (毫秒)

    Returns
    -------
    pd.DataFrame
        K 线数据
    """
    import requests
    import time

    symbol = trading_pair.replace("-", "")
    url = "https://api.binance.com/api/v3/klines"

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"Error fetching {trading_pair}: {resp.status_code}")
                break

            klines = resp.json()
            if not klines:
                break

            all_klines.extend(klines)
            current_start = klines[-1][0] + 1  # 下一个时间戳

            print(f"  Fetched {len(all_klines)} klines for {trading_pair}...")

            # 避免 API 限流
            time.sleep(0.1)

        except requests.RequestException as e:
            print(f"Request error for {trading_pair}: {e}")
            break

    if not all_klines:
        return pd.DataFrame()

    # 转换为 DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    return df


async def fetch_binance_klines_async(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据 (异步版本)

    Parameters
    ----------
    trading_pair : str
        交易对，如 "BTC-USDT"
    interval : str
        K 线间隔，如 "1h", "1d"
    start_time : int
        开始时间戳 (毫秒)
    end_time : int
        结束时间戳 (毫秒)

    Returns
    -------
    pd.DataFrame
        K 线数据
    """
    import aiohttp

    symbol = trading_pair.replace("-", "")
    url = "https://api.binance.com/api/v3/klines"

    all_klines = []
    current_start = start_time

    async with aiohttp.ClientSession() as session:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"Error fetching {trading_pair}: {resp.status}")
                    break

                klines = await resp.json()
                if not klines:
                    break

                all_klines.extend(klines)
                current_start = klines[-1][0] + 1  # 下一个时间戳

                print(f"  Fetched {len(all_klines)} klines for {trading_pair}...")

    if not all_klines:
        return pd.DataFrame()

    # 转换为 DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    return df


async def fetch_binance_klines(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
    use_sync: bool = False,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据 (自动选择同步/异步)

    如果 use_sync=True 或在 Windows 上 aiohttp 失败，自动回退到同步模式。
    """
    if use_sync:
        return fetch_binance_klines_sync(trading_pair, interval, start_time, end_time)

    try:
        return await fetch_binance_klines_async(trading_pair, interval, start_time, end_time)
    except Exception as e:
        print(f"  Async fetch failed ({e}), falling back to sync mode...")
        return fetch_binance_klines_sync(trading_pair, interval, start_time, end_time)


def detect_timestamp_unit(timestamp: int) -> str:
    """自动检测时间戳单位 (秒/毫秒)"""
    if timestamp > 1e12:
        return "ms"
    return "s"


def convert_to_parquet_format(
    df: pd.DataFrame,
    trading_pair: str,
) -> pd.DataFrame:
    """
    将 Binance K 线数据转换为 Parquet 格式

    输出格式:
    - Index: datetime (UTC)
    - Columns: open, high, low, close, volume, quote_volume
    """
    if df.empty:
        return pd.DataFrame()

    # 转换时间戳
    unit = detect_timestamp_unit(df["timestamp"].iloc[0])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    df = df.set_index("datetime")

    # 只保留需要的列，使用简单列名
    result = pd.DataFrame({
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "quote_volume": df["quote_volume"].astype(float),
    })

    return result


def save_to_parquet(
    data: Dict[str, pd.DataFrame],
    output_dir: Path,
    freq: str,
):
    """
    保存为 Parquet 格式

    目录结构:
    output_dir/
    └── {freq}/
        ├── btcusdt.parquet
        ├── ethusdt.parquet
        └── metadata.json
    """
    freq_dir = output_dir / freq
    freq_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "freq": freq,
        "timezone": "UTC",
        "instruments": [],
        "columns": ["open", "high", "low", "close", "volume", "quote_volume"],
    }

    for pair, df in data.items():
        instrument = pair.lower().replace("-", "")
        file_path = freq_dir / f"{instrument}.parquet"

        # 保存 Parquet
        df.to_parquet(file_path, engine="pyarrow")

        # 更新元数据
        metadata["instruments"].append({
            "name": instrument,
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "rows": len(df),
            "gaps": int(df["close"].isna().sum()),
        })
        print(f"  Saved {instrument}: {len(df)} rows")

    # 保存元数据
    with open(freq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data saved to {freq_dir}")


async def main():
    parser = argparse.ArgumentParser(description="Prepare crypto data (Parquet)")
    parser.add_argument(
        "--trading-pairs",
        type=str,
        nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="Trading pairs to fetch",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Candle interval (1h, 4h, 1d)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.algvex/data",  # v10.0.0: 统一使用 ~/.algvex/
        help="Output directory",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous requests (recommended for Windows if async fails)",
    )

    args = parser.parse_args()

    # Windows 自动检测：如果是 Windows 且未显式指定 --sync，打印提示
    use_sync = args.sync
    if sys.platform == "win32" and not use_sync:
        print("ℹ️ Windows detected. If download fails, try: --sync")

    # 转换时间
    start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(args.end_date, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)

    output_dir = Path(args.output_dir).expanduser()

    # 获取所有交易对数据
    all_data = {}
    for pair in args.trading_pairs:
        print(f"Fetching {pair}...")
        df = await fetch_binance_klines(pair, args.interval, start_ts, end_ts, use_sync=use_sync)
        if not df.empty:
            parquet_df = convert_to_parquet_format(df, pair)
            all_data[pair] = parquet_df
            print(f"  Total: {len(parquet_df)} records")

    if not all_data:
        print("No data fetched!")
        print("\n💡 Troubleshooting tips:")
        print("  1. Check your internet connection")
        print("  2. Try with --sync flag: python scripts/prepare_crypto_data.py --sync")
        print("  3. Use mock data: python scripts/generate_mock_data.py")
        return

    # 保存为 Parquet
    save_to_parquet(all_data, output_dir, args.interval)
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    asyncio.run(main())
