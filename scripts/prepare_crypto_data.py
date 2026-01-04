"""
加密货币数据准备脚本 (v10.0.6)

从 Binance 获取历史 K 线数据，输出为 Parquet 格式。

输出目录: ~/.algvex/data/{freq}/ (可通过 ALGVEX_DATA_DIR 环境变量自定义)
输出文件: {instrument}.parquet

用法:
    python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --interval 1h

环境变量:
    ALGVEX_DATA_DIR: 自定义数据目录 (默认 ~/.algvex/data)
    HTTPS_PROXY: 代理服务器 (如 http://127.0.0.1:7890)

Windows 兼容性:
    - 自动检测 Windows 并设置正确的事件循环策略
    - 如果 aiohttp 失败，会自动回退到同步模式 (requests)

网络问题排查:
    - 中国用户: 需要代理，设置 HTTPS_PROXY 环境变量
    - 美国用户: 可以尝试 --api-base https://api.binance.us
    - 网络不稳定: 脚本会自动重试 3 次
"""

import json
import os
import sys
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

# Windows 兼容性修复
if sys.platform == "win32":
    # Windows 上使用 SelectorEventLoop 以兼容 aiohttp
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ============================================================================
# 错误码定义
# ============================================================================

class BinanceError:
    """Binance API 错误码及其含义"""

    ERROR_MESSAGES = {
        400: "请求参数错误，请检查交易对格式 (如 BTC-USDT)",
        403: "访问被拒绝。可能原因:\n"
             "   - 您的 IP 所在地区被 Binance 限制 (中国大陆、美国等)\n"
             "   - 请设置代理: export HTTPS_PROXY=http://127.0.0.1:7890",
        418: "您的 IP 已被 Binance 临时封禁 (请求过于频繁)，请等待几分钟后重试",
        429: "请求频率过高，触发了 API 限流。请稍等后重试",
        451: "您所在的地区无法使用 Binance 服务",
        500: "Binance 服务器内部错误，请稍后重试",
        502: "Binance 网关错误，请稍后重试",
        503: "Binance 服务暂时不可用，请稍后重试",
    }

    @classmethod
    def get_message(cls, status_code: int) -> str:
        """获取错误码对应的中文说明"""
        return cls.ERROR_MESSAGES.get(status_code, f"未知错误 (HTTP {status_code})")


# ============================================================================
# 依赖检查
# ============================================================================

def check_dependencies() -> Tuple[bool, bool]:
    """
    检查必要的依赖是否已安装

    Returns
    -------
    Tuple[bool, bool]
        (requests_available, aiohttp_available)
    """
    requests_available = False
    aiohttp_available = False

    try:
        import requests  # noqa: F401
        requests_available = True
    except ImportError:
        pass

    try:
        import aiohttp  # noqa: F401
        aiohttp_available = True
    except ImportError:
        pass

    return requests_available, aiohttp_available


def check_pyarrow() -> bool:
    """检查 pyarrow 是否可用"""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# 同步版本 (使用 requests)
# ============================================================================

def fetch_binance_klines_sync(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
    api_base: str = "https://api.binance.com",
    proxy: Optional[str] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据 (同步版本)

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
    api_base : str
        API 基础 URL
    proxy : str, optional
        代理服务器地址
    max_retries : int
        最大重试次数

    Returns
    -------
    pd.DataFrame
        K 线数据
    """
    import requests
    from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError

    symbol = trading_pair.replace("-", "")
    url = f"{api_base}/api/v3/klines"

    all_klines = []
    current_start = start_time
    consecutive_errors = 0

    # 配置代理
    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}
        print(f"  Using proxy: {proxy}")

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }

        # 重试逻辑
        for retry in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    timeout=30,
                    proxies=proxies
                )

                if resp.status_code == 200:
                    klines = resp.json()
                    if not klines:
                        # 没有更多数据
                        return _convert_klines_to_df(all_klines)

                    all_klines.extend(klines)
                    current_start = klines[-1][0] + 1
                    consecutive_errors = 0
                    print(f"  Fetched {len(all_klines)} klines for {trading_pair}...")

                    # 避免 API 限流
                    time.sleep(0.1)
                    break

                elif resp.status_code == 429:
                    # 速率限制，等待更长时间
                    wait_time = 2 ** (retry + 2)  # 4, 8, 16 秒
                    print(f"  Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                else:
                    error_msg = BinanceError.get_message(resp.status_code)
                    print(f"  Error: {error_msg}")

                    # 403/451 是地区限制，重试无意义
                    if resp.status_code in (403, 451):
                        return pd.DataFrame()

                    # 其他错误，重试
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry  # 1, 2, 4 秒
                        print(f"  Retrying in {wait_time}s... ({retry + 1}/{max_retries})")
                        time.sleep(wait_time)
                    continue

            except Timeout:
                print(f"  Request timeout, retrying... ({retry + 1}/{max_retries})")
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)
                continue

            except ReqConnectionError as e:
                print(f"  Connection error: {e}")
                if "Connection refused" in str(e):
                    print("  💡 提示: 请检查网络连接或代理设置")
                if retry < max_retries - 1:
                    print(f"  Retrying in {2 ** retry}s... ({retry + 1}/{max_retries})")
                    time.sleep(2 ** retry)
                continue

            except RequestException as e:
                print(f"  Request error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    print("  Too many consecutive errors, stopping.")
                    return _convert_klines_to_df(all_klines)
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)
                continue
        else:
            # 所有重试都失败了
            print(f"  Failed after {max_retries} retries, stopping.")
            break

    return _convert_klines_to_df(all_klines)


# ============================================================================
# 异步版本 (使用 aiohttp)
# ============================================================================

async def fetch_binance_klines_async(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
    api_base: str = "https://api.binance.com",
    proxy: Optional[str] = None,
    max_retries: int = 3,
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
    api_base : str
        API 基础 URL
    proxy : str, optional
        代理服务器地址
    max_retries : int
        最大重试次数

    Returns
    -------
    pd.DataFrame
        K 线数据
    """
    import aiohttp
    from aiohttp import ClientTimeout, ClientError

    symbol = trading_pair.replace("-", "")
    url = f"{api_base}/api/v3/klines"

    all_klines = []
    current_start = start_time

    # 设置超时
    timeout = ClientTimeout(total=30)

    if proxy:
        print(f"  Using proxy: {proxy}")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            }

            for retry in range(max_retries):
                try:
                    async with session.get(url, params=params, proxy=proxy) as resp:
                        if resp.status == 200:
                            klines = await resp.json()
                            if not klines:
                                return _convert_klines_to_df(all_klines)

                            all_klines.extend(klines)
                            current_start = klines[-1][0] + 1
                            print(f"  Fetched {len(all_klines)} klines for {trading_pair}...")

                            # 避免 API 限流
                            await asyncio.sleep(0.1)
                            break

                        elif resp.status == 429:
                            wait_time = 2 ** (retry + 2)
                            print(f"  Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        else:
                            error_msg = BinanceError.get_message(resp.status)
                            print(f"  Error: {error_msg}")

                            if resp.status in (403, 451):
                                return pd.DataFrame()

                            if retry < max_retries - 1:
                                await asyncio.sleep(2 ** retry)
                            continue

                except asyncio.TimeoutError:
                    print(f"  Request timeout, retrying... ({retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    continue

                except ClientError as e:
                    print(f"  Client error: {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(2 ** retry)
                    continue

            else:
                print(f"  Failed after {max_retries} retries, stopping.")
                break

    return _convert_klines_to_df(all_klines)


# ============================================================================
# 统一入口
# ============================================================================

async def fetch_binance_klines(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
    use_sync: bool = False,
    api_base: str = "https://api.binance.com",
    proxy: Optional[str] = None,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据 (自动选择同步/异步)

    如果 use_sync=True 或 aiohttp 不可用/失败，自动回退到同步模式。
    """
    if use_sync:
        return fetch_binance_klines_sync(
            trading_pair, interval, start_time, end_time,
            api_base=api_base, proxy=proxy
        )

    try:
        return await fetch_binance_klines_async(
            trading_pair, interval, start_time, end_time,
            api_base=api_base, proxy=proxy
        )
    except Exception as e:
        print(f"  Async fetch failed ({e}), falling back to sync mode...")
        return fetch_binance_klines_sync(
            trading_pair, interval, start_time, end_time,
            api_base=api_base, proxy=proxy
        )


# ============================================================================
# 数据转换工具
# ============================================================================

def _convert_klines_to_df(klines: list) -> pd.DataFrame:
    """将 K 线列表转换为 DataFrame"""
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    return df


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


# ============================================================================
# 主函数
# ============================================================================

def get_default_data_dir() -> str:
    """获取默认数据目录，支持环境变量自定义"""
    return os.environ.get("ALGVEX_DATA_DIR", "~/.algvex/data")


def print_troubleshooting_tips(has_proxy: bool, is_china: bool = False):
    """打印故障排除提示"""
    print("\n" + "=" * 60)
    print("💡 数据下载故障排除指南")
    print("=" * 60)

    print("\n1. 网络连接问题:")
    print("   - 检查您的网络连接是否正常")
    print("   - 尝试访问 https://api.binance.com/api/v3/ping")

    print("\n2. 地区限制问题:")
    print("   - 中国大陆用户需要使用代理")
    print("   - 美国用户可以使用 Binance.US:")
    print("     python scripts/prepare_crypto_data.py --api-base https://api.binance.us")

    if not has_proxy:
        print("\n3. 代理设置:")
        print("   方法 1 - 环境变量:")
        print("     export HTTPS_PROXY=http://127.0.0.1:7890")
        print("   方法 2 - 命令行参数:")
        print("     python scripts/prepare_crypto_data.py --proxy http://127.0.0.1:7890")

    print("\n4. 其他解决方案:")
    print("   - 使用 --sync 标志强制同步模式:")
    print("     python scripts/prepare_crypto_data.py --sync")
    print("   - 使用模拟数据进行测试:")
    print("     python scripts/generate_mock_data.py")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Prepare crypto data from Binance (Parquet format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT

  # 使用代理 (中国用户)
  python scripts/prepare_crypto_data.py --proxy http://127.0.0.1:7890

  # 美国用户使用 Binance.US
  python scripts/prepare_crypto_data.py --api-base https://api.binance.us

  # 自定义输出目录
  python scripts/prepare_crypto_data.py --output-dir /path/to/data

Environment Variables:
  ALGVEX_DATA_DIR  - Custom data directory (default: ~/.algvex/data)
  HTTPS_PROXY      - Proxy server for network requests
        """
    )
    parser.add_argument(
        "--trading-pairs",
        type=str,
        nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="Trading pairs to fetch (default: BTC-USDT ETH-USDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d (default: 1h)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date YYYY-MM-DD (default: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # 将在运行时从环境变量获取默认值
        help="Output directory (default: ~/.algvex/data or ALGVEX_DATA_DIR)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Force synchronous requests (recommended if async fails)",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy server URL (e.g., http://127.0.0.1:7890)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.binance.com",
        help="Binance API base URL (default: https://api.binance.com)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 依赖检查
    # -------------------------------------------------------------------------
    print("Checking dependencies...")

    requests_available, aiohttp_available = check_dependencies()

    if not requests_available and not aiohttp_available:
        print("ERROR: Neither 'requests' nor 'aiohttp' is installed!")
        print("Please install at least one:")
        print("  pip install requests")
        print("  pip install aiohttp")
        sys.exit(1)

    if not check_pyarrow():
        print("ERROR: 'pyarrow' is not installed!")
        print("Please install it:")
        print("  pip install pyarrow")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 确定是否使用同步模式
    # -------------------------------------------------------------------------
    use_sync = args.sync

    if not aiohttp_available:
        print("Note: aiohttp not available, using synchronous mode")
        use_sync = True
    elif sys.platform == "win32" and not use_sync:
        print("Note: Windows detected. If download fails, try: --sync")

    if not requests_available and use_sync:
        print("ERROR: --sync flag requires 'requests' package!")
        print("  pip install requests")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 代理设置
    # -------------------------------------------------------------------------
    proxy = args.proxy
    if not proxy:
        # 检查环境变量
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    # -------------------------------------------------------------------------
    # 输出目录
    # -------------------------------------------------------------------------
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = Path(get_default_data_dir()).expanduser()

    print(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # 转换时间
    # -------------------------------------------------------------------------
    try:
        start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(args.end_date, "%Y-%m-%d")
                     .replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        print("Please use YYYY-MM-DD format (e.g., 2023-01-01)")
        sys.exit(1)

    if start_ts >= end_ts:
        print("ERROR: start-date must be before end-date")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 获取数据
    # -------------------------------------------------------------------------
    print(f"\nFetching data from {args.start_date} to {args.end_date}")
    print(f"API: {args.api_base}")
    print(f"Interval: {args.interval}")
    print(f"Trading pairs: {', '.join(args.trading_pairs)}")
    print()

    all_data = {}
    failed_pairs = []

    for pair in args.trading_pairs:
        print(f"Fetching {pair}...")
        df = await fetch_binance_klines(
            pair, args.interval, start_ts, end_ts,
            use_sync=use_sync,
            api_base=args.api_base,
            proxy=proxy
        )

        if not df.empty:
            parquet_df = convert_to_parquet_format(df, pair)
            all_data[pair] = parquet_df
            print(f"  Total: {len(parquet_df)} records\n")
        else:
            failed_pairs.append(pair)
            print(f"  No data fetched for {pair}\n")

    # -------------------------------------------------------------------------
    # 结果处理
    # -------------------------------------------------------------------------
    if not all_data:
        print("ERROR: No data fetched for any trading pair!")
        print_troubleshooting_tips(has_proxy=bool(proxy))
        sys.exit(1)

    if failed_pairs:
        print(f"Warning: Failed to fetch data for: {', '.join(failed_pairs)}")

    # 保存为 Parquet
    save_to_parquet(all_data, output_dir, args.interval)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"Successfully downloaded: {', '.join(all_data.keys())}")
    print(f"Data location: {output_dir / args.interval}")

    if failed_pairs:
        print(f"\nFailed pairs: {', '.join(failed_pairs)}")
        print("Use --proxy option if you're in a restricted region.")


if __name__ == "__main__":
    asyncio.run(main())
