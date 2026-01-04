"""
加密货币数据准备脚本 (v10.2.0) - 直接官方数据源

直接从 Binance 官方数据仓库 (data.binance.vision) 下载历史 K 线数据。
无第三方包依赖，仅使用 requests + pandas。

数据源: https://data.binance.vision/
输出目录: ~/.algvex/data/{interval}/
输出格式: Parquet

用法:
    python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --interval 1h

环境变量:
    ALGVEX_DATA_DIR: 自定义数据目录 (默认 ~/.algvex/data)
    HTTPS_PROXY: 代理服务器 (如 http://127.0.0.1:7890)
"""

import os
import sys
import io
import json
import zipfile
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd


# ============================================================================
# 常量定义
# ============================================================================

# Binance 官方数据源 URL
BASE_URL = "https://data.binance.vision/data/spot"

# 支持的时间间隔
SUPPORTED_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# CSV 列名 (Binance 官方格式)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]


# ============================================================================
# 工具函数
# ============================================================================

def trading_pair_to_symbol(pair: str) -> str:
    """BTC-USDT -> BTCUSDT"""
    return pair.replace("-", "").upper()


def get_default_data_dir() -> Path:
    """获取默认数据目录"""
    return Path(os.environ.get("ALGVEX_DATA_DIR", Path.home() / ".algvex" / "data"))


def get_proxy_config() -> Optional[Dict[str, str]]:
    """获取代理配置"""
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        return {"http": proxy, "https": proxy}
    return None


def generate_month_list(start_date: date, end_date: date) -> List[Tuple[int, int]]:
    """生成月份列表: [(year, month), ...]"""
    months = []
    current = date(start_date.year, start_date.month, 1)
    end = date(end_date.year, end_date.month, 1)

    while current <= end:
        months.append((current.year, current.month))
        # 下一个月
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return months


# ============================================================================
# 数据下载
# ============================================================================

def download_monthly_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    proxies: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Optional[pd.DataFrame]:
    """
    下载单月 K 线数据

    URL 格式: https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip
    """
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{filename}"

    try:
        response = requests.get(url, proxies=proxies, timeout=timeout)

        if response.status_code == 200:
            # 解压 ZIP 并读取 CSV
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_name = filename.replace(".zip", ".csv")
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                    return df

        elif response.status_code == 404:
            # 数据不存在 (可能是未来月份或交易对不存在)
            return None
        else:
            print(f"      HTTP {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"      超时")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"      连接错误: {e}")
        return None
    except Exception as e:
        print(f"      错误: {e}")
        return None


def download_symbol_data(
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    proxies: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    下载指定交易对的完整数据
    """
    months = generate_month_list(start_date, end_date)
    all_dfs = []

    print(f"\n   下载 {symbol} ({len(months)} 个月份)...")

    for i, (year, month) in enumerate(months):
        print(f"      [{i+1}/{len(months)}] {year}-{month:02d}", end=" ")

        df = download_monthly_klines(symbol, interval, year, month, proxies)

        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"✓ {len(df)} 行")
        else:
            print("- 无数据")

    if not all_dfs:
        return pd.DataFrame()

    # 合并所有月份数据
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"])
    combined = combined.sort_values("open_time")

    return combined


# ============================================================================
# 数据转换
# ============================================================================

def convert_to_parquet_format(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    将 Binance CSV 格式转换为 AlgVex Parquet 格式

    输出:
    - Index: datetime (UTC)
    - Columns: open, high, low, close, volume, quote_volume
    """
    if df.empty:
        return pd.DataFrame()

    # 检测时间戳单位
    sample_ts = df["open_time"].iloc[0]
    if sample_ts > 1e15:
        unit = "us"  # 微秒 (2025年起)
    elif sample_ts > 1e12:
        unit = "ms"  # 毫秒
    else:
        unit = "s"   # 秒

    # 转换时间戳
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.set_index("datetime")

    # 过滤时间范围
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.index >= start_ts) & (df.index < end_ts)]

    # 只保留需要的列
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
    interval: str,
) -> None:
    """保存为 Parquet 格式"""
    freq_dir = output_dir / interval
    freq_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "freq": interval,
        "timezone": "UTC",
        "source": "data.binance.vision",
        "version": "v10.2.0",
        "download_time": datetime.now().isoformat(),
        "instruments": [],
        "columns": ["open", "high", "low", "close", "volume", "quote_volume"],
    }

    for symbol, df in data.items():
        instrument = symbol.lower()
        file_path = freq_dir / f"{instrument}.parquet"

        df.to_parquet(file_path, engine="pyarrow")

        metadata["instruments"].append({
            "name": instrument,
            "symbol": symbol,
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "rows": len(df),
        })

        print(f"   ✅ {instrument}.parquet: {len(df):,} 行")

    # 保存元数据
    with open(freq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n📁 数据保存到: {freq_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 Binance 官方数据源下载历史 K 线数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT
  python scripts/prepare_crypto_data.py --interval 4h --start-date 2024-01-01

数据源: https://data.binance.vision/
环境变量:
  ALGVEX_DATA_DIR  - 自定义数据目录
  HTTPS_PROXY      - 代理服务器
        """
    )

    parser.add_argument(
        "--trading-pairs", type=str, nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="交易对列表 (默认: BTC-USDT ETH-USDT)",
    )
    parser.add_argument(
        "--interval", type=str, default="1h",
        choices=SUPPORTED_INTERVALS,
        help="K线间隔 (默认: 1h)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2023-01-01",
        help="开始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-12-31",
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认: ~/.algvex/data)",
    )

    # 兼容旧参数 (忽略)
    parser.add_argument("--sync", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--proxy", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--api-base", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # =========================================================================
    # 初始化
    # =========================================================================
    print("=" * 60)
    print("AlgVex 数据准备工具 v10.2.0")
    print("数据源: data.binance.vision (Binance 官方)")
    print("=" * 60)

    # 检查依赖
    try:
        import pyarrow
    except ImportError:
        print("\n❌ 缺少依赖: pyarrow")
        print("   pip install pyarrow")
        sys.exit(1)

    # 代理配置
    proxies = get_proxy_config()
    if proxies:
        print(f"\n🌐 代理: {proxies['https']}")
    else:
        print("\n🌐 未配置代理 (如需代理，设置 HTTPS_PROXY 环境变量)")

    # 解析日期
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"\n❌ 日期格式错误: {e}")
        sys.exit(1)

    if start_date >= end_date:
        print("\n❌ start-date 必须早于 end-date")
        sys.exit(1)

    # 输出目录
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else get_default_data_dir()

    # 转换交易对
    symbols = [trading_pair_to_symbol(p) for p in args.trading_pairs]

    print(f"\n📊 配置:")
    print(f"   交易对: {', '.join(symbols)}")
    print(f"   间隔: {args.interval}")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   输出目录: {output_dir}")

    # =========================================================================
    # 下载数据
    # =========================================================================
    print("\n" + "=" * 60)
    print("📥 开始下载")
    print("=" * 60)

    all_data = {}
    failed = []

    for symbol in symbols:
        raw_df = download_symbol_data(symbol, args.interval, start_date, end_date, proxies)

        if raw_df.empty:
            failed.append(symbol)
            print(f"   ⚠️ {symbol}: 无数据")
            continue

        # 转换格式
        parquet_df = convert_to_parquet_format(raw_df, start_date, end_date)

        if parquet_df.empty:
            failed.append(symbol)
            print(f"   ⚠️ {symbol}: 转换后无数据")
            continue

        all_data[symbol] = parquet_df
        print(f"   ✅ {symbol}: {len(parquet_df):,} 行 ({parquet_df.index.min().date()} ~ {parquet_df.index.max().date()})")

    # =========================================================================
    # 保存数据
    # =========================================================================
    if not all_data:
        print("\n" + "=" * 60)
        print("❌ 未下载到任何数据")
        print("=" * 60)
        print("\n💡 故障排除:")
        print("   1. 检查网络连接")
        print("   2. 设置代理: export HTTPS_PROXY=http://127.0.0.1:7890")
        print("   3. 使用模拟数据: python scripts/generate_mock_data.py")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("💾 保存数据")
    print("=" * 60)

    save_to_parquet(all_data, output_dir, args.interval)

    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ 数据准备完成!")
    print("=" * 60)
    print(f"   成功: {', '.join(all_data.keys())}")
    print(f"   位置: {output_dir / args.interval}")

    if failed:
        print(f"   失败: {', '.join(failed)}")


if __name__ == "__main__":
    main()
