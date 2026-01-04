"""
加密货币数据准备脚本 (v10.1.0) - 官方数据源版本

使用 Binance 官方数据源 (data.binance.vision) 下载历史 K 线数据。
相比 REST API 方式，速度快 10-100 倍，无 API 限流，有校验保证数据完整性。

输出目录: ~/.algvex/data/{freq}/ (可通过 ALGVEX_DATA_DIR 环境变量自定义)
输出文件: {instrument}.parquet

用法:
    pip install binance-historical-data
    python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --interval 1h

环境变量:
    ALGVEX_DATA_DIR: 自定义数据目录 (默认 ~/.algvex/data)
    HTTPS_PROXY: 代理服务器 (如 http://127.0.0.1:7890)

数据源: https://data.binance.vision/
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import date, datetime, timezone
from typing import List, Dict, Optional

import pandas as pd


# ============================================================================
# 常量定义
# ============================================================================

# 支持的时间间隔
SUPPORTED_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# CSV 列名映射 (Binance 官方格式)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]


# ============================================================================
# 工具函数
# ============================================================================

def trading_pair_to_symbol(trading_pair: str) -> str:
    """将交易对转换为 Binance symbol: BTC-USDT -> BTCUSDT"""
    return trading_pair.replace("-", "").upper()


def trading_pair_to_instrument(trading_pair: str) -> str:
    """将交易对转换为 instrument 名称: BTC-USDT -> btcusdt"""
    return trading_pair.replace("-", "").lower()


def get_default_data_dir() -> Path:
    """获取默认数据目录"""
    return Path(os.environ.get("ALGVEX_DATA_DIR", Path.home() / ".algvex" / "data"))


def check_binance_historical_data() -> bool:
    """检查 binance-historical-data 包是否已安装"""
    try:
        from binance_historical_data import BinanceDataDumper
        return True
    except ImportError:
        return False


# ============================================================================
# 数据下载 (使用官方包)
# ============================================================================

def download_with_official_package(
    symbols: List[str],
    interval: str,
    start_date: date,
    end_date: date,
    temp_dir: Path,
) -> Dict[str, Path]:
    """
    使用 binance-historical-data 官方包下载数据

    Returns
    -------
    Dict[str, Path]
        symbol -> 数据目录路径
    """
    from binance_historical_data import BinanceDataDumper

    print(f"\n📥 使用官方数据源下载 (data.binance.vision)")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   交易对: {', '.join(symbols)}")
    print(f"   间隔: {interval}")
    print()

    # 创建下载器
    dumper = BinanceDataDumper(
        path_dir_where_to_dump=str(temp_dir),
        asset_class="spot",
        data_type="klines",
        data_frequency=interval,
    )

    # 下载数据
    dumper.dump_data(
        tickers=symbols,
        date_start=start_date,
        date_end=end_date,
        is_to_update_existing=False,
    )

    # 返回数据目录
    result = {}
    for symbol in symbols:
        data_dir = temp_dir / "spot" / "klines" / symbol / interval
        if data_dir.exists():
            result[symbol] = data_dir
        else:
            print(f"   ⚠️ {symbol} 数据目录不存在")

    return result


# ============================================================================
# 数据转换
# ============================================================================

def load_csv_files(data_dir: Path) -> pd.DataFrame:
    """
    加载目录下所有 CSV 文件并合并

    Parameters
    ----------
    data_dir : Path
        包含 CSV 文件的目录

    Returns
    -------
    pd.DataFrame
        合并后的数据
    """
    all_files = sorted(data_dir.glob("*.csv"))

    if not all_files:
        return pd.DataFrame()

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 读取 {f.name} 失败: {e}")

    if not dfs:
        return pd.DataFrame()

    # 合并并去重
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"])
    combined = combined.sort_values("open_time")

    return combined


def convert_to_parquet_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Binance CSV 格式转换为 AlgVex Parquet 格式

    输出格式:
    - Index: datetime (UTC)
    - Columns: open, high, low, close, volume, quote_volume
    """
    if df.empty:
        return pd.DataFrame()

    # 检测时间戳单位 (2025年起 Binance 使用微秒)
    sample_ts = df["open_time"].iloc[0]
    if sample_ts > 1e15:  # 微秒
        unit = "us"
    elif sample_ts > 1e12:  # 毫秒
        unit = "ms"
    else:  # 秒
        unit = "s"

    # 转换时间戳
    df["datetime"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df = df.set_index("datetime")

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
    """
    保存为 Parquet 格式

    目录结构:
    output_dir/
    └── {interval}/
        ├── btcusdt.parquet
        ├── ethusdt.parquet
        └── metadata.json
    """
    freq_dir = output_dir / interval
    freq_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "freq": interval,
        "timezone": "UTC",
        "source": "data.binance.vision",
        "version": "v10.1.0",
        "instruments": [],
        "columns": ["open", "high", "low", "close", "volume", "quote_volume"],
    }

    for symbol, df in data.items():
        instrument = symbol.lower()
        file_path = freq_dir / f"{instrument}.parquet"

        # 保存 Parquet
        df.to_parquet(file_path, engine="pyarrow")

        # 更新元数据
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

    print(f"\n📁 数据已保存到: {freq_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 Binance 官方数据源下载历史 K 线数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT

  # 指定时间范围
  python scripts/prepare_crypto_data.py --start-date 2023-01-01 --end-date 2024-12-31

  # 自定义输出目录
  python scripts/prepare_crypto_data.py --output-dir /path/to/data

环境变量:
  ALGVEX_DATA_DIR  - 自定义数据目录 (默认: ~/.algvex/data)
  HTTPS_PROXY      - 代理服务器地址

数据源: https://data.binance.vision/
        """
    )

    parser.add_argument(
        "--trading-pairs",
        type=str,
        nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="交易对列表 (默认: BTC-USDT ETH-USDT)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=SUPPORTED_INTERVALS,
        help="K线间隔 (默认: 1h)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="开始日期 YYYY-MM-DD (默认: 2023-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="结束日期 YYYY-MM-DD (默认: 2024-12-31)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: ~/.algvex/data 或 ALGVEX_DATA_DIR)",
    )

    # 兼容旧参数 (忽略)
    parser.add_argument("--sync", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--proxy", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--api-base", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 检查依赖
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("AlgVex 数据准备工具 v10.1.0 (官方数据源)")
    print("=" * 60)

    if not check_binance_historical_data():
        print("\n❌ 缺少依赖: binance-historical-data")
        print("\n请安装:")
        print("  pip install binance-historical-data")
        print("\n或使用模拟数据:")
        print("  python scripts/generate_mock_data.py")
        sys.exit(1)

    try:
        import pyarrow
    except ImportError:
        print("\n❌ 缺少依赖: pyarrow")
        print("  pip install pyarrow")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 检查代理
    # -------------------------------------------------------------------------
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        print(f"\n🌐 代理已配置: {proxy}")
    else:
        print("\n🌐 未配置代理 (如遇下载问题，请设置 HTTPS_PROXY)")

    # -------------------------------------------------------------------------
    # 解析参数
    # -------------------------------------------------------------------------
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"\n❌ 日期格式错误: {e}")
        print("   请使用 YYYY-MM-DD 格式")
        sys.exit(1)

    if start_date >= end_date:
        print("\n❌ start-date 必须早于 end-date")
        sys.exit(1)

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = get_default_data_dir()

    # 转换交易对为 Binance symbol
    symbols = [trading_pair_to_symbol(p) for p in args.trading_pairs]

    print(f"\n📊 配置:")
    print(f"   交易对: {', '.join(args.trading_pairs)}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   间隔: {args.interval}")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   输出目录: {output_dir}")

    # -------------------------------------------------------------------------
    # 下载数据
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            data_dirs = download_with_official_package(
                symbols=symbols,
                interval=args.interval,
                start_date=start_date,
                end_date=end_date,
                temp_dir=temp_path,
            )
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("\n💡 故障排除:")
            print("   1. 检查网络连接")
            print("   2. 如果在中国，设置代理: export HTTPS_PROXY=http://127.0.0.1:7890")
            print("   3. 使用模拟数据: python scripts/generate_mock_data.py")
            sys.exit(1)

        if not data_dirs:
            print("\n❌ 未下载到任何数据")
            sys.exit(1)

        # ---------------------------------------------------------------------
        # 转换格式
        # ---------------------------------------------------------------------
        print("\n🔄 转换为 Parquet 格式...")

        converted_data = {}
        for symbol, data_dir in data_dirs.items():
            print(f"   处理 {symbol}...")

            # 加载 CSV
            raw_df = load_csv_files(data_dir)
            if raw_df.empty:
                print(f"   ⚠️ {symbol} 无数据")
                continue

            # 转换格式
            parquet_df = convert_to_parquet_format(raw_df)

            # 过滤时间范围
            start_ts = pd.Timestamp(start_date, tz="UTC")
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            parquet_df = parquet_df[(parquet_df.index >= start_ts) & (parquet_df.index < end_ts)]

            if not parquet_df.empty:
                converted_data[symbol] = parquet_df

    if not converted_data:
        print("\n❌ 转换后无有效数据")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 保存数据
    # -------------------------------------------------------------------------
    print("\n💾 保存数据...")
    save_to_parquet(converted_data, output_dir, args.interval)

    # -------------------------------------------------------------------------
    # 完成
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ 数据准备完成!")
    print("=" * 60)
    print(f"   下载: {', '.join(converted_data.keys())}")
    print(f"   位置: {output_dir / args.interval}")
    print(f"   数据源: data.binance.vision (官方)")


if __name__ == "__main__":
    main()
