"""
币安数据采集器 - 采集永续合约所有免费数据 (Qlib 风格增强版)

数据类型:
1. K线数据 (OHLCV)
2. 资金费率 (Funding Rate)
3. 持仓量 (Open Interest)
4. 多空比 (Long/Short Ratio)
5. 大户持仓比 (Top Trader Ratio)
6. 主动买卖比 (Taker Buy/Sell)
7. 订单簿快照 (Order Book)

增强特性 (来自 Qlib):
- 自动重试机制 (max_collector_count)
- 并行采集支持 (joblib)
- 数据完整性验证 (check_data_length)
- 详细进度追踪
- 本地缓存机制 (类似 Qlib 的 provider_uri 和 DiskDatasetCache)

缓存机制说明:
    类似 Qlib 的 provider_uri，数据存储在 data_dir 目录下
    使用 use_cache=True 时，优先从本地加载，只下载缺失/过期的数据
    缓存有效期由 cache_valid_days 控制
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# Qlib 风格的重试和并行装饰器
# ============================================================

def retry_on_failure(max_retries: int = 3, delay: float = 2.0):
    """
    重试装饰器 (来自 Qlib 的 max_collector_count 模式)

    Args:
        max_retries: 最大重试次数
        delay: 重试间隔(秒)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")
            return None
        return wrapper
    return decorator


class BinanceDataCollector:
    """
    币安永续合约数据采集器 (Qlib 风格增强版)

    特性:
    - 自动重试: max_collector_count 次失败后重试
    - 并行采集: 使用 ThreadPoolExecutor 加速
    - 数据验证: check_data_length 确保数据完整性
    - 进度追踪: 详细的采集进度日志
    """

    BASE_URL = "https://fapi.binance.com"
    DATA_URL = "https://fapi.binance.com/futures/data"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = "~/.cryptoquant/data",
        rate_limit_delay: float = 0.1,
        # Qlib 风格的新参数
        max_collector_count: int = 3,       # 最大重试次数
        retry_delay: float = 2.0,           # 重试间隔
        max_workers: int = 4,               # 并行工作数
        check_data_length: int = 0,         # 最小数据条数验证 (0=不验证)
        # 缓存参数 (类似 Qlib 的 provider_uri 机制)
        use_cache: bool = True,             # 是否使用本地缓存
        cache_valid_days: int = 7,          # 缓存有效期(天)
    ):
        """
        初始化采集器

        Args:
            symbols: 交易对列表, 如 ['BTCUSDT', 'ETHUSDT']
            data_dir: 数据存储目录 (类似 Qlib 的 provider_uri)
            rate_limit_delay: API调用间隔(秒)
            max_collector_count: 最大重试次数 (Qlib 风格)
            retry_delay: 重试间隔(秒)
            max_workers: 并行采集的工作线程数
            check_data_length: 最小数据条数，低于此值视为采集失败
            use_cache: 是否使用本地缓存 (类似 Qlib 的 DiskDatasetCache)
            cache_valid_days: 缓存有效期(天)，超过此时间需重新下载
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # Qlib 风格参数
        self.max_collector_count = max_collector_count
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.check_data_length = check_data_length

        # 缓存参数
        self.use_cache = use_cache
        self.cache_valid_days = cache_valid_days

        # 统计信息
        self._stats = {
            "success": 0,
            "failed": 0,
            "retried": 0,
            "cache_hit": 0,
            "cache_miss": 0,
        }

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["klines", "funding", "oi", "ls_ratio", "taker", "orderbook"]:
            (self.data_dir / subdir).mkdir(exist_ok=True)

        logger.info(f"BinanceDataCollector initialized with {len(self.symbols)} symbols")
        logger.info(f"  max_retries={max_collector_count}, workers={max_workers}, min_length={check_data_length}")
        logger.info(f"  cache={'enabled' if use_cache else 'disabled'}, valid_days={cache_valid_days}")

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """
        发送请求 (带自动重试)

        Args:
            url: API URL
            params: 请求参数

        Returns:
            JSON 响应或 None
        """
        last_error = None

        for attempt in range(self.max_collector_count):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                time.sleep(self.rate_limit_delay)
                self._stats["success"] += 1
                return resp.json()

            except requests.RequestException as e:
                last_error = e
                self._stats["retried"] += 1

                if attempt < self.max_collector_count - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_collector_count}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Request failed after {self.max_collector_count} attempts: {url}")
                    self._stats["failed"] += 1

        return None

    def _validate_data(self, df: pd.DataFrame, data_type: str, symbol: str) -> bool:
        """
        验证数据完整性 (Qlib 的 check_data_length 模式)

        Args:
            df: 数据 DataFrame
            data_type: 数据类型
            symbol: 交易对

        Returns:
            True 如果数据有效
        """
        if df is None or df.empty:
            logger.warning(f"No data for {symbol} ({data_type})")
            return False

        if self.check_data_length > 0 and len(df) < self.check_data_length:
            logger.warning(
                f"Data length {len(df)} < {self.check_data_length} for {symbol} ({data_type})"
            )
            return False

        return True

    def get_stats(self) -> Dict[str, int]:
        """获取采集统计信息"""
        return self._stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self._stats = {"success": 0, "failed": 0, "retried": 0, "cache_hit": 0, "cache_miss": 0}

    # ==================== 缓存管理 (Qlib 风格) ====================

    def check_cache_valid(self, data_type: str, symbol: str) -> tuple:
        """
        检查缓存是否有效 (类似 Qlib 的 DiskDatasetCache 检查)

        Args:
            data_type: 数据类型 ('klines', 'funding', 'oi', 等)
            symbol: 交易对

        Returns:
            tuple: (is_valid, file_path, cache_age_days)
        """
        cache_path = self.data_dir / data_type / f"{symbol}.parquet"

        if not cache_path.exists():
            return (False, cache_path, None)

        # 检查缓存时间
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        cache_age = (datetime.now() - mtime).days

        is_valid = cache_age <= self.cache_valid_days
        return (is_valid, cache_path, cache_age)

    def load_symbol_cache(self, data_type: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载单个交易对的缓存数据

        Args:
            data_type: 数据类型
            symbol: 交易对

        Returns:
            DataFrame 或 None
        """
        is_valid, cache_path, cache_age = self.check_cache_valid(data_type, symbol)

        if is_valid:
            try:
                df = pd.read_parquet(cache_path)
                self._stats["cache_hit"] += 1
                logger.debug(f"Cache hit: {symbol}/{data_type} ({cache_age} days old)")
                return df
            except Exception as e:
                logger.warning(f"Cache read error for {symbol}/{data_type}: {e}")

        self._stats["cache_miss"] += 1
        return None

    def save_symbol_cache(self, data_type: str, symbol: str, df: pd.DataFrame):
        """
        保存单个交易对的缓存数据

        Args:
            data_type: 数据类型
            symbol: 交易对
            df: 数据
        """
        if df is None or df.empty:
            return

        cache_path = self.data_dir / data_type / f"{symbol}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.debug(f"Cache saved: {symbol}/{data_type} ({len(df)} rows)")

    def get_cache_info(self) -> Dict[str, Dict]:
        """
        获取缓存状态信息

        Returns:
            dict: 各数据类型的缓存信息
        """
        info = {}
        data_types = ["klines", "funding", "oi", "ls_ratio", "top_ls_ratio", "taker"]

        for data_type in data_types:
            type_dir = self.data_dir / data_type
            if not type_dir.exists():
                info[data_type] = {"files": 0, "symbols": []}
                continue

            files = list(type_dir.glob("*.parquet"))
            symbols = [f.stem for f in files]

            # 计算最新和最旧的缓存
            if files:
                mtimes = [datetime.fromtimestamp(f.stat().st_mtime) for f in files]
                info[data_type] = {
                    "files": len(files),
                    "symbols": symbols,
                    "oldest": min(mtimes),
                    "newest": max(mtimes),
                }
            else:
                info[data_type] = {"files": 0, "symbols": []}

        return info

    def clear_cache(self, data_type: str = None, symbol: str = None):
        """
        清除缓存

        Args:
            data_type: 数据类型 (None=所有类型)
            symbol: 交易对 (None=所有交易对)
        """
        data_types = [data_type] if data_type else ["klines", "funding", "oi", "ls_ratio", "top_ls_ratio", "taker"]

        for dt in data_types:
            type_dir = self.data_dir / dt
            if not type_dir.exists():
                continue

            if symbol:
                cache_file = type_dir / f"{symbol}.parquet"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cache cleared: {symbol}/{dt}")
            else:
                for f in type_dir.glob("*.parquet"):
                    f.unlink()
                logger.info(f"Cache cleared: all/{dt}")

    # ==================== K线数据 ====================
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 交易对
            interval: K线周期 (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: 开始时间戳(ms)
            end_time: 结束时间戳(ms)
            limit: 最大数量
        """
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])

        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df["symbol"] = symbol

        # 转换类型
        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)

        return df[["datetime", "symbol", "open", "high", "low", "close",
                   "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "trades"]]

    # ==================== 资金费率 ====================
    def fetch_funding_rate(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取资金费率历史"""
        url = f"{self.BASE_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        # 处理空字符串和无效值
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors='coerce')
        df["mark_price"] = pd.to_numeric(df.get("markPrice", None), errors='coerce')
        df["symbol"] = symbol

        # 移除无效行
        df = df.dropna(subset=["funding_rate"])

        return df[["datetime", "symbol", "funding_rate", "mark_price"]]

    # ==================== 持仓量 ====================
    def fetch_open_interest(self, symbol: str) -> pd.DataFrame:
        """获取当前持仓量"""
        url = f"{self.BASE_URL}/fapi/v1/openInterest"
        params = {"symbol": symbol}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        return pd.DataFrame([{
            "datetime": pd.Timestamp.now(),
            "symbol": symbol,
            "open_interest": float(data["openInterest"]),
        }])

    def fetch_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取持仓量历史"""
        url = f"{self.DATA_URL}/openInterestHist"
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors='coerce')
        df["open_interest_value"] = pd.to_numeric(df["sumOpenInterestValue"], errors='coerce')
        df["symbol"] = symbol

        # 移除无效行
        df = df.dropna(subset=["open_interest"])

        return df[["datetime", "symbol", "open_interest", "open_interest_value"]]

    # ==================== 多空比 ====================
    def fetch_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取全市场多空账户比"""
        url = f"{self.DATA_URL}/globalLongShortAccountRatio"
        params = {"symbol": symbol, "period": period, "limit": limit}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["ls_ratio"] = pd.to_numeric(df["longShortRatio"], errors='coerce')
        df["long_account"] = pd.to_numeric(df["longAccount"], errors='coerce')
        df["short_account"] = pd.to_numeric(df["shortAccount"], errors='coerce')
        df["symbol"] = symbol

        df = df.dropna(subset=["ls_ratio"])
        return df[["datetime", "symbol", "ls_ratio", "long_account", "short_account"]]

    def fetch_top_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取大户多空持仓比"""
        url = f"{self.DATA_URL}/topLongShortPositionRatio"
        params = {"symbol": symbol, "period": period, "limit": limit}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["top_ls_ratio"] = pd.to_numeric(df["longShortRatio"], errors='coerce')
        df["top_long_account"] = pd.to_numeric(df["longAccount"], errors='coerce')
        df["top_short_account"] = pd.to_numeric(df["shortAccount"], errors='coerce')
        df["symbol"] = symbol

        df = df.dropna(subset=["top_ls_ratio"])
        return df[["datetime", "symbol", "top_ls_ratio", "top_long_account", "top_short_account"]]

    # ==================== 主动买卖比 ====================
    def fetch_taker_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """获取主动买卖比"""
        url = f"{self.DATA_URL}/takerlongshortRatio"
        params = {"symbol": symbol, "period": period, "limit": limit}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["taker_buy_sell_ratio"] = pd.to_numeric(df["buySellRatio"], errors='coerce')
        df["taker_buy_vol"] = pd.to_numeric(df["buyVol"], errors='coerce')
        df["taker_sell_vol"] = pd.to_numeric(df["sellVol"], errors='coerce')
        df["symbol"] = symbol

        df = df.dropna(subset=["taker_buy_sell_ratio"])
        return df[["datetime", "symbol", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]]

    # ==================== 订单簿 ====================
    def fetch_orderbook(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """获取订单簿快照"""
        url = f"{self.BASE_URL}/fapi/v1/depth"
        params = {"symbol": symbol, "limit": limit}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        row = {
            "datetime": pd.Timestamp.now(),
            "symbol": symbol,
        }

        # 添加买卖档位
        for i, (price, qty) in enumerate(data.get("bids", [])[:20]):
            row[f"bid{i+1}"] = float(price)
            row[f"bsize{i+1}"] = float(qty)

        for i, (price, qty) in enumerate(data.get("asks", [])[:20]):
            row[f"ask{i+1}"] = float(price)
            row[f"asize{i+1}"] = float(qty)

        return pd.DataFrame([row])

    # ==================== 批量采集 ====================
    def _collect_symbol_data(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        interval: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        采集单个交易对的所有数据 (用于并行执行)

        Args:
            symbol: 交易对
            start_ts: 开始时间戳
            end_ts: 结束时间戳
            interval: K线周期

        Returns:
            该交易对的所有数据
        """
        result = {}

        # K线
        klines = self.fetch_klines(symbol, interval, start_ts, end_ts)
        if self._validate_data(klines, "klines", symbol):
            result["klines"] = klines

        # 资金费率
        funding = self.fetch_funding_rate(symbol, start_ts, end_ts)
        if self._validate_data(funding, "funding", symbol):
            result["funding"] = funding

        # 持仓量历史
        oi = self.fetch_open_interest_history(symbol, interval)
        if self._validate_data(oi, "oi", symbol):
            result["oi"] = oi

        # 多空比
        ls = self.fetch_long_short_ratio(symbol, interval)
        if self._validate_data(ls, "ls_ratio", symbol):
            result["ls_ratio"] = ls

        # 大户多空比
        top_ls = self.fetch_top_long_short_ratio(symbol, interval)
        if self._validate_data(top_ls, "top_ls_ratio", symbol):
            result["top_ls_ratio"] = top_ls

        # 主动买卖比
        taker = self.fetch_taker_long_short_ratio(symbol, interval)
        if self._validate_data(taker, "taker", symbol):
            result["taker"] = taker

        return result

    def collect_all(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        parallel: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        采集所有数据 (支持并行)

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            interval: K线周期
            parallel: 是否并行采集

        Returns:
            包含所有数据的字典
        """
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        self.reset_stats()
        all_data = {
            "klines": [],
            "funding": [],
            "oi": [],
            "ls_ratio": [],
            "top_ls_ratio": [],
            "taker": [],
        }

        logger.info(f"Starting data collection for {len(self.symbols)} symbols...")
        logger.info(f"  Period: {start_date} ~ {end_date}")
        logger.info(f"  Parallel: {parallel} (workers: {self.max_workers})")

        if parallel and self.max_workers > 1:
            # 并行采集 (Qlib 风格)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._collect_symbol_data, symbol, start_ts, end_ts, interval
                    ): symbol
                    for symbol in self.symbols
                }

                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        for key, df in result.items():
                            if df is not None and not df.empty:
                                all_data[key].append(df)
                        logger.info(f"✅ {symbol}: collected {len(result)} data types")
                    except Exception as e:
                        logger.error(f"❌ {symbol}: collection failed - {e}")
        else:
            # 串行采集
            for i, symbol in enumerate(self.symbols):
                logger.info(f"[{i+1}/{len(self.symbols)}] Collecting {symbol}...")
                result = self._collect_symbol_data(symbol, start_ts, end_ts, interval)
                for key, df in result.items():
                    if df is not None and not df.empty:
                        all_data[key].append(df)

        # 合并数据
        result = {}
        for key, dfs in all_data.items():
            if dfs:
                result[key] = pd.concat(dfs, ignore_index=True)
            else:
                result[key] = pd.DataFrame()

        # 打印统计
        stats = self.get_stats()
        logger.info(f"Collection completed: success={stats['success']}, failed={stats['failed']}, retried={stats['retried']}")

        return result

    def save_data(self, data: Dict[str, pd.DataFrame], date_str: str = None):
        """保存数据到Parquet文件"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        for key, df in data.items():
            if not df.empty:
                path = self.data_dir / key / f"{date_str}.parquet"
                df.to_parquet(path, index=False)
                logger.info(f"Saved {len(df)} rows to {path}")

    def load_data(
        self,
        data_type: str,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """加载历史数据"""
        data_path = self.data_dir / data_type
        if not data_path.exists():
            return pd.DataFrame()

        files = sorted(data_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            date_str = f.stem
            if start_date and date_str < start_date.replace("-", ""):
                continue
            if end_date and date_str > end_date.replace("-", ""):
                continue
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).drop_duplicates()

    # ==================== 缓存优先采集 (Qlib 风格) ====================

    def collect_with_cache(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        parallel: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        缓存优先的数据采集 (类似 Qlib 的 provider_uri 机制)

        工作流程:
        1. 检查本地缓存是否存在且有效
        2. 从缓存加载有效数据
        3. 只下载缺失/过期的数据
        4. 保存新下载的数据到缓存

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            interval: K线周期
            parallel: 是否并行采集
            force_refresh: 强制刷新 (忽略缓存)

        Returns:
            dict: {data_type: {symbol: DataFrame}}
        """
        self.reset_stats()
        data_types = ["klines", "funding", "oi", "ls_ratio", "top_ls_ratio", "taker"]

        # 结果存储
        result = {dt: {} for dt in data_types}

        # 需要下载的交易对
        need_download = {dt: [] for dt in data_types}

        logger.info(f"Collect with cache: {len(self.symbols)} symbols")
        logger.info(f"  Period: {start_date} ~ {end_date}")
        logger.info(f"  Cache: {'disabled (force_refresh)' if force_refresh else 'enabled'}")

        # Step 1: 检查缓存
        if self.use_cache and not force_refresh:
            logger.info("Step 1: Checking cache...")
            for dt in data_types:
                for symbol in self.symbols:
                    cached_data = self.load_symbol_cache(dt, symbol)
                    if cached_data is not None:
                        result[dt][symbol] = cached_data
                    else:
                        need_download[dt].append(symbol)

            # 统计缓存命中
            total_cached = sum(len(result[dt]) for dt in data_types)
            total_need = sum(len(need_download[dt]) for dt in data_types)
            logger.info(f"  Cache hit: {total_cached}, Cache miss: {total_need}")

            if total_need == 0:
                logger.info("All data loaded from cache!")
                return result
        else:
            # 禁用缓存，全部需要下载
            for dt in data_types:
                need_download[dt] = self.symbols.copy()

        # Step 2: 下载缺失的数据
        logger.info("Step 2: Downloading missing data...")

        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        # 找出需要下载任何数据的交易对
        symbols_to_download = set()
        for dt in data_types:
            symbols_to_download.update(need_download[dt])

        if not symbols_to_download:
            logger.info("No data to download!")
            return result

        logger.info(f"  Symbols to download: {list(symbols_to_download)}")

        # 采集数据
        for symbol in symbols_to_download:
            logger.info(f"  Collecting {symbol}...")

            # 只采集需要的数据类型
            if symbol in need_download["klines"]:
                klines = self.fetch_klines(symbol, interval, start_ts, end_ts)
                if self._validate_data(klines, "klines", symbol):
                    result["klines"][symbol] = klines
                    self.save_symbol_cache("klines", symbol, klines)

            if symbol in need_download["funding"]:
                funding = self.fetch_funding_rate(symbol, start_ts, end_ts)
                if self._validate_data(funding, "funding", symbol):
                    result["funding"][symbol] = funding
                    self.save_symbol_cache("funding", symbol, funding)

            if symbol in need_download["oi"]:
                oi = self.fetch_open_interest_history(symbol, interval)
                if self._validate_data(oi, "oi", symbol):
                    result["oi"][symbol] = oi
                    self.save_symbol_cache("oi", symbol, oi)

            if symbol in need_download["ls_ratio"]:
                ls = self.fetch_long_short_ratio(symbol, interval)
                if self._validate_data(ls, "ls_ratio", symbol):
                    result["ls_ratio"][symbol] = ls
                    self.save_symbol_cache("ls_ratio", symbol, ls)

            if symbol in need_download["top_ls_ratio"]:
                top_ls = self.fetch_top_long_short_ratio(symbol, interval)
                if self._validate_data(top_ls, "top_ls_ratio", symbol):
                    result["top_ls_ratio"][symbol] = top_ls
                    self.save_symbol_cache("top_ls_ratio", symbol, top_ls)

            if symbol in need_download["taker"]:
                taker = self.fetch_taker_long_short_ratio(symbol, interval)
                if self._validate_data(taker, "taker", symbol):
                    result["taker"][symbol] = taker
                    self.save_symbol_cache("taker", symbol, taker)

        # Step 3: 打印统计
        stats = self.get_stats()
        logger.info(f"Collection completed:")
        logger.info(f"  API: success={stats['success']}, failed={stats['failed']}, retried={stats['retried']}")
        logger.info(f"  Cache: hit={stats['cache_hit']}, miss={stats['cache_miss']}")

        return result


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置日志
    logger.add("collector.log", rotation="10 MB")

    # 初始化采集器 (启用缓存)
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    collector = BinanceDataCollector(
        symbols=symbols,
        use_cache=True,          # 启用本地缓存 (类似 Qlib provider_uri)
        cache_valid_days=7,      # 缓存7天有效
    )

    # 采集最近30天数据 (缓存优先)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 方式1: 缓存优先采集 (推荐)
    # 首次运行会下载，后续运行直接从缓存加载
    data = collector.collect_with_cache(start_date, end_date, interval="1h")

    # 打印统计
    print("\n=== 数据统计 ===")
    for data_type, symbol_data in data.items():
        total_rows = sum(len(df) for df in symbol_data.values())
        print(f"{data_type}: {len(symbol_data)} symbols, {total_rows} rows")

    # 查看缓存状态
    print("\n=== 缓存状态 ===")
    cache_info = collector.get_cache_info()
    for dt, info in cache_info.items():
        print(f"{dt}: {info['files']} files")

    # 方式2: 强制刷新 (忽略缓存)
    # data = collector.collect_with_cache(start_date, end_date, force_refresh=True)

    # 方式3: 传统方式 (不使用缓存)
    # data = collector.collect_all(start_date, end_date, interval="1h")
    # collector.save_data(data)
