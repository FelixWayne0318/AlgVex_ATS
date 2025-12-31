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
    ):
        """
        初始化采集器

        Args:
            symbols: 交易对列表, 如 ['BTCUSDT', 'ETHUSDT']
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
            max_collector_count: 最大重试次数 (Qlib 风格)
            retry_delay: 重试间隔(秒)
            max_workers: 并行采集的工作线程数
            check_data_length: 最小数据条数，低于此值视为采集失败
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # Qlib 风格参数
        self.max_collector_count = max_collector_count
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.check_data_length = check_data_length

        # 统计信息
        self._stats = {
            "success": 0,
            "failed": 0,
            "retried": 0,
        }

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["klines", "funding", "oi", "ls_ratio", "taker", "orderbook"]:
            (self.data_dir / subdir).mkdir(exist_ok=True)

        logger.info(f"BinanceDataCollector initialized with {len(self.symbols)} symbols")
        logger.info(f"  max_retries={max_collector_count}, workers={max_workers}, min_length={check_data_length}")

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
        self._stats = {"success": 0, "failed": 0, "retried": 0}

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


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 配置日志
    logger.add("collector.log", rotation="10 MB")

    # 初始化采集器
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    collector = BinanceDataCollector(symbols)

    # 采集最近30天数据
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    data = collector.collect_all(start_date, end_date, interval="1h")
    collector.save_data(data)

    # 打印统计
    for key, df in data.items():
        print(f"{key}: {len(df)} rows")
