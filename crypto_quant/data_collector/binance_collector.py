"""
币安数据采集器 - 采集永续合约所有免费数据

数据类型:
1. K线数据 (OHLCV)
2. 资金费率 (Funding Rate)
3. 持仓量 (Open Interest)
4. 多空比 (Long/Short Ratio)
5. 大户持仓比 (Top Trader Ratio)
6. 主动买卖比 (Taker Buy/Sell)
7. 订单簿快照 (Order Book)
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests
from loguru import logger


class BinanceDataCollector:
    """币安永续合约数据采集器 (全部免费API)"""

    BASE_URL = "https://fapi.binance.com"
    DATA_URL = "https://fapi.binance.com/futures/data"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = "~/.cryptoquant/data",
        rate_limit_delay: float = 0.1,
    ):
        """
        初始化采集器

        Args:
            symbols: 交易对列表, 如 ['BTCUSDT', 'ETHUSDT']
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.symbols = symbols or ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "klines").mkdir(exist_ok=True)
        (self.data_dir / "funding").mkdir(exist_ok=True)
        (self.data_dir / "oi").mkdir(exist_ok=True)
        (self.data_dir / "ls_ratio").mkdir(exist_ok=True)
        (self.data_dir / "taker").mkdir(exist_ok=True)
        (self.data_dir / "orderbook").mkdir(exist_ok=True)

        logger.info(f"BinanceDataCollector initialized with {len(self.symbols)} symbols")

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """发送请求并处理错误"""
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

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
                   "volume", "quote_volume", "taker_buy_volume", "trades"]]

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
        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["mark_price"] = df["markPrice"].astype(float) if "markPrice" in df else None
        df["symbol"] = symbol

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
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
        df["symbol"] = symbol

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
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["ls_ratio"] = df["longShortRatio"].astype(float)
        df["long_account"] = df["longAccount"].astype(float)
        df["short_account"] = df["shortAccount"].astype(float)
        df["symbol"] = symbol

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
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["top_ls_ratio"] = df["longShortRatio"].astype(float)
        df["top_long_account"] = df["longAccount"].astype(float)
        df["top_short_account"] = df["shortAccount"].astype(float)
        df["symbol"] = symbol

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
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["taker_buy_sell_ratio"] = df["buySellRatio"].astype(float)
        df["taker_buy_vol"] = df["buyVol"].astype(float)
        df["taker_sell_vol"] = df["sellVol"].astype(float)
        df["symbol"] = symbol

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
    def collect_all(
        self,
        start_date: str,
        end_date: str,
        interval: str = "1h",
    ) -> Dict[str, pd.DataFrame]:
        """
        采集所有数据

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            interval: K线周期

        Returns:
            包含所有数据的字典
        """
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        all_data = {
            "klines": [],
            "funding": [],
            "oi": [],
            "ls_ratio": [],
            "top_ls_ratio": [],
            "taker": [],
        }

        for symbol in self.symbols:
            logger.info(f"Collecting data for {symbol}...")

            # K线
            klines = self.fetch_klines(symbol, interval, start_ts, end_ts)
            if not klines.empty:
                all_data["klines"].append(klines)

            # 资金费率
            funding = self.fetch_funding_rate(symbol, start_ts, end_ts)
            if not funding.empty:
                all_data["funding"].append(funding)

            # 持仓量历史
            oi = self.fetch_open_interest_history(symbol, interval, start_ts, end_ts)
            if not oi.empty:
                all_data["oi"].append(oi)

            # 多空比
            ls = self.fetch_long_short_ratio(symbol, interval)
            if not ls.empty:
                all_data["ls_ratio"].append(ls)

            # 大户多空比
            top_ls = self.fetch_top_long_short_ratio(symbol, interval)
            if not top_ls.empty:
                all_data["top_ls_ratio"].append(top_ls)

            # 主动买卖比
            taker = self.fetch_taker_long_short_ratio(symbol, interval)
            if not taker.empty:
                all_data["taker"].append(taker)

        # 合并数据
        result = {}
        for key, dfs in all_data.items():
            if dfs:
                result[key] = pd.concat(dfs, ignore_index=True)
            else:
                result[key] = pd.DataFrame()

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
