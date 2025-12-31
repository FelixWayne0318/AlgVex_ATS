"""
Bybit 数据采集器 - 永续合约数据

用于多交易所Basis计算和套利机会检测 (Step 11)

数据类型:
1. K线数据 (OHLCV)
2. 资金费率 (Funding Rate)
3. 持仓量 (Open Interest)
4. 订单簿快照 (Order Book)

API文档: https://bybit-exchange.github.io/docs/v5/intro
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BybitDataCollector:
    """Bybit 永续合约数据采集器 (免费API)"""

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = "~/.cryptoquant/data/bybit",
        rate_limit_delay: float = 0.1,
    ):
        """
        初始化采集器

        Args:
            symbols: 交易对列表, 如 ['BTCUSDT', 'ETHUSDT']
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "klines").mkdir(exist_ok=True)
        (self.data_dir / "funding").mkdir(exist_ok=True)
        (self.data_dir / "oi").mkdir(exist_ok=True)
        (self.data_dir / "orderbook").mkdir(exist_ok=True)

        logger.info(f"BybitDataCollector initialized with {len(self.symbols)} symbols")

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """发送请求并处理错误"""
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)
            data = resp.json()
            if data.get("retCode") == 0:
                return data.get("result", {})
            else:
                logger.error(f"Bybit API error: {data.get('retMsg')}")
                return None
        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

    # ==================== K线数据 ====================
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",  # 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 交易对 (如 BTCUSDT)
            interval: K线周期 (分钟: 1,3,5,15,30,60,120,240,360,720; 或 D,W,M)
            start_time: 开始时间戳(ms)
            end_time: 结束时间戳(ms)
            limit: 最大数量 (max 1000)
        """
        url = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        result = self._request(url, params)
        if not result or "list" not in result:
            return pd.DataFrame()

        # Bybit返回: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        data = result["list"]
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume", "turnover"
        ])

        df["datetime"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
        df["symbol"] = symbol
        df["exchange"] = "bybit"

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        # 按时间升序排序
        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "open", "high", "low", "close", "volume", "turnover"]]

    # ==================== 资金费率 ====================
    def fetch_funding_rate(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """获取资金费率历史"""
        url = f"{self.BASE_URL}/v5/market/funding/history"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 200),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        result = self._request(url, params)
        if not result or "list" not in result:
            return pd.DataFrame()

        data = result["list"]
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["fundingRateTimestamp"].astype(int), unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["symbol"] = symbol
        df["exchange"] = "bybit"

        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "funding_rate"]]

    # ==================== 持仓量 ====================
    def fetch_open_interest(
        self,
        symbol: str,
        interval: str = "1h",  # 5min, 15min, 30min, 1h, 4h, 1d
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """获取持仓量历史"""
        url = f"{self.BASE_URL}/v5/market/open-interest"
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval,
            "limit": min(limit, 200),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        result = self._request(url, params)
        if not result or "list" not in result:
            return pd.DataFrame()

        data = result["list"]
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df["open_interest"] = df["openInterest"].astype(float)
        df["symbol"] = symbol
        df["exchange"] = "bybit"

        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "open_interest"]]

    # ==================== 订单簿 ====================
    def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 50,
    ) -> Dict:
        """获取订单簿快照"""
        url = f"{self.BASE_URL}/v5/market/orderbook"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 200),
        }

        result = self._request(url, params)
        if not result:
            return {}

        bids = result.get("b", [])
        asks = result.get("a", [])

        # 计算基础指标
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = (best_ask - best_bid) / mid_price * 10000 if mid_price else 0  # bps

        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

        return {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "exchange": "bybit",
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance": imbalance,
            "bids": bids[:20],
            "asks": asks[:20],
        }

    # ==================== Ticker (当前价格) ====================
    def fetch_ticker(self, symbol: str) -> Dict:
        """获取当前行情"""
        url = f"{self.BASE_URL}/v5/market/tickers"
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        result = self._request(url, params)
        if not result or "list" not in result or not result["list"]:
            return {}

        ticker = result["list"][0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "exchange": "bybit",
            "last_price": float(ticker.get("lastPrice", 0)),
            "index_price": float(ticker.get("indexPrice", 0)),
            "mark_price": float(ticker.get("markPrice", 0)),
            "funding_rate": float(ticker.get("fundingRate", 0)),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "turnover_24h": float(ticker.get("turnover24h", 0)),
            "open_interest": float(ticker.get("openInterest", 0)),
        }

    # ==================== 批量获取 ====================
    def fetch_all_tickers(self) -> pd.DataFrame:
        """获取所有交易对的行情"""
        url = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": "linear"}

        result = self._request(url, params)
        if not result or "list" not in result:
            return pd.DataFrame()

        data = []
        for ticker in result["list"]:
            data.append({
                "timestamp": datetime.utcnow(),
                "symbol": ticker.get("symbol"),
                "exchange": "bybit",
                "last_price": float(ticker.get("lastPrice", 0)),
                "index_price": float(ticker.get("indexPrice", 0)),
                "mark_price": float(ticker.get("markPrice", 0)),
                "funding_rate": float(ticker.get("fundingRate", 0)),
                "volume_24h": float(ticker.get("volume24h", 0)),
                "open_interest": float(ticker.get("openInterest", 0)),
            })

        return pd.DataFrame(data)

    # ==================== 计算Basis ====================
    def calculate_basis(self, symbol: str) -> Dict:
        """计算永续-现货基差"""
        ticker = self.fetch_ticker(symbol)
        if not ticker:
            return {}

        mark_price = ticker.get("mark_price", 0)
        index_price = ticker.get("index_price", 0)

        if index_price == 0:
            return {}

        basis = mark_price - index_price
        basis_pct = basis / index_price * 100

        return {
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "exchange": "bybit",
            "mark_price": mark_price,
            "index_price": index_price,
            "basis": basis,
            "basis_pct": basis_pct,
            "funding_rate": ticker.get("funding_rate", 0),
        }
