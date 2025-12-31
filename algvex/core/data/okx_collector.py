"""
OKX 数据采集器 - 永续合约数据

用于多交易所Basis计算和套利机会检测 (Step 11)

数据类型:
1. K线数据 (OHLCV)
2. 资金费率 (Funding Rate)
3. 持仓量 (Open Interest)
4. 订单簿快照 (Order Book)

API文档: https://www.okx.com/docs-v5/en/
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


class OKXDataCollector:
    """OKX 永续合约数据采集器 (免费API)"""

    BASE_URL = "https://www.okx.com"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = "~/.cryptoquant/data/okx",
        rate_limit_delay: float = 0.1,
    ):
        """
        初始化采集器

        Args:
            symbols: 交易对列表, 如 ['BTC-USDT-SWAP', 'ETH-USDT-SWAP']
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        # OKX使用不同的符号格式
        self.symbols = symbols or ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "klines").mkdir(exist_ok=True)
        (self.data_dir / "funding").mkdir(exist_ok=True)
        (self.data_dir / "oi").mkdir(exist_ok=True)
        (self.data_dir / "orderbook").mkdir(exist_ok=True)

        logger.info(f"OKXDataCollector initialized with {len(self.symbols)} symbols")

    @staticmethod
    def to_okx_symbol(symbol: str) -> str:
        """转换为OKX符号格式 (BTCUSDT -> BTC-USDT-SWAP)"""
        if "-SWAP" in symbol:
            return symbol
        # 假设输入是BTCUSDT格式
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USDT-SWAP"
        return symbol

    @staticmethod
    def from_okx_symbol(symbol: str) -> str:
        """从OKX符号格式转换 (BTC-USDT-SWAP -> BTCUSDT)"""
        if "-SWAP" in symbol:
            parts = symbol.replace("-SWAP", "").split("-")
            return "".join(parts)
        return symbol

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """发送请求并处理错误"""
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)
            data = resp.json()
            if data.get("code") == "0":
                return data.get("data", [])
            else:
                logger.error(f"OKX API error: {data.get('msg')}")
                return None
        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

    # ==================== K线数据 ====================
    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1H",  # 1m,3m,5m,15m,30m,1H,2H,4H,6H,12H,1D,1W,1M
        after: Optional[int] = None,
        before: Optional[int] = None,
        limit: int = 300,
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 交易对 (如 BTC-USDT-SWAP 或 BTCUSDT)
            interval: K线周期
            after: 请求此时间戳之后的数据(ms)
            before: 请求此时间戳之前的数据(ms)
            limit: 最大数量 (max 300)
        """
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/market/candles"
        params = {
            "instId": okx_symbol,
            "bar": interval,
            "limit": str(min(limit, 300)),
        }
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        # OKX返回: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "vol_ccy", "vol_quote", "confirm"
        ])

        df["datetime"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
        df["symbol"] = self.from_okx_symbol(okx_symbol)
        df["exchange"] = "okx"

        for col in ["open", "high", "low", "close", "volume", "vol_quote"]:
            df[col] = df[col].astype(float)

        # 按时间升序排序
        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "open", "high", "low", "close", "volume", "vol_quote"]]

    # ==================== 资金费率 ====================
    def fetch_funding_rate(
        self,
        symbol: str,
        after: Optional[int] = None,
        before: Optional[int] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """获取资金费率历史"""
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/public/funding-rate-history"
        params = {
            "instId": okx_symbol,
            "limit": str(min(limit, 100)),
        }
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["symbol"] = self.from_okx_symbol(okx_symbol)
        df["exchange"] = "okx"

        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "funding_rate"]]

    # ==================== 持仓量 ====================
    def fetch_open_interest(
        self,
        symbol: str,
    ) -> Dict:
        """获取当前持仓量"""
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/public/open-interest"
        params = {"instId": okx_symbol}

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        item = data[0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
            "open_interest": float(item.get("oi", 0)),
            "open_interest_ccy": float(item.get("oiCcy", 0)),
        }

    def fetch_open_interest_history(
        self,
        symbol: str,
        period: str = "1H",  # 5m, 1H, 1D
        after: Optional[int] = None,
        before: Optional[int] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """获取持仓量历史"""
        okx_symbol = self.to_okx_symbol(symbol)
        # 提取币种
        ccy = okx_symbol.split("-")[0]

        url = f"{self.BASE_URL}/api/v5/rubik/stat/contracts/open-interest-history"
        params = {
            "instType": "SWAP",
            "ccy": ccy,
            "period": period,
            "limit": str(min(limit, 100)),
        }
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        # 返回格式: [ts, oi, oiCcy]
        df = pd.DataFrame(data, columns=["timestamp", "open_interest", "open_interest_ccy"])
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df["open_interest"] = df["open_interest"].astype(float)
        df["symbol"] = self.from_okx_symbol(okx_symbol)
        df["exchange"] = "okx"

        df = df.sort_values("datetime").reset_index(drop=True)

        return df[["datetime", "symbol", "exchange", "open_interest"]]

    # ==================== 订单簿 ====================
    def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 20,
    ) -> Dict:
        """获取订单簿快照"""
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/market/books"
        params = {
            "instId": okx_symbol,
            "sz": str(min(depth, 400)),
        }

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        book = data[0]
        bids = book.get("bids", [])
        asks = book.get("asks", [])

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
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
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
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/market/ticker"
        params = {"instId": okx_symbol}

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        ticker = data[0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
            "last_price": float(ticker.get("last", 0)),
            "bid_price": float(ticker.get("bidPx", 0)),
            "ask_price": float(ticker.get("askPx", 0)),
            "volume_24h": float(ticker.get("vol24h", 0)),
            "volume_ccy_24h": float(ticker.get("volCcy24h", 0)),
            "open_24h": float(ticker.get("open24h", 0)),
            "high_24h": float(ticker.get("high24h", 0)),
            "low_24h": float(ticker.get("low24h", 0)),
        }

    # ==================== 标记价格和指数价格 ====================
    def fetch_mark_price(self, symbol: str) -> Dict:
        """获取标记价格"""
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/public/mark-price"
        params = {
            "instType": "SWAP",
            "instId": okx_symbol,
        }

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        item = data[0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
            "mark_price": float(item.get("markPx", 0)),
        }

    def fetch_index_price(self, symbol: str) -> Dict:
        """获取指数价格"""
        okx_symbol = self.to_okx_symbol(symbol)
        # 从swap符号提取指数符号
        index_id = okx_symbol.replace("-SWAP", "")

        url = f"{self.BASE_URL}/api/v5/market/index-tickers"
        params = {"instId": index_id}

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        item = data[0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
            "index_price": float(item.get("idxPx", 0)),
        }

    # ==================== 当前资金费率 ====================
    def fetch_current_funding_rate(self, symbol: str) -> Dict:
        """获取当前资金费率"""
        okx_symbol = self.to_okx_symbol(symbol)
        url = f"{self.BASE_URL}/api/v5/public/funding-rate"
        params = {"instId": okx_symbol}

        data = self._request(url, params)
        if not data or len(data) == 0:
            return {}

        item = data[0]
        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(okx_symbol),
            "exchange": "okx",
            "funding_rate": float(item.get("fundingRate", 0)),
            "next_funding_rate": float(item.get("nextFundingRate", 0)) if item.get("nextFundingRate") else None,
            "funding_time": int(item.get("fundingTime", 0)),
            "next_funding_time": int(item.get("nextFundingTime", 0)) if item.get("nextFundingTime") else None,
        }

    # ==================== 计算Basis ====================
    def calculate_basis(self, symbol: str) -> Dict:
        """计算永续-现货基差"""
        mark = self.fetch_mark_price(symbol)
        index = self.fetch_index_price(symbol)
        funding = self.fetch_current_funding_rate(symbol)

        if not mark or not index:
            return {}

        mark_price = mark.get("mark_price", 0)
        index_price = index.get("index_price", 0)

        if index_price == 0:
            return {}

        basis = mark_price - index_price
        basis_pct = basis / index_price * 100

        return {
            "timestamp": datetime.utcnow(),
            "symbol": self.from_okx_symbol(self.to_okx_symbol(symbol)),
            "exchange": "okx",
            "mark_price": mark_price,
            "index_price": index_price,
            "basis": basis,
            "basis_pct": basis_pct,
            "funding_rate": funding.get("funding_rate", 0) if funding else 0,
        }

    # ==================== 批量获取 ====================
    def fetch_all_swap_tickers(self) -> pd.DataFrame:
        """获取所有永续合约行情"""
        url = f"{self.BASE_URL}/api/v5/market/tickers"
        params = {"instType": "SWAP"}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        records = []
        for ticker in data:
            inst_id = ticker.get("instId", "")
            if "USDT" in inst_id:  # 只要USDT结算的
                records.append({
                    "timestamp": datetime.utcnow(),
                    "symbol": self.from_okx_symbol(inst_id),
                    "exchange": "okx",
                    "last_price": float(ticker.get("last", 0)),
                    "volume_24h": float(ticker.get("vol24h", 0)),
                    "volume_ccy_24h": float(ticker.get("volCcy24h", 0)),
                })

        return pd.DataFrame(records)
