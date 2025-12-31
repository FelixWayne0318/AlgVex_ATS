"""
Yahoo Finance 宏观数据采集器

数据类型:
1. 美元指数 (DXY)
2. 国债收益率 (US10Y, US2Y)
3. 股指 (SPX, VIX)
4. 商品 (Gold)

使用示例:
    collector = YahooMacroCollector()
    macro_data = collector.fetch_all()
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class YahooMacroCollector:
    """Yahoo Finance 宏观数据采集器"""

    # Yahoo Finance符号映射
    SYMBOLS = {
        "dxy": "DX-Y.NYB",      # 美元指数
        "us10y": "^TNX",        # 10年期国债收益率
        "us2y": "^IRX",         # 2年期国债收益率 (近似)
        "spx": "^GSPC",         # 标普500
        "vix": "^VIX",          # VIX波动率指数
        "gold": "GC=F",         # 黄金期货
        "btc": "BTC-USD",       # BTC价格
    }

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(
        self,
        symbols: List[str] = None,
        data_dir: str = "~/.cryptoquant/data/macro",
        rate_limit_delay: float = 0.5,
    ):
        """
        初始化采集器

        Args:
            symbols: 要采集的符号列表
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.symbols = symbols or list(self.SYMBOLS.keys())
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"YahooMacroCollector initialized with {len(self.symbols)} symbols")

    def _request(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Optional[Dict]:
        """发送API请求"""
        yahoo_symbol = self.SYMBOLS.get(symbol, symbol)
        url = f"{self.BASE_URL}{yahoo_symbol}"

        params = {
            "period1": int((datetime.now() - timedelta(days=365*2)).timestamp()),
            "period2": int(datetime.now().timestamp()),
            "interval": interval,
            "includePrePost": "false",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)

            data = resp.json()
            if "chart" in data and "result" in data["chart"]:
                return data["chart"]["result"][0]
            return None

        except requests.RequestException as e:
            logger.error(f"Request failed for {symbol}: {e}")
            return None

    def fetch_symbol(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        获取单个符号的历史数据

        Args:
            symbol: 符号名称 (dxy, spx, etc.)
            days: 历史天数

        Returns:
            包含OHLCV数据的DataFrame
        """
        data = self._request(symbol)

        if not data:
            return pd.DataFrame()

        try:
            timestamps = data.get("timestamp", [])
            indicators = data.get("indicators", {})
            quote = indicators.get("quote", [{}])[0]

            if not timestamps:
                return pd.DataFrame()

            df = pd.DataFrame({
                "datetime": pd.to_datetime(timestamps, unit="s"),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            })

            df["symbol"] = symbol
            df = df.dropna(subset=["close"])

            return df

        except Exception as e:
            logger.error(f"Error parsing data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all(self, days: int = 365) -> pd.DataFrame:
        """
        获取所有宏观指标

        Returns:
            宽表格式的DataFrame (每列一个指标)
        """
        all_data = {}

        for symbol in self.symbols:
            logger.info(f"Fetching {symbol}...")
            df = self.fetch_symbol(symbol, days)

            if not df.empty:
                # 使用datetime作为索引，close作为值
                series = df.set_index("datetime")["close"]
                all_data[symbol] = series

        if not all_data:
            return pd.DataFrame()

        # 合并为宽表
        result = pd.DataFrame(all_data)
        result = result.sort_index()

        # 前向填充缺失值 (交易日差异)
        result = result.ffill()

        return result

    def fetch_latest(self) -> Dict[str, float]:
        """获取所有指标的最新值"""
        df = self.fetch_all(days=7)

        if df.empty:
            return {}

        return df.iloc[-1].to_dict()

    def save_data(self, df: pd.DataFrame, date_str: str = None):
        """保存数据到Parquet文件"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        if not df.empty:
            path = self.data_dir / f"macro_{date_str}.parquet"
            df.to_parquet(path)
            logger.info(f"Saved {len(df)} rows to {path}")

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """加载历史数据"""
        files = sorted(self.data_dir.glob("macro_*.parquet"))

        if not files:
            return pd.DataFrame()

        # 加载最新的文件
        df = pd.read_parquet(files[-1])

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        return df


# 使用示例
if __name__ == "__main__":
    collector = YahooMacroCollector()

    # 获取所有宏观数据
    macro_data = collector.fetch_all(days=30)
    print("Macro data shape:", macro_data.shape)
    print(macro_data.tail())

    # 获取最新值
    latest = collector.fetch_latest()
    print("\nLatest values:")
    for k, v in latest.items():
        print(f"  {k}: {v:.2f}")

    # 保存数据
    collector.save_data(macro_data)
