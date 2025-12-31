"""
情绪数据采集器 - 免费API

数据类型:
1. 恐惧贪婪指数 (Alternative.me)
2. 稳定币供应量 (DefiLlama)
3. TVL数据 (DefiLlama)
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from loguru import logger


class SentimentDataCollector:
    """情绪数据采集器 (全部免费API)"""

    def __init__(self, data_dir: str = "~/.cryptoquant/data"):
        self.data_dir = Path(data_dir).expanduser()
        (self.data_dir / "sentiment").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "stablecoin").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "defi").mkdir(parents=True, exist_ok=True)

    def _request(self, url: str, params: dict = None) -> Optional[dict]:
        """发送请求"""
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(0.5)
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

    # ==================== 恐惧贪婪指数 ====================
    def fetch_fear_greed_index(self, limit: int = 365) -> pd.DataFrame:
        """
        获取恐惧贪婪指数

        指数范围: 0-100
        - 0-25: 极度恐惧 (买入信号)
        - 25-45: 恐惧
        - 45-55: 中性
        - 55-75: 贪婪
        - 75-100: 极度贪婪 (卖出信号)
        """
        url = "https://api.alternative.me/fng/"
        params = {"limit": limit, "format": "json"}

        data = self._request(url, params)
        if not data or "data" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["fear_greed_index"] = df["value"].astype(int)
        df["fear_greed_class"] = df["value_classification"]

        return df[["datetime", "fear_greed_index", "fear_greed_class"]].sort_values("datetime")

    # ==================== 稳定币数据 ====================
    def fetch_stablecoin_supply(self) -> pd.DataFrame:
        """获取主要稳定币供应量"""
        url = "https://stablecoins.llama.fi/stablecoins"
        params = {"includePrices": "true"}

        data = self._request(url, params)
        if not data or "peggedAssets" not in data:
            return pd.DataFrame()

        rows = []
        for coin in data["peggedAssets"]:
            if coin.get("symbol") in ["USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD"]:
                rows.append({
                    "datetime": pd.Timestamp.now(),
                    "symbol": coin["symbol"],
                    "name": coin["name"],
                    "circulating": coin.get("circulating", {}).get("peggedUSD", 0),
                    "price": coin.get("price", 1.0),
                })

        return pd.DataFrame(rows)

    def fetch_stablecoin_history(self, stablecoin: str = "tether") -> pd.DataFrame:
        """获取稳定币历史数据"""
        url = f"https://stablecoins.llama.fi/stablecoincharts/all"
        params = {"stablecoin": stablecoin}

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        rows = []
        for item in data:
            rows.append({
                "datetime": pd.to_datetime(item["date"], unit="s"),
                "total_circulating": item.get("totalCirculating", {}).get("peggedUSD", 0),
            })

        df = pd.DataFrame(rows)
        df["stablecoin"] = stablecoin
        return df

    # ==================== DeFi TVL ====================
    def fetch_tvl_history(self, chain: str = "Ethereum") -> pd.DataFrame:
        """获取链上TVL历史"""
        url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"

        data = self._request(url)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["date"], unit="s")
        df["tvl"] = df["tvl"].astype(float)
        df["chain"] = chain

        return df[["datetime", "chain", "tvl"]]

    def fetch_total_tvl(self) -> pd.DataFrame:
        """获取总TVL历史"""
        url = "https://api.llama.fi/v2/historicalChainTvl"

        data = self._request(url)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["date"], unit="s")
        df["tvl"] = df["tvl"].astype(float)

        return df[["datetime", "tvl"]]

    # ==================== 批量采集 ====================
    def collect_all(self) -> dict:
        """采集所有情绪数据"""
        logger.info("Collecting sentiment data...")

        data = {}

        # 恐惧贪婪指数
        data["fear_greed"] = self.fetch_fear_greed_index()

        # 稳定币供应
        data["stablecoin_supply"] = self.fetch_stablecoin_supply()

        # USDT历史
        data["usdt_history"] = self.fetch_stablecoin_history("tether")

        # 总TVL
        data["tvl_total"] = self.fetch_total_tvl()

        # ETH TVL
        data["tvl_eth"] = self.fetch_tvl_history("Ethereum")

        return data

    def save_data(self, data: dict, date_str: str = None):
        """保存数据"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        for key, df in data.items():
            if df is not None and not df.empty:
                if "fear_greed" in key:
                    path = self.data_dir / "sentiment" / f"{key}_{date_str}.parquet"
                elif "stablecoin" in key or "usdt" in key:
                    path = self.data_dir / "stablecoin" / f"{key}_{date_str}.parquet"
                else:
                    path = self.data_dir / "defi" / f"{key}_{date_str}.parquet"

                df.to_parquet(path, index=False)
                logger.info(f"Saved {len(df)} rows to {path}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    collector = SentimentDataCollector()
    data = collector.collect_all()
    collector.save_data(data)

    for key, df in data.items():
        if df is not None:
            print(f"{key}: {len(df)} rows")
            print(df.head())
            print()
