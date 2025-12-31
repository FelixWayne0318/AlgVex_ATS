"""
DeFiLlama 数据采集器 - 采集链上 DeFi 指标

数据类型:
1. TVL (Total Value Locked)
2. Protocol TVL 历史
3. Stablecoins 流通量
4. 桥接资金流
5. DEX 交易量
6. Yields (收益率)

使用示例:
    collector = DeFiLlamaCollector()
    tvl = collector.fetch_protocol_tvl("aave")
    stables = collector.fetch_stablecoin_chart("tether")
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


class DeFiLlamaCollector:
    """DeFiLlama DeFi 数据采集器"""

    BASE_URL = "https://api.llama.fi"
    COINS_URL = "https://coins.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"
    BRIDGES_URL = "https://bridges.llama.fi"
    YIELDS_URL = "https://yields.llama.fi"

    def __init__(
        self,
        data_dir: str = "~/.cryptoquant/data/defillama",
        rate_limit_delay: float = 0.2,
    ):
        """
        初始化采集器

        Args:
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tvl").mkdir(exist_ok=True)
        (self.data_dir / "stablecoins").mkdir(exist_ok=True)
        (self.data_dir / "bridges").mkdir(exist_ok=True)
        (self.data_dir / "dex").mkdir(exist_ok=True)
        (self.data_dir / "yields").mkdir(exist_ok=True)

        logger.info("DeFiLlamaCollector initialized")

    def _request(self, url: str, params: dict = None) -> Optional[Any]:
        """发送 API 请求"""
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)
            return resp.json()

        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

    # ==================== TVL 数据 ====================
    def fetch_protocols(self) -> pd.DataFrame:
        """获取所有协议列表"""
        url = f"{self.BASE_URL}/protocols"
        data = self._request(url)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    def fetch_protocol_tvl(self, protocol: str) -> pd.DataFrame:
        """
        获取协议 TVL 历史

        Args:
            protocol: 协议名 (slug), 如 'aave', 'uniswap'
        """
        url = f"{self.BASE_URL}/protocol/{protocol}"
        data = self._request(url)

        if not data or "tvl" not in data:
            return pd.DataFrame()

        tvl_data = data["tvl"]
        df = pd.DataFrame(tvl_data)
        df["datetime"] = pd.to_datetime(df["date"], unit="s")
        df["protocol"] = protocol

        return df[["datetime", "protocol", "totalLiquidityUSD"]]

    def fetch_chain_tvl(self, chain: str) -> pd.DataFrame:
        """
        获取链 TVL 历史

        Args:
            chain: 链名, 如 'Ethereum', 'BSC', 'Arbitrum'
        """
        url = f"{self.BASE_URL}/v2/historicalChainTvl/{chain}"
        data = self._request(url)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["date"], unit="s")
        df["chain"] = chain

        return df[["datetime", "chain", "tvl"]]

    def fetch_total_tvl(self) -> pd.DataFrame:
        """获取全网 TVL 历史"""
        url = f"{self.BASE_URL}/v2/historicalChainTvl"
        data = self._request(url)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["date"], unit="s")

        return df[["datetime", "tvl"]]

    # ==================== Stablecoins ====================
    def fetch_stablecoins(self) -> pd.DataFrame:
        """获取所有稳定币列表"""
        url = f"{self.STABLECOINS_URL}/stablecoins"
        data = self._request(url)

        if not data or "peggedAssets" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["peggedAssets"])

    def fetch_stablecoin_chart(self, stablecoin_id: str) -> pd.DataFrame:
        """
        获取稳定币流通量历史

        Args:
            stablecoin_id: 稳定币 ID (数字) 或 名称
        """
        url = f"{self.STABLECOINS_URL}/stablecoincharts/all"
        params = {"stablecoin": stablecoin_id}
        data = self._request(url, params)

        if not data:
            return pd.DataFrame()

        result = []
        for item in data:
            result.append({
                "datetime": pd.Timestamp(item["date"], unit="s"),
                "stablecoin": stablecoin_id,
                "circulating": item.get("totalCirculating", {}).get("peggedUSD", 0),
            })

        return pd.DataFrame(result)

    def fetch_stablecoin_chains(self) -> pd.DataFrame:
        """获取各链稳定币分布"""
        url = f"{self.STABLECOINS_URL}/stablecoinchains"
        data = self._request(url)

        if not data:
            return pd.DataFrame()

        result = []
        for chain_data in data:
            result.append({
                "datetime": pd.Timestamp.now(),
                "chain": chain_data.get("name"),
                "total_circulating_usd": chain_data.get("totalCirculatingUSD", {}).get("peggedUSD", 0),
            })

        return pd.DataFrame(result)

    # ==================== 桥接资金 ====================
    def fetch_bridges(self) -> pd.DataFrame:
        """获取所有桥列表"""
        url = f"{self.BRIDGES_URL}/bridges"
        data = self._request(url)

        if not data or "bridges" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["bridges"])

    def fetch_bridge_volume(self, bridge_id: int) -> pd.DataFrame:
        """
        获取桥接交易量历史

        Args:
            bridge_id: 桥 ID
        """
        url = f"{self.BRIDGES_URL}/bridgevolume/{bridge_id}"
        data = self._request(url)

        if not data:
            return pd.DataFrame()

        result = []
        for item in data:
            result.append({
                "datetime": pd.Timestamp(item["date"], unit="s"),
                "bridge_id": bridge_id,
                "volume_deposit_usd": item.get("depositUSD", 0),
                "volume_withdraw_usd": item.get("withdrawUSD", 0),
            })

        return pd.DataFrame(result)

    def fetch_bridge_day_stats(self) -> pd.DataFrame:
        """获取桥接日统计"""
        url = f"{self.BRIDGES_URL}/bridgedaystats/all"
        params = {"chain": "all"}
        data = self._request(url, params)

        if not data:
            return pd.DataFrame()

        result = []
        for item in data:
            result.append({
                "datetime": pd.Timestamp(item["date"], unit="s"),
                "deposit_usd": item.get("depositUSD", 0),
                "withdraw_usd": item.get("withdrawUSD", 0),
                "net_flow": item.get("depositUSD", 0) - item.get("withdrawUSD", 0),
            })

        return pd.DataFrame(result)

    # ==================== DEX 交易量 ====================
    def fetch_dexs(self) -> pd.DataFrame:
        """获取所有 DEX 列表"""
        url = f"{self.BASE_URL}/overview/dexs"
        data = self._request(url)

        if not data or "protocols" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["protocols"])

    def fetch_dex_volume(self, protocol: str) -> pd.DataFrame:
        """
        获取 DEX 交易量历史

        Args:
            protocol: DEX 名称 (slug)
        """
        url = f"{self.BASE_URL}/summary/dexs/{protocol}"
        data = self._request(url)

        if not data or "totalDataChart" not in data:
            return pd.DataFrame()

        result = []
        for item in data["totalDataChart"]:
            result.append({
                "datetime": pd.Timestamp(item[0], unit="s"),
                "dex": protocol,
                "volume_usd": item[1],
            })

        return pd.DataFrame(result)

    def fetch_chain_dex_volume(self, chain: str) -> pd.DataFrame:
        """获取链上 DEX 总交易量"""
        url = f"{self.BASE_URL}/overview/dexs/{chain}"
        data = self._request(url)

        if not data or "totalDataChart" not in data:
            return pd.DataFrame()

        result = []
        for item in data["totalDataChart"]:
            result.append({
                "datetime": pd.Timestamp(item[0], unit="s"),
                "chain": chain,
                "volume_usd": item[1],
            })

        return pd.DataFrame(result)

    # ==================== Yields ====================
    def fetch_yields(self) -> pd.DataFrame:
        """获取所有收益率池"""
        url = f"{self.YIELDS_URL}/pools"
        data = self._request(url)

        if not data or "data" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        df["datetime"] = pd.Timestamp.now()

        return df

    def fetch_pool_yield_history(self, pool_id: str) -> pd.DataFrame:
        """
        获取池收益率历史

        Args:
            pool_id: 池 ID
        """
        url = f"{self.YIELDS_URL}/chart/{pool_id}"
        data = self._request(url)

        if not data or "data" not in data:
            return pd.DataFrame()

        result = []
        for item in data["data"]:
            result.append({
                "datetime": pd.to_datetime(item["timestamp"]),
                "pool_id": pool_id,
                "apy": item.get("apy"),
                "tvl_usd": item.get("tvlUsd"),
            })

        return pd.DataFrame(result)

    # ==================== 价格数据 ====================
    def fetch_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """
        获取代币当前价格

        Args:
            coins: 代币地址列表, 格式 "chain:address"
                   如 ["ethereum:0x...", "coingecko:bitcoin"]
        """
        coins_str = ",".join(coins)
        url = f"{self.COINS_URL}/prices/current/{coins_str}"
        data = self._request(url)

        if not data or "coins" not in data:
            return {}

        result = {}
        for coin, info in data["coins"].items():
            result[coin] = info.get("price", 0)

        return result

    def fetch_historical_prices(
        self,
        coins: List[str],
        timestamp: int = None,
    ) -> Dict[str, float]:
        """
        获取代币历史价格

        Args:
            coins: 代币地址列表
            timestamp: Unix 时间戳 (秒)
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())

        coins_str = ",".join(coins)
        url = f"{self.COINS_URL}/prices/historical/{timestamp}/{coins_str}"
        data = self._request(url)

        if not data or "coins" not in data:
            return {}

        result = {}
        for coin, info in data["coins"].items():
            result[coin] = info.get("price", 0)

        return result

    # ==================== 批量采集 ====================
    def collect_tvl_data(
        self,
        protocols: List[str] = None,
        chains: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        采集 TVL 数据

        Args:
            protocols: 协议列表
            chains: 链列表
        """
        if protocols is None:
            protocols = ["aave", "uniswap", "lido", "makerdao", "curve"]
        if chains is None:
            chains = ["Ethereum", "BSC", "Arbitrum", "Polygon", "Optimism"]

        result = {
            "protocol_tvl": [],
            "chain_tvl": [],
            "total_tvl": None,
        }

        # 协议 TVL
        for protocol in protocols:
            logger.info(f"Fetching TVL for {protocol}...")
            df = self.fetch_protocol_tvl(protocol)
            if not df.empty:
                result["protocol_tvl"].append(df)

        # 链 TVL
        for chain in chains:
            logger.info(f"Fetching TVL for {chain}...")
            df = self.fetch_chain_tvl(chain)
            if not df.empty:
                result["chain_tvl"].append(df)

        # 总 TVL
        logger.info("Fetching total TVL...")
        result["total_tvl"] = self.fetch_total_tvl()

        # 合并
        if result["protocol_tvl"]:
            result["protocol_tvl"] = pd.concat(result["protocol_tvl"], ignore_index=True)
        else:
            result["protocol_tvl"] = pd.DataFrame()

        if result["chain_tvl"]:
            result["chain_tvl"] = pd.concat(result["chain_tvl"], ignore_index=True)
        else:
            result["chain_tvl"] = pd.DataFrame()

        return result

    def collect_stablecoin_data(
        self,
        stablecoins: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        采集稳定币数据

        Args:
            stablecoins: 稳定币 ID 列表
        """
        if stablecoins is None:
            stablecoins = ["1", "2", "3"]  # USDT, USDC, DAI

        result = {
            "stablecoin_supply": [],
            "chain_distribution": None,
        }

        for stable_id in stablecoins:
            logger.info(f"Fetching stablecoin {stable_id}...")
            df = self.fetch_stablecoin_chart(stable_id)
            if not df.empty:
                result["stablecoin_supply"].append(df)

        result["chain_distribution"] = self.fetch_stablecoin_chains()

        if result["stablecoin_supply"]:
            result["stablecoin_supply"] = pd.concat(
                result["stablecoin_supply"], ignore_index=True
            )
        else:
            result["stablecoin_supply"] = pd.DataFrame()

        return result

    def save_data(self, data: Dict[str, pd.DataFrame], date_str: str = None):
        """保存数据到 Parquet 文件"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        for key, df in data.items():
            if df is not None and not df.empty:
                # 确定子目录
                if "tvl" in key.lower():
                    subdir = "tvl"
                elif "stable" in key.lower():
                    subdir = "stablecoins"
                elif "bridge" in key.lower():
                    subdir = "bridges"
                elif "dex" in key.lower() or "volume" in key.lower():
                    subdir = "dex"
                else:
                    subdir = "tvl"

                path = self.data_dir / subdir / f"{key}_{date_str}.parquet"
                df.to_parquet(path, index=False)
                logger.info(f"Saved {len(df)} rows to {path}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    logger.add("defillama_collector.log", rotation="10 MB")

    collector = DeFiLlamaCollector()

    # 采集 TVL 数据
    tvl_data = collector.collect_tvl_data()
    collector.save_data(tvl_data)

    # 采集稳定币数据
    stable_data = collector.collect_stablecoin_data()
    collector.save_data(stable_data)

    # 打印统计
    print("\nTVL Data:")
    for key, df in tvl_data.items():
        if df is not None and not df.empty:
            print(f"  {key}: {len(df)} rows")

    print("\nStablecoin Data:")
    for key, df in stable_data.items():
        if df is not None and not df.empty:
            print(f"  {key}: {len(df)} rows")
