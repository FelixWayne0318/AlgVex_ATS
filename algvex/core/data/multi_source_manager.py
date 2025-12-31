"""
AlgVex 多数据源管理器

统一管理多种数据来源:
1. 交易所数据 (币安)
2. 链上数据 (Glassnode, CryptoQuant)
3. 情绪数据 (恐慌贪婪, 社交媒体)
4. 宏观数据 (利率, CPI)

所有数据统一转换为 Qlib MultiIndex 格式
"""

import asyncio
from abc import ABC, abstractmethod
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


class DataSource(ABC):
    """数据源基类"""

    def __init__(self, name: str, rate_limit: float = 0.5):
        self.name = name
        self.rate_limit = rate_limit
        self._cache: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """获取数据"""
        pass

    @abstractmethod
    def get_field_mapping(self) -> Dict[str, str]:
        """返回字段映射 (原始字段 -> Qlib字段)"""
        pass

    def to_qlib_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为 Qlib 格式"""
        if df.empty:
            return df

        mapping = self.get_field_mapping()
        df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        return df


# ==================== 交易所数据源 ====================

class BinanceDataSource(DataSource):
    """币安永续合约数据源"""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, symbols: List[str] = None):
        super().__init__("binance")
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]

    def fetch(
        self,
        start_date: str,
        end_date: str,
        data_type: str = "klines",
        interval: str = "1h",
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取币安数据

        Args:
            data_type: klines, funding, oi, ls_ratio, taker
        """
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)

        all_data = []
        for symbol in self.symbols:
            if data_type == "klines":
                df = self._fetch_klines(symbol, interval, start_ts, end_ts)
            elif data_type == "funding":
                df = self._fetch_funding(symbol, start_ts, end_ts)
            elif data_type == "oi":
                df = self._fetch_oi(symbol, interval, start_ts, end_ts)
            elif data_type == "ls_ratio":
                df = self._fetch_ls_ratio(symbol, interval)
            else:
                continue

            if not df.empty:
                all_data.append(df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _fetch_klines(self, symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500}

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            logger.error(f"Fetch klines failed: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore"
        ])

        df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
        df["symbol"] = symbol

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        return df[["datetime", "symbol", "open", "high", "low", "close", "volume", "quote_volume"]]

    def _fetch_funding(self, symbol: str, start: int, end: int) -> pd.DataFrame:
        url = f"{self.BASE_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000}

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            logger.error(f"Fetch funding failed: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["symbol"] = symbol

        return df[["datetime", "symbol", "funding_rate"]]

    def _fetch_oi(self, symbol: str, period: str, start: int, end: int) -> pd.DataFrame:
        url = f"{self.BASE_URL}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": period, "startTime": start, "endTime": end, "limit": 500}

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            logger.error(f"Fetch OI failed: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["symbol"] = symbol

        return df[["datetime", "symbol", "open_interest"]]

    def _fetch_ls_ratio(self, symbol: str, period: str) -> pd.DataFrame:
        url = f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio"
        params = {"symbol": symbol, "period": period, "limit": 500}

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            logger.error(f"Fetch LS ratio failed: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["ls_ratio"] = df["longShortRatio"].astype(float)
        df["symbol"] = symbol

        return df[["datetime", "symbol", "ls_ratio"]]

    def get_field_mapping(self) -> Dict[str, str]:
        return {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "quote_volume": "$quote_volume",
            "funding_rate": "$funding_rate",
            "open_interest": "$open_interest",
            "ls_ratio": "$ls_ratio",
        }


# ==================== 链上数据源 ====================

class OnChainDataSource(DataSource):
    """链上数据源 (Glassnode/CryptoQuant 风格)"""

    def __init__(self, api_key: str = None):
        super().__init__("onchain")
        self.api_key = api_key

    def fetch(
        self,
        start_date: str,
        end_date: str,
        metrics: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取链上数据

        常用指标:
        - exchange_netflow: 交易所净流入
        - active_addresses: 活跃地址数
        - nvt_ratio: NVT比率
        - mvrv_ratio: MVRV比率
        - sopr: 已花费输出利润率
        """
        metrics = metrics or ["exchange_netflow", "active_addresses"]

        # 这里是示例实现，实际需要接入 Glassnode/CryptoQuant API
        # 目前返回模拟数据结构

        logger.warning("OnChainDataSource: Using mock data. Integrate real API for production.")

        dates = pd.date_range(start_date, end_date, freq="D")
        data = {
            "datetime": dates,
            "exchange_netflow": [0.0] * len(dates),  # 需要真实API
            "active_addresses": [0] * len(dates),
            "nvt_ratio": [0.0] * len(dates),
            "mvrv_ratio": [0.0] * len(dates),
        }

        return pd.DataFrame(data)

    def get_field_mapping(self) -> Dict[str, str]:
        return {
            "exchange_netflow": "$exchange_netflow",
            "active_addresses": "$active_addresses",
            "nvt_ratio": "$nvt_ratio",
            "mvrv_ratio": "$mvrv_ratio",
            "sopr": "$sopr",
        }


# ==================== 情绪数据源 ====================

class SentimentDataSource(DataSource):
    """情绪数据源"""

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        super().__init__("sentiment")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """获取恐慌贪婪指数"""
        try:
            # 计算需要的天数
            days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
            resp = requests.get(f"{self.FEAR_GREED_URL}?limit={days}", timeout=30)
            data = resp.json().get("data", [])
        except Exception as e:
            logger.error(f"Fetch fear greed failed: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["fear_greed_index"] = df["value"].astype(int)
        df["fear_greed_class"] = df["value_classification"]

        # 过滤日期范围
        df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

        return df[["datetime", "fear_greed_index", "fear_greed_class"]]

    def get_field_mapping(self) -> Dict[str, str]:
        return {
            "fear_greed_index": "$fear_greed",
            "fear_greed_class": "$fear_greed_class",
        }


# ==================== 宏观数据源 ====================

class MacroDataSource(DataSource):
    """宏观经济数据源"""

    def __init__(self):
        super().__init__("macro")

    def fetch(
        self,
        start_date: str,
        end_date: str,
        indicators: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取宏观数据

        指标:
        - dxy: 美元指数
        - us10y: 美国10年期国债收益率
        - sp500: 标普500
        - gold: 黄金价格
        """
        # 实际需要接入 Yahoo Finance 或 FRED API
        logger.warning("MacroDataSource: Using mock data. Integrate real API for production.")

        dates = pd.date_range(start_date, end_date, freq="D")
        data = {
            "datetime": dates,
            "dxy": [100.0] * len(dates),
            "us10y": [4.0] * len(dates),
            "sp500": [5000.0] * len(dates),
            "gold": [2000.0] * len(dates),
        }

        return pd.DataFrame(data)

    def get_field_mapping(self) -> Dict[str, str]:
        return {
            "dxy": "$dxy",
            "us10y": "$us10y",
            "sp500": "$sp500",
            "gold": "$gold",
        }


# ==================== 统一数据管理器 ====================

class MultiSourceDataManager:
    """
    多数据源统一管理器

    功能:
    1. 注册多个数据源
    2. 统一获取和合并数据
    3. 自动转换为 Qlib 格式
    4. 处理不同频率数据对齐
    """

    def __init__(self, data_dir: str = "~/.algvex/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sources: Dict[str, DataSource] = {}

        # 注册默认数据源
        self.register_source("binance", BinanceDataSource())
        self.register_source("sentiment", SentimentDataSource())

    def register_source(self, name: str, source: DataSource) -> None:
        """注册数据源"""
        self.sources[name] = source
        logger.info(f"Registered data source: {name}")

    def fetch_all(
        self,
        start_date: str,
        end_date: str,
        sources: List[str] = None,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """
        获取并合并所有数据源

        Args:
            start_date: 开始日期
            end_date: 结束日期
            sources: 要获取的数据源列表 (None=全部)
            freq: 目标频率

        Returns:
            合并后的 Qlib 格式 DataFrame
        """
        sources = sources or list(self.sources.keys())
        all_data = {}

        for source_name in sources:
            if source_name not in self.sources:
                logger.warning(f"Unknown source: {source_name}")
                continue

            source = self.sources[source_name]
            logger.info(f"Fetching from {source_name}...")

            try:
                df = source.fetch(start_date, end_date)
                if not df.empty:
                    df = source.to_qlib_format(df)
                    all_data[source_name] = df
            except Exception as e:
                logger.error(f"Failed to fetch {source_name}: {e}")

        # 合并数据
        return self._merge_data(all_data, freq)

    def _merge_data(
        self,
        data: Dict[str, pd.DataFrame],
        freq: str,
    ) -> pd.DataFrame:
        """合并多数据源"""
        if not data:
            return pd.DataFrame()

        # 以交易所数据为主表
        if "binance" in data:
            main_df = data["binance"].copy()
        else:
            main_df = list(data.values())[0].copy()

        # 确保 datetime 列
        if "datetime" not in main_df.columns and main_df.index.name != "datetime":
            logger.error("Main data missing datetime column")
            return pd.DataFrame()

        # 合并其他数据源
        for source_name, df in data.items():
            if source_name == "binance":
                continue

            if df.empty:
                continue

            # 对齐频率 (低频数据向高频填充)
            if "datetime" in df.columns:
                df = df.set_index("datetime")

            # 重采样到目标频率
            df_resampled = df.resample(freq).ffill().reset_index()

            # 合并
            merge_cols = ["datetime"] + [c for c in df_resampled.columns
                                         if c.startswith("$") and c not in main_df.columns]

            if len(merge_cols) > 1:
                main_df = main_df.merge(
                    df_resampled[merge_cols],
                    on="datetime",
                    how="left"
                )

        logger.info(f"Merged data shape: {main_df.shape}, columns: {list(main_df.columns)}")
        return main_df

    def to_qlib_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为 Qlib MultiIndex 格式"""
        if df.empty:
            return df

        if "symbol" in df.columns:
            df["instrument"] = df["symbol"].str.lower()
            df = df.drop(columns=["symbol"])

        if "datetime" in df.columns and "instrument" in df.columns:
            df = df.set_index(["datetime", "instrument"]).sort_index()

        return df

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """保存数据"""
        path = self.data_dir / f"{filename}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved to {path}")
        return path

    def load(self, filename: str) -> pd.DataFrame:
        """加载数据"""
        path = self.data_dir / f"{filename}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def get_available_fields(self) -> Dict[str, List[str]]:
        """获取所有可用字段"""
        fields = {}
        for name, source in self.sources.items():
            fields[name] = list(source.get_field_mapping().values())
        return fields


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化管理器
    manager = MultiSourceDataManager()

    # 注册额外数据源
    manager.register_source("onchain", OnChainDataSource())
    manager.register_source("macro", MacroDataSource())

    # 获取所有数据
    df = manager.fetch_all(
        start_date="2024-01-01",
        end_date="2024-01-31",
        freq="1h",
    )

    # 转换为 Qlib 格式
    qlib_df = manager.to_qlib_multiindex(df)

    print("可用字段:")
    for source, fields in manager.get_available_fields().items():
        print(f"  {source}: {fields}")

    print(f"\n数据形状: {qlib_df.shape}")
    print(qlib_df.head())
