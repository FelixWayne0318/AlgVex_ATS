"""
Fear & Greed Index 数据采集器

数据源: Alternative.me Fear & Greed Index API
- 免费API，无需认证
- 每日更新一次
- 历史数据可回溯多年

使用示例:
    collector = FearGreedCollector()
    current = collector.fetch_current()
    history = collector.fetch_history(limit=365)
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


class FearGreedCollector:
    """Fear & Greed Index 数据采集器"""

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(
        self,
        data_dir: str = "~/.cryptoquant/data/fear_greed",
        rate_limit_delay: float = 0.5,
    ):
        """
        初始化采集器

        Args:
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FearGreedCollector initialized")

    def _request(self, params: dict = None) -> Optional[Dict]:
        """发送API请求"""
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)

            data = resp.json()
            if data.get("metadata", {}).get("error"):
                logger.error(f"API error: {data['metadata']['error']}")
                return None

            return data

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def fetch_current(self) -> pd.DataFrame:
        """获取当前Fear & Greed指数"""
        data = self._request({"limit": 1})

        if not data or "data" not in data:
            return pd.DataFrame()

        item = data["data"][0]

        return pd.DataFrame([{
            "datetime": pd.Timestamp(int(item["timestamp"]), unit="s"),
            "value": int(item["value"]),
            "value_classification": item["value_classification"],
        }])

    def fetch_history(self, limit: int = 365) -> pd.DataFrame:
        """
        获取历史Fear & Greed指数

        Args:
            limit: 获取天数 (最大约2000天)

        Returns:
            包含历史数据的DataFrame
        """
        data = self._request({"limit": limit, "format": "json"})

        if not data or "data" not in data:
            return pd.DataFrame()

        records = []
        for item in data["data"]:
            records.append({
                "datetime": pd.Timestamp(int(item["timestamp"]), unit="s"),
                "value": int(item["value"]),
                "value_classification": item["value_classification"],
            })

        df = pd.DataFrame(records)
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def fetch_time_range(
        self,
        start_date: str,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取指定时间范围的数据

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD' (默认为今天)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        days = (end - start).days + 1

        df = self.fetch_history(limit=days + 30)  # 多取一些以确保覆盖

        if df.empty:
            return df

        # 过滤时间范围
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        return df[mask].reset_index(drop=True)

    def save_data(self, df: pd.DataFrame, date_str: str = None):
        """保存数据到Parquet文件"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        if not df.empty:
            path = self.data_dir / f"fear_greed_{date_str}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} rows to {path}")

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """加载历史数据"""
        files = sorted(self.data_dir.glob("fear_greed_*.parquet"))

        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            date_str = f.stem.replace("fear_greed_", "")
            if start_date and date_str < start_date.replace("-", ""):
                continue
            if end_date and date_str > end_date.replace("-", ""):
                continue
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["datetime"])


# 使用示例
if __name__ == "__main__":
    collector = FearGreedCollector()

    # 获取当前值
    current = collector.fetch_current()
    print("Current Fear & Greed:")
    print(current)

    # 获取历史数据
    history = collector.fetch_history(limit=30)
    print(f"\nLast 30 days: {len(history)} records")
    print(history.head())

    # 保存数据
    collector.save_data(history)
