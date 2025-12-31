"""
Google Trends 数据采集器

数据类型:
1. Bitcoin 搜索热度
2. Cryptocurrency 搜索热度
3. Ethereum 搜索热度

注意:
- 需要安装 pytrends: pip install pytrends
- Google可能会限制请求频率
- 数据为相对值 (0-100)

使用示例:
    collector = GoogleTrendsCollector()
    trends = collector.fetch_trends(['bitcoin', 'ethereum'])
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 尝试导入pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. GoogleTrendsCollector will be limited.")


class GoogleTrendsCollector:
    """Google Trends 数据采集器"""

    DEFAULT_KEYWORDS = ["bitcoin", "cryptocurrency", "ethereum", "crypto"]

    def __init__(
        self,
        keywords: List[str] = None,
        data_dir: str = "~/.cryptoquant/data/google_trends",
        rate_limit_delay: float = 2.0,
        geo: str = "",  # 全球
        hl: str = "en-US",
    ):
        """
        初始化采集器

        Args:
            keywords: 关键词列表
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
            geo: 地区代码 (空=全球)
            hl: 语言
        """
        self.keywords = keywords or self.DEFAULT_KEYWORDS
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay
        self.geo = geo
        self.hl = hl

        self.data_dir.mkdir(parents=True, exist_ok=True)

        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl=hl, tz=0)
            logger.info(f"GoogleTrendsCollector initialized with keywords: {self.keywords}")
        else:
            self.pytrends = None
            logger.warning("GoogleTrendsCollector running in limited mode (no pytrends)")

    def fetch_interest_over_time(
        self,
        keywords: List[str] = None,
        timeframe: str = "today 12-m",
    ) -> pd.DataFrame:
        """
        获取关键词的搜索热度时间序列

        Args:
            keywords: 关键词列表 (最多5个)
            timeframe: 时间范围
                - 'today 12-m': 过去12个月
                - 'today 3-m': 过去3个月
                - 'now 7-d': 过去7天
                - '2024-01-01 2024-12-31': 指定范围

        Returns:
            包含搜索热度的DataFrame (0-100)
        """
        if not PYTRENDS_AVAILABLE:
            return self._fetch_fallback()

        keywords = keywords or self.keywords[:5]

        try:
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo=self.geo,
                gprop=""
            )

            time.sleep(self.rate_limit_delay)

            df = self.pytrends.interest_over_time()

            if df.empty:
                return pd.DataFrame()

            # 移除 isPartial 列
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df = df.reset_index()
            df = df.rename(columns={"date": "datetime"})

            return df

        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return self._fetch_fallback()

    def _fetch_fallback(self) -> pd.DataFrame:
        """在pytrends不可用时返回模拟数据"""
        logger.warning("Using fallback data (simulated)")

        dates = pd.date_range(
            end=datetime.now(),
            periods=365,
            freq="D"
        )

        # 生成模拟数据 (带有一些周期性)
        import numpy as np
        np.random.seed(42)

        data = {"datetime": dates}
        for kw in self.keywords[:5]:
            # 基础趋势 + 周期性 + 噪声
            trend = 50 + 10 * np.sin(np.arange(len(dates)) / 30)
            noise = np.random.normal(0, 5, len(dates))
            data[kw] = np.clip(trend + noise, 0, 100).astype(int)

        return pd.DataFrame(data)

    def fetch_related_queries(
        self,
        keyword: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        获取相关搜索词

        Args:
            keyword: 主关键词

        Returns:
            包含 'top' 和 'rising' 相关词的字典
        """
        if not PYTRENDS_AVAILABLE:
            return {"top": pd.DataFrame(), "rising": pd.DataFrame()}

        try:
            self.pytrends.build_payload([keyword], timeframe="today 3-m", geo=self.geo)
            time.sleep(self.rate_limit_delay)

            related = self.pytrends.related_queries()
            return related.get(keyword, {})

        except Exception as e:
            logger.error(f"Error fetching related queries: {e}")
            return {"top": pd.DataFrame(), "rising": pd.DataFrame()}

    def fetch_trends_history(
        self,
        keywords: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取历史搜索趋势

        Args:
            keywords: 关键词列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        """
        if start_date and end_date:
            timeframe = f"{start_date} {end_date}"
        else:
            timeframe = "today 12-m"

        return self.fetch_interest_over_time(keywords, timeframe)

    def save_data(self, df: pd.DataFrame, date_str: str = None):
        """保存数据到Parquet文件"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        if not df.empty:
            path = self.data_dir / f"trends_{date_str}.parquet"
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} rows to {path}")

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """加载历史数据"""
        files = sorted(self.data_dir.glob("trends_*.parquet"))

        if not files:
            return pd.DataFrame()

        # 加载最新的文件
        df = pd.read_parquet(files[-1])

        if "datetime" in df.columns:
            if start_date:
                df = df[df["datetime"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["datetime"] <= pd.Timestamp(end_date)]

        return df

    def calculate_momentum(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        计算搜索趋势动量

        Args:
            df: 搜索趋势数据
            window: 动量窗口 (天)
        """
        if df.empty:
            return df

        result = df.copy()

        for col in df.columns:
            if col == "datetime":
                continue
            result[f"{col}_momentum"] = df[col].pct_change(window)
            result[f"{col}_ma7"] = df[col].rolling(7).mean()

        return result


# 使用示例
if __name__ == "__main__":
    collector = GoogleTrendsCollector()

    # 获取搜索趋势
    trends = collector.fetch_interest_over_time()
    print("Trends data shape:", trends.shape)
    print(trends.tail())

    # 计算动量
    with_momentum = collector.calculate_momentum(trends)
    print("\nWith momentum:")
    print(with_momentum.tail())

    # 保存数据
    collector.save_data(trends)
