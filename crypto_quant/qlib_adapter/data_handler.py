"""
Qlib数据处理器 - 将采集的数据转换为Qlib格式

Qlib数据格式要求:
- MultiIndex: (datetime, instrument)
- 字段名以$开头: $open, $close, $funding_rate等
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger


class CryptoDataHandler:
    """
    加密货币数据处理器

    将采集的Parquet数据转换为Qlib兼容格式:
    - 合并多源数据 (K线、资金费率、持仓量等)
    - 统一时间索引
    - 字段重命名为Qlib格式
    """

    # Qlib字段映射
    FIELD_MAPPING = {
        # K线数据
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
        "quote_volume": "$quote_volume",
        "taker_buy_volume": "$taker_buy_volume",
        "trades": "$trades",

        # 资金费率
        "funding_rate": "$funding_rate",
        "mark_price": "$mark_price",

        # 持仓量
        "open_interest": "$open_interest",
        "open_interest_value": "$open_interest_value",

        # 多空比
        "ls_ratio": "$ls_ratio",
        "long_account": "$long_account",
        "short_account": "$short_account",
        "top_ls_ratio": "$top_ls_ratio",

        # 主动买卖
        "taker_buy_sell_ratio": "$taker_buy_sell_ratio",
        "taker_buy_vol": "$taker_buy_vol",
        "taker_sell_vol": "$taker_sell_vol",

        # 情绪
        "fear_greed_index": "$fear_greed_index",

        # 订单簿
        "bid1": "$bid1", "bsize1": "$bsize1",
        "ask1": "$ask1", "asize1": "$asize1",
        # ... 更多档位
    }

    def __init__(self, data_dir: str = "~/.cryptoquant/data"):
        self.data_dir = Path(data_dir).expanduser()

    def _load_parquet_files(self, subdir: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载Parquet文件"""
        path = self.data_dir / subdir
        if not path.exists():
            return pd.DataFrame()

        files = sorted(path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            date_str = f.stem.split("_")[-1] if "_" in f.stem else f.stem
            if start_date and date_str < start_date.replace("-", ""):
                continue
            if end_date and date_str > end_date.replace("-", ""):
                continue
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def load_and_merge(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """
        加载并合并所有数据源

        Args:
            symbols: 交易对列表
            start_date: 开始日期
            end_date: 结束日期
            freq: 数据频率

        Returns:
            合并后的DataFrame (MultiIndex: datetime, instrument)
        """
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")

        # 1. 加载K线数据 (主表)
        klines = self._load_parquet_files("klines", start_date, end_date)
        if klines.empty:
            logger.error("No klines data found")
            return pd.DataFrame()

        klines = klines[klines["symbol"].isin(symbols)]
        klines["datetime"] = pd.to_datetime(klines["datetime"])

        # 2. 加载资金费率
        funding = self._load_parquet_files("funding", start_date, end_date)
        if not funding.empty:
            funding = funding[funding["symbol"].isin(symbols)]
            funding["datetime"] = pd.to_datetime(funding["datetime"])

        # 3. 加载持仓量
        oi = self._load_parquet_files("oi", start_date, end_date)
        if not oi.empty:
            oi = oi[oi["symbol"].isin(symbols)]
            oi["datetime"] = pd.to_datetime(oi["datetime"])

        # 4. 加载多空比
        ls_ratio = self._load_parquet_files("ls_ratio", start_date, end_date)
        if not ls_ratio.empty:
            ls_ratio = ls_ratio[ls_ratio["symbol"].isin(symbols)]
            ls_ratio["datetime"] = pd.to_datetime(ls_ratio["datetime"])

        # 5. 加载主动买卖比
        taker = self._load_parquet_files("taker", start_date, end_date)
        if not taker.empty:
            taker = taker[taker["symbol"].isin(symbols)]
            taker["datetime"] = pd.to_datetime(taker["datetime"])

        # 6. 加载情绪数据
        sentiment = self._load_parquet_files("sentiment", start_date, end_date)
        if not sentiment.empty:
            sentiment["datetime"] = pd.to_datetime(sentiment["datetime"])

        # 合并数据
        df = klines.copy()

        # 合并资金费率 (资金费率每8小时一次，需要forward fill)
        if not funding.empty:
            funding_pivot = funding.pivot(index="datetime", columns="symbol", values="funding_rate")
            funding_pivot = funding_pivot.resample(freq).ffill()
            funding_melted = funding_pivot.reset_index().melt(
                id_vars=["datetime"], var_name="symbol", value_name="funding_rate"
            )
            df = df.merge(funding_melted, on=["datetime", "symbol"], how="left")

        # 合并持仓量
        if not oi.empty:
            df = df.merge(
                oi[["datetime", "symbol", "open_interest", "open_interest_value"]],
                on=["datetime", "symbol"],
                how="left"
            )

        # 合并多空比
        if not ls_ratio.empty:
            df = df.merge(
                ls_ratio[["datetime", "symbol", "ls_ratio", "long_account", "short_account"]],
                on=["datetime", "symbol"],
                how="left"
            )

        # 合并主动买卖比
        if not taker.empty:
            df = df.merge(
                taker[["datetime", "symbol", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]],
                on=["datetime", "symbol"],
                how="left"
            )

        # 合并情绪数据 (广播到所有symbol)
        if not sentiment.empty:
            sentiment_daily = sentiment.set_index("datetime").resample("D").last()
            sentiment_daily = sentiment_daily.resample(freq).ffill().reset_index()
            df = df.merge(sentiment_daily[["datetime", "fear_greed_index"]], on="datetime", how="left")

        # 填充缺失值
        df = df.fillna(method="ffill")

        logger.info(f"Merged data shape: {df.shape}")
        return df

    def to_qlib_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换为Qlib格式

        Args:
            df: 原始DataFrame

        Returns:
            Qlib格式的DataFrame (MultiIndex: datetime, instrument)
        """
        if df.empty:
            return df

        # 重命名列
        rename_map = {}
        for old_name, new_name in self.FIELD_MAPPING.items():
            if old_name in df.columns:
                rename_map[old_name] = new_name

        df = df.rename(columns=rename_map)

        # 添加必要的Qlib字段
        df["$factor"] = 1.0  # 永续合约无复权
        df["$change"] = df.groupby("symbol")["$close"].pct_change()

        # 计算衍生字段
        if "$close" in df.columns:
            # 收益率
            df["$return_1h"] = df.groupby("symbol")["$close"].pct_change()

            # 波动率 (20周期)
            df["$volatility"] = df.groupby("symbol")["$return_1h"].transform(
                lambda x: x.rolling(20).std()
            )

        # 转换symbol为instrument
        df["instrument"] = df["symbol"].str.lower()

        # 设置MultiIndex
        df = df.set_index(["datetime", "instrument"])
        df = df.sort_index()

        # 删除无用列
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        return df

    def get_qlib_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """
        获取Qlib格式数据 (一键调用)

        Args:
            symbols: 交易对列表
            start_date: 开始日期
            end_date: 结束日期
            freq: 数据频率

        Returns:
            Qlib格式的DataFrame
        """
        df = self.load_and_merge(symbols, start_date, end_date, freq)
        return self.to_qlib_format(df)

    def get_feature_names(self) -> List[str]:
        """获取所有可用特征名"""
        return list(self.FIELD_MAPPING.values())


# ==================== 使用示例 ====================
if __name__ == "__main__":
    handler = CryptoDataHandler()

    symbols = ["BTCUSDT", "ETHUSDT"]
    df = handler.get_qlib_data(
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    print("Qlib格式数据:")
    print(df.head(20))
    print(f"\n列: {df.columns.tolist()}")
    print(f"形状: {df.shape}")
