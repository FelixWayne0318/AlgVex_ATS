"""
Deribit 数据采集器 - 采集期权和期货数据

数据类型:
1. 隐含波动率 (IV Surface)
2. Greeks (Delta, Gamma, Theta, Vega)
3. 期货价格和持仓量
4. 历史波动率 (HV)
5. DVOL 指数 (Deribit Volatility Index)

使用示例:
    collector = DeribitDataCollector()
    iv_data = collector.fetch_iv_surface("BTC")
    vol_index = collector.fetch_volatility_index("BTC")
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


class DeribitDataCollector:
    """Deribit 期权数据采集器"""

    BASE_URL = "https://www.deribit.com/api/v2"

    def __init__(
        self,
        currencies: List[str] = None,
        data_dir: str = "~/.cryptoquant/data/deribit",
        rate_limit_delay: float = 0.2,
    ):
        """
        初始化采集器

        Args:
            currencies: 货币列表, 如 ['BTC', 'ETH']
            data_dir: 数据存储目录
            rate_limit_delay: API调用间隔(秒)
        """
        self.currencies = currencies or ["BTC", "ETH"]
        self.data_dir = Path(data_dir).expanduser()
        self.rate_limit_delay = rate_limit_delay

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "iv_surface").mkdir(exist_ok=True)
        (self.data_dir / "greeks").mkdir(exist_ok=True)
        (self.data_dir / "futures").mkdir(exist_ok=True)
        (self.data_dir / "vol_index").mkdir(exist_ok=True)
        (self.data_dir / "hv").mkdir(exist_ok=True)

        logger.info(f"DeribitDataCollector initialized with {len(self.currencies)} currencies")

    def _request(
        self,
        method: str,
        params: dict = None,
    ) -> Optional[Dict[str, Any]]:
        """发送 API 请求"""
        url = f"{self.BASE_URL}/public/{method}"

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(self.rate_limit_delay)

            data = resp.json()
            if data.get("result"):
                return data["result"]
            else:
                logger.error(f"API error: {data.get('error', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            logger.error(f"Request failed: {url}, error: {e}")
            return None

    # ==================== 期权链 ====================
    def fetch_instruments(
        self,
        currency: str,
        kind: str = "option",
        expired: bool = False,
    ) -> pd.DataFrame:
        """
        获取期权/期货合约列表

        Args:
            currency: 货币 (BTC, ETH)
            kind: 合约类型 (option, future)
            expired: 是否包含已过期
        """
        params = {
            "currency": currency,
            "kind": kind,
            "expired": str(expired).lower(),
        }

        data = self._request("get_instruments", params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    # ==================== 隐含波动率曲面 ====================
    def fetch_book_summary_by_currency(
        self,
        currency: str,
        kind: str = "option",
    ) -> pd.DataFrame:
        """
        获取所有期权的市场数据 (含 IV)

        Args:
            currency: 货币
            kind: 合约类型
        """
        params = {"currency": currency, "kind": kind}

        data = self._request("get_book_summary_by_currency", params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.Timestamp.now()
        df["currency"] = currency

        return df

    def fetch_iv_surface(self, currency: str) -> pd.DataFrame:
        """
        构建 IV 曲面数据

        Returns:
            包含 strike, expiry, iv, delta, gamma, theta, vega 的 DataFrame
        """
        df = self.fetch_book_summary_by_currency(currency, "option")
        if df.empty:
            return pd.DataFrame()

        # 解析期权名称
        # 格式: BTC-31DEC24-100000-C
        def parse_instrument(name):
            parts = name.split("-")
            if len(parts) >= 4:
                return {
                    "underlying": parts[0],
                    "expiry": parts[1],
                    "strike": float(parts[2]),
                    "option_type": parts[3],
                }
            return None

        parsed = df["instrument_name"].apply(parse_instrument)
        parsed_df = pd.DataFrame(parsed.tolist())

        result = pd.concat([df, parsed_df], axis=1)

        # 选择关键列
        columns = [
            "datetime", "currency", "instrument_name",
            "underlying", "expiry", "strike", "option_type",
            "mark_iv", "mark_price", "underlying_price",
            "bid_price", "ask_price", "volume_usd",
            "open_interest",
        ]
        available_cols = [c for c in columns if c in result.columns]

        return result[available_cols]

    # ==================== Greeks ====================
    def fetch_option_data(self, instrument_name: str) -> Optional[Dict]:
        """获取单个期权的详细数据 (含 Greeks)"""
        params = {"instrument_name": instrument_name}
        return self._request("ticker", params)

    def fetch_greeks_batch(self, currency: str) -> pd.DataFrame:
        """
        批量获取期权 Greeks

        注意: 这会发送多个请求,谨慎使用
        """
        instruments = self.fetch_instruments(currency, "option")
        if instruments.empty:
            return pd.DataFrame()

        greeks_data = []

        for _, row in instruments.iterrows():
            name = row["instrument_name"]
            ticker = self.fetch_option_data(name)

            if ticker and "greeks" in ticker:
                greeks = ticker["greeks"]
                greeks_data.append({
                    "datetime": pd.Timestamp.now(),
                    "instrument_name": name,
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "theta": greeks.get("theta"),
                    "vega": greeks.get("vega"),
                    "rho": greeks.get("rho"),
                    "mark_iv": ticker.get("mark_iv"),
                    "underlying_price": ticker.get("underlying_price"),
                })

        if not greeks_data:
            return pd.DataFrame()

        return pd.DataFrame(greeks_data)

    # ==================== 波动率指数 (DVOL) ====================
    def fetch_volatility_index(self, currency: str) -> pd.DataFrame:
        """
        获取 DVOL (Deribit Volatility Index)

        类似 VIX,是 30 天隐含波动率指数
        """
        params = {"currency": currency}
        data = self._request("get_volatility_index_data", params)

        if not data:
            return pd.DataFrame()

        # data 是一个嵌套列表
        result = []
        for item in data.get("data", []):
            if len(item) >= 2:
                result.append({
                    "datetime": pd.Timestamp(item[0], unit="ms"),
                    "currency": currency,
                    "dvol": item[1],
                })

        return pd.DataFrame(result)

    def fetch_current_dvol(self, currency: str) -> Optional[Dict]:
        """获取当前 DVOL 值"""
        index_name = f"{currency.lower()}_usd"
        params = {"index_name": f"dvol_{index_name}"}
        return self._request("get_index_price", params)

    # ==================== 期货数据 ====================
    def fetch_futures(self, currency: str) -> pd.DataFrame:
        """获取所有期货合约数据"""
        df = self.fetch_book_summary_by_currency(currency, "future")
        if df.empty:
            return pd.DataFrame()

        columns = [
            "datetime", "instrument_name", "mark_price",
            "underlying_price", "bid_price", "ask_price",
            "volume_usd", "open_interest", "funding_8h",
        ]
        available_cols = [c for c in columns if c in df.columns]

        df["currency"] = currency
        return df[available_cols + ["currency"]]

    def fetch_perpetual_funding(self, currency: str) -> pd.DataFrame:
        """获取永续合约资金费率"""
        instrument = f"{currency}-PERPETUAL"
        params = {"instrument_name": instrument}

        data = self._request("ticker", params)
        if not data:
            return pd.DataFrame()

        return pd.DataFrame([{
            "datetime": pd.Timestamp.now(),
            "instrument": instrument,
            "funding_8h": data.get("funding_8h"),
            "current_funding": data.get("current_funding"),
            "mark_price": data.get("mark_price"),
            "index_price": data.get("index_price"),
            "open_interest": data.get("open_interest"),
        }])

    # ==================== 历史波动率 ====================
    def fetch_historical_volatility(
        self,
        currency: str,
    ) -> pd.DataFrame:
        """
        获取历史波动率数据

        Returns:
            包含 10/30/60/90/180 天历史波动率
        """
        params = {"currency": currency}
        data = self._request("get_historical_volatility", params)

        if not data:
            return pd.DataFrame()

        # data 是一个嵌套列表 [[timestamp, hv], ...]
        result = []
        for item in data:
            if len(item) >= 2:
                result.append({
                    "datetime": pd.Timestamp(item[0], unit="ms"),
                    "currency": currency,
                    "hv": item[1],
                })

        return pd.DataFrame(result)

    # ==================== 批量采集 ====================
    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """
        采集所有数据

        Returns:
            包含所有数据的字典
        """
        all_data = {
            "iv_surface": [],
            "futures": [],
            "vol_index": [],
            "hv": [],
            "funding": [],
        }

        for currency in self.currencies:
            logger.info(f"Collecting Deribit data for {currency}...")

            # IV 曲面
            iv = self.fetch_iv_surface(currency)
            if not iv.empty:
                all_data["iv_surface"].append(iv)

            # 期货
            futures = self.fetch_futures(currency)
            if not futures.empty:
                all_data["futures"].append(futures)

            # DVOL
            dvol = self.fetch_volatility_index(currency)
            if not dvol.empty:
                all_data["vol_index"].append(dvol)

            # 历史波动率
            hv = self.fetch_historical_volatility(currency)
            if not hv.empty:
                all_data["hv"].append(hv)

            # 永续资金费率
            funding = self.fetch_perpetual_funding(currency)
            if not funding.empty:
                all_data["funding"].append(funding)

        # 合并数据
        result = {}
        for key, dfs in all_data.items():
            if dfs:
                result[key] = pd.concat(dfs, ignore_index=True)
            else:
                result[key] = pd.DataFrame()

        return result

    def save_data(self, data: Dict[str, pd.DataFrame], date_str: str = None):
        """保存数据到 Parquet 文件"""
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


# ==================== ATM IV 计算 ====================
def calculate_atm_iv(iv_surface: pd.DataFrame, underlying_price: float) -> Dict:
    """
    计算 ATM 隐含波动率

    Args:
        iv_surface: IV 曲面数据
        underlying_price: 标的价格

    Returns:
        各到期日的 ATM IV
    """
    if iv_surface.empty:
        return {}

    result = {}

    for expiry in iv_surface["expiry"].unique():
        exp_data = iv_surface[iv_surface["expiry"] == expiry]

        # 找到最接近 ATM 的 strike
        exp_data = exp_data.copy()
        exp_data["strike_diff"] = abs(exp_data["strike"] - underlying_price)
        atm = exp_data.loc[exp_data["strike_diff"].idxmin()]

        result[expiry] = {
            "strike": atm["strike"],
            "iv": atm.get("mark_iv"),
            "option_type": atm.get("option_type"),
        }

    return result


# ==================== 使用示例 ====================
if __name__ == "__main__":
    logger.add("deribit_collector.log", rotation="10 MB")

    collector = DeribitDataCollector(["BTC", "ETH"])

    # 采集数据
    data = collector.collect_all()
    collector.save_data(data)

    # 打印统计
    for key, df in data.items():
        print(f"{key}: {len(df)} rows")

    # 计算 ATM IV
    if "iv_surface" in data and not data["iv_surface"].empty:
        btc_iv = data["iv_surface"][data["iv_surface"]["currency"] == "BTC"]
        if not btc_iv.empty and "underlying_price" in btc_iv.columns:
            price = btc_iv["underlying_price"].iloc[0]
            atm_ivs = calculate_atm_iv(btc_iv, price)
            print(f"\nBTC ATM IVs: {atm_ivs}")
