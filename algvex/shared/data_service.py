"""
AlgVex 数据服务接口

功能:
- 统一数据访问入口 (P0-2: DataManager唯一入口)
- 禁止其他模块直接访问数据库/Redis
- 提供一致的数据获取API
- 集成可见性检查

使用方式:
    from shared.data_service import DataService

    # 创建数据服务
    ds = DataService()

    # 获取K线数据
    klines = ds.get_klines("BTCUSDT", start="2024-01-01", end="2024-01-31")

    # 获取带可见性检查的数据
    klines = ds.get_klines_safe("BTCUSDT", signal_time=datetime.now())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .visibility_checker import VisibilityChecker, VisibilityResult
from .time_provider import TimeProvider


@dataclass
class DataRequest:
    """数据请求"""
    source_id: str
    symbol: str
    start_time: datetime
    end_time: datetime
    fields: Optional[List[str]] = None
    signal_time: Optional[datetime] = None  # 用于可见性检查


@dataclass
class DataResponse:
    """数据响应"""
    source_id: str
    symbol: str
    data: pd.DataFrame
    start_time: datetime
    end_time: datetime
    visibility_results: Optional[List[VisibilityResult]] = None
    metadata: Dict[str, Any] = None


class DataProvider(ABC):
    """数据提供者抽象基类"""

    @abstractmethod
    def fetch(self, request: DataRequest) -> DataResponse:
        """获取数据"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


class DataService:
    """
    统一数据服务

    规则:
    - 所有数据访问必须通过 DataService
    - 禁止直接访问数据库/Redis
    - 自动进行可见性检查
    """

    def __init__(
        self,
        visibility_config: str = "config/visibility.yaml",
        enable_visibility_check: bool = True,
    ):
        """
        初始化数据服务

        Args:
            visibility_config: 可见性配置文件路径
            enable_visibility_check: 是否启用可见性检查
        """
        self.enable_visibility_check = enable_visibility_check
        self._visibility_checker = VisibilityChecker(visibility_config)
        self._providers: Dict[str, DataProvider] = {}
        self._cache: Dict[str, pd.DataFrame] = {}

    def register_provider(self, source_id: str, provider: DataProvider):
        """
        注册数据提供者

        Args:
            source_id: 数据源ID
            provider: 数据提供者实例
        """
        self._providers[source_id] = provider

    def get_data(
        self,
        source_id: str,
        symbol: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        signal_time: Optional[datetime] = None,
        fields: Optional[List[str]] = None,
    ) -> DataResponse:
        """
        获取数据（通用接口）

        Args:
            source_id: 数据源ID
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            signal_time: 信号时间（用于可见性检查）
            fields: 需要的字段列表

        Returns:
            DataResponse 数据响应
        """
        # 解析时间
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        # 创建请求
        request = DataRequest(
            source_id=source_id,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            fields=fields,
            signal_time=signal_time,
        )

        # 可见性检查
        visibility_results = None
        if self.enable_visibility_check and signal_time is not None:
            visibility_results = self._check_visibility(request)

        # 获取数据
        if source_id in self._providers:
            response = self._providers[source_id].fetch(request)
        else:
            # 返回空数据（或从缓存获取）
            response = DataResponse(
                source_id=source_id,
                symbol=symbol,
                data=pd.DataFrame(),
                start_time=start_time,
                end_time=end_time,
            )

        response.visibility_results = visibility_results
        return response

    def _check_visibility(self, request: DataRequest) -> List[VisibilityResult]:
        """检查数据可见性"""
        results = []

        if request.signal_time is None:
            return results

        # 检查结束时间的可见性
        result = self._visibility_checker.check_visibility(
            source_id=request.source_id,
            data_time=request.end_time,
            signal_time=request.signal_time,
        )
        results.append(result)

        return results

    def get_klines(
        self,
        symbol: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        interval: str = "5m",
        signal_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        获取K线数据

        Args:
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            interval: K线间隔
            signal_time: 信号时间

        Returns:
            K线 DataFrame
        """
        source_id = f"klines_{interval}"
        response = self.get_data(
            source_id=source_id,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            signal_time=signal_time,
        )
        return response.data

    def get_open_interest(
        self,
        symbol: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        signal_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        获取持仓量数据

        Args:
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            signal_time: 信号时间

        Returns:
            持仓量 DataFrame
        """
        response = self.get_data(
            source_id="open_interest_5m",
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            signal_time=signal_time,
        )
        return response.data

    def get_funding_rate(
        self,
        symbol: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        signal_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        获取资金费率数据

        Args:
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            signal_time: 信号时间

        Returns:
            资金费率 DataFrame
        """
        response = self.get_data(
            source_id="funding_8h",
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            signal_time=signal_time,
        )
        return response.data

    def get_usable_data_time(
        self,
        source_id: str,
        signal_time: Optional[datetime] = None,
    ) -> datetime:
        """
        获取在信号时间可用的最新数据时间

        Args:
            source_id: 数据源ID
            signal_time: 信号时间

        Returns:
            可用的最新数据时间
        """
        if signal_time is None:
            signal_time = TimeProvider.utcnow()

        return self._visibility_checker.get_usable_data_time(source_id, signal_time)

    def get_latest_data(
        self,
        source_id: str,
        symbol: str,
        lookback_bars: int = 1,
        signal_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        获取最新的可用数据

        Args:
            source_id: 数据源ID
            symbol: 交易对
            lookback_bars: 回看的bar数量
            signal_time: 信号时间

        Returns:
            最新数据 DataFrame
        """
        if signal_time is None:
            signal_time = TimeProvider.utcnow()

        # 计算可用的数据时间范围
        usable_end = self.get_usable_data_time(source_id, signal_time)

        # 根据数据源确定bar间隔
        if "5m" in source_id:
            bar_seconds = 300
        elif "1h" in source_id:
            bar_seconds = 3600
        elif "8h" in source_id:
            bar_seconds = 28800
        else:
            bar_seconds = 300

        usable_start = usable_end - timedelta(seconds=bar_seconds * lookback_bars)

        return self.get_data(
            source_id=source_id,
            symbol=symbol,
            start_time=usable_start,
            end_time=usable_end,
            signal_time=signal_time,
        ).data

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

    def get_available_sources(self) -> List[str]:
        """获取可用的数据源列表"""
        return list(self._providers.keys())


# 内存数据提供者（用于测试）
class InMemoryDataProvider(DataProvider):
    """内存数据提供者（用于测试）"""

    def __init__(self, data: Dict[str, pd.DataFrame] = None):
        self._data = data or {}

    def set_data(self, key: str, df: pd.DataFrame):
        """设置数据"""
        self._data[key] = df

    def fetch(self, request: DataRequest) -> DataResponse:
        """获取数据"""
        key = f"{request.source_id}:{request.symbol}"
        df = self._data.get(key, pd.DataFrame())

        # 过滤时间范围
        if not df.empty and "timestamp" in df.columns:
            mask = (df["timestamp"] >= request.start_time) & (
                df["timestamp"] <= request.end_time
            )
            df = df[mask]

        return DataResponse(
            source_id=request.source_id,
            symbol=request.symbol,
            data=df,
            start_time=request.start_time,
            end_time=request.end_time,
        )

    def is_available(self) -> bool:
        return True


# 测试代码
if __name__ == "__main__":
    # 创建数据服务
    ds = DataService(enable_visibility_check=True)

    # 创建测试数据
    test_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "open": [100 + i * 0.1 for i in range(100)],
            "high": [101 + i * 0.1 for i in range(100)],
            "low": [99 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )

    # 注册测试提供者
    provider = InMemoryDataProvider()
    provider.set_data("klines_5m:BTCUSDT", test_data)
    ds.register_provider("klines_5m", provider)

    # 获取数据
    print("=== 获取K线数据 ===")
    response = ds.get_data(
        source_id="klines_5m",
        symbol="BTCUSDT",
        start_time="2024-01-01T00:00:00",
        end_time="2024-01-01T01:00:00",
        signal_time=datetime(2024, 1, 1, 1, 5, 0),
    )
    print(f"获取到 {len(response.data)} 条数据")

    # 可见性检查
    print("\n=== 可见性检查 ===")
    if response.visibility_results:
        for result in response.visibility_results:
            print(f"可见: {result.is_visible}")
            print(result.message)

    # 获取可用数据时间
    print("\n=== 可用数据时间 ===")
    signal_time = datetime(2024, 1, 1, 10, 5, 0)
    usable_time = ds.get_usable_data_time("klines_5m", signal_time)
    print(f"信号时间: {signal_time}")
    print(f"可用数据时间: {usable_time}")

    usable_time_oi = ds.get_usable_data_time("open_interest_5m", signal_time)
    print(f"OI可用数据时间: {usable_time_oi}")
