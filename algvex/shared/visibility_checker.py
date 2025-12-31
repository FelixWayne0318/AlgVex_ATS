"""
AlgVex 可见性检查器

功能:
- 检查数据在指定时间点是否可见
- 防止未来信息泄露 (Lookahead)
- 根据数据类型应用不同的可见性规则

核心规则:
任何数据可用 当且仅当: visible_time <= snapshot_cutoff

可见性类型:
- realtime: 实时数据，event_time + latency_buffer
- bar_close: K线数据，bar_close_time
- bar_close_delayed: 延迟数据，bar_close_time + delay
- scheduled: 定时数据，scheduled_time
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import yaml


class VisibilityType(Enum):
    """可见性类型"""
    REALTIME = "realtime"
    BAR_CLOSE = "bar_close"
    BAR_CLOSE_DELAYED = "bar_close_delayed"
    SCHEDULED = "scheduled"


class VisibilityViolationError(Exception):
    """可见性违规错误"""
    pass


@dataclass
class VisibilityResult:
    """可见性检查结果"""
    is_visible: bool
    data_time: datetime
    visible_time: datetime
    snapshot_cutoff: datetime
    source_id: str
    message: str


class VisibilityChecker:
    """可见性检查器"""

    def __init__(self, config_path: str = "config/visibility.yaml"):
        """
        初始化可见性检查器

        Args:
            config_path: 可见性配置文件路径
        """
        self.config = self._load_config(config_path)
        # 默认不启用严格模式，允许检查返回结果而非抛出异常
        self.strict_mode = self.config.get("validation", {}).get("strict_mode", False)
        self.on_violation = self.config.get("validation", {}).get("on_violation", "warn")

        # 解析安全边际
        self.safety_margin = self._parse_duration(
            self.config.get("safety_margin", "1s")
        )

        # 解析可见性规则
        self.visibility_types = self.config.get("visibility_types", {})
        self.source_visibility_map = self.config.get("source_visibility_map", {})
        self.source_delay_overrides = self.config.get("source_delay_overrides", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载可见性配置"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # 返回默认配置
            return {
                "safety_margin": "1s",
                "visibility_types": {
                    "realtime": {"latency_buffer": "1s"},
                    "bar_close": {},
                    "bar_close_delayed": {"publication_delay": "5min"},
                    "scheduled": {"publication_delay": "0s"},
                },
                "source_visibility_map": {
                    "klines_5m": "bar_close",
                    "open_interest_5m": "bar_close_delayed",
                    "funding_8h": "scheduled",
                },
            }

    @staticmethod
    def _parse_duration(duration_str: str) -> timedelta:
        """
        解析时间持续字符串

        Args:
            duration_str: 如 "5min", "1s", "1h"

        Returns:
            timedelta对象
        """
        if not duration_str:
            return timedelta(seconds=0)

        duration_str = duration_str.strip().lower()

        if duration_str.endswith("min"):
            return timedelta(minutes=int(duration_str[:-3]))
        elif duration_str.endswith("m") and not duration_str.endswith("min"):
            return timedelta(minutes=int(duration_str[:-1]))
        elif duration_str.endswith("s"):
            return timedelta(seconds=int(duration_str[:-1]))
        elif duration_str.endswith("h"):
            return timedelta(hours=int(duration_str[:-1]))
        elif duration_str.endswith("d"):
            return timedelta(days=int(duration_str[:-1]))
        else:
            return timedelta(seconds=int(duration_str))

    def get_visibility_type(self, source_id: str) -> str:
        """获取数据源的可见性类型"""
        return self.source_visibility_map.get(source_id, "bar_close")

    def get_publication_delay(self, source_id: str) -> timedelta:
        """获取数据源的发布延迟"""
        # 检查是否有覆盖配置
        if source_id in self.source_delay_overrides:
            delay_str = self.source_delay_overrides[source_id].get(
                "publication_delay", "0s"
            )
            return self._parse_duration(delay_str)

        # 使用类型默认值
        vis_type = self.get_visibility_type(source_id)
        type_config = self.visibility_types.get(vis_type, {})
        delay_str = type_config.get("publication_delay", "0s")

        return self._parse_duration(delay_str)

    def compute_visible_time(
        self,
        source_id: str,
        data_time: datetime,
        bar_close_time: Optional[datetime] = None,
        scheduled_time: Optional[datetime] = None,
    ) -> datetime:
        """
        计算数据的可见时间

        Args:
            source_id: 数据源ID
            data_time: 数据/事件时间
            bar_close_time: K线收盘时间（可选）
            scheduled_time: 定时发布时间（可选）

        Returns:
            可见时间
        """
        vis_type = self.get_visibility_type(source_id)

        if vis_type == "realtime":
            # 实时数据: event_time + latency_buffer
            latency = self._parse_duration(
                self.visibility_types.get("realtime", {}).get("latency_buffer", "1s")
            )
            return data_time + latency

        elif vis_type == "bar_close":
            # K线数据: bar_close_time
            return bar_close_time or data_time

        elif vis_type == "bar_close_delayed":
            # 延迟数据: bar_close_time + delay
            base_time = bar_close_time or data_time
            delay = self.get_publication_delay(source_id)
            return base_time + delay

        elif vis_type == "scheduled":
            # 定时数据: scheduled_time
            return scheduled_time or data_time

        else:
            # 默认: 数据时间
            return data_time

    def compute_snapshot_cutoff(self, signal_time: datetime) -> datetime:
        """
        计算快照截止时间

        Args:
            signal_time: 信号生成时间

        Returns:
            快照截止时间
        """
        return signal_time - self.safety_margin

    def check_visibility(
        self,
        source_id: str,
        data_time: datetime,
        signal_time: datetime,
        bar_close_time: Optional[datetime] = None,
        scheduled_time: Optional[datetime] = None,
    ) -> VisibilityResult:
        """
        检查数据在信号时间点是否可见

        Args:
            source_id: 数据源ID
            data_time: 数据时间
            signal_time: 信号生成时间
            bar_close_time: K线收盘时间（可选）
            scheduled_time: 定时发布时间（可选）

        Returns:
            VisibilityResult 检查结果
        """
        # 计算可见时间
        visible_time = self.compute_visible_time(
            source_id, data_time, bar_close_time, scheduled_time
        )

        # 计算快照截止时间
        snapshot_cutoff = self.compute_snapshot_cutoff(signal_time)

        # 判断是否可见
        is_visible = visible_time <= snapshot_cutoff

        # 生成消息
        if is_visible:
            message = f"数据可见: visible_time={visible_time} <= cutoff={snapshot_cutoff}"
        else:
            message = (
                f"数据不可见 (未来数据):\n"
                f"  visible_time={visible_time}\n"
                f"  snapshot_cutoff={snapshot_cutoff}\n"
                f"  差异={(visible_time - snapshot_cutoff).total_seconds()}秒"
            )

        result = VisibilityResult(
            is_visible=is_visible,
            data_time=data_time,
            visible_time=visible_time,
            snapshot_cutoff=snapshot_cutoff,
            source_id=source_id,
            message=message,
        )

        # 根据配置处理违规
        if not is_visible and self.strict_mode:
            if self.on_violation == "error":
                raise VisibilityViolationError(message)

        return result

    def check_factor_visibility(
        self,
        factor_id: str,
        factor_value: Any,
        signal_time: datetime,
        data_dependencies: Dict[str, datetime],
    ) -> List[VisibilityResult]:
        """
        检查因子计算中所有数据依赖的可见性

        Args:
            factor_id: 因子ID
            factor_value: 因子值
            signal_time: 信号时间
            data_dependencies: {数据源ID: 数据时间}

        Returns:
            可见性检查结果列表
        """
        results = []

        for source_id, data_time in data_dependencies.items():
            result = self.check_visibility(
                source_id=source_id,
                data_time=data_time,
                signal_time=signal_time,
            )
            results.append(result)

        return results

    def get_usable_data_time(
        self,
        source_id: str,
        signal_time: datetime,
    ) -> datetime:
        """
        获取在信号时间可以使用的最新数据时间

        Args:
            source_id: 数据源ID
            signal_time: 信号时间

        Returns:
            可用的最新数据时间
        """
        snapshot_cutoff = self.compute_snapshot_cutoff(signal_time)
        delay = self.get_publication_delay(source_id)

        # visible_time = data_time + delay
        # visible_time <= snapshot_cutoff
        # data_time <= snapshot_cutoff - delay
        usable_time = snapshot_cutoff - delay

        return usable_time

    def filter_visible_data(
        self,
        df: "pd.DataFrame",
        current_time: datetime,
        delay_minutes: int = 0,
        source_id: Optional[str] = None,
    ) -> "pd.DataFrame":
        """
        过滤 DataFrame，只保留在指定时间可见的数据

        Args:
            df: 带有时间索引的 DataFrame
            current_time: 当前时间（信号时间）
            delay_minutes: 额外延迟（分钟）
            source_id: 数据源ID（可选，用于获取配置的延迟）

        Returns:
            过滤后只包含可见数据的 DataFrame
        """
        import pandas as pd

        if df.empty:
            return df

        # 计算快照截止时间
        snapshot_cutoff = self.compute_snapshot_cutoff(current_time)

        # 计算总延迟
        total_delay = timedelta(minutes=delay_minutes)
        if source_id:
            total_delay += self.get_publication_delay(source_id)

        # 数据可见条件: data_time + total_delay <= snapshot_cutoff
        # 即: data_time <= snapshot_cutoff - total_delay
        visible_cutoff = snapshot_cutoff - total_delay

        # 过滤数据
        if isinstance(df.index, pd.DatetimeIndex):
            return df[df.index <= visible_cutoff]
        else:
            # 如果索引不是时间类型，尝试使用 datetime 列
            if "datetime" in df.columns:
                return df[df["datetime"] <= visible_cutoff]
            elif "time" in df.columns:
                return df[df["time"] <= visible_cutoff]
            else:
                # 无法过滤，返回原始数据
                return df


def check_visibility(
    source_id: str,
    data_time: datetime,
    signal_time: datetime,
    config_path: str = "config/visibility.yaml",
) -> bool:
    """
    便捷函数：检查数据可见性

    Args:
        source_id: 数据源ID
        data_time: 数据时间
        signal_time: 信号时间
        config_path: 配置文件路径

    Returns:
        是否可见
    """
    checker = VisibilityChecker(config_path)
    result = checker.check_visibility(source_id, data_time, signal_time)
    return result.is_visible


# 测试代码
if __name__ == "__main__":
    from datetime import datetime

    checker = VisibilityChecker()

    # 测试K线数据可见性
    signal_time = datetime(2024, 1, 1, 10, 5, 0)  # 10:05
    kline_time = datetime(2024, 1, 1, 10, 0, 0)   # 10:00 K线

    result = checker.check_visibility(
        source_id="klines_5m",
        data_time=kline_time,
        signal_time=signal_time,
        bar_close_time=datetime(2024, 1, 1, 10, 5, 0),
    )
    print(f"K线可见性: {result.is_visible}")
    print(result.message)

    # 测试OI数据可见性（有延迟）
    oi_time = datetime(2024, 1, 1, 10, 0, 0)  # 10:00 OI
    result = checker.check_visibility(
        source_id="open_interest_5m",
        data_time=oi_time,
        signal_time=signal_time,
        bar_close_time=datetime(2024, 1, 1, 10, 5, 0),
    )
    print(f"\nOI可见性: {result.is_visible}")
    print(result.message)
