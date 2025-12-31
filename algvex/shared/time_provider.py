"""
AlgVex 时间提供者

功能:
- 统一时间获取接口，禁止直接使用 datetime.now()
- 支持 Replay 模式（使用模拟时间）
- 支持 Live 模式（使用真实时间）
- 确保 Replay 确定性 (P0-4)

使用方式:
    from shared.time_provider import TimeProvider, get_current_time

    # 获取当前时间
    now = get_current_time()

    # Replay 模式
    TimeProvider.set_replay_mode(True)
    TimeProvider.set_simulated_time(datetime(2024, 1, 1, 10, 0, 0))
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import threading


class TimeProvider:
    """
    统一时间提供者

    支持两种使用方式:
    1. 类方法 (静态/单例模式): TimeProvider.now(), TimeProvider.set_replay_mode()
    2. 实例方法: provider = TimeProvider(mode="replay"); provider.now()

    规则:
    - 禁止在业务代码中直接使用 datetime.now()
    - 所有时间获取必须通过 TimeProvider
    - Replay 模式下使用模拟时间，保证确定性
    """

    # 类级别变量 (用于类方法模式)
    _lock = threading.Lock()
    _replay_mode: bool = False
    _simulated_time: Optional[datetime] = None
    _time_offset: timedelta = timedelta(0)

    def __init__(self, mode: str = "live", fixed_time: Optional[datetime] = None):
        """
        创建 TimeProvider 实例

        Args:
            mode: "live" 或 "replay"
            fixed_time: Replay 模式下的固定时间
        """
        self._instance_mode = mode
        self._instance_time = fixed_time
        self._instance_offset = timedelta(0)

    @property
    def mode(self) -> str:
        """获取当前模式"""
        return self._instance_mode

    def now(self, tz: Optional[timezone] = None) -> datetime:
        """
        获取当前时间 (实例方法)

        Args:
            tz: 时区，如果为 None 则保留原始时区信息

        Returns:
            当前时间
        """
        if self._instance_mode == "replay" and self._instance_time is not None:
            result = self._instance_time + self._instance_offset
            # 保留输入时间的时区信息（naive 或 aware）
            return result
        else:
            # Live 模式
            if tz is not None:
                result = datetime.now(tz)
            else:
                result = datetime.utcnow()
            return result

    def advance_time(self, delta: timedelta):
        """
        推进模拟时间 (实例方法)

        Args:
            delta: 时间增量
        """
        if self._instance_mode != "replay":
            raise RuntimeError("只能在 Replay 模式下推进时间")
        self._instance_offset += delta

    def set_time(self, time: datetime):
        """
        设置模拟时间 (实例方法)

        Args:
            time: 新的模拟时间
        """
        if self._instance_mode != "replay":
            raise RuntimeError("只能在 Replay 模式下设置时间")
        self._instance_time = time
        self._instance_offset = timedelta(0)

    def switch_to_replay(self, fixed_time: datetime):
        """
        切换到 Replay 模式

        Args:
            fixed_time: 固定时间
        """
        self._instance_mode = "replay"
        self._instance_time = fixed_time
        self._instance_offset = timedelta(0)

    def switch_to_live(self):
        """切换到 Live 模式"""
        self._instance_mode = "live"
        self._instance_time = None
        self._instance_offset = timedelta(0)

    # ==================== 类方法 (向后兼容) ====================

    @classmethod
    def is_replay_mode(cls) -> bool:
        """是否处于 Replay 模式"""
        return cls._replay_mode

    @classmethod
    def set_replay_mode(cls, enabled: bool):
        """
        设置 Replay 模式

        Args:
            enabled: 是否启用 Replay 模式
        """
        with cls._lock:
            cls._replay_mode = enabled
            if not enabled:
                cls._simulated_time = None

    @classmethod
    def set_simulated_time(cls, time: datetime):
        """
        设置模拟时间 (仅 Replay 模式有效)

        Args:
            time: 模拟的当前时间
        """
        with cls._lock:
            if not cls._replay_mode:
                raise RuntimeError("只能在 Replay 模式下设置模拟时间")
            cls._simulated_time = time

    @classmethod
    def class_advance_time(cls, delta: timedelta):
        """
        推进模拟时间 (仅 Replay 模式有效) - 类方法版本

        Args:
            delta: 时间增量
        """
        with cls._lock:
            if not cls._replay_mode:
                raise RuntimeError("只能在 Replay 模式下推进时间")
            if cls._simulated_time is None:
                raise RuntimeError("请先设置模拟时间")
            cls._simulated_time += delta

    @classmethod
    def set_time_offset(cls, offset: timedelta):
        """
        设置时间偏移 (用于测试)

        Args:
            offset: 时间偏移量
        """
        with cls._lock:
            cls._time_offset = offset

    @classmethod
    def class_now(cls, tz: Optional[timezone] = None) -> datetime:
        """
        获取当前时间 (类方法版本)

        Args:
            tz: 时区，默认 UTC

        Returns:
            当前时间
        """
        if tz is None:
            tz = timezone.utc

        with cls._lock:
            if cls._replay_mode and cls._simulated_time is not None:
                # Replay 模式：使用模拟时间
                result = cls._simulated_time
            else:
                # Live 模式：使用真实时间
                result = datetime.now(tz)

            # 应用时间偏移
            result = result + cls._time_offset

        # 确保有时区信息
        if result.tzinfo is None:
            result = result.replace(tzinfo=tz)

        return result

    @classmethod
    def utcnow(cls) -> datetime:
        """获取当前 UTC 时间"""
        return cls.class_now(timezone.utc)

    @classmethod
    def today(cls) -> datetime:
        """获取今天的日期（UTC）"""
        return cls.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    @classmethod
    def reset(cls):
        """重置所有设置"""
        with cls._lock:
            cls._replay_mode = False
            cls._simulated_time = None
            cls._time_offset = timedelta(0)

    @classmethod
    def get_bar_time(cls, bar_duration_seconds: int = 300) -> datetime:
        """
        获取当前 Bar 的开始时间

        Args:
            bar_duration_seconds: Bar 持续时间（秒），默认 300（5分钟）

        Returns:
            Bar 开始时间
        """
        now = cls.utcnow()
        timestamp = now.timestamp()
        bar_start = int(timestamp // bar_duration_seconds) * bar_duration_seconds
        return datetime.fromtimestamp(bar_start, tz=timezone.utc)

    @classmethod
    def get_next_bar_time(cls, bar_duration_seconds: int = 300) -> datetime:
        """
        获取下一个 Bar 的开始时间

        Args:
            bar_duration_seconds: Bar 持续时间（秒）

        Returns:
            下一个 Bar 开始时间
        """
        current_bar = cls.get_bar_time(bar_duration_seconds)
        return current_bar + timedelta(seconds=bar_duration_seconds)

    @classmethod
    def seconds_until_next_bar(cls, bar_duration_seconds: int = 300) -> float:
        """
        计算距离下一个 Bar 的秒数

        Args:
            bar_duration_seconds: Bar 持续时间（秒）

        Returns:
            距离下一个 Bar 的秒数
        """
        now = cls.utcnow()
        next_bar = cls.get_next_bar_time(bar_duration_seconds)
        return (next_bar - now).total_seconds()


def get_current_time(tz: Optional[timezone] = None) -> datetime:
    """
    便捷函数：获取当前时间

    Args:
        tz: 时区，默认 UTC

    Returns:
        当前时间
    """
    return TimeProvider.class_now(tz)


def get_utc_now() -> datetime:
    """便捷函数：获取当前 UTC 时间"""
    return TimeProvider.utcnow()


# 单例管理
_time_provider_instance: Optional[TimeProvider] = None


def get_time_provider() -> TimeProvider:
    """获取全局 TimeProvider 实例"""
    global _time_provider_instance
    if _time_provider_instance is None:
        _time_provider_instance = TimeProvider()
    return _time_provider_instance


def reset_time_provider():
    """重置 TimeProvider 状态"""
    global _time_provider_instance
    TimeProvider.reset()
    _time_provider_instance = None


# 上下文管理器：用于临时进入 Replay 模式
class ReplayContext:
    """
    Replay 模式上下文管理器

    使用方式:
        with ReplayContext(datetime(2024, 1, 1, 10, 0, 0)):
            now = get_current_time()  # 返回模拟时间
    """

    def __init__(self, simulated_time: datetime):
        self.simulated_time = simulated_time
        self._previous_replay_mode = False
        self._previous_simulated_time = None

    def __enter__(self):
        self._previous_replay_mode = TimeProvider.is_replay_mode()
        self._previous_simulated_time = TimeProvider._simulated_time

        TimeProvider.set_replay_mode(True)
        TimeProvider.set_simulated_time(self.simulated_time)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TimeProvider._replay_mode = self._previous_replay_mode
        TimeProvider._simulated_time = self._previous_simulated_time
        return False


# 测试代码
if __name__ == "__main__":
    # Live 模式
    print("=== Live 模式 ===")
    print(f"当前时间: {get_current_time()}")
    print(f"当前 Bar: {TimeProvider.get_bar_time()}")
    print(f"下一 Bar: {TimeProvider.get_next_bar_time()}")
    print(f"距下一 Bar: {TimeProvider.seconds_until_next_bar():.1f} 秒")

    # Replay 模式
    print("\n=== Replay 模式 ===")
    replay_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    TimeProvider.set_replay_mode(True)
    TimeProvider.set_simulated_time(replay_time)
    print(f"模拟时间: {get_current_time()}")
    print(f"模拟 Bar: {TimeProvider.get_bar_time()}")

    # 推进时间
    TimeProvider.class_advance_time(timedelta(minutes=5))
    print(f"推进 5 分钟后: {get_current_time()}")

    # 重置
    TimeProvider.reset()
    print(f"\n重置后: {get_current_time()}")
