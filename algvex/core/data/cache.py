# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# AlgVex 缓存模块 - Qlib 0.9.7 原版复刻

"""
缓存模块 (Qlib 原版)

提供多级缓存机制:
- L1: 内存缓存 (MemCache)
- L2: 磁盘缓存 (DiskCache)
- L3: Redis 分布式缓存 (RedisCache)
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import hashlib
import contextlib
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd

from loguru import logger

# Redis 支持 (可选)
try:
    import redis
    import redis_lock
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    redis_lock = None


# ============================================================
# 缓存异常
# ============================================================

class CacheException(RuntimeError):
    """缓存异常"""
    pass


# ============================================================
# 哈希工具
# ============================================================

def hash_args(*args) -> str:
    """
    计算参数哈希值

    Parameters
    ----------
    *args
        参数列表

    Returns
    -------
    str
        MD5 哈希值
    """
    content = str(args).encode('utf-8')
    return hashlib.md5(content).hexdigest()


# ============================================================
# 内存缓存单元 (Qlib 原版)
# ============================================================

class MemCacheUnit(ABC):
    """
    内存缓存单元 (Qlib 原版)

    Memory Cache Unit with LRU eviction.
    """

    def __init__(self, size_limit: int = 0, **kwargs):
        """
        初始化缓存单元

        Parameters
        ----------
        size_limit
            缓存大小限制 (0 表示无限制)
        """
        self.size_limit = size_limit
        self._size = 0
        self.od = OrderedDict()

    def __setitem__(self, key, value):
        # 计算新大小
        self._adjust_size(key, value)
        # 存储
        self.od.__setitem__(key, value)
        # 移到末尾 (最近使用)
        self.od.move_to_end(key)

        # LRU 淘汰
        if self.limited:
            while self._size > self.size_limit:
                self.popitem(last=False)

    def __getitem__(self, key):
        v = self.od.__getitem__(key)
        self.od.move_to_end(key)
        return v

    def __contains__(self, key):
        return key in self.od

    def __len__(self):
        return len(self.od)

    def __repr__(self):
        limit_str = self.size_limit if self.limited else 'no limit'
        return f"{self.__class__.__name__}<size_limit:{limit_str} total_size:{self._size}>"

    def set_limit_size(self, limit: int):
        """设置大小限制"""
        self.size_limit = limit

    @property
    def limited(self) -> bool:
        """是否有大小限制"""
        return self.size_limit > 0

    @property
    def total_size(self) -> int:
        """当前总大小"""
        return self._size

    def clear(self):
        """清空缓存"""
        self._size = 0
        self.od.clear()

    def popitem(self, last: bool = True) -> Tuple[Any, Any]:
        """弹出项目"""
        k, v = self.od.popitem(last=last)
        self._size -= self._get_value_size(v)
        return k, v

    def pop(self, key) -> Any:
        """弹出指定键"""
        v = self.od.pop(key)
        self._size -= self._get_value_size(v)
        return v

    def _adjust_size(self, key, value):
        """调整大小计数"""
        if key in self.od:
            self._size -= self._get_value_size(self.od[key])
        self._size += self._get_value_size(value)

    @abstractmethod
    def _get_value_size(self, value) -> int:
        """获取值的大小"""
        raise NotImplementedError


class MemCacheLengthUnit(MemCacheUnit):
    """基于数量的缓存单元"""

    def _get_value_size(self, value) -> int:
        return 1


class MemCacheSizeofUnit(MemCacheUnit):
    """基于内存大小的缓存单元"""

    def _get_value_size(self, value) -> int:
        return sys.getsizeof(value)


# ============================================================
# 内存缓存 (Qlib 原版)
# ============================================================

class MemCache:
    """
    内存缓存 (Qlib 原版)

    Memory cache with multiple cache units.
    """

    def __init__(
        self,
        size_limit: int = 1000,
        limit_type: str = "length",
    ):
        """
        初始化内存缓存

        Parameters
        ----------
        size_limit
            缓存大小限制
        limit_type
            限制类型: "length" (数量) 或 "sizeof" (内存大小)
        """
        if limit_type == "length":
            klass = MemCacheLengthUnit
        elif limit_type == "sizeof":
            klass = MemCacheSizeofUnit
        else:
            raise ValueError(f"limit_type must be 'length' or 'sizeof', got {limit_type}")

        self.__calendar_cache = klass(size_limit)
        self.__instrument_cache = klass(size_limit)
        self.__feature_cache = klass(size_limit)
        self.__general_cache = klass(size_limit)

    def __getitem__(self, key: str) -> MemCacheUnit:
        if key == "c" or key == "calendar":
            return self.__calendar_cache
        elif key == "i" or key == "instrument":
            return self.__instrument_cache
        elif key == "f" or key == "feature":
            return self.__feature_cache
        elif key == "g" or key == "general":
            return self.__general_cache
        else:
            raise KeyError(f"Unknown cache unit: {key}")

    def clear(self):
        """清空所有缓存"""
        self.__calendar_cache.clear()
        self.__instrument_cache.clear()
        self.__feature_cache.clear()
        self.__general_cache.clear()

    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "calendar": len(self.__calendar_cache),
            "instrument": len(self.__instrument_cache),
            "feature": len(self.__feature_cache),
            "general": len(self.__general_cache),
        }


class MemCacheExpire:
    """
    带过期时间的缓存 (Qlib 原版)

    Memory cache with expiration.
    """

    CACHE_EXPIRE = 3600  # 默认过期时间 (秒)

    @staticmethod
    def set_cache(mem_cache: MemCacheUnit, key: str, value: Any):
        """
        设置缓存

        Parameters
        ----------
        mem_cache
            缓存单元
        key
            缓存键
        value
            缓存值
        """
        mem_cache[key] = (value, time.time())

    @staticmethod
    def get_cache(mem_cache: MemCacheUnit, key: str) -> Tuple[Any, bool]:
        """
        获取缓存

        Parameters
        ----------
        mem_cache
            缓存单元
        key
            缓存键

        Returns
        -------
        Tuple[Any, bool]
            (值, 是否过期)
        """
        value = None
        expire = False
        if key in mem_cache:
            value, latest_time = mem_cache[key]
            expire = (time.time() - latest_time) > MemCacheExpire.CACHE_EXPIRE
        return value, expire

    @classmethod
    def set_expire_time(cls, seconds: int):
        """设置过期时间"""
        cls.CACHE_EXPIRE = seconds


# ============================================================
# 磁盘缓存 (Qlib 原版)
# ============================================================

class DiskCache:
    """
    磁盘缓存 (Qlib 原版)

    Disk-based cache with pickle serialization.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./cache",
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ):
        """
        初始化磁盘缓存

        Parameters
        ----------
        cache_dir
            缓存目录
        protocol
            pickle 协议版本
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"

    def _get_meta_path(self, key: str) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"{key}.meta"

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        return self._get_cache_path(key).exists()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存

        Parameters
        ----------
        key
            缓存键
        default
            默认值

        Returns
        -------
        Any
            缓存值或默认值
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
                return default
        return default

    def set(self, key: str, value: Any, meta: Dict = None):
        """
        设置缓存

        Parameters
        ----------
        key
            缓存键
        value
            缓存值
        meta
            元数据
        """
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f, protocol=self.protocol)

            # 保存元数据
            if meta:
                meta_path = self._get_meta_path(key)
                meta['last_update'] = time.time()
                with open(meta_path, 'wb') as f:
                    pickle.dump(meta, f, protocol=self.protocol)
        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")

    def delete(self, key: str):
        """删除缓存"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        for p in [cache_path, meta_path]:
            if p.exists():
                p.unlink()

    def clear(self):
        """清空所有缓存"""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        for f in self.cache_dir.glob("*.meta"):
            f.unlink()

    def get_meta(self, key: str) -> Optional[Dict]:
        """获取元数据"""
        meta_path = self._get_meta_path(key)
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None


# ============================================================
# Redis 缓存 (Qlib 原版)
# ============================================================

class RedisCache:
    """
    Redis 分布式缓存 (Qlib 原版)

    Redis-based distributed cache with locking support.
    """

    LOCK_ID = "ALGVEX"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "algvex:",
        expire: int = 86400,  # 默认 1 天过期
        decode_responses: bool = False,
    ):
        """
        初始化 Redis 缓存

        Parameters
        ----------
        host
            Redis 主机
        port
            Redis 端口
        db
            Redis 数据库编号
        password
            Redis 密码
        prefix
            键前缀
        expire
            过期时间 (秒)
        decode_responses
            是否解码响应
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis and redis_lock packages are required for RedisCache")

        self.prefix = prefix
        self.expire = expire

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
        )

        # 测试连接
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}/{db}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """生成完整的键名"""
        return f"{self.prefix}{key}"

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        return self.client.exists(self._make_key(key)) > 0

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存

        Parameters
        ----------
        key
            缓存键
        default
            默认值

        Returns
        -------
        Any
            缓存值或默认值
        """
        full_key = self._make_key(key)
        data = self.client.get(full_key)
        if data is not None:
            try:
                return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Failed to deserialize cache {key}: {e}")
                return default
        return default

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        """
        设置缓存

        Parameters
        ----------
        key
            缓存键
        value
            缓存值
        expire
            过期时间 (秒)
        """
        full_key = self._make_key(key)
        try:
            data = pickle.dumps(value)
            self.client.set(full_key, data, ex=expire or self.expire)
        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")

    def delete(self, key: str):
        """删除缓存"""
        self.client.delete(self._make_key(key))

    def clear(self, pattern: str = "*"):
        """
        清空缓存

        Parameters
        ----------
        pattern
            键模式 (默认所有)
        """
        full_pattern = self._make_key(pattern)
        keys = self.client.keys(full_pattern)
        if keys:
            self.client.delete(*keys)

    @contextlib.contextmanager
    def reader_lock(self, lock_name: str, timeout: int = 60):
        """
        读锁上下文管理器 (Qlib 原版)

        Parameters
        ----------
        lock_name
            锁名称
        timeout
            超时时间 (秒)
        """
        if not REDIS_AVAILABLE:
            yield
            return

        rlock = redis_lock.Lock(self.client, f"{lock_name}-rlock")
        wlock = redis_lock.Lock(self.client, f"{lock_name}-wlock")
        reader_key = f"{lock_name}-reader"

        # 获取读锁
        rlock.acquire(timeout=timeout)
        try:
            readers = self.client.get(reader_key)
            if readers is None or int(readers) == 0:
                self._acquire_lock(wlock, lock_name)
            self.client.incr(reader_key)
        finally:
            rlock.release()

        try:
            yield
        finally:
            # 释放读锁
            rlock.acquire(timeout=timeout)
            try:
                self.client.decr(reader_key)
                if int(self.client.get(reader_key)) == 0:
                    self.client.delete(reader_key)
                    wlock.reset()
            finally:
                rlock.release()

    @contextlib.contextmanager
    def writer_lock(self, lock_name: str, timeout: int = 60):
        """
        写锁上下文管理器 (Qlib 原版)

        Parameters
        ----------
        lock_name
            锁名称
        timeout
            超时时间 (秒)
        """
        if not REDIS_AVAILABLE:
            yield
            return

        wlock = redis_lock.Lock(self.client, f"{lock_name}-wlock", id=self.LOCK_ID)
        self._acquire_lock(wlock, lock_name)
        try:
            yield
        finally:
            wlock.release()

    def _acquire_lock(self, lock, lock_name: str):
        """获取锁"""
        try:
            lock.acquire()
        except redis_lock.AlreadyAcquired as e:
            raise CacheException(
                f"Lock {lock_name} is already acquired. "
                f"You can use redis-cli to clear the lock."
            ) from e

    def reset_locks(self):
        """重置所有锁"""
        if REDIS_AVAILABLE:
            redis_lock.reset_all(self.client)


# ============================================================
# 多级缓存 (AlgVex 扩展)
# ============================================================

class MultiLevelCache:
    """
    多级缓存 (AlgVex 扩展)

    Combines L1 (memory), L2 (disk), and L3 (Redis) caches.
    """

    def __init__(
        self,
        mem_cache: Optional[MemCache] = None,
        disk_cache: Optional[DiskCache] = None,
        redis_cache: Optional[RedisCache] = None,
    ):
        """
        初始化多级缓存

        Parameters
        ----------
        mem_cache
            L1 内存缓存
        disk_cache
            L2 磁盘缓存
        redis_cache
            L3 Redis 缓存
        """
        self.l1 = mem_cache
        self.l2 = disk_cache
        self.l3 = redis_cache

    def get(
        self,
        key: str,
        category: str = "general",
        default: Any = None,
    ) -> Any:
        """
        获取缓存 (按级别查找)

        Parameters
        ----------
        key
            缓存键
        category
            缓存类别 (用于内存缓存)
        default
            默认值

        Returns
        -------
        Any
            缓存值或默认值
        """
        # L1: 内存缓存
        if self.l1 is not None:
            value, expired = MemCacheExpire.get_cache(self.l1[category], key)
            if value is not None and not expired:
                return value

        # L2: 磁盘缓存
        if self.l2 is not None:
            value = self.l2.get(key)
            if value is not None:
                # 回填 L1
                if self.l1 is not None:
                    MemCacheExpire.set_cache(self.l1[category], key, value)
                return value

        # L3: Redis 缓存
        if self.l3 is not None:
            value = self.l3.get(key)
            if value is not None:
                # 回填 L1 和 L2
                if self.l1 is not None:
                    MemCacheExpire.set_cache(self.l1[category], key, value)
                if self.l2 is not None:
                    self.l2.set(key, value)
                return value

        return default

    def set(
        self,
        key: str,
        value: Any,
        category: str = "general",
        levels: Tuple[bool, bool, bool] = (True, True, True),
    ):
        """
        设置缓存

        Parameters
        ----------
        key
            缓存键
        value
            缓存值
        category
            缓存类别
        levels
            要写入的级别 (L1, L2, L3)
        """
        l1_enabled, l2_enabled, l3_enabled = levels

        if l1_enabled and self.l1 is not None:
            MemCacheExpire.set_cache(self.l1[category], key, value)

        if l2_enabled and self.l2 is not None:
            self.l2.set(key, value)

        if l3_enabled and self.l3 is not None:
            self.l3.set(key, value)

    def delete(self, key: str, category: str = "general"):
        """删除缓存"""
        if self.l1 is not None and key in self.l1[category]:
            self.l1[category].pop(key)
        if self.l2 is not None:
            self.l2.delete(key)
        if self.l3 is not None:
            self.l3.delete(key)

    def clear(self):
        """清空所有缓存"""
        if self.l1 is not None:
            self.l1.clear()
        if self.l2 is not None:
            self.l2.clear()
        if self.l3 is not None:
            self.l3.clear()


# ============================================================
# 数据集缓存 (Qlib 风格)
# ============================================================

class DatasetCache:
    """
    数据集缓存 (Qlib 风格)

    Dataset cache mechanism for feature data.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "./cache/dataset",
        use_redis: bool = False,
        redis_config: Dict = None,
    ):
        """
        初始化数据集缓存

        Parameters
        ----------
        cache_dir
            缓存目录
        use_redis
            是否使用 Redis
        redis_config
            Redis 配置
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.disk_cache = DiskCache(cache_dir)
        self.redis_cache = None

        if use_redis and REDIS_AVAILABLE:
            redis_config = redis_config or {}
            try:
                self.redis_cache = RedisCache(**redis_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")

    def _make_uri(
        self,
        instruments: Any,
        fields: list,
        start_time: str,
        end_time: str,
        freq: str,
    ) -> str:
        """生成缓存 URI"""
        return hash_args(
            str(instruments),
            str(sorted(fields)),
            str(start_time),
            str(end_time),
            freq,
        )

    def get_dataset(
        self,
        instruments: Any,
        fields: list,
        start_time: str = None,
        end_time: str = None,
        freq: str = "day",
    ) -> Optional[pd.DataFrame]:
        """
        获取缓存的数据集

        Parameters
        ----------
        instruments
            标的列表
        fields
            字段列表
        start_time
            开始时间
        end_time
            结束时间
        freq
            频率

        Returns
        -------
        Optional[pd.DataFrame]
            缓存的数据集或 None
        """
        uri = self._make_uri(instruments, fields, start_time, end_time, freq)

        # 尝试从磁盘缓存获取
        data = self.disk_cache.get(uri)
        if data is not None:
            return data

        # 尝试从 Redis 获取
        if self.redis_cache is not None:
            data = self.redis_cache.get(uri)
            if data is not None:
                # 回填磁盘缓存
                self.disk_cache.set(uri, data)
                return data

        return None

    def set_dataset(
        self,
        data: pd.DataFrame,
        instruments: Any,
        fields: list,
        start_time: str = None,
        end_time: str = None,
        freq: str = "day",
    ):
        """
        缓存数据集

        Parameters
        ----------
        data
            数据集
        instruments
            标的列表
        fields
            字段列表
        start_time
            开始时间
        end_time
            结束时间
        freq
            频率
        """
        uri = self._make_uri(instruments, fields, start_time, end_time, freq)

        # 保存到磁盘
        meta = {
            "instruments": instruments,
            "fields": fields,
            "start_time": start_time,
            "end_time": end_time,
            "freq": freq,
        }
        self.disk_cache.set(uri, data, meta=meta)

        # 保存到 Redis
        if self.redis_cache is not None:
            self.redis_cache.set(uri, data)


# ============================================================
# 全局缓存实例
# ============================================================

# 全局内存缓存
_global_mem_cache: Optional[MemCache] = None


def get_mem_cache(size_limit: int = 1000) -> MemCache:
    """获取全局内存缓存"""
    global _global_mem_cache
    if _global_mem_cache is None:
        _global_mem_cache = MemCache(size_limit=size_limit)
    return _global_mem_cache


def clear_mem_cache():
    """清空全局内存缓存"""
    global _global_mem_cache
    if _global_mem_cache is not None:
        _global_mem_cache.clear()


__all__ = [
    # 异常
    "CacheException",
    # 工具
    "hash_args",
    # 内存缓存
    "MemCacheUnit",
    "MemCacheLengthUnit",
    "MemCacheSizeofUnit",
    "MemCache",
    "MemCacheExpire",
    # 磁盘缓存
    "DiskCache",
    # Redis 缓存
    "RedisCache",
    "REDIS_AVAILABLE",
    # 多级缓存
    "MultiLevelCache",
    # 数据集缓存
    "DatasetCache",
    # 全局函数
    "get_mem_cache",
    "clear_mem_cache",
]
