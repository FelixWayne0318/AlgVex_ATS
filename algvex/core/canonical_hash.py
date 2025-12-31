"""
Canonical Hashing (S9)

功能:
- 计算配置文件的规范化哈希
- 计算数据的规范化哈希
- 计算 trace 的规范化哈希
- 确保跨环境/跨时间的哈希一致性

规范:
- 使用 SHA256 算法
- JSON 序列化时排序所有键
- 浮点数格式化为 8 位精度
- 时间戳使用 ISO8601 UTC 格式

配置: config/hashing_spec.yaml
"""

import hashlib
import json
import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import yaml


logger = logging.getLogger(__name__)


class CanonicalHasher:
    """
    规范化哈希计算器

    使用方法:
        hasher = CanonicalHasher()

        # 计算配置哈希
        config_hash = hasher.compute_hash(config_dict)

        # 计算 DataFrame 哈希
        df_hash = hasher.hash_dataframe(df)

        # 计算因子值哈希
        factor_hash = hasher.hash_factors(factor_dict)
    """

    # 默认配置
    DEFAULT_ALGORITHM = "sha256"
    DEFAULT_OUTPUT_FORMAT = "sha256:{hash_value}"
    DEFAULT_FLOAT_PRECISION = 8

    def __init__(self, config_path: str = "config/hashing_spec.yaml"):
        """
        初始化哈希计算器

        Args:
            config_path: 哈希规范配置文件路径
        """
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """加载配置"""
        config_file = Path(self.config_path)

        if config_file.exists():
            with open(config_file, encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            logger.debug(f"哈希配置文件不存在: {self.config_path}, 使用默认配置")
            config = {}

        global_config = config.get("global_config", {})
        self.algorithm = global_config.get("algorithm", self.DEFAULT_ALGORITHM)
        self.output_format = global_config.get("output_format", self.DEFAULT_OUTPUT_FORMAT)

        # 浮点数处理配置
        float_handling = config.get("data_hash_spec", {}).get("dataframe", {}).get("float_handling", {})
        self.float_precision = float_handling.get("precision", self.DEFAULT_FLOAT_PRECISION)
        self.strip_trailing_zeros = float_handling.get("strip_trailing_zeros", True)

    def _normalize_value(self, value: Any) -> Any:
        """
        规范化值以确保哈希一致性

        Args:
            value: 任意值

        Returns:
            规范化后的值
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, np.integer)):
            return int(value)

        if isinstance(value, (float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return str(value)  # "nan", "inf", "-inf"
            # 格式化为指定精度
            formatted = f"{value:.{self.float_precision}f}"
            if self.strip_trailing_zeros:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted

        if isinstance(value, Decimal):
            return str(value)

        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if isinstance(value, date):
            return value.isoformat()

        if isinstance(value, (set, frozenset)):
            return sorted(self._normalize_value(v) for v in value)

        if isinstance(value, (list, tuple)):
            return [self._normalize_value(v) for v in value]

        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in sorted(value.items())}

        if isinstance(value, np.ndarray):
            return self._normalize_value(value.tolist())

        if isinstance(value, bytes):
            return value.hex()

        if isinstance(value, Path):
            return str(value)

        # 其他类型转为字符串
        return str(value)

    def _compute_sha256(self, data: bytes) -> str:
        """计算 SHA256 哈希"""
        return hashlib.sha256(data).hexdigest()

    def compute_hash(
        self,
        data: Dict[str, Any],
        exclude_keys: Optional[Set[str]] = None,
        truncate: int = 16,
    ) -> str:
        """
        计算配置/数据字典的规范化哈希

        Args:
            data: 数据字典
            exclude_keys: 要排除的键 (如 "config_hash")
            truncate: 哈希截断长度 (默认 16)

        Returns:
            格式化的哈希字符串 (如 "sha256:abc123...")
        """
        # 复制数据以避免修改原始数据
        data_copy = dict(data) if hasattr(data, 'items') else data

        # 排除指定的键
        if exclude_keys is None:
            exclude_keys = {"config_hash"}

        for key in exclude_keys:
            data_copy.pop(key, None)

        # 规范化数据
        normalized = self._normalize_value(data_copy)

        # JSON 序列化 (排序键，紧凑格式)
        json_str = json.dumps(
            normalized,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        )

        # 计算哈希
        hash_value = self._compute_sha256(json_str.encode('utf-8'))

        # 截断并格式化
        if truncate > 0:
            hash_value = hash_value[:truncate]

        return f"sha256:{hash_value}"

    def hash_dataframe(self, df, include_index: bool = True) -> str:
        """
        计算 DataFrame 的哈希

        Args:
            df: pandas DataFrame
            include_index: 是否包含索引

        Returns:
            格式化的哈希字符串
        """
        # 按列名排序
        df_sorted = df.reindex(sorted(df.columns), axis=1)

        # 按索引排序
        if include_index:
            df_sorted = df_sorted.sort_index()

        # 转换为 CSV 字符串
        csv_str = df_sorted.to_csv(index=include_index)
        hash_value = self._compute_sha256(csv_str.encode('utf-8'))[:16]

        return f"sha256:{hash_value}"

    def hash_factors(self, factors: Dict[str, float]) -> str:
        """
        计算因子值字典的哈希

        Args:
            factors: 因子名 -> 值 的字典

        Returns:
            格式化的哈希字符串
        """
        return self.compute_hash(factors, exclude_keys=set())

    def hash_file(self, filepath: Union[str, Path]) -> str:
        """
        计算文件的哈希

        Args:
            filepath: 文件路径

        Returns:
            格式化的哈希字符串
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'rb') as f:
            content = f.read()

        hash_value = self._compute_sha256(content)[:16]
        return f"sha256:{hash_value}"

    def hash_files(self, filepaths: List[Union[str, Path]]) -> str:
        """
        计算多个文件的联合哈希

        Args:
            filepaths: 文件路径列表

        Returns:
            格式化的哈希字符串
        """
        combined = []
        for filepath in sorted(str(p) for p in filepaths):
            filepath = Path(filepath)
            if filepath.exists():
                file_hash = self.hash_file(filepath)
                combined.append(f"{filepath.name}:{file_hash}")

        combined_str = "\n".join(combined)
        hash_value = self._compute_sha256(combined_str.encode('utf-8'))[:16]

        return f"sha256:{hash_value}"

    def verify_hash(
        self,
        data: Dict[str, Any],
        expected_hash: str,
        exclude_keys: Optional[Set[str]] = None,
    ) -> bool:
        """
        验证哈希是否匹配

        Args:
            data: 数据字典
            expected_hash: 期望的哈希值
            exclude_keys: 要排除的键

        Returns:
            True 如果匹配
        """
        actual_hash = self.compute_hash(data, exclude_keys)
        return actual_hash == expected_hash


def compute_contract_hash() -> str:
    """
    计算所有契约配置的联合哈希

    用于确保 Live 和 Replay 使用相同的契约定义
    """
    hasher = CanonicalHasher()
    contract_files = [
        "config/visibility.yaml",
        "config/data_contracts/klines_5m.yaml",
        "config/data_contracts/open_interest_5m.yaml",
        "config/data_contracts/funding_8h.yaml",
    ]

    # 过滤存在的文件
    existing_files = [f for f in contract_files if Path(f).exists()]

    if not existing_files:
        logger.warning("没有找到契约配置文件")
        return "sha256:no_contracts"

    return hasher.hash_files(existing_files)


# 测试代码
if __name__ == "__main__":
    hasher = CanonicalHasher()

    # 测试配置哈希
    test_config = {
        "config_version": "1.0.0",
        "config_hash": "will_be_excluded",
        "settings": {
            "enabled": True,
            "threshold": 0.001234567890123,
            "items": ["a", "b", "c"],
        }
    }

    hash_result = hasher.compute_hash(test_config)
    print(f"配置哈希: {hash_result}")

    # 测试因子哈希
    factors = {
        "return_5m": 0.0123456789,
        "atr_288": 0.0234567890,
        "oi_change_rate": -0.0012345678,
    }
    factor_hash = hasher.hash_factors(factors)
    print(f"因子哈希: {factor_hash}")

    # 测试哈希验证
    verified = hasher.verify_hash(test_config, hash_result)
    print(f"哈希验证: {'通过' if verified else '失败'}")
