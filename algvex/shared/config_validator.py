"""
AlgVex 配置验证器

功能:
- 加载和解析YAML配置文件
- 计算配置文件的规范化哈希
- 验证配置版本和完整性
- 运行时配置校验

规范化哈希 (Canonical Hashing):
- 将YAML转为规范化JSON（排序键、去除空白）
- 计算SHA256哈希
- 保证相同内容产生相同哈希
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigValidator:
    """配置验证器"""

    def __init__(self, config_dir: str = "config"):
        """
        初始化配置验证器

        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self._loaded_configs: Dict[str, Dict] = {}
        self._config_hashes: Dict[str, str] = {}

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_name: 配置文件名（不含.yaml后缀）

        Returns:
            配置字典
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise ConfigValidationError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 缓存配置
        self._loaded_configs[config_name] = config

        # 计算并缓存哈希
        self._config_hashes[config_name] = self.compute_hash(config)

        return config

    def load_data_contract(self, source_id: str) -> Dict[str, Any]:
        """
        加载数据契约配置

        Args:
            source_id: 数据源ID（如 klines_5m）

        Returns:
            数据契约配置
        """
        contract_path = self.config_dir / "data_contracts" / f"{source_id}.yaml"

        if not contract_path.exists():
            raise ConfigValidationError(f"数据契约不存在: {contract_path}")

        with open(contract_path, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        return contract

    @staticmethod
    def compute_hash(config: Dict[str, Any]) -> str:
        """
        计算配置的规范化哈希

        规范化规则:
        1. 移除 config_hash 字段（避免循环）
        2. 将字典转为排序的JSON字符串
        3. 计算SHA256哈希

        Args:
            config: 配置字典

        Returns:
            SHA256哈希字符串
        """
        # 复制配置，移除hash字段
        config_copy = config.copy()
        config_copy.pop("config_hash", None)

        # 规范化JSON（排序键，无缩进）
        canonical_json = json.dumps(
            config_copy,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # 计算SHA256
        hash_value = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return f"sha256:{hash_value[:16]}"  # 取前16位

    def validate_hash(self, config_name: str) -> Tuple[bool, str]:
        """
        验证配置文件的哈希值

        Args:
            config_name: 配置文件名

        Returns:
            (是否通过, 消息)
        """
        if config_name not in self._loaded_configs:
            self.load_config(config_name)

        config = self._loaded_configs[config_name]
        stored_hash = config.get("config_hash", "")
        computed_hash = self._config_hashes[config_name]

        if not stored_hash:
            return True, f"配置 {config_name} 无存储哈希，计算哈希: {computed_hash}"

        if stored_hash == computed_hash:
            return True, f"配置 {config_name} 哈希验证通过"
        else:
            return False, (
                f"配置 {config_name} 哈希不匹配!\n"
                f"  存储: {stored_hash}\n"
                f"  计算: {computed_hash}"
            )

    def validate_all(self) -> Dict[str, Tuple[bool, str]]:
        """
        验证所有配置文件

        Returns:
            {配置名: (是否通过, 消息)}
        """
        results = {}

        # 验证主配置文件
        for yaml_file in self.config_dir.glob("*.yaml"):
            config_name = yaml_file.stem
            try:
                results[config_name] = self.validate_hash(config_name)
            except Exception as e:
                results[config_name] = (False, f"验证失败: {e}")

        # 验证数据契约
        contracts_dir = self.config_dir / "data_contracts"
        if contracts_dir.exists():
            for yaml_file in contracts_dir.glob("*.yaml"):
                config_name = f"data_contracts/{yaml_file.stem}"
                try:
                    contract = self.load_data_contract(yaml_file.stem)
                    stored_hash = contract.get("contract_hash", "")
                    computed_hash = self.compute_hash(contract)

                    if not stored_hash:
                        results[config_name] = (
                            True,
                            f"无存储哈希，计算: {computed_hash}",
                        )
                    elif stored_hash == computed_hash:
                        results[config_name] = (True, "哈希验证通过")
                    else:
                        results[config_name] = (False, f"哈希不匹配")
                except Exception as e:
                    results[config_name] = (False, f"验证失败: {e}")

        return results

    def get_config_version(self, config_name: str) -> str:
        """获取配置版本"""
        if config_name not in self._loaded_configs:
            self.load_config(config_name)

        return self._loaded_configs[config_name].get("config_version", "unknown")

    def get_all_hashes(self) -> Dict[str, str]:
        """获取所有已加载配置的哈希"""
        return self._config_hashes.copy()

    def update_config_hash(self, config_name: str) -> str:
        """
        更新配置文件中的哈希值

        Args:
            config_name: 配置文件名

        Returns:
            新的哈希值
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise ConfigValidationError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        config = yaml.safe_load(content)
        new_hash = self.compute_hash(config)

        # 更新哈希值
        if "config_hash" in config:
            content = content.replace(
                f'config_hash: "{config["config_hash"]}"',
                f'config_hash: "{new_hash}"',
            )
        else:
            # 在 config_version 后添加
            content = content.replace(
                f'config_version: "{config.get("config_version", "1.0.0")}"',
                f'config_version: "{config.get("config_version", "1.0.0")}"\nconfig_hash: "{new_hash}"',
            )

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)

        return new_hash


def validate_config_hash(config_dir: str = "config") -> bool:
    """
    验证所有配置文件的哈希值

    Args:
        config_dir: 配置目录

    Returns:
        是否全部通过

    Raises:
        ConfigValidationError: 如果验证失败
    """
    validator = ConfigValidator(config_dir)
    results = validator.validate_all()

    all_passed = True
    for name, (passed, message) in results.items():
        if passed:
            print(f"✅ {name}: {message}")
        else:
            print(f"❌ {name}: {message}")
            all_passed = False

    return all_passed


# 命令行入口
if __name__ == "__main__":
    import sys

    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    success = validate_config_hash(config_dir)
    sys.exit(0 if success else 1)
