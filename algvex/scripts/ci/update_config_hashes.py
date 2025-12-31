#!/usr/bin/env python3
"""
配置哈希自动更新脚本

功能:
- 遍历所有配置文件
- 计算规范化哈希
- 更新 config_hash 字段
- 保留注释和格式 (使用 ruamel.yaml)

关键: 使用 ruamel.yaml RoundTrip 模式保留注释和格式!
- 读: ruamel.yaml RoundTripLoader
- 写: ruamel.yaml RoundTripDumper
- hash 计算: 基于解析后对象的 canonical JSON (不依赖 YAML 文本)

用法:
    python scripts/ci/update_config_hashes.py

    # 指定配置目录
    python scripts/ci/update_config_hashes.py --config-dir config/

    # 仅检查不更新
    python scripts/ci/update_config_hashes.py --check-only
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


# 添加项目路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))


def normalize_value(value: Any) -> Any:
    """规范化值以确保哈希一致性"""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        # 格式化为 8 位精度
        formatted = f"{value:.8f}"
        return formatted.rstrip('0').rstrip('.')

    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]

    if isinstance(value, dict):
        # 递归处理并排序键
        return {str(k): normalize_value(v) for k, v in sorted(value.items())}

    return str(value)


def compute_config_hash(config: Dict[str, Any], exclude_key: str = "config_hash") -> str:
    """
    计算配置的规范化哈希

    Args:
        config: 配置字典
        exclude_key: 要排除的键

    Returns:
        格式化的哈希字符串
    """
    # 转换为普通 dict (ruamel 返回的是 CommentedMap)
    config_dict = dict(config) if hasattr(config, 'items') else config

    # 排除 config_hash 字段
    config_copy = {k: v for k, v in config_dict.items() if k != exclude_key}

    # 规范化
    normalized = normalize_value(config_copy)

    # JSON 序列化 (排序键，紧凑格式)
    json_str = json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )

    # 计算 SHA256
    hash_value = hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]

    return f"sha256:{hash_value}"


def update_all_config_hashes(
    config_dir: str = "config",
    check_only: bool = False,
    verbose: bool = True,
) -> int:
    """
    更新所有配置文件的哈希 (保留注释和格式)

    Args:
        config_dir: 配置目录
        check_only: 仅检查不更新
        verbose: 输出详细信息

    Returns:
        更新的文件数量
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        print("错误: 需要安装 ruamel.yaml")
        print("  pip install ruamel.yaml")
        sys.exit(1)

    config_path = Path(config_dir)

    if not config_path.exists():
        print(f"错误: 配置目录不存在: {config_path}")
        return 0

    # 使用 ruamel.yaml RoundTrip 模式 (保留注释/顺序/格式)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    updated_count = 0
    checked_count = 0
    mismatched = []

    for yaml_file in sorted(config_path.rglob("*.yaml")):
        checked_count += 1

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.load(f)
        except Exception as e:
            if verbose:
                print(f"跳过 {yaml_file}: 无法解析 ({e})")
            continue

        if config is None:
            continue

        # 计算新哈希 (基于解析后的 dict, 不是 YAML 文本)
        new_hash = compute_config_hash(config)
        old_hash = config.get("config_hash", "")

        # 幂等检查 - 只在哈希真正变化时更新
        if old_hash == new_hash:
            if verbose:
                print(f"  {yaml_file.name}: 无变化")
            continue

        # 记录不匹配
        mismatched.append({
            "file": str(yaml_file),
            "old_hash": old_hash,
            "new_hash": new_hash,
        })

        if check_only:
            print(f"  {yaml_file.name}: 需要更新")
            print(f"    旧: {old_hash}")
            print(f"    新: {new_hash}")
            continue

        # 更新哈希 (只改这一个字段，保留其他一切)
        config["config_hash"] = new_hash

        # 写回 (ruamel 保留注释/顺序/格式，PR diff 最小化)
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        old_display = f"{old_hash[:20]}..." if old_hash else "None"
        print(f"  更新 {yaml_file.name}: {old_display} -> {new_hash}")
        updated_count += 1

    # 输出摘要
    print()
    if check_only:
        if mismatched:
            print(f"检查结果: {len(mismatched)} 个文件需要更新哈希")
            return len(mismatched)
        else:
            print("检查结果: 所有配置哈希已是最新")
            return 0
    else:
        if updated_count == 0:
            print("所有配置哈希已是最新，无需更新")
        else:
            print(f"共更新 {updated_count} 个配置文件")
        return updated_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AlgVex 配置哈希更新工具",
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=str,
        default="config",
        help="配置目录 (默认: config)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查不更新"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("        AlgVex 配置哈希更新工具")
    print("=" * 60)
    print()

    # 切换到项目根目录
    import os
    os.chdir(project_root)
    print(f"项目目录: {project_root}")
    print(f"配置目录: {args.config_dir}")
    print()

    count = update_all_config_hashes(
        config_dir=args.config_dir,
        check_only=args.check_only,
        verbose=not args.quiet,
    )

    # 返回码
    if args.check_only:
        # 检查模式: 有需要更新的文件返回 1
        sys.exit(1 if count > 0 else 0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
