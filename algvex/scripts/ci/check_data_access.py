#!/usr/bin/env python3
"""
CI门禁: 检查非法数据访问

规则:
1. 禁止直接 import psycopg2/redis/sqlalchemy (除了 infrastructure/ 和 shared/data_service.py)
2. 禁止直接使用 open() 读取数据文件 (除了配置文件)
3. 所有数据访问必须通过 DataService 接口

用法:
    python scripts/ci/check_data_access.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


# 直接数据库访问模块 (禁止在业务代码中直接导入)
FORBIDDEN_DB_IMPORTS = [
    "psycopg2",
    "redis",
    "sqlalchemy",
    "pymongo",
    "motor",
    "asyncpg",
]

# 允许直接访问数据库的目录/文件
ALLOWED_DB_ACCESS_PATHS = [
    "algvex/infrastructure/",
    "algvex/shared/data_service.py",
    "algvex/api/database.py",
    "algvex/core/data/",
    "tests/",
]

# 禁止的文件操作模式 (数据文件读取)
FORBIDDEN_FILE_PATTERNS = [
    r'open\s*\([^)]*\.parquet',
    r'open\s*\([^)]*\.csv',
    r'open\s*\([^)]*\.h5',
    r'open\s*\([^)]*\.feather',
    r'pd\.read_parquet\s*\(',
    r'pd\.read_csv\s*\(',
    r'pd\.read_hdf\s*\(',
]

# 允许直接文件访问的路径
ALLOWED_FILE_ACCESS_PATHS = [
    "algvex/shared/data_service.py",
    "algvex/core/data/",
    "algvex/infrastructure/",
    "scripts/",
    "tests/",
]


class DataAccessChecker:
    """数据访问检查器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.violations: List[Tuple[str, int, str, str]] = []

    def is_allowed_path(self, filepath: str, allowed_paths: List[str]) -> bool:
        """检查文件是否在允许的路径中"""
        for allowed in allowed_paths:
            if allowed in str(filepath):
                return True
        return False

    def check_db_imports(self, filepath: Path) -> List[Tuple[int, str, str]]:
        """检查直接数据库导入"""
        violations = []

        if self.is_allowed_path(str(filepath), ALLOWED_DB_ACCESS_PATHS):
            return violations

        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_DB_IMPORTS:
                        if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                            violations.append((
                                node.lineno,
                                f"import {alias.name}",
                                f"禁止直接导入数据库模块 {forbidden}"
                            ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in FORBIDDEN_DB_IMPORTS:
                        if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                            names = ", ".join(a.name for a in node.names)
                            violations.append((
                                node.lineno,
                                f"from {node.module} import {names}",
                                f"禁止直接导入数据库模块 {forbidden}"
                            ))

        return violations

    def check_file_access(self, filepath: Path) -> List[Tuple[int, str, str]]:
        """检查直接文件数据访问"""
        violations = []

        if self.is_allowed_path(str(filepath), ALLOWED_FILE_ACCESS_PATHS):
            return violations

        try:
            with open(filepath, encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return violations

        for line_no, line in enumerate(lines, 1):
            for pattern in FORBIDDEN_FILE_PATTERNS:
                if re.search(pattern, line):
                    violations.append((
                        line_no,
                        line.strip()[:80],
                        "禁止直接读取数据文件，请使用 DataService"
                    ))
                    break

        return violations

    def check_directory(self, directory: str) -> int:
        """检查目录下所有Python文件"""
        dir_path = self.base_dir / directory
        if not dir_path.exists():
            return 0

        violation_count = 0
        for py_file in dir_path.rglob("*.py"):
            rel_path = py_file.relative_to(self.base_dir)

            # 检查数据库导入
            db_violations = self.check_db_imports(py_file)
            for line_no, code, reason in db_violations:
                self.violations.append((str(rel_path), line_no, code, reason))
                violation_count += 1

            # 检查文件访问
            file_violations = self.check_file_access(py_file)
            for line_no, code, reason in file_violations:
                self.violations.append((str(rel_path), line_no, code, reason))
                violation_count += 1

        return violation_count

    def run(self) -> bool:
        """运行所有检查"""
        print("=" * 60)
        print("       AlgVex 数据访问检查 (Data Access Check)")
        print("=" * 60)
        print()
        print("规则:")
        print("  1. 禁止在业务代码中直接导入数据库模块")
        print("  2. 禁止直接读取数据文件 (parquet/csv/h5)")
        print("  3. 所有数据访问必须通过 DataService 接口")
        print()

        # 检查的目录 (相对于 algvex 根目录)
        directories_to_check = [
            "production",
            "research",
            "api/routers",
            "api/services",
            "core/strategy",
            "core/model",
        ]

        total_violations = 0
        for directory in directories_to_check:
            print(f"检查 {directory}/...")
            count = self.check_directory(directory)
            total_violations += count

            if count == 0:
                print(f"  ✅ 通过")
            else:
                print(f"  ❌ 发现 {count} 个违规")

        print()

        # 输出详细违规信息
        if self.violations:
            print("=" * 60)
            print("违规详情:")
            print("=" * 60)
            for filepath, line_no, code, reason in self.violations:
                print(f"  {filepath}:{line_no}")
                print(f"    代码: {code}")
                print(f"    原因: {reason}")
                print()

        # 最终结果
        print("=" * 60)
        if total_violations == 0:
            print("✅ 数据访问检查通过! 所有数据访问符合规范。")
            return True
        else:
            print(f"❌ 数据访问检查失败! 发现 {total_violations} 个违规。")
            print("请通过 DataService 接口访问数据。")
            return False


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    # script_dir = algvex/scripts/ci/
    # algvex_root = algvex/
    algvex_root = script_dir.parent.parent

    # 检查 algvex 根目录
    if not algvex_root.exists():
        print(f"错误: 找不到 algvex 目录")
        sys.exit(1)

    print(f"AlgVex 根目录: {algvex_root}")
    print()

    checker = DataAccessChecker(algvex_root)
    success = checker.run()

    # 如果没有检查到任何目录 (可能是 CI 环境), 也算通过
    if not success and len(checker.violations) == 0:
        print("注意: 未找到需要检查的目录，跳过检查")
        sys.exit(0)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
