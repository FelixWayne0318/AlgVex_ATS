#!/usr/bin/env python3
"""
CI门禁: 检查 production/ 的非法导入

规则:
1. production/ 禁止 import qlib
2. production/ 禁止 import research/
3. shared/ 禁止 import production/ 或 research/

用法:
    python scripts/ci/check_import_boundary.py
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


# 边界规则定义
BOUNDARY_RULES = {
    "algvex/production": {
        "forbidden": [
            "qlib",
            "algvex.research",
        ],
        "description": "生产代码禁止依赖 Qlib 和研究代码",
    },
    "algvex/shared": {
        "forbidden": [
            "algvex.production",
            "algvex.research",
        ],
        "description": "共享代码禁止依赖生产和研究代码 (防止循环依赖)",
    },
}

# 允许的例外 (如果有的话)
ALLOWED_EXCEPTIONS = [
    # ("algvex/production/some_file.py", "qlib.utils"),  # 示例
]


class ImportBoundaryChecker:
    """导入边界检查器"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.violations: List[Tuple[str, int, str, str]] = []

    def check_file(self, filepath: Path, forbidden_imports: List[str]) -> List[Tuple[str, int, str]]:
        """
        检查单个文件的非法导入

        Returns:
            List of (line_number, import_statement, forbidden_package)
        """
        violations = []

        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"  警告: 无法解析 {filepath}: {e}")
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_imports:
                        if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                            violations.append((node.lineno, f"import {alias.name}", forbidden))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in forbidden_imports:
                        if node.module == forbidden or node.module.startswith(f"{forbidden}."):
                            names = ", ".join(a.name for a in node.names)
                            violations.append((node.lineno, f"from {node.module} import {names}", forbidden))

        return violations

    def is_exception(self, filepath: str, forbidden: str) -> bool:
        """检查是否为允许的例外"""
        rel_path = str(filepath)
        for exc_path, exc_import in ALLOWED_EXCEPTIONS:
            if exc_path in rel_path and forbidden.startswith(exc_import.split('.')[0]):
                return True
        return False

    def check_directory(self, directory: str, forbidden_imports: List[str]) -> int:
        """检查目录下所有Python文件"""
        dir_path = self.base_dir / directory
        if not dir_path.exists():
            print(f"  目录不存在: {directory}")
            return 0

        violation_count = 0
        for py_file in dir_path.rglob("*.py"):
            rel_path = py_file.relative_to(self.base_dir)
            file_violations = self.check_file(py_file, forbidden_imports)

            for line_no, import_stmt, forbidden in file_violations:
                if not self.is_exception(str(rel_path), forbidden):
                    self.violations.append((str(rel_path), line_no, import_stmt, forbidden))
                    violation_count += 1

        return violation_count

    def run(self) -> bool:
        """运行所有边界检查"""
        print("=" * 60)
        print("       AlgVex 导入边界检查 (Import Boundary Check)")
        print("=" * 60)
        print()

        total_violations = 0

        for directory, rules in BOUNDARY_RULES.items():
            print(f"检查 {directory}/")
            print(f"  规则: {rules['description']}")
            print(f"  禁止: {rules['forbidden']}")

            count = self.check_directory(directory, rules["forbidden"])
            total_violations += count

            if count == 0:
                print(f"  结果: ✅ 通过")
            else:
                print(f"  结果: ❌ 发现 {count} 个违规")
            print()

        # 输出详细违规信息
        if self.violations:
            print("=" * 60)
            print("违规详情:")
            print("=" * 60)
            for filepath, line_no, import_stmt, forbidden in self.violations:
                print(f"  {filepath}:{line_no}")
                print(f"    {import_stmt}")
                print(f"    违反规则: 禁止导入 {forbidden}")
                print()

        # 最终结果
        print("=" * 60)
        if total_violations == 0:
            print("✅ 边界检查通过! 所有导入符合规范。")
            return True
        else:
            print(f"❌ 边界检查失败! 发现 {total_violations} 个违规。")
            print("请修复以上违规后重新提交。")
            return False


def main():
    """主函数"""
    # 查找项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # scripts/ci -> scripts -> algvex

    if not (project_root / "production").exists():
        # 可能从 qlib 根目录运行
        project_root = project_root / "algvex"

    if not (project_root / "production").exists():
        print(f"错误: 找不到 production/ 目录")
        print(f"当前搜索路径: {project_root}")
        sys.exit(1)

    print(f"项目根目录: {project_root}")
    print()

    checker = ImportBoundaryChecker(project_root)
    success = checker.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
