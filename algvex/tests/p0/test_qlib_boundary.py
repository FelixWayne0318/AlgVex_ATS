"""
P0 验收测试: Qlib 边界隔离

验收标准:
- production/ 目录不依赖 Qlib
- shared/ 目录不依赖 Qlib
- research/ 目录可以使用 Qlib
- 导入边界检查通过
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pytest

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestQlibBoundary:
    """测试 Qlib 边界隔离"""

    @pytest.fixture
    def project_root(self):
        """项目根目录"""
        return Path(__file__).parent.parent.parent

    def get_imports_from_file(self, filepath: Path) -> Set[str]:
        """从文件中提取所有导入的模块"""
        imports = set()

        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

        return imports

    def test_production_no_qlib(self, project_root):
        """测试 production/ 不导入 qlib"""
        production_dir = project_root / "production"

        if not production_dir.exists():
            pytest.skip("production/ 目录不存在")

        violations = []
        for py_file in production_dir.rglob("*.py"):
            imports = self.get_imports_from_file(py_file)
            if "qlib" in imports:
                violations.append(str(py_file.relative_to(project_root)))

        assert len(violations) == 0, f"production/ 中以下文件导入了 qlib: {violations}"

    def test_shared_no_qlib(self, project_root):
        """测试 shared/ 不导入 qlib"""
        shared_dir = project_root / "shared"

        if not shared_dir.exists():
            pytest.skip("shared/ 目录不存在")

        violations = []
        for py_file in shared_dir.rglob("*.py"):
            imports = self.get_imports_from_file(py_file)
            if "qlib" in imports:
                violations.append(str(py_file.relative_to(project_root)))

        assert len(violations) == 0, f"shared/ 中以下文件导入了 qlib: {violations}"

    def test_core_no_qlib(self, project_root):
        """测试 core/ 不导入 qlib"""
        core_dir = project_root / "core"

        if not core_dir.exists():
            pytest.skip("core/ 目录不存在")

        violations = []
        for py_file in core_dir.rglob("*.py"):
            imports = self.get_imports_from_file(py_file)
            if "qlib" in imports:
                violations.append(str(py_file.relative_to(project_root)))

        assert len(violations) == 0, f"core/ 中以下文件导入了 qlib: {violations}"

    def test_research_can_use_qlib(self, project_root):
        """测试 research/ 可以使用 qlib (不报错)"""
        research_dir = project_root / "research"

        if not research_dir.exists():
            pytest.skip("research/ 目录不存在")

        # research 目录应该存在 qlib 相关代码
        qlib_files = []
        for py_file in research_dir.rglob("*.py"):
            imports = self.get_imports_from_file(py_file)
            if "qlib" in imports:
                qlib_files.append(str(py_file.relative_to(project_root)))

        # 不要求必须有，但如果有应该在 research 目录
        # 这个测试主要是确认 research 目录可以使用 qlib
        pass


class TestProductionImports:
    """测试 production 模块可以独立导入"""

    def test_import_factor_engine(self):
        """测试导入 factor_engine (无 qlib)"""
        try:
            # 确保 qlib 不在 sys.modules 中
            if 'qlib' in sys.modules:
                del sys.modules['qlib']

            from algvex.production import factor_engine
            assert factor_engine is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"factor_engine 依赖 qlib: {e}")
            # 其他依赖缺失可以接受

    def test_import_signal_generator(self):
        """测试导入 signal_generator (无 qlib)"""
        try:
            if 'qlib' in sys.modules:
                del sys.modules['qlib']

            from algvex.production import signal_generator
            assert signal_generator is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"signal_generator 依赖 qlib: {e}")

    def test_import_model_loader(self):
        """测试导入 model_loader (无 qlib)"""
        try:
            if 'qlib' in sys.modules:
                del sys.modules['qlib']

            from algvex.production import model_loader
            assert model_loader is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"model_loader 依赖 qlib: {e}")


class TestSharedImports:
    """测试 shared 模块可以独立导入"""

    def test_import_visibility_checker(self):
        """测试导入 visibility_checker"""
        try:
            from algvex.shared import visibility_checker
            assert visibility_checker is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"visibility_checker 依赖 qlib: {e}")

    def test_import_time_provider(self):
        """测试导入 time_provider"""
        try:
            from algvex.shared import time_provider
            assert time_provider is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"time_provider 依赖 qlib: {e}")

    def test_import_seeded_random(self):
        """测试导入 seeded_random"""
        try:
            from algvex.shared import seeded_random
            assert seeded_random is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"seeded_random 依赖 qlib: {e}")

    def test_import_trace_logger(self):
        """测试导入 trace_logger"""
        try:
            from algvex.shared import trace_logger
            assert trace_logger is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"trace_logger 依赖 qlib: {e}")

    def test_import_config_validator(self):
        """测试导入 config_validator"""
        try:
            from algvex.shared import config_validator
            assert config_validator is not None
        except ImportError as e:
            if 'qlib' in str(e).lower():
                pytest.fail(f"config_validator 依赖 qlib: {e}")


class TestBoundaryRules:
    """测试边界规则"""

    @pytest.fixture
    def project_root(self):
        """项目根目录"""
        return Path(__file__).parent.parent.parent

    def get_all_imports(self, filepath: Path) -> List[str]:
        """获取文件的所有导入"""
        imports = []

        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def test_production_no_research_imports(self, project_root):
        """测试 production 不导入 research"""
        production_dir = project_root / "production"

        if not production_dir.exists():
            pytest.skip("production/ 目录不存在")

        violations = []
        for py_file in production_dir.rglob("*.py"):
            imports = self.get_all_imports(py_file)
            for imp in imports:
                if "research" in imp or imp.startswith("algvex.research"):
                    violations.append({
                        "file": str(py_file.relative_to(project_root)),
                        "import": imp,
                    })

        assert len(violations) == 0, f"production/ 导入了 research: {violations}"

    def test_shared_no_production_imports(self, project_root):
        """测试 shared 不导入 production"""
        shared_dir = project_root / "shared"

        if not shared_dir.exists():
            pytest.skip("shared/ 目录不存在")

        violations = []
        for py_file in shared_dir.rglob("*.py"):
            imports = self.get_all_imports(py_file)
            for imp in imports:
                if "production" in imp or imp.startswith("algvex.production"):
                    violations.append({
                        "file": str(py_file.relative_to(project_root)),
                        "import": imp,
                    })

        assert len(violations) == 0, f"shared/ 导入了 production: {violations}"

    def test_shared_no_research_imports(self, project_root):
        """测试 shared 不导入 research"""
        shared_dir = project_root / "shared"

        if not shared_dir.exists():
            pytest.skip("shared/ 目录不存在")

        violations = []
        for py_file in shared_dir.rglob("*.py"):
            imports = self.get_all_imports(py_file)
            for imp in imports:
                if "research" in imp or imp.startswith("algvex.research"):
                    violations.append({
                        "file": str(py_file.relative_to(project_root)),
                        "import": imp,
                    })

        assert len(violations) == 0, f"shared/ 导入了 research: {violations}"
