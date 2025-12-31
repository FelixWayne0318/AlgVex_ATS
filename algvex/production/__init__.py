"""
AlgVex Production - 生产环境模块

规则:
- 只包含 MVP 验证过的代码
- 不依赖 Qlib（Qlib 仅用于研究）
- 不允许导入 research/ 目录的代码
- 所有因子/模型必须经过验证

包含:
- factor_engine: MVP 因子计算引擎
- model_loader: 模型加载器（从 Qlib 导出的权重）
- signal_generator: 信号生成器
"""

__version__ = "1.0.0"

# 边界检查：禁止导入 research 模块
def _check_import_boundary():
    """检查导入边界"""
    import sys
    for module_name in sys.modules:
        if "algvex.research" in module_name:
            raise ImportError(
                f"生产环境不允许导入研究模块: {module_name}\n"
                "请将验证过的代码移至 production/ 目录"
            )

# 导出
from .factor_engine import MVPFactorEngine
from .model_loader import ModelLoader
from .signal_generator import SignalGenerator

__all__ = [
    "MVPFactorEngine",
    "ModelLoader",
    "SignalGenerator",
]
