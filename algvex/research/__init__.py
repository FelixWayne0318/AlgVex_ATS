"""
AlgVex Research - 研究环境模块

规则:
- 可以使用 Qlib 和其他研究工具
- 可以访问 180 个研究因子
- 不能直接用于生产
- 验证通过的代码需移至 production/

包含:
- qlib_adapter: Qlib 适配器
- factor_research: 因子研究工具
- backtest_research: 回测研究工具
"""

__version__ = "1.0.0"

# 尝试导入 Qlib（研究环境可用）
try:
    import qlib
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("警告: Qlib 未安装，部分研究功能不可用")

from .qlib_adapter import QlibAdapter
from .factor_research import FactorResearch

__all__ = [
    "QLIB_AVAILABLE",
    "QlibAdapter",
    "FactorResearch",
]
