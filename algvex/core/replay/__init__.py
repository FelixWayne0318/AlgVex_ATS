"""
AlgVex Replay - 回放与对齐模块

功能:
- 使用历史快照重放信号
- 验证 Live vs Replay 一致性
- 确保系统确定性

包含:
- ReplayRunner: 重放运行器
- AlignmentChecker: 对齐检查器
"""

from .replay_runner import ReplayRunner, ReplayResult, AlignmentResult

__all__ = [
    "ReplayRunner",
    "ReplayResult",
    "AlignmentResult",
]
