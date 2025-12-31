"""
P0 验收测试 (Acceptance Tests)

P0 测试确保以下核心功能正常工作:
- Live vs Replay 对齐
- 快照存储/恢复
- 确定性保证
- Qlib 边界隔离
- 生产因子计算 (无 Qlib 依赖)

运行方式:
    pytest tests/p0/ -v --tb=short
"""
