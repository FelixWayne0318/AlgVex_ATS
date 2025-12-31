#!/usr/bin/env python3
"""
AlgVex 每日对齐检查脚本

功能:
- 每日自动运行 Replay 对齐检查
- 比对 Live 输出与 Replay 结果
- 生成对齐报告
- 差异超阈值时告警

使用方式:
    # 检查昨天的对齐情况
    python scripts/daily_alignment.py

    # 检查指定日期
    python scripts/daily_alignment.py --date 2024-01-15

    # 连续检查多天
    python scripts/daily_alignment.py --start 2024-01-10 --end 2024-01-15
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.replay.replay_runner import ReplayRunner, AlignmentResult


def run_daily_check(
    date: str,
    tolerance: float = 0.001,
    snapshot_dir: str = "data/snapshots",
    live_dir: str = "data/live_outputs",
    output_dir: str = "data/replay_outputs",
) -> AlignmentResult:
    """
    运行每日对齐检查

    Args:
        date: 日期 (YYYY-MM-DD)
        tolerance: 容差
        snapshot_dir: 快照目录
        live_dir: 实盘输出目录
        output_dir: 重放输出目录

    Returns:
        对齐检查结果
    """
    print(f"\n{'='*60}")
    print(f"📅 日期: {date}")
    print(f"{'='*60}")

    runner = ReplayRunner(
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
    )

    result = runner.daily_replay_check(
        date=date,
        live_outputs_dir=live_dir,
        tolerance=tolerance,
    )

    print(result.summary)

    if not result.aligned:
        print("\n❌ 检测到对齐问题:")
        for i, mismatch in enumerate(result.mismatches[:10]):  # 只显示前10个
            print(f"  [{i+1}] {mismatch}")
        if len(result.mismatches) > 10:
            print(f"  ... 还有 {len(result.mismatches) - 10} 个问题")

    return result


def run_range_check(
    start_date: str,
    end_date: str,
    tolerance: float = 0.001,
) -> dict:
    """检查日期范围内的对齐情况"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    results = {}
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            result = run_daily_check(date_str, tolerance)
            results[date_str] = {
                "aligned": result.aligned,
                "max_signal_diff": result.max_signal_diff,
                "snapshots_checked": result.snapshots_checked,
            }
        except Exception as e:
            results[date_str] = {
                "aligned": False,
                "error": str(e),
            }
        current += timedelta(days=1)

    # 汇总报告
    print(f"\n{'='*60}")
    print("📊 汇总报告")
    print(f"{'='*60}")

    aligned_days = sum(1 for r in results.values() if r.get("aligned", False))
    total_days = len(results)

    print(f"检查天数: {total_days}")
    print(f"对齐天数: {aligned_days}")
    print(f"对齐率: {aligned_days/total_days*100:.1f}%")

    if aligned_days == total_days:
        print("\n✅ 所有日期对齐检查通过!")
    else:
        print("\n❌ 存在未对齐的日期:")
        for date_str, r in results.items():
            if not r.get("aligned", False):
                error = r.get("error", f"max_diff={r.get('max_signal_diff', 'N/A')}")
                print(f"  {date_str}: {error}")

    return results


def main():
    parser = argparse.ArgumentParser(description="AlgVex 每日对齐检查")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="检查日期 (YYYY-MM-DD), 默认昨天",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="开始日期 (用于范围检查)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="结束日期 (用于范围检查)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="容差阈值 (默认 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="data/snapshots",
        help="快照目录",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="data/live_outputs",
        help="实盘输出目录",
    )

    args = parser.parse_args()

    # 范围检查
    if args.start and args.end:
        results = run_range_check(args.start, args.end, args.tolerance)
        all_aligned = all(r.get("aligned", False) for r in results.values())
        sys.exit(0 if all_aligned else 1)

    # 单日检查
    if args.date:
        date = args.date
    else:
        # 默认检查昨天
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    result = run_daily_check(
        date=date,
        tolerance=args.tolerance,
        snapshot_dir=args.snapshot_dir,
        live_dir=args.live_dir,
    )

    sys.exit(0 if result.aligned else 1)


if __name__ == "__main__":
    main()
