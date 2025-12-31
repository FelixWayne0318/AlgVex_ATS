"""
AlgVex Replay 运行器

功能:
- 使用历史快照重放信号生成过程
- 验证回测与实盘的一致性
- 确保 Replay 确定性 (P0-4)
- 支持逐日/逐小时重放

使用方式:
    from core.replay.replay_runner import ReplayRunner

    runner = ReplayRunner()

    # 单次重放
    result = runner.replay(snapshot_id)

    # 每日重放对齐检查
    report = runner.daily_replay_check(date="2024-01-15")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.snapshot_manager import SnapshotManager
from ...production.factor_engine import MVPFactorEngine
from ...production.signal_generator import SignalGenerator, Signal
from ...shared.time_provider import TimeProvider, ReplayContext
from ...shared.seeded_random import SeededRandom
from ...shared.trace_logger import TraceLogger


@dataclass
class ReplayResult:
    """重放结果"""
    snapshot_id: str
    signal_time: str
    signals: List[Dict[str, Any]]
    factors: Dict[str, Dict[str, float]]
    execution_time_ms: float
    deterministic: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """对齐检查结果"""
    date: str
    snapshots_checked: int
    signals_compared: int
    aligned: bool
    max_signal_diff: float
    max_factor_diff: float
    mismatches: List[Dict[str, Any]]
    summary: str


class ReplayRunner:
    """Replay 运行器"""

    def __init__(
        self,
        snapshot_dir: str = "data/snapshots",
        output_dir: str = "data/replay_outputs",
        factor_engine: Optional[MVPFactorEngine] = None,
        signal_generator: Optional[SignalGenerator] = None,
    ):
        """
        初始化 Replay 运行器

        Args:
            snapshot_dir: 快照目录
            output_dir: 输出目录
            factor_engine: 因子引擎
            signal_generator: 信号生成器
        """
        self.snapshot_manager = SnapshotManager(snapshot_dir)
        self.output_dir = Path(output_dir)
        self.factor_engine = factor_engine or MVPFactorEngine()
        self.signal_generator = signal_generator or SignalGenerator(
            factor_engine=self.factor_engine,
            enable_trace=False,
        )

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def replay(
        self,
        snapshot_id: str,
        verify_determinism: bool = True,
    ) -> ReplayResult:
        """
        重放单个快照

        Args:
            snapshot_id: 快照ID
            verify_determinism: 是否验证确定性

        Returns:
            重放结果
        """
        import time
        start_time = time.time()

        # 加载快照
        data, metadata = self.snapshot_manager.load_snapshot(snapshot_id)
        signal_time = datetime.fromisoformat(metadata.signal_time)

        # 进入 Replay 模式
        with ReplayContext(signal_time):
            # 设置确定性随机数
            rng = SeededRandom.from_context(
                date=signal_time.strftime("%Y-%m-%d"),
                symbol="REPLAY",
            )

            # 准备数据
            klines_data = self._prepare_klines(data)
            oi_data = self._prepare_oi(data)
            funding_data = self._prepare_funding(data)
            symbols = list(klines_data.keys())

            # 生成信号
            signals = self.signal_generator.generate(
                symbols=symbols,
                klines_data=klines_data,
                oi_data=oi_data,
                funding_data=funding_data,
                signal_time=signal_time,
            )

            # 收集因子值
            factors = {}
            for symbol in symbols:
                if symbol in klines_data:
                    factor_values = self.factor_engine.compute_all_factors(
                        klines=klines_data[symbol],
                        oi=oi_data.get(symbol) if oi_data else None,
                        funding=funding_data.get(symbol) if funding_data else None,
                        signal_time=signal_time,
                    )
                    factors[symbol] = {
                        k: v.value for k, v in factor_values.items()
                        if v.is_valid
                    }

        execution_time = (time.time() - start_time) * 1000

        # 验证确定性（重复运行应得到相同结果）
        deterministic = True
        if verify_determinism:
            second_result = self._replay_once(snapshot_id, data, metadata)
            deterministic = self._compare_results(signals, second_result)

        result = ReplayResult(
            snapshot_id=snapshot_id,
            signal_time=metadata.signal_time,
            signals=[s.to_dict() for s in signals],
            factors=factors,
            execution_time_ms=execution_time,
            deterministic=deterministic,
            metadata={
                "data_hash": metadata.data_hash,
                "symbols": symbols,
            },
        )

        # 保存结果
        self._save_result(result)

        return result

    def _replay_once(
        self,
        snapshot_id: str,
        data: Dict[str, pd.DataFrame],
        metadata,
    ) -> List[Signal]:
        """执行一次重放"""
        signal_time = datetime.fromisoformat(metadata.signal_time)

        with ReplayContext(signal_time):
            rng = SeededRandom.from_context(
                date=signal_time.strftime("%Y-%m-%d"),
                symbol="REPLAY",
            )

            klines_data = self._prepare_klines(data)
            oi_data = self._prepare_oi(data)
            funding_data = self._prepare_funding(data)

            return self.signal_generator.generate(
                symbols=list(klines_data.keys()),
                klines_data=klines_data,
                oi_data=oi_data,
                funding_data=funding_data,
                signal_time=signal_time,
            )

    def _compare_results(
        self,
        signals1: List[Signal],
        signals2: List[Signal],
        tolerance: float = 1e-10,
    ) -> bool:
        """比较两次结果是否一致"""
        if len(signals1) != len(signals2):
            return False

        for s1, s2 in zip(signals1, signals2):
            if s1.symbol != s2.symbol:
                return False
            if s1.signal_type != s2.signal_type:
                return False
            if abs(s1.strength - s2.strength) > tolerance:
                return False

        return True

    def daily_replay_check(
        self,
        date: str,
        live_outputs_dir: str = "data/live_outputs",
        tolerance: float = 0.001,
    ) -> AlignmentResult:
        """
        每日 Replay 对齐检查

        Args:
            date: 日期 (YYYY-MM-DD)
            live_outputs_dir: 实盘输出目录
            tolerance: 容差

        Returns:
            对齐检查结果
        """
        # 加载实盘输出
        live_outputs = self._load_live_outputs(live_outputs_dir, date)

        # 获取当日快照
        snapshots = self.snapshot_manager.list_snapshots(
            start_date=date,
            end_date=date,
        )

        mismatches = []
        max_signal_diff = 0.0
        max_factor_diff = 0.0
        signals_compared = 0

        for metadata in snapshots:
            # 重放
            replay_result = self.replay(metadata.snapshot_id, verify_determinism=True)

            # 查找对应的实盘输出
            live_output = live_outputs.get(metadata.snapshot_id)
            if live_output is None:
                continue

            # 比较信号
            for replay_signal in replay_result.signals:
                signals_compared += 1
                live_signal = self._find_matching_signal(
                    replay_signal, live_output.get("signals", [])
                )

                if live_signal is None:
                    mismatches.append({
                        "type": "missing_live_signal",
                        "snapshot_id": metadata.snapshot_id,
                        "symbol": replay_signal["symbol"],
                    })
                    continue

                # 计算差异
                signal_diff = abs(
                    replay_signal["strength"] - live_signal.get("strength", 0)
                )
                max_signal_diff = max(max_signal_diff, signal_diff)

                if signal_diff > tolerance:
                    mismatches.append({
                        "type": "signal_mismatch",
                        "snapshot_id": metadata.snapshot_id,
                        "symbol": replay_signal["symbol"],
                        "replay_strength": replay_signal["strength"],
                        "live_strength": live_signal.get("strength"),
                        "diff": signal_diff,
                    })

            # 比较因子
            live_factors = live_output.get("factors", {})
            for symbol, replay_factors in replay_result.factors.items():
                if symbol not in live_factors:
                    continue

                for factor_id, replay_value in replay_factors.items():
                    live_value = live_factors[symbol].get(factor_id)
                    if live_value is None:
                        continue

                    factor_diff = abs(replay_value - live_value)
                    max_factor_diff = max(max_factor_diff, factor_diff)

                    if factor_diff > tolerance:
                        mismatches.append({
                            "type": "factor_mismatch",
                            "snapshot_id": metadata.snapshot_id,
                            "symbol": symbol,
                            "factor_id": factor_id,
                            "replay_value": replay_value,
                            "live_value": live_value,
                            "diff": factor_diff,
                        })

        aligned = len(mismatches) == 0 and max_signal_diff <= tolerance

        summary = (
            f"日期: {date}\n"
            f"快照数: {len(snapshots)}\n"
            f"信号比对数: {signals_compared}\n"
            f"最大信号差异: {max_signal_diff:.6f}\n"
            f"最大因子差异: {max_factor_diff:.6f}\n"
            f"不匹配数: {len(mismatches)}\n"
            f"对齐状态: {'✅ 通过' if aligned else '❌ 未通过'}"
        )

        result = AlignmentResult(
            date=date,
            snapshots_checked=len(snapshots),
            signals_compared=signals_compared,
            aligned=aligned,
            max_signal_diff=max_signal_diff,
            max_factor_diff=max_factor_diff,
            mismatches=mismatches,
            summary=summary,
        )

        # 保存报告
        self._save_alignment_report(result)

        return result

    def _prepare_klines(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """准备K线数据"""
        result = {}
        klines_key = None

        for key in data.keys():
            if "klines" in key.lower():
                klines_key = key
                break

        if klines_key and not data[klines_key].empty:
            df = data[klines_key]
            if "symbol" in df.columns:
                for symbol in df["symbol"].unique():
                    result[symbol] = df[df["symbol"] == symbol].copy()
                    if "datetime" in result[symbol].columns:
                        result[symbol].set_index("datetime", inplace=True)

        return result

    def _prepare_oi(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, pd.DataFrame]]:
        """准备持仓量数据"""
        result = {}
        oi_key = None

        for key in data.keys():
            if "oi" in key.lower() or "open_interest" in key.lower():
                oi_key = key
                break

        if oi_key and not data[oi_key].empty:
            df = data[oi_key]
            if "symbol" in df.columns:
                for symbol in df["symbol"].unique():
                    result[symbol] = df[df["symbol"] == symbol].copy()
                    if "datetime" in result[symbol].columns:
                        result[symbol].set_index("datetime", inplace=True)

        return result if result else None

    def _prepare_funding(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, pd.DataFrame]]:
        """准备资金费率数据"""
        result = {}
        funding_key = None

        for key in data.keys():
            if "funding" in key.lower():
                funding_key = key
                break

        if funding_key and not data[funding_key].empty:
            df = data[funding_key]
            if "symbol" in df.columns:
                for symbol in df["symbol"].unique():
                    result[symbol] = df[df["symbol"] == symbol].copy()
                    if "datetime" in result[symbol].columns:
                        result[symbol].set_index("datetime", inplace=True)

        return result if result else None

    def _load_live_outputs(
        self,
        live_dir: str,
        date: str,
    ) -> Dict[str, Dict]:
        """加载实盘输出"""
        live_path = Path(live_dir)
        outputs = {}

        # 查找当日输出文件
        output_file = live_path / f"live_output_{date}.jsonl"
        if not output_file.exists():
            return outputs

        with open(output_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    snapshot_id = record.get("snapshot_id")
                    if snapshot_id:
                        outputs[snapshot_id] = record
                except Exception:
                    pass

        return outputs

    def _find_matching_signal(
        self,
        replay_signal: Dict,
        live_signals: List[Dict],
    ) -> Optional[Dict]:
        """查找匹配的实盘信号"""
        for live_signal in live_signals:
            if live_signal.get("symbol") == replay_signal.get("symbol"):
                return live_signal
        return None

    def _save_result(self, result: ReplayResult):
        """保存重放结果"""
        date_str = result.signal_time[:10]
        output_file = self.output_dir / f"replay_output_{date_str}.jsonl"

        with open(output_file, "a") as f:
            f.write(json.dumps(result.__dict__, default=str) + "\n")

    def _save_alignment_report(self, result: AlignmentResult):
        """保存对齐报告"""
        report_file = self.output_dir / f"alignment_report_{result.date}.json"

        with open(report_file, "w") as f:
            json.dump(result.__dict__, f, indent=2, default=str)


# 初始化目录
def init_replay_dirs():
    """初始化 Replay 相关目录"""
    Path("algvex/core/replay").mkdir(parents=True, exist_ok=True)


# 测试代码
if __name__ == "__main__":
    print("ReplayRunner 已定义")
    print("使用方式:")
    print("  runner = ReplayRunner()")
    print("  result = runner.replay(snapshot_id)")
    print("  report = runner.daily_replay_check(date='2024-01-15')")
