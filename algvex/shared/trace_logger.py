"""
AlgVex Trace 日志记录器

功能:
- 记录每条信号的完整追溯信息
- 包含配置哈希、代码版本、数据快照
- 支持审计和问题排查
- 符合 P6 可复现与可追责原则

Trace Schema:
每条信号必须能回答:
- 当时看到的数据快照是什么？
- 因子值怎么来的？
- 使用了什么配置？
"""

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False


@dataclass
class FactorTrace:
    """因子追溯信息"""
    factor_id: str
    value: float
    data_dependencies: Dict[str, str]  # {source_id: data_time}
    visible_time: str
    formula: Optional[str] = None


@dataclass
class SignalTrace:
    """信号追溯信息"""
    # 唯一标识
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # 时间信息
    signal_time: str = ""
    snapshot_cutoff: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # 信号内容
    symbol: str = ""
    signal_type: str = ""  # "long" | "short" | "close"
    signal_strength: float = 0.0
    confidence: float = 0.0

    # 因子追溯
    factors_used: List[FactorTrace] = field(default_factory=list)

    # 模型信息
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_weights_hash: Optional[str] = None

    # 配置追溯
    config_hashes: Dict[str, str] = field(default_factory=dict)
    code_hash: Optional[str] = None
    git_commit: Optional[str] = None

    # 数据快照
    snapshot_id: Optional[str] = None
    data_hash: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceLogger:
    """Trace 日志记录器"""

    def __init__(
        self,
        output_dir: str = "traces",
        config_dir: str = "config",
        enable_git: bool = True,
    ):
        """
        初始化 Trace 记录器

        Args:
            output_dir: Trace 输出目录
            config_dir: 配置文件目录
            enable_git: 是否记录 Git 信息
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir)
        self.enable_git = enable_git and GIT_AVAILABLE

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 缓存配置哈希
        self._config_hashes: Dict[str, str] = {}
        self._code_hash: Optional[str] = None
        self._git_commit: Optional[str] = None

        # 初始化
        self._load_config_hashes()
        self._load_git_info()

    def _load_config_hashes(self):
        """加载所有配置文件的哈希"""
        if not self.config_dir.exists():
            return

        for yaml_file in self.config_dir.glob("**/*.yaml"):
            try:
                with open(yaml_file, "rb") as f:
                    content = f.read()
                hash_value = hashlib.sha256(content).hexdigest()[:16]
                rel_path = yaml_file.relative_to(self.config_dir)
                self._config_hashes[str(rel_path)] = f"sha256:{hash_value}"
            except Exception:
                pass

    def _load_git_info(self):
        """加载 Git 版本信息"""
        if not self.enable_git:
            return

        try:
            repo = git.Repo(search_parent_directories=True)
            self._git_commit = repo.head.commit.hexsha[:8]

            # 检查是否有未提交的更改
            if repo.is_dirty():
                self._git_commit += "-dirty"

        except Exception:
            self._git_commit = None

    def _compute_code_hash(self, code_files: List[str] = None) -> str:
        """计算代码文件的哈希"""
        if code_files is None:
            # 默认计算 shared/ 目录的代码
            code_dir = Path(__file__).parent
            code_files = list(code_dir.glob("*.py"))

        hasher = hashlib.sha256()
        for file_path in sorted(code_files):
            try:
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
            except Exception:
                pass

        return f"sha256:{hasher.hexdigest()[:16]}"

    def create_trace(
        self,
        symbol: str,
        signal_type: str,
        signal_strength: float,
        signal_time: datetime,
        factors: List[Dict[str, Any]],
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        data_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalTrace:
        """
        创建信号追溯记录

        Args:
            symbol: 交易对
            signal_type: 信号类型
            signal_strength: 信号强度
            signal_time: 信号时间
            factors: 使用的因子列表
            model_id: 模型ID
            model_version: 模型版本
            snapshot_id: 数据快照ID
            data_hash: 数据哈希
            metadata: 额外元数据

        Returns:
            SignalTrace 对象
        """
        # 转换因子信息
        factor_traces = []
        for f in factors:
            factor_traces.append(
                FactorTrace(
                    factor_id=f.get("factor_id", ""),
                    value=f.get("value", 0.0),
                    data_dependencies=f.get("data_dependencies", {}),
                    visible_time=f.get("visible_time", ""),
                    formula=f.get("formula"),
                )
            )

        # 计算代码哈希
        if self._code_hash is None:
            self._code_hash = self._compute_code_hash()

        # 创建 Trace
        trace = SignalTrace(
            signal_time=signal_time.isoformat(),
            snapshot_cutoff=(
                signal_time - __import__("datetime").timedelta(seconds=1)
            ).isoformat(),
            symbol=symbol,
            signal_type=signal_type,
            signal_strength=signal_strength,
            factors_used=factor_traces,
            model_id=model_id,
            model_version=model_version,
            config_hashes=self._config_hashes.copy(),
            code_hash=self._code_hash,
            git_commit=self._git_commit,
            snapshot_id=snapshot_id,
            data_hash=data_hash,
            metadata=metadata or {},
        )

        return trace

    def log_trace(self, trace: SignalTrace, flush: bool = True) -> str:
        """
        记录 Trace 到文件

        Args:
            trace: SignalTrace 对象
            flush: 是否立即写入

        Returns:
            Trace 文件路径
        """
        # 按日期组织文件
        date_str = trace.created_at[:10]
        output_file = self.output_dir / f"traces_{date_str}.jsonl"

        # 转换为 JSON
        trace_dict = self._trace_to_dict(trace)
        trace_json = json.dumps(trace_dict, ensure_ascii=False)

        # 写入文件
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(trace_json + "\n")
            if flush:
                f.flush()

        return str(output_file)

    def _trace_to_dict(self, trace: SignalTrace) -> Dict[str, Any]:
        """将 SignalTrace 转换为字典"""
        result = {
            "trace_id": trace.trace_id,
            "signal_time": trace.signal_time,
            "snapshot_cutoff": trace.snapshot_cutoff,
            "created_at": trace.created_at,
            "symbol": trace.symbol,
            "signal_type": trace.signal_type,
            "signal_strength": trace.signal_strength,
            "confidence": trace.confidence,
            "factors_used": [
                {
                    "factor_id": f.factor_id,
                    "value": f.value,
                    "data_dependencies": f.data_dependencies,
                    "visible_time": f.visible_time,
                    "formula": f.formula,
                }
                for f in trace.factors_used
            ],
            "model_id": trace.model_id,
            "model_version": trace.model_version,
            "model_weights_hash": trace.model_weights_hash,
            "config_hashes": trace.config_hashes,
            "code_hash": trace.code_hash,
            "git_commit": trace.git_commit,
            "snapshot_id": trace.snapshot_id,
            "data_hash": trace.data_hash,
            "metadata": trace.metadata,
        }
        return result

    def load_traces(
        self,
        date: str,
        symbol: Optional[str] = None,
    ) -> List[SignalTrace]:
        """
        加载指定日期的 Trace 记录

        Args:
            date: 日期字符串 (YYYY-MM-DD)
            symbol: 过滤特定交易对

        Returns:
            SignalTrace 列表
        """
        trace_file = self.output_dir / f"traces_{date}.jsonl"

        if not trace_file.exists():
            return []

        traces = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                if symbol and data.get("symbol") != symbol:
                    continue

                # 重建 FactorTrace 对象
                factor_traces = [
                    FactorTrace(**f_data)
                    for f_data in data.get("factors_used", [])
                ]

                trace = SignalTrace(
                    trace_id=data.get("trace_id", ""),
                    signal_time=data.get("signal_time", ""),
                    snapshot_cutoff=data.get("snapshot_cutoff", ""),
                    created_at=data.get("created_at", ""),
                    symbol=data.get("symbol", ""),
                    signal_type=data.get("signal_type", ""),
                    signal_strength=data.get("signal_strength", 0.0),
                    confidence=data.get("confidence", 0.0),
                    factors_used=factor_traces,
                    model_id=data.get("model_id"),
                    model_version=data.get("model_version"),
                    model_weights_hash=data.get("model_weights_hash"),
                    config_hashes=data.get("config_hashes", {}),
                    code_hash=data.get("code_hash"),
                    git_commit=data.get("git_commit"),
                    snapshot_id=data.get("snapshot_id"),
                    data_hash=data.get("data_hash"),
                    metadata=data.get("metadata", {}),
                )
                traces.append(trace)

        return traces


def create_trace(
    symbol: str,
    signal_type: str,
    signal_strength: float,
    signal_time: datetime,
    factors: List[Dict[str, Any]],
    **kwargs,
) -> SignalTrace:
    """
    便捷函数：创建并记录 Trace

    Args:
        symbol: 交易对
        signal_type: 信号类型
        signal_strength: 信号强度
        signal_time: 信号时间
        factors: 因子列表
        **kwargs: 其他参数

    Returns:
        SignalTrace 对象
    """
    logger = TraceLogger()
    trace = logger.create_trace(
        symbol=symbol,
        signal_type=signal_type,
        signal_strength=signal_strength,
        signal_time=signal_time,
        factors=factors,
        **kwargs,
    )
    logger.log_trace(trace)
    return trace


# 测试代码
if __name__ == "__main__":
    logger = TraceLogger(output_dir="test_traces")

    # 创建测试 Trace
    trace = logger.create_trace(
        symbol="BTCUSDT",
        signal_type="long",
        signal_strength=0.85,
        signal_time=datetime.utcnow(),
        factors=[
            {
                "factor_id": "return_5m",
                "value": 0.002,
                "data_dependencies": {"klines_5m": "2024-01-01T10:00:00"},
                "visible_time": "2024-01-01T10:05:00",
            },
            {
                "factor_id": "ma_cross",
                "value": 0.015,
                "data_dependencies": {"klines_5m": "2024-01-01T10:00:00"},
                "visible_time": "2024-01-01T10:05:00",
            },
        ],
        model_id="lightgbm_v1",
        model_version="1.0.0",
    )

    # 记录 Trace
    output_path = logger.log_trace(trace)
    print(f"Trace 已记录到: {output_path}")
    print(f"Trace ID: {trace.trace_id}")
    print(f"Config Hashes: {trace.config_hashes}")
    print(f"Git Commit: {trace.git_commit}")
