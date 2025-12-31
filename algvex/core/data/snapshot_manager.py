"""
AlgVex 快照管理器

功能:
- 创建数据快照用于回测复现
- 保存完整的数据状态
- 支持快照恢复和比对
- 确保 Replay 确定性 (P0-4)

使用方式:
    from core.data.snapshot_manager import SnapshotManager

    manager = SnapshotManager()

    # 创建快照
    snapshot_id = manager.create_snapshot(data_dict, signal_time)

    # 加载快照
    data = manager.load_snapshot(snapshot_id)

    # 比对快照
    diff = manager.compare_snapshots(snapshot_id_1, snapshot_id_2)
"""

import hashlib
import json
import os
import pickle
import gzip
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pandas as pd
import numpy as np


@dataclass
class SnapshotMetadata:
    """快照元数据"""
    snapshot_id: str
    created_at: str
    signal_time: str
    data_sources: List[str]
    symbols: List[str]
    data_hash: str
    row_counts: Dict[str, int]
    config_hashes: Dict[str, str]
    description: str = ""
    extra_metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


class SnapshotManager:
    """快照管理器"""

    def __init__(
        self,
        snapshot_dir: str = "data/snapshots",
        compression: bool = True,
        base_dir: Optional[str] = None,  # 别名参数
    ):
        """
        初始化快照管理器

        Args:
            snapshot_dir: 快照存储目录
            compression: 是否压缩快照
            base_dir: 快照存储目录 (别名，优先使用)
        """
        # base_dir 是 snapshot_dir 的别名
        actual_dir = base_dir if base_dir is not None else snapshot_dir
        self.snapshot_dir = Path(actual_dir)
        self.compression = compression

        # 创建目录
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshot_dir / "metadata").mkdir(exist_ok=True)
        (self.snapshot_dir / "data").mkdir(exist_ok=True)

    def create_snapshot(
        self,
        data: Dict[str, pd.DataFrame],
        signal_time: Optional[datetime] = None,
        config_hashes: Optional[Dict[str, str]] = None,
        description: str = "",
        cutoff_time: Optional[datetime] = None,  # 别名参数
        metadata: Optional[Dict[str, Any]] = None,  # 额外元数据
    ) -> str:
        """
        创建数据快照

        Args:
            data: 数据字典 {source_id: DataFrame}
            signal_time: 信号时间
            config_hashes: 配置哈希
            description: 描述
            cutoff_time: 信号时间 (别名)
            metadata: 额外元数据

        Returns:
            快照ID
        """
        # 处理别名参数: cutoff_time 优先于 signal_time
        actual_signal_time = cutoff_time if cutoff_time is not None else signal_time
        if actual_signal_time is None:
            actual_signal_time = datetime.utcnow()

        # 生成快照ID
        snapshot_id = self._generate_snapshot_id(actual_signal_time)

        # 计算数据哈希
        data_hash = self._compute_data_hash(data)

        # 收集统计信息
        data_sources = list(data.keys())
        symbols = self._extract_symbols(data)
        row_counts = {k: len(v) for k, v in data.items()}

        # 创建元数据 (注意: 参数名为 metadata，需要用不同变量名)
        snapshot_metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            created_at=datetime.utcnow().isoformat(),
            signal_time=actual_signal_time.isoformat(),
            data_sources=data_sources,
            symbols=symbols,
            data_hash=data_hash,
            row_counts=row_counts,
            config_hashes=config_hashes or {},
            description=description,
            extra_metadata=metadata or {},  # 存储额外元数据
        )

        # 保存元数据
        self._save_metadata(snapshot_metadata)

        # 保存数据
        self._save_data(snapshot_id, data)

        print(f"✅ 快照已创建: {snapshot_id}")
        print(f"   数据源: {len(data_sources)}")
        print(f"   总行数: {sum(row_counts.values())}")
        print(f"   哈希: {data_hash}")

        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Dict[str, pd.DataFrame]:
        """
        加载快照

        Args:
            snapshot_id: 快照ID

        Returns:
            数据字典 {source_id: DataFrame}
        """
        # 加载元数据
        metadata = self._load_metadata(snapshot_id)
        if metadata is None:
            raise FileNotFoundError(f"快照不存在: {snapshot_id}")

        # 加载数据
        data = self._load_data(snapshot_id)

        # 验证数据完整性
        computed_hash = self._compute_data_hash(data)
        if computed_hash != metadata.data_hash:
            raise ValueError(
                f"数据完整性校验失败!\n"
                f"  期望: {metadata.data_hash}\n"
                f"  实际: {computed_hash}"
            )

        return data

    def load_snapshot_with_metadata(self, snapshot_id: str) -> Tuple[Dict[str, pd.DataFrame], SnapshotMetadata]:
        """
        加载快照及元数据

        Args:
            snapshot_id: 快照ID

        Returns:
            (数据字典, 元数据)
        """
        # 加载元数据
        metadata = self._load_metadata(snapshot_id)
        if metadata is None:
            raise FileNotFoundError(f"快照不存在: {snapshot_id}")

        # 加载数据
        data = self._load_data(snapshot_id)

        # 验证数据完整性
        computed_hash = self._compute_data_hash(data)
        if computed_hash != metadata.data_hash:
            raise ValueError(
                f"数据完整性校验失败!\n"
                f"  期望: {metadata.data_hash}\n"
                f"  实际: {computed_hash}"
            )

        return data, metadata

    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        获取快照信息

        Args:
            snapshot_id: 快照ID

        Returns:
            快照信息字典，如果快照不存在则返回 None
        """
        metadata = self._load_metadata(snapshot_id)
        if metadata is None:
            return None

        return {
            "snapshot_id": metadata.snapshot_id,
            "created_at": metadata.created_at,
            "cutoff_time": metadata.signal_time,  # 别名
            "signal_time": metadata.signal_time,
            "data_sources": metadata.data_sources,
            "symbols": metadata.symbols,
            "data_hash": metadata.data_hash,
            "row_counts": metadata.row_counts,
            "config_hashes": metadata.config_hashes,
            "description": metadata.description,
            "metadata": metadata.extra_metadata,
        }

    def list_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SnapshotMetadata]:
        """列出所有快照"""
        snapshots = []

        metadata_dir = self.snapshot_dir / "metadata"
        for meta_file in sorted(metadata_dir.glob("*.json")):
            try:
                with open(meta_file, "r") as f:
                    meta_dict = json.load(f)
                    metadata = SnapshotMetadata(**meta_dict)

                    # 过滤日期
                    signal_date = metadata.signal_time[:10]
                    if start_date and signal_date < start_date:
                        continue
                    if end_date and signal_date > end_date:
                        continue

                    snapshots.append(metadata)
            except Exception as e:
                print(f"警告: 无法读取元数据 {meta_file}: {e}")

        return snapshots

    def compare_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Dict[str, Any]:
        """
        比较两个快照

        Args:
            snapshot_id_1: 第一个快照ID
            snapshot_id_2: 第二个快照ID

        Returns:
            差异报告
        """
        data1, meta1 = self.load_snapshot(snapshot_id_1)
        data2, meta2 = self.load_snapshot(snapshot_id_2)

        diff = {
            "snapshot_1": snapshot_id_1,
            "snapshot_2": snapshot_id_2,
            "hash_match": meta1.data_hash == meta2.data_hash,
            "signal_time_1": meta1.signal_time,
            "signal_time_2": meta2.signal_time,
            "data_sources_diff": {
                "only_in_1": set(meta1.data_sources) - set(meta2.data_sources),
                "only_in_2": set(meta2.data_sources) - set(meta1.data_sources),
                "common": set(meta1.data_sources) & set(meta2.data_sources),
            },
            "row_count_diff": {},
            "value_diffs": {},
        }

        # 比较每个数据源
        for source in diff["data_sources_diff"]["common"]:
            df1 = data1.get(source, pd.DataFrame())
            df2 = data2.get(source, pd.DataFrame())

            # 行数差异
            diff["row_count_diff"][source] = {
                "snapshot_1": len(df1),
                "snapshot_2": len(df2),
                "difference": len(df1) - len(df2),
            }

            # 值差异（仅数值列）
            if not df1.empty and not df2.empty:
                numeric_cols = df1.select_dtypes(include=[np.number]).columns
                value_diff = {}

                for col in numeric_cols:
                    if col in df2.columns:
                        try:
                            # 对齐索引后比较
                            common_idx = df1.index.intersection(df2.index)
                            if len(common_idx) > 0:
                                v1 = df1.loc[common_idx, col]
                                v2 = df2.loc[common_idx, col]
                                max_diff = (v1 - v2).abs().max()
                                if max_diff > 1e-10:
                                    value_diff[col] = float(max_diff)
                        except Exception:
                            pass

                if value_diff:
                    diff["value_diffs"][source] = value_diff

        return diff

    def delete_snapshot(self, snapshot_id: str):
        """删除快照"""
        # 删除元数据
        meta_file = self.snapshot_dir / "metadata" / f"{snapshot_id}.json"
        if meta_file.exists():
            meta_file.unlink()

        # 删除数据
        ext = ".pkl.gz" if self.compression else ".pkl"
        data_file = self.snapshot_dir / "data" / f"{snapshot_id}{ext}"
        if data_file.exists():
            data_file.unlink()

        print(f"快照已删除: {snapshot_id}")

    def cleanup_old_snapshots(self, keep_days: int = 30):
        """清理旧快照"""
        cutoff = datetime.utcnow().timestamp() - (keep_days * 86400)
        deleted = 0

        for metadata in self.list_snapshots():
            created_at = datetime.fromisoformat(metadata.created_at).timestamp()
            if created_at < cutoff:
                self.delete_snapshot(metadata.snapshot_id)
                deleted += 1

        print(f"已清理 {deleted} 个旧快照")

    def _generate_snapshot_id(self, signal_time: datetime) -> str:
        """生成快照ID"""
        date_str = signal_time.strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        return f"snap_{date_str}_{uid}"

    def _compute_data_hash(self, data: Dict[str, pd.DataFrame]) -> str:
        """计算数据哈希"""
        hasher = hashlib.sha256()

        for key in sorted(data.keys()):
            df = data[key]
            if not df.empty:
                # 规范化数据
                df_sorted = df.sort_index()
                # 转为bytes
                df_bytes = pickle.dumps(df_sorted.values.tobytes())
                hasher.update(key.encode())
                hasher.update(df_bytes)

        return f"sha256:{hasher.hexdigest()[:16]}"

    def _extract_symbols(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """提取所有交易对"""
        symbols = set()
        for df in data.values():
            if "symbol" in df.columns:
                symbols.update(df["symbol"].unique())
        return sorted(symbols)

    def _save_metadata(self, metadata: SnapshotMetadata):
        """保存元数据"""
        meta_file = self.snapshot_dir / "metadata" / f"{metadata.snapshot_id}.json"
        with open(meta_file, "w") as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)

    def _load_metadata(self, snapshot_id: str) -> Optional[SnapshotMetadata]:
        """加载元数据"""
        meta_file = self.snapshot_dir / "metadata" / f"{snapshot_id}.json"
        if not meta_file.exists():
            return None

        with open(meta_file, "r") as f:
            meta_dict = json.load(f)
            return SnapshotMetadata(**meta_dict)

    def _save_data(self, snapshot_id: str, data: Dict[str, pd.DataFrame]):
        """保存数据"""
        ext = ".pkl.gz" if self.compression else ".pkl"
        data_file = self.snapshot_dir / "data" / f"{snapshot_id}{ext}"

        if self.compression:
            with gzip.open(data_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(data_file, "wb") as f:
                pickle.dump(data, f)

    def _load_data(self, snapshot_id: str) -> Dict[str, pd.DataFrame]:
        """加载数据"""
        # 尝试压缩格式
        data_file = self.snapshot_dir / "data" / f"{snapshot_id}.pkl.gz"
        if data_file.exists():
            with gzip.open(data_file, "rb") as f:
                return pickle.load(f)

        # 尝试非压缩格式
        data_file = self.snapshot_dir / "data" / f"{snapshot_id}.pkl"
        if data_file.exists():
            with open(data_file, "rb") as f:
                return pickle.load(f)

        raise ValueError(f"快照数据不存在: {snapshot_id}")


# 测试代码
if __name__ == "__main__":
    import numpy as np

    # 创建测试数据
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    test_data = {
        "klines_5m": pd.DataFrame({
            "datetime": dates,
            "symbol": "BTCUSDT",
            "open": 100 + np.random.randn(100),
            "high": 101 + np.random.randn(100),
            "low": 99 + np.random.randn(100),
            "close": 100.5 + np.random.randn(100),
            "volume": 1000 + np.random.randint(0, 500, 100),
        }),
        "funding_8h": pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=10, freq="8h"),
            "symbol": "BTCUSDT",
            "funding_rate": np.random.randn(10) * 0.0001,
        }),
    }

    # 测试快照管理器
    manager = SnapshotManager(snapshot_dir="test_snapshots")

    # 创建快照
    signal_time = datetime(2024, 1, 1, 10, 5, 0)
    snapshot_id = manager.create_snapshot(
        data=test_data,
        signal_time=signal_time,
        description="测试快照",
    )

    # 加载快照
    loaded_data, metadata = manager.load_snapshot(snapshot_id)
    print(f"\n加载的快照:")
    print(f"  信号时间: {metadata.signal_time}")
    print(f"  数据源: {metadata.data_sources}")
    print(f"  行数: {metadata.row_counts}")

    # 清理
    import shutil
    shutil.rmtree("test_snapshots", ignore_errors=True)
