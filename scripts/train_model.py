"""
模型训练脚本 (v10.0.0)

从 Parquet 读取数据，使用统一特征计算。

用法:
    python scripts/train_model.py --instruments btcusdt ethusdt

重要变更 (v10.0.0):
    - 从 Parquet 读取数据 (不使用 Qlib D.features)
    - 修复时间切分类型问题 (pd.Timestamp)
    - 使用 valid_start 参数
    - 保存训练元数据
"""

import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入统一特征模块 (兼容不同运行目录)
try:
    from scripts.unified_features import (
        compute_unified_features,
        compute_label,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )
except ImportError:
    from unified_features import (
        compute_unified_features,
        compute_label,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )


def load_parquet_data(data_dir: Path, instruments: list, freq: str) -> dict:
    """从 Parquet 加载 OHLCV 数据"""
    freq_dir = data_dir / freq
    if not freq_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {freq_dir}")

    data = {}
    for inst in instruments:
        file_path = freq_dir / f"{inst}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pd.read_parquet(file_path)
        data[inst] = df
        print(f"  Loaded {inst}: {len(df)} rows")

    return data


def parse_datetime(dt_str: str, end_of_day: bool = False) -> pd.Timestamp:
    """解析日期字符串为 pd.Timestamp (UTC)"""
    ts = pd.Timestamp(dt_str, tz="UTC")
    if end_of_day and len(dt_str) == 10:  # 只有日期，没有时间
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def train_model(
    data_dir: str,
    output_dir: str,
    instruments: list,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    freq: str = "1h",
):
    """训练 LightGBM 模型"""

    data_path = Path(data_dir).expanduser()

    # 解析时间边界 (修复 C3: 类型问题)
    train_start_ts = parse_datetime(train_start)
    train_end_ts = parse_datetime(train_end, end_of_day=True)
    valid_start_ts = parse_datetime(valid_start)
    valid_end_ts = parse_datetime(valid_end, end_of_day=True)

    print(f"Train period: {train_start_ts} to {train_end_ts}")
    print(f"Valid period: {valid_start_ts} to {valid_end_ts}")

    # 加载数据
    print(f"Loading data for {instruments}...")
    all_data = load_parquet_data(data_path, instruments, freq)

    # 计算统一特征
    print("Computing unified features...")
    all_features = []
    all_labels = []

    for inst, df in all_data.items():
        features = compute_unified_features(df)
        labels = compute_label(df)

        features["datetime"] = features.index
        features["instrument"] = inst
        labels.name = "label"

        all_features.append(features)
        all_labels.append(labels)

    features_df = pd.concat(all_features)
    labels_df = pd.concat(all_labels)

    # 合并并清理
    data = features_df.copy()
    data["label"] = labels_df.values
    data = data.dropna()

    print(f"Features shape: {data.shape}")

    # 分割训练/验证集 (修复 C3: 使用 pd.Timestamp 比较)
    train_mask = (data["datetime"] >= train_start_ts) & (data["datetime"] <= train_end_ts)
    valid_mask = (data["datetime"] >= valid_start_ts) & (data["datetime"] <= valid_end_ts)

    X_train = data.loc[train_mask, FEATURE_COLUMNS]
    y_train = data.loc[train_mask, "label"]
    X_valid = data.loc[valid_mask, FEATURE_COLUMNS]
    y_valid = data.loc[valid_mask, "label"]

    print(f"Train: {len(X_train)}, Valid: {len(X_valid)}")

    if len(X_train) == 0:
        raise ValueError("No training data! Check date range.")
    if len(X_valid) == 0:
        raise ValueError("No validation data! Check date range.")

    # 归一化特征 (仅用训练集 fit，防止泄漏)
    print("Normalizing features (fit on train only)...")
    normalizer = FeatureNormalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_valid_norm = normalizer.transform(X_valid, strict=False)  # 训练时可宽松

    # 训练 LightGBM
    print("Training LightGBM model...")
    train_data = lgb.Dataset(X_train_norm, label=y_train)
    valid_data = lgb.Dataset(X_valid_norm, label=y_valid, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(50)],
    )

    # 保存模型
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_path / "lgb_model.txt"))
    normalizer.save(str(output_path / "normalizer.pkl"))

    with open(output_path / "feature_columns.pkl", "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)

    # 保存元数据
    metadata = {
        "version": "v10.0.0",
        "created": datetime.utcnow().isoformat(),
        "instruments": instruments,
        "freq": freq,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "feature_count": len(FEATURE_COLUMNS),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {output_path}")

    # 验证
    predictions = model.predict(X_valid_norm)
    ic = np.corrcoef(predictions, y_valid)[0, 1]
    print(f"Validation IC: {ic:.4f}")

    return model, normalizer


def main():
    parser = argparse.ArgumentParser(description="Train model (v10.0.0)")
    parser.add_argument("--data-dir", type=str, default="~/.algvex/data")
    parser.add_argument("--output-dir", type=str, default="~/.algvex/models/qlib_alpha")
    parser.add_argument("--instruments", type=str, nargs="+", default=["btcusdt", "ethusdt"])
    # 时间切分: 训练/验证集之间建议留 1-2 周 gap 避免数据泄漏
    # 示例: train-end=06-15, valid-start=07-01 (留 2 周 gap)
    parser.add_argument("--train-start", type=str, default="2023-01-01")
    parser.add_argument("--train-end", type=str, default="2024-06-15")  # 提前 2 周
    parser.add_argument("--valid-start", type=str, default="2024-07-01")
    parser.add_argument("--valid-end", type=str, default="2024-12-31")
    parser.add_argument("--freq", type=str, default="1h")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        instruments=args.instruments,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
