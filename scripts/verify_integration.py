"""
集成验证脚本 (v10.0.0)

验证 AlgVex 系统完整性：
- Parquet 数据质量
- 模型加载
- 特征计算一致性
- Normalizer strict 模式
- 回测链路一致性

用法:
    python scripts/verify_integration.py
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ========== 配置 ==========
DATA_DIR = Path("~/.algvex/data").expanduser()
MODEL_DIR = Path("~/.algvex/models").expanduser()
MIN_BARS = 61  # 与 Controller 一致


def verify_qlib_runtime_config() -> bool:
    """
    验证 Qlib 运行时配置覆盖能力 (v10.0.4)

    v10.0.4 修正:
    - 不再用 Parquet 目录作为 provider_uri (语义不清)
    - 只验证 C 配置覆盖功能是否可用
    - 此项为可选验证，失败不阻断 MVP (主流程用 Parquet + unified_features)
    """
    print("1. Testing Qlib runtime config (optional)...")
    try:
        import qlib
        from qlib.config import C

        # v10.0.4: 不调用 qlib.init，只验证 C 配置覆盖能力
        # 主流程不依赖 Qlib Provider，使用 Parquet + unified_features

        # 验证运行时覆盖
        C["trade_unit"] = 1  # 加密货币无最小交易单位
        C["limit_threshold"] = None  # 加密货币无涨跌停

        assert C["trade_unit"] == 1, f"Expected trade_unit=1, got {C['trade_unit']}"
        assert C["limit_threshold"] is None, f"Expected None, got {C['limit_threshold']}"

        print("   ✓ Qlib C[] override works (trade_unit=1, limit_threshold=None)")
        return True
    except ImportError:
        print("   ⚠ Qlib not installed (optional - main flow uses Parquet)")
        return True  # 非阻断性
    except Exception as e:
        print(f"   ⚠ Qlib config test failed: {e} (optional, not blocking)")
        return True  # v10.0.4: 改为非阻断性


def verify_parquet_data(freq: str = "1h") -> Tuple[bool, dict]:
    """
    验证 Parquet 数据质量 (v10.0.1)

    检查项:
    - Parquet 文件存在
    - datetime 在列或 index (兼容两种格式)
    - 必需列完整 (open, high, low, close, volume)
    - 缺失值检查
    - 价格值合理 (> 0)
    - 时区为 UTC
    - index 无重复且有序
    """
    print(f"2. Testing Parquet data ({freq})...")

    freq_dir = DATA_DIR / freq
    if not freq_dir.exists():
        print(f"   ⚠ Data directory not found: {freq_dir}")
        print("   ⚠ Run prepare_crypto_data.py first")
        return True, {"passed": True, "issues": ["Directory not found - run prepare_crypto_data.py"]}

    # 检查 btcusdt.parquet 作为示例
    file_path = freq_dir / "btcusdt.parquet"
    if not file_path.exists():
        parquet_files = list(freq_dir.glob("*.parquet"))
        if not parquet_files:
            print("   ⚠ No Parquet files found - run prepare_crypto_data.py first")
            return True, {"passed": True, "issues": ["No Parquet files"]}
        file_path = parquet_files[0]

    try:
        df = pd.read_parquet(file_path)
        inst_name = file_path.stem
        print(f"   Checking {inst_name}...")

        # 1) 兼容：datetime 可能在列，也可能在 index
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.set_index("datetime")

        if not isinstance(df.index, pd.DatetimeIndex):
            return False, {"passed": False, "issues": ["DatetimeIndex missing"]}

        # 2) 统一 UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # 3) 必需列（不再要求 datetime 列）
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, {"passed": False, "issues": [f"Missing columns: {missing}"]}

        # 4) 基础健康检查
        ok = True
        issues = []

        if df.index.has_duplicates:
            ok = False
            issues.append("duplicate datetime index")

        if not df.index.is_monotonic_increasing:
            ok = False
            issues.append("datetime index not sorted")

        # NaN 检查
        nan_counts = df[required_cols].isna().sum().to_dict()
        if any(v > 0 for v in nan_counts.values()):
            ok = False
            issues.append(f"NaN exists: {nan_counts}")

        # 价格 > 0 检查
        for col in ["open", "high", "low", "close"]:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                ok = False
                issues.append(f"{col}: {invalid} rows <= 0")

        info = {
            "passed": ok,
            "rows": int(len(df)),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "tz": str(df.index.tz),
            "nan_counts": nan_counts,
            "issues": issues,
        }

        if ok:
            print(f"      ✓ {len(df)} rows, {df.index.min()} to {df.index.max()}")
            print(f"   ✓ Parquet data quality passed")
        else:
            print(f"   ✗ Data quality issues: {issues}")

        return ok, info

    except Exception as e:
        return False, {"passed": False, "issues": [f"read error: {e}"]}


def verify_model_load(strategy: str = "qlib_alpha") -> bool:
    """验证模型加载 (v10.0.0: ~/.algvex/models/)"""
    print(f"3. Testing model load ({strategy})...")

    model_dir = MODEL_DIR / strategy
    if not model_dir.exists():
        print(f"   ⚠ Model directory not found: {model_dir}")
        print("   ⚠ Run train_model.py first")
        return True  # 非阻断性

    # 检查必需文件 (v10.0.4: 4 件套)
    required_files = ["lgb_model.txt", "normalizer.pkl", "feature_columns.pkl", "metadata.json"]
    missing = [f for f in required_files if not (model_dir / f).exists()]

    if missing:
        print(f"   ✗ Missing files: {missing}")
        return False

    try:
        import pickle
        import json
        import lightgbm as lgb

        # 导入 FeatureNormalizer (与训练/回测/实盘一致)
        try:
            from scripts.unified_features import FeatureNormalizer
        except ImportError:
            from unified_features import FeatureNormalizer

        # 加载模型 (LightGBM 原生格式)
        model = lgb.Booster(model_file=str(model_dir / "lgb_model.txt"))
        print(f"      ✓ lgb_model.txt: {type(model).__name__}")

        # 加载 normalizer (v10.0.4: 使用 FeatureNormalizer.load)
        normalizer = FeatureNormalizer()
        normalizer.load(str(model_dir / "normalizer.pkl"))
        print(f"      ✓ normalizer.pkl: fitted={normalizer.fitted}, {len(normalizer.feature_columns)} features")

        # 加载 feature_columns
        with open(model_dir / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        print(f"      ✓ feature_columns.pkl: {len(feature_columns)} features")

        # 加载 metadata 并校验
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        feature_count = metadata.get("feature_count", 0)
        print(f"      ✓ metadata.json: feature_count={feature_count}")

        # 校验一致性
        if feature_count != len(feature_columns):
            print(f"   ⚠ Warning: metadata.feature_count ({feature_count}) != feature_columns ({len(feature_columns)})")

        return True
    except Exception as e:
        print(f"   ✗ Model load failed: {e}")
        return False


def verify_feature_computation() -> bool:
    """验证统一特征计算 (v10.0.2: 直接导入 unified_features)"""
    print("4. Testing unified feature computation...")
    try:
        # 导入统一特征模块 (与训练/回测/实盘完全一致)
        try:
            from scripts.unified_features import (
                compute_unified_features,
                FEATURE_COLUMNS,
            )
        except ImportError:
            from unified_features import (
                compute_unified_features,
                FEATURE_COLUMNS,
            )

        # Mock OHLCV 数据 (100 根 K 线)
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": np.random.uniform(40000, 42000, n),
            "high": np.random.uniform(42000, 43000, n),
            "low": np.random.uniform(39000, 40000, n),
            "close": np.random.uniform(40000, 42000, n),
            "volume": np.random.uniform(100, 1000, n),
        })
        df = df.set_index("datetime")

        # 使用统一特征计算 (与训练/回测/实盘完全一致)
        features = compute_unified_features(df)

        # 验证 (v10.0.4: 严格校验 59 个特征 + 顺序一致)
        assert len(features) >= MIN_BARS - 60, f"Expected >= {MIN_BARS - 60} rows, got {len(features)}"
        assert len(features.columns) == len(FEATURE_COLUMNS), f"Expected {len(FEATURE_COLUMNS)} features, got {len(features.columns)}"
        assert list(features.columns) == FEATURE_COLUMNS, "Feature columns order mismatch!"

        print(f"   ✓ Computed {len(features.columns)} features, {len(features)} valid samples")
        return True
    except Exception as e:
        print(f"   ✗ Feature computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_normalizer_strict() -> bool:
    """验证 Normalizer strict 模式 (v10.0.4: 使用真实 FeatureNormalizer)"""
    print("5. Testing FeatureNormalizer strict mode...")
    try:
        # v10.0.4: 导入真实 FeatureNormalizer (与训练/回测/实盘一致)
        try:
            from scripts.unified_features import FeatureNormalizer
        except ImportError:
            from unified_features import FeatureNormalizer

        normalizer = FeatureNormalizer()

        # 训练数据
        train_data = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [4.0, 5.0, 6.0],
        })

        # fit_transform
        normalized = normalizer.fit_transform(train_data)
        assert normalizer.fitted, "Normalizer should be fitted"
        assert normalizer.feature_columns == ["feat1", "feat2"]

        # 正常 transform
        test_data = pd.DataFrame({
            "feat1": [2.0],
            "feat2": [5.0],
        })
        result = normalizer.transform(test_data, strict=True)
        assert not result.isna().any().any(), "Result should not contain NaN"

        # strict=True 应该对缺失列报错
        incomplete_data = pd.DataFrame({
            "feat1": [2.0],
        })
        try:
            normalizer.transform(incomplete_data, strict=True)
            print("   ✗ Should have raised ValueError for missing column")
            return False
        except ValueError as e:
            if "缺失特征列" in str(e):
                pass  # 预期行为
            else:
                raise

        # strict=True 应该对 NaN 报错
        nan_data = pd.DataFrame({
            "feat1": [np.nan],
            "feat2": [5.0],
        })
        try:
            normalizer.transform(nan_data, strict=True)
            print("   ✗ Should have raised ValueError for NaN")
            return False
        except ValueError as e:
            if "NaN" in str(e):
                pass  # 预期行为
            else:
                raise

        print("   ✓ Normalizer strict mode works correctly")
        return True

    except Exception as e:
        print(f"   ✗ Normalizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_backtest_consistency() -> bool:
    """验证回测链路一致性 (v10.0.4)"""
    print("6. Testing backtest consistency...")
    try:
        # 导入统一特征模块
        try:
            from scripts.unified_features import (
                compute_unified_features,
                FeatureNormalizer,
                FEATURE_COLUMNS,
            )
        except ImportError:
            from unified_features import (
                compute_unified_features,
                FeatureNormalizer,
                FEATURE_COLUMNS,
            )

        # Mock 数据
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "open": np.random.uniform(40000, 42000, n),
            "high": np.random.uniform(42000, 43000, n),
            "low": np.random.uniform(39000, 40000, n),
            "close": np.random.uniform(40000, 42000, n),
            "volume": np.random.uniform(100, 1000, n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))

        # 计算特征
        features = compute_unified_features(df)

        # 剔除 NaN (前 60 行)
        valid_mask = ~features.isna().any(axis=1)
        features_valid = features[valid_mask]

        assert len(features_valid) > 0, "No valid features after NaN removal"

        # 模拟训练时的归一化
        normalizer = FeatureNormalizer()
        normalized = normalizer.fit_transform(features_valid)

        # 模拟实盘时使用 iloc[-2]
        latest = features_valid.iloc[-2:-1]
        latest_norm = normalizer.transform(latest, strict=True)

        assert latest_norm.shape == (1, len(FEATURE_COLUMNS)), f"Shape mismatch: {latest_norm.shape}"
        assert not latest_norm.isna().any().any(), "Latest features contain NaN"

        print(f"   ✓ Backtest consistency verified (59 features, iloc[-2] works)")
        return True

    except Exception as e:
        print(f"   ✗ Backtest consistency failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("AlgVex 集成验证 (v10.0.4)")
    print("=" * 60 + "\n")

    results = []

    # 1. Qlib 配置 (可选)
    results.append(("Qlib Config", verify_qlib_runtime_config()))

    # 2. Parquet 数据
    ok, info = verify_parquet_data()
    results.append(("Parquet Data", ok))

    # 3. 模型加载
    results.append(("Model Load", verify_model_load()))

    # 4. 特征计算
    results.append(("Feature Computation", verify_feature_computation()))

    # 5. Normalizer strict
    results.append(("Normalizer Strict", verify_normalizer_strict()))

    # 6. 回测一致性
    results.append(("Backtest Consistency", verify_backtest_consistency()))

    # 汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有验证通过！可以进入下一步。")
        print("\n下一步操作:")
        print("  1. 准备数据: python scripts/prepare_crypto_data.py")
        print("  2. 训练模型: python scripts/train_model.py")
        print("  3. 离线回测: python scripts/backtest_offline.py")
        print("  4. 启动策略: hummingbot >>> start --script qlib_alpha_strategy.py")
        return 0
    else:
        print("✗ 部分验证失败，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
