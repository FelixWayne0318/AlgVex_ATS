#!/usr/bin/env python
"""
模型训练脚本

用法:
    python scripts/train_model.py
    python scripts/train_model.py --symbol btcusdt --horizon 24
    python scripts/train_model.py --cross-validate --folds 5
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from qlib_adapter.data_handler import CryptoDataHandler
from qlib_adapter.feature_engine import CryptoFeatureEngine
from strategy.ml_strategy import MLStrategy


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level,
    )


def load_config() -> dict:
    """加载配置"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_symbols() -> list:
    """加载标的列表"""
    symbols_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    if symbols_path.exists():
        with open(symbols_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return [s["symbol"] for s in config.get("primary", []) if s.get("enabled", True)]
    return ["btcusdt", "ethusdt"]


def train_model(
    symbol: str,
    start_date: str,
    end_date: str,
    target_horizon: int = 24,
    feature_groups: list = None,
    cross_validate: bool = False,
    n_folds: int = 5,
    data_dir: str = None,
):
    """
    训练模型

    Args:
        symbol: 标的
        start_date: 开始日期
        end_date: 结束日期
        target_horizon: 预测周期
        feature_groups: 特征组
        cross_validate: 是否交叉验证
        n_folds: 交叉验证折数
        data_dir: 数据目录
    """
    if data_dir is None:
        data_dir = Path("~/.cryptoquant/data").expanduser()
    else:
        data_dir = Path(data_dir)

    if feature_groups is None:
        feature_groups = ["momentum", "volatility", "volume", "funding", "oi", "ls_ratio"]

    logger.info(f"Training model for {symbol}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Target horizon: {target_horizon} periods")
    logger.info(f"Feature groups: {feature_groups}")

    # 1. 加载数据
    logger.info("Loading data...")
    data_handler = CryptoDataHandler(data_dir=str(data_dir))

    try:
        df = data_handler.get_qlib_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            freq="1h",
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    if df.empty:
        logger.error("No data loaded!")
        return None

    logger.info(f"Loaded {len(df)} records")

    # 2. 计算特征
    logger.info("Calculating features...")
    feature_engine = CryptoFeatureEngine()
    df_features = feature_engine.calculate_all_features(df, groups=feature_groups)

    if df_features.empty:
        logger.error("Failed to calculate features!")
        return None

    logger.info(f"Generated {len(df_features.columns)} features")

    # 3. 创建标签
    logger.info("Creating labels...")
    strategy = MLStrategy(target_horizon=target_horizon)

    # 从特征数据中提取价格
    if "$close" in df_features.columns:
        price_col = "$close"
    elif "close" in df_features.columns:
        price_col = "close"
    else:
        logger.error("No price column found!")
        return None

    labels = strategy.create_labels(df_features, price_col=price_col, threshold=0.01)
    df_features["label"] = labels

    # 删除缺失值
    df_features = df_features.dropna()
    logger.info(f"Samples after cleaning: {len(df_features)}")

    if len(df_features) < 100:
        logger.error("Insufficient data for training!")
        return None

    # 4. 准备特征列表
    feature_cols = [c for c in df_features.columns if c not in ["label", price_col, "$open", "$high", "$low", "$volume"]]
    logger.info(f"Using {len(feature_cols)} features")

    # 5. 训练模型
    if cross_validate:
        logger.info(f"Running {n_folds}-fold cross-validation...")
        cv_results = strategy.cross_validate(
            X=df_features[feature_cols],
            y=df_features["label"],
            n_splits=n_folds,
            symbol=symbol,
        )

        logger.info("\nCross-validation results:")
        for k, v in cv_results["mean"].items():
            logger.info(f"  {k}: {v:.4f}")
    else:
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = strategy.prepare_data(
            df_features,
            feature_cols=feature_cols,
            label_col="label",
            train_ratio=0.8,
        )

        # 训练
        strategy.train(X_train, y_train, symbol=symbol)

        # 评估
        metrics = strategy.evaluate(X_test, y_test, symbol=symbol)

        logger.info("\nTest metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # 保存模型
        strategy.save_model(symbol)

        # 输出特征重要性
        logger.info("\nTop 10 important features:")
        importance = strategy.get_feature_importance(symbol, top_n=10)
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return strategy


def train_all_symbols(
    start_date: str,
    end_date: str,
    **kwargs,
):
    """训练所有标的"""
    symbols = load_symbols()
    logger.info(f"Training models for {len(symbols)} symbols")

    results = {}
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        try:
            strategy = train_model(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
            results[symbol] = "success" if strategy else "failed"
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}")
            results[symbol] = "error"

    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("Training Summary:")
    for symbol, status in results.items():
        logger.info(f"  {symbol}: {status}")


def main():
    parser = argparse.ArgumentParser(description="CryptoQuant Model Training")

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol to train (default: all enabled symbols)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Prediction horizon (periods)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="momentum,volatility,volume,funding,oi,ls_ratio",
        help="Feature groups (comma-separated)",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Use cross-validation",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    feature_groups = [f.strip() for f in args.features.split(",")]

    if args.symbol:
        train_model(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            target_horizon=args.horizon,
            feature_groups=feature_groups,
            cross_validate=args.cross_validate,
            n_folds=args.folds,
            data_dir=args.data_dir,
        )
    else:
        train_all_symbols(
            start_date=args.start,
            end_date=args.end,
            target_horizon=args.horizon,
            feature_groups=feature_groups,
            cross_validate=args.cross_validate,
            n_folds=args.folds,
            data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
