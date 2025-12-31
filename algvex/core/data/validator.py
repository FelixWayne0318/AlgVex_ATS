"""
数据质量验证器

检查:
1. 缺失值
2. 异常值
3. 时间连续性
4. 数据一致性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    issues: List[Dict]
    stats: Dict


class DataValidator:
    """
    数据质量验证器

    在数据进入因子计算前进行质量检查
    """

    def __init__(
        self,
        max_missing_ratio: float = 0.1,  # 最大缺失率 10%
        max_gap_hours: int = 4,  # 最大允许间隔 4 小时
        outlier_std: float = 5.0,  # 异常值阈值 (5个标准差)
    ):
        self.max_missing_ratio = max_missing_ratio
        self.max_gap_hours = max_gap_hours
        self.outlier_std = outlier_std

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        全面验证数据质量

        Args:
            df: 待验证的 DataFrame

        Returns:
            ValidationResult
        """
        issues = []

        # 1. 检查缺失值
        missing_issues = self._check_missing(df)
        issues.extend(missing_issues)

        # 2. 检查时间连续性
        gap_issues = self._check_time_gaps(df)
        issues.extend(gap_issues)

        # 3. 检查异常值
        outlier_issues = self._check_outliers(df)
        issues.extend(outlier_issues)

        # 4. 检查数据一致性
        consistency_issues = self._check_consistency(df)
        issues.extend(consistency_issues)

        # 统计信息
        stats = self._compute_stats(df)

        # 判断是否通过
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        passed = len(critical_issues) == 0

        return ValidationResult(
            passed=passed,
            issues=issues,
            stats=stats,
        )

    def _check_missing(self, df: pd.DataFrame) -> List[Dict]:
        """检查缺失值"""
        issues = []

        # 关键字段
        critical_fields = ["$open", "$high", "$low", "$close", "$volume"]
        optional_fields = ["$funding_rate", "$open_interest", "$ls_ratio"]

        for col in critical_fields:
            if col not in df.columns:
                issues.append({
                    "type": "missing_column",
                    "column": col,
                    "severity": "critical",
                    "message": f"关键字段 {col} 不存在",
                })
                continue

            missing_ratio = df[col].isna().mean()
            if missing_ratio > self.max_missing_ratio:
                issues.append({
                    "type": "high_missing_ratio",
                    "column": col,
                    "severity": "critical",
                    "missing_ratio": missing_ratio,
                    "message": f"{col} 缺失率 {missing_ratio:.1%} 超过阈值 {self.max_missing_ratio:.1%}",
                })
            elif missing_ratio > 0:
                issues.append({
                    "type": "has_missing",
                    "column": col,
                    "severity": "warning",
                    "missing_ratio": missing_ratio,
                    "message": f"{col} 有 {missing_ratio:.1%} 缺失",
                })

        return issues

    def _check_time_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """检查时间连续性"""
        issues = []

        if df.index.nlevels == 2:  # MultiIndex
            # 按 instrument 检查
            for instrument in df.index.get_level_values("instrument").unique():
                inst_df = df.xs(instrument, level="instrument")
                gaps = self._find_gaps(inst_df.index.get_level_values(0))
                for gap in gaps:
                    issues.append({
                        "type": "time_gap",
                        "instrument": instrument,
                        "severity": "warning",
                        "gap_start": gap[0].isoformat(),
                        "gap_end": gap[1].isoformat(),
                        "gap_hours": gap[2],
                        "message": f"{instrument} 在 {gap[0]} 到 {gap[1]} 有 {gap[2]:.1f} 小时间隔",
                    })
        else:
            if "datetime" in df.columns:
                times = pd.to_datetime(df["datetime"])
            else:
                times = df.index

            gaps = self._find_gaps(times)
            for gap in gaps:
                issues.append({
                    "type": "time_gap",
                    "severity": "warning",
                    "gap_start": gap[0].isoformat(),
                    "gap_end": gap[1].isoformat(),
                    "gap_hours": gap[2],
                    "message": f"在 {gap[0]} 到 {gap[1]} 有 {gap[2]:.1f} 小时间隔",
                })

        return issues

    def _find_gaps(self, times: pd.DatetimeIndex) -> List[Tuple]:
        """查找时间间隔"""
        gaps = []
        times = pd.DatetimeIndex(times).sort_values()

        for i in range(1, len(times)):
            diff = (times[i] - times[i-1]).total_seconds() / 3600
            if diff > self.max_gap_hours:
                gaps.append((times[i-1], times[i], diff))

        return gaps

    def _check_outliers(self, df: pd.DataFrame) -> List[Dict]:
        """检查异常值"""
        issues = []

        price_cols = ["$open", "$high", "$low", "$close"]

        for col in price_cols:
            if col not in df.columns:
                continue

            values = df[col].dropna()
            if len(values) < 10:
                continue

            # 计算收益率的异常值
            returns = values.pct_change().dropna()
            mean_ret = returns.mean()
            std_ret = returns.std()

            outliers = returns[abs(returns - mean_ret) > self.outlier_std * std_ret]

            if len(outliers) > 0:
                issues.append({
                    "type": "outlier",
                    "column": col,
                    "severity": "warning",
                    "count": len(outliers),
                    "max_deviation": float(abs(outliers).max()),
                    "message": f"{col} 发现 {len(outliers)} 个异常值 (>{self.outlier_std} 标准差)",
                })

        return issues

    def _check_consistency(self, df: pd.DataFrame) -> List[Dict]:
        """检查数据一致性"""
        issues = []

        # 检查 OHLC 关系: High >= Low, High >= Open/Close, Low <= Open/Close
        if all(c in df.columns for c in ["$open", "$high", "$low", "$close"]):
            invalid_hl = (df["$high"] < df["$low"]).sum()
            if invalid_hl > 0:
                issues.append({
                    "type": "invalid_ohlc",
                    "severity": "critical",
                    "count": int(invalid_hl),
                    "message": f"发现 {invalid_hl} 条记录 High < Low",
                })

            invalid_h = ((df["$high"] < df["$open"]) | (df["$high"] < df["$close"])).sum()
            if invalid_h > 0:
                issues.append({
                    "type": "invalid_high",
                    "severity": "warning",
                    "count": int(invalid_h),
                    "message": f"发现 {invalid_h} 条记录 High < Open/Close",
                })

        # 检查成交量非负
        if "$volume" in df.columns:
            negative_vol = (df["$volume"] < 0).sum()
            if negative_vol > 0:
                issues.append({
                    "type": "negative_volume",
                    "severity": "critical",
                    "count": int(negative_vol),
                    "message": f"发现 {negative_vol} 条负成交量",
                })

        # 检查资金费率范围 (-0.1 到 0.1 合理)
        if "$funding_rate" in df.columns:
            extreme_funding = (abs(df["$funding_rate"]) > 0.1).sum()
            if extreme_funding > 0:
                issues.append({
                    "type": "extreme_funding",
                    "severity": "warning",
                    "count": int(extreme_funding),
                    "message": f"发现 {extreme_funding} 条极端资金费率 (>10%)",
                })

        return issues

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        """计算统计信息"""
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
        }

        # 时间范围
        if df.index.nlevels == 2:
            times = df.index.get_level_values(0)
            stats["instruments"] = list(df.index.get_level_values("instrument").unique())
        else:
            times = pd.to_datetime(df.get("datetime", df.index))
            stats["instruments"] = list(df.get("symbol", pd.Series()).unique())

        if len(times) > 0:
            stats["start_time"] = str(times.min())
            stats["end_time"] = str(times.max())
            stats["time_span_days"] = (times.max() - times.min()).days

        # 字段统计
        for col in ["$close", "$volume", "$funding_rate"]:
            if col in df.columns:
                stats[f"{col}_mean"] = float(df[col].mean())
                stats[f"{col}_std"] = float(df[col].std())
                stats[f"{col}_missing"] = float(df[col].isna().mean())

        return stats

    def fix_issues(self, df: pd.DataFrame, issues: List[Dict]) -> pd.DataFrame:
        """
        自动修复可修复的问题

        Args:
            df: 原始数据
            issues: 问题列表

        Returns:
            修复后的数据
        """
        df = df.copy()

        for issue in issues:
            if issue["type"] == "has_missing" and issue["severity"] == "warning":
                col = issue["column"]
                # 使用前向填充
                df[col] = df[col].fillna(method="ffill")
                logger.info(f"已填充 {col} 的缺失值")

            elif issue["type"] == "outlier":
                col = issue["column"]
                # 用滚动中位数替换异常值
                values = df[col]
                returns = values.pct_change()
                mean_ret = returns.mean()
                std_ret = returns.std()

                mask = abs(returns - mean_ret) > self.outlier_std * std_ret
                df.loc[mask, col] = values.rolling(5, center=True).median()
                logger.info(f"已修复 {col} 的 {mask.sum()} 个异常值")

        return df
