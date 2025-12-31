"""
AlgVex 因子研究工具

功能:
- 因子分析和可视化
- 因子相关性检查
- 因子 IC 分析
- 因子准入审查

使用方式:
    from research.factor_research import FactorResearch

    research = FactorResearch()
    report = research.analyze_factor(factor_values, labels)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorAnalysisReport:
    """因子分析报告"""
    factor_id: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    auto_correlation: float
    turnover: float
    coverage: float
    is_qualified: bool
    rejection_reasons: List[str]


class FactorResearch:
    """因子研究工具"""

    # 准入标准
    ADMISSION_CRITERIA = {
        "min_ic": 0.02,
        "min_ir": 0.3,
        "min_ic_positive_ratio": 0.5,
        "max_auto_correlation": 0.9,
        "min_coverage": 0.8,
        "max_correlation_with_existing": 0.7,
    }

    def __init__(self):
        """初始化因子研究工具"""
        self.existing_factors: Dict[str, pd.Series] = {}

    def analyze_factor(
        self,
        factor_values: pd.Series,
        labels: pd.Series,
        factor_id: str = "new_factor",
    ) -> FactorAnalysisReport:
        """
        分析因子

        Args:
            factor_values: 因子值 (MultiIndex: datetime, instrument)
            labels: 标签值 (未来收益)
            factor_id: 因子ID

        Returns:
            因子分析报告
        """
        rejection_reasons = []

        # 对齐数据
        common_idx = factor_values.index.intersection(labels.index)
        factor_aligned = factor_values.loc[common_idx]
        label_aligned = labels.loc[common_idx]

        # 1. 计算 IC
        df = pd.DataFrame({
            "factor": factor_aligned,
            "label": label_aligned,
        })

        daily_ic = df.groupby(level=0).apply(
            lambda x: x["factor"].corr(x["label"])
        ).dropna()

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = (daily_ic > 0).mean()

        # 检查 IC 标准
        if abs(ic_mean) < self.ADMISSION_CRITERIA["min_ic"]:
            rejection_reasons.append(
                f"IC均值 {ic_mean:.4f} < {self.ADMISSION_CRITERIA['min_ic']}"
            )
        if ir < self.ADMISSION_CRITERIA["min_ir"]:
            rejection_reasons.append(
                f"IR {ir:.4f} < {self.ADMISSION_CRITERIA['min_ir']}"
            )
        if ic_positive_ratio < self.ADMISSION_CRITERIA["min_ic_positive_ratio"]:
            rejection_reasons.append(
                f"IC正比例 {ic_positive_ratio:.2%} < {self.ADMISSION_CRITERIA['min_ic_positive_ratio']:.0%}"
            )

        # 2. 计算自相关性
        auto_corr = daily_ic.autocorr()
        if auto_corr > self.ADMISSION_CRITERIA["max_auto_correlation"]:
            rejection_reasons.append(
                f"自相关 {auto_corr:.4f} > {self.ADMISSION_CRITERIA['max_auto_correlation']}"
            )

        # 3. 计算换手率
        factor_rank = df.groupby(level=0)["factor"].rank(pct=True)
        turnover = factor_rank.groupby(level=1).diff().abs().mean()

        # 4. 计算覆盖率
        total_count = len(factor_values)
        valid_count = factor_values.notna().sum()
        coverage = valid_count / total_count if total_count > 0 else 0

        if coverage < self.ADMISSION_CRITERIA["min_coverage"]:
            rejection_reasons.append(
                f"覆盖率 {coverage:.2%} < {self.ADMISSION_CRITERIA['min_coverage']:.0%}"
            )

        # 5. 检查与现有因子的相关性
        for existing_id, existing_values in self.existing_factors.items():
            common = factor_values.index.intersection(existing_values.index)
            if len(common) > 100:
                corr = factor_values.loc[common].corr(existing_values.loc[common])
                if abs(corr) > self.ADMISSION_CRITERIA["max_correlation_with_existing"]:
                    rejection_reasons.append(
                        f"与 {existing_id} 相关性 {corr:.4f} > {self.ADMISSION_CRITERIA['max_correlation_with_existing']}"
                    )

        is_qualified = len(rejection_reasons) == 0

        return FactorAnalysisReport(
            factor_id=factor_id,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_positive_ratio=ic_positive_ratio,
            auto_correlation=auto_corr if not np.isnan(auto_corr) else 0,
            turnover=turnover,
            coverage=coverage,
            is_qualified=is_qualified,
            rejection_reasons=rejection_reasons,
        )

    def add_existing_factor(self, factor_id: str, values: pd.Series):
        """添加现有因子用于相关性检查"""
        self.existing_factors[factor_id] = values

    def compute_factor_correlation_matrix(
        self,
        factors: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factors: {因子ID: 因子值}

        Returns:
            相关性矩阵
        """
        factor_df = pd.DataFrame(factors)
        return factor_df.corr()

    def find_redundant_factors(
        self,
        factors: Dict[str, pd.Series],
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        """
        找出冗余因子对

        Args:
            factors: {因子ID: 因子值}
            threshold: 相关性阈值

        Returns:
            [(因子1, 因子2, 相关性), ...]
        """
        corr_matrix = self.compute_factor_correlation_matrix(factors)

        redundant = []
        factor_ids = list(factors.keys())

        for i, f1 in enumerate(factor_ids):
            for f2 in factor_ids[i + 1:]:
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) > threshold:
                    redundant.append((f1, f2, corr))

        return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)

    def generate_admission_report(
        self,
        report: FactorAnalysisReport,
    ) -> str:
        """
        生成因子准入报告

        Args:
            report: 因子分析报告

        Returns:
            Markdown 格式报告
        """
        status = "✅ 通过" if report.is_qualified else "❌ 未通过"

        md = f"""
# 因子准入审查报告

## 基本信息
- **因子ID**: {report.factor_id}
- **审查状态**: {status}
- **审查时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 指标详情

| 指标 | 值 | 标准 | 状态 |
|------|-----|------|------|
| IC均值 | {report.ic_mean:.4f} | > {self.ADMISSION_CRITERIA['min_ic']} | {'✅' if abs(report.ic_mean) >= self.ADMISSION_CRITERIA['min_ic'] else '❌'} |
| IC标准差 | {report.ic_std:.4f} | - | - |
| IR | {report.ir:.4f} | > {self.ADMISSION_CRITERIA['min_ir']} | {'✅' if report.ir >= self.ADMISSION_CRITERIA['min_ir'] else '❌'} |
| IC正比例 | {report.ic_positive_ratio:.2%} | > {self.ADMISSION_CRITERIA['min_ic_positive_ratio']:.0%} | {'✅' if report.ic_positive_ratio >= self.ADMISSION_CRITERIA['min_ic_positive_ratio'] else '❌'} |
| 自相关 | {report.auto_correlation:.4f} | < {self.ADMISSION_CRITERIA['max_auto_correlation']} | {'✅' if report.auto_correlation < self.ADMISSION_CRITERIA['max_auto_correlation'] else '❌'} |
| 换手率 | {report.turnover:.4f} | - | - |
| 覆盖率 | {report.coverage:.2%} | > {self.ADMISSION_CRITERIA['min_coverage']:.0%} | {'✅' if report.coverage >= self.ADMISSION_CRITERIA['min_coverage'] else '❌'} |

"""

        if report.rejection_reasons:
            md += "\n## 拒绝原因\n"
            for reason in report.rejection_reasons:
                md += f"- {reason}\n"

        return md


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    instruments = ["BTC", "ETH", "SOL"]

    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=["datetime", "instrument"]
    )

    # 创建有一定 IC 的因子
    labels = pd.Series(np.random.randn(len(index)) * 0.02, index=index)
    factor_values = labels * 0.3 + pd.Series(
        np.random.randn(len(index)) * 0.01, index=index
    )

    # 分析因子
    research = FactorResearch()
    report = research.analyze_factor(
        factor_values=factor_values,
        labels=labels,
        factor_id="test_momentum",
    )

    print("=== 因子分析报告 ===")
    print(f"因子ID: {report.factor_id}")
    print(f"是否合格: {report.is_qualified}")
    print(f"IC均值: {report.ic_mean:.4f}")
    print(f"IR: {report.ir:.4f}")
    print(f"IC正比例: {report.ic_positive_ratio:.2%}")

    if report.rejection_reasons:
        print("\n拒绝原因:")
        for reason in report.rejection_reasons:
            print(f"  - {reason}")

    # 生成 Markdown 报告
    print("\n" + "=" * 50)
    print(research.generate_admission_report(report))
