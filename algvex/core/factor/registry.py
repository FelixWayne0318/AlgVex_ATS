"""
AlgVex 因子注册表

管理所有 180 个核心因子 + 21 个 P1 扩展因子。

因子分类:
- 基础价量因子 (50个): 动量20 + 波动率15 + 成交量15
- 永续合约专用因子 (45个): 资金费率12 + 持仓量12 + 订单流21
- 期权/波动率因子 (20个): 隐含波动率10 + 期权持仓10
- 衍生品结构因子 (15个): 基差8 + 市场结构7
- 链上因子 (10个): 稳定币5 + DeFi TVL 5
- 情绪因子 (10个): Fear&Greed 5 + Google Trends 5
- 宏观关联因子 (15个): 美元/利率8 + 风险资产7
- 复合/ML因子 (15个)

P1扩展 (21个): L2深度8 + 清算5 + 多交易所Basis 8
"""

from typing import Dict, List, Optional, Type

from .base import BaseFactor, FactorFamily, FactorMetadata


class FactorRegistry:
    """
    因子注册表

    单例模式管理所有因子。

    使用示例:
        registry = FactorRegistry.get_instance()
        factor = registry.get_factor("return_1h")
        all_factors = registry.get_all_factors()
    """

    _instance = None

    def __init__(self):
        self._factors: Dict[str, BaseFactor] = {}
        self._factor_classes: Dict[str, Type[BaseFactor]] = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "FactorRegistry":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化因子注册表"""
        if self._initialized:
            return

        # 导入所有因子模块
        from .momentum import MOMENTUM_FACTORS
        from .volatility import VOLATILITY_FACTORS
        from .perpetual import PERPETUAL_FACTORS

        # 注册动量因子
        for factor_class in MOMENTUM_FACTORS:
            self.register(factor_class)

        # 注册波动率因子
        for factor_class in VOLATILITY_FACTORS:
            self.register(factor_class)

        # 注册永续合约因子
        for factor_class in PERPETUAL_FACTORS:
            self.register(factor_class)

        # 注册扩展因子模块 (每个因子单独处理错误)
        self._register_module_factors("volume", "VOLUME_FACTORS")
        self._register_module_factors("options", "OPTIONS_FACTORS")
        self._register_module_factors("derivatives", "DERIVATIVES_FACTORS")
        self._register_module_factors("onchain", "ONCHAIN_FACTORS")
        self._register_module_factors("sentiment", "SENTIMENT_FACTORS")
        self._register_module_factors("macro", "MACRO_FACTORS")
        self._register_module_factors("composite", "COMPOSITE_FACTORS")
        self._register_module_factors("p1_extension", "P1_EXTENSION_FACTORS")

        self._initialized = True

    def _register_module_factors(self, module_name: str, list_name: str):
        """注册模块因子，每个因子单独处理错误"""
        try:
            import importlib
            mod = importlib.import_module(f".{module_name}", package="algvex.core.factor")
            factors = getattr(mod, list_name, [])
            for factor_class in factors:
                try:
                    self.register(factor_class)
                except ValueError:
                    # 因子ID冲突，跳过
                    pass
        except ImportError:
            # 模块不存在，跳过
            pass

    def register(self, factor_class: Type[BaseFactor]):
        """
        注册因子类

        Args:
            factor_class: 因子类
        """
        instance = factor_class()
        metadata = instance.get_metadata()
        factor_id = metadata.factor_id

        if factor_id in self._factor_classes:
            raise ValueError(f"Factor {factor_id} already registered")

        self._factor_classes[factor_id] = factor_class
        self._factors[factor_id] = instance

    def get_factor(self, factor_id: str) -> Optional[BaseFactor]:
        """
        获取因子实例

        Args:
            factor_id: 因子ID

        Returns:
            因子实例
        """
        return self._factors.get(factor_id)

    def get_factor_class(self, factor_id: str) -> Optional[Type[BaseFactor]]:
        """获取因子类"""
        return self._factor_classes.get(factor_id)

    def get_all_factors(self) -> List[BaseFactor]:
        """获取所有因子实例"""
        return list(self._factors.values())

    def get_factors_by_family(self, family: FactorFamily) -> List[BaseFactor]:
        """
        按因子族获取因子

        Args:
            family: 因子族

        Returns:
            该族的所有因子
        """
        return [f for f in self._factors.values() if f.family == family]

    def get_mvp_factors(self) -> List[BaseFactor]:
        """获取MVP因子"""
        return [f for f in self._factors.values() if f.get_metadata().is_mvp]

    def get_all_factor_ids(self) -> List[str]:
        """获取所有因子ID"""
        return list(self._factors.keys())

    def get_metadata(self, factor_id: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        factor = self.get_factor(factor_id)
        if factor:
            return factor.get_metadata()
        return None

    def get_all_metadata(self) -> Dict[str, FactorMetadata]:
        """获取所有因子的元数据"""
        return {fid: f.get_metadata() for fid, f in self._factors.items()}

    def count(self) -> int:
        """获取因子总数"""
        return len(self._factors)

    def count_by_family(self) -> Dict[str, int]:
        """按因子族统计数量"""
        counts = {}
        for factor in self._factors.values():
            family = factor.family.value
            counts[family] = counts.get(family, 0) + 1
        return counts

    def summary(self) -> str:
        """生成因子摘要"""
        counts = self.count_by_family()
        mvp_count = len(self.get_mvp_factors())

        lines = [
            "=" * 50,
            "AlgVex 因子注册表摘要",
            "=" * 50,
            f"总因子数: {self.count()}",
            f"MVP因子数: {mvp_count}",
            "",
            "按因子族统计:",
        ]

        for family, count in sorted(counts.items()):
            lines.append(f"  - {family}: {count}")

        return "\n".join(lines)


# 因子目录 (用于文档和参考)
FACTOR_CATALOG = {
    "price_volume": {
        "name": "基础价量因子",
        "count": 50,
        "subcategories": {
            "momentum": 20,
            "volatility": 15,
            "volume": 15,
        },
    },
    "perpetual": {
        "name": "永续合约专用因子",
        "count": 45,
        "subcategories": {
            "funding": 12,
            "open_interest": 12,
            "order_flow": 21,
        },
    },
    "options": {
        "name": "期权/波动率因子",
        "count": 20,
        "subcategories": {
            "implied_vol": 10,
            "option_positions": 10,
        },
    },
    "derivatives": {
        "name": "衍生品结构因子",
        "count": 15,
        "subcategories": {
            "basis": 8,
            "market_structure": 7,
        },
    },
    "on_chain": {
        "name": "链上因子",
        "count": 10,
        "subcategories": {
            "stablecoin": 5,
            "defi_tvl": 5,
        },
    },
    "sentiment": {
        "name": "情绪因子",
        "count": 10,
        "subcategories": {
            "fear_greed": 5,
            "google_trends": 5,
        },
    },
    "macro": {
        "name": "宏观关联因子",
        "count": 15,
        "subcategories": {
            "usd_rates": 8,
            "risk_assets": 7,
        },
    },
    "composite": {
        "name": "复合/ML因子",
        "count": 15,
        "subcategories": {},
    },
}


def get_factor_catalog_summary() -> str:
    """生成因子目录摘要"""
    lines = [
        "=" * 60,
        "AlgVex 因子目录 (180 核心 + 21 P1扩展 = 201 总计)",
        "=" * 60,
    ]

    total = 0
    for cat_id, cat_info in FACTOR_CATALOG.items():
        count = cat_info["count"]
        total += count
        lines.append(f"\n{cat_info['name']} ({count}个)")
        lines.append("-" * 40)
        for sub_name, sub_count in cat_info.get("subcategories", {}).items():
            lines.append(f"  - {sub_name}: {sub_count}")

    lines.append(f"\n{'=' * 60}")
    lines.append(f"核心因子总计: {total}")
    lines.append(f"P1扩展因子: 21 (L2深度8 + 清算5 + 多交易所Basis8)")
    lines.append(f"总计: {total + 21}")

    return "\n".join(lines)
