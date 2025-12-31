"""
AlgVex 价格语义统一器 (P0-2)

功能:
- 统一永续合约的价格类型定义
- 确保不同场景使用正确的价格类型
- 防止价格语义混淆导致的 PnL 计算错误

价格类型:
- mark_price: 标记价格 - 用于计算盈亏和强平
- index_price: 指数价格 - 多交易所加权
- last_price: 最新成交价 - 实际交易价格
- close_price: 收盘价 - K线收盘

使用场景映射:
- pnl_calculation: mark_price
- liquidation_check: mark_price
- entry_exit_signal: close_price
- order_execution: last_price
- backtest_fill: close_price
- funding_settlement: mark_price
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PriceType(Enum):
    """价格类型枚举"""
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"
    LAST_PRICE = "last_price"
    CLOSE_PRICE = "close_price"


class PriceScenario(Enum):
    """价格使用场景枚举"""
    PNL_CALCULATION = "pnl_calculation"
    LIQUIDATION_CHECK = "liquidation_check"
    ENTRY_EXIT_SIGNAL = "entry_exit_signal"
    ORDER_EXECUTION = "order_execution"
    BACKTEST_FILL = "backtest_fill"
    FUNDING_SETTLEMENT = "funding_settlement"
    MARGIN_CALCULATION = "margin_calculation"
    POSITION_VALUATION = "position_valuation"


@dataclass
class PriceData:
    """价格数据容器"""
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    last_price: Optional[float] = None
    close_price: Optional[float] = None
    timestamp: Optional[int] = None  # Unix timestamp in ms

    def get(self, price_type: PriceType) -> Optional[float]:
        """根据价格类型获取价格"""
        mapping = {
            PriceType.MARK_PRICE: self.mark_price,
            PriceType.INDEX_PRICE: self.index_price,
            PriceType.LAST_PRICE: self.last_price,
            PriceType.CLOSE_PRICE: self.close_price,
        }
        return mapping.get(price_type)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "last_price": self.last_price,
            "close_price": self.close_price,
            "timestamp": self.timestamp,
        }


class PriceSemantics:
    """
    价格语义统一器 - 永续合约专用

    确保不同场景使用正确的价格类型，防止语义混淆。

    使用方法:
        semantics = PriceSemantics()

        # 获取场景对应的价格类型
        price_type = semantics.get_price_type(PriceScenario.PNL_CALCULATION)

        # 直接获取价格
        price = semantics.get_price(PriceScenario.PNL_CALCULATION, price_data)
    """

    # 价格类型描述
    PRICE_TYPE_DESCRIPTIONS = {
        PriceType.MARK_PRICE: "标记价格 - 用于计算盈亏和强平",
        PriceType.INDEX_PRICE: "指数价格 - 多交易所加权",
        PriceType.LAST_PRICE: "最新成交价 - 实际交易价格",
        PriceType.CLOSE_PRICE: "收盘价 - K线收盘",
    }

    # 场景-价格映射 (必须严格遵守)
    PRICE_USAGE_MAP: Dict[PriceScenario, PriceType] = {
        PriceScenario.PNL_CALCULATION: PriceType.MARK_PRICE,
        PriceScenario.LIQUIDATION_CHECK: PriceType.MARK_PRICE,
        PriceScenario.ENTRY_EXIT_SIGNAL: PriceType.CLOSE_PRICE,
        PriceScenario.ORDER_EXECUTION: PriceType.LAST_PRICE,
        PriceScenario.BACKTEST_FILL: PriceType.CLOSE_PRICE,
        PriceScenario.FUNDING_SETTLEMENT: PriceType.MARK_PRICE,
        PriceScenario.MARGIN_CALCULATION: PriceType.MARK_PRICE,
        PriceScenario.POSITION_VALUATION: PriceType.MARK_PRICE,
    }

    def __init__(self, custom_mapping: Optional[Dict[PriceScenario, PriceType]] = None):
        """
        初始化价格语义统一器

        Args:
            custom_mapping: 自定义场景-价格映射 (会覆盖默认映射)
        """
        self.price_map = self.PRICE_USAGE_MAP.copy()
        if custom_mapping:
            self.price_map.update(custom_mapping)

        logger.info("PriceSemantics initialized")

    def get_price_type(self, scenario: PriceScenario) -> PriceType:
        """
        获取场景对应的价格类型

        Args:
            scenario: 使用场景

        Returns:
            对应的价格类型

        Raises:
            ValueError: 未知场景
        """
        if scenario not in self.price_map:
            raise ValueError(f"Unknown scenario: {scenario}")
        return self.price_map[scenario]

    def get_price(
        self,
        scenario: PriceScenario,
        price_data: PriceData,
        fallback: bool = False,
    ) -> float:
        """
        根据场景获取正确的价格

        Args:
            scenario: 使用场景
            price_data: 价格数据
            fallback: 是否允许回退到其他价格类型

        Returns:
            对应场景的价格

        Raises:
            ValueError: 价格不可用且不允许回退
        """
        price_type = self.get_price_type(scenario)
        price = price_data.get(price_type)

        if price is not None:
            return price

        if fallback:
            # 回退顺序: mark -> last -> close -> index
            fallback_order = [
                PriceType.MARK_PRICE,
                PriceType.LAST_PRICE,
                PriceType.CLOSE_PRICE,
                PriceType.INDEX_PRICE,
            ]
            for fb_type in fallback_order:
                fb_price = price_data.get(fb_type)
                if fb_price is not None:
                    logger.warning(
                        f"Price fallback: {scenario.value} requested {price_type.value}, "
                        f"using {fb_type.value} instead"
                    )
                    return fb_price

        raise ValueError(
            f"Price not available for scenario {scenario.value}: "
            f"required {price_type.value}"
        )

    def get_price_from_dict(
        self,
        scenario: PriceScenario,
        data: Dict[str, Any],
        fallback: bool = False,
    ) -> float:
        """
        从字典获取价格 (便捷方法)

        Args:
            scenario: 使用场景
            data: 包含价格的字典
            fallback: 是否允许回退

        Returns:
            对应场景的价格
        """
        price_data = PriceData(
            mark_price=data.get("mark_price"),
            index_price=data.get("index_price"),
            last_price=data.get("last_price"),
            close_price=data.get("close_price") or data.get("close"),
            timestamp=data.get("timestamp"),
        )
        return self.get_price(scenario, price_data, fallback)

    def validate_price_usage(
        self,
        scenario: PriceScenario,
        used_price_type: PriceType,
    ) -> bool:
        """
        验证价格使用是否正确

        Args:
            scenario: 使用场景
            used_price_type: 实际使用的价格类型

        Returns:
            是否正确
        """
        expected_type = self.get_price_type(scenario)
        is_correct = used_price_type == expected_type

        if not is_correct:
            logger.error(
                f"Price semantics violation! Scenario {scenario.value} "
                f"should use {expected_type.value}, but {used_price_type.value} was used"
            )

        return is_correct

    def calculate_pnl(
        self,
        entry_price: float,
        current_price_data: PriceData,
        quantity: float,
        side: str,  # "long" or "short"
    ) -> float:
        """
        计算 PnL (使用正确的价格语义)

        Args:
            entry_price: 入场价格
            current_price_data: 当前价格数据
            quantity: 持仓数量
            side: 持仓方向

        Returns:
            未实现盈亏
        """
        # PnL 计算必须使用 mark_price
        mark_price = self.get_price(PriceScenario.PNL_CALCULATION, current_price_data)

        if side.lower() == "long":
            pnl = (mark_price - entry_price) * quantity
        else:
            pnl = (entry_price - mark_price) * quantity

        return pnl

    def check_liquidation(
        self,
        position_value: float,
        margin: float,
        current_price_data: PriceData,
        maintenance_margin_rate: float = 0.005,  # 0.5%
    ) -> bool:
        """
        检查是否触发强平 (使用正确的价格语义)

        Args:
            position_value: 持仓价值
            margin: 保证金
            current_price_data: 当前价格数据
            maintenance_margin_rate: 维持保证金率

        Returns:
            是否触发强平
        """
        # 强平检查必须使用 mark_price
        mark_price = self.get_price(PriceScenario.LIQUIDATION_CHECK, current_price_data)

        # 计算所需维持保证金
        maintenance_margin = position_value * maintenance_margin_rate

        # 如果保证金低于维持保证金，触发强平
        return margin < maintenance_margin

    def get_mapping_report(self) -> Dict[str, str]:
        """获取当前的场景-价格映射报告"""
        return {
            scenario.value: price_type.value
            for scenario, price_type in self.price_map.items()
        }


# 全局单例
_price_semantics: Optional[PriceSemantics] = None


def get_price_semantics() -> PriceSemantics:
    """获取全局 PriceSemantics 实例"""
    global _price_semantics
    if _price_semantics is None:
        _price_semantics = PriceSemantics()
    return _price_semantics


def reset_price_semantics():
    """重置全局实例 (用于测试)"""
    global _price_semantics
    _price_semantics = None
