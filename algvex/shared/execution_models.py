"""
AlgVex 执行模型 (P0-6)

功能:
- 动态滑点模型 - 考虑市场条件
- VIP费率模型 - 考虑等级和maker/taker
- 成交模型验证器 - 确保回测与实盘一致

目标: 让回测的成交模型与实盘尽可能一致，避免回测收益成为幻觉。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VIPLevel(Enum):
    """VIP等级"""
    VIP0 = "VIP0"
    VIP1 = "VIP1"
    VIP2 = "VIP2"
    VIP3 = "VIP3"
    VIP4 = "VIP4"
    VIP5 = "VIP5"
    VIP6 = "VIP6"
    VIP7 = "VIP7"
    VIP8 = "VIP8"
    VIP9 = "VIP9"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class MarketConditions:
    """市场条件"""
    avg_daily_volume: float  # 日均成交量 (USD)
    volatility: float  # 波动率 (年化)
    bid_ask_spread: float  # 买卖价差 (比例)
    depth_imbalance: float = 0.0  # 深度不平衡 (-1 到 1)
    recent_trades_velocity: float = 0.0  # 最近成交速度
    funding_rate: float = 0.0  # 当前资金费率


@dataclass
class SlippageEstimate:
    """滑点估计结果"""
    base_slippage: float  # 基础滑点
    size_impact: float  # 订单大小影响
    volatility_impact: float  # 波动率影响
    spread_impact: float  # 价差影响
    total_slippage: float  # 总滑点
    confidence: float  # 置信度 (0-1)


class DynamicSlippageModel:
    """
    动态滑点模型 - 考虑市场条件

    比静态 0.01% 更真实，考虑:
    - 订单大小 (大单冲击成本)
    - 市场波动率
    - 买卖价差
    - 流动性状况
    """

    # 默认参数
    DEFAULT_BASE_SLIPPAGE = 0.0001  # 0.01%
    DEFAULT_SIZE_IMPACT_FACTOR = 0.1  # 占日成交量比例的10%作为冲击成本
    DEFAULT_VOLATILITY_BASELINE = 0.02  # 假设正常波动率 2%
    MAX_SLIPPAGE = 0.01  # 最大滑点 1%

    def __init__(
        self,
        base_slippage: float = DEFAULT_BASE_SLIPPAGE,
        size_impact_factor: float = DEFAULT_SIZE_IMPACT_FACTOR,
        volatility_baseline: float = DEFAULT_VOLATILITY_BASELINE,
        max_slippage: float = MAX_SLIPPAGE,
    ):
        """
        初始化动态滑点模型

        Args:
            base_slippage: 基础滑点
            size_impact_factor: 订单大小影响因子
            volatility_baseline: 波动率基准
            max_slippage: 最大滑点
        """
        self.base_slippage = base_slippage
        self.size_impact_factor = size_impact_factor
        self.volatility_baseline = volatility_baseline
        self.max_slippage = max_slippage

        logger.info(f"DynamicSlippageModel initialized: base={base_slippage:.4%}")

    def estimate_slippage(
        self,
        symbol: str,
        order_size_usd: float,
        market_conditions: MarketConditions,
        order_type: OrderType = OrderType.MARKET,
    ) -> SlippageEstimate:
        """
        估计滑点 - 基于订单大小和市场条件

        Args:
            symbol: 交易对
            order_size_usd: 订单价值 (USD)
            market_conditions: 市场条件
            order_type: 订单类型

        Returns:
            滑点估计结果
        """
        # 1. 基础滑点
        base = self.base_slippage

        # 2. 订单大小影响 (大单冲击成本)
        if market_conditions.avg_daily_volume > 0:
            size_ratio = order_size_usd / market_conditions.avg_daily_volume
            size_impact = size_ratio * self.size_impact_factor
        else:
            size_impact = 0.001  # 无法评估时使用保守值

        # 3. 波动率影响
        if self.volatility_baseline > 0:
            vol_multiplier = market_conditions.volatility / self.volatility_baseline
            volatility_impact = base * (vol_multiplier - 1) if vol_multiplier > 1 else 0
        else:
            volatility_impact = 0

        # 4. 价差影响
        spread_impact = market_conditions.bid_ask_spread / 2

        # 5. 深度不平衡影响 (可选)
        depth_impact = abs(market_conditions.depth_imbalance) * 0.0001

        # 6. 计算总滑点
        total = base + size_impact + volatility_impact + spread_impact + depth_impact

        # 限制类型调整
        if order_type == OrderType.LIMIT:
            total *= 0.3  # 限价单滑点较小
        elif order_type in [OrderType.STOP_MARKET, OrderType.TAKE_PROFIT]:
            total *= 1.5  # 止损/止盈单滑点较大

        # 应用上限
        total = min(total, self.max_slippage)

        # 计算置信度 (基于数据质量)
        confidence = self._calculate_confidence(market_conditions)

        return SlippageEstimate(
            base_slippage=base,
            size_impact=size_impact,
            volatility_impact=volatility_impact,
            spread_impact=spread_impact,
            total_slippage=total,
            confidence=confidence,
        )

    def _calculate_confidence(self, conditions: MarketConditions) -> float:
        """计算估计置信度"""
        confidence = 1.0

        # 成交量太低，置信度降低
        if conditions.avg_daily_volume < 1000000:  # < 100万USD
            confidence *= 0.7

        # 波动率太高，置信度降低
        if conditions.volatility > 0.05:  # > 5%
            confidence *= 0.8

        # 价差太大，置信度降低
        if conditions.bid_ask_spread > 0.001:  # > 0.1%
            confidence *= 0.8

        return confidence

    def get_slippage_for_backtest(
        self,
        symbol: str,
        order_size_usd: float,
        avg_daily_volume: float,
        volatility: float = 0.02,
        spread: float = 0.0001,
    ) -> float:
        """
        回测使用的简化滑点接口

        Args:
            symbol: 交易对
            order_size_usd: 订单价值
            avg_daily_volume: 日均成交量
            volatility: 波动率
            spread: 买卖价差

        Returns:
            滑点比例
        """
        conditions = MarketConditions(
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            bid_ask_spread=spread,
        )
        estimate = self.estimate_slippage(symbol, order_size_usd, conditions)
        return estimate.total_slippage

    def calibrate_from_history(
        self,
        historical_trades: List[Dict],
    ) -> Dict[str, float]:
        """
        从历史交易数据校准模型

        Args:
            historical_trades: 历史交易列表，每个包含:
                - expected_price: 预期价格
                - actual_price: 实际成交价
                - order_size_usd: 订单大小
                - market_conditions: 市场条件

        Returns:
            校准后的参数
        """
        if not historical_trades:
            return {"base_slippage": self.base_slippage}

        actual_slippages = []
        predicted_slippages = []

        for trade in historical_trades:
            expected = trade.get("expected_price", 0)
            actual = trade.get("actual_price", 0)
            if expected > 0:
                actual_slip = abs(actual - expected) / expected
                actual_slippages.append(actual_slip)

                # 预测滑点
                conditions = trade.get("market_conditions")
                if conditions:
                    estimate = self.estimate_slippage(
                        trade.get("symbol", ""),
                        trade.get("order_size_usd", 0),
                        conditions,
                    )
                    predicted_slippages.append(estimate.total_slippage)

        if actual_slippages:
            avg_actual = sum(actual_slippages) / len(actual_slippages)

            # 调整基础滑点
            if predicted_slippages:
                avg_predicted = sum(predicted_slippages) / len(predicted_slippages)
                if avg_predicted > 0:
                    adjustment = avg_actual / avg_predicted
                    self.base_slippage *= adjustment

            return {
                "base_slippage": self.base_slippage,
                "avg_actual_slippage": avg_actual,
                "sample_size": len(actual_slippages),
            }

        return {"base_slippage": self.base_slippage}


class FeeModel:
    """
    费率模型 - 考虑VIP等级和maker/taker

    币安永续合约费率示例 (2024):
    - VIP0: maker 0.02%, taker 0.04%
    - VIP1: maker 0.016%, taker 0.04%
    - VIP2: maker 0.014%, taker 0.035%
    ...
    """

    # 币安永续合约费率表
    BINANCE_FEE_TIERS = {
        VIPLevel.VIP0: {"maker": 0.0002, "taker": 0.0004},
        VIPLevel.VIP1: {"maker": 0.00016, "taker": 0.0004},
        VIPLevel.VIP2: {"maker": 0.00014, "taker": 0.00035},
        VIPLevel.VIP3: {"maker": 0.00012, "taker": 0.00032},
        VIPLevel.VIP4: {"maker": 0.0001, "taker": 0.0003},
        VIPLevel.VIP5: {"maker": 0.00008, "taker": 0.00027},
        VIPLevel.VIP6: {"maker": 0.00006, "taker": 0.00025},
        VIPLevel.VIP7: {"maker": 0.00004, "taker": 0.00022},
        VIPLevel.VIP8: {"maker": 0.00002, "taker": 0.0002},
        VIPLevel.VIP9: {"maker": 0.0, "taker": 0.00017},
    }

    # 其他交易所费率表
    BYBIT_FEE_TIERS = {
        VIPLevel.VIP0: {"maker": 0.0001, "taker": 0.0006},
        VIPLevel.VIP1: {"maker": 0.00006, "taker": 0.0005},
        # ... 可扩展
    }

    OKX_FEE_TIERS = {
        VIPLevel.VIP0: {"maker": 0.0002, "taker": 0.0005},
        VIPLevel.VIP1: {"maker": 0.00015, "taker": 0.00045},
        # ... 可扩展
    }

    def __init__(
        self,
        exchange: str = "binance",
        vip_level: VIPLevel = VIPLevel.VIP0,
        custom_fees: Optional[Dict[str, float]] = None,
    ):
        """
        初始化费率模型

        Args:
            exchange: 交易所名称
            vip_level: VIP等级
            custom_fees: 自定义费率 {"maker": x, "taker": y}
        """
        self.exchange = exchange.lower()
        self.vip_level = vip_level
        self.custom_fees = custom_fees

        # 选择费率表
        if exchange.lower() == "binance":
            self.fee_tiers = self.BINANCE_FEE_TIERS
        elif exchange.lower() == "bybit":
            self.fee_tiers = self.BYBIT_FEE_TIERS
        elif exchange.lower() == "okx":
            self.fee_tiers = self.OKX_FEE_TIERS
        else:
            self.fee_tiers = self.BINANCE_FEE_TIERS  # 默认使用币安

        logger.info(f"FeeModel initialized: {exchange} {vip_level.value}")

    def get_fee(self, is_maker: bool) -> float:
        """
        获取费率

        Args:
            is_maker: 是否为 maker

        Returns:
            费率
        """
        if self.custom_fees:
            return self.custom_fees.get("maker" if is_maker else "taker", 0.0004)

        tier = self.fee_tiers.get(self.vip_level, self.fee_tiers[VIPLevel.VIP0])
        return tier["maker"] if is_maker else tier["taker"]

    def get_maker_fee(self) -> float:
        """获取 maker 费率"""
        return self.get_fee(is_maker=True)

    def get_taker_fee(self) -> float:
        """获取 taker 费率"""
        return self.get_fee(is_maker=False)

    def calculate_fee(
        self,
        order_value: float,
        is_maker: bool,
    ) -> float:
        """
        计算手续费

        Args:
            order_value: 订单价值
            is_maker: 是否为 maker

        Returns:
            手续费金额
        """
        fee_rate = self.get_fee(is_maker)
        return order_value * fee_rate

    def estimate_trade_cost(
        self,
        order_value: float,
        order_type: OrderType = OrderType.MARKET,
    ) -> Dict[str, float]:
        """
        估算交易成本

        Args:
            order_value: 订单价值
            order_type: 订单类型

        Returns:
            成本明细
        """
        # 市价单通常是 taker
        # 限价单可能是 maker 或 taker
        if order_type == OrderType.MARKET:
            is_maker = False
            maker_prob = 0.0
        elif order_type == OrderType.LIMIT:
            is_maker = True  # 假设限价单大概率成为 maker
            maker_prob = 0.7
        else:
            is_maker = False
            maker_prob = 0.0

        maker_fee = self.calculate_fee(order_value, is_maker=True)
        taker_fee = self.calculate_fee(order_value, is_maker=False)

        # 预期费用 (加权平均)
        expected_fee = maker_fee * maker_prob + taker_fee * (1 - maker_prob)

        return {
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "expected_fee": expected_fee,
            "maker_probability": maker_prob,
            "fee_rate": self.get_fee(is_maker),
        }

    def get_fee_summary(self) -> Dict[str, Any]:
        """获取费率摘要"""
        return {
            "exchange": self.exchange,
            "vip_level": self.vip_level.value,
            "maker_fee": self.get_maker_fee(),
            "taker_fee": self.get_taker_fee(),
        }


@dataclass
class AlignmentCheckItem:
    """对齐检查项"""
    item: str
    description: str
    backtest_value: Any
    live_value: Any
    is_aligned: bool
    severity: str = "high"  # high, medium, low


class ExecutionModelValidator:
    """
    成交模型验证器 - 确保回测与实盘一致

    检查项:
    - fill_price: 成交价格模型
    - partial_fill: 部分成交处理
    - fee_model: 费率模型
    - slippage_model: 滑点模型
    - reduce_only: 仅减仓订单处理
    - position_mode: 仓位模式
    - trigger_logic: 触发单逻辑
    - leverage_handling: 杠杆处理
    - liquidation_logic: 爆仓逻辑
    """

    # 需要对齐的成交模型要素
    ALIGNMENT_CHECKLIST = {
        "fill_price": {
            "description": "成交价格模型 (last vs close vs mid)",
            "severity": "high",
        },
        "partial_fill": {
            "description": "部分成交处理",
            "severity": "medium",
        },
        "fee_model": {
            "description": "费率模型 (maker/taker, VIP等级)",
            "severity": "high",
        },
        "slippage_model": {
            "description": "滑点模型 (静态 vs 动态 vs 冲击成本)",
            "severity": "high",
        },
        "reduce_only": {
            "description": "仅减仓订单处理",
            "severity": "medium",
        },
        "position_mode": {
            "description": "仓位模式 (单向 vs 双向)",
            "severity": "high",
        },
        "trigger_logic": {
            "description": "触发单逻辑 (止损/止盈触发条件)",
            "severity": "medium",
        },
        "leverage_handling": {
            "description": "杠杆处理 (保证金计算)",
            "severity": "high",
        },
        "liquidation_logic": {
            "description": "爆仓逻辑 (与交易所一致)",
            "severity": "high",
        },
    }

    def __init__(self):
        """初始化成交模型验证器"""
        self.check_items = list(self.ALIGNMENT_CHECKLIST.keys())
        logger.info("ExecutionModelValidator initialized")

    def validate_alignment(
        self,
        backtest_engine: Any,
        live_engine: Any,
    ) -> Tuple[bool, List[AlignmentCheckItem]]:
        """
        验证回测与实盘的成交模型对齐

        Args:
            backtest_engine: 回测引擎
            live_engine: 实盘引擎

        Returns:
            (是否全部对齐, 检查结果列表)
        """
        results = []
        all_aligned = True

        for item, info in self.ALIGNMENT_CHECKLIST.items():
            # 获取回测和实盘的实现
            bt_getter = f"get_{item}_impl"
            live_getter = f"get_{item}_impl"

            bt_value = None
            live_value = None

            if hasattr(backtest_engine, bt_getter):
                bt_value = getattr(backtest_engine, bt_getter)()
            elif hasattr(backtest_engine, item):
                bt_value = getattr(backtest_engine, item)

            if hasattr(live_engine, live_getter):
                live_value = getattr(live_engine, live_getter)()
            elif hasattr(live_engine, item):
                live_value = getattr(live_engine, item)

            # 比较
            is_aligned = self._compare_values(bt_value, live_value)

            if not is_aligned:
                all_aligned = False
                logger.warning(
                    f"Alignment mismatch: {item} - "
                    f"backtest={bt_value}, live={live_value}"
                )

            check_result = AlignmentCheckItem(
                item=item,
                description=info["description"],
                backtest_value=bt_value,
                live_value=live_value,
                is_aligned=is_aligned,
                severity=info["severity"],
            )
            results.append(check_result)

        return all_aligned, results

    def _compare_values(self, v1: Any, v2: Any) -> bool:
        """比较两个值是否相等"""
        if v1 is None or v2 is None:
            return v1 == v2

        if isinstance(v1, float) and isinstance(v2, float):
            return abs(v1 - v2) < 1e-8

        if isinstance(v1, dict) and isinstance(v2, dict):
            return v1 == v2

        return str(v1) == str(v2)

    def validate_with_config(
        self,
        backtest_config: Dict[str, Any],
        live_config: Dict[str, Any],
    ) -> Tuple[bool, List[AlignmentCheckItem]]:
        """
        使用配置验证对齐

        Args:
            backtest_config: 回测配置
            live_config: 实盘配置

        Returns:
            (是否全部对齐, 检查结果列表)
        """
        results = []
        all_aligned = True

        for item, info in self.ALIGNMENT_CHECKLIST.items():
            bt_value = backtest_config.get(item)
            live_value = live_config.get(item)

            is_aligned = self._compare_values(bt_value, live_value)

            if not is_aligned:
                all_aligned = False

            check_result = AlignmentCheckItem(
                item=item,
                description=info["description"],
                backtest_value=bt_value,
                live_value=live_value,
                is_aligned=is_aligned,
                severity=info["severity"],
            )
            results.append(check_result)

        return all_aligned, results

    def generate_report(
        self,
        results: List[AlignmentCheckItem],
    ) -> Dict[str, Any]:
        """
        生成对齐报告

        Args:
            results: 检查结果列表

        Returns:
            报告字典
        """
        total = len(results)
        aligned = sum(1 for r in results if r.is_aligned)
        misaligned = total - aligned

        high_severity_issues = [
            r for r in results
            if not r.is_aligned and r.severity == "high"
        ]

        return {
            "total_checks": total,
            "aligned": aligned,
            "misaligned": misaligned,
            "alignment_ratio": aligned / total if total > 0 else 0,
            "is_valid": misaligned == 0,
            "high_severity_issues": len(high_severity_issues),
            "details": [
                {
                    "item": r.item,
                    "description": r.description,
                    "backtest": r.backtest_value,
                    "live": r.live_value,
                    "aligned": r.is_aligned,
                    "severity": r.severity,
                }
                for r in results
            ],
            "issues": [
                {
                    "item": r.item,
                    "description": r.description,
                    "backtest": r.backtest_value,
                    "live": r.live_value,
                }
                for r in results if not r.is_aligned
            ],
        }

    def create_default_config(self) -> Dict[str, Any]:
        """创建默认的对齐配置"""
        return {
            "fill_price": "close_price",  # 使用收盘价成交
            "partial_fill": True,  # 支持部分成交
            "fee_model": {
                "exchange": "binance",
                "vip_level": "VIP0",
                "maker": 0.0002,
                "taker": 0.0004,
            },
            "slippage_model": {
                "type": "dynamic",
                "base": 0.0001,
                "max": 0.01,
            },
            "reduce_only": True,  # 支持 reduce_only
            "position_mode": "one_way",  # 单向持仓
            "trigger_logic": {
                "stop_loss": "mark_price",  # 使用标记价格触发
                "take_profit": "last_price",
            },
            "leverage_handling": {
                "max_leverage": 10,
                "cross_margin": True,
            },
            "liquidation_logic": {
                "use_mark_price": True,
                "maintenance_margin_rate": 0.005,
            },
        }


# 全局单例
_slippage_model: Optional[DynamicSlippageModel] = None
_fee_model: Optional[FeeModel] = None
_execution_validator: Optional[ExecutionModelValidator] = None


def get_slippage_model(**kwargs) -> DynamicSlippageModel:
    """获取全局 DynamicSlippageModel 实例"""
    global _slippage_model
    if _slippage_model is None:
        _slippage_model = DynamicSlippageModel(**kwargs)
    return _slippage_model


def get_fee_model(**kwargs) -> FeeModel:
    """获取全局 FeeModel 实例"""
    global _fee_model
    if _fee_model is None:
        _fee_model = FeeModel(**kwargs)
    return _fee_model


def get_execution_validator() -> ExecutionModelValidator:
    """获取全局 ExecutionModelValidator 实例"""
    global _execution_validator
    if _execution_validator is None:
        _execution_validator = ExecutionModelValidator()
    return _execution_validator


def reset_execution_models():
    """重置全局实例 (用于测试)"""
    global _slippage_model, _fee_model, _execution_validator
    _slippage_model = None
    _fee_model = None
    _execution_validator = None
