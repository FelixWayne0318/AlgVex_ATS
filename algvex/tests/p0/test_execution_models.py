"""
P0-6 验收测试: 执行模型

验收标准:
- 动态滑点模型考虑市场条件
- VIP费率模型正确
- 回测与实盘成交模型对齐
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.execution_models import (
    DynamicSlippageModel,
    FeeModel,
    ExecutionModelValidator,
    MarketConditions,
    VIPLevel,
    OrderType,
    get_slippage_model,
    get_fee_model,
    get_execution_validator,
    reset_execution_models,
)


class TestDynamicSlippageModel:
    """测试动态滑点模型"""

    def setup_method(self):
        """每个测试前重置"""
        reset_execution_models()

    def test_base_slippage(self):
        """测试基础滑点"""
        model = DynamicSlippageModel(base_slippage=0.0001)

        conditions = MarketConditions(
            avg_daily_volume=100_000_000,  # 1亿美元日成交量
            volatility=0.02,  # 2% 波动率
            bid_ask_spread=0.0001,  # 0.01% 价差
        )

        estimate = model.estimate_slippage(
            symbol="BTCUSDT",
            order_size_usd=1000,  # 小单
            market_conditions=conditions,
        )

        # 小单基础滑点应该较低
        assert estimate.base_slippage == 0.0001
        assert estimate.total_slippage < 0.001  # 小于 0.1%

    def test_size_impact(self):
        """测试订单大小影响"""
        model = DynamicSlippageModel()

        conditions = MarketConditions(
            avg_daily_volume=10_000_000,  # 1000万日成交量
            volatility=0.02,
            bid_ask_spread=0.0001,
        )

        # 小单
        small_estimate = model.estimate_slippage(
            symbol="BTCUSDT",
            order_size_usd=1000,
            market_conditions=conditions,
        )

        # 大单 (占日成交量 1%)
        large_estimate = model.estimate_slippage(
            symbol="BTCUSDT",
            order_size_usd=100000,  # 10万
            market_conditions=conditions,
        )

        # 大单滑点应该更大
        assert large_estimate.size_impact > small_estimate.size_impact
        assert large_estimate.total_slippage > small_estimate.total_slippage

    def test_volatility_impact(self):
        """测试波动率影响"""
        model = DynamicSlippageModel()

        # 正常波动率
        normal_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,  # 正常
            bid_ask_spread=0.0001,
        )

        # 高波动率
        high_vol_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.05,  # 高波动
            bid_ask_spread=0.0001,
        )

        normal_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, normal_conditions
        )
        high_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, high_vol_conditions
        )

        # 高波动时滑点应该更大
        assert high_estimate.volatility_impact > normal_estimate.volatility_impact

    def test_spread_impact(self):
        """测试价差影响"""
        model = DynamicSlippageModel()

        # 窄价差
        narrow_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,
            bid_ask_spread=0.0001,  # 0.01%
        )

        # 宽价差
        wide_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,
            bid_ask_spread=0.001,  # 0.1%
        )

        narrow_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, narrow_conditions
        )
        wide_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, wide_conditions
        )

        # 宽价差滑点更大
        assert wide_estimate.spread_impact > narrow_estimate.spread_impact

    def test_max_slippage_cap(self):
        """测试最大滑点上限"""
        model = DynamicSlippageModel(max_slippage=0.01)

        # 极端条件
        extreme_conditions = MarketConditions(
            avg_daily_volume=1_000_000,  # 低流动性
            volatility=0.10,  # 高波动
            bid_ask_spread=0.01,  # 大价差
        )

        estimate = model.estimate_slippage(
            "BTCUSDT", 500000, extreme_conditions  # 大单
        )

        # 应该被限制在最大值
        assert estimate.total_slippage <= 0.01

    def test_order_type_adjustment(self):
        """测试订单类型调整"""
        model = DynamicSlippageModel()

        conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,
            bid_ask_spread=0.0001,
        )

        # 市价单
        market_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, conditions, OrderType.MARKET
        )

        # 限价单
        limit_estimate = model.estimate_slippage(
            "BTCUSDT", 10000, conditions, OrderType.LIMIT
        )

        # 限价单滑点较小
        assert limit_estimate.total_slippage < market_estimate.total_slippage

    def test_backtest_interface(self):
        """测试回测接口"""
        model = DynamicSlippageModel()

        slippage = model.get_slippage_for_backtest(
            symbol="BTCUSDT",
            order_size_usd=10000,
            avg_daily_volume=100_000_000,
            volatility=0.02,
            spread=0.0001,
        )

        assert 0 < slippage < 0.01

    def test_global_singleton(self):
        """测试全局单例"""
        reset_execution_models()

        m1 = get_slippage_model()
        m2 = get_slippage_model()

        assert m1 is m2


class TestFeeModel:
    """测试费率模型"""

    def setup_method(self):
        """每个测试前重置"""
        reset_execution_models()

    def test_binance_vip0_fees(self):
        """测试币安 VIP0 费率"""
        model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)

        assert model.get_maker_fee() == 0.0002  # 0.02%
        assert model.get_taker_fee() == 0.0004  # 0.04%

    def test_binance_vip_tiers(self):
        """测试币安 VIP 等级"""
        model_vip0 = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)
        model_vip3 = FeeModel(exchange="binance", vip_level=VIPLevel.VIP3)
        model_vip9 = FeeModel(exchange="binance", vip_level=VIPLevel.VIP9)

        # VIP 等级越高，费率越低
        assert model_vip0.get_taker_fee() > model_vip3.get_taker_fee()
        assert model_vip3.get_taker_fee() > model_vip9.get_taker_fee()

        # VIP9 maker 费率为 0
        assert model_vip9.get_maker_fee() == 0.0

    def test_calculate_fee(self):
        """测试计算手续费"""
        model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)

        # 10000 USD 订单
        maker_fee = model.calculate_fee(order_value=10000, is_maker=True)
        taker_fee = model.calculate_fee(order_value=10000, is_maker=False)

        assert maker_fee == 2.0  # 10000 * 0.0002
        assert taker_fee == 4.0  # 10000 * 0.0004

    def test_custom_fees(self):
        """测试自定义费率"""
        model = FeeModel(custom_fees={"maker": 0.0001, "taker": 0.0003})

        assert model.get_maker_fee() == 0.0001
        assert model.get_taker_fee() == 0.0003

    def test_estimate_trade_cost(self):
        """测试估算交易成本"""
        model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)

        # 市价单 (taker)
        market_cost = model.estimate_trade_cost(
            order_value=10000,
            order_type=OrderType.MARKET,
        )
        assert market_cost["expected_fee"] == 4.0  # 全部 taker

        # 限价单 (可能是 maker)
        limit_cost = model.estimate_trade_cost(
            order_value=10000,
            order_type=OrderType.LIMIT,
        )
        # 预期费用应该介于 maker 和 taker 之间
        assert 2.0 <= limit_cost["expected_fee"] <= 4.0

    def test_fee_summary(self):
        """测试费率摘要"""
        model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP3)

        summary = model.get_fee_summary()

        assert summary["exchange"] == "binance"
        assert summary["vip_level"] == "VIP3"
        assert "maker_fee" in summary
        assert "taker_fee" in summary

    def test_global_singleton(self):
        """测试全局单例"""
        reset_execution_models()

        m1 = get_fee_model()
        m2 = get_fee_model()

        assert m1 is m2


class TestExecutionModelValidator:
    """测试成交模型验证器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_execution_models()

    def test_alignment_checklist(self):
        """测试对齐检查清单"""
        validator = ExecutionModelValidator()

        assert "fill_price" in validator.check_items
        assert "fee_model" in validator.check_items
        assert "slippage_model" in validator.check_items
        assert "position_mode" in validator.check_items

    def test_validate_with_config_aligned(self):
        """测试配置对齐"""
        validator = ExecutionModelValidator()

        # 相同配置
        config = {
            "fill_price": "close_price",
            "fee_model": {"maker": 0.0002, "taker": 0.0004},
            "slippage_model": {"type": "dynamic", "base": 0.0001},
            "position_mode": "one_way",
        }

        is_aligned, results = validator.validate_with_config(config, config)

        assert is_aligned is True
        for result in results:
            if result.backtest_value is not None:
                assert result.is_aligned is True

    def test_validate_with_config_misaligned(self):
        """测试配置不对齐"""
        validator = ExecutionModelValidator()

        backtest_config = {
            "fill_price": "close_price",
            "position_mode": "one_way",
        }

        live_config = {
            "fill_price": "last_price",  # 不同
            "position_mode": "hedge",  # 不同
        }

        is_aligned, results = validator.validate_with_config(
            backtest_config, live_config
        )

        assert is_aligned is False

        # 检查具体不对齐项
        misaligned = [r for r in results if not r.is_aligned]
        assert len(misaligned) >= 2

    def test_generate_report(self):
        """测试生成报告"""
        validator = ExecutionModelValidator()

        config = {"fill_price": "close_price", "position_mode": "one_way"}
        _, results = validator.validate_with_config(config, config)

        report = validator.generate_report(results)

        assert "total_checks" in report
        assert "aligned" in report
        assert "is_valid" in report
        assert "details" in report

    def test_default_config(self):
        """测试默认配置"""
        validator = ExecutionModelValidator()
        config = validator.create_default_config()

        assert "fill_price" in config
        assert "fee_model" in config
        assert "slippage_model" in config
        assert "position_mode" in config
        assert "liquidation_logic" in config

    def test_global_singleton(self):
        """测试全局单例"""
        reset_execution_models()

        v1 = get_execution_validator()
        v2 = get_execution_validator()

        assert v1 is v2


class TestExecutionModelsIntegration:
    """执行模型集成测试"""

    def test_complete_trade_cost(self):
        """测试完整交易成本计算"""
        slippage_model = DynamicSlippageModel()
        fee_model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)

        conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,
            bid_ask_spread=0.0001,
        )

        order_value = 10000

        # 计算滑点
        slippage = slippage_model.estimate_slippage(
            "BTCUSDT", order_value, conditions
        )
        slippage_cost = order_value * slippage.total_slippage

        # 计算手续费 (假设 taker)
        fee_cost = fee_model.calculate_fee(order_value, is_maker=False)

        # 总成本
        total_cost = slippage_cost + fee_cost

        # 成本应该合理
        assert total_cost > 0
        assert total_cost < order_value * 0.01  # 不超过1%

    def test_slippage_realistic(self):
        """测试滑点真实性"""
        model = DynamicSlippageModel()

        normal_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.02,
            bid_ask_spread=0.0001,
        )

        # 小单滑点应该很小
        small_slippage = model.estimate_slippage(
            "BTCUSDT", 1000, normal_conditions
        ).total_slippage
        assert small_slippage < 0.0005, "小单滑点过大"

        # 大单滑点应该更大
        large_slippage = model.estimate_slippage(
            "BTCUSDT", 1000000, normal_conditions
        ).total_slippage
        assert large_slippage > small_slippage, "大单滑点应大于小单"

        # 高波动时滑点应该更大
        volatile_conditions = MarketConditions(
            avg_daily_volume=100_000_000,
            volatility=0.05,  # 高波动
            bid_ask_spread=0.001,
        )
        volatile_slippage = model.estimate_slippage(
            "BTCUSDT", 1000, volatile_conditions
        ).total_slippage
        assert volatile_slippage > small_slippage, "高波动时滑点应更大"
