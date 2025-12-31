"""
ExecutionModel 单元测试

测试统一成交模型的各项功能:
- 成交价格计算
- 手续费计算
- 滑点计算
- 订单执行
- 回测-实盘对齐验证
"""

import pytest
from datetime import datetime

from algvex.core.backtest.execution_model import (
    ExecutionModel,
    ExecutionConfig,
    FillResult,
    OrderSide,
    OrderType,
    create_backtest_execution_model,
    create_live_execution_model,
)
from shared.execution_models import VIPLevel


class TestExecutionConfig:
    """测试执行配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ExecutionConfig()
        assert config.exchange == "binance"
        assert config.vip_level == VIPLevel.VIP0
        assert config.slippage_model == "dynamic"
        assert config.base_slippage == 0.0001
        assert config.fill_price_type == "close_price"

    def test_custom_config(self):
        """测试自定义配置"""
        config = ExecutionConfig(
            exchange="okx",
            vip_level=VIPLevel.VIP1,
            maker_fee=0.0001,
            taker_fee=0.0003,
            base_slippage=0.0002,
        )
        assert config.exchange == "okx"
        assert config.vip_level == VIPLevel.VIP1
        assert config.maker_fee == 0.0001
        assert config.taker_fee == 0.0003


class TestExecutionModel:
    """测试执行模型"""

    @pytest.fixture
    def model(self):
        """创建测试用执行模型"""
        config = ExecutionConfig(
            exchange="binance",
            vip_level=VIPLevel.VIP0,
            taker_fee=0.0004,
            maker_fee=0.0002,
            base_slippage=0.0001,
        )
        return ExecutionModel(config)

    @pytest.fixture
    def market_data(self):
        """创建测试用市场数据"""
        return {
            "symbol": "BTCUSDT",
            "close_price": 50000.0,
            "last_price": 50000.0,
            "mark_price": 50010.0,
            "order_size_usd": 10000.0,
            "avg_daily_volume": 10_000_000.0,
        }

    def test_calculate_fill_price_market_buy(self, model, market_data):
        """测试市价买入成交价格"""
        fill_price = model.calculate_fill_price(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            market_data=market_data,
        )
        # 买入应该高于基准价 (加滑点)
        assert fill_price >= market_data["close_price"]

    def test_calculate_fill_price_market_sell(self, model, market_data):
        """测试市价卖出成交价格"""
        fill_price = model.calculate_fill_price(
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            market_data=market_data,
        )
        # 卖出应该低于基准价 (减滑点)
        assert fill_price <= market_data["close_price"]

    def test_calculate_fill_price_limit(self, model, market_data):
        """测试限价单成交价格"""
        limit_price = 49800.0
        fill_price = model.calculate_fill_price(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            market_data=market_data,
            limit_price=limit_price,
        )
        # 限价单使用限价
        assert fill_price == limit_price

    def test_calculate_fee_taker(self, model):
        """测试 Taker 手续费"""
        notional = 10000.0
        fee = model.calculate_fee(notional, is_maker=False)
        # VIP0 taker fee 约 0.04%
        assert fee > 0
        assert fee < notional * 0.01  # 应该小于 1%

    def test_calculate_fee_maker(self, model):
        """测试 Maker 手续费"""
        notional = 10000.0
        fee = model.calculate_fee(notional, is_maker=True)
        taker_fee = model.calculate_fee(notional, is_maker=False)
        # Maker 手续费应该低于 Taker
        assert fee <= taker_fee

    def test_calculate_slippage(self, model, market_data):
        """测试滑点计算"""
        slippage = model.calculate_slippage(
            symbol="BTCUSDT",
            order_size_usd=10000.0,
            market_data=market_data,
        )
        # 滑点应该在合理范围内
        assert slippage >= 0
        assert slippage <= model.config.max_slippage

    def test_execute_order_market(self, model, market_data):
        """测试执行市价单"""
        result = model.execute_order(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
            timestamp=datetime.now(),
        )
        assert isinstance(result, FillResult)
        assert result.filled is True
        assert result.fill_quantity == 0.1
        assert result.fill_price > 0
        assert result.fee > 0
        assert result.slippage >= 0

    def test_execute_order_limit(self, model, market_data):
        """测试执行限价单"""
        result = model.execute_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            market_data=market_data,
            limit_price=49500.0,
        )
        assert result.filled is True
        assert result.fill_price == 49500.0

    def test_execute_order_metadata(self, model, market_data):
        """测试执行订单元数据"""
        result = model.execute_order(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            market_data=market_data,
        )
        assert "symbol" in result.metadata
        assert result.metadata["symbol"] == "BTCUSDT"
        assert result.metadata["side"] == "buy"
        assert result.metadata["order_type"] == "market"


class TestExecutionModelAlignment:
    """测试回测-实盘对齐"""

    def test_backtest_model_config(self):
        """测试回测模型配置"""
        model = create_backtest_execution_model()
        assert model.config.fill_price_type == "close_price"

    def test_live_model_config(self):
        """测试实盘模型配置"""
        model = create_live_execution_model()
        assert model.config.fill_price_type == "last_price"

    def test_alignment_same_config(self):
        """测试相同配置的对齐验证"""
        model1 = create_backtest_execution_model()
        model2 = create_backtest_execution_model()

        result = model1.validate_alignment(model2)
        assert result["all_aligned"] is True

    def test_alignment_different_config(self):
        """测试不同配置的对齐验证"""
        backtest = create_backtest_execution_model()
        live = create_live_execution_model()

        result = backtest.validate_alignment(live)
        # fill_price 不同
        assert result["fill_price"]["aligned"] is False
        assert result["fill_price"]["self"] == "close_price"
        assert result["fill_price"]["other"] == "last_price"

    def test_get_impl_methods(self):
        """测试获取实现方法"""
        model = create_backtest_execution_model()

        assert model.get_fill_price_impl() == "close_price"
        assert model.get_partial_fill_impl() is True
        assert isinstance(model.get_fee_model_impl(), dict)
        assert isinstance(model.get_slippage_model_impl(), dict)


class TestFillResult:
    """测试成交结果"""

    def test_fill_result_creation(self):
        """测试创建成交结果"""
        result = FillResult(
            filled=True,
            fill_price=50000.0,
            fill_quantity=0.1,
            fee=2.0,
            slippage=0.0001,
            slippage_cost=0.5,
            total_cost=2.5,
        )
        assert result.filled is True
        assert result.fill_price == 50000.0
        assert result.fill_quantity == 0.1
        assert result.fee == 2.0
        assert result.total_cost == 2.5

    def test_fill_result_partial(self):
        """测试部分成交"""
        result = FillResult(
            filled=True,
            fill_price=50000.0,
            fill_quantity=0.05,
            fee=1.0,
            slippage=0.0001,
            slippage_cost=0.25,
            total_cost=1.25,
            partial=True,
            fill_ratio=0.5,
        )
        assert result.partial is True
        assert result.fill_ratio == 0.5

    def test_fill_result_metadata(self):
        """测试成交结果元数据"""
        result = FillResult(
            filled=True,
            fill_price=50000.0,
            fill_quantity=0.1,
            fee=2.0,
            slippage=0.0001,
            slippage_cost=0.5,
            total_cost=2.5,
            metadata={"order_id": "123"},
        )
        assert result.metadata["order_id"] == "123"
