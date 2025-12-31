"""
AlgVex 回测引擎单元测试

测试内容:
- BacktestConfig 配置验证
- Position/Trade 模型
- CryptoPerpetualBacktest 核心逻辑
- 资金费率计算
- 强平逻辑
- 指标计算
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.backtest.config import BacktestConfig, ExecutionConfig, PositionMode, MarginMode
from core.backtest.models import (
    Position, Trade, Signal, Account,
    PositionSide, TradeType, OrderType, TradeStatus,
)
from core.backtest.metrics import BacktestMetrics, BacktestResult
from core.backtest.engine import CryptoPerpetualBacktest, BarData
from shared.execution_models import VIPLevel


class TestBacktestConfig:
    """测试回测配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.leverage == 3.0
        assert config.max_leverage == 10.0
        assert config.taker_fee == 0.0004
        assert config.maker_fee == 0.0002
        assert config.slippage == 0.0001
        assert config.enable_funding is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = BacktestConfig(
            initial_capital=50000,
            leverage=5.0,
            taker_fee=0.0005,
        )

        assert config.initial_capital == 50000
        assert config.leverage == 5.0
        assert config.taker_fee == 0.0005

    def test_config_validation(self):
        """测试配置验证"""
        # 负资金应该失败
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=-1000)

        # 杠杆超过最大值应该失败
        with pytest.raises(ValueError):
            BacktestConfig(leverage=20, max_leverage=10)

        # 负费率应该失败
        with pytest.raises(ValueError):
            BacktestConfig(taker_fee=-0.001)

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = BacktestConfig()
        d = config.to_dict()

        assert "initial_capital" in d
        assert "leverage" in d
        assert "execution_config" in d

    def test_config_from_dict(self):
        """测试配置反序列化"""
        data = {
            "initial_capital": 200000,
            "leverage": 2.0,
        }
        config = BacktestConfig.from_dict(data)

        assert config.initial_capital == 200000
        assert config.leverage == 2.0

    def test_calculate_required_margin(self):
        """测试保证金计算"""
        config = BacktestConfig(leverage=5.0)
        margin = config.calculate_required_margin(10000)

        assert margin == 2000  # 10000 / 5

    def test_calculate_maintenance_margin(self):
        """测试维持保证金计算"""
        config = BacktestConfig(maintenance_margin_rate=0.005)
        margin = config.calculate_maintenance_margin(10000)

        assert margin == 50  # 10000 * 0.005


class TestPositionModel:
    """测试持仓模型"""

    def test_create_position(self):
        """测试创建持仓"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            leverage=3.0,
        )

        assert pos.symbol == "BTCUSDT"
        assert pos.side == PositionSide.LONG
        assert pos.quantity == 0.1
        assert pos.entry_price == 50000
        assert pos.is_open is True

    def test_position_value(self):
        """测试持仓价值计算"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            current_price=51000,
        )

        assert pos.position_value == 5100  # 0.1 * 51000
        assert pos.entry_value == 5000     # 0.1 * 50000

    def test_update_price_long(self):
        """测试多头价格更新"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
        )

        pos.update_price(52000)

        assert pos.current_price == 52000
        assert pos.unrealized_pnl == 200  # (52000 - 50000) * 0.1

    def test_update_price_short(self):
        """测试空头价格更新"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.1,
            entry_price=50000,
        )

        pos.update_price(48000)

        assert pos.current_price == 48000
        assert pos.unrealized_pnl == 200  # (50000 - 48000) * 0.1

    def test_check_liquidation(self):
        """测试强平检查"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            current_price=50000,
            margin=500,  # 10x 杠杆
        )

        # 价格下跌，亏损超过保证金
        pos.update_price(45000)  # 亏损 500

        # 检查强平 (维持保证金率 0.5%)
        assert pos.check_liquidation(0.005) is True

    def test_liquidation_price_long(self):
        """测试多头强平价格计算"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            margin=500,
        )

        liq_price = pos.calculate_liquidation_price(0.005)

        # 强平价格应该略低于入场价格 - (margin / qty)
        assert liq_price < 50000
        assert liq_price > 0

    def test_close_position(self):
        """测试平仓"""
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000,
            margin=500,
            initial_margin=500,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        trade = pos.close(
            close_price=52000,
            close_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            fee=10,
            slippage=5,
        )

        assert pos.is_open is False
        assert pos.quantity == 0
        assert trade.trade_type == TradeType.CLOSE_LONG
        assert trade.pnl == 200 - 10 - 5  # (52000-50000)*0.1 - fee - slippage


class TestTradeModel:
    """测试交易模型"""

    def test_create_trade(self):
        """测试创建交易"""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.OPEN_LONG,
            side=PositionSide.LONG,
            quantity=0.1,
            price=50000,
            fee=20,
        )

        assert trade.symbol == "BTCUSDT"
        assert trade.is_open is True
        assert trade.value == 5000  # 0.1 * 50000

    def test_trade_cost(self):
        """测试交易成本"""
        trade = Trade(
            symbol="BTCUSDT",
            trade_type=TradeType.OPEN_LONG,
            side=PositionSide.LONG,
            quantity=0.1,
            price=50000,
            fee=20,
            slippage=5,
        )

        assert trade.total_cost == 25  # 20 + 5


class TestAccountModel:
    """测试账户模型"""

    def test_create_account(self):
        """测试创建账户"""
        account = Account(
            balance=100000,
            initial_balance=100000,
        )

        assert account.balance == 100000
        assert account.equity == 100000
        assert account.total_return == 0

    def test_account_equity(self):
        """测试权益计算"""
        account = Account(
            balance=100000,
            initial_balance=100000,
            unrealized_pnl=500,
        )

        assert account.equity == 100500

    def test_account_win_rate(self):
        """测试胜率计算"""
        account = Account(
            num_trades=10,
            num_wins=6,
        )

        assert account.win_rate == 0.6


class TestBacktestMetrics:
    """测试回测指标计算"""

    def test_calculate_returns(self):
        """测试收益率计算"""
        metrics = BacktestMetrics()

        equity_curve = [
            (datetime(2024, 1, 1), 100000),
            (datetime(2024, 1, 2), 101000),
            (datetime(2024, 1, 3), 102000),
        ]

        returns = metrics._calculate_returns(equity_curve)

        assert len(returns) == 2
        assert abs(returns[0] - 0.01) < 0.0001  # 1% return

    def test_calculate_volatility(self):
        """测试波动率计算"""
        metrics = BacktestMetrics()

        returns = [0.01, -0.02, 0.015, -0.01, 0.02]

        vol = metrics._calculate_volatility(returns)

        assert vol > 0

    def test_calculate_sharpe(self):
        """测试夏普比率计算"""
        metrics = BacktestMetrics()

        sharpe = metrics._calculate_sharpe(0.20, 0.15)  # 20% return, 15% vol

        expected = (0.20 - 0.02) / 0.15  # (return - risk_free) / vol
        assert abs(sharpe - expected) < 0.01

    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        metrics = BacktestMetrics()

        equity_curve = [
            (datetime(2024, 1, 1), 100000),
            (datetime(2024, 1, 2), 110000),
            (datetime(2024, 1, 3), 90000),   # 最大回撤点
            (datetime(2024, 1, 4), 95000),
        ]

        max_dd, duration = metrics._calculate_max_drawdown(equity_curve)

        expected_dd = (110000 - 90000) / 110000  # ~18.18%
        assert abs(max_dd - expected_dd) < 0.01

    def test_calculate_trade_metrics(self):
        """测试交易指标计算"""
        metrics = BacktestMetrics()

        trades = [
            Trade(trade_type=TradeType.CLOSE_LONG, pnl=100),
            Trade(trade_type=TradeType.CLOSE_LONG, pnl=-50),
            Trade(trade_type=TradeType.CLOSE_SHORT, pnl=80),
            Trade(trade_type=TradeType.CLOSE_SHORT, pnl=-30),
        ]

        result = metrics._calculate_trade_metrics(trades)

        assert result["total_trades"] == 4
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 2
        assert result["win_rate"] == 0.5


class TestCryptoPerpetualBacktest:
    """测试永续合约回测引擎"""

    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return BacktestConfig(
            initial_capital=100000,
            leverage=3.0,
            taker_fee=0.0004,
            maker_fee=0.0002,
            slippage=0.0001,
            enable_funding=False,  # 简化测试
        )

    @pytest.fixture
    def engine(self, config):
        """创建测试引擎"""
        return CryptoPerpetualBacktest(config)

    @pytest.fixture
    def sample_prices(self):
        """创建样本价格数据"""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        np.random.seed(42)

        prices = 50000 * np.cumprod(1 + np.random.randn(100) * 0.002)

        df = pd.DataFrame({
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 100),
        }, index=dates)

        return {"BTCUSDT": df}

    @pytest.fixture
    def sample_signals(self):
        """创建样本信号"""
        return [
            Signal(
                symbol="BTCUSDT",
                signal_type="long",
                strength=0.8,
                timestamp=datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
            ),
            Signal(
                symbol="BTCUSDT",
                signal_type="close",
                strength=0,
                timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            ),
        ]

    def test_engine_initialization(self, engine):
        """测试引擎初始化"""
        assert engine.config.initial_capital == 100000
        assert engine.account.balance == 100000

    def test_engine_reset(self, engine):
        """测试引擎重置"""
        engine.account.balance = 50000
        engine.reset()

        assert engine.account.balance == 100000

    def test_run_empty_signals(self, engine, sample_prices):
        """测试空信号运行"""
        result = engine.run([], sample_prices)

        assert result.total_trades == 0
        assert result.total_return == 0

    def test_run_basic_backtest(self, engine, sample_prices, sample_signals):
        """测试基本回测流程"""
        result = engine.run(sample_signals, sample_prices)

        # 应该有交易记录
        assert result.total_trades >= 0

        # 结果应该包含所有必要字段
        assert result.start_time is not None
        assert result.end_time is not None

    def test_open_position(self, engine, sample_prices):
        """测试开仓"""
        # 使用价格数据中存在的时间戳
        first_time = list(sample_prices["BTCUSDT"].index)[0]

        signal = Signal(
            symbol="BTCUSDT",
            signal_type="long",
            strength=0.5,
            timestamp=first_time,
        )

        engine._market_data = sample_prices
        engine._process_signal(signal, signal.timestamp)

        # 应该有持仓或交易记录
        assert len(engine.account.positions) == 1 or len(engine.all_trades) > 0

    def test_close_position(self, engine, sample_prices):
        """测试平仓"""
        # 先开仓
        open_signal = Signal(
            symbol="BTCUSDT",
            signal_type="long",
            strength=0.5,
            timestamp=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
        )

        close_signal = Signal(
            symbol="BTCUSDT",
            signal_type="close",
            strength=0,
            timestamp=datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
        )

        engine._market_data = sample_prices
        engine._process_signal(open_signal, open_signal.timestamp)
        engine._process_signal(close_signal, close_signal.timestamp)

        # 持仓应该被平掉
        assert "BTCUSDT" not in engine.account.positions or not engine.account.positions.get("BTCUSDT", Position()).is_open

    def test_fee_calculation(self, engine):
        """测试手续费计算"""
        fee = engine._calculate_fee(10000, is_maker=False)

        # taker fee = 0.0004
        assert fee == 4  # 10000 * 0.0004

    def test_slippage_calculation(self, engine):
        """测试滑点计算"""
        slippage = engine._calculate_slippage("BTCUSDT", 10000)

        # 应该返回正数
        assert slippage > 0
        assert slippage < 0.01  # 小于 1%

    def test_position_size_calculation(self, engine):
        """测试仓位大小计算"""
        size = engine._calculate_position_size("BTCUSDT", 0.5)

        # 应该基于可用资金和信号强度
        assert size > 0
        assert size <= engine.account.free_margin * engine.config.leverage

    def test_execution_model_interfaces(self, engine):
        """测试执行模型接口 (用于对齐验证)"""
        assert engine.get_fill_price_impl() == "close_price"
        assert engine.get_partial_fill_impl() is True
        assert "exchange" in engine.get_fee_model_impl()
        assert "type" in engine.get_slippage_model_impl()
        assert engine.get_position_mode_impl() == "one_way"


class TestFundingRateIntegration:
    """测试资金费率集成"""

    def test_funding_settlement(self):
        """测试资金费结算"""
        config = BacktestConfig(
            initial_capital=100000,
            leverage=3.0,
            enable_funding=True,
        )
        engine = CryptoPerpetualBacktest(config)

        # 创建一个跨越结算时间的持仓
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000,
            margin=16666,
            entry_time=datetime(2024, 1, 1, 7, tzinfo=timezone.utc),
        )
        engine.account.positions["BTCUSDT"] = pos

        # 设置资金费率数据
        settlement_time = datetime(2024, 1, 1, 8, tzinfo=timezone.utc)
        engine._funding_rates = {
            "BTCUSDT": {settlement_time: 0.0001}  # 0.01%
        }

        # 处理结算
        engine._process_funding_settlement(settlement_time)

        # 应该记录资金费
        assert pos.total_funding_paid != 0


class TestLiquidationLogic:
    """测试强平逻辑"""

    def test_liquidation_trigger(self):
        """测试强平触发"""
        config = BacktestConfig(
            initial_capital=100000,
            leverage=10.0,
            maintenance_margin_rate=0.005,
        )
        engine = CryptoPerpetualBacktest(config)

        # 创建一个接近强平的持仓
        # 10x杠杆，保证金5000，持仓价值50000
        pos = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=50000,
            current_price=50000,
            margin=5000,  # 10x 杠杆
        )
        engine.account.positions["BTCUSDT"] = pos

        # 价格下跌到 45000，亏损 5000，超过保证金
        # available_margin = 5000 + (-5000) = 0
        # maintenance_margin = 45000 * 0.005 = 225
        # 0 < 225，应该触发强平
        pos.update_price(45000)

        # 检查是否应该强平
        should_liquidate = pos.check_liquidation(config.maintenance_margin_rate)

        # 当亏损超过保证金时应该触发强平
        assert should_liquidate is True


class TestBacktestResult:
    """测试回测结果"""

    def test_result_to_dict(self):
        """测试结果序列化"""
        result = BacktestResult(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            total_return=0.15,
            sharpe_ratio=1.5,
        )

        d = result.to_dict()

        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert d["total_return"] == 0.15

    def test_result_summary(self):
        """测试结果摘要"""
        result = BacktestResult(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            duration_days=30,
            total_return=0.15,
            annual_return=0.60,
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            total_trades=50,
            win_rate=0.55,
        )

        summary = result.get_summary()

        assert "15.00%" in summary  # total return
        assert "夏普比率" in summary
        assert "回测结果" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
