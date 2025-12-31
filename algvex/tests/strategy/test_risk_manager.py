"""
RiskManager 单元测试

测试风险管理器的核心功能:
- 交易检查
- 回撤控制
- 仓位限制
- 风险指标计算
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from algvex.core.strategy import (
    RiskManager,
    RiskConfig,
    RiskMetrics,
    RiskLevel,
    RiskAction,
    create_conservative_risk_manager,
    create_moderate_risk_manager,
    create_aggressive_risk_manager,
)


class TestRiskConfig:
    """测试风险配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = RiskConfig()
        assert config.max_drawdown == 0.15
        assert config.max_daily_loss == 0.05
        assert config.max_position == 0.5
        assert config.max_leverage == 5.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = RiskConfig(
            max_drawdown=0.10,
            max_daily_loss=0.03,
            max_position=0.3,
        )
        assert config.max_drawdown == 0.10
        assert config.max_daily_loss == 0.03
        assert config.max_position == 0.3

    def test_invalid_max_drawdown(self):
        """测试无效最大回撤"""
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown=1.5)

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = RiskConfig(max_drawdown=0.12)
        data = config.to_dict()
        assert data["max_drawdown"] == 0.12

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {"max_drawdown": 0.20, "max_daily_loss": 0.08}
        config = RiskConfig.from_dict(data)
        assert config.max_drawdown == 0.20


class TestRiskMetrics:
    """测试风险指标"""

    def test_metrics_creation(self):
        """测试创建风险指标"""
        metrics = RiskMetrics(
            current_drawdown=0.05,
            daily_pnl=-0.02,
            total_position=0.3,
            risk_level=RiskLevel.MEDIUM,
        )
        assert metrics.current_drawdown == 0.05
        assert metrics.daily_pnl == -0.02
        assert metrics.risk_level == RiskLevel.MEDIUM

    def test_metrics_to_dict(self):
        """测试指标转字典"""
        metrics = RiskMetrics(
            current_drawdown=0.08,
            daily_pnl=-0.01,
            total_position=0.2,
            risk_level=RiskLevel.LOW,
            action=RiskAction.ALLOW,
        )
        data = metrics.to_dict()
        assert data["current_drawdown"] == 0.08
        assert data["risk_level"] == "low"
        assert data["action"] == "allow"


class TestRiskManager:
    """测试风险管理器"""

    @pytest.fixture
    def manager(self):
        """创建风险管理器"""
        config = RiskConfig(
            max_drawdown=0.15,
            max_daily_loss=0.05,
            max_position=0.5,
            max_leverage=5.0,
        )
        return RiskManager(config)

    def test_manager_init(self, manager):
        """测试管理器初始化"""
        assert manager.config is not None
        assert manager.config.max_drawdown == 0.15

    def test_check_trade_allowed(self, manager):
        """测试允许交易"""
        can_trade, reason = manager.check_trade(
            current_drawdown=0.05,
            current_position=0.2,
            daily_pnl=-0.01,
        )
        assert can_trade is True
        assert "allowed" in reason.lower()

    def test_check_trade_max_drawdown(self, manager):
        """测试最大回撤阻止交易"""
        can_trade, reason = manager.check_trade(
            current_drawdown=0.20,  # 超过 15%
            current_position=0.1,
        )
        assert can_trade is False
        assert "drawdown" in reason.lower()

    def test_check_trade_daily_loss(self, manager):
        """测试日内亏损阻止交易"""
        can_trade, reason = manager.check_trade(
            current_drawdown=0.05,
            daily_pnl=-0.06,  # 超过 5%
        )
        assert can_trade is False
        assert "daily loss" in reason.lower()

    def test_check_trade_position_limit(self, manager):
        """测试仓位限制"""
        can_trade, reason = manager.check_trade(
            current_drawdown=0.05,
            current_position=0.4,
            proposed_position=0.2,  # 总共 0.6, 超过 0.5
        )
        assert can_trade is False
        assert "position" in reason.lower()

    def test_check_trade_leverage_limit(self, manager):
        """测试杠杆限制"""
        can_trade, reason = manager.check_trade(
            current_drawdown=0.05,
            current_position=0.2,
            leverage=6.0,  # 超过 5x
        )
        assert can_trade is False
        assert "leverage" in reason.lower()


class TestRiskManagerMetrics:
    """测试风险指标计算"""

    @pytest.fixture
    def manager(self):
        """创建风险管理器"""
        config = RiskConfig(
            max_drawdown=0.15,
            max_position=0.5,
        )
        return RiskManager(config)

    def test_calculate_metrics_basic(self, manager):
        """测试基本指标计算"""
        equity_curve = np.array([100, 105, 110, 108, 112])
        positions = {"BTCUSDT": 0.2, "ETHUSDT": 0.1}

        metrics = manager.calculate_metrics(
            equity_curve=equity_curve,
            positions=positions,
        )

        assert metrics.current_drawdown >= 0
        assert abs(metrics.total_position - 0.3) < 1e-9  # Floating point comparison
        assert metrics.risk_level is not None

    def test_calculate_metrics_high_drawdown(self, manager):
        """测试高回撤风险等级"""
        equity_curve = np.array([100, 110, 95, 85])  # 22% 回撤

        metrics = manager.calculate_metrics(
            equity_curve=equity_curve,
        )

        assert metrics.current_drawdown > 0.15
        assert metrics.risk_level == RiskLevel.CRITICAL
        assert metrics.action == RiskAction.CLOSE_ALL

    def test_calculate_metrics_medium_risk(self, manager):
        """测试中等风险等级"""
        equity_curve = np.array([100, 110, 100])  # 9% 回撤 (约60% of 15%)

        metrics = manager.calculate_metrics(
            equity_curve=equity_curve,
        )

        assert metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]


class TestPositionAdjustment:
    """测试仓位调整建议"""

    @pytest.fixture
    def manager(self):
        """创建风险管理器"""
        return RiskManager(RiskConfig())

    def test_adjustment_critical(self, manager):
        """测试危急时全平"""
        adjustment = manager.get_position_adjustment(
            current_position=0.3,
            risk_level=RiskLevel.CRITICAL,
        )
        assert adjustment == -0.3  # 全平

    def test_adjustment_high(self, manager):
        """测试高风险时减半"""
        adjustment = manager.get_position_adjustment(
            current_position=0.4,
            risk_level=RiskLevel.HIGH,
        )
        assert adjustment == -0.2  # 减半

    def test_adjustment_medium(self, manager):
        """测试中等风险时减25%"""
        adjustment = manager.get_position_adjustment(
            current_position=0.4,
            risk_level=RiskLevel.MEDIUM,
        )
        assert adjustment == -0.1  # 减25%

    def test_adjustment_low(self, manager):
        """测试低风险无调整"""
        adjustment = manager.get_position_adjustment(
            current_position=0.4,
            risk_level=RiskLevel.LOW,
        )
        assert adjustment == 0.0


class TestConcentrationCheck:
    """测试集中度检查"""

    @pytest.fixture
    def manager(self):
        """创建风险管理器"""
        config = RiskConfig(max_concentration=0.3)
        return RiskManager(config)

    def test_concentration_ok(self, manager):
        """测试集中度合规"""
        # Total = 0.5, each position <= 30%: 0.15/0.5=30%, 0.15/0.5=30%, 0.20/0.5=40%
        # Need: each position's concentration <= 30%
        positions = {"BTCUSDT": 0.15, "ETHUSDT": 0.15, "BNBUSDT": 0.20}
        # BTC = 0.15/0.5 = 30%, ETH = 30%, BNB = 40% - still fails
        # Let's use 4 equal positions: 0.125/0.5 = 25% each
        positions = {"BTCUSDT": 0.125, "ETHUSDT": 0.125, "BNBUSDT": 0.125, "SOLUSDT": 0.125}
        is_ok, over_limit = manager.check_concentration(positions)
        assert is_ok is True
        assert len(over_limit) == 0

    def test_concentration_exceeded(self, manager):
        """测试集中度超限"""
        positions = {"BTCUSDT": 0.5, "ETHUSDT": 0.1}  # BTC 占 83%
        is_ok, over_limit = manager.check_concentration(positions)
        assert is_ok is False
        assert "BTCUSDT" in over_limit


class TestPresetManagers:
    """测试预设风险管理器"""

    def test_conservative_manager(self):
        """测试保守型"""
        manager = create_conservative_risk_manager()
        assert manager.config.max_drawdown == 0.10
        assert manager.config.max_leverage == 2.0

    def test_moderate_manager(self):
        """测试中等型"""
        manager = create_moderate_risk_manager()
        assert manager.config.max_drawdown == 0.15
        assert manager.config.max_leverage == 5.0

    def test_aggressive_manager(self):
        """测试激进型"""
        manager = create_aggressive_risk_manager()
        assert manager.config.max_drawdown == 0.25
        assert manager.config.max_leverage == 10.0
