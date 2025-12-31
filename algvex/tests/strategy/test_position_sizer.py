"""
PositionSizer 单元测试

测试仓位管理器的核心功能:
- 仓位计算方法
- Kelly 准则
- 波动率调整
- 仓位限制
"""

import pytest
import numpy as np

from algvex.core.strategy import (
    PositionSizer,
    PositionSizeConfig,
    PositionSize,
    SizingMethod,
    create_conservative_sizer,
    create_moderate_sizer,
    create_aggressive_sizer,
)


class TestPositionSizeConfig:
    """测试仓位配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = PositionSizeConfig()
        assert config.method == SizingMethod.KELLY
        assert config.base_position == 0.1
        assert config.max_position == 0.3
        assert config.kelly_fraction == 0.5

    def test_custom_config(self):
        """测试自定义配置"""
        config = PositionSizeConfig(
            method=SizingMethod.FIXED,
            base_position=0.2,
            max_position=0.5,
        )
        assert config.method == SizingMethod.FIXED
        assert config.base_position == 0.2
        assert config.max_position == 0.5

    def test_string_method(self):
        """测试字符串方法转换"""
        config = PositionSizeConfig(method="volatility")
        assert config.method == SizingMethod.VOLATILITY

    def test_invalid_max_position(self):
        """测试无效最大仓位"""
        with pytest.raises(ValueError):
            PositionSizeConfig(max_position=1.5)

    def test_invalid_kelly_fraction(self):
        """测试无效 Kelly 分数"""
        with pytest.raises(ValueError):
            PositionSizeConfig(kelly_fraction=0)

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = PositionSizeConfig(base_position=0.15)
        data = config.to_dict()
        assert data["base_position"] == 0.15
        assert data["method"] == "kelly"


class TestPositionSize:
    """测试仓位结果"""

    def test_position_size_creation(self):
        """测试创建仓位结果"""
        size = PositionSize(
            size=0.2,
            notional=10000.0,
            quantity=0.2,
            method="kelly",
        )
        assert size.size == 0.2
        assert size.notional == 10000.0
        assert size.is_valid is True

    def test_position_size_to_dict(self):
        """测试结果转字典"""
        size = PositionSize(size=0.15, method="kelly")
        data = size.to_dict()
        assert data["size"] == 0.15
        assert data["method"] == "kelly"


class TestPositionSizer:
    """测试仓位管理器"""

    @pytest.fixture
    def kelly_sizer(self):
        """创建 Kelly 仓位管理器"""
        config = PositionSizeConfig(
            method=SizingMethod.KELLY,
            base_position=0.1,
            max_position=0.3,
            kelly_fraction=0.5,
        )
        return PositionSizer(config)

    @pytest.fixture
    def fixed_sizer(self):
        """创建固定仓位管理器"""
        config = PositionSizeConfig(
            method=SizingMethod.FIXED,
            base_position=0.1,
            max_position=0.3,
        )
        return PositionSizer(config)

    def test_sizer_init(self, kelly_sizer):
        """测试仓位管理器初始化"""
        assert kelly_sizer.config is not None
        assert kelly_sizer.config.method == SizingMethod.KELLY

    def test_fixed_position(self, fixed_sizer):
        """测试固定仓位计算"""
        result = fixed_sizer.calculate(signal_strength=1.0)
        assert result.size == 0.1  # base_position

    def test_fixed_position_with_strength(self, fixed_sizer):
        """测试固定仓位带信号强度"""
        result = fixed_sizer.calculate(signal_strength=0.5)
        assert result.size == 0.05  # base_position * 0.5

    def test_kelly_position(self, kelly_sizer):
        """测试 Kelly 仓位计算"""
        result = kelly_sizer.calculate(
            signal_strength=1.0,
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.02,
        )
        assert result.size > 0
        assert result.size <= kelly_sizer.config.max_position

    def test_kelly_position_default_values(self, kelly_sizer):
        """测试 Kelly 使用默认值"""
        result = kelly_sizer.calculate(signal_strength=1.0)
        assert result.size > 0

    def test_position_max_limit(self, kelly_sizer):
        """测试仓位最大限制"""
        result = kelly_sizer.calculate(
            signal_strength=1.0,
            win_rate=0.9,  # 高胜率
            avg_win=0.10,
            avg_loss=0.01,
        )
        # 应该被限制在 max_position
        assert result.size <= kelly_sizer.config.max_position

    def test_position_min_limit(self, kelly_sizer):
        """测试仓位最小限制"""
        result = kelly_sizer.calculate(
            signal_strength=0.01,  # 很弱的信号
        )
        assert result.size >= kelly_sizer.config.min_position

    def test_calculate_with_capital(self, fixed_sizer):
        """测试带资金的仓位计算"""
        result = fixed_sizer.calculate(
            signal_strength=1.0,
            capital=100000.0,
            price=50000.0,
        )
        assert result.notional == 10000.0  # 0.1 * 100000
        assert result.quantity == 0.2  # 10000 / 50000

    def test_calculate_quantity(self, fixed_sizer):
        """测试计算数量"""
        quantity = fixed_sizer.calculate_quantity(
            capital=100000.0,
            price=50000.0,
            position_size=0.1,
        )
        assert quantity == 0.2


class TestVolatilitySizer:
    """测试波动率仓位管理器"""

    @pytest.fixture
    def vol_sizer(self):
        """创建波动率仓位管理器"""
        config = PositionSizeConfig(
            method=SizingMethod.VOLATILITY,
            base_position=0.1,
            max_position=0.5,
            target_volatility=0.15,
        )
        return PositionSizer(config)

    def test_volatility_sizing_low_vol(self, vol_sizer):
        """测试低波动率时仓位增加"""
        result = vol_sizer.calculate(
            signal_strength=1.0,
            current_volatility=0.10,  # 低于目标
        )
        # 低波动率应该增加仓位
        assert result.size > vol_sizer.config.base_position

    def test_volatility_sizing_high_vol(self, vol_sizer):
        """测试高波动率时仓位减少"""
        result = vol_sizer.calculate(
            signal_strength=1.0,
            current_volatility=0.30,  # 高于目标
        )
        # 高波动率应该减少仓位
        assert result.size < vol_sizer.config.base_position


class TestPresetSizers:
    """测试预设仓位管理器"""

    def test_conservative_sizer(self):
        """测试保守型"""
        sizer = create_conservative_sizer()
        assert sizer.config.max_position == 0.15
        assert sizer.config.kelly_fraction == 0.25

    def test_moderate_sizer(self):
        """测试中等型"""
        sizer = create_moderate_sizer()
        assert sizer.config.max_position == 0.3
        assert sizer.config.kelly_fraction == 0.5

    def test_aggressive_sizer(self):
        """测试激进型"""
        sizer = create_aggressive_sizer()
        assert sizer.config.max_position == 0.5
        assert sizer.config.kelly_fraction == 0.75
