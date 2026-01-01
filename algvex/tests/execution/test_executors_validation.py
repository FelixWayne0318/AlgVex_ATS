"""
执行器输入验证测试

测试 TWAPExecutor 和 GridExecutor 的输入验证逻辑。
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock

# 模拟依赖
import sys
sys.modules['aiohttp'] = MagicMock()

from algvex.core.execution.exchange_connectors import OrderSide, OrderType


class MockConnector:
    """模拟连接器"""
    pass


class TestTWAPExecutorValidation:
    """TWAP 执行器验证测试"""

    def setup_method(self):
        """测试前准备"""
        self.connector = MockConnector()

    def test_validates_num_slices_minimum(self):
        """测试 num_slices 最小值验证"""
        from algvex.core.execution.executors import TWAPExecutor

        with pytest.raises(ValueError, match="num_slices must be >= 1"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("1.0"),
                duration_minutes=60,
                num_slices=0
            )

    def test_validates_duration_minutes_minimum(self):
        """测试 duration_minutes 最小值验证"""
        from algvex.core.execution.executors import TWAPExecutor

        with pytest.raises(ValueError, match="duration_minutes must be >= 1"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("1.0"),
                duration_minutes=0,
                num_slices=12
            )

    def test_validates_total_quantity_positive(self):
        """测试 total_quantity 必须为正数"""
        from algvex.core.execution.executors import TWAPExecutor

        with pytest.raises(ValueError, match="total_quantity must be > 0"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("0"),
                duration_minutes=60,
                num_slices=12
            )

        with pytest.raises(ValueError, match="total_quantity must be > 0"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("-1.0"),
                duration_minutes=60,
                num_slices=12
            )

    def test_validates_max_deviation_range(self):
        """测试 max_deviation 范围验证 [0, 1]"""
        from algvex.core.execution.executors import TWAPExecutor

        with pytest.raises(ValueError, match="max_deviation must be in"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("1.0"),
                duration_minutes=60,
                num_slices=12,
                max_deviation=-0.1
            )

        with pytest.raises(ValueError, match="max_deviation must be in"):
            TWAPExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("1.0"),
                duration_minutes=60,
                num_slices=12,
                max_deviation=1.5
            )

    def test_valid_parameters_accepted(self):
        """测试有效参数被接受"""
        from algvex.core.execution.executors import TWAPExecutor

        # 不应该抛出异常
        executor = TWAPExecutor(
            connector=self.connector,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            total_quantity=Decimal("1.0"),
            duration_minutes=60,
            num_slices=12,
            max_deviation=0.1
        )
        assert executor.num_slices == 12
        assert executor.duration_minutes == 60


class TestGridExecutorValidation:
    """Grid 执行器验证测试"""

    def setup_method(self):
        """测试前准备"""
        self.connector = MockConnector()

    def test_validates_num_grids_minimum(self):
        """测试 num_grids 最小值验证 (必须 >= 2)"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="num_grids must be >= 2"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("40000"),
                upper_price=Decimal("50000"),
                num_grids=1
            )

    def test_validates_lower_price_positive(self):
        """测试 lower_price 必须为正数"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="lower_price must be > 0"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("0"),
                upper_price=Decimal("50000"),
                num_grids=10
            )

        with pytest.raises(ValueError, match="lower_price must be > 0"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("-1000"),
                upper_price=Decimal("50000"),
                num_grids=10
            )

    def test_validates_upper_price_positive(self):
        """测试 upper_price 必须为正数"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="upper_price must be > 0"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("40000"),
                upper_price=Decimal("0"),
                num_grids=10
            )

    def test_validates_price_order(self):
        """测试 lower_price 必须小于 upper_price"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="lower_price.*must be < upper_price"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("50000"),
                upper_price=Decimal("40000"),
                num_grids=10
            )

        # 相等也不行
        with pytest.raises(ValueError, match="lower_price.*must be < upper_price"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("45000"),
                upper_price=Decimal("45000"),
                num_grids=10
            )

    def test_validates_total_quantity_positive(self):
        """测试 total_quantity 必须为正数"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="total_quantity must be > 0"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("0"),
                lower_price=Decimal("40000"),
                upper_price=Decimal("50000"),
                num_grids=10
            )

    def test_validates_grid_type(self):
        """测试 grid_type 必须是 'arithmetic' 或 'geometric'"""
        from algvex.core.execution.executors import GridExecutor

        with pytest.raises(ValueError, match="grid_type must be 'arithmetic' or 'geometric'"):
            GridExecutor(
                connector=self.connector,
                symbol="BTCUSDT",
                total_quantity=Decimal("1.0"),
                lower_price=Decimal("40000"),
                upper_price=Decimal("50000"),
                num_grids=10,
                grid_type="invalid"
            )

    def test_valid_arithmetic_grid(self):
        """测试有效的等差网格参数"""
        from algvex.core.execution.executors import GridExecutor

        executor = GridExecutor(
            connector=self.connector,
            symbol="BTCUSDT",
            total_quantity=Decimal("1.0"),
            lower_price=Decimal("40000"),
            upper_price=Decimal("50000"),
            num_grids=10,
            grid_type="arithmetic"
        )
        assert executor.num_grids == 10
        assert executor.grid_type == "arithmetic"

    def test_valid_geometric_grid(self):
        """测试有效的等比网格参数"""
        from algvex.core.execution.executors import GridExecutor

        executor = GridExecutor(
            connector=self.connector,
            symbol="BTCUSDT",
            total_quantity=Decimal("1.0"),
            lower_price=Decimal("40000"),
            upper_price=Decimal("50000"),
            num_grids=10,
            grid_type="geometric"
        )
        assert executor.num_grids == 10
        assert executor.grid_type == "geometric"
