"""
P0-2 验收测试: 价格语义统一

验收标准:
- PriceSemantics 正确映射场景到价格类型
- 不同场景使用正确的价格
- 价格语义一致性验证
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.price_semantics import (
    PriceSemantics,
    PriceData,
    PriceType,
    PriceScenario,
    get_price_semantics,
    reset_price_semantics,
)


class TestPriceSemantics:
    """测试价格语义统一器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_price_semantics()

    def test_price_type_mapping(self):
        """测试价格类型映射"""
        semantics = PriceSemantics()

        # PnL 计算使用 mark_price
        assert semantics.get_price_type(PriceScenario.PNL_CALCULATION) == PriceType.MARK_PRICE

        # 强平检查使用 mark_price
        assert semantics.get_price_type(PriceScenario.LIQUIDATION_CHECK) == PriceType.MARK_PRICE

        # 入场/出场信号使用 close_price
        assert semantics.get_price_type(PriceScenario.ENTRY_EXIT_SIGNAL) == PriceType.CLOSE_PRICE

        # 订单执行使用 last_price
        assert semantics.get_price_type(PriceScenario.ORDER_EXECUTION) == PriceType.LAST_PRICE

        # 回测成交使用 close_price
        assert semantics.get_price_type(PriceScenario.BACKTEST_FILL) == PriceType.CLOSE_PRICE

        # 资金费率结算使用 mark_price
        assert semantics.get_price_type(PriceScenario.FUNDING_SETTLEMENT) == PriceType.MARK_PRICE

    def test_get_price(self):
        """测试获取价格"""
        semantics = PriceSemantics()
        price_data = PriceData(
            mark_price=50000.0,
            index_price=50010.0,
            last_price=49990.0,
            close_price=50005.0,
        )

        # PnL 计算应返回 mark_price
        pnl_price = semantics.get_price(PriceScenario.PNL_CALCULATION, price_data)
        assert pnl_price == 50000.0

        # 订单执行应返回 last_price
        exec_price = semantics.get_price(PriceScenario.ORDER_EXECUTION, price_data)
        assert exec_price == 49990.0

        # 入场信号应返回 close_price
        signal_price = semantics.get_price(PriceScenario.ENTRY_EXIT_SIGNAL, price_data)
        assert signal_price == 50005.0

    def test_get_price_from_dict(self):
        """测试从字典获取价格"""
        semantics = PriceSemantics()
        data = {
            "mark_price": 50000.0,
            "last_price": 49990.0,
            "close": 50005.0,  # 使用 close 而不是 close_price
        }

        # 应该正确处理 close vs close_price
        price = semantics.get_price_from_dict(PriceScenario.ENTRY_EXIT_SIGNAL, data)
        assert price == 50005.0

    def test_price_not_available(self):
        """测试价格不可用时的处理"""
        semantics = PriceSemantics()
        price_data = PriceData(last_price=49990.0)  # 只有 last_price

        # 没有 mark_price 时应该抛出异常
        with pytest.raises(ValueError, match="Price not available"):
            semantics.get_price(PriceScenario.PNL_CALCULATION, price_data)

    def test_price_fallback(self):
        """测试价格回退"""
        semantics = PriceSemantics()
        price_data = PriceData(last_price=49990.0)  # 只有 last_price

        # 允许回退时应返回 last_price
        price = semantics.get_price(PriceScenario.PNL_CALCULATION, price_data, fallback=True)
        assert price == 49990.0

    def test_validate_price_usage(self):
        """测试价格使用验证"""
        semantics = PriceSemantics()

        # 正确使用
        assert semantics.validate_price_usage(
            PriceScenario.PNL_CALCULATION, PriceType.MARK_PRICE
        ) is True

        # 错误使用
        assert semantics.validate_price_usage(
            PriceScenario.PNL_CALCULATION, PriceType.LAST_PRICE
        ) is False

    def test_calculate_pnl(self):
        """测试 PnL 计算"""
        semantics = PriceSemantics()
        price_data = PriceData(mark_price=51000.0)

        # 多头盈利
        pnl = semantics.calculate_pnl(
            entry_price=50000.0,
            current_price_data=price_data,
            quantity=1.0,
            side="long",
        )
        assert pnl == 1000.0

        # 空头盈利
        price_data.mark_price = 49000.0
        pnl = semantics.calculate_pnl(
            entry_price=50000.0,
            current_price_data=price_data,
            quantity=1.0,
            side="short",
        )
        assert pnl == 1000.0

    def test_custom_mapping(self):
        """测试自定义映射"""
        custom_mapping = {
            PriceScenario.BACKTEST_FILL: PriceType.LAST_PRICE,  # 覆盖默认
        }
        semantics = PriceSemantics(custom_mapping=custom_mapping)

        assert semantics.get_price_type(PriceScenario.BACKTEST_FILL) == PriceType.LAST_PRICE

    def test_global_singleton(self):
        """测试全局单例"""
        reset_price_semantics()

        s1 = get_price_semantics()
        s2 = get_price_semantics()

        assert s1 is s2

    def test_mapping_report(self):
        """测试映射报告"""
        semantics = PriceSemantics()
        report = semantics.get_mapping_report()

        assert "pnl_calculation" in report
        assert report["pnl_calculation"] == "mark_price"
        assert "order_execution" in report
        assert report["order_execution"] == "last_price"


class TestPriceData:
    """测试价格数据容器"""

    def test_get_by_type(self):
        """测试按类型获取"""
        data = PriceData(
            mark_price=50000.0,
            index_price=50010.0,
            last_price=49990.0,
            close_price=50005.0,
        )

        assert data.get(PriceType.MARK_PRICE) == 50000.0
        assert data.get(PriceType.INDEX_PRICE) == 50010.0
        assert data.get(PriceType.LAST_PRICE) == 49990.0
        assert data.get(PriceType.CLOSE_PRICE) == 50005.0

    def test_to_dict(self):
        """测试转换为字典"""
        data = PriceData(mark_price=50000.0, timestamp=1234567890)
        d = data.to_dict()

        assert d["mark_price"] == 50000.0
        assert d["timestamp"] == 1234567890
        assert d["last_price"] is None
