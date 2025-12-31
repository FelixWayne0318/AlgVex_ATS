"""
状态同步器测试

测试 StateSynchronizer 的仓位对账功能
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.execution.state_synchronizer import (
    StateSynchronizer,
    PositionManager,
    PositionRecord,
    SyncResult,
    SyncStatus,
    ProtectionMode,
)


class TestPositionRecord:
    """PositionRecord 数据类测试"""

    def test_position_creation(self):
        """测试仓位记录创建"""
        pos = PositionRecord(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        assert pos.symbol == "BTCUSDT"
        assert pos.side == "LONG"
        assert pos.amount == Decimal("0.1")
        assert pos.entry_price == Decimal("50000")
        assert pos.unrealized_pnl == Decimal("0")
        assert pos.leverage == 1

    def test_to_dict(self):
        """测试转换为字典"""
        pos = PositionRecord(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
            leverage=5,
        )

        d = pos.to_dict()

        assert d["symbol"] == "BTCUSDT"
        assert d["side"] == "LONG"
        assert d["amount"] == "0.1"
        assert d["entry_price"] == "50000"
        assert d["leverage"] == 5


class TestPositionManager:
    """PositionManager 测试"""

    @pytest.fixture
    def manager(self):
        """创建仓位管理器"""
        return PositionManager()

    def test_update_position(self, manager):
        """测试更新仓位"""
        manager.update_position(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        pos = manager.get_position("BTCUSDT")
        assert pos is not None
        assert pos.side == "LONG"
        assert pos.amount == Decimal("0.1")

    def test_close_position(self, manager):
        """测试关闭仓位"""
        manager.update_position(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        manager.close_position("BTCUSDT")

        pos = manager.get_position("BTCUSDT")
        assert pos is None

    def test_close_position_by_zero_amount(self, manager):
        """测试通过零数量关闭仓位"""
        manager.update_position(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        # 数量为0时自动关闭
        manager.update_position(
            symbol="BTCUSDT",
            side="LONG",
            amount=Decimal("0"),
            entry_price=Decimal("50000"),
        )

        pos = manager.get_position("BTCUSDT")
        assert pos is None

    def test_get_all_positions(self, manager):
        """测试获取所有仓位"""
        manager.update_position("BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000"))
        manager.update_position("ETHUSDT", "SHORT", Decimal("1"), Decimal("3000"))

        positions = manager.get_all_positions()

        assert len(positions) == 2
        assert "BTCUSDT" in positions
        assert "ETHUSDT" in positions

    def test_protection_mode(self, manager):
        """测试保护模式"""
        manager.update_position("BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000"))

        # 进入保护模式
        manager.enter_protection_mode()
        assert manager.mode == ProtectionMode.PROTECTION

        # 保护模式下的更新会被暂存
        manager.update_position("BTCUSDT", "LONG", Decimal("0.2"), Decimal("51000"))

        # 原仓位不变
        pos = manager.get_position("BTCUSDT")
        assert pos.amount == Decimal("0.1")

        # 退出保护模式，应用暂存的更新
        manager.exit_protection_mode()
        assert manager.mode == ProtectionMode.NORMAL

        pos = manager.get_position("BTCUSDT")
        assert pos.amount == Decimal("0.2")


class TestSyncResult:
    """SyncResult 测试"""

    def test_all_synced_true(self):
        """测试全部同步成功"""
        result = SyncResult(
            timestamp=0,
            symbols_checked=5,
            synced=5,
            mismatched=0,
            missing_local=0,
            missing_exchange=0,
            errors=[],
        )

        assert result.all_synced is True

    def test_all_synced_false_mismatch(self):
        """测试有不匹配"""
        result = SyncResult(
            timestamp=0,
            symbols_checked=5,
            synced=4,
            mismatched=1,
            missing_local=0,
            missing_exchange=0,
            errors=[],
        )

        assert result.all_synced is False

    def test_all_synced_false_errors(self):
        """测试有错误"""
        result = SyncResult(
            timestamp=0,
            symbols_checked=5,
            synced=5,
            mismatched=0,
            missing_local=0,
            missing_exchange=0,
            errors=["Some error"],
        )

        assert result.all_synced is False


class TestStateSynchronizer:
    """StateSynchronizer 测试"""

    @pytest.fixture
    def synchronizer(self):
        """创建同步器"""
        return StateSynchronizer(sync_interval=1.0)

    @pytest.mark.asyncio
    async def test_sync_empty(self, synchronizer):
        """测试空同步"""
        result = await synchronizer.sync()

        assert result.symbols_checked == 0
        assert result.all_synced is True

    @pytest.mark.asyncio
    async def test_sync_with_matching_positions(self, synchronizer):
        """测试匹配的仓位同步"""
        # 设置本地仓位
        synchronizer.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        # 设置交易所数据获取
        async def fetch_positions():
            return {
                "BTCUSDT": {
                    "side": "LONG",
                    "amount": "0.1",
                    "entry_price": "50000",
                }
            }

        synchronizer.set_exchange_fetcher(fetch_positions)

        result = await synchronizer.sync()

        assert result.symbols_checked == 1
        assert result.synced == 1
        assert result.mismatched == 0
        assert result.all_synced is True

    @pytest.mark.asyncio
    async def test_sync_missing_local(self, synchronizer):
        """测试本地缺失仓位"""
        # 只有交易所有仓位
        async def fetch_positions():
            return {
                "BTCUSDT": {
                    "side": "LONG",
                    "amount": "0.1",
                    "entry_price": "50000",
                }
            }

        synchronizer.set_exchange_fetcher(fetch_positions)

        result = await synchronizer.sync()

        assert result.missing_local == 1

        # 同步后本地应该有仓位
        pos = synchronizer.position_manager.get_position("BTCUSDT")
        assert pos is not None
        assert pos.amount == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_sync_missing_exchange(self, synchronizer):
        """测试交易所缺失仓位"""
        # 只有本地有仓位
        synchronizer.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        async def fetch_positions():
            return {}

        synchronizer.set_exchange_fetcher(fetch_positions)

        result = await synchronizer.sync()

        assert result.missing_exchange == 1

        # 同步后本地仓位应该被清除
        pos = synchronizer.position_manager.get_position("BTCUSDT")
        assert pos is None

    @pytest.mark.asyncio
    async def test_sync_amount_mismatch(self, synchronizer):
        """测试数量不匹配"""
        # 设置本地仓位
        synchronizer.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        # 交易所数量不同
        async def fetch_positions():
            return {
                "BTCUSDT": {
                    "side": "LONG",
                    "amount": "0.2",  # 不同
                    "entry_price": "50000",
                }
            }

        synchronizer.set_exchange_fetcher(fetch_positions)

        result = await synchronizer.sync()

        assert result.mismatched == 1

        # 以交易所为准
        pos = synchronizer.position_manager.get_position("BTCUSDT")
        assert pos.amount == Decimal("0.2")

    @pytest.mark.asyncio
    async def test_disconnect_reconnect(self, synchronizer):
        """测试断线重连"""
        synchronizer.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        async def fetch_positions():
            return {
                "BTCUSDT": {
                    "side": "LONG",
                    "amount": "0.1",
                    "entry_price": "50000",
                }
            }

        synchronizer.set_exchange_fetcher(fetch_positions)

        # 断线
        await synchronizer.on_disconnect()
        assert synchronizer.position_manager.mode == ProtectionMode.PROTECTION

        # 重连
        await synchronizer.on_reconnect()
        assert synchronizer.position_manager.mode == ProtectionMode.NORMAL

    @pytest.mark.asyncio
    async def test_mismatch_callback(self, synchronizer):
        """测试不匹配回调"""
        mismatches = []

        def on_mismatch(symbol, details):
            mismatches.append((symbol, details))

        synchronizer._on_mismatch_detected = on_mismatch

        # 设置本地仓位
        synchronizer.position_manager.update_position(
            "BTCUSDT", "LONG", Decimal("0.1"), Decimal("50000")
        )

        # 交易所数量不同
        async def fetch_positions():
            return {
                "BTCUSDT": {
                    "side": "LONG",
                    "amount": "0.2",
                    "entry_price": "50000",
                }
            }

        synchronizer.set_exchange_fetcher(fetch_positions)

        await synchronizer.sync()

        assert len(mismatches) == 1
        assert mismatches[0][0] == "BTCUSDT"
        assert mismatches[0][1]["type"] == "amount_mismatch"

    def test_get_statistics(self, synchronizer):
        """测试获取统计信息"""
        stats = synchronizer.get_statistics()

        assert "total_syncs" in stats
        assert "success_rate" in stats
        assert "mode" in stats
        assert stats["mode"] == "normal"

    def test_tolerance_check(self, synchronizer):
        """测试容差检查"""
        # 差异小于容差应该视为同步
        tolerance = StateSynchronizer.POSITION_TOLERANCE

        assert tolerance == Decimal("0.00001")
