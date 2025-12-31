"""
状态同步器

负责 AlgVex 与 Hummingbot 之间的仓位状态同步和对账
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """同步状态"""
    SYNCED = "synced"
    MISMATCHED = "mismatched"
    MISSING_LOCAL = "missing_local"
    MISSING_EXCHANGE = "missing_exchange"


class ProtectionMode(Enum):
    """保护模式"""
    NORMAL = "normal"
    PROTECTION = "protection"  # 断线保护
    RECOVERY = "recovery"      # 恢复中


@dataclass
class PositionRecord:
    """仓位记录"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    amount: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    liquidation_price: Optional[Decimal] = None
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "amount": str(self.amount),
            "entry_price": str(self.entry_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "leverage": self.leverage,
            "liquidation_price": str(self.liquidation_price) if self.liquidation_price else None,
            "last_update": self.last_update,
        }


@dataclass
class SyncResult:
    """同步结果"""
    timestamp: float
    symbols_checked: int
    synced: int
    mismatched: int
    missing_local: int
    missing_exchange: int
    errors: List[str]

    @property
    def all_synced(self) -> bool:
        """是否全部同步"""
        return (
            self.mismatched == 0 and
            self.missing_local == 0 and
            len(self.errors) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "symbols_checked": self.symbols_checked,
            "synced": self.synced,
            "mismatched": self.mismatched,
            "missing_local": self.missing_local,
            "missing_exchange": self.missing_exchange,
            "errors": self.errors,
            "all_synced": self.all_synced,
        }


class PositionManager:
    """
    仓位管理器

    管理本地仓位状态，支持保护模式
    """

    def __init__(self):
        self._positions: Dict[str, PositionRecord] = {}
        self._mode: ProtectionMode = ProtectionMode.NORMAL
        self._pending_updates: List[Dict] = []

    def get_position(self, symbol: str) -> Optional[PositionRecord]:
        """获取仓位"""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionRecord]:
        """获取所有仓位"""
        return dict(self._positions)

    def update_position(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        entry_price: Decimal,
        **kwargs
    ):
        """
        更新仓位

        在保护模式下，更新会被暂存
        """
        update = {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            **kwargs
        }

        if self._mode == ProtectionMode.PROTECTION:
            # 保护模式下暂存更新
            self._pending_updates.append(update)
            logger.warning(f"Position update queued (protection mode): {symbol}")
            return

        self._apply_position_update(update)

    def _apply_position_update(self, update: Dict):
        """应用仓位更新"""
        symbol = update["symbol"]

        if update["amount"] == Decimal("0"):
            # 数量为0，删除仓位
            if symbol in self._positions:
                del self._positions[symbol]
                logger.info(f"Position closed: {symbol}")
        else:
            # 更新或创建仓位
            self._positions[symbol] = PositionRecord(
                symbol=symbol,
                side=update["side"],
                amount=update["amount"],
                entry_price=update["entry_price"],
                unrealized_pnl=update.get("unrealized_pnl", Decimal("0")),
                leverage=update.get("leverage", 1),
                liquidation_price=update.get("liquidation_price"),
                last_update=time.time(),
            )
            logger.debug(f"Position updated: {symbol}")

    def close_position(self, symbol: str):
        """关闭仓位"""
        if symbol in self._positions:
            del self._positions[symbol]
            logger.info(f"Position closed: {symbol}")

    def enter_protection_mode(self):
        """进入保护模式"""
        self._mode = ProtectionMode.PROTECTION
        self._pending_updates.clear()
        logger.warning("Entered protection mode - position updates queued")

    def exit_protection_mode(self):
        """退出保护模式"""
        self._mode = ProtectionMode.RECOVERY

        # 应用暂存的更新
        for update in self._pending_updates:
            self._apply_position_update(update)

        self._pending_updates.clear()
        self._mode = ProtectionMode.NORMAL
        logger.info("Exited protection mode - all pending updates applied")

    @property
    def mode(self) -> ProtectionMode:
        """当前模式"""
        return self._mode


class StateSynchronizer:
    """
    状态同步器

    职责:
    1. 定期同步仓位状态
    2. 检测并处理状态不一致
    3. 处理断线重连

    设计原则:
    - 以交易所为事实源（Source of Truth）
    - 不一致时以交易所数据为准
    - 支持保护模式防止断线期间误操作
    """

    # 仓位差异阈值（超过此值视为不一致）
    POSITION_TOLERANCE = Decimal("0.00001")

    def __init__(
        self,
        position_manager: Optional[PositionManager] = None,
        sync_interval: float = 60.0,  # 60秒同步一次
        max_consecutive_errors: int = 5,
        on_sync_complete: Optional[Callable[[SyncResult], None]] = None,
        on_mismatch_detected: Optional[Callable[[str, Dict], None]] = None,
    ):
        """
        Args:
            position_manager: 仓位管理器
            sync_interval: 同步间隔（秒）
            max_consecutive_errors: 最大连续错误次数
            on_sync_complete: 同步完成回调
            on_mismatch_detected: 发现不一致回调
        """
        self.position_manager = position_manager or PositionManager()
        self.sync_interval = sync_interval
        self.max_consecutive_errors = max_consecutive_errors
        self._on_sync_complete = on_sync_complete
        self._on_mismatch_detected = on_mismatch_detected

        self._running = False
        self._consecutive_errors = 0
        self._sync_history: List[SyncResult] = []
        self._max_history = 100

        # 交易所数据获取函数（由外部注入）
        self._exchange_position_fetcher: Optional[Callable] = None

    def set_exchange_fetcher(self, fetcher: Callable):
        """设置交易所数据获取函数"""
        self._exchange_position_fetcher = fetcher

    async def start(self):
        """启动同步循环"""
        self._running = True
        logger.info(f"StateSynchronizer started, interval={self.sync_interval}s")

        while self._running:
            try:
                result = await self.sync()
                self._consecutive_errors = 0

                if self._on_sync_complete:
                    try:
                        self._on_sync_complete(result)
                    except Exception as e:
                        logger.error(f"Error in sync complete callback: {e}")

            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"Sync failed ({self._consecutive_errors}/{self.max_consecutive_errors}): {e}")

                if self._consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Max consecutive errors reached, entering protection mode")
                    self.position_manager.enter_protection_mode()

            await asyncio.sleep(self.sync_interval)

    async def stop(self):
        """停止同步循环"""
        self._running = False
        logger.info("StateSynchronizer stopped")

    async def sync(self) -> SyncResult:
        """
        执行一次同步

        Returns:
            SyncResult: 同步结果
        """
        timestamp = time.time()
        errors = []

        # 1. 获取交易所仓位
        try:
            exchange_positions = await self._fetch_exchange_positions()
        except Exception as e:
            logger.error(f"Failed to fetch exchange positions: {e}")
            raise

        # 2. 获取本地仓位
        local_positions = self.position_manager.get_all_positions()

        # 3. 合并所有symbol
        all_symbols = set(exchange_positions.keys()) | set(local_positions.keys())

        # 4. 对比并处理差异
        synced = 0
        mismatched = 0
        missing_local = 0
        missing_exchange = 0

        for symbol in all_symbols:
            try:
                status = await self._sync_symbol(
                    symbol,
                    exchange_positions.get(symbol),
                    local_positions.get(symbol)
                )

                if status == SyncStatus.SYNCED:
                    synced += 1
                elif status == SyncStatus.MISMATCHED:
                    mismatched += 1
                elif status == SyncStatus.MISSING_LOCAL:
                    missing_local += 1
                elif status == SyncStatus.MISSING_EXCHANGE:
                    missing_exchange += 1

            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                logger.error(f"Error syncing {symbol}: {e}")

        # 5. 创建结果
        result = SyncResult(
            timestamp=timestamp,
            symbols_checked=len(all_symbols),
            synced=synced,
            mismatched=mismatched,
            missing_local=missing_local,
            missing_exchange=missing_exchange,
            errors=errors,
        )

        # 6. 记录历史
        self._sync_history.append(result)
        if len(self._sync_history) > self._max_history:
            self._sync_history = self._sync_history[-self._max_history:]

        logger.info(
            f"Sync completed: {result.symbols_checked} checked, "
            f"{synced} synced, {mismatched} mismatched, "
            f"{missing_local} missing local, {missing_exchange} missing exchange"
        )

        return result

    async def _fetch_exchange_positions(self) -> Dict[str, Dict]:
        """获取交易所仓位"""
        if self._exchange_position_fetcher:
            return await self._exchange_position_fetcher()

        # 默认返回空（需要外部注入实际的fetcher）
        logger.warning("No exchange position fetcher configured")
        return {}

    async def _sync_symbol(
        self,
        symbol: str,
        exchange_pos: Optional[Dict],
        local_pos: Optional[PositionRecord],
    ) -> SyncStatus:
        """
        同步单个品种

        以交易所为准，更新本地状态
        """
        if exchange_pos and not local_pos:
            # 交易所有仓位，本地没有 -> 更新本地
            logger.warning(f"Missing local position for {symbol}, syncing from exchange")

            self.position_manager.update_position(
                symbol=symbol,
                side=exchange_pos.get("side", "LONG"),
                amount=Decimal(str(exchange_pos.get("amount", 0))),
                entry_price=Decimal(str(exchange_pos.get("entry_price", 0))),
                unrealized_pnl=Decimal(str(exchange_pos.get("unrealized_pnl", 0))),
                leverage=exchange_pos.get("leverage", 1),
                liquidation_price=Decimal(str(exchange_pos["liquidation_price"]))
                    if exchange_pos.get("liquidation_price") else None,
            )

            if self._on_mismatch_detected:
                self._on_mismatch_detected(symbol, {
                    "type": "missing_local",
                    "exchange": exchange_pos,
                })

            return SyncStatus.MISSING_LOCAL

        elif local_pos and not exchange_pos:
            # 本地有仓位，交易所没有 -> 可能已平仓
            logger.warning(f"Position {symbol} closed on exchange, updating local")
            self.position_manager.close_position(symbol)

            if self._on_mismatch_detected:
                self._on_mismatch_detected(symbol, {
                    "type": "missing_exchange",
                    "local": local_pos.to_dict(),
                })

            return SyncStatus.MISSING_EXCHANGE

        elif exchange_pos and local_pos:
            # 两边都有，检查数量是否一致
            exchange_amt = Decimal(str(exchange_pos.get("amount", 0)))
            local_amt = local_pos.amount

            if abs(exchange_amt - local_amt) > self.POSITION_TOLERANCE:
                logger.error(
                    f"Position mismatch for {symbol}: "
                    f"exchange={exchange_amt}, local={local_amt}"
                )

                # 以交易所为准
                self.position_manager.update_position(
                    symbol=symbol,
                    side=exchange_pos.get("side", local_pos.side),
                    amount=exchange_amt,
                    entry_price=Decimal(str(exchange_pos.get("entry_price", local_pos.entry_price))),
                    unrealized_pnl=Decimal(str(exchange_pos.get("unrealized_pnl", 0))),
                    leverage=exchange_pos.get("leverage", local_pos.leverage),
                )

                if self._on_mismatch_detected:
                    self._on_mismatch_detected(symbol, {
                        "type": "amount_mismatch",
                        "exchange_amount": str(exchange_amt),
                        "local_amount": str(local_amt),
                    })

                return SyncStatus.MISMATCHED

            return SyncStatus.SYNCED

        # 两边都没有
        return SyncStatus.SYNCED

    async def on_disconnect(self):
        """处理断线"""
        logger.warning("Connector disconnected, entering protection mode")
        self.position_manager.enter_protection_mode()
        self._consecutive_errors = 0

    async def on_reconnect(self):
        """处理重连"""
        logger.info("Connector reconnected, performing full sync")

        try:
            await self.sync()
            self.position_manager.exit_protection_mode()
            self._consecutive_errors = 0
        except Exception as e:
            logger.error(f"Failed to sync on reconnect: {e}")

    def get_sync_history(self, limit: int = 10) -> List[Dict]:
        """获取同步历史"""
        return [r.to_dict() for r in self._sync_history[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._sync_history:
            return {
                "total_syncs": 0,
                "success_rate": 0,
                "avg_symbols_per_sync": 0,
                "total_mismatches": 0,
                "consecutive_errors": self._consecutive_errors,
                "mode": self.position_manager.mode.value,
            }

        total = len(self._sync_history)
        successful = sum(1 for r in self._sync_history if r.all_synced)
        total_symbols = sum(r.symbols_checked for r in self._sync_history)
        total_mismatches = sum(r.mismatched for r in self._sync_history)

        return {
            "total_syncs": total,
            "success_rate": successful / total if total > 0 else 0,
            "avg_symbols_per_sync": total_symbols / total if total > 0 else 0,
            "total_mismatches": total_mismatches,
            "consecutive_errors": self._consecutive_errors,
            "mode": self.position_manager.mode.value,
            "last_sync": self._sync_history[-1].to_dict() if self._sync_history else None,
        }
