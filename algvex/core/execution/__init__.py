"""
AlgVex 执行层
基于 Hummingbot 的企业级执行引擎

组件:
- HummingbotBridge: 信号 → 订单转换桥接层
- OrderTracker: 订单生命周期追踪 (InFlightOrder)
- StateSynchronizer: 仓位状态同步和对账
- EventHandler: 订单事件处理和 trace 写入
- AlgVexController: Strategy V2 集成控制器
- RiskManager: 风控管理
- PositionManager: 仓位分配和再平衡
"""

from .hummingbot_bridge import HummingbotBridge
from .risk_manager import RiskManager
from .position_manager import PositionManager
from .order_tracker import AlgVexOrderTracker, OrderState, TrackedOrder
from .state_synchronizer import (
    StateSynchronizer,
    PositionManager as StatePositionManager,
    PositionRecord,
    SyncResult,
    SyncStatus,
    ProtectionMode,
)
from .event_handlers import (
    AlgVexEventHandler,
    EventType,
    OrderEvent,
    FundingPaymentEvent,
    TraceWriter,
)
from .controllers import AlgVexController, AlgVexControllerConfig

__all__ = [
    # Core bridge
    "HummingbotBridge",
    # Order tracking
    "AlgVexOrderTracker",
    "OrderState",
    "TrackedOrder",
    # State synchronization
    "StateSynchronizer",
    "StatePositionManager",
    "PositionRecord",
    "SyncResult",
    "SyncStatus",
    "ProtectionMode",
    # Event handling
    "AlgVexEventHandler",
    "EventType",
    "OrderEvent",
    "FundingPaymentEvent",
    "TraceWriter",
    # Strategy V2 controller
    "AlgVexController",
    "AlgVexControllerConfig",
    # Position management
    "RiskManager",
    "PositionManager",
]
