"""
WebSocket 消息处理器
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

from .manager import ConnectionManager, get_manager


logger = logging.getLogger(__name__)


class BaseHandler:
    """基础处理器"""

    def __init__(self, manager: Optional[ConnectionManager] = None):
        self.manager = manager or get_manager()

    async def handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict[str, Any],
    ):
        """处理消息"""
        raise NotImplementedError


class SignalHandler(BaseHandler):
    """
    信号处理器

    频道:
    - signals: 所有信号
    - signals:{symbol}: 特定标的信号
    """

    CHANNEL_PREFIX = "signals"

    async def handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict[str, Any],
    ):
        """处理信号订阅消息"""
        action = message.get("action")
        symbol = message.get("symbol")

        if action == "subscribe":
            if symbol:
                channel = f"{self.CHANNEL_PREFIX}:{symbol}"
            else:
                channel = self.CHANNEL_PREFIX

            self.manager.subscribe(connection_id, channel)
            await self.manager.send_personal_message(
                {"type": "subscribed", "channel": channel},
                connection_id,
            )

        elif action == "unsubscribe":
            if symbol:
                channel = f"{self.CHANNEL_PREFIX}:{symbol}"
            else:
                channel = self.CHANNEL_PREFIX

            self.manager.unsubscribe(connection_id, channel)
            await self.manager.send_personal_message(
                {"type": "unsubscribed", "channel": channel},
                connection_id,
            )

    async def publish_signal(self, signal_data: Dict[str, Any]):
        """
        发布信号

        Args:
            signal_data: 信号数据
        """
        message = {
            "type": "signal",
            "data": signal_data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # 广播到全局信号频道
        await self.manager.broadcast(message, self.CHANNEL_PREFIX)

        # 广播到特定标的频道
        symbol = signal_data.get("symbol")
        if symbol:
            await self.manager.broadcast(message, f"{self.CHANNEL_PREFIX}:{symbol}")


class MarketHandler(BaseHandler):
    """
    行情处理器

    频道:
    - market:{symbol}: 特定标的行情
    - market:all: 所有行情
    """

    CHANNEL_PREFIX = "market"

    async def handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict[str, Any],
    ):
        """处理行情订阅消息"""
        action = message.get("action")
        symbols = message.get("symbols", [])

        if action == "subscribe":
            for symbol in symbols:
                channel = f"{self.CHANNEL_PREFIX}:{symbol}"
                self.manager.subscribe(connection_id, channel)

            await self.manager.send_personal_message(
                {"type": "subscribed", "channels": [f"{self.CHANNEL_PREFIX}:{s}" for s in symbols]},
                connection_id,
            )

        elif action == "unsubscribe":
            for symbol in symbols:
                channel = f"{self.CHANNEL_PREFIX}:{symbol}"
                self.manager.unsubscribe(connection_id, channel)

            await self.manager.send_personal_message(
                {"type": "unsubscribed", "channels": [f"{self.CHANNEL_PREFIX}:{s}" for s in symbols]},
                connection_id,
            )

    async def publish_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """
        发布行情数据

        Args:
            symbol: 标的
            market_data: 行情数据
        """
        message = {
            "type": "market",
            "symbol": symbol,
            "data": market_data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        await self.manager.broadcast(message, f"{self.CHANNEL_PREFIX}:{symbol}")


class AlignmentHandler(BaseHandler):
    """
    对齐状态处理器

    频道:
    - alignment: 对齐检查结果
    """

    CHANNEL = "alignment"

    async def handle_message(
        self,
        websocket: WebSocket,
        connection_id: str,
        message: Dict[str, Any],
    ):
        """处理对齐订阅消息"""
        action = message.get("action")

        if action == "subscribe":
            self.manager.subscribe(connection_id, self.CHANNEL)
            await self.manager.send_personal_message(
                {"type": "subscribed", "channel": self.CHANNEL},
                connection_id,
            )

        elif action == "unsubscribe":
            self.manager.unsubscribe(connection_id, self.CHANNEL)
            await self.manager.send_personal_message(
                {"type": "unsubscribed", "channel": self.CHANNEL},
                connection_id,
            )

    async def publish_alignment_result(self, result: Dict[str, Any]):
        """
        发布对齐检查结果

        Args:
            result: 对齐检查结果
        """
        message = {
            "type": "alignment",
            "data": result,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        await self.manager.broadcast(message, self.CHANNEL)


async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[int] = None,
):
    """
    WebSocket 端点

    消息格式:
    {
        "type": "signal" | "market" | "alignment",
        "action": "subscribe" | "unsubscribe",
        "symbol": "BTCUSDT",  // 可选
        "symbols": ["BTCUSDT", "ETHUSDT"],  // 可选
    }
    """
    manager = get_manager()
    connection_id = await manager.connect(websocket, user_id)

    # 初始化处理器
    handlers = {
        "signal": SignalHandler(manager),
        "market": MarketHandler(manager),
        "alignment": AlignmentHandler(manager),
    }

    try:
        # 发送连接确认
        await manager.send_personal_message(
            {
                "type": "connected",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            connection_id,
        )

        while True:
            # 接收消息
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type in handlers:
                    await handlers[msg_type].handle_message(
                        websocket,
                        connection_id,
                        message,
                    )
                else:
                    await manager.send_personal_message(
                        {"type": "error", "message": f"Unknown message type: {msg_type}"},
                        connection_id,
                    )

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON"},
                    connection_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        manager.disconnect(connection_id)
