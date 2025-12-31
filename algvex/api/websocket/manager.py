"""
WebSocket 连接管理器
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect


logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket 连接管理器

    功能:
    - 管理 WebSocket 连接
    - 支持订阅/取消订阅频道
    - 广播消息到订阅者
    """

    def __init__(self):
        # 活跃连接: {connection_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # 订阅关系: {channel: set(connection_ids)}
        self.subscriptions: Dict[str, Set[str]] = {}

        # 连接信息: {connection_id: {user_id, subscribed_channels, connected_at}}
        self.connection_info: Dict[str, Dict[str, Any]] = {}

        # 连接计数
        self._connection_counter = 0

    def _generate_connection_id(self) -> str:
        """生成连接ID"""
        self._connection_counter += 1
        return f"conn_{self._connection_counter}_{datetime.utcnow().strftime('%H%M%S')}"

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[int] = None,
    ) -> str:
        """
        接受新连接

        Args:
            websocket: WebSocket 连接
            user_id: 用户ID (可选)

        Returns:
            连接ID
        """
        await websocket.accept()

        connection_id = self._generate_connection_id()
        self.active_connections[connection_id] = websocket
        self.connection_info[connection_id] = {
            "user_id": user_id,
            "subscribed_channels": set(),
            "connected_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"WebSocket 连接已建立: {connection_id}, user_id={user_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        """
        断开连接

        Args:
            connection_id: 连接ID
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # 清理订阅
        if connection_id in self.connection_info:
            channels = self.connection_info[connection_id].get("subscribed_channels", set())
            for channel in channels:
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(connection_id)

            del self.connection_info[connection_id]

        logger.info(f"WebSocket 连接已断开: {connection_id}")

    def subscribe(self, connection_id: str, channel: str) -> bool:
        """
        订阅频道

        Args:
            connection_id: 连接ID
            channel: 频道名

        Returns:
            是否成功
        """
        if connection_id not in self.active_connections:
            return False

        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()

        self.subscriptions[channel].add(connection_id)

        if connection_id in self.connection_info:
            self.connection_info[connection_id]["subscribed_channels"].add(channel)

        logger.debug(f"订阅: {connection_id} -> {channel}")
        return True

    def unsubscribe(self, connection_id: str, channel: str) -> bool:
        """
        取消订阅

        Args:
            connection_id: 连接ID
            channel: 频道名

        Returns:
            是否成功
        """
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(connection_id)

        if connection_id in self.connection_info:
            self.connection_info[connection_id]["subscribed_channels"].discard(channel)

        return True

    async def send_personal_message(
        self,
        message: Dict[str, Any],
        connection_id: str,
    ):
        """
        发送私人消息

        Args:
            message: 消息内容
            connection_id: 连接ID
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"发送消息失败 {connection_id}: {e}")
                self.disconnect(connection_id)

    async def broadcast(self, message: Dict[str, Any], channel: str):
        """
        广播消息到频道

        Args:
            message: 消息内容
            channel: 频道名
        """
        if channel not in self.subscriptions:
            return

        disconnected = []
        for connection_id in self.subscriptions[channel]:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"广播失败 {connection_id}: {e}")
                    disconnected.append(connection_id)

        # 清理断开的连接
        for conn_id in disconnected:
            self.disconnect(conn_id)

    async def broadcast_all(self, message: Dict[str, Any]):
        """
        广播消息到所有连接

        Args:
            message: 消息内容
        """
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"广播失败 {connection_id}: {e}")
                disconnected.append(connection_id)

        for conn_id in disconnected:
            self.disconnect(conn_id)

    def get_connection_count(self) -> int:
        """获取连接数"""
        return len(self.active_connections)

    def get_channel_subscribers(self, channel: str) -> int:
        """获取频道订阅者数"""
        return len(self.subscriptions.get(channel, set()))

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_connections": len(self.active_connections),
            "channels": {
                channel: len(subscribers)
                for channel, subscribers in self.subscriptions.items()
            },
        }


# 全局连接管理器实例
manager = ConnectionManager()


def get_manager() -> ConnectionManager:
    """获取连接管理器"""
    return manager
