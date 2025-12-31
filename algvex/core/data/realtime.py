"""
实时数据流

通过 WebSocket 接收币安永续合约实时数据
支持:
1. K线推送
2. 资金费率推送
3. 订单簿推送
4. 成交推送
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import websockets
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class StreamType(Enum):
    KLINE = "kline"
    FUNDING_RATE = "markPrice"
    DEPTH = "depth"
    AGG_TRADE = "aggTrade"
    LIQUIDATION = "forceOrder"


@dataclass
class KlineData:
    """K线数据"""
    symbol: str
    interval: str
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    is_closed: bool


@dataclass
class FundingData:
    """资金费率数据"""
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: int


@dataclass
class DepthData:
    """订单簿数据"""
    symbol: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    timestamp: int


class RealtimeStream:
    """
    币安永续合约实时数据流

    使用方法:
        stream = RealtimeStream(symbols=["btcusdt", "ethusdt"])

        # 注册回调
        stream.on_kline(my_kline_handler)
        stream.on_funding(my_funding_handler)

        # 启动
        await stream.start()
    """

    BASE_URL = "wss://fstream.binance.com"

    def __init__(
        self,
        symbols: List[str],
        streams: List[StreamType] = None,
    ):
        """
        初始化实时数据流

        Args:
            symbols: 交易对列表 (小写)
            streams: 数据流类型列表
        """
        self.symbols = [s.lower() for s in symbols]
        self.streams = streams or [StreamType.KLINE, StreamType.FUNDING_RATE]

        self._ws = None
        self._running = False
        self._callbacks: Dict[StreamType, List[Callable]] = {st: [] for st in StreamType}
        self._reconnect_delay = 1

        # 最新数据缓存
        self._latest_klines: Dict[str, KlineData] = {}
        self._latest_funding: Dict[str, FundingData] = {}
        self._latest_depth: Dict[str, DepthData] = {}

    def _build_url(self) -> str:
        """构建 WebSocket URL"""
        stream_names = []

        for symbol in self.symbols:
            for stream_type in self.streams:
                if stream_type == StreamType.KLINE:
                    stream_names.append(f"{symbol}@kline_1m")  # 1分钟K线
                elif stream_type == StreamType.FUNDING_RATE:
                    stream_names.append(f"{symbol}@markPrice@1s")
                elif stream_type == StreamType.DEPTH:
                    stream_names.append(f"{symbol}@depth20@100ms")
                elif stream_type == StreamType.AGG_TRADE:
                    stream_names.append(f"{symbol}@aggTrade")
                elif stream_type == StreamType.LIQUIDATION:
                    stream_names.append(f"{symbol}@forceOrder")

        streams_param = "/".join(stream_names)
        return f"{self.BASE_URL}/stream?streams={streams_param}"

    def on_kline(self, callback: Callable[[KlineData], None]) -> None:
        """注册K线回调"""
        self._callbacks[StreamType.KLINE].append(callback)

    def on_funding(self, callback: Callable[[FundingData], None]) -> None:
        """注册资金费率回调"""
        self._callbacks[StreamType.FUNDING_RATE].append(callback)

    def on_depth(self, callback: Callable[[DepthData], None]) -> None:
        """注册订单簿回调"""
        self._callbacks[StreamType.DEPTH].append(callback)

    async def start(self) -> None:
        """启动数据流"""
        self._running = True
        url = self._build_url()

        logger.info(f"Connecting to {url[:100]}...")

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1  # 重置重连延迟
                    logger.info("WebSocket connected")

                    async for message in ws:
                        await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    async def stop(self) -> None:
        """停止数据流"""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket stopped")

    async def _handle_message(self, message: str) -> None:
        """处理接收到的消息"""
        try:
            data = json.loads(message)

            # 组合流格式: {"stream": "...", "data": {...}}
            if "stream" in data:
                stream_name = data["stream"]
                payload = data["data"]
            else:
                payload = data
                stream_name = ""

            # 解析数据类型
            if "@kline" in stream_name or "e" in payload and payload.get("e") == "kline":
                await self._handle_kline(payload)
            elif "@markPrice" in stream_name or payload.get("e") == "markPriceUpdate":
                await self._handle_funding(payload)
            elif "@depth" in stream_name:
                await self._handle_depth(payload, stream_name)
            elif "@aggTrade" in stream_name or payload.get("e") == "aggTrade":
                await self._handle_trade(payload)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_kline(self, data: Dict) -> None:
        """处理K线数据"""
        k = data.get("k", data)

        kline = KlineData(
            symbol=data.get("s", k.get("s", "")),
            interval=k.get("i", "1m"),
            open_time=k.get("t", 0),
            close_time=k.get("T", 0),
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
            quote_volume=float(k.get("q", 0)),
            trades=int(k.get("n", 0)),
            is_closed=k.get("x", False),
        )

        self._latest_klines[kline.symbol] = kline

        # 触发回调
        for callback in self._callbacks[StreamType.KLINE]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(kline)
                else:
                    callback(kline)
            except Exception as e:
                logger.error(f"Kline callback error: {e}")

    async def _handle_funding(self, data: Dict) -> None:
        """处理资金费率数据"""
        funding = FundingData(
            symbol=data.get("s", ""),
            mark_price=float(data.get("p", 0)),
            index_price=float(data.get("i", 0)),
            funding_rate=float(data.get("r", 0)),
            next_funding_time=int(data.get("T", 0)),
        )

        self._latest_funding[funding.symbol] = funding

        # 触发回调
        for callback in self._callbacks[StreamType.FUNDING_RATE]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(funding)
                else:
                    callback(funding)
            except Exception as e:
                logger.error(f"Funding callback error: {e}")

    async def _handle_depth(self, data: Dict, stream_name: str) -> None:
        """处理订单簿数据"""
        # 从stream名称提取symbol
        symbol = stream_name.split("@")[0].upper()

        depth = DepthData(
            symbol=symbol,
            bids=[[float(p), float(q)] for p, q in data.get("b", [])],
            asks=[[float(p), float(q)] for p, q in data.get("a", [])],
            timestamp=data.get("T", int(datetime.now().timestamp() * 1000)),
        )

        self._latest_depth[depth.symbol] = depth

        # 触发回调
        for callback in self._callbacks[StreamType.DEPTH]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(depth)
                else:
                    callback(depth)
            except Exception as e:
                logger.error(f"Depth callback error: {e}")

    async def _handle_trade(self, data: Dict) -> None:
        """处理成交数据"""
        # 可扩展
        pass

    # ==================== 便捷方法 ====================

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        symbol = symbol.upper()
        if symbol in self._latest_klines:
            return self._latest_klines[symbol].close
        if symbol in self._latest_funding:
            return self._latest_funding[symbol].mark_price
        return None

    def get_latest_funding_rate(self, symbol: str) -> Optional[float]:
        """获取最新资金费率"""
        symbol = symbol.upper()
        if symbol in self._latest_funding:
            return self._latest_funding[symbol].funding_rate
        return None

    def get_best_bid_ask(self, symbol: str) -> Optional[tuple]:
        """获取最佳买卖价"""
        symbol = symbol.upper()
        if symbol in self._latest_depth:
            depth = self._latest_depth[symbol]
            if depth.bids and depth.asks:
                return (depth.bids[0][0], depth.asks[0][0])
        return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """获取买卖价差"""
        ba = self.get_best_bid_ask(symbol)
        if ba:
            return (ba[1] - ba[0]) / ba[0]
        return None


# ==================== 使用示例 ====================
async def main():
    stream = RealtimeStream(
        symbols=["btcusdt", "ethusdt"],
        streams=[StreamType.KLINE, StreamType.FUNDING_RATE],
    )

    def on_kline(kline: KlineData):
        if kline.is_closed:
            print(f"[Kline] {kline.symbol} close={kline.close} vol={kline.volume}")

    def on_funding(funding: FundingData):
        print(f"[Funding] {funding.symbol} rate={funding.funding_rate:.6f}")

    stream.on_kline(on_kline)
    stream.on_funding(on_funding)

    try:
        await stream.start()
    except KeyboardInterrupt:
        await stream.stop()


if __name__ == "__main__":
    asyncio.run(main())
