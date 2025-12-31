"""
AlgVex 交易所连接器

支持多交易所连接，基于 Hummingbot connector 架构:
- Binance Perpetual
- Bybit Perpetual
- OKX Perpetual
- Gate.io Perpetual
- Deribit

版本: 2.0.0
"""

import asyncio
import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """交易所类型"""
    BINANCE_PERPETUAL = "binance_perpetual"
    BYBIT_PERPETUAL = "bybit_perpetual"
    OKX_PERPETUAL = "okx_perpetual"
    GATE_PERPETUAL = "gate_perpetual"
    DERIBIT = "deribit"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # 单向持仓模式


@dataclass
class ExchangeConfig:
    """交易所配置"""
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    passphrase: str = ""  # OKX 需要
    testnet: bool = True
    rate_limit: int = 10  # 每秒请求数
    timeout: int = 30  # 请求超时秒数


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    leverage: int = 1
    reduce_only: bool = False
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    position_side: PositionSide = PositionSide.BOTH


@dataclass
class OrderResponse:
    """订单响应"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    fee: Decimal = Decimal("0")
    fee_asset: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int
    margin: Decimal
    margin_type: str = "cross"  # cross or isolated
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Balance:
    """账户余额"""
    asset: str
    total: Decimal
    available: Decimal
    locked: Decimal
    unrealized_pnl: Decimal = Decimal("0")


@dataclass
class Ticker:
    """行情数据"""
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


class BaseExchangeConnector(ABC):
    """交易所连接器基类"""

    def __init__(self, config: ExchangeConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_connected = False
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
        self._last_request_time = 0.0

        # 回调函数
        self._order_update_callback: Optional[Callable] = None
        self._position_update_callback: Optional[Callable] = None

    @property
    @abstractmethod
    def base_url(self) -> str:
        """API 基础 URL"""
        pass

    @property
    @abstractmethod
    def ws_url(self) -> str:
        """WebSocket URL"""
        pass

    async def connect(self) -> bool:
        """连接到交易所"""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            # 测试连接
            await self._test_connection()
            self._is_connected = True
            logger.info(f"Connected to {self.config.exchange_type.value}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """断开连接"""
        self._is_connected = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"Disconnected from {self.config.exchange_type.value}")

    @abstractmethod
    async def _test_connection(self):
        """测试连接"""
        pass

    @abstractmethod
    def _sign_request(self, method: str, path: str, params: Dict) -> Dict:
        """签名请求"""
        pass

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        signed: bool = False,
    ) -> Dict:
        """发送请求"""
        async with self._rate_limiter:
            # 速率限制
            elapsed = time.time() - self._last_request_time
            if elapsed < 1.0 / self.config.rate_limit:
                await asyncio.sleep(1.0 / self.config.rate_limit - elapsed)

            url = f"{self.base_url}{path}"
            headers = {"Content-Type": "application/json"}

            if signed:
                sign_data = self._sign_request(method, path, params or data or {})
                headers.update(sign_data.get("headers", {}))
                if "params" in sign_data:
                    params = sign_data["params"]

            self._last_request_time = time.time()

            async with self._session.request(
                method, url, params=params, json=data, headers=headers
            ) as response:
                result = await response.json()
                if response.status >= 400:
                    raise Exception(f"API Error: {result}")
                return result

    # ============== 交易接口 ==============

    @abstractmethod
    async def create_order(self, request: OrderRequest) -> OrderResponse:
        """创建订单"""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        """查询订单"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """获取挂单"""
        pass

    # ============== 仓位接口 ==============

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """获取持仓"""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """设置杠杆"""
        pass

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """设置保证金模式"""
        pass

    # ============== 账户接口 ==============

    @abstractmethod
    async def get_balance(self) -> List[Balance]:
        """获取余额"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """获取行情"""
        pass

    # ============== 回调注册 ==============

    def on_order_update(self, callback: Callable):
        """注册订单更新回调"""
        self._order_update_callback = callback

    def on_position_update(self, callback: Callable):
        """注册持仓更新回调"""
        self._position_update_callback = callback


class BinancePerpetualConnector(BaseExchangeConnector):
    """Binance 永续合约连接器"""

    @property
    def base_url(self) -> str:
        if self.config.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    @property
    def ws_url(self) -> str:
        if self.config.testnet:
            return "wss://stream.binancefuture.com"
        return "wss://fstream.binance.com"

    async def _test_connection(self):
        result = await self._request("GET", "/fapi/v1/time")
        logger.debug(f"Binance server time: {result}")

    def _sign_request(self, method: str, path: str, params: Dict) -> Dict:
        timestamp = int(time.time() * 1000)
        params["timestamp"] = timestamp

        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

        params["signature"] = signature

        return {
            "headers": {"X-MBX-APIKEY": self.config.api_key},
            "params": params,
        }

    async def create_order(self, request: OrderRequest) -> OrderResponse:
        params = {
            "symbol": request.symbol.upper().replace("-", "").replace("/", ""),
            "side": request.side.value.upper(),
            "type": self._convert_order_type(request.order_type),
            "quantity": str(request.quantity),
        }

        if request.price:
            params["price"] = str(request.price)
            params["timeInForce"] = request.time_in_force

        if request.stop_price:
            params["stopPrice"] = str(request.stop_price)

        if request.reduce_only:
            params["reduceOnly"] = "true"

        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        if request.position_side != PositionSide.BOTH:
            params["positionSide"] = request.position_side.value.upper()

        result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)

        return OrderResponse(
            order_id=str(result["orderId"]),
            client_order_id=result.get("clientOrderId", ""),
            symbol=result["symbol"],
            side=OrderSide(result["side"].lower()),
            order_type=request.order_type,
            status=self._convert_order_status(result["status"]),
            quantity=Decimal(result["origQty"]),
            filled_quantity=Decimal(result.get("executedQty", "0")),
            price=Decimal(result["price"]) if result.get("price") else None,
            average_price=Decimal(result["avgPrice"]) if result.get("avgPrice") else None,
            raw=result,
        )

    def _convert_order_type(self, order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_MARKET: "STOP_MARKET",
            OrderType.STOP_LIMIT: "STOP",
            OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT",
            OrderType.TRAILING_STOP: "TRAILING_STOP_MARKET",
        }
        return mapping.get(order_type, "MARKET")

    def _convert_order_status(self, status: str) -> OrderStatus:
        mapping = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.PENDING)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "orderId": order_id,
        }
        try:
            await self._request("DELETE", "/fapi/v1/order", params=params, signed=True)
            return True
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    async def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "orderId": order_id,
        }
        result = await self._request("GET", "/fapi/v1/order", params=params, signed=True)

        return OrderResponse(
            order_id=str(result["orderId"]),
            client_order_id=result.get("clientOrderId", ""),
            symbol=result["symbol"],
            side=OrderSide(result["side"].lower()),
            order_type=OrderType.LIMIT,  # 简化
            status=self._convert_order_status(result["status"]),
            quantity=Decimal(result["origQty"]),
            filled_quantity=Decimal(result.get("executedQty", "0")),
            price=Decimal(result["price"]) if result.get("price") else None,
            average_price=Decimal(result["avgPrice"]) if result.get("avgPrice") else None,
            raw=result,
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        params = {}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")

        results = await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)

        orders = []
        for r in results:
            orders.append(OrderResponse(
                order_id=str(r["orderId"]),
                client_order_id=r.get("clientOrderId", ""),
                symbol=r["symbol"],
                side=OrderSide(r["side"].lower()),
                order_type=OrderType.LIMIT,
                status=self._convert_order_status(r["status"]),
                quantity=Decimal(r["origQty"]),
                filled_quantity=Decimal(r.get("executedQty", "0")),
                price=Decimal(r["price"]) if r.get("price") else None,
                raw=r,
            ))
        return orders

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        result = await self._request("GET", "/fapi/v2/positionRisk", params={}, signed=True)

        positions = []
        for r in result:
            qty = Decimal(r["positionAmt"])
            if qty == 0 and symbol:
                continue
            if symbol and r["symbol"] != symbol.upper().replace("-", "").replace("/", ""):
                continue

            positions.append(Position(
                symbol=r["symbol"],
                side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                quantity=abs(qty),
                entry_price=Decimal(r["entryPrice"]),
                mark_price=Decimal(r["markPrice"]),
                liquidation_price=Decimal(r["liquidationPrice"]) if r["liquidationPrice"] != "0" else Decimal("0"),
                unrealized_pnl=Decimal(r["unRealizedProfit"]),
                realized_pnl=Decimal("0"),  # 需要另外查询
                leverage=int(r["leverage"]),
                margin=Decimal(r.get("isolatedMargin", "0")),
                margin_type=r.get("marginType", "cross"),
                raw=r,
            ))
        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "leverage": leverage,
        }
        try:
            await self._request("POST", "/fapi/v1/leverage", params=params, signed=True)
            return True
        except Exception as e:
            logger.error(f"Set leverage failed: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        params = {
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "marginType": margin_type.upper(),
        }
        try:
            await self._request("POST", "/fapi/v1/marginType", params=params, signed=True)
            return True
        except Exception as e:
            # 可能已经是该模式
            if "No need to change margin type" in str(e):
                return True
            logger.error(f"Set margin type failed: {e}")
            return False

    async def get_balance(self) -> List[Balance]:
        result = await self._request("GET", "/fapi/v2/balance", params={}, signed=True)

        balances = []
        for r in result:
            balances.append(Balance(
                asset=r["asset"],
                total=Decimal(r["balance"]),
                available=Decimal(r["availableBalance"]),
                locked=Decimal(r["balance"]) - Decimal(r["availableBalance"]),
                unrealized_pnl=Decimal(r.get("crossUnPnl", "0")),
            ))
        return balances

    async def get_ticker(self, symbol: str) -> Ticker:
        params = {"symbol": symbol.upper().replace("-", "").replace("/", "")}
        result = await self._request("GET", "/fapi/v1/ticker/24hr", params=params)

        return Ticker(
            symbol=result["symbol"],
            last_price=Decimal(result["lastPrice"]),
            bid_price=Decimal(result["bidPrice"]),
            ask_price=Decimal(result["askPrice"]),
            volume_24h=Decimal(result["volume"]),
            high_24h=Decimal(result["highPrice"]),
            low_24h=Decimal(result["lowPrice"]),
            timestamp=datetime.fromtimestamp(result["closeTime"] / 1000),
        )


class BybitPerpetualConnector(BaseExchangeConnector):
    """Bybit 永续合约连接器"""

    @property
    def base_url(self) -> str:
        if self.config.testnet:
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"

    @property
    def ws_url(self) -> str:
        if self.config.testnet:
            return "wss://stream-testnet.bybit.com"
        return "wss://stream.bybit.com"

    async def _test_connection(self):
        result = await self._request("GET", "/v5/market/time")
        logger.debug(f"Bybit server time: {result}")

    def _sign_request(self, method: str, path: str, params: Dict) -> Dict:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        if method == "GET":
            query_string = urlencode(params) if params else ""
            sign_payload = f"{timestamp}{self.config.api_key}{recv_window}{query_string}"
        else:
            import json
            sign_payload = f"{timestamp}{self.config.api_key}{recv_window}{json.dumps(params)}"

        signature = hmac.new(
            self.config.api_secret.encode(),
            sign_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "headers": {
                "X-BAPI-API-KEY": self.config.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-SIGN": signature,
                "X-BAPI-RECV-WINDOW": recv_window,
            },
            "params": params,
        }

    async def create_order(self, request: OrderRequest) -> OrderResponse:
        data = {
            "category": "linear",
            "symbol": request.symbol.upper().replace("-", "").replace("/", ""),
            "side": "Buy" if request.side == OrderSide.BUY else "Sell",
            "orderType": "Market" if request.order_type == OrderType.MARKET else "Limit",
            "qty": str(request.quantity),
        }

        if request.price:
            data["price"] = str(request.price)

        if request.reduce_only:
            data["reduceOnly"] = True

        if request.client_order_id:
            data["orderLinkId"] = request.client_order_id

        result = await self._request("POST", "/v5/order/create", data=data, signed=True)
        result_data = result.get("result", {})

        return OrderResponse(
            order_id=result_data.get("orderId", ""),
            client_order_id=result_data.get("orderLinkId", ""),
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            status=OrderStatus.OPEN,
            quantity=request.quantity,
            raw=result,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        data = {
            "category": "linear",
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "orderId": order_id,
        }
        try:
            await self._request("POST", "/v5/order/cancel", data=data, signed=True)
            return True
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    async def get_order(self, symbol: str, order_id: str) -> OrderResponse:
        params = {
            "category": "linear",
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "orderId": order_id,
        }
        result = await self._request("GET", "/v5/order/realtime", params=params, signed=True)
        orders = result.get("result", {}).get("list", [])

        if not orders:
            raise Exception(f"Order not found: {order_id}")

        r = orders[0]
        return OrderResponse(
            order_id=r["orderId"],
            client_order_id=r.get("orderLinkId", ""),
            symbol=r["symbol"],
            side=OrderSide.BUY if r["side"] == "Buy" else OrderSide.SELL,
            order_type=OrderType.MARKET if r["orderType"] == "Market" else OrderType.LIMIT,
            status=self._convert_order_status(r["orderStatus"]),
            quantity=Decimal(r["qty"]),
            filled_quantity=Decimal(r.get("cumExecQty", "0")),
            price=Decimal(r["price"]) if r.get("price") else None,
            average_price=Decimal(r["avgPrice"]) if r.get("avgPrice") else None,
            raw=r,
        )

    def _convert_order_status(self, status: str) -> OrderStatus:
        mapping = {
            "New": OrderStatus.OPEN,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.PENDING)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")

        result = await self._request("GET", "/v5/order/realtime", params=params, signed=True)
        orders = []

        for r in result.get("result", {}).get("list", []):
            orders.append(OrderResponse(
                order_id=r["orderId"],
                client_order_id=r.get("orderLinkId", ""),
                symbol=r["symbol"],
                side=OrderSide.BUY if r["side"] == "Buy" else OrderSide.SELL,
                order_type=OrderType.MARKET if r["orderType"] == "Market" else OrderType.LIMIT,
                status=self._convert_order_status(r["orderStatus"]),
                quantity=Decimal(r["qty"]),
                filled_quantity=Decimal(r.get("cumExecQty", "0")),
                raw=r,
            ))
        return orders

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        params = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol.upper().replace("-", "").replace("/", "")

        result = await self._request("GET", "/v5/position/list", params=params, signed=True)
        positions = []

        for r in result.get("result", {}).get("list", []):
            qty = Decimal(r["size"])
            if qty == 0:
                continue

            positions.append(Position(
                symbol=r["symbol"],
                side=PositionSide.LONG if r["side"] == "Buy" else PositionSide.SHORT,
                quantity=qty,
                entry_price=Decimal(r["avgPrice"]),
                mark_price=Decimal(r["markPrice"]),
                liquidation_price=Decimal(r["liqPrice"]) if r.get("liqPrice") else Decimal("0"),
                unrealized_pnl=Decimal(r["unrealisedPnl"]),
                realized_pnl=Decimal(r.get("cumRealisedPnl", "0")),
                leverage=int(r["leverage"]),
                margin=Decimal(r.get("positionIM", "0")),
                raw=r,
            ))
        return positions

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        data = {
            "category": "linear",
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        try:
            await self._request("POST", "/v5/position/set-leverage", data=data, signed=True)
            return True
        except Exception as e:
            logger.error(f"Set leverage failed: {e}")
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        # Bybit 通过 switch-isolated 接口
        data = {
            "category": "linear",
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
            "tradeMode": 1 if margin_type.lower() == "isolated" else 0,
            "buyLeverage": "10",
            "sellLeverage": "10",
        }
        try:
            await self._request("POST", "/v5/position/switch-isolated", data=data, signed=True)
            return True
        except Exception as e:
            logger.error(f"Set margin type failed: {e}")
            return False

    async def get_balance(self) -> List[Balance]:
        params = {"accountType": "UNIFIED"}
        result = await self._request("GET", "/v5/account/wallet-balance", params=params, signed=True)

        balances = []
        for account in result.get("result", {}).get("list", []):
            for coin in account.get("coin", []):
                balances.append(Balance(
                    asset=coin["coin"],
                    total=Decimal(coin["walletBalance"]),
                    available=Decimal(coin["availableToWithdraw"]),
                    locked=Decimal(coin["locked"]),
                    unrealized_pnl=Decimal(coin.get("unrealisedPnl", "0")),
                ))
        return balances

    async def get_ticker(self, symbol: str) -> Ticker:
        params = {
            "category": "linear",
            "symbol": symbol.upper().replace("-", "").replace("/", ""),
        }
        result = await self._request("GET", "/v5/market/tickers", params=params)
        tickers = result.get("result", {}).get("list", [])

        if not tickers:
            raise Exception(f"Ticker not found: {symbol}")

        r = tickers[0]
        return Ticker(
            symbol=r["symbol"],
            last_price=Decimal(r["lastPrice"]),
            bid_price=Decimal(r["bid1Price"]),
            ask_price=Decimal(r["ask1Price"]),
            volume_24h=Decimal(r["volume24h"]),
            high_24h=Decimal(r["highPrice24h"]),
            low_24h=Decimal(r["lowPrice24h"]),
            timestamp=datetime.now(),
        )


# ============== 连接器工厂 ==============

class ConnectorFactory:
    """连接器工厂"""

    _registry: Dict[ExchangeType, type] = {
        ExchangeType.BINANCE_PERPETUAL: BinancePerpetualConnector,
        ExchangeType.BYBIT_PERPETUAL: BybitPerpetualConnector,
        # OKX, Gate, Deribit 可以类似实现
    }

    @classmethod
    def create(cls, config: ExchangeConfig) -> BaseExchangeConnector:
        """创建连接器"""
        connector_cls = cls._registry.get(config.exchange_type)
        if connector_cls is None:
            raise ValueError(f"不支持的交易所: {config.exchange_type}")
        return connector_cls(config)

    @classmethod
    def list_supported(cls) -> List[str]:
        """列出支持的交易所"""
        return [t.value for t in cls._registry.keys()]


# 测试代码
if __name__ == "__main__":
    print("AlgVex Exchange Connectors v2.0.0")
    print("=" * 50)
    print("\n支持的交易所:")
    for exchange in ConnectorFactory.list_supported():
        print(f"  - {exchange}")
