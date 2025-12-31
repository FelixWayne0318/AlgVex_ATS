"""市场数据路由"""

from fastapi import APIRouter, Query
from typing import List, Optional
from datetime import datetime

router = APIRouter()


@router.get("/symbols")
async def list_symbols():
    """获取支持的交易对列表"""
    return {
        "success": True,
        "data": [
            {"symbol": "BTCUSDT", "base": "BTC", "quote": "USDT"},
            {"symbol": "ETHUSDT", "base": "ETH", "quote": "USDT"},
            {"symbol": "BNBUSDT", "base": "BNB", "quote": "USDT"},
        ]
    }


@router.get("/klines/{symbol}")
async def get_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = Query(100, le=1500),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
):
    """获取K线数据"""
    return {"success": True, "data": []}


@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str):
    """获取最新价格"""
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "price": 0,
            "change_24h": 0,
            "volume_24h": 0,
        }
    }


@router.get("/funding-rate/{symbol}")
async def get_funding_rate(symbol: str, limit: int = 100):
    """获取资金费率历史"""
    return {"success": True, "data": []}


@router.get("/open-interest/{symbol}")
async def get_open_interest(symbol: str, limit: int = 100):
    """获取持仓量历史"""
    return {"success": True, "data": []}


@router.get("/long-short-ratio/{symbol}")
async def get_long_short_ratio(symbol: str, limit: int = 100):
    """获取多空比历史"""
    return {"success": True, "data": []}


@router.get("/fear-greed")
async def get_fear_greed(limit: int = 30):
    """获取恐惧贪婪指数"""
    return {"success": True, "data": []}


@router.get("/overview")
async def get_market_overview():
    """获取市场概览"""
    return {
        "success": True,
        "data": {
            "btc_dominance": 0,
            "total_market_cap": 0,
            "fear_greed_index": 50,
            "top_gainers": [],
            "top_losers": [],
        }
    }
