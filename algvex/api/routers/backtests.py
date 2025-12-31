"""回测管理路由"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import date

router = APIRouter()


class BacktestCreate(BaseModel):
    strategy_id: str
    start_date: date
    end_date: date
    symbols: List[str]
    initial_capital: float = 100000
    leverage: float = 1
    config: dict = {}


@router.get("/")
async def list_backtests():
    """获取回测列表"""
    return {"success": True, "data": [], "total": 0}


@router.post("/")
async def create_backtest(backtest: BacktestCreate):
    """创建回测任务"""
    return {
        "success": True,
        "message": "回测任务已创建",
        "data": {"id": "backtest_123", "status": "pending"}
    }


@router.get("/{backtest_id}")
async def get_backtest(backtest_id: str):
    """获取回测详情"""
    return {
        "success": True,
        "data": {
            "id": backtest_id,
            "status": "completed",
            "metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.1,
                "win_rate": 0.55,
            }
        }
    }


@router.delete("/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """删除回测"""
    return {"success": True, "message": "回测删除成功"}


@router.get("/{backtest_id}/equity")
async def get_backtest_equity(backtest_id: str):
    """获取回测权益曲线"""
    return {"success": True, "data": []}


@router.get("/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: str):
    """获取回测交易记录"""
    return {"success": True, "data": [], "total": 0}


@router.post("/{backtest_id}/compare")
async def compare_backtests(backtest_id: str, compare_ids: List[str]):
    """对比多个回测结果"""
    return {"success": True, "data": {}}
