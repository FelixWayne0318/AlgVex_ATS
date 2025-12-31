"""
策略 Schema
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StrategyBase(BaseModel):
    """策略基础字段"""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    factors: List[str] = Field(default_factory=list, description="使用的因子列表")
    symbols: List[str] = Field(default_factory=list, description="交易标的")
    frequency: str = Field(default="5m", description="时间框架")


class StrategyCreate(StrategyBase):
    """创建策略"""

    config: Optional[Dict[str, Any]] = Field(None, description="策略配置")


class StrategyUpdate(BaseModel):
    """更新策略"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    factors: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    frequency: Optional[str] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None


class StrategyResponse(StrategyBase):
    """策略响应"""

    id: int
    owner_id: int
    is_active: bool
    is_public: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StrategyVersionResponse(BaseModel):
    """策略版本响应"""

    id: int
    strategy_id: int
    version: str
    config_hash: str
    changelog: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StrategyWithVersions(StrategyResponse):
    """包含版本的策略响应"""

    versions: List[StrategyVersionResponse] = Field(default_factory=list)
