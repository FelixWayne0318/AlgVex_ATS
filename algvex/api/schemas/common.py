"""
通用 Schema
"""

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应"""

    items: List[T]
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    pages: int = Field(..., description="总页数")

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """错误响应"""

    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    details: Optional[Any] = Field(None, description="详细信息")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="状态: healthy/unhealthy")
    version: str = Field(..., description="版本号")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: Optional[dict] = Field(None, description="组件状态")


class ConfigResponse(BaseModel):
    """配置响应"""

    config_version: str
    config_hash: str
    mvp_factors: List[str]
    allowed_symbols: List[str]
    allowed_frequencies: List[str]
