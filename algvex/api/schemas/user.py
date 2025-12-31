"""
用户 Schema
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """用户基础字段"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    """创建用户"""

    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """更新用户"""

    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """用户响应"""

    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """用户登录"""

    username: str
    password: str


class Token(BaseModel):
    """JWT Token"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="过期时间 (秒)")


class TokenPayload(BaseModel):
    """Token 载荷"""

    sub: int  # user_id
    exp: datetime
    iat: datetime
