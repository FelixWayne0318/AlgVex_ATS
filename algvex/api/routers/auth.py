"""
认证路由
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import settings

router = APIRouter()

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ==============================================================================
# Schemas
# ==============================================================================

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_active: bool
    is_verified: bool
    role: str
    created_at: datetime


# ==============================================================================
# 工具函数
# ==============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """创建刷新令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception

    # TODO: 从数据库获取用户
    # user = await get_user_by_id(token_data.user_id)
    # if user is None:
    #     raise credentials_exception
    # return user

    return {"id": user_id, "email": payload.get("email")}


# ==============================================================================
# 路由
# ==============================================================================

@router.post("/register", response_model=dict)
async def register(user: UserRegister):
    """
    用户注册

    - **email**: 邮箱地址
    - **username**: 用户名
    - **password**: 密码 (至少8位)
    """
    # TODO: 检查邮箱是否已存在
    # TODO: 检查用户名是否已存在
    # TODO: 创建用户

    hashed_password = get_password_hash(user.password)

    return {
        "success": True,
        "message": "注册成功，请查收验证邮件",
        "data": {
            "email": user.email,
            "username": user.username,
        }
    }


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    用户登录

    使用邮箱和密码登录，返回访问令牌
    """
    # TODO: 从数据库获取用户并验证

    # 临时模拟
    if form_data.username != "admin@algvex.com":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建令牌
    access_token = create_access_token(
        data={"sub": "user_id_123", "email": form_data.username},
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_refresh_token(
        data={"sub": "user_id_123", "email": form_data.username}
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    刷新访问令牌

    使用刷新令牌获取新的访问令牌
    """
    try:
        payload = jwt.decode(
            refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        user_id = payload.get("sub")
        email = payload.get("email")

        access_token = create_access_token(
            data={"sub": user_id, "email": email},
            expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        new_refresh_token = create_refresh_token(
            data={"sub": user_id, "email": email}
        )

        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=dict)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    获取当前用户信息
    """
    return {
        "success": True,
        "data": current_user,
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    用户登出

    使当前令牌失效
    """
    # TODO: 将令牌加入黑名单

    return {
        "success": True,
        "message": "已登出"
    }
