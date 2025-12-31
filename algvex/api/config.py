"""
AlgVex API 配置管理
"""

from typing import List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""

    # 应用
    APP_NAME: str = "AlgVex"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev_secret_key_change_in_production"

    # 数据库
    DATABASE_URL: str = "postgresql://algvex:algvex@localhost:5432/algvex"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    JWT_SECRET_KEY: str = "jwt_secret_key_change_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://algvex.com",
        "https://www.algvex.com",
    ]

    # 币安
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True

    # 加密
    ENCRYPTION_MASTER_KEY: str = "encryption_master_key_32_chars_"

    # 日志
    LOG_LEVEL: str = "INFO"

    # 数据目录
    DATA_DIR: str = "/opt/algvex/data"
    MODEL_DIR: str = "/opt/algvex/data/models"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
