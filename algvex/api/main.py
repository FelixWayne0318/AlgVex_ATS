"""
AlgVex API - FastAPI 应用入口

功能:
- 用户认证与授权
- 策略管理
- 回测服务
- 交易服务
- 市场数据
- WebSocket 实时推送
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import time

from .config import settings
from .database import engine, Base
from .routers import auth, users, strategies, backtests, trades, market


# ==============================================================================
# 生命周期管理
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("AlgVex API starting...")

    # 创建数据库表 (开发环境)
    if settings.DEBUG:
        async with engine.begin() as conn:
            # await conn.run_sync(Base.metadata.create_all)
            pass

    logger.info(f"AlgVex API started on {settings.APP_ENV} environment")

    yield

    # 关闭时
    logger.info("AlgVex API shutting down...")
    await engine.dispose()


# ==============================================================================
# 创建应用
# ==============================================================================

app = FastAPI(
    title="AlgVex API",
    description="""
    AlgVex - 专业加密货币量化交易平台

    ## 功能

    * **用户管理** - 注册、登录、API密钥管理
    * **策略管理** - 创建、编辑、版本控制
    * **回测服务** - 历史回测、性能分析
    * **交易服务** - 模拟盘、实盘交易
    * **市场数据** - K线、资金费率、持仓量

    ## 认证

    使用 JWT Bearer Token 认证。在请求头中添加:
    ```
    Authorization: Bearer <your_token>
    ```
    """,
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)


# ==============================================================================
# 中间件
# ==============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.3f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# ==============================================================================
# 异常处理
# ==============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """验证错误处理"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Validation error",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error",
        },
    )


# ==============================================================================
# 路由注册
# ==============================================================================

app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(users.router, prefix="/api/users", tags=["用户"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["策略"])
app.include_router(backtests.router, prefix="/api/backtests", tags=["回测"])
app.include_router(trades.router, prefix="/api/trades", tags=["交易"])
app.include_router(market.router, prefix="/api/market", tags=["市场"])


# ==============================================================================
# 基础路由
# ==============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """根路径"""
    return {
        "name": "AlgVex API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.APP_ENV,
    }


@app.get("/api/status", tags=["系统"])
async def api_status():
    """API状态"""
    return {
        "success": True,
        "data": {
            "name": "AlgVex",
            "version": "1.0.0",
            "environment": settings.APP_ENV,
            "features": {
                "trading": True,
                "backtest": True,
                "paper_trading": True,
                "live_trading": settings.APP_ENV == "production",
            },
        },
    }


# ==============================================================================
# WebSocket (实时数据)
# ==============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from typing import List


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {len(self.active_connections)} connections")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {len(self.active_connections)} connections")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            # 处理订阅请求
            if data.get("action") == "subscribe":
                channel = data.get("channel")
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": channel,
                })

            # 处理心跳
            elif data.get("action") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==============================================================================
# 启动入口
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
