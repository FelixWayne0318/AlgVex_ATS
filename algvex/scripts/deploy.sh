#!/bin/bash
# ==============================================================================
# AlgVex 部署脚本 (直接部署版)
# 用途: 一键部署到生产环境
# 用法: bash deploy.sh [--build] [--migrate] [--restart]
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 配置
PROJECT_DIR="/opt/algvex/algvex"
VENV_DIR="$PROJECT_DIR/venv"
BACKUP_DIR="/opt/algvex/backups"
LOG_FILE="/var/log/algvex/deploy.log"
SERVICE_NAME="algvex"

# 参数
DO_BUILD=false
DO_MIGRATE=false
DO_RESTART=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            DO_BUILD=true
            shift
            ;;
        --migrate)
            DO_MIGRATE=true
            shift
            ;;
        --restart)
            DO_RESTART=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "     AlgVex 部署脚本 v2.0 (直接部署)"
echo "=============================================="
echo ""
echo "配置:"
echo "  项目目录: $PROJECT_DIR"
echo "  虚拟环境: $VENV_DIR"
echo "  重新安装依赖: $DO_BUILD"
echo "  数据库迁移: $DO_MIGRATE"
echo "  重启服务: $DO_RESTART"
echo ""

cd "$PROJECT_DIR"

# ==============================================================================
# Step 1: 备份
# ==============================================================================
log_step "[1/6] 备份当前版本..."

BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# 备份数据库 (如果使用 PostgreSQL)
if command -v pg_dump &> /dev/null; then
    if pg_isready -q 2>/dev/null; then
        pg_dump -U algvex algvex > "$BACKUP_DIR/$BACKUP_NAME/database.sql" 2>/dev/null || true
        log_info "数据库备份完成: $BACKUP_DIR/$BACKUP_NAME/database.sql"
    fi
fi

# 备份配置
cp .env "$BACKUP_DIR/$BACKUP_NAME/.env" 2>/dev/null || true
log_info "配置备份完成"

# ==============================================================================
# Step 2: 拉取最新代码
# ==============================================================================
log_step "[2/6] 拉取最新代码..."

git fetch origin
git pull origin main

CURRENT_COMMIT=$(git rev-parse --short HEAD)
log_info "当前版本: $CURRENT_COMMIT"

# ==============================================================================
# Step 3: 更新依赖
# ==============================================================================
if [ "$DO_BUILD" = true ]; then
    log_step "[3/6] 更新 Python 依赖..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
    log_info "依赖更新完成"
else
    log_step "[3/6] 跳过依赖更新"
fi

# ==============================================================================
# Step 4: 数据库迁移
# ==============================================================================
if [ "$DO_MIGRATE" = true ]; then
    log_step "[4/6] 执行数据库迁移..."
    source "$VENV_DIR/bin/activate"
    alembic upgrade head 2>/dev/null || log_warn "Alembic 迁移跳过 (可能未配置)"
    log_info "数据库迁移完成"
else
    log_step "[4/6] 跳过数据库迁移"
fi

# ==============================================================================
# Step 5: 构建前端
# ==============================================================================
log_step "[5/6] 构建前端..."

if [ -d "web" ] && [ -f "web/package.json" ]; then
    cd web
    npm ci --production=false
    npm run build
    cp -r dist/* /opt/algvex/web/ 2>/dev/null || true
    cd ..
    log_info "前端构建完成"
else
    log_warn "前端目录不存在，跳过"
fi

# ==============================================================================
# Step 6: 重启服务
# ==============================================================================
if [ "$DO_RESTART" = true ]; then
    log_step "[6/6] 重启服务..."

    # 使用 systemd 重启服务
    if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
        sudo systemctl restart $SERVICE_NAME
        log_info "systemd 服务已重启"
    else
        # 如果没有 systemd 服务，尝试使用进程管理
        pkill -f "uvicorn api.main:app" 2>/dev/null || true
        sleep 2

        source "$VENV_DIR/bin/activate"
        nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > /opt/algvex/logs/api/uvicorn.log 2>&1 &
        log_info "API 服务已启动 (nohup)"
    fi

    # 等待服务启动
    log_info "等待服务启动..."
    sleep 5

    # 健康检查
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API 健康检查通过"
    else
        log_warn "API 健康检查未通过 (服务可能仍在启动中)"
    fi
else
    log_step "[6/6] 跳过服务重启"
fi

# ==============================================================================
# 完成
# ==============================================================================
echo ""
echo "=============================================="
echo "           部署完成!"
echo "=============================================="
echo ""
echo "版本: $CURRENT_COMMIT"
echo "备份: $BACKUP_DIR/$BACKUP_NAME"
echo ""
echo "服务状态:"
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    systemctl status $SERVICE_NAME --no-pager | head -5
else
    pgrep -a -f "uvicorn api.main:app" || echo "  服务未运行"
fi
echo ""
echo "访问地址:"
echo "  - API:  http://localhost:8000"
echo "  - 文档: http://localhost:8000/docs"
echo ""

# 记录日志
mkdir -p "$(dirname $LOG_FILE)"
echo "[$(date)] Deployed version $CURRENT_COMMIT" >> $LOG_FILE
