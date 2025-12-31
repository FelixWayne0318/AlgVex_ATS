#!/bin/bash
# ==============================================================================
# AlgVex 项目设置脚本 (直接部署版)
# 用途: 克隆依赖项目、安装依赖、初始化环境
# 用法: bash setup.sh [--with-db]
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="/opt/algvex/deps"

# 参数
INSTALL_DB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-db)
            INSTALL_DB=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "   AlgVex 项目设置脚本 v2.0 (直接部署)"
echo "=============================================="
echo ""

cd "$PROJECT_DIR"

# ==============================================================================
# Step 1: 克隆依赖项目
# ==============================================================================
log_step "[1/6] 克隆依赖项目..."

mkdir -p "$DEPS_DIR"
cd "$DEPS_DIR"

# 克隆 Qlib
if [ ! -d "qlib" ]; then
    log_info "克隆 Microsoft Qlib..."
    git clone --depth 1 --branch v0.9.7 https://github.com/microsoft/qlib.git
    log_info "Qlib 克隆完成: v0.9.7"
else
    log_warn "Qlib 已存在，跳过"
fi

# 克隆 Hummingbot (企业级执行引擎)
if [ ! -d "hummingbot" ]; then
    log_info "克隆 Hummingbot v2.11.0..."
    git clone --depth 1 --branch v2.11.0 https://github.com/hummingbot/hummingbot.git
    log_info "Hummingbot 克隆完成: v2.11.0 (15k+ stars, 企业级执行引擎)"
else
    log_warn "Hummingbot 已存在，跳过"
fi

cd "$PROJECT_DIR"

# ==============================================================================
# Step 2: 创建 Python 虚拟环境
# ==============================================================================
log_step "[2/6] 创建 Python 虚拟环境..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "虚拟环境创建完成"
else
    log_warn "虚拟环境已存在"
fi

source venv/bin/activate

# ==============================================================================
# Step 3: 安装 Python 依赖
# ==============================================================================
log_step "[3/6] 安装 Python 依赖..."

pip install --upgrade pip
pip install -r requirements.txt

log_info "Python 依赖安装完成"

# ==============================================================================
# Step 4: 配置环境变量
# ==============================================================================
log_step "[4/6] 配置环境变量..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    log_info "已创建 .env 文件，请编辑配置"
    log_warn "重要: 请修改 .env 中的敏感配置!"
else
    log_warn ".env 已存在，跳过"
fi

# ==============================================================================
# Step 5: 安装数据库 (可选)
# ==============================================================================
if [ "$INSTALL_DB" = true ]; then
    log_step "[5/6] 安装数据库服务..."

    if command -v apt &> /dev/null; then
        # Ubuntu/Debian
        sudo apt update
        sudo apt install -y postgresql redis-server

        # 启动服务
        sudo systemctl enable postgresql redis-server
        sudo systemctl start postgresql redis-server

        # 创建数据库和用户
        sudo -u postgres psql -c "CREATE USER algvex WITH PASSWORD 'algvex_password';" 2>/dev/null || true
        sudo -u postgres psql -c "CREATE DATABASE algvex OWNER algvex;" 2>/dev/null || true

        log_info "PostgreSQL 和 Redis 安装完成"
        log_warn "数据库密码: algvex_password (请在 .env 中修改)"
    else
        log_warn "非 apt 系统，请手动安装 PostgreSQL 和 Redis"
    fi
else
    log_step "[5/6] 跳过数据库安装 (使用 --with-db 参数安装)"
fi

# ==============================================================================
# Step 6: 安装前端依赖
# ==============================================================================
log_step "[6/6] 安装前端依赖..."

if [ -d "web" ] && [ -f "web/package.json" ]; then
    cd web
    npm install
    cd ..
    log_info "前端依赖安装完成"
else
    log_warn "前端目录不存在或未初始化"
fi

# ==============================================================================
# 完成
# ==============================================================================
echo ""
echo "=============================================="
echo "           项目设置完成!"
echo "=============================================="
echo ""
echo "依赖项目位置:"
echo "  - Qlib:       $DEPS_DIR/qlib (信号层)"
echo "  - Hummingbot: $DEPS_DIR/hummingbot (执行层)"
echo ""
echo "下一步操作:"
echo ""
echo "  1. 编辑配置文件:"
echo "     vim .env"
echo ""
echo "  2. 激活虚拟环境:"
echo "     source venv/bin/activate"
echo ""
echo "  3. 运行回测:"
echo "     python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT"
echo ""
echo "  4. 启动后端 API (可选):"
echo "     uvicorn api.main:app --reload --port 8000"
echo ""
echo "  5. 启动前端 (另一个终端):"
echo "     cd web && npm run dev"
echo ""
echo "  6. 访问:"
echo "     - 前端: http://localhost:3000"
echo "     - API:  http://localhost:8000/docs"
echo ""
