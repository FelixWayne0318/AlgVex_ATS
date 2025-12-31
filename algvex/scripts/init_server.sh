#!/bin/bash
# ==============================================================================
# AlgVex 服务器初始化脚本 (直接部署版)
# 用途: 在全新Ubuntu 22.04服务器上安装所有依赖
# 用法: sudo bash init_server.sh [--with-db]
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

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
echo "  AlgVex 服务器初始化脚本 v2.0 (直接部署)"
echo "=============================================="
echo ""

# 检查root权限
if [ "$EUID" -ne 0 ]; then
    log_error "请使用 sudo 运行此脚本"
    exit 1
fi

# 1. 系统更新
log_info "[1/8] 更新系统包..."
apt update && apt upgrade -y

# 2. 安装基础工具
log_info "[2/8] 安装基础工具..."
apt install -y \
    curl wget git vim htop tree \
    build-essential cmake \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg lsb-release \
    unzip jq

# 3. 安装 Node.js 20
log_info "[3/8] 安装 Node.js 20..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
    log_info "Node.js 安装完成: $(node --version)"
else
    log_warn "Node.js 已安装: $(node --version)"
fi

# 4. 安装 Python 3.11
log_info "[4/8] 安装 Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    log_info "Python 安装完成: $(python3.11 --version)"
else
    log_warn "Python 3.11 已安装: $(python3.11 --version)"
fi

# 5. 安装数据库 (可选)
if [ "$INSTALL_DB" = true ]; then
    log_info "[5/8] 安装数据库服务..."

    # PostgreSQL + TimescaleDB
    if ! command -v psql &> /dev/null; then
        apt install -y postgresql postgresql-contrib
        systemctl enable postgresql
        systemctl start postgresql
        log_info "PostgreSQL 安装完成"
    else
        log_warn "PostgreSQL 已安装"
    fi

    # Redis
    if ! command -v redis-cli &> /dev/null; then
        apt install -y redis-server
        systemctl enable redis-server
        systemctl start redis-server
        log_info "Redis 安装完成"
    else
        log_warn "Redis 已安装"
    fi
else
    log_info "[5/8] 跳过数据库安装 (使用 --with-db 参数安装)"
fi

# 6. 创建项目目录结构
log_info "[6/8] 创建项目目录..."
mkdir -p /opt/algvex/{data/{raw,processed,models},logs/{api,worker,trading},backups,web}
mkdir -p /var/log/algvex

# 设置权限
chown -R $SUDO_USER:$SUDO_USER /opt/algvex

# 7. 配置防火墙
log_info "[7/8] 配置防火墙..."
if command -v ufw &> /dev/null; then
    ufw allow 22/tcp comment 'SSH'
    ufw allow 80/tcp comment 'HTTP'
    ufw allow 443/tcp comment 'HTTPS'
    ufw allow 8000/tcp comment 'API'
    ufw --force enable
    log_info "防火墙配置完成"
else
    log_warn "UFW 未安装，跳过防火墙配置"
fi

# 8. 配置系统参数
log_info "[8/8] 优化系统参数..."
cat >> /etc/sysctl.conf << 'EOF'
# AlgVex 优化配置
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_tw_reuse = 1
vm.overcommit_memory = 1
EOF
sysctl -p

# 配置时区
timedatectl set-timezone UTC
log_info "时区设置为 UTC"

# 完成
echo ""
echo "=============================================="
echo "       服务器初始化完成!"
echo "=============================================="
echo ""
echo "已安装组件:"
echo "  - Python:  $(python3.11 --version 2>/dev/null || echo '未安装')"
echo "  - Node.js: $(node --version 2>/dev/null || echo '未安装')"
if [ "$INSTALL_DB" = true ]; then
    echo "  - PostgreSQL: $(psql --version 2>/dev/null || echo '未安装')"
    echo "  - Redis: $(redis-cli --version 2>/dev/null || echo '未安装')"
fi
echo ""
echo "下一步操作:"
echo "  1. cd /opt/algvex"
echo "  2. git clone <your-repo> algvex"
echo "  3. cd algvex && bash scripts/setup.sh"
echo ""
echo "目录结构:"
tree -L 2 /opt/algvex
echo ""
