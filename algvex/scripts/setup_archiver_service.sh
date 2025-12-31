#!/bin/bash
# AlgVex 数据落盘服务安装脚本
#
# 用法:
#   ./setup_archiver_service.sh install   # 安装并启动服务
#   ./setup_archiver_service.sh uninstall # 卸载服务
#   ./setup_archiver_service.sh status    # 查看服务状态
#   ./setup_archiver_service.sh logs      # 查看服务日志

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="algvex-archiver"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_PATH=$(which python3)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

install_service() {
    log_info "Installing AlgVex Data Archiver service..."

    # 检查是否有 root 权限
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi

    # 创建 systemd 服务文件
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AlgVex Cryptocurrency Data Archiver
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${SUDO_USER:-$USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON_PATH} ${SCRIPT_DIR}/data_archiver.py --interval 3600
Restart=always
RestartSec=60

# 环境变量
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/..

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# 资源限制
MemoryMax=1G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

    # 重新加载 systemd
    systemctl daemon-reload

    # 启用并启动服务
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"

    log_info "Service installed and started successfully!"
    log_info "Use 'sudo systemctl status ${SERVICE_NAME}' to check status"
    log_info "Use 'journalctl -u ${SERVICE_NAME} -f' to follow logs"
}

uninstall_service() {
    log_info "Uninstalling AlgVex Data Archiver service..."

    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi

    # 停止并禁用服务
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true

    # 删除服务文件
    rm -f "$SERVICE_FILE"

    # 重新加载 systemd
    systemctl daemon-reload

    log_info "Service uninstalled successfully!"
}

show_status() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service is running"
        systemctl status "$SERVICE_NAME" --no-pager
    else
        log_warn "Service is not running"
        log_info "Start with: sudo systemctl start ${SERVICE_NAME}"
    fi
}

show_logs() {
    log_info "Showing logs (Ctrl+C to exit)..."
    journalctl -u "$SERVICE_NAME" -f
}

setup_cron() {
    log_info "Setting up cron job..."

    # 添加 cron 任务 (每小时运行一次)
    CRON_CMD="0 * * * * cd ${PROJECT_DIR} && ${PYTHON_PATH} ${SCRIPT_DIR}/data_archiver.py --once >> ~/.algvex/logs/cron.log 2>&1"

    # 检查是否已存在
    if crontab -l 2>/dev/null | grep -q "data_archiver.py"; then
        log_warn "Cron job already exists"
        crontab -l | grep "data_archiver.py"
        return
    fi

    # 添加新的 cron 任务
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

    log_info "Cron job added successfully!"
    log_info "Current cron jobs:"
    crontab -l | grep "data_archiver.py"
}

remove_cron() {
    log_info "Removing cron job..."

    if ! crontab -l 2>/dev/null | grep -q "data_archiver.py"; then
        log_warn "No cron job found"
        return
    fi

    crontab -l | grep -v "data_archiver.py" | crontab -

    log_info "Cron job removed successfully!"
}

show_help() {
    echo "AlgVex Data Archiver Service Manager"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install     Install and start systemd service (requires sudo)"
    echo "  uninstall   Stop and remove systemd service (requires sudo)"
    echo "  status      Show service status"
    echo "  logs        Follow service logs"
    echo "  cron        Setup hourly cron job (no sudo required)"
    echo "  cron-remove Remove cron job"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  sudo $0 install   # Install as system service"
    echo "  $0 cron           # Setup as cron job (user level)"
    echo "  $0 status         # Check if service is running"
}

# 主逻辑
case "${1:-help}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    cron)
        setup_cron
        ;;
    cron-remove)
        remove_cron
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
