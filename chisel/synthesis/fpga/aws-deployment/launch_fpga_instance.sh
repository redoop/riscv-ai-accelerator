#!/bin/bash
# 启动 AWS F2 FPGA 实例
# 注意: AWS F1 实例已于 2024 年退役

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         AWS F2 FPGA 实例启动向导                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}注意: AWS F1 实例已于 2024 年退役${NC}"
echo -e "${YELLOW}本项目现在使用 F2 实例进行 FPGA 开发${NC}"
echo ""

echo -e "${BLUE}F2 实例信息:${NC}"
echo "   • 设备: xcvu47p (Virtex UltraScale+ VU47P)"
echo "   • 逻辑单元: 9M LUTs"
echo "   • 用途: FPGA 开发和测试"
echo "   • 成本: ~\$1.00/小时 (Spot)"
echo "   • 实例: f2.6xlarge"
echo ""

read -p "启动 F2 Spot 实例？(y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}✓ 启动 F2 实例...${NC}"
    if [ -f "$SCRIPT_DIR/launch_f2_vivado.sh" ]; then
        bash "$SCRIPT_DIR/launch_f2_vivado.sh"
    else
        echo -e "${RED}❌ 未找到 launch_f2_vivado.sh${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${BLUE}已取消${NC}"
    exit 0
fi
