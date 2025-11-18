#!/bin/bash
# 智能选择并启动 FPGA 实例（F1 或 F2）

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
echo -e "${CYAN}║         AWS FPGA 实例启动向导                             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}选择实例类型:${NC}"
echo ""
echo -e "${GREEN}1. F1 Spot 实例 (推荐 - 最便宜)${NC}"
echo "   • 设备: xcvu9p (Virtex UltraScale+ VU9P)"
echo "   • 用途: AFI 创建和测试"
echo "   • 兼容性: ✓ 完全支持 AWS AFI 服务"
echo "   • 成本: ~\$0.50-0.60/小时"
echo "   • 实例: f1.2xlarge (Spot)"
echo "   • 注意: 可能因容量不足而失败"
echo ""
echo -e "${GREEN}2. F1 按需实例 (推荐 - 最可靠)${NC}"
echo "   • 设备: xcvu9p (Virtex UltraScale+ VU9P)"
echo "   • 用途: AFI 创建和测试"
echo "   • 兼容性: ✓ 完全支持 AWS AFI 服务"
echo "   • 成本: \$1.65/小时"
echo "   • 实例: f1.2xlarge (On-Demand)"
echo "   • 优势: 保证可用，不会被中断"
echo ""
echo -e "${YELLOW}3. F2 实例 (实验性 - 不推荐)${NC}"
echo "   • 设备: xcvu47p (Virtex UltraScale+ VU47P)"
echo "   • 用途: 仅用于开发和测试"
echo "   • 兼容性: ✗ 不支持 AWS AFI 服务"
echo "   • 成本: ~\$2.30/小时 (Spot)"
echo "   • 实例: f2.6xlarge"
echo "   • 注意: 生成的 DCP 无法用于 AFI 创建"
echo ""
echo -e "${BLUE}4. 查看详细对比${NC}"
echo ""

read -p "请选择 [1/2/3/4]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}✓ 选择 F1 Spot 实例${NC}"
        echo ""
        if [ -f "$SCRIPT_DIR/launch_f1_vivado.sh" ]; then
            bash "$SCRIPT_DIR/launch_f1_vivado.sh"
        else
            echo -e "${RED}❌ 未找到 launch_f1_vivado.sh${NC}"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo -e "${GREEN}✓ 选择 F1 按需实例${NC}"
        echo ""
        echo "按需实例特点:"
        echo "  ✓ 保证可用，不会因容量不足而失败"
        echo "  ✓ 不会被中断"
        echo "  ✓ 适合长时间构建任务"
        echo "  ✗ 成本是 Spot 的 3 倍（\$1.65/小时 vs \$0.50/小时）"
        echo ""
        read -p "确认启动按需实例？(y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            if [ -f "$SCRIPT_DIR/launch_f1_ondemand.sh" ]; then
                bash "$SCRIPT_DIR/launch_f1_ondemand.sh"
            else
                echo -e "${RED}❌ 未找到 launch_f1_ondemand.sh${NC}"
                exit 1
            fi
        else
            echo ""
            echo -e "${BLUE}已取消${NC}"
            exit 0
        fi
        ;;
    3)
        echo ""
        echo -e "${YELLOW}⚠️  警告: F2 实例不支持 AFI 创建${NC}"
        echo ""
        echo "F2 实例生成的 DCP 文件使用 xcvu47p 设备，"
        echo "无法被 AWS AFI 服务接受（需要 xcvu9p）。"
        echo ""
        echo "F2 实例仅适用于:"
        echo "  • 本地开发和调试"
        echo "  • 设计验证"
        echo "  • 不需要创建 AFI 的场景"
        echo ""
        read -p "确定要继续使用 F2 吗？(y/N): " confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${YELLOW}启动 F2 实例...${NC}"
            if [ -f "$SCRIPT_DIR/launch_f2_vivado.sh" ]; then
                bash "$SCRIPT_DIR/launch_f2_vivado.sh"
            else
                echo -e "${RED}❌ 未找到 launch_f2_vivado.sh${NC}"
                exit 1
            fi
        else
            echo ""
            echo -e "${BLUE}已取消，请选择 F1 实例${NC}"
            exit 0
        fi
        ;;
    4)
        echo ""
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}F1 vs F2 详细对比${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        
        echo "┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐"
        echo "│ 特性            │ F1 Spot (推荐)   │ F1 按需 (可靠)   │ F2 (实验性)      │"
        echo "├─────────────────┼──────────────────┼──────────────────┼──────────────────┤"
        echo "│ FPGA 设备       │ xcvu9p           │ xcvu9p           │ xcvu47p          │"
        echo "│ 逻辑单元        │ 2.5M             │ 2.5M             │ 9M               │"
        echo "│ AFI 支持        │ ✓ 完全支持       │ ✓ 完全支持       │ ✗ 不支持         │"
        echo "│ 价格/小时       │ ~\$0.50          │ \$1.65           │ ~\$2.30          │"
        echo "│ 可用性          │ 可能不足         │ 保证可用         │ 可能不足         │"
        echo "│ 可中断性        │ 可能被中断       │ 不会中断         │ 可能被中断       │"
        echo "│ 推荐用途        │ 成本敏感         │ 可靠性优先       │ 仅开发调试       │"
        echo "│ 构建时间        │ 2-4 小时         │ 2-4 小时         │ 2-4 小时         │"
        echo "└─────────────────┴──────────────────┴──────────────────┴──────────────────┘"
        echo ""
        
        echo -e "${BLUE}设备兼容性:${NC}"
        echo ""
        echo "F1 (xcvu9p):"
        echo "  ✓ AWS Shell 使用 xcvu9p"
        echo "  ✓ AFI 服务接受 xcvu9p DCP"
        echo "  ✓ 可以创建和加载 AFI"
        echo ""
        echo "F2 (xcvu47p):"
        echo "  ✗ AWS Shell 使用 xcvu9p"
        echo "  ✗ AFI 服务拒绝 xcvu47p DCP"
        echo "  ✗ 无法创建 AFI"
        echo "  ✓ 可用于本地开发"
        echo ""
        
        echo -e "${BLUE}成本对比（4小时构建）:${NC}"
        echo ""
        echo "F1 Spot:      4 × \$0.50 = \$2.00   ⭐ 最便宜"
        echo "F1 按需:      4 × \$1.65 = \$6.60   ⭐ 最可靠"
        echo "F2 Spot:      4 × \$2.30 = \$9.20   ❌ 无法创建 AFI"
        echo "F2 按需:      4 × \$7.65 = \$30.60  ❌ 无法创建 AFI"
        echo ""
        
        echo -e "${BLUE}选择建议:${NC}"
        echo ""
        echo -e "${GREEN}F1 Spot 实例 - 适合:${NC}"
        echo "  ✓ 成本敏感的项目"
        echo "  ✓ 可以接受偶尔失败重试"
        echo "  ✓ 非紧急任务"
        echo ""
        echo -e "${GREEN}F1 按需实例 - 适合:${NC}"
        echo "  ✓ 需要保证成功"
        echo "  ✓ 紧急任务"
        echo "  ✓ 长时间构建（避免中断）"
        echo "  ✓ 生产环境"
        echo ""
        echo -e "${YELLOW}F2 实例 - 仅适合:${NC}"
        echo "  ✓ 需要 9M LUTs 的大型设计"
        echo "  ✓ 本地开发和验证"
        echo "  ✗ 无法部署到 AWS FPGA"
        echo ""
        
        read -p "按回车键返回选择菜单..." dummy
        exec "$0"
        ;;
    *)
        echo ""
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac
