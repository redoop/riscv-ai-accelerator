#!/bin/bash
# test.sh - 运行 Chisel 测试的便捷脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RISC-V AI SoC 测试套件 ===${NC}"
echo ""

# 解析参数
TEST_TARGET="${1:-all}"

case "$TEST_TARGET" in
  all)
    echo -e "${GREEN}运行所有测试...${NC}"
    sbt test
    ;;
    
  uart)
    echo -e "${GREEN}运行 UART 测试...${NC}"
    sbt "testOnly riscv.ai.peripherals.RealUARTTest"
    ;;
    
  lcd)
    echo -e "${GREEN}运行 TFT LCD 测试...${NC}"
    sbt "testOnly riscv.ai.peripherals.TFTLCDTest"
    ;;
    
  ai|accel|accelerator)
    echo -e "${GREEN}运行 AI 加速器测试...${NC}"
    sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
    ;;
    
  soc)
    echo -e "${GREEN}运行完整 SoC 测试...${NC}"
    sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
    ;;
    
  cpu|picorv32)
    echo -e "${GREEN}运行 PicoRV32 CPU 测试...${NC}"
    sbt "testOnly riscv.ai.PicoRV32CoreTest"
    ;;
    
  peripherals)
    echo -e "${GREEN}运行所有外设测试...${NC}"
    sbt "testOnly riscv.ai.peripherals.*"
    ;;
    
  quick)
    echo -e "${GREEN}运行快速测试（跳过长时间测试）...${NC}"
    sbt "testOnly riscv.ai.peripherals.RealUARTTest riscv.ai.peripherals.TFTLCDTest"
    ;;
    
  list)
    echo -e "${YELLOW}可用的测试目标:${NC}"
    echo "  all          - 运行所有测试（默认）"
    echo "  uart         - UART 控制器测试"
    echo "  lcd          - TFT LCD 控制器测试"
    echo "  ai           - AI 加速器测试"
    echo "  soc          - 完整 SoC 测试"
    echo "  cpu          - PicoRV32 CPU 测试"
    echo "  peripherals  - 所有外设测试"
    echo "  quick        - 快速测试"
    echo "  list         - 显示此帮助信息"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./test.sh all        # 运行所有测试"
    echo "  ./test.sh uart       # 只运行 UART 测试"
    echo "  ./test.sh quick      # 快速测试"
    ;;
    
  *)
    echo -e "${YELLOW}未知的测试目标: $TEST_TARGET${NC}"
    echo "使用 './test.sh list' 查看可用选项"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}✓ 测试完成${NC}"
