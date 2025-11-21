#!/bin/bash
# 运行完整的时钟验证测试

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CHISEL_DIR="$PROJECT_ROOT/chisel"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           时钟验证测试套件                                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查依赖
echo -e "${BLUE}[1/4] 检查依赖...${NC}"
if ! command -v sbt &> /dev/null; then
    echo -e "${RED}❌ sbt 未安装${NC}"
    echo "安装方法:"
    echo "  macOS: brew install sbt"
    echo "  Linux: sudo apt install sbt"
    exit 1
fi
echo -e "${GREEN}✓ sbt 已安装${NC}"
echo ""

# 进入 Chisel 目录
cd "$CHISEL_DIR"

# 编译项目
echo -e "${BLUE}[2/4] 编译项目...${NC}"
sbt compile > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 编译成功${NC}"
else
    echo -e "${RED}❌ 编译失败${NC}"
    exit 1
fi
echo ""

# 运行时钟验证测试
echo -e "${BLUE}[3/4] 运行时钟验证测试...${NC}"
echo ""

# 创建测试结果目录
mkdir -p test_results

# 运行测试并保存输出
sbt "testOnly riscv.ai.ClockVerificationTest" 2>&1 | tee test_results/clock_verification.log

TEST_RESULT=${PIPESTATUS[0]}

echo ""

# 检查测试结果
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ 所有测试通过${NC}"
else
    echo -e "${RED}❌ 部分测试失败${NC}"
    echo ""
    echo "查看详细日志:"
    echo "  cat $CHISEL_DIR/test_results/clock_verification.log"
fi

echo ""

# 检查生成的 VCD 文件
echo -e "${BLUE}[4/4] 检查生成的波形文件...${NC}"

VCD_FILES=$(find test_run_dir -name "*.vcd" 2>/dev/null | head -5)

if [ -n "$VCD_FILES" ]; then
    echo -e "${GREEN}✓ 找到波形文件:${NC}"
    echo "$VCD_FILES" | while read file; do
        echo "  - $file"
    done
    echo ""
    echo "使用 GTKWave 查看波形:"
    echo "  gtkwave <vcd_file>"
else
    echo -e "${YELLOW}⚠ 未找到 VCD 文件${NC}"
fi

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           测试完成                                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 生成测试报告摘要
echo "测试报告摘要:"
echo "  日志文件: $CHISEL_DIR/test_results/clock_verification.log"
echo "  波形文件: $CHISEL_DIR/test_run_dir/"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ 验证通过，可以继续后端工作${NC}"
    exit 0
else
    echo -e "${RED}✗ 验证失败，请检查时钟配置${NC}"
    exit 1
fi
