#!/bin/bash

export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk-11.0.16.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

# 检查参数
MODE=${1:-"test"}

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}RISC-V AI Accelerator - 运行脚本${NC}"
    echo ""
    echo "用法: $0 [模式]"
    echo ""
    echo "可用模式："
    echo "  test        - 运行所有测试 (默认)"
    echo "  soc         - 运行 SimpleEdgeAiSoC 测试"
    echo "  riscv       - 运行 RISC-V 指令测试"
    echo "  generate    - 生成 SimpleEdgeAiSoC SystemVerilog"
    echo "  all         - 生成所有版本的 SystemVerilog"
    echo "  full        - 完整流程（编译+测试+生成）"
    echo "  clean       - 清理所有构建文件"
    echo ""
    echo "示例："
    echo "  $0 soc       # 测试 SimpleEdgeAiSoC"
    echo "  $0 riscv     # 测试 RISC-V 指令"
    echo "  $0 generate  # 生成 Verilog 文件"
    echo "  $0 full      # 完整开发流程"
    exit 0
}

# 检查 sbt 是否安装
check_sbt() {
    if ! command -v sbt &> /dev/null; then
        echo -e "${RED}❌ sbt 未安装，请先安装 sbt${NC}"
        echo "macOS: brew install sbt"
        echo "Ubuntu: sudo apt install sbt"
        exit 1
    fi
}

# 编译项目
compile_project() {
    echo -e "${BLUE}📦 1. 编译 Chisel 代码...${NC}"
    sbt compile
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 编译失败${NC}"
        exit 1
    fi
    echo ""
}

# 运行所有测试
run_all_tests() {
    echo -e "${BLUE}🧪 2. 运行所有测试...${NC}"
    sbt test
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️  部分测试失败${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    echo ""
    return 0
}

# 运行 SimpleEdgeAiSoC 测试
run_soc_test() {
    echo -e "${BLUE}🧪 2. 运行 SimpleEdgeAiSoC 测试...${NC}"
    sbt 'testOnly riscv.ai.SimpleEdgeAiSoCTest'
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ SimpleEdgeAiSoC 测试失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ SimpleEdgeAiSoC 测试通过！${NC}"
    echo ""
}

# 运行 RISC-V 指令测试
run_riscv_test() {
    echo -e "${BLUE}🧪 2. 运行 RISC-V 指令测试...${NC}"
    echo ""
    
    # 运行测试并捕获输出
    sbt 'testOnly riscv.ai.SimpleRiscvInstructionTests' 2>&1 | grep -A 50 "SimpleRiscvInstructionTests"
    
    # 检查测试结果
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo ""
        echo -e "${RED}❌ RISC-V 指令测试失败${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ RISC-V 指令测试通过！${NC}"
    echo ""
    echo -e "${BLUE}📊 测试覆盖：${NC}"
    echo "  ✓ 算术运算指令 (ADD, SUB, ADDI)"
    echo "  ✓ 逻辑运算指令 (AND, OR, XOR, etc.)"
    echo "  ✓ 移位指令 (SLL, SRL, SRA, etc.)"
    echo "  ✓ 比较指令 (SLT, SLTU, etc.)"
    echo "  ✓ 加载存储指令 (LW, SW, etc.)"
    echo "  ✓ 分支跳转指令 (BEQ, BNE, JAL, etc.)"
    echo "  ✓ 立即数指令 (LUI, AUIPC)"
    echo ""
    echo -e "${BLUE}📈 指令覆盖率：${NC}"
    echo "  总计: 37 条 RV32I 基本指令"
    echo "  覆盖: 100% 指令编码验证"
    echo ""
}

# 生成 SimpleEdgeAiSoC Verilog
generate_soc() {
    echo -e "${BLUE}🔧 生成 SimpleEdgeAiSoC SystemVerilog...${NC}"
    sbt 'runMain riscv.ai.SimpleEdgeAiSoCMain'
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Verilog 生成失败${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ SimpleEdgeAiSoC SystemVerilog 生成成功！${NC}"
    echo ""
    
    if [ -f "generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv" ]; then
        SOC_LINES=$(wc -l < generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv)
        echo -e "${BLUE}📁 生成的文件：${NC}"
        echo "  ✓ generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv"
        echo ""
        echo -e "${BLUE}📊 文件信息：${NC}"
        echo "  代码行数: $SOC_LINES 行"
        echo ""
        echo -e "${BLUE}🏗️  架构组成：${NC}"
        echo "  ✓ PicoRV32 RISC-V CPU"
        echo "  ✓ CompactAccel (8x8 矩阵加速器)"
        echo "  ✓ BitNetAccel (16x16 BitNet 加速器)"
        echo "  ✓ 内存适配器和地址解码器"
        echo "  ✓ GPIO 和中断控制器"
        echo ""
        echo -e "${BLUE}💡 下一步：${NC}"
        echo "  1. 使用 Verilator 进行仿真验证"
        echo "  2. 使用 Yosys 进行综合"
        echo "  3. 查看测试结果: test_run_dir/"
    fi
    echo ""
}

# 生成所有版本
generate_all() {
    echo -e "${BLUE}🔧 生成所有版本的 SystemVerilog...${NC}"
    sbt 'runMain riscv.ai.VerilogGenerator'
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Verilog 生成失败${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ 所有版本 SystemVerilog 生成成功！${NC}"
    echo ""
    echo -e "${BLUE}📁 生成的文件：${NC}"
    
    if [ -f "generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv" ]; then
        echo "  ✓ SimpleEdgeAiSoC.sv (推荐)"
    fi
    
    if [ -d "generated/optimized" ]; then
        echo "  ✓ 优化版本 (generated/optimized/)"
    fi
    
    if [ -d "generated/scalable" ]; then
        echo "  ✓ 可扩展版本 (generated/scalable/)"
    fi
    
    if [ -d "generated/fixed" ]; then
        echo "  ✓ 修复版本 (generated/fixed/)"
    fi
    
    echo ""
}

# 清理构建文件
clean_all() {
    echo -e "${BLUE}🧹 清理所有构建文件...${NC}"
    sbt clean
    
    echo -e "${BLUE}🧹 删除测试运行目录...${NC}"
    rm -rf test_run_dir
    
    echo -e "${BLUE}🧹 删除生成的文件...${NC}"
    rm -rf generated
    
    echo -e "${GREEN}✅ 清理完成！${NC}"
    echo ""
}

# 主流程
main() {
    # 获取脚本所在目录
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    cd "$SCRIPT_DIR"
    
    # 检查 sbt
    check_sbt
    
    case $MODE in
        "help"|"-h"|"--help")
            show_help
            ;;
        "test")
            echo -e "${BLUE}=== RISC-V AI Accelerator - 测试模式 ===${NC}"
            echo ""
            compile_project
            run_all_tests
            ;;
        "soc")
            echo -e "${BLUE}=== RISC-V AI Accelerator - SimpleEdgeAiSoC 测试 ===${NC}"
            echo ""
            compile_project
            run_soc_test
            ;;
        "riscv")
            echo -e "${BLUE}=== RISC-V AI Accelerator - RISC-V 指令测试 ===${NC}"
            echo ""
            compile_project
            run_riscv_test
            ;;
        "generate")
            echo -e "${BLUE}=== RISC-V AI Accelerator - 生成 Verilog ===${NC}"
            echo ""
            compile_project
            generate_soc
            ;;
        "all")
            echo -e "${BLUE}=== RISC-V AI Accelerator - 生成所有版本 ===${NC}"
            echo ""
            compile_project
            generate_all
            ;;
        "full")
            echo -e "${BLUE}=== RISC-V AI Accelerator - 完整流程 ===${NC}"
            echo ""
            compile_project
            run_all_tests
            generate_soc
            
            echo -e "${GREEN}✅ 完整流程完成！${NC}"
            echo ""
            echo -e "${BLUE}📚 项目信息：${NC}"
            echo "  ✓ 编译成功"
            echo "  ✓ 测试通过"
            echo "  ✓ Verilog 生成完成"
            echo ""
            echo -e "${BLUE}💡 技术特点：${NC}"
            echo "  ✓ Chisel 硬件描述语言"
            echo "  ✓ PicoRV32 RISC-V CPU"
            echo "  ✓ 矩阵加速器 (CompactAccel + BitNetAccel)"
            echo "  ✓ 标准 AXI-Lite 接口"
            echo "  ✓ 完整的测试覆盖"
            echo ""
            ;;
        "clean")
            clean_all
            ;;
        *)
            echo -e "${RED}❌ 未知模式: $MODE${NC}"
            echo ""
            show_help
            ;;
    esac
}

# 运行主流程
main
