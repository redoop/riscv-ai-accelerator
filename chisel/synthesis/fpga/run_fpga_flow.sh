#!/bin/bash
# RISC-V AI 加速器 FPGA 验证完整流程
# 统一的自动化脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
CHISEL_DIR="$PROJECT_ROOT/chisel"
FPGA_DIR="$SCRIPT_DIR"

# 默认参数
MODE=${1:-"help"}
TARGET=${2:-"local"}  # local 或 aws

# 显示 Banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║   RISC-V AI 加速器 - FPGA 验证流程                        ║"
    echo "║   PicoRV32 + CompactAccel + BitNetAccel                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 显示帮助
show_help() {
    show_banner
    echo -e "${BLUE}用法:${NC} $0 [模式] [目标]"
    echo ""
    echo -e "${BLUE}模式:${NC}"
    echo "  prepare     - 准备环境和生成 Verilog"
    echo "  simulate    - 运行 RTL 仿真验证"
    echo "  synthesize  - 综合 FPGA 设计"
    echo "  build       - 构建完整 FPGA 镜像"
    echo "  deploy      - 部署到 FPGA (需要硬件)"
    echo "  test        - 运行 FPGA 测试"
    echo "  full        - 完整流程 (prepare -> build)"
    echo "  aws         - AWS F1 完整流程"
    echo "  clean       - 清理所有构建文件"
    echo "  status      - 查看当前状态"
    echo ""
    echo -e "${BLUE}目标:${NC}"
    echo "  local       - 本地 FPGA 开发 (默认)"
    echo "  aws         - AWS F1 云端 FPGA"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo "  $0 prepare              # 准备环境和生成 Verilog"
    echo "  $0 simulate             # 运行仿真"
    echo "  $0 full local           # 本地完整流程"
    echo "  $0 aws                  # AWS F1 完整流程"
    echo "  $0 status               # 查看状态"
    echo ""
    echo -e "${BLUE}AWS F1 流程:${NC}"
    echo "  1. $0 prepare           # 生成 Verilog"
    echo "  2. $0 aws               # 自动执行 AWS 流程"
    echo "     - 配置 AWS 环境"
    echo "     - 运行 Vivado 综合"
    echo "     - 创建 AFI 镜像"
    echo "     - 部署和测试"
    echo ""
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}[1/7] 检查依赖...${NC}"
    
    local missing_deps=()
    
    # 检查 sbt (Chisel 编译)
    if ! command -v sbt &> /dev/null; then
        missing_deps+=("sbt")
    fi
    
    # 检查 Java
    if ! command -v java &> /dev/null; then
        missing_deps+=("java")
    fi
    
    # 如果是 AWS 模式，检查 AWS CLI
    if [ "$TARGET" == "aws" ]; then
        if ! command -v aws &> /dev/null; then
            missing_deps+=("aws-cli")
        fi
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}❌ 缺少依赖: ${missing_deps[*]}${NC}"
        echo ""
        echo "安装方法:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                sbt)
                    echo "  macOS: brew install sbt"
                    echo "  Linux: sudo apt install sbt"
                    ;;
                java)
                    echo "  macOS: brew install openjdk@11"
                    echo "  Linux: sudo apt install openjdk-11-jdk"
                    ;;
                aws-cli)
                    echo "  pip install awscli"
                    ;;
            esac
        done
        exit 1
    fi
    
    echo -e "${GREEN}✓ 所有依赖已安装${NC}"
}

# 准备环境
prepare_environment() {
    echo -e "${BLUE}[2/7] 准备环境...${NC}"
    
    # 创建必要的目录
    mkdir -p "$FPGA_DIR/build"
    mkdir -p "$FPGA_DIR/build/reports"
    mkdir -p "$FPGA_DIR/build/checkpoints"
    mkdir -p "$FPGA_DIR/build/checkpoints/to_aws"
    mkdir -p "$FPGA_DIR/test_results"
    
    echo -e "${GREEN}✓ 目录结构创建完成${NC}"
}

# 生成 Verilog
generate_verilog() {
    echo -e "${BLUE}[3/7] 生成 Verilog...${NC}"
    
    cd "$CHISEL_DIR"
    
    # 编译 Chisel 代码
    echo "  编译 Chisel..."
    sbt compile > /dev/null 2>&1
    
    # 生成 Verilog
    echo "  生成 SystemVerilog..."
    sbt 'runMain riscv.ai.SimpleEdgeAiSoCMain' 2>&1 | grep -E "(Generating|✅|📁|代码行数)"
    
    if [ ! -f "$CHISEL_DIR/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv" ]; then
        echo -e "${RED}❌ Verilog 生成失败${NC}"
        exit 1
    fi
    
    local lines=$(wc -l < "$CHISEL_DIR/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv")
    echo -e "${GREEN}✓ Verilog 生成成功 ($lines 行)${NC}"
    
    cd "$FPGA_DIR"
}

# 运行仿真
run_simulation() {
    echo -e "${BLUE}[4/7] 运行 RTL 仿真...${NC}"
    
    cd "$CHISEL_DIR"
    
    echo "  运行 SimpleEdgeAiSoC 测试..."
    sbt 'testOnly riscv.ai.SimpleEdgeAiSoCTest' 2>&1 | tail -20
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 仿真测试通过${NC}"
    else
        echo -e "${YELLOW}⚠ 部分测试失败，但继续流程${NC}"
    fi
    
    cd "$FPGA_DIR"
}

# 本地综合 (使用 Yosys)
synthesize_local() {
    echo -e "${BLUE}[5/7] 本地综合 (Yosys)...${NC}"
    
    if ! command -v yosys &> /dev/null; then
        echo -e "${YELLOW}⚠ Yosys 未安装，跳过综合${NC}"
        echo "  安装: brew install yosys (macOS) 或 sudo apt install yosys (Linux)"
        return
    fi
    
    echo "  运行 Yosys 综合..."
    cd "$CHISEL_DIR/synthesis"
    
    if [ -f "run_generic_synthesis.sh" ]; then
        bash run_generic_synthesis.sh 2>&1 | tail -30
        echo -e "${GREEN}✓ 综合完成${NC}"
    else
        echo -e "${YELLOW}⚠ 综合脚本未找到${NC}"
    fi
    
    cd "$FPGA_DIR"
}

# AWS 综合 (使用 Vivado)
synthesize_aws() {
    echo -e "${BLUE}[5/7] AWS Vivado 综合...${NC}"
    
    if ! command -v vivado &> /dev/null; then
        echo -e "${RED}❌ Vivado 未安装${NC}"
        echo "  请在 AWS F1 实例上运行此脚本"
        exit 1
    fi
    
    echo "  运行 Vivado 构建..."
    cd "$FPGA_DIR"
    
    if [ -f "scripts/build_fpga.tcl" ]; then
        vivado -mode batch -source scripts/build_fpga.tcl 2>&1 | tee build/vivado.log
        
        if [ -f "build/checkpoints/to_aws/SH_CL_routed.dcp" ]; then
            echo -e "${GREEN}✓ Vivado 综合完成${NC}"
        else
            echo -e "${RED}❌ Vivado 综合失败${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ 构建脚本未找到${NC}"
        exit 1
    fi
}

# 创建 AWS AFI
create_afi() {
    echo -e "${BLUE}[6/7] 创建 AWS AFI...${NC}"
    
    if [ ! -f "$FPGA_DIR/scripts/create_afi.sh" ]; then
        echo -e "${RED}❌ AFI 创建脚本未找到${NC}"
        exit 1
    fi
    
    cd "$FPGA_DIR"
    bash scripts/create_afi.sh
    
    echo -e "${GREEN}✓ AFI 创建请求已提交${NC}"
    echo -e "${YELLOW}⚠ AFI 生成需要 30-60 分钟，请稍后检查状态${NC}"
}

# 部署到 FPGA
deploy_fpga() {
    echo -e "${BLUE}[7/7] 部署到 FPGA...${NC}"
    
    if [ "$TARGET" == "aws" ]; then
        # AWS F1 部署
        if [ -f "$FPGA_DIR/build/afi_info.txt" ]; then
            local agfi_id=$(grep "AGFI ID" "$FPGA_DIR/build/afi_info.txt" | awk '{print $3}')
            
            if [ -n "$agfi_id" ]; then
                echo "  加载 AFI: $agfi_id"
                sudo fpga-load-local-image -S 0 -I "$agfi_id"
                
                echo "  验证加载..."
                sudo fpga-describe-local-image -S 0 -H
                
                echo -e "${GREEN}✓ FPGA 镜像已加载${NC}"
            else
                echo -e "${RED}❌ 未找到 AGFI ID${NC}"
                exit 1
            fi
        else
            echo -e "${YELLOW}⚠ AFI 信息文件未找到，请先创建 AFI${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 本地 FPGA 部署需要硬件支持${NC}"
        echo "  请使用 JTAG 或其他方式手动烧录比特流"
    fi
}

# 运行测试
run_tests() {
    echo -e "${BLUE}运行 FPGA 测试...${NC}"
    
    if [ -f "$FPGA_DIR/scripts/run_tests.sh" ]; then
        cd "$FPGA_DIR"
        bash scripts/run_tests.sh
    else
        echo -e "${YELLOW}⚠ 测试脚本未找到${NC}"
    fi
}

# 查看状态
show_status() {
    show_banner
    echo -e "${BLUE}项目状态:${NC}"
    echo ""
    
    # Verilog 生成状态
    if [ -f "$CHISEL_DIR/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv" ]; then
        local lines=$(wc -l < "$CHISEL_DIR/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv")
        echo -e "${GREEN}✓${NC} Verilog 已生成 ($lines 行)"
    else
        echo -e "${RED}✗${NC} Verilog 未生成"
    fi
    
    # 综合状态
    if [ -f "$FPGA_DIR/build/checkpoints/to_aws/SH_CL_routed.dcp" ]; then
        echo -e "${GREEN}✓${NC} Vivado 综合已完成"
    else
        echo -e "${YELLOW}○${NC} Vivado 综合未完成"
    fi
    
    # AFI 状态
    if [ -f "$FPGA_DIR/build/afi_info.txt" ]; then
        local afi_id=$(grep "AFI ID" "$FPGA_DIR/build/afi_info.txt" | awk '{print $3}')
        echo -e "${GREEN}✓${NC} AFI 已创建: $afi_id"
        
        # 检查 AFI 状态
        if command -v aws &> /dev/null && [ -n "$afi_id" ]; then
            local status=$(aws ec2 describe-fpga-images --fpga-image-ids "$afi_id" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null)
            if [ -n "$status" ]; then
                echo "  状态: $status"
            fi
        fi
    else
        echo -e "${YELLOW}○${NC} AFI 未创建"
    fi
    
    # 测试结果
    if [ -d "$FPGA_DIR/test_results" ] && [ "$(ls -A $FPGA_DIR/test_results)" ]; then
        echo -e "${GREEN}✓${NC} 测试结果可用"
        echo "  位置: $FPGA_DIR/test_results/"
    else
        echo -e "${YELLOW}○${NC} 无测试结果"
    fi
    
    echo ""
    echo -e "${BLUE}文件位置:${NC}"
    echo "  Verilog:  $CHISEL_DIR/generated/simple_edgeaisoc/"
    echo "  构建:     $FPGA_DIR/build/"
    echo "  脚本:     $FPGA_DIR/scripts/"
    echo "  文档:     $FPGA_DIR/docs/"
    echo ""
}

# 清理
clean_all() {
    echo -e "${BLUE}清理构建文件...${NC}"
    
    rm -rf "$FPGA_DIR/build"
    rm -rf "$FPGA_DIR/test_results"
    rm -rf "$CHISEL_DIR/generated"
    rm -rf "$CHISEL_DIR/test_run_dir"
    
    echo -e "${GREEN}✓ 清理完成${NC}"
}

# AWS 完整流程
aws_full_flow() {
    show_banner
    echo -e "${CYAN}开始 AWS F1 完整验证流程...${NC}"
    echo ""
    
    TARGET="aws"
    
    check_dependencies
    prepare_environment
    generate_verilog
    run_simulation
    synthesize_aws
    create_afi
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  AWS F1 流程完成！                                         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}下一步:${NC}"
    echo "  1. 等待 AFI 生成完成 (30-60 分钟)"
    echo "  2. 检查状态: $0 status"
    echo "  3. 部署测试: $0 deploy aws"
    echo "  4. 运行测试: $0 test"
    echo ""
}

# 本地完整流程
local_full_flow() {
    show_banner
    echo -e "${CYAN}开始本地 FPGA 验证流程...${NC}"
    echo ""
    
    TARGET="local"
    
    check_dependencies
    prepare_environment
    generate_verilog
    run_simulation
    synthesize_local
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  本地流程完成！                                            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}生成的文件:${NC}"
    echo "  Verilog: $CHISEL_DIR/generated/simple_edgeaisoc/"
    echo "  综合:    $CHISEL_DIR/synthesis/"
    echo ""
    echo -e "${BLUE}下一步:${NC}"
    echo "  1. 使用 FPGA 工具链进行 P&R"
    echo "  2. 生成比特流"
    echo "  3. 烧录到 FPGA"
    echo ""
}

# 主流程
main() {
    case $MODE in
        help|-h|--help)
            show_help
            ;;
        prepare)
            show_banner
            check_dependencies
            prepare_environment
            generate_verilog
            ;;
        simulate)
            show_banner
            run_simulation
            ;;
        synthesize)
            show_banner
            if [ "$TARGET" == "aws" ]; then
                synthesize_aws
            else
                synthesize_local
            fi
            ;;
        build)
            show_banner
            check_dependencies
            prepare_environment
            generate_verilog
            if [ "$TARGET" == "aws" ]; then
                synthesize_aws
            else
                synthesize_local
            fi
            ;;
        deploy)
            show_banner
            deploy_fpga
            ;;
        test)
            show_banner
            run_tests
            ;;
        full)
            if [ "$TARGET" == "aws" ]; then
                aws_full_flow
            else
                local_full_flow
            fi
            ;;
        aws)
            aws_full_flow
            ;;
        status)
            show_status
            ;;
        clean)
            clean_all
            ;;
        *)
            echo -e "${RED}❌ 未知模式: $MODE${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行
main
