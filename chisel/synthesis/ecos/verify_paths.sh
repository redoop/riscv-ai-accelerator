#!/bin/bash
# ============================================================================
# ECOS 项目路径验证脚本
# ============================================================================
# 用途: 验证所有路径配置是否正确
# 使用: cd chisel/synthesis/ecos && ./verify_paths.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "ECOS 项目路径验证"
echo "=========================================="
echo ""
echo "工作目录: $SCRIPT_DIR"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

# 验证函数
check_path() {
    local desc="$1"
    local path="$2"
    local required="$3"  # "required" or "optional"
    
    if [ -e "$path" ]; then
        echo -e "${GREEN}✓${NC} $desc"
        echo "  路径: $path"
        if [ -f "$path" ]; then
            local size=$(du -h "$path" | cut -f1)
            echo "  大小: $size"
        fi
        ((PASS++))
        return 0
    else
        if [ "$required" = "required" ]; then
            echo -e "${RED}✗${NC} $desc"
            echo "  路径: $path (不存在)"
            ((FAIL++))
            return 1
        else
            echo -e "${YELLOW}⚠${NC} $desc"
            echo "  路径: $path (可选，不存在)"
            ((WARN++))
            return 0
        fi
    fi
}

echo "=========================================="
echo "1. 基础目录结构"
echo "=========================================="
echo ""

check_path "ECOS 根目录" "." "required"
check_path "filelist 目录" "filelist" "required"
check_path "project 目录" "project" "required"
check_path "project/verilog 目录" "project/verilog" "required"
check_path "project/netlist 目录" "project/netlist" "required"
check_path "run 目录" "run" "required"
check_path "tb 目录" "tb" "required"
check_path "utils 目录" "utils" "required"
check_path "rcu 目录" "rcu" "required"
check_path "lib 目录" "lib" "required"

echo ""
echo "=========================================="
echo "2. 关键文件"
echo "=========================================="
echo ""

check_path "综合脚本" "run_synthesis.sh" "required"
check_path "ASIC 顶层" "asic_top.sv" "required"
check_path "Makefile.iverilog" "run/Makefile.iverilog" "required"
check_path "run_sim.py" "run/run_sim.py" "required"

echo ""
echo "=========================================="
echo "3. Filelist 文件"
echo "=========================================="
echo ""

check_path "asic_top.f" "filelist/asic_top.f" "required"
check_path "ip.f" "filelist/ip.f" "required"
check_path "lib.f" "filelist/lib.f" "required"
check_path "soc.f" "filelist/soc.f" "optional"
check_path "asic_tblist.f" "filelist/asic_tblist.f" "required"

echo ""
echo "=========================================="
echo "4. Chisel RTL (生成文件)"
echo "=========================================="
echo ""

check_path "Chisel RTL 源" "../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv" "optional"
check_path "Chisel RTL 副本" "project/verilog/SimpleEdgeAiSoC.sv" "optional"

echo ""
echo "=========================================="
echo "5. 综合输出 (生成文件)"
echo "=========================================="
echo ""

check_path "综合网表" "project/netlist/asic_top_ics55.v" "optional"
check_path "PDK Verilog 副本" "project/netlist/ics55_LLSC_H7CL.v" "optional"
check_path "综合统计" "project/netlist/synthesis_stats.txt" "optional"
check_path "综合日志" "project/netlist/synthesis.log" "optional"

echo ""
echo "=========================================="
echo "6. PDK 文件"
echo "=========================================="
echo ""

check_path "PDK 根目录" "pdk/icsprout55-pdk" "optional"
if [ -d "pdk/icsprout55-pdk" ]; then
    check_path "PDK Liberty" "pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib" "optional"
    check_path "PDK Verilog" "pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v" "optional"
fi

echo ""
echo "=========================================="
echo "7. 从 run/ 目录验证路径"
echo "=========================================="
echo ""

cd run

echo "当前目录: $(pwd)"
echo ""

check_path "网表 (从 run/)" "../project/netlist/asic_top_ics55.v" "optional"
check_path "PDK (从 run/)" "../pdk/icsprout55-pdk" "optional"
check_path "Chisel RTL (从 run/)" "../project/verilog/SimpleEdgeAiSoC.sv" "optional"

cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "8. Makefile 配置验证"
echo "=========================================="
echo ""

cd run

# 提取 Makefile 中的路径配置
echo "Makefile.iverilog 配置:"
echo ""

if [ -f "Makefile.iverilog" ]; then
    PDK_ROOT=$(grep "^PDK_ROOT :=" Makefile.iverilog | awk '{print $3}')
    NETLIST_DIR=$(grep "^NETLIST_DIR :=" Makefile.iverilog | awk '{print $3}')
    NETLIST_FILE_BASE=$(grep "^NETLIST_FILE :=" Makefile.iverilog | sed 's/.*\///')
    CHISEL_RTL=$(grep "^CHISEL_RTL :=" Makefile.iverilog | awk '{print $3}')
    
    echo "  PDK_ROOT = $PDK_ROOT"
    if [ "$PDK_ROOT" = "../pdk/icsprout55-pdk" ]; then
        echo -e "    ${GREEN}✓ 正确${NC}"
        ((PASS++))
    else
        echo -e "    ${RED}✗ 错误，应该是 ../pdk/icsprout55-pdk${NC}"
        ((FAIL++))
    fi
    
    echo ""
    echo "  NETLIST_DIR = $NETLIST_DIR"
    if [ "$NETLIST_DIR" = "../project/netlist" ]; then
        echo -e "    ${GREEN}✓ 正确${NC}"
        ((PASS++))
    else
        echo -e "    ${RED}✗ 错误，应该是 ../project/netlist${NC}"
        ((FAIL++))
    fi
    
    echo ""
    echo "  NETLIST_FILE = \$(NETLIST_DIR)/$NETLIST_FILE_BASE"
    if [[ "$NETLIST_FILE_BASE" == *"asic_top_ics55.v"* ]]; then
        echo -e "    ${GREEN}✓ 正确${NC}"
        ((PASS++))
    else
        echo -e "    ${RED}✗ 错误，应该是 asic_top_ics55.v${NC}"
        ((FAIL++))
    fi
    
    echo ""
    echo "  CHISEL_RTL = $CHISEL_RTL"
    if [ "$CHISEL_RTL" = "../project/verilog/SimpleEdgeAiSoC.sv" ]; then
        echo -e "    ${GREEN}✓ 正确${NC}"
        ((PASS++))
    else
        echo -e "    ${RED}✗ 错误，应该是 ../project/verilog/SimpleEdgeAiSoC.sv${NC}"
        ((FAIL++))
    fi
fi

cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "验证总结"
echo "=========================================="
echo ""
echo -e "${GREEN}通过: $PASS${NC}"
echo -e "${YELLOW}警告: $WARN${NC}"
echo -e "${RED}失败: $FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ 所有必需的路径配置正确！${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 如果 Chisel RTL 不存在，运行: ./run_synthesis.sh"
    echo "  2. 如果 PDK 不存在，参考: pdk/README.md"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 发现 $FAIL 个路径配置错误！${NC}"
    echo ""
    echo "请检查:"
    echo "  1. Makefile.iverilog 中的路径配置"
    echo "  2. run_synthesis.sh 中的路径配置"
    echo "  3. 参考: PATH_VERIFICATION.md"
    echo ""
    exit 1
fi
