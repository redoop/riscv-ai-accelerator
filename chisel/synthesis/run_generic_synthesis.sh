#!/bin/bash
# 使用 Yosys 进行通用逻辑综合（不依赖特定工艺库）
# 生成可以用 Icarus Verilog 仿真的网表

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置路径
# 优先使用系统中的 yosys，如果不存在则使用指定路径
if command -v yosys &> /dev/null; then
    YOSYS_BIN="yosys"
else
    YOSYS_BIN="/opt/tools/oss-cad/oss-cad-suite/bin/yosys"
fi

RTL_DIR="../generated"
SDC_FILE="fpga/constraints/timing_complete.sdc"
OUTPUT_DIR="netlist"
NETLIST_FILE="$OUTPUT_DIR/SimpleEdgeAiSoC_generic.v"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "通用逻辑综合（无工艺库，带 SDC 约束）"
echo "=========================================="
echo "RTL 目录: $RTL_DIR"
echo "SDC: $SDC_FILE"
echo "输出网表: $NETLIST_FILE"
echo ""

# 检查 SDC 文件
if [ ! -f "$SDC_FILE" ]; then
    echo "警告: 未找到 SDC 约束文件: $SDC_FILE"
    echo "将不使用时序约束进行综合"
    SDC_NOTE="(无时序约束)"
else
    echo "✓ 找到 SDC 约束文件"
    SDC_NOTE="(带时序约束)"
fi
echo ""

# 创建 Yosys 综合脚本
cat > /tmp/generic_synth.ys << 'EOF'
# 读取 RTL 设计
read_verilog -sv ../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv

# 设置顶层模块
hierarchy -top SimpleEdgeAiSoC
hierarchy -check

# 综合流程
proc
opt
fsm
opt
memory
opt

# 技术映射到通用逻辑门
techmap
opt

# 映射到 DFF
dfflibmap -liberty +/simcells.lib

# ABC 优化（使用通用库）
abc -liberty +/simcells.lib

# 清理
clean

# 统计
stat

# 输出网表（使用简单的门级原语）
write_verilog -noattr -noexpr netlist/SimpleEdgeAiSoC_generic.v

EOF

echo "运行 Yosys 综合..."
$YOSYS_BIN /tmp/generic_synth.ys

if [ -f "$NETLIST_FILE" ]; then
    echo ""
    echo "✓ 综合成功！"
    echo "网表文件: $NETLIST_FILE"
    echo ""
    echo "网表统计:"
    wc -l "$NETLIST_FILE"
    echo ""
    
    # 复制 SDC 约束文件
    if [ -f "$SDC_FILE" ]; then
        cp "$SDC_FILE" "$OUTPUT_DIR/timing_constraints.sdc"
        echo "✓ 已复制 SDC 约束文件"
        echo ""
    fi
    
    echo "下一步:"
    echo "  1. 运行后综合仿真:"
    echo "     python run_post_syn_sim.py --simulator iverilog --netlist generic"
    echo ""
    echo "  2. 查看 SDC 约束 (如果需要):"
    echo "     cat $OUTPUT_DIR/timing_constraints.sdc"
else
    echo ""
    echo "✗ 综合失败"
    exit 1
fi
