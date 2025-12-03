#!/bin/bash
# 综合完整的 ysyxSoCTop
# 使用 ICS55 PDK

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 使用 oss-cad-suite 的 yosys
YOSYS_BIN="/opt/tools/oss-cad/oss-cad-suite/bin/yosys"
PDK_ROOT="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"
OUTPUT_DIR="synth_ysyxsoc"
NETLIST_FILE="$OUTPUT_DIR/ysyxSoCTop_ics55.v"

# 检查 PDK
if [ ! -f "$LIBERTY_FILE" ]; then
    echo "错误: 未找到 ICS55 PDK"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ysyxSoCTop ICS55 综合"
echo "=========================================="
echo "Yosys: $YOSYS_BIN"
echo "PDK: ICS55 55nm"
echo "顶层: ysyxSoCTop"
echo "输出: $NETLIST_FILE"
echo ""

# 创建 Yosys 综合脚本
cat > /tmp/ysyxsoc_synth.ys << EOF
# 加载 slang 插件
plugin -i slang

# 使用 slang 读取所有文件（支持 SystemVerilog）
read_slang SimpleEdgeAiSoC.sv ysyx_26000001_with_ai.v ysyxSoCFull.v \\
    flash_fixed.v uart_*.v spi_*.v sdram*.v \\
    apb_delayer.v axi4_delayer.v bitrev.v \\
    gpio_top_apb.v ps2_top_apb.v psram*.v \\
    EF_PSRAM_CTRL*.v vga_top_apb.v raminfr.v \\
    --compat-mode --keep-hierarchy \\
    --allow-use-before-declare --ignore-unknown-modules \\
    --ignore-timing --ignore-initial

# 设置顶层
hierarchy -top ysyxSoCTop
hierarchy -check

# 综合流程
proc
opt
fsm
opt
memory
opt
techmap
opt

# 映射到 ICS55 标准单元
dfflibmap -liberty $LIBERTY_FILE
abc -liberty $LIBERTY_FILE -D 10000

# 清理
clean

# 统计
tee -o $OUTPUT_DIR/synthesis_stats.txt stat -liberty $LIBERTY_FILE

# 输出网表
write_verilog -noattr -noexpr $NETLIST_FILE
EOF

echo "运行 Yosys 综合..."
$YOSYS_BIN /tmp/ysyxsoc_synth.ys 2>&1 | tee "$OUTPUT_DIR/synthesis.log"

if [ -f "$NETLIST_FILE" ]; then
    echo ""
    echo "✓ 综合成功！"
    echo "网表: $NETLIST_FILE"
    echo "统计: $OUTPUT_DIR/synthesis_stats.txt"
    echo "日志: $OUTPUT_DIR/synthesis.log"
    echo ""
    wc -l "$NETLIST_FILE"
    
    # 复制标准单元库
    cp "$VERILOG_MODEL" "$OUTPUT_DIR/ics55_LLSC_H7CL.v"
    echo "✓ 已复制标准单元库"
else
    echo ""
    echo "✗ 综合失败"
    exit 1
fi
