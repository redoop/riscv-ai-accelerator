#!/bin/bash
# 使用 ICS55 PDK 综合 ysyxSoC AI 加速器
# 日期: 2025-12-03

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置路径
YOSYS_BIN="yosys"
PDK_ROOT="/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"
OUTPUT_DIR="synth_output"
NETLIST_FILE="$OUTPUT_DIR/ysyxSoCTop_ics55.v"

# 检查 PDK
if [ ! -f "$LIBERTY_FILE" ]; then
    echo "错误: 未找到 ICS55 PDK Liberty 文件"
    echo "路径: $LIBERTY_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ICS55 PDK 逻辑综合"
echo "=========================================="
echo "PDK: ICS55 55nm"
echo "Liberty: $(basename $LIBERTY_FILE)"
echo "顶层: ysyxSoCTop"
echo "输出: $NETLIST_FILE"
echo ""

# 创建 Yosys 综合脚本
cat > /tmp/ysyxsoc_synth.ys << 'EOF'
# 读取所有 Verilog 文件
read_verilog -sv SimpleEdgeAiSoC.sv
read_verilog ysyx_26000001_with_ai.v
read_verilog ysyxSoCFull.v
read_verilog flash_fixed.v
read_verilog uart_*.v
read_verilog spi_*.v
read_verilog sdram*.v
read_verilog apb_delayer.v
read_verilog axi4_delayer.v
read_verilog bitrev.v
read_verilog gpio_top_apb.v
read_verilog ps2_top_apb.v
read_verilog psram*.v
read_verilog EF_PSRAM_CTRL*.v
read_verilog vga_top_apb.v
read_verilog raminfr.v

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
dfflibmap -liberty LIBERTY_FILE_PLACEHOLDER
abc -liberty LIBERTY_FILE_PLACEHOLDER -D 10000

# 清理
clean

# 统计
tee -o OUTPUT_DIR_PLACEHOLDER/synthesis_stats.txt stat -liberty LIBERTY_FILE_PLACEHOLDER

# 输出网表
write_verilog -noattr -noexpr NETLIST_FILE_PLACEHOLDER
EOF

# 替换占位符
sed -i "s|LIBERTY_FILE_PLACEHOLDER|$LIBERTY_FILE|g" /tmp/ysyxsoc_synth.ys
sed -i "s|OUTPUT_DIR_PLACEHOLDER|$OUTPUT_DIR|g" /tmp/ysyxsoc_synth.ys
sed -i "s|NETLIST_FILE_PLACEHOLDER|$NETLIST_FILE|g" /tmp/ysyxsoc_synth.ys

echo "运行 Yosys 综合..."
$YOSYS_BIN /tmp/ysyxsoc_synth.ys 2>&1 | tee "$OUTPUT_DIR/synthesis.log"

if [ -f "$NETLIST_FILE" ]; then
    echo ""
    echo "✓ 综合成功！"
    echo "网表文件: $NETLIST_FILE"
    echo ""
    echo "网表统计:"
    wc -l "$NETLIST_FILE"
    echo ""
    
    # 复制标准单元模型
    if [ -f "$VERILOG_MODEL" ]; then
        cp "$VERILOG_MODEL" "$OUTPUT_DIR/ics55_LLSC_H7CL.v"
        echo "✓ 已复制标准单元 Verilog 模型"
    fi
    
    echo ""
    echo "综合报告: $OUTPUT_DIR/synthesis_stats.txt"
    echo "综合日志: $OUTPUT_DIR/synthesis.log"
else
    echo ""
    echo "✗ 综合失败，请查看日志: $OUTPUT_DIR/synthesis.log"
    exit 1
fi
