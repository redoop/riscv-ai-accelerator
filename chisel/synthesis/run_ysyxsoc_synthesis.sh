#!/bin/bash
# ysyxSoC Wrapper 逻辑综合脚本
# 综合 ysyx_26000001 (包含 AI 加速器的 ysyxSoC 包装器)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置路径
YOSYS_BIN="/opt/tools/oss-cad/oss-cad-suite/bin/yosys"
PDK_ROOT="$SCRIPT_DIR/pdk/icsprout55-pdk"
# 使用 H7CL (Low) 标准单元库，典型角度
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"
FILELIST="filelist_ysyxsoc.f"
SDC_FILE="fpga/constraints/timing_complete.sdc"
OUTPUT_DIR="netlist"
NETLIST_FILE="$OUTPUT_DIR/ysyx_26000001_ics55.v"
TOP_MODULE="ysyx_26000001"

# 检查 SimpleEdgeAiSoC.sv 是否存在
SIMPLE_EDGEAI_SRC="../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv"
SIMPLE_EDGEAI_DEST="ysyxSoc/all_verilog/SimpleEdgeAiSoC.sv"

if [ ! -f "$SIMPLE_EDGEAI_SRC" ]; then
    echo "错误: 未找到 SimpleEdgeAiSoC.sv: $SIMPLE_EDGEAI_SRC"
    echo "请先生成 Chisel RTL: cd ../chisel && make"
    exit 1
fi

# 复制最新的 SimpleEdgeAiSoC.sv
echo "复制 SimpleEdgeAiSoC.sv 到 ysyxSoc 目录..."
cp "$SIMPLE_EDGEAI_SRC" "$SIMPLE_EDGEAI_DEST"

# 检查 filelist
if [ ! -f "$FILELIST" ]; then
    echo "错误: 未找到文件列表: $FILELIST"
    exit 1
fi

# 检查 PDK 是否存在
if [ ! -d "$PDK_ROOT" ]; then
    echo "错误: 未找到 ICS55 PDK"
    echo "请运行: python pdk/get_ics55_pdk.py"
    exit 1
fi

if [ ! -f "$LIBERTY_FILE" ]; then
    echo "错误: 未找到 ICS55 PDK Liberty 文件: $LIBERTY_FILE"
    echo "请检查 PDK 安装"
    exit 1
fi

if [ ! -f "$VERILOG_MODEL" ]; then
    echo "错误: 未找到 ICS55 PDK Verilog 模型: $VERILOG_MODEL"
    echo "请检查 PDK 安装"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "ysyxSoC Wrapper 逻辑综合 (ICS55 PDK)"
echo "=========================================="
echo "顶层模块: $TOP_MODULE"
echo "PDK: ICS55"
echo "Liberty: $LIBERTY_FILE"
echo "Verilog: $VERILOG_MODEL"
echo "文件列表: $FILELIST"
echo "SDC: $SDC_FILE"
echo "输出: $NETLIST_FILE"
echo ""

# 检查 SDC 文件
if [ ! -f "$SDC_FILE" ]; then
    echo "警告: 未找到 SDC 约束文件: $SDC_FILE"
    echo "将不使用时序约束进行综合"
    SDC_CONSTRAINT=""
else
    echo "✓ 找到 SDC 约束文件"
    SDC_CONSTRAINT="-constr $SDC_FILE"
fi

# 读取文件列表并生成 Yosys 读取命令
echo "生成 Yosys 综合脚本..."
READ_COMMANDS=""
while IFS= read -r line; do
    # 跳过注释和空行
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue
    
    # 检查文件是否存在
    if [ ! -f "$line" ]; then
        echo "警告: 文件不存在: $line"
        continue
    fi
    
    # 根据文件扩展名选择读取命令
    if [[ "$line" == *.sv ]]; then
        READ_COMMANDS="${READ_COMMANDS}read_slang $line --compat-mode --keep-hierarchy --allow-use-before-declare --ignore-unknown-modules --ignore-timing --ignore-initial\n"
    else
        READ_COMMANDS="${READ_COMMANDS}read_verilog -sv $line\n"
    fi
done < "$FILELIST"

# 创建 Yosys 综合脚本
cat > /tmp/ysyxsoc_synth.ys << EOF
# 加载 slang 插件以支持完整的 SystemVerilog
plugin -i slang

# 读取设计文件
$(echo -e "$READ_COMMANDS")

# 设置顶层模块
hierarchy -top $TOP_MODULE
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

# 映射到 ICS55 标准单元（带时序约束）
dfflibmap -liberty $LIBERTY_FILE
abc -liberty $LIBERTY_FILE $SDC_CONSTRAINT -D 10000

# 清理
clean

# 统计
tee -o $OUTPUT_DIR/synthesis_stats_ysyxsoc.txt stat -liberty $LIBERTY_FILE

# 输出网表
write_verilog -noattr -noexpr $NETLIST_FILE

# 如果有 SDC 文件，复制到输出目录
EOF

if [ -f "$SDC_FILE" ]; then
    echo "cp $SDC_FILE $OUTPUT_DIR/timing_constraints_ysyxsoc.sdc" >> /tmp/ysyxsoc_synth.ys
fi

echo "运行 Yosys 综合..."
$YOSYS_BIN /tmp/ysyxsoc_synth.ys 2>&1 | tee "$OUTPUT_DIR/synthesis_ysyxsoc.log"

if [ -f "$NETLIST_FILE" ]; then
    echo ""
    echo "✓ 综合成功！"
    echo "网表文件: $NETLIST_FILE"
    echo ""
    echo "网表统计:"
    wc -l "$NETLIST_FILE"
    echo ""
    
    # 复制 Verilog 模型到 netlist 目录以便仿真
    cp "$VERILOG_MODEL" "$OUTPUT_DIR/ics55_LLSC_H7CL.v"
    echo "✓ 已复制标准单元 Verilog 模型"
    
    # 复制 SDC 约束文件
    if [ -f "$SDC_FILE" ]; then
        cp "$SDC_FILE" "$OUTPUT_DIR/timing_constraints_ysyxsoc.sdc"
        echo "✓ 已复制 SDC 约束文件"
    fi
    echo ""
    
    echo "设计信息:"
    echo "  - 顶层模块: $TOP_MODULE"
    echo "  - 包含模块:"
    echo "    * PicoRV32 CPU"
    echo "    * SimpleCompactAccel (矩阵加速器)"
    echo "    * SimpleBitNetAccel (BitNet 加速器)"
    echo "    * SPIFlash 控制器"
    echo "    * PSRAM 控制器"
    echo ""
    
    echo "下一步:"
    echo "  1. 查看综合统计:"
    echo "     cat $OUTPUT_DIR/synthesis_stats_ysyxsoc.txt"
    echo ""
    echo "  2. 运行后综合仿真:"
    echo "     python run_post_syn_sim.py --simulator iverilog --netlist ysyxsoc"
    echo ""
    echo "  3. 运行静态时序分析 (需要 OpenSTA):"
    echo "     sta -f $OUTPUT_DIR/timing_constraints_ysyxsoc.sdc $NETLIST_FILE"
else
    echo ""
    echo "✗ 综合失败，请查看日志: $OUTPUT_DIR/synthesis_ysyxsoc.log"
    exit 1
fi
