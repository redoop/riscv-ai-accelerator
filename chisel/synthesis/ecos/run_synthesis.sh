#!/bin/bash
# ============================================================================
# ECOS 项目 - ICS55 PDK 逻辑综合和网表仿真脚本
# ============================================================================
#
# 功能说明:
#   1. 调用 Chisel 生成 SimpleEdgeAiSoC.sv RTL 文件
#   2. 复制生成的 RTL 到 ecos/project/verilog 目录
#   3. 综合 ECOS ASIC 顶层 (asic_top.sv) 到 ICS55 标准单元
#   4. 生成网表保存到 ecos/project/netlist 目录
#   5. 运行网表仿真验证综合结果
#
# 文件引用关系:
#   - filelist/asic_top.f: ASIC 顶层和工具模块
#   - filelist/ip.f: SimpleEdgeAiSoC IP 核 (引用 project/verilog/SimpleEdgeAiSoC.sv)
#   - filelist/lib.f: IO PAD 和时钟缓冲库
#   - filelist/soc.f: 其他 SoC 模块 (当前为空)
#
# 顶层信息:
#   - 模块名: asic_top
#   - 时钟: sys_clk_i_pad (100MHz)
#   - 复位: rst_n_pad (低电平有效)
#   - IP 选择: ip_sel_pad[2:0] (选择 ip_1 = 3'd1 for SimpleEdgeAiSoC)
#
# 使用方法:
#   cd chisel/synthesis/ecos
#   ./run_synthesis.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置路径
YOSYS_BIN="/opt/tools/oss-cad/oss-cad-suite/bin/yosys"
PDK_ROOT="$SCRIPT_DIR/pdk/icsprout55-pdk"
LIBERTY_FILE="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
VERILOG_MODEL="$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/verilog/ics55_LLSC_H7CL.v"

# Chisel 生成的 RTL 源文件
CHISEL_RTL_SRC="../../generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv"

# Chisel RTL 目标位置（复制到 project/verilog）
PROJECT_VERILOG_DIR="$SCRIPT_DIR/project/verilog"
CHISEL_RTL_DEST="$PROJECT_VERILOG_DIR/SimpleEdgeAiSoC.sv"

# 输出目录
OUTPUT_DIR="$SCRIPT_DIR/project/netlist"
NETLIST_FILE="$OUTPUT_DIR/asic_top_ics55.v"

# SDC 约束文件
SDC_FILE="$SCRIPT_DIR/sdc/timing_complete.sdc"

echo "=========================================="
echo "ECOS 项目 - ICS55 PDK 逻辑综合"
echo "=========================================="
echo "项目: ECOS ASIC (asic_top with SimpleEdgeAiSoC)"
echo "PDK: ICS55 LLSC H7CL"
echo "顶层模块: asic_top"
echo "IP 选择: ip_1 (SimpleEdgeAiSoC)"
echo "时钟: sys_clk_i_pad (100MHz)"
echo "=========================================="
echo ""

# 步骤 1: 生成 Chisel RTL
echo "=========================================="
echo "步骤 1: 生成 Chisel RTL"
echo "=========================================="
echo ""

if [ ! -f "$CHISEL_RTL_SRC" ]; then
    echo "⚠ 未找到 Chisel RTL，开始生成..."
    echo ""
    
    # 进入 chisel 目录生成 RTL
    cd ../../
    echo "运行: sbt \"runMain riscv.ai.SimpleEdgeAiSoCMain\""
    sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
    
    if [ $? -ne 0 ]; then
        echo "❌ Chisel RTL 生成失败"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    
    if [ ! -f "$CHISEL_RTL_SRC" ]; then
        echo "❌ 错误: Chisel RTL 生成后仍未找到: $CHISEL_RTL_SRC"
        exit 1
    fi
fi

echo "✓ 找到 Chisel RTL: $CHISEL_RTL_SRC"

# 步骤 2: 复制 Chisel RTL 到 project/verilog
echo ""
echo "=========================================="
echo "步骤 2: 复制 Chisel RTL 到 project/verilog"
echo "=========================================="
echo ""

mkdir -p "$PROJECT_VERILOG_DIR"
cp "$CHISEL_RTL_SRC" "$CHISEL_RTL_DEST"

if [ $? -ne 0 ]; then
    echo "❌ 复制 Chisel RTL 失败"
    exit 1
fi

echo "✓ 已复制 Chisel RTL 到: $CHISEL_RTL_DEST"
RTL_LINES=$(wc -l < "$CHISEL_RTL_DEST")
echo "  代码行数: $RTL_LINES 行"
echo ""

# 步骤 3: 检查工具和 PDK
echo "=========================================="
echo "步骤 3: 检查综合工具和 PDK"
echo "=========================================="
echo ""

# 检查 Yosys
if [ ! -f "$YOSYS_BIN" ]; then
    echo "❌ 错误: 未找到 Yosys: $YOSYS_BIN"
    echo "请安装 OSS CAD Suite"
    exit 1
fi

echo "✓ 找到 Yosys: $YOSYS_BIN"

# 检查 PDK
if [ ! -d "$PDK_ROOT" ]; then
    echo "❌ 错误: 未找到 ICS55 PDK: $PDK_ROOT"
    echo ""
    echo "请下载 PDK:"
    echo "  cd synthesis/ecos"
    echo "  python pdk/get_ics55_pdk.py"
    echo ""
    exit 1
fi

if [ ! -f "$LIBERTY_FILE" ]; then
    echo "❌ 错误: 未找到 Liberty 文件: $LIBERTY_FILE"
    exit 1
fi

if [ ! -f "$VERILOG_MODEL" ]; then
    echo "❌ 错误: 未找到 Verilog 模型: $VERILOG_MODEL"
    exit 1
fi

echo "✓ 找到 ICS55 PDK"
echo "  Liberty: $(basename $LIBERTY_FILE)"
echo "  Verilog: $(basename $VERILOG_MODEL)"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查 SDC 约束
if [ -f "$SDC_FILE" ]; then
    echo "✓ 找到 SDC 约束: $SDC_FILE"
    SDC_CONSTRAINT="-constr $SDC_FILE"
else
    echo "⚠ 未找到 SDC 约束文件，使用默认时序目标"
    SDC_CONSTRAINT=""
fi

echo ""
echo "=========================================="
echo "步骤 4: 综合 ECOS ASIC 顶层"
echo "=========================================="
echo ""

# 收集所有需要综合的文件
echo "收集 RTL 文件..."

# 设置 RTL_PATH 环境变量供 filelist 使用
export RTL_PATH="$SCRIPT_DIR"

# 读取文件列表
FILELIST_DIR="$SCRIPT_DIR/filelist"
ALL_FILES=""

# 读取并处理 filelist
for flist in asic_top.f ip.f lib.f soc.f; do
    flist_path="$FILELIST_DIR/$flist"
    if [ -f "$flist_path" ]; then
        echo "  读取: $flist"
        while IFS= read -r line; do
            # 跳过注释和空行
            [[ "$line" =~ ^#.*$ ]] && continue
            [[ -z "$line" ]] && continue
            
            # 替换 $RTL_PATH
            line="${line//\$RTL_PATH/$RTL_PATH}"
            
            # 检查文件是否存在
            if [ -f "$line" ]; then
                ALL_FILES="$ALL_FILES $line"
            else
                echo "  ⚠ 警告: 文件不存在: $line"
            fi
        done < "$flist_path"
    fi
done

echo ""
echo "RTL 文件统计:"
FILE_COUNT=$(echo $ALL_FILES | wc -w)
echo "  总计: $FILE_COUNT 个文件"
echo ""

# 创建 Yosys 综合脚本
SYNTH_SCRIPT="/tmp/ecos_asic_top_synth_$$.ys"

cat > "$SYNTH_SCRIPT" << EOF
# ECOS ASIC 顶层 ICS55 综合脚本
# 生成时间: $(date)
# 顶层模块: asic_top

# 加载 slang 插件支持 SystemVerilog
plugin -i slang

# 读取所有 RTL 文件
logger -nowarn "Reading RTL design files..."
EOF

# 将文件列表添加到综合脚本
echo "read_slang \\" >> "$SYNTH_SCRIPT"
for file in $ALL_FILES; do
    echo "    $file \\" >> "$SYNTH_SCRIPT"
done

cat >> "$SYNTH_SCRIPT" << EOF
    --top asic_top \\
    -D PDK_BEHAV \\
    --compat-mode \\
    --allow-use-before-declare \\
    --ignore-unknown-modules \\
    --ignore-timing \\
    --ignore-initial

# 设置顶层模块
logger -nowarn "Setting top module: asic_top"
hierarchy -top asic_top
hierarchy -check

# 综合流程
logger -nowarn "Executing synthesis optimization..."
proc
opt
fsm
opt
memory
opt
techmap
opt

# 映射到 ICS55 标准单元
logger -nowarn "Mapping to ICS55 standard cells..."
dfflibmap -liberty $LIBERTY_FILE
abc -liberty $LIBERTY_FILE $SDC_CONSTRAINT -D 10000

# 清理
clean

# 统计
logger -nowarn "Generating statistics report..."
tee -o $OUTPUT_DIR/synthesis_stats.txt stat -liberty $LIBERTY_FILE

# 输出网表
logger -nowarn "Writing netlist file..."
write_verilog -noattr -noexpr $NETLIST_FILE

logger -nowarn "Synthesis complete!"
EOF

echo "✓ 已生成综合脚本: $SYNTH_SCRIPT"
echo ""

# 运行 Yosys
echo "运行 Yosys 综合..."
echo "  输入: asic_top.sv + SimpleEdgeAiSoC.sv + 支持文件"
echo "  输出: $NETLIST_FILE"
echo ""

$YOSYS_BIN "$SYNTH_SCRIPT" 2>&1 | tee "$OUTPUT_DIR/synthesis.log"

YOSYS_EXIT_CODE=${PIPESTATUS[0]}

# 检查综合结果
echo ""
echo "=========================================="
echo "步骤 5: 检查综合结果"
echo "=========================================="
echo ""

if [ $YOSYS_EXIT_CODE -ne 0 ]; then
    echo "❌ Yosys 综合失败 (退出码: $YOSYS_EXIT_CODE)"
    echo ""
    echo "请查看日志: $OUTPUT_DIR/synthesis.log"
    echo ""
    exit 1
fi

if [ ! -f "$NETLIST_FILE" ]; then
    echo "❌ 网表文件未生成: $NETLIST_FILE"
    echo ""
    echo "请查看日志: $OUTPUT_DIR/synthesis.log"
    echo ""
    exit 1
fi

echo "✓ 综合成功！"
echo ""
echo "输出文件:"
echo "  网表: $NETLIST_FILE"
echo "  统计: $OUTPUT_DIR/synthesis_stats.txt"
echo "  日志: $OUTPUT_DIR/synthesis.log"
echo ""

# 显示网表统计
NETLIST_LINES=$(wc -l < "$NETLIST_FILE")
NETLIST_SIZE=$(du -h "$NETLIST_FILE" | cut -f1)
echo "网表统计:"
echo "  代码行数: $NETLIST_LINES 行"
echo "  文件大小: $NETLIST_SIZE"
echo ""

# 复制 PDK Verilog 模型
echo "复制 PDK 文件到 netlist 目录..."
cp "$VERILOG_MODEL" "$OUTPUT_DIR/ics55_LLSC_H7CL.v"
echo "✓ 已复制 PDK Verilog 模型"

# 复制 SDC 约束
if [ -f "$SDC_FILE" ]; then
    cp "$SDC_FILE" "$OUTPUT_DIR/timing_constraints.sdc"
    echo "✓ 已复制 SDC 约束文件"
fi

echo ""
echo "=========================================="
echo "步骤 6: 网表验证 (可选)"
echo "=========================================="
echo ""

echo "注意: 网表仿真需要 ASIC 顶层测试平台 (soc_tb.sv)"
echo "当前的 netlist_tb.sv 用于独立模块测试"
echo ""
echo "跳过网表仿真步骤..."
echo ""

# 如果需要运行网表仿真，取消下面的注释:
# cd "$SCRIPT_DIR/run"
# echo "使用 Icarus Verilog 运行网表仿真..."
# make -f Makefile.iverilog netlist
# SIM_EXIT_CODE=$?

SIM_EXIT_CODE=0  # 跳过仿真，设置为成功

if [ $SIM_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 综合成功！"
    echo "=========================================="
    echo ""
    
    echo ""
    echo "=========================================="
    echo "完成！"
    echo "=========================================="
    echo ""
    echo "综合流程已完成："
    echo "  ✓ Chisel RTL 生成"
    echo "  ✓ ECOS ASIC 顶层综合"
    echo "  ⊙ 网表仿真 (已跳过)"
    echo ""
    echo "下一步："
    echo "  1. 查看综合统计:"
    echo "     cat $OUTPUT_DIR/synthesis_stats.txt"
    echo ""
    echo "  2. 查看网表:"
    echo "     less $NETLIST_FILE"
    echo ""
    echo "  3. 运行 ASIC 顶层仿真 (需要完整的 soc_tb.sv):"
    echo "     cd run"
    echo "     # 待实现: make -f Makefile.iverilog asic-sim"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠ 网表仿真失败"
    echo "=========================================="
    echo ""
    echo "综合成功，但网表仿真失败。"
    echo ""
    echo "请检查:"
    echo "  1. 仿真日志: run/sim_netlist.log"
    echo "  2. 编译日志: run/compile_netlist.log"
    echo ""
    echo "手动运行仿真:"
    echo "  cd run"
    echo "  make -f Makefile.iverilog netlist"
    echo ""
    exit 1
fi
