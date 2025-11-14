#!/usr/bin/tclsh
# 逻辑综合后网表仿真脚本
# 用于验证综合后网表的功能正确性

puts "=========================================="
puts "逻辑综合后网表仿真脚本"
puts "=========================================="
puts ""

# 配置参数
set DESIGN_NAME "SimpleEdgeAiSoC"
set NETLIST_DIR "./synthesis/netlist"
set TB_DIR "./synthesis/testbench"
set SIM_DIR "./synthesis/sim"
set WAVE_DIR "./synthesis/waves"

# 创建目录
file mkdir $SIM_DIR
file mkdir $WAVE_DIR

puts "设计名称: $DESIGN_NAME"
puts "网表目录: $NETLIST_DIR"
puts "仿真目录: $SIM_DIR"
puts ""

# 检查网表文件是否存在
set netlist_file "$NETLIST_DIR/${DESIGN_NAME}_syn.v"
if {![file exists $netlist_file]} {
    puts "错误: 网表文件不存在: $netlist_file"
    puts "请先运行逻辑综合生成网表"
    exit 1
}

puts "✓ 找到网表文件: $netlist_file"
puts ""

# 编译网表和测试平台
puts "编译网表和测试平台..."
puts "----------------------------------------"

# 这里使用 VCS 作为示例，可以根据实际工具修改
# VCS 编译命令
set compile_cmd "vcs -full64 -sverilog \
    -timescale=1ns/1ps \
    +v2k \
    -debug_all \
    -kdb \
    -lca \
    -f $TB_DIR/filelist.f \
    $netlist_file \
    -o $SIM_DIR/simv \
    -l $SIM_DIR/compile.log"

puts "编译命令: $compile_cmd"
puts ""

# 运行仿真
puts "运行仿真..."
puts "----------------------------------------"

set sim_cmd "$SIM_DIR/simv \
    +vcs+finish+100000000 \
    -l $SIM_DIR/sim.log \
    -ucli -i $TB_DIR/sim.tcl"

puts "仿真命令: $sim_cmd"
puts ""

puts "=========================================="
puts "仿真完成"
puts "=========================================="
puts ""
puts "查看结果:"
puts "  编译日志: $SIM_DIR/compile.log"
puts "  仿真日志: $SIM_DIR/sim.log"
puts "  波形文件: $WAVE_DIR/"
puts ""
