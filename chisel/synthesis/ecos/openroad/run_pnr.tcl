# OpenROAD 完整 P&R 流程
# 针对 RISC-V AI 加速器芯片

# 配置
set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"
set CLOCK_PERIOD 40.0
set CLOCK_PORT "sys_clk_i_pad"

puts "=========================================="
puts "OpenROAD P&R 流程"
puts "设计: $DESIGN_NAME"
puts "时钟周期: ${CLOCK_PERIOD}ns (25MHz)"
puts "=========================================="

# 1. 读取设计
puts "\nStep 1: 读取设计..."
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 2. 时钟约束
puts "\nStep 2: 创建时钟约束..."
create_clock -name sys_clk -period $CLOCK_PERIOD $CLOCK_PORT
set_clock_uncertainty 2.0 sys_clk

# 3. Floorplan
puts "\nStep 3: 布图规划..."
initialize_floorplan \
    -utilization 50 \
    -aspect_ratio 1.0 \
    -core_space 10

# 4. Placement
puts "\nStep 4: 全局布局..."
global_placement -density 0.5

puts "\n详细布局..."
detailed_placement

# 5. CTS
puts "\nStep 5: 时钟树综合..."
clock_tree_synthesis \
    -root_buf BUFX4H7L \
    -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L"

# 6. Routing
puts "\nStep 6: 全局布线..."
global_route

puts "\n详细布线..."
detailed_route

# 保存结果
puts "\n保存结果..."
write_def results/final.def
write_verilog results/final.v

# 时序报告
puts "\n=========================================="
puts "时序报告"
puts "=========================================="
report_checks -path_delay max
report_checks -path_delay min
report_tns
report_wns

puts "\n✅ P&R 流程完成"
puts "结果保存在: results/"
