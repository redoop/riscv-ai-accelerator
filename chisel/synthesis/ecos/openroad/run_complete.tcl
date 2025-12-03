# 完整 P&R 流程 - 25MHz

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"

puts "=========================================="
puts "完整 P&R 流程 - 25MHz"
puts "=========================================="

# 1. 读取设计
puts "\n\[1/6\] 读取设计..."
read_lef "tech.lef"
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 2. 时钟约束
puts "\n\[2/6\] 创建时钟 (25MHz)..."
create_clock -period 40.0 sys_clk_i_pad

# 3. Floorplan
puts "\n\[3/6\] Floorplan..."
initialize_floorplan \
    -site core7 \
    -utilization 30 \
    -aspect_ratio 1.0 \
    -core_space 50
write_def results/1_floorplan.def

# 4. Placement
puts "\n\[4/6\] Placement..."
global_placement -density 0.35
detailed_placement
write_def results/2_placement.def

# 5. CTS
puts "\n\[5/6\] 时钟树综合..."
clock_tree_synthesis \
    -root_buf BUFX4H7L \
    -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L"
write_def results/3_cts.def

# 6. Routing
puts "\n\[6/6\] 布线..."
global_route
detailed_route
write_def results/4_routing.def
write_verilog results/final.v

# 时序报告
puts "\n=========================================="
puts "最终时序报告"
puts "=========================================="
report_checks -path_delay max -format full_clock_expanded
report_checks -path_delay min
report_tns
report_wns
report_clock_skew

puts "\n✅ 完整 P&R 流程完成！"
puts "结果: results/"
