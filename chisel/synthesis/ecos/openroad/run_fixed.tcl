# 修复后的完整 P&R 流程

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"

puts "=========================================="
puts "修复后的完整 P&R 流程"
puts "=========================================="

# 1. 读取设计
puts "\n\[1/6\] 读取设计..."
read_lef "tech_complete.lef"
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 2. 时钟约束 - 使用内部时钟信号
puts "\n\[2/6\] 创建时钟..."
create_clock -period 40.0 -name sys_clk [get_pins u_sys_clk_pad/C]
set_wire_rc -clock -layer MET3

# 3. Floorplan
puts "\n\[3/6\] Floorplan..."
initialize_floorplan \
    -site core7 \
    -utilization 30 \
    -aspect_ratio 1.0 \
    -core_space 50

# 定义布线轨道
make_tracks MET1 -x_offset 0 -x_pitch 0.38 -y_offset 0 -y_pitch 2.8
make_tracks MET2 -x_offset 0 -x_pitch 0.38 -y_offset 0 -y_pitch 2.8
make_tracks MET3 -x_offset 0 -x_pitch 0.76 -y_offset 0 -y_pitch 2.8
make_tracks MET4 -x_offset 0 -x_pitch 0.76 -y_offset 0 -y_pitch 2.8
make_tracks MET5 -x_offset 0 -x_pitch 1.52 -y_offset 0 -y_pitch 2.8
make_tracks MET6 -x_offset 0 -x_pitch 1.52 -y_offset 0 -y_pitch 2.8

# 定义电源网络
add_global_connection -net VDD -pin_pattern {^VDD$} -power
add_global_connection -net VSS -pin_pattern {^VSS$} -ground

write_def results/1_floorplan_fixed.def

# 4. Placement
puts "\n\[4/6\] Placement..."
global_placement -density 0.35
detailed_placement
write_def results/2_placement_fixed.def

# 5. CTS
puts "\n\[5/6\] CTS..."
clock_tree_synthesis \
    -root_buf BUFX4H7L \
    -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L"
write_def results/3_cts_fixed.def

# 6. Routing
puts "\n\[6/6\] Routing..."
set_routing_layers -signal MET1-MET6
global_route
detailed_route
write_def results/4_routing_fixed.def
write_verilog results/final_fixed.v

# 时序报告
puts "\n=========================================="
puts "最终时序报告"
puts "=========================================="
report_checks -path_delay max
report_checks -path_delay min
report_tns
report_wns
report_clock_skew

puts "\n✅ 完整 P&R 流程完成！"
