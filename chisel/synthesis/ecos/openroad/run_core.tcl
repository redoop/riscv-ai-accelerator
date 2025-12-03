# 核心模块完整 P&R 流程 (无 IO PAD)

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "asic_top_core.v"

puts "=========================================="
puts "核心模块完整 P&R 流程"
puts "=========================================="

# 1. 读取设计
puts "\n\[1/5\] 读取设计..."
read_lef "tech_complete.lef"
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 2. 时钟约束
puts "\n\[2/5\] 创建时钟..."
create_clock -period 40.0 -name sys_clk [get_ports sys_clk_i_pad]

# 3. Floorplan
puts "\n\[3/5\] Floorplan..."
initialize_floorplan \
    -site core7 \
    -utilization 30 \
    -aspect_ratio 1.0 \
    -core_space 50

make_tracks MET1 -x_offset 0 -x_pitch 0.38 -y_offset 0 -y_pitch 2.8
make_tracks MET2 -x_offset 0 -x_pitch 0.38 -y_offset 0 -y_pitch 2.8
make_tracks MET3 -x_offset 0 -x_pitch 0.76 -y_offset 0 -y_pitch 2.8
make_tracks MET4 -x_offset 0 -x_pitch 0.76 -y_offset 0 -y_pitch 2.8
make_tracks MET5 -x_offset 0 -x_pitch 1.52 -y_offset 0 -y_pitch 2.8
make_tracks MET6 -x_offset 0 -x_pitch 1.52 -y_offset 0 -y_pitch 2.8

write_def results/core_1_floorplan.def

# 4. Placement
puts "\n\[4/5\] Placement..."
global_placement -density 0.35
detailed_placement
write_def results/core_2_placement.def

# 5. Routing
puts "\n\[5/5\] Routing..."
set_routing_layers -signal MET2-MET5
global_route -congestion_iterations 100
detailed_route -output_drc results/core_drc.rpt
write_def results/core_3_routing.def
write_verilog results/core_final.v

# 时序报告
puts "\n=========================================="
puts "时序报告"
puts "=========================================="
report_checks -path_delay max -format full_clock_expanded
report_checks -path_delay min
report_tns
report_wns

puts "\n✅ 核心模块 P&R 完成！"
puts "结果: results/core_*.def"
