# 简化的 25MHz P&R 流程

set PDK_ROOT "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set STD_CELL_PATH "$PDK_ROOT/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set DESIGN_NAME "asic_top"
set VERILOG_FILE "../project/netlist/asic_top_ics55.v"

puts "=========================================="
puts "简化 P&R 流程 - 25MHz"
puts "=========================================="

# 读取设计
puts "\n读取 Tech LEF..."
read_lef "tech.lef"

puts "读取 Cell LEF..."
read_lef "$STD_CELL_PATH/lef/ics55_LLSC_H7CL.lef"

puts "读取 Liberty..."
read_liberty "$STD_CELL_PATH/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"

puts "读取网表..."
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 时钟约束 - 25MHz (40ns)
puts "\n创建时钟 (25MHz)..."
create_clock -period 40.0 sys_clk_i_pad

# Floorplan
puts "\nFloorplan..."
initialize_floorplan \
    -site core7 \
    -utilization 30 \
    -aspect_ratio 1.0 \
    -core_space 50

puts "\n✅ Floorplan 完成"
write_def results/floorplan_25mhz.def

# Placement
puts "\nPlacement..."
global_placement -density 0.35
detailed_placement

puts "\n✅ Placement 完成"
write_def results/placement_25mhz.def

# 时序报告
puts "\n=========================================="
puts "时序报告 (25MHz)"
puts "=========================================="
report_checks -path_delay max -format full_clock_expanded
report_tns
report_wns

puts "\n✅ 完成"
