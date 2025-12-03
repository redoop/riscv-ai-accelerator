# OpenROAD 完整 P&R 流程

puts "=========================================="
puts "OpenROAD P&R 流程 - 时钟树综合"
puts "=========================================="

set pdk_path "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set std_cell_path "$pdk_path/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set io_path "$pdk_path/IP/IO/ICsprout_55LLULP1233_IO_251013"

puts "\nStep 1: 读取 LEF 文件..."
read_lef "$std_cell_path/lef/ics55_LLSC_H7CL.lef"
read_lef "$io_path/lef/ICSIOA_N55_3P3_1P6M1TM.lef"

puts "\nStep 2: 读取 Liberty 库..."
read_liberty "$std_cell_path/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_liberty "$io_path/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib"

puts "\nStep 3: 读取网表..."
read_verilog "project/netlist/asic_top_ics55.v"
link_design asic_top

puts "\nStep 4: 创建时钟约束..."
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 0.5 [get_clocks sys_clk]

puts "\nStep 5: 布图规划..."
initialize_floorplan \
  -die_area "0 0 600 600" \
  -core_area "50 50 550 550" \
  -site core7

puts "\nStep 6: 全局布局..."
global_placement -density 0.7

puts "\nStep 7: 详细布局..."
detailed_placement

puts "\nStep 8: 时钟树综合..."
clock_tree_synthesis \
  -root_buf BUFX8H7L \
  -buf_list "BUFX2H7L BUFX4H7L BUFX8H7L" \
  -wire_unit 20

puts "\n输出结果..."
write_verilog "project/netlist/asic_top_cts.v"
write_def "project/netlist/asic_top_cts.def"

puts "\n=========================================="
puts "时钟树综合完成"
puts "=========================================="

puts "\n时钟 Skew 报告:"
report_clock_skew

puts "\n时序报告:"
report_checks -path_delay max -format summary

puts "\n输出文件:"
puts "  - 网表: project/netlist/asic_top_cts.v"
puts "  - DEF: project/netlist/asic_top_cts.def"
