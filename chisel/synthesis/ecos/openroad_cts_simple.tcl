# OpenROAD 简化 CTS 流程
# 由于完整 P&R 需要更多配置，这里演示 CTS 概念

puts "=========================================="
puts "OpenROAD CTS 演示"
puts "=========================================="

set pdk_path "/opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk"
set std_cell_path "$pdk_path/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL"
set io_path "$pdk_path/IP/IO/ICsprout_55LLULP1233_IO_251013"

puts "\n读取 LEF..."
read_lef "$std_cell_path/lef/ics55_LLSC_H7CL.lef"

puts "\n读取 Liberty..."
read_liberty "$std_cell_path/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib"
read_liberty "$io_path/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib"

puts "\n读取网表..."
read_verilog "project/netlist/asic_top_ics55.v"
link_design asic_top

puts "\n创建时钟..."
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

puts "\n=========================================="
puts "说明："
puts "=========================================="
puts ""
puts "完整的 CTS 需要先完成布局（Placement）"
puts "这需要："
puts "  1. 定义芯片尺寸（Floorplan）"
puts "  2. 放置所有单元（Placement）"
puts "  3. 然后才能进行 CTS"
puts ""
puts "由于当前网表较大（96,087 个单元），"
puts "完整的 P&R 流程需要："
puts "  - 更多的配置"
puts "  - 更长的运行时间（数小时）"
puts "  - 更多的内存"
puts ""
puts "建议："
puts "  1. 使用方案 1（25MHz）进行功能验证"
puts "  2. 学习 OpenROAD-flow-scripts"
puts "  3. 使用专业的 P&R 流程"
puts ""
puts "参考："
puts "  https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts"
