# OpenROAD 简化流程 - ICS55 55nm

puts "========== 1. 读取 Liberty 文件 =========="
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

puts "========== 2. 读取 LEF 文件 =========="
read_lef /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL.lef

puts "========== 3. 读取网表 =========="
read_verilog design.v
link_design asic_top

puts "========== 4. 创建时钟 =========="
create_clock -name sys_clk -period 40.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 2.0 [get_clocks sys_clk]

puts "========== 5. 布图规划 (使用 core7 site) =========="
initialize_floorplan \
    -die_area "0 0 3000 3000" \
    -core_area "200 200 2800 2800" \
    -site core7

puts "========== 6. 放置 IO (简化) =========="
# 跳过 IO 放置，因为 IO PAD 单元缺失

puts "========== 7. 全局布局 =========="
global_placement -density 0.25 -skip_io

puts "========== 8. 详细布局 =========="
detailed_placement

puts "========== 9. 输出 DEF =========="
write_def placed.def

puts "========== 10. 输出网表 =========="
write_verilog placed.v

puts "========== 11. 时序报告 =========="
report_checks -path_delay max -format full_clock_expanded > timing_max.rpt || true
report_checks -path_delay min -format full_clock_expanded > timing_min.rpt || true

puts "========== 12. 面积报告 =========="
report_design_area > area.rpt

puts "========== 完成 =========="
puts "注意: 由于缺少 IO PAD 单元和 latch 单元，流程已简化"
puts "输出文件: placed.def, placed.v, timing_max.rpt, area.rpt"
exit
