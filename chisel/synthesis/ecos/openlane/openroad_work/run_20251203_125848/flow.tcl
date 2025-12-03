# OpenROAD 完整流程 - ICS55 55nm

# 1. 读取 Liberty 文件
puts "========== 1. 读取 Liberty 文件 =========="
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 2. 读取 LEF 文件
puts "========== 2. 读取 LEF 文件 =========="
read_lef /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/lef/ics55_LLSC_H7CL.lef

# 3. 读取网表
puts "========== 3. 读取网表 =========="
read_verilog design.v
link_design asic_top

# 4. 创建时钟
puts "========== 4. 创建时钟 =========="
create_clock -name sys_clk -period 40.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 2.0 [get_clocks sys_clk]

# 5. 布图规划
puts "========== 5. 布图规划 =========="
initialize_floorplan \
    -die_area "0 0 2000 2000" \
    -core_area "100 100 1900 1900" \
    -site CoreSite

# 6. 放置 IO
puts "========== 6. 放置 IO =========="
place_pins -hor_layers metal3 -ver_layers metal4

# 7. 电源网络
puts "========== 7. 电源网络 =========="
add_global_connection -net VDD -pin_pattern {^VDD$} -power
add_global_connection -net VSS -pin_pattern {^VSS$} -ground

set_voltage_domain -power VDD -ground VSS

define_pdn_grid -name main_grid
add_pdn_stripe -grid main_grid -layer metal1 -width 0.48 -pitch 5.0 -offset 2.5
add_pdn_stripe -grid main_grid -layer metal4 -width 1.6 -pitch 50.0 -offset 25.0
add_pdn_connect -grid main_grid -layers {metal1 metal4}

pdngen

# 8. 全局布局
puts "========== 8. 全局布局 =========="
global_placement -density 0.30

# 9. 详细布局
puts "========== 9. 详细布局 =========="
detailed_placement

# 10. 时钟树综合
puts "========== 10. 时钟树综合 =========="
clock_tree_synthesis \
    -root_buf BUFX4 \
    -buf_list {BUFX2 BUFX4 BUFX8} \
    -wire_unit 20

# 11. 全局布线
puts "========== 11. 全局布线 =========="
global_route -guide_file route.guide \
    -layers metal1:metal6 \
    -clock_layers metal3:metal5

# 12. 详细布线
puts "========== 12. 详细布线 =========="
detailed_route -guide route.guide \
    -output_drc drc.rpt \
    -output_maze maze.log

# 13. 填充单元
puts "========== 13. 填充单元 =========="
filler_placement FILL*

# 14. 输出结果
puts "========== 14. 输出结果 =========="
write_def final.def
write_verilog final.v

# 15. 时序报告
puts "========== 15. 时序报告 =========="
report_checks -path_delay max -format full_clock_expanded > timing_max.rpt
report_checks -path_delay min -format full_clock_expanded > timing_min.rpt
report_tns > tns.rpt
report_wns > wns.rpt

# 16. 面积报告
puts "========== 16. 面积报告 =========="
report_design_area > area.rpt

puts "========== 完成 =========="
exit
