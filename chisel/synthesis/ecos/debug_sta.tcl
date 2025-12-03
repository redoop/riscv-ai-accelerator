# 调试 STA 问题

read_liberty pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 在输入端口定义时钟
create_clock -name sys_clk -period 10.000 [get_ports sys_clk_i_pad]

# 设置时钟传播 (关键!)
set_propagated_clock [all_clocks]

# 添加基本约束
set_input_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_output_delay -clock sys_clk -max 2.0 [get_ports io_pad*]

# 报告路径
puts "\n========== Setup 路径 =========="
report_checks -path_delay max -format full_clock_expanded

puts "\n========== Hold 路径 =========="
report_checks -path_delay min -format full_clock_expanded

puts "\n========== 时序摘要 =========="
report_tns
report_wns

