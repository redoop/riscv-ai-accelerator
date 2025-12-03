# OpenSTA 时序分析 - 包含 PAD 模型

# 读取 Liberty 库
read_liberty pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取 PAD 功能模型
read_verilog lib/pad_models.v

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 定义时钟
create_clock -name sys_clk -period 10.000 [get_ports sys_clk_i_pad]

# 时钟约束
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# I/O 约束
set_input_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_pad*]
set_output_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_pad*]

# 假路径
set_false_path -from [get_ports rst_n_pad]

# 环境
set_input_transition 0.5 [all_inputs]
set_load 0.02 [all_outputs]

# 报告
puts "\n========== Setup 关键路径 (Top 5) =========="
report_checks -path_delay max -format full_clock_expanded -digits 3 -endpoint_count 5

puts "\n========== Hold 关键路径 (Top 5) =========="
report_checks -path_delay min -format full_clock_expanded -digits 3 -endpoint_count 5

puts "\n========== 时序摘要 =========="
report_tns
report_wns

puts "\n=========================================="
puts "时序分析完成"
puts "=========================================="
