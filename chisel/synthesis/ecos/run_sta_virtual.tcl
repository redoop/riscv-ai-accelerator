# OpenSTA 时序分析 - 虚拟时钟方案

# 读取 Liberty 库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 创建虚拟时钟
create_clock -name sys_clk -period 10.0

# 设置时钟延迟（模拟 PAD + 时钟树）
set_clock_latency 2.0 [get_clocks sys_clk]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 设置输入延迟
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]

# 设置输出延迟
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]

# 报告时序
puts "\n========== Setup Timing (Max Delay) =========="
report_checks -path_delay max -format full_clock_expanded -fields {slew cap input nets fanout} -digits 3 -path_group sys_clk

puts "\n========== Hold Timing (Min Delay) =========="
report_checks -path_delay min -format full_clock_expanded -fields {slew cap input nets fanout} -digits 3 -path_group sys_clk

puts "\n========== Timing Summary =========="
report_worst_slack -max
report_worst_slack -min
report_tns
report_checks -path_delay max -format summary -group_count 10

puts "\n========== Done =========="
