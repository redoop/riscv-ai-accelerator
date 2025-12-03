# OpenSTA 时序分析 - 组合方案

# 读取 Liberty 库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 读取约束
read_sdc sdc/sta_combined.sdc

# 报告时序
puts "\n========== Setup Timing (Max Delay) =========="
report_checks -path_delay max -fields {slew cap input nets fanout} -format full_clock_expanded -digits 3

puts "\n========== Hold Timing (Min Delay) =========="
report_checks -path_delay min -fields {slew cap input nets fanout} -format full_clock_expanded -digits 3

# 报告 WNS/TNS
puts "\n========== Timing Summary =========="
report_worst_slack -max
report_worst_slack -min
report_tns
report_checks -path_delay max -format summary -group_count 10

puts "\n========== Done =========="
