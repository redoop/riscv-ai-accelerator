# OpenSTA 时序分析 - 绕过 PAD 黑盒
# 直接在内部时钟网络上定义时钟

# 读取 Liberty 库 (typical corner: TT 1.2V 25C)
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 读取约束
read_sdc sdc/sta_bypass_pad.sdc

# 报告时钟
puts "\n========== Clock Report =========="
report_checks -path_delay min_max -format full_clock_expanded

# 报告时序
puts "\n========== Timing Report =========="
report_checks -path_delay min_max -fields {slew cap input nets fanout} -digits 3

# 报告 WNS/TNS
puts "\n========== Summary =========="
report_worst_slack
report_tns
report_checks -path_delay max -format summary

puts "\n========== Done =========="
