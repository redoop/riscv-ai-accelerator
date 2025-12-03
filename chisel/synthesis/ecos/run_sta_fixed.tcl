# OpenSTA 时序分析脚本 (修正版)

# 读取 Liberty 库
read_liberty pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 读取修正的 SDC 约束
read_sdc sdc/sta_timing.sdc

# 报告关键路径 (Setup)
puts "\n========== Setup 关键路径 =========="
report_checks -path_delay max -format full_clock_expanded -digits 3

# 报告保持时间路径 (Hold)
puts "\n========== Hold 关键路径 =========="
report_checks -path_delay min -format full_clock_expanded -digits 3

# 报告时序摘要
puts "\n========== 时序摘要 =========="
report_tns
report_wns

puts "\n=========================================="
puts "时序分析完成"
puts "=========================================="

