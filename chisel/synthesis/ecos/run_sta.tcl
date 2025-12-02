# OpenSTA 时序分析脚本
# 用于分析综合后网表的时序

# 读取 Liberty 库
read_liberty pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 读取 SDC 约束
read_sdc sdc/asic_top_timing.sdc

# 报告时钟
report_checks -path_delay min_max -format full_clock_expanded

# 报告关键路径
report_checks -path_delay max -fields {slew cap input nets fanout} -format full

# 报告时序摘要
report_tns
report_wns

# 报告所有违例
report_checks -path_delay max -slack_max 0.0

puts "\n=========================================="
puts "时序分析完成"
puts "=========================================="
