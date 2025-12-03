# OpenSTA 时序分析脚本 (最终版)

read_liberty pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top
read_sdc sdc/sta_timing_final.sdc

puts "\n========== Setup 关键路径 (Top 10) =========="
report_checks -path_delay max -format full_clock_expanded -digits 3 -endpoint_count 10

puts "\n========== Hold 关键路径 (Top 10) =========="
report_checks -path_delay min -format full_clock_expanded -digits 3 -endpoint_count 10

puts "\n========== 时序摘要 =========="
report_tns
report_wns

puts "\n========== 时序违例 (如果有) =========="
report_checks -path_delay max -slack_max 0.0

puts "\n=========================================="
puts "时序分析完成"
puts "WNS (Worst Negative Slack): [get_property [lindex [find_timing_paths -sort_by_slack] 0] slack]"
puts "=========================================="
