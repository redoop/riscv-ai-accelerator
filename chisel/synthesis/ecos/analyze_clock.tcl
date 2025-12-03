# 分析时钟网络

read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/IO/ICsprout_55LLULP1233_IO_251013/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

puts "\n========== Clock Network Analysis =========="
puts "Clock: sys_clk"
puts "Period: [get_property [get_clocks sys_clk] period]"

# 检查时钟网络延迟
puts "\n========== Clock Latency =========="
report_clock_properties [get_clocks sys_clk]

# 检查触发器 _161209_ 的时钟路径
puts "\n========== Clock Path to _161209_ =========="
report_checks -from [get_clocks sys_clk] -to _161209_/CK -path_delay max

puts "\n========== Done =========="
