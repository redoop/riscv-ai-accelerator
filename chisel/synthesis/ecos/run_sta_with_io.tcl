# OpenSTA 时序分析 - 包含 IO PAD 库

# 读取标准单元库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib

# 读取 IO PAD 库
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/IO/ICsprout_55LLULP1233_IO_251013/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib

# 读取网表
read_verilog project/netlist/asic_top_ics55.v

# 链接设计
link_design asic_top

# 在输入端口创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 设置输入延迟
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]

# 设置输出延迟
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]

# 报告时序
puts "\n========== Setup Timing =========="
report_checks -path_delay max -format full_clock_expanded -digits 3

puts "\n========== Hold Timing =========="
report_checks -path_delay min -format full_clock_expanded -digits 3

puts "\n========== Summary =========="
report_worst_slack -max
report_worst_slack -min
report_tns
report_checks -path_delay max -format summary -group_count 10

puts "\n========== Done =========="
