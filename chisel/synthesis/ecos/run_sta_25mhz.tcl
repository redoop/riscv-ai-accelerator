# OpenSTA 时序分析 - 25MHz

read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/IO/ICsprout_55LLULP1233_IO_251013/liberty/ICSIOA_N55_3P3_tt_1p2_3p3_25c.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 25MHz = 40 ns 周期
create_clock -name sys_clk -period 40.0 [get_ports sys_clk_i_pad]
set_clock_uncertainty 0.5 [get_clocks sys_clk]
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]

puts "\n========== 25MHz Timing Analysis =========="
puts "Clock Period: 40 ns"
puts "Target Frequency: 25 MHz"

puts "\n========== Setup Timing =========="
report_checks -path_delay max -digits 3

puts "\n========== Summary =========="
report_worst_slack -max
report_worst_slack -min
report_tns

puts "\n========== Done =========="
