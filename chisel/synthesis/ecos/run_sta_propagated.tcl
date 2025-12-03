# OpenSTA 时序分析 - 传播时钟方案

read_liberty /opt/github/riscv-ai-accelerator/chisel/synthesis/pdk/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CL/liberty/ics55_LLSC_H7CL_typ_tt_1p2_25_nldm.lib
read_verilog project/netlist/asic_top_ics55.v
link_design asic_top

# 在输入端口创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 设置为传播时钟（尝试通过网络传播）
set_propagated_clock [get_clocks sys_clk]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 设置输入/输出延迟
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]

puts "\n========== Setup Timing =========="
report_checks -path_delay max -format full_clock_expanded -digits 3

puts "\n========== Summary =========="
report_worst_slack
report_tns

puts "\n========== Done =========="
