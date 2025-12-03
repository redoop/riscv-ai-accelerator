# 组合方案：端口时钟 + 理想网络

# 在输入端口上创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 将内部时钟网络设置为理想网络（绕过 PAD 黑盒）
set_ideal_network [get_nets sys_clk]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 设置输入延迟（排除时钟端口）
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]

# 设置输出延迟
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]
