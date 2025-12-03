# 在 PAD 输出引脚上定义时钟

# 在输入端口上创建时钟
create_clock -name sys_clk -period 10.0 [get_ports sys_clk_i_pad]

# 在 PAD 的输出引脚上也创建生成时钟
create_generated_clock -name sys_clk_internal -source [get_ports sys_clk_i_pad] -divide_by 1 [get_pins u_sys_clk_pad/XC]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [all_clocks]

# 设置输入延迟
set_input_delay -clock sys_clk 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]

# 设置输出延迟
set_output_delay -clock sys_clk 2.0 [get_ports sys_clk_o_pad]
