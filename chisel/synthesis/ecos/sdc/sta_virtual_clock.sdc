# 虚拟时钟方案 - 忽略 PAD 延迟
# 注意：这种方法不包括 PAD 的延迟，仅用于评估内部逻辑时序

# 创建虚拟时钟（不关联到任何端口或引脚）
create_clock -name sys_clk_virtual -period 10.0

# 将虚拟时钟关联到所有由 sys_clk 网络驱动的寄存器
# 通过设置时钟延迟来模拟从端口到内部网络的传播
set_clock_latency 2.0 [get_clocks sys_clk_virtual]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk_virtual]

# 设置输入/输出延迟
set_input_delay -clock sys_clk_virtual 2.0 [get_ports {ip_sel_pad0 ip_sel_pad1 ip_sel_pad2 rst_n_pad}]
set_output_delay -clock sys_clk_virtual 2.0 [get_ports sys_clk_o_pad]

# 设置所有寄存器的时钟
# 注意：这需要手动指定所有寄存器，或者使用通配符
# set_case_analysis 可以用来设置特定信号的值
