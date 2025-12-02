# ASIC 顶层时序约束 (asic_top)
# 适配 ECOS ASIC 封装后的端口名称

# 主时钟 - 100MHz (从 25MHz 输入 PLL 倍频)
create_clock -name sys_clk -period 10.000 [get_ports sys_clk_i_pad]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# 时钟延迟
set_clock_latency -source 0.5 [get_clocks sys_clk]
set_clock_latency 0.3 [get_clocks sys_clk]

# 时钟转换时间
set_clock_transition 0.1 [get_clocks sys_clk]

# 输入延迟 - GPIO
set_input_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_pad*]

# 输入延迟 - 复位
set_input_delay -clock sys_clk -max 2.0 [get_ports rst_n_pad]
set_input_delay -clock sys_clk -min 0.5 [get_ports rst_n_pad]

# 输入延迟 - IP 选择
set_input_delay -clock sys_clk -max 2.0 [get_ports ip_sel_pad*]
set_input_delay -clock sys_clk -min 0.5 [get_ports ip_sel_pad*]

# 输出延迟 - GPIO
set_output_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_pad*]

# 假路径 - 复位是异步的
set_false_path -from [get_ports rst_n_pad]

# 输入转换时间
set_input_transition 0.5 [all_inputs]

# 输出负载
set_load 2.0 [all_outputs]

# 设计规则
set_max_fanout 16 [current_design]
set_max_transition 0.5 [current_design]
set_max_capacitance 0.5 [current_design]
