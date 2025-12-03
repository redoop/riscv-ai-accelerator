# 绕过 PAD 黑盒的时序约束
# 在所有触发器的时钟引脚上定义时钟

# 主时钟：100MHz (10ns)
create_clock -name sys_clk -period 10.0 [get_pins -of_objects [get_nets sys_clk] -filter "direction==in"]

# 设置时钟不确定性
set_clock_uncertainty 0.5 [get_clocks sys_clk]

# 设置输入/输出延迟
set_input_delay -clock sys_clk 2.0 [all_inputs]
set_output_delay -clock sys_clk 2.0 [all_outputs]
