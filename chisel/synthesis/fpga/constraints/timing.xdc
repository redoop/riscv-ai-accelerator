# 时序约束文件 - RISC-V AI 加速器

# 主时钟约束
create_clock -period 10.000 -name sys_clk [get_ports clock]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clock]

# 输入延迟约束
set_input_delay -clock sys_clk -max 2.0 [get_ports reset]
set_input_delay -clock sys_clk -max 2.0 [get_ports io_uart_rx]
set_input_delay -clock sys_clk -max 2.0 [get_ports io_gpio_in*]

# 输出延迟约束
set_output_delay -clock sys_clk -max 2.0 [get_ports io_uart_tx]
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_out*]
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_oe*]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# 最大延迟约束
set_max_delay 10.0 -from [all_inputs] -to [all_outputs]

# 多周期路径（如果需要）
# set_multicycle_path -setup 2 -from [get_pins ...] -to [get_pins ...]

# 假路径（跨时钟域）
# set_false_path -from [get_clocks clk_a] -to [get_clocks clk_b]

# 时序例外
set_false_path -from [get_ports reset]
