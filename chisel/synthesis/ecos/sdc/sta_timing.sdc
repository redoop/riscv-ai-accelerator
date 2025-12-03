# OpenSTA 时序约束文件
# 针对综合后网表的实际信号名称

# 主时钟 - 100MHz
# 在输入端口上定义时钟 (PAD 是黑盒)
create_clock -name sys_clk -period 10.000 [get_ports sys_clk_i_pad]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# 输入延迟 (排除时钟端口)
set_input_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_pad*]

set_input_delay -clock sys_clk -max 2.0 [get_ports ip_sel_pad*]
set_input_delay -clock sys_clk -min 0.5 [get_ports ip_sel_pad*]

# 输出延迟
set_output_delay -clock sys_clk -max 2.0 [get_ports io_pad*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_pad*]

# 假路径 - 复位
set_false_path -from [get_ports rst_n_pad]

# 环境约束
set_input_transition 0.5 [all_inputs]
set_load 0.02 [all_outputs]

