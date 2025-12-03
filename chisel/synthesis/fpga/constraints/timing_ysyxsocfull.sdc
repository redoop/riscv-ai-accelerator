# ysyxSoCFull 时序约束文件
# 用于 ICS55 PDK 逻辑综合

# 设置时钟周期 (50MHz = 20ns)
# 可根据目标频率调整
set CLOCK_PERIOD 20.0
set CLOCK_PORT clock

# 创建时钟
create_clock -name clk -period $CLOCK_PERIOD [get_ports $CLOCK_PORT]

# 设置时钟不确定性 (jitter + skew)
set_clock_uncertainty 0.5 [get_clocks clk]

# 设置时钟转换时间
set_clock_transition 0.1 [get_clocks clk]

# 输入延迟约束 (假设外部逻辑延迟为 30% 时钟周期)
set INPUT_DELAY [expr $CLOCK_PERIOD * 0.3]
set_input_delay -clock clk -max $INPUT_DELAY [all_inputs]
set_input_delay -clock clk -min 0 [all_inputs]

# 输出延迟约束 (假设外部逻辑延迟为 30% 时钟周期)
set OUTPUT_DELAY [expr $CLOCK_PERIOD * 0.3]
set_output_delay -clock clk -max $OUTPUT_DELAY [all_outputs]
set_output_delay -clock clk -min 0 [all_outputs]

# 复位信号不需要时序约束
set_false_path -from [get_ports reset]

# 设置最大扇出
set_max_fanout 16 [current_design]

# 设置最大转换时间
set_max_transition 0.5 [current_design]

# 设置负载
set_load 0.1 [all_outputs]

# 设置驱动强度
set_driving_cell -lib_cell BUFX2 -library ics55_LLSC_H7CL [all_inputs]

# 异步信号路径 (如果有的话)
# set_false_path -from [get_ports async_input] -to [get_registers *]

# 多周期路径 (如果有的话)
# set_multicycle_path -setup 2 -from [get_registers src_reg] -to [get_registers dst_reg]
# set_multicycle_path -hold 1 -from [get_registers src_reg] -to [get_registers dst_reg]

# 时钟组约束 (如果有多个时钟域)
# set_clock_groups -asynchronous -group [get_clocks clk1] -group [get_clocks clk2]
