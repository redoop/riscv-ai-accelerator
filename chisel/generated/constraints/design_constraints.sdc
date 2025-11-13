
# SDC约束文件 - PhysicalOptimizedRiscvAiChip
# 自动生成，用于解决物理验证违例

# 时钟约束
create_clock -name "clk" -period 10.0 [get_ports clock]
set_clock_uncertainty 0.1 [get_clocks clk]

# 输入延迟约束
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# 输出延迟约束
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# 负载约束
set_load 0.1 [all_outputs]

# 驱动强度约束
set_driving_cell -lib_cell BUFX2 [all_inputs]

# 面积约束
set_max_area 0

# 功耗约束
set_max_dynamic_power 100.0
set_max_leakage_power 10.0

# 时序例外
set_false_path -from [get_ports reset]
set_multicycle_path -setup 2 -from [get_pins */matrixMult/*] -to [get_pins */result_reg*]

# DRC约束
set_min_pulse_width -high 0.4 [get_clocks clk]
set_min_pulse_width -low 0.4 [get_clocks clk]
