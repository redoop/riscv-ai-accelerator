# F2 实例时序约束 - 简化版本

# 主时钟定义 - 100 MHz
create_clock -period 10.000 -name clk_main -waveform {0.000 5.000} [get_ports clk_main_a0]

# 时钟不确定性
set_clock_uncertainty 0.200 [get_clocks clk_main]

# 输入延迟约束
set_input_delay -clock [get_clocks clk_main] -min 1.000 [get_ports rst_main_n]
set_input_delay -clock [get_clocks clk_main] -max 3.000 [get_ports rst_main_n]

set_input_delay -clock [get_clocks clk_main] -min 1.000 [get_ports pcie_bar_addr[*]]
set_input_delay -clock [get_clocks clk_main] -max 3.000 [get_ports pcie_bar_addr[*]]

set_input_delay -clock [get_clocks clk_main] -min 1.000 [get_ports pcie_bar_wdata[*]]
set_input_delay -clock [get_clocks clk_main] -max 3.000 [get_ports pcie_bar_wdata[*]]

set_input_delay -clock [get_clocks clk_main] -min 1.000 [get_ports pcie_bar_wen]
set_input_delay -clock [get_clocks clk_main] -max 3.000 [get_ports pcie_bar_wen]

set_input_delay -clock [get_clocks clk_main] -min 1.000 [get_ports pcie_bar_ren]
set_input_delay -clock [get_clocks clk_main] -max 3.000 [get_ports pcie_bar_ren]

# 输出延迟约束
set_output_delay -clock [get_clocks clk_main] -min 1.000 [get_ports pcie_bar_rdata[*]]
set_output_delay -clock [get_clocks clk_main] -max 3.000 [get_ports pcie_bar_rdata[*]]

set_output_delay -clock [get_clocks clk_main] -min 1.000 [get_ports debug_status[*]]
set_output_delay -clock [get_clocks clk_main] -max 3.000 [get_ports debug_status[*]]

# 复位路径 - 异步复位，设置为 false path
set_false_path -from [get_ports rst_main_n]

# 多周期路径（如果需要）
# set_multicycle_path -setup 2 -from [get_pins ...] -to [get_pins ...]
