# ============================================================================
# 基本时序约束文件 (SDC 格式) - RISC-V AI 加速器
# 简化版本，用于快速开始
# ============================================================================

# 主时钟 - 100MHz
create_clock -name sys_clk -period 10.000 [get_ports clock]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# SPI 生成时钟 - 10MHz
create_generated_clock -name spi_clk \
  -source [get_ports clock] \
  -divide_by 10 \
  [get_pins -hierarchical *spiClkReg*/Q]

# 输入延迟
set_input_delay -clock sys_clk -max 2.0 [get_ports io_uart_rx]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_uart_rx]
set_input_delay -clock sys_clk -max 2.0 [get_ports reset]
set_input_delay -clock sys_clk -min 0.5 [get_ports reset]

# 输出延迟 - 系统时钟域
set_output_delay -clock sys_clk -max 2.0 [get_ports io_uart_tx]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_uart_tx]

# 输出延迟 - SPI 时钟域
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_mosi]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_mosi]

# 假路径
set_false_path -from [get_ports reset]

# 设计规则
set_max_fanout 16 [current_design]
set_max_transition 0.5 [current_design]
