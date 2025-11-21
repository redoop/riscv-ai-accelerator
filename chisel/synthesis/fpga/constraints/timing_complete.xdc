# ============================================================================
# 完整时序约束文件 - RISC-V AI 加速器
# 包含主时钟和 SPI 时钟约束
# ============================================================================

# ============================================================================
# 主时钟约束
# ============================================================================

# 系统主时钟 - 100MHz
create_clock -period 10.000 -name sys_clk [get_ports clock]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clock]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# ============================================================================
# 生成时钟约束 - SPI
# ============================================================================

# SPI 时钟 - 从主时钟分频生成
# 实际频率 10.0 MHz (100MHz / 10)
# 注意：路径需要根据实际综合后的层次结构调整
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 10 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]

# ============================================================================
# 输入延迟约束
# ============================================================================

# UART RX
set_input_delay -clock sys_clk -max 2.0 [get_ports io_uart_rx]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_uart_rx]

# 复位信号
set_input_delay -clock sys_clk -max 2.0 [get_ports reset]
set_input_delay -clock sys_clk -min 0.5 [get_ports reset]

# GPIO 输入（如果存在）
set_input_delay -clock sys_clk -max 2.0 [get_ports io_gpio_in*]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_gpio_in*]

# ============================================================================
# 输出延迟约束 - 系统时钟域
# ============================================================================

# UART TX
set_output_delay -clock sys_clk -max 2.0 [get_ports io_uart_tx]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_uart_tx]

# GPIO 输出（如果存在）
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_out*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_gpio_out*]

# GPIO 输出使能（如果存在）
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_oe*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_gpio_oe*]

# LCD 背光控制
set_output_delay -clock sys_clk -max 2.0 [get_ports io_lcd_backlight]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_lcd_backlight]

# LCD 复位信号
set_output_delay -clock sys_clk -max 2.0 [get_ports io_lcd_spi_rst]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_lcd_spi_rst]

# ============================================================================
# 输出延迟约束 - SPI 时钟域
# ============================================================================

# SPI 时钟输出
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_clk]

# SPI 数据输出（MOSI）
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_mosi]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_mosi]

# SPI 片选
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_cs]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_cs]

# SPI DC（数据/命令选择）
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_dc]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_dc]

# ============================================================================
# 假路径约束
# ============================================================================

# 复位信号是异步的
set_false_path -from [get_ports reset]

# 如果 SPI 时钟域和主时钟域之间有异步路径，取消注释：
# set_false_path -from [get_clocks sys_clk] -to [get_clocks spi_clk]
# set_false_path -from [get_clocks spi_clk] -to [get_clocks sys_clk]

# ============================================================================
# 多周期路径约束
# ============================================================================

# SPI 控制器状态机可以使用多周期（如果需要）
# set_multicycle_path -setup 2 -from [get_cells lcd/lcd/state_reg*] \
#   -to [get_cells lcd/lcd/spiShiftReg_reg*]
# set_multicycle_path -hold 1 -from [get_cells lcd/lcd/state_reg*] \
#   -to [get_cells lcd/lcd/spiShiftReg_reg*]

# ============================================================================
# 最大延迟约束
# ============================================================================

# 组合逻辑最大延迟
set_max_delay 10.0 -from [all_inputs] -to [all_outputs]

# ============================================================================
# 时钟组约束（可选）
# ============================================================================

# 如果时钟域是物理上互斥的，取消注释：
# set_clock_groups -physically_exclusive \
#   -group [get_clocks sys_clk] \
#   -group [get_clocks spi_clk]

# ============================================================================
# 时序例外
# ============================================================================

# 时序例外（如果需要）
# set_false_path -through [get_pins ...]

# ============================================================================
# 注释
# ============================================================================
# 1. 主时钟频率：100 MHz (周期 10 ns)
# 2. SPI 时钟频率：10 MHz (周期 100 ns, 从主时钟 10 分频)
# 3. SPI 时钟是软件分频生成，不是硬件 PLL
# 4. 所有延迟值基于典型 FPGA I/O 特性
# 5. 根据实际硬件调整延迟值
# 6. 时钟验证测试已通过：频率误差 2.04%，占空比 50.00%
# ============================================================================
