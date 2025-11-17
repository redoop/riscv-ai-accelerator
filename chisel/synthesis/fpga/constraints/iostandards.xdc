# I/O 标准约束 - AWS F1/F2 FPGA
# 注意：AWS Shell-CL 接口不需要物理引脚位置，但需要指定 I/O 标准

# 为所有端口设置默认 I/O 标准（LVCMOS18 是常用标准）
set_property IOSTANDARD LVCMOS18 [get_ports clock]
set_property IOSTANDARD LVCMOS18 [get_ports reset]

# UART 接口
set_property IOSTANDARD LVCMOS18 [get_ports io_uart_tx]
set_property IOSTANDARD LVCMOS18 [get_ports io_uart_rx]

# LCD SPI 接口
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_spi_clk]
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_spi_mosi]
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_spi_cs]
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_spi_dc]
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_spi_rst]
set_property IOSTANDARD LVCMOS18 [get_ports io_lcd_backlight]

# GPIO 接口（32-bit）
set_property IOSTANDARD LVCMOS18 [get_ports io_gpio_out[*]]
set_property IOSTANDARD LVCMOS18 [get_ports io_gpio_in[*]]

# 中断和陷阱信号
set_property IOSTANDARD LVCMOS18 [get_ports io_trap]
set_property IOSTANDARD LVCMOS18 [get_ports io_compact_irq]
set_property IOSTANDARD LVCMOS18 [get_ports io_bitnet_irq]
set_property IOSTANDARD LVCMOS18 [get_ports io_uart_tx_irq]
set_property IOSTANDARD LVCMOS18 [get_ports io_uart_rx_irq]

# 对于 AWS F1/F2，这些端口实际上会通过 Shell-CL 接口连接
# 物理实现由 AWS Shell 处理，这里只需要满足 DRC 要求
