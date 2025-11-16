# 引脚约束文件 - AWS F1 FPGA

# 注意：AWS F1 使用 Shell-CL 接口，实际引脚由 AWS Shell 管理
# 这里定义的是逻辑端口到 Shell 接口的映射

# 时钟和复位（来自 AWS Shell）
# set_property PACKAGE_PIN XX [get_ports clock]
# set_property IOSTANDARD LVCMOS18 [get_ports clock]

# set_property PACKAGE_PIN XX [get_ports reset]
# set_property IOSTANDARD LVCMOS18 [get_ports reset]

# UART 接口（通过 PCIe 或 USB-UART 桥接）
# 实际实现中，UART 会映射到 PCIe BAR 寄存器
# set_property PACKAGE_PIN XX [get_ports io_uart_tx]
# set_property IOSTANDARD LVCMOS18 [get_ports io_uart_tx]

# set_property PACKAGE_PIN XX [get_ports io_uart_rx]
# set_property IOSTANDARD LVCMOS18 [get_ports io_uart_rx]

# GPIO 接口（映射到 PCIe BAR 寄存器）
# 32-bit GPIO 输出
# for {set i 0} {$i < 32} {incr i} {
#     set_property PACKAGE_PIN XX [get_ports io_gpio_out[$i]]
#     set_property IOSTANDARD LVCMOS18 [get_ports io_gpio_out[$i]]
# }

# 32-bit GPIO 输入
# for {set i 0} {$i < 32} {incr i} {
#     set_property PACKAGE_PIN XX [get_ports io_gpio_in[$i]]
#     set_property IOSTANDARD LVCMOS18 [get_ports io_gpio_in[$i]]
# }

# 32-bit GPIO 输出使能
# for {set i 0} {$i < 32} {incr i} {
#     set_property PACKAGE_PIN XX [get_ports io_gpio_oe[$i]]
#     set_property IOSTANDARD LVCMOS18 [get_ports io_gpio_oe[$i]]
# }

# 注意：在 AWS F1 上，大部分 IO 通过 PCIe 接口访问
# 物理引脚约束由 AWS Shell 提供的 XDC 文件定义
