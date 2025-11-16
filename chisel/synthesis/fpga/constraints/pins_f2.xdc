# F2 实例引脚约束 - 简化版本
# 用于综合和实现验证，不生成实际比特流

# 只设置 IOSTANDARD，不设置 PACKAGE_PIN
# 这样可以通过综合和实现，但不会尝试映射到实际引脚

# 所有端口使用 LVCMOS18 标准
set_property IOSTANDARD LVCMOS18 [get_ports clk_main_a0]
set_property IOSTANDARD LVCMOS18 [get_ports rst_main_n]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_bar_addr[*]]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_bar_wdata[*]]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_bar_rdata[*]]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_bar_wen]
set_property IOSTANDARD LVCMOS18 [get_ports pcie_bar_ren]
set_property IOSTANDARD LVCMOS18 [get_ports debug_status[*]]

# 注意：没有设置 PACKAGE_PIN，因此会触发 UCIO-1 警告
# 这个警告将在 TCL 脚本中被降级为 Warning
# 这样可以完成实现流程而不需要实际的引脚分配

# 这个约束文件仅用于验证设计的综合和实现
# 不用于生成实际可用的比特流
