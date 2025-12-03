# Vivado 综合脚本
# 日期: 2025-12-03
# 目标: ysyxSoC AI 加速器

# 创建项目
create_project ysyxsoc_ai ./vivado_project -part xc7a100tcsg324-1 -force

# 添加所有 Verilog 文件
add_files [glob *.v]
add_files [glob *.sv]

# 设置顶层模块
set_property top ysyxSoCTop [current_fileset]

# 设置 SystemVerilog 支持
set_property file_type SystemVerilog [get_files *.sv]

# 综合设置
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]

# 运行综合
launch_runs synth_1
wait_on_run synth_1

# 打开综合设计
open_run synth_1

# 生成报告
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
report_power -file power.rpt

# 输出网表
write_verilog -force synth_netlist.v

puts "综合完成！"
puts "报告文件:"
puts "  - utilization.rpt"
puts "  - timing.rpt"
puts "  - power.rpt"
puts "  - synth_netlist.v"
