# OpenROAD 完整 P&R 流程

puts "=========================================="
puts "OpenROAD 完整 P&R 流程"
puts "=========================================="
puts ""
puts "设计: asic_top"
puts "目标频率: 100 MHz"
puts "工艺: ICS55 55nm"
puts ""

set start_time [clock seconds]

# 1. Floorplan
puts "\nStep 1: 执行 Floorplan..."
source scripts/1_floorplan.tcl

# 2. Placement
puts "\nStep 2: 执行 Placement..."
source scripts/2_placement.tcl

# 3. CTS
puts "\nStep 3: 执行 CTS..."
source scripts/3_cts.tcl

# 4. Routing
puts "\nStep 4: 执行 Routing..."
source scripts/4_routing.tcl

set end_time [clock seconds]
set elapsed [expr $end_time - $start_time]

puts "\n=========================================="
puts "P&R 流程完成"
puts "=========================================="
puts "总耗时: $elapsed 秒"
puts ""
puts "输出文件:"
puts "  - Floorplan: results/1_floorplan.def"
puts "  - Placement: results/2_placement.def"
puts "  - CTS: results/3_cts.def, results/3_cts.v"
puts "  - Routing: results/4_routing.def, results/4_routing.v"
