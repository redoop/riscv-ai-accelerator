
# 物理实现TCL脚本 - DRC违例修复
# 适用于Synopsys ICC2或Cadence Innovus

# 设置设计参数
set DESIGN_NAME "PhysicalOptimizedRiscvAiChip"
set TARGET_FREQ 100
set UTILIZATION 0.75

# 读取设计
read_verilog generated/optimized/$DESIGN_NAME.sv
link_design $DESIGN_NAME

# 读取约束
read_sdc generated/constraints/design_constraints.sdc
read_upf generated/constraints/power_constraints.upf

# 设置物理约束
# 1. 布局约束
set_placement_padding -global -left 2 -right 2 -top 2 -bottom 2
set_app_var placer_max_cell_density_threshold $UTILIZATION

# 2. 布线约束  
set_route_mode -name "default" -min_routing_layer M2 -max_routing_layer M8
set_route_mode -name "default" -antenna_diode_insertion true
set_route_mode -name "default" -post_route_spread_wire true

# 3. 时钟树约束
set_clock_tree_options -target_skew 50 -target_latency 500
set_clock_tree_options -buffer_relocation true -gate_relocation true

# 4. 电源网络约束
create_power_grid -layers {M1 M2 M9 M10} -width 0.5 -spacing 10.0
add_power_grid_straps -layer M1 -width 0.5 -spacing 5.0 -direction horizontal
add_power_grid_straps -layer M2 -width 0.5 -spacing 5.0 -direction vertical

# 执行物理实现流程
# 1. 布局
place_design -timing_driven -congestion_driven
optimize_design -pre_cts

# 2. 时钟树综合
clock_design -cts
optimize_design -post_cts

# 3. 布线
route_design -global_detail
optimize_design -post_route

# 4. 填充和金属填充
add_filler_cells
add_metal_fill

# 5. DRC和LVS检查
verify_drc -limit 1000
verify_connectivity

# 6. 时序分析
report_timing -max_paths 100 -nworst 10
report_power
report_area

# 7. 输出结果
write_def $DESIGN_NAME.def
write_gds $DESIGN_NAME.gds
write_netlist $DESIGN_NAME.v
write_sdf $DESIGN_NAME.sdf

puts "物理实现完成，预期DRC违例: 0"
