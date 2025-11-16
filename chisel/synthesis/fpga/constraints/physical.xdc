# 物理约束文件 - RISC-V AI 加速器

# Pblock 约束（可选，用于控制布局）
# create_pblock pblock_accel
# add_cells_to_pblock [get_pblocks pblock_accel] [get_cells -hierarchical -filter {NAME =~ "*accel*"}]
# resize_pblock [get_pblocks pblock_accel] -add {SLICE_X0Y0:SLICE_X50Y50}

# 布局策略
set_property STRATEGY Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# 关键路径优化
# set_property HIGH_PRIORITY true [get_nets critical_path_net]

# 时钟区域约束
# set_property CLOCK_REGION X0Y0 [get_cells clock_gen_inst]

# 功耗优化
set_property POWER_OPT_DESIGN true [current_design]

# 布线优化
set_property ROUTE.DELAY_OPTIMIZATION true [current_design]
