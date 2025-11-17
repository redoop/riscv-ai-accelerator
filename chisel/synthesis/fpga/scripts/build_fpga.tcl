# Vivado 构建脚本 - RISC-V AI 加速器

# 设置项目参数
set project_name "riscv_ai_accel"
set top_module "fpga_top"
set part "xcvu9p-flgb2104-2-i"
set build_dir "./build"

# 创建必要的目录
puts "创建项目目录..."
file mkdir $build_dir
file mkdir $build_dir/reports
file mkdir $build_dir/checkpoints
file mkdir $build_dir/checkpoints/to_aws

# 创建项目
puts "创建 Vivado 项目..."
create_project $project_name $build_dir -part $part -force

# 添加源文件
puts "添加 RTL 源文件..."

# 添加生成的 Verilog 文件
add_files [glob ../../generated/simple_edgeaisoc/*.sv]

# 添加 FPGA 特定源文件
# add_files [glob ./src/*.sv]

# 添加约束文件
puts "添加约束文件..."
add_files -fileset constrs_1 [glob ./constraints/*.xdc]

# 设置顶层模块
set_property top $top_module [current_fileset]

# 综合设置
puts "配置综合选项..."
set_property strategy "Flow_PerfOptimized_high" [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE "AlternateRoutability" [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# 实现设置
puts "配置实现选项..."
set_property strategy "Performance_ExplorePostRoutePhysOpt" [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE "Explore" [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE "Explore" [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE "Explore" [get_runs impl_1]

# 禁用 write_bitstream 步骤（AWS 只需要 DCP）
set_property STEPS.WRITE_BITSTREAM.IS_ENABLED false [get_runs impl_1]

# 运行综合
puts "开始综合..."
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# 检查综合结果
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "错误：综合失败"
    exit 1
}

# 打开综合设计
open_run synth_1

# 生成综合报告
puts "生成综合报告..."
report_utilization -file $build_dir/reports/utilization_synth.rpt
report_timing_summary -file $build_dir/reports/timing_synth.rpt

# 运行实现（直接使用 TCL 命令，不生成比特流）
puts "开始实现（优化）..."
opt_design -directive Explore

puts "开始布局..."
place_design -directive Explore

puts "开始布线..."
route_design -directive Explore

puts "实现完成！"

# 生成实现报告
puts "生成实现报告..."
report_utilization -file $build_dir/reports/utilization_impl.rpt
report_timing_summary -file $build_dir/reports/timing_impl.rpt
report_power -file $build_dir/reports/power.rpt

# 检查时序
set wns [get_property SLACK [get_timing_paths]]
puts "WNS (Worst Negative Slack): $wns"

if {$wns < 0} {
    puts "警告：存在时序违例！"
} else {
    puts "时序收敛成功！"
}

# 生成 DCP 文件（用于 AWS AFI）
# 注意：AWS F1/F2 不需要比特流，只需要 DCP 文件
puts "生成 DCP 文件（用于 AWS AFI）..."

# 保存 DCP 检查点
write_checkpoint -force $build_dir/checkpoints/to_aws/SH_CL_routed.dcp

puts ""
puts "=========================================="
puts "构建完成！"
puts "=========================================="
puts "DCP 文件：$build_dir/checkpoints/to_aws/SH_CL_routed.dcp"
puts "报告目录：$build_dir/reports/"
puts ""
puts "下一步："
puts "1. 上传 DCP 到 S3"
puts "2. 创建 AFI"
puts "3. 等待 AFI 生成完成"
puts "4. 加载 AFI 到 F1/F2 实例"
puts ""
