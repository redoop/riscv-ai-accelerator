# 4. Routing - 布线

cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
source config.tcl

puts "=========================================="
puts "Step 4: Routing"
puts "=========================================="

# 读取 CTS 结果
puts "\n读取 CTS 结果..."
read_lef $TECH_LEF
read_lef $IO_LEF

foreach lib $LIB_FILES {
    read_liberty $lib
}

read_db $RESULTS_DIR/3_cts.odb

# 全局布线
puts "\n执行全局布线..."
global_route

# 详细布线
puts "\n执行详细布线..."
detailed_route

# 保存结果
puts "\n保存结果..."
write_def $RESULTS_DIR/4_routing.def
write_db $RESULTS_DIR/4_routing.odb
write_verilog $RESULTS_DIR/4_routing.v

# 最终报告
puts "\n最终时序报告:"
report_checks -path_delay max
report_worst_slack
report_tns

puts "\n✅ Routing 完成"
