# 3. CTS - 时钟树综合

cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
source config.tcl

puts "=========================================="
puts "Step 3: Clock Tree Synthesis"
puts "=========================================="

# 读取 Placement 结果
puts "\n读取 Placement 结果..."
read_lef $TECH_LEF
read_lef $IO_LEF

foreach lib $LIB_FILES {
    read_liberty $lib
}

read_db $RESULTS_DIR/2_placement.odb

# 时钟树综合
puts "\n执行时钟树综合..."
clock_tree_synthesis \
    -root_buf $CTS_ROOT_BUF \
    -buf_list $CTS_BUF_LIST \
    -wire_unit $CTS_WIRE_UNIT

# 保存结果
puts "\n保存结果..."
write_def $RESULTS_DIR/3_cts.def
write_db $RESULTS_DIR/3_cts.odb
write_verilog $RESULTS_DIR/3_cts.v

# 报告
puts "\n时钟树报告:"
report_clock_skew

puts "\n时序报告:"
report_checks -path_delay max
report_worst_slack

puts "\n✅ CTS 完成"
