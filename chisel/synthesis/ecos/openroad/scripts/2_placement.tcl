# 2. Placement - 布局

cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
source config.tcl

puts "=========================================="
puts "Step 2: Placement"
puts "=========================================="

# 读取 Floorplan 结果
puts "\n读取 Floorplan 结果..."
read_lef $TECH_LEF
read_lef $IO_LEF

foreach lib $LIB_FILES {
    read_liberty $lib
}

read_db $RESULTS_DIR/1_floorplan.odb

# 全局布局
puts "\n执行全局布局..."
global_placement -density $PLACE_DENSITY

# 详细布局
puts "\n执行详细布局..."
detailed_placement

# 保存结果
puts "\n保存结果..."
write_def $RESULTS_DIR/2_placement.def
write_db $RESULTS_DIR/2_placement.odb

puts "\n✅ Placement 完成"
