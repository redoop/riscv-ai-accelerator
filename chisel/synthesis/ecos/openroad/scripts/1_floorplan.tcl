# 1. Floorplan - 布图规划

cd /opt/github/riscv-ai-accelerator/chisel/synthesis/ecos/openroad
source config.tcl

puts "=========================================="
puts "Step 1: Floorplan"
puts "=========================================="

# 读取 LEF
puts "\n读取 LEF 文件..."
read_lef $TECH_LEF
read_lef $IO_LEF

# 读取 Liberty
puts "\n读取 Liberty 库..."
foreach lib $LIB_FILES {
    read_liberty $lib
}

# 读取网表
puts "\n读取网表..."
read_verilog $VERILOG_FILE
link_design $DESIGN_NAME

# 创建时钟
puts "\n创建时钟约束..."
create_clock -name sys_clk -period $CLOCK_PERIOD [get_ports $CLOCK_PORT]
set_clock_uncertainty $CLOCK_UNCERTAINTY [get_clocks sys_clk]

# 布图规划（使用固定尺寸）
puts "\n执行布图规划..."
initialize_floorplan \
    -die_area $DIE_AREA \
    -core_area $CORE_AREA

# 插入电源网格
puts "\n插入电源网格..."
add_global_connection -net VDD -pin_pattern {^VDD$} -power
add_global_connection -net VSS -pin_pattern {^VSS$} -ground

# 保存结果
puts "\n保存结果..."
write_def $RESULTS_DIR/1_floorplan.def
write_db $RESULTS_DIR/1_floorplan.odb

puts "\n✅ Floorplan 完成"
