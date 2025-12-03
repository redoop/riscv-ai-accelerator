# 配置验证脚本

source config.tcl

puts "=========================================="
puts "配置验证"
puts "=========================================="

puts "\n检查文件..."

# 检查 LEF
if {[file exists $TECH_LEF]} {
    puts "✅ Tech LEF: $TECH_LEF"
} else {
    puts "❌ Tech LEF 不存在: $TECH_LEF"
}

if {[file exists $IO_LEF]} {
    puts "✅ IO LEF: $IO_LEF"
} else {
    puts "❌ IO LEF 不存在: $IO_LEF"
}

# 检查 Liberty
foreach lib $LIB_FILES {
    if {[file exists $lib]} {
        puts "✅ Liberty: [file tail $lib]"
    } else {
        puts "❌ Liberty 不存在: $lib"
    }
}

# 检查网表
if {[file exists $VERILOG_FILE]} {
    puts "✅ Verilog: $VERILOG_FILE"
    set size [file size $VERILOG_FILE]
    puts "   大小: [expr $size / 1024 / 1024] MB"
} else {
    puts "❌ Verilog 不存在: $VERILOG_FILE"
}

puts "\n配置参数:"
puts "  设计名称: $DESIGN_NAME"
puts "  时钟端口: $CLOCK_PORT"
puts "  时钟周期: $CLOCK_PERIOD ns (100 MHz)"
puts "  芯片尺寸: $DIE_AREA um"
puts "  核心区域: $CORE_AREA um"
puts "  利用率: $CORE_UTILIZATION"

puts "\n✅ 配置验证完成"
puts "\n下一步:"
puts "  ./run.sh all"
