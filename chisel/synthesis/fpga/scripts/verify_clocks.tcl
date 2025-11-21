# ============================================================================
# 时钟约束验证脚本
# ============================================================================

puts "=========================================="
puts "时钟约束验证脚本"
puts "=========================================="

# 创建报告目录
file mkdir reports

# ============================================================================
# 1. 检查主时钟
# ============================================================================
puts "\n\[1/7\] 检查主时钟..."
set main_clk [get_clocks -quiet sys_clk]
if {[llength $main_clk] == 0} {
    puts "ERROR: 主时钟 sys_clk 未定义!"
    puts "请检查约束文件中是否有: create_clock -period 10.000 -name sys_clk \[get_ports clock\]"
    exit 1
} else {
    set period [get_property PERIOD $main_clk]
    set freq [expr {1000.0 / $period}]
    puts "✓ 主时钟: sys_clk"
    puts "  周期: $period ns"
    puts "  频率: [format %.2f $freq] MHz"
    
    if {abs($period - 10.0) > 0.1} {
        puts "WARNING: 主时钟周期应该是 10 ns (100 MHz)，当前是 $period ns"
    }
}

# ============================================================================
# 2. 检查 SPI 生成时钟
# ============================================================================
puts "\n\[2/7\] 检查 SPI 生成时钟..."
set spi_clk [get_clocks -quiet spi_clk]
if {[llength $spi_clk] == 0} {
    puts "INFO: SPI 时钟 spi_clk 未定义为生成时钟"
    puts "  这是正常的，SPI 时钟是通过软件分频生成的"
} else {
    set period [get_property PERIOD $spi_clk]
    set freq [expr {1000.0 / $period}]
    puts "✓ SPI 时钟: spi_clk"
    puts "  周期: $period ns"
    puts "  频率: [format %.2f $freq] MHz"
    
    set expected_period 100.0
    if {abs($period - $expected_period) > 10.0} {
        puts "WARNING: SPI 时钟周期应该约为 $expected_period ns (10 MHz)，当前是 $period ns"
    }
}

# ============================================================================
# 3. 检查 SPI 时钟生成器
# ============================================================================
puts "\n\[3/7\] 检查 SPI 时钟生成器..."
set spi_clk_cells [get_cells -quiet -hierarchical -filter {NAME =~ *spiClkReg*}]
if {[llength $spi_clk_cells] == 0} {
    puts "WARNING: 未找到 SPI 时钟寄存器 (spiClkReg)"
    puts "  可能的原因:"
    puts "  1. 设计尚未综合"
    puts "  2. 寄存器名称不同"
    puts "  3. SPI 模块未实例化"
} else {
    puts "✓ 找到 [llength $spi_clk_cells] 个 SPI 时钟相关寄存器:"
    foreach cell $spi_clk_cells {
        puts "  - [get_property NAME $cell]"
    }
}

# ============================================================================
# 4. 检查 SPI 输出端口
# ============================================================================
puts "\n\[4/7\] 检查 SPI 输出端口..."
set spi_ports [get_ports -quiet io_lcd_spi_*]
if {[llength $spi_ports] == 0} {
    puts "WARNING: 未找到 SPI 端口 (io_lcd_spi_*)"
    puts "  可能的原因:"
    puts "  1. 端口名称不同"
    puts "  2. LCD 模块未连接到顶层"
} else {
    puts "✓ 找到 [llength $spi_ports] 个 SPI 端口:"
    foreach port $spi_ports {
        set port_name [get_property NAME $port]
        puts "  - $port_name"
        
        # 检查输出延迟约束
        set output_delays [get_property OUTPUT_DELAY $port]
        if {[llength $output_delays] == 0} {
            puts "    WARNING: 端口 $port_name 没有输出延迟约束"
        } else {
            puts "    ✓ 有输出延迟约束"
        }
    }
}

# ============================================================================
# 5. 检查未约束的路径
# ============================================================================
puts "\n\[5/7\] 检查未约束的路径..."
set unconstrained_paths [get_timing_paths -max_paths 10 -quiet \
  -filter {STARTPOINT_CLOCK == "" || ENDPOINT_CLOCK == ""}]

if {[llength $unconstrained_paths] > 0} {
    puts "WARNING: 发现 [llength $unconstrained_paths] 条未约束的路径"
    puts "  前 5 条未约束路径:"
    set count 0
    foreach path $unconstrained_paths {
        if {$count >= 5} break
        set startpoint [get_property STARTPOINT_PIN $path]
        set endpoint [get_property ENDPOINT_PIN $path]
        puts "  $count: $startpoint -> $endpoint"
        incr count
    }
    puts "  详细信息请查看: reports/unconstrained_paths.txt"
    report_timing -of $unconstrained_paths -max_paths 10 \
      -file reports/unconstrained_paths.txt
} else {
    puts "✓ 所有路径都已约束"
}

# ============================================================================
# 6. 检查时序违例
# ============================================================================
puts "\n\[6/7\] 检查时序违例..."

# Setup 时序
set setup_paths [get_timing_paths -max_paths 1 -setup -quiet]
if {[llength $setup_paths] > 0} {
    set wns [get_property SLACK [lindex $setup_paths 0]]
    puts "  Setup WNS (Worst Negative Slack): [format %.3f $wns] ns"
    
    if {$wns < 0} {
        puts "  ✗ Setup 时序违例!"
        puts "  最差路径详情请查看: reports/worst_setup_path.txt"
        report_timing -of $setup_paths -file reports/worst_setup_path.txt
    } else {
        puts "  ✓ Setup 时序满足"
    }
} else {
    puts "  INFO: 无 Setup 时序路径"
}

# Hold 时序
set hold_paths [get_timing_paths -max_paths 1 -hold -quiet]
if {[llength $hold_paths] > 0} {
    set whs [get_property SLACK [lindex $hold_paths 0]]
    puts "  Hold WHS (Worst Hold Slack): [format %.3f $whs] ns"
    
    if {$whs < 0} {
        puts "  ✗ Hold 时序违例!"
        puts "  最差路径详情请查看: reports/worst_hold_path.txt"
        report_timing -of $hold_paths -file reports/worst_hold_path.txt
    } else {
        puts "  ✓ Hold 时序满足"
    }
} else {
    puts "  INFO: 无 Hold 时序路径"
}

# ============================================================================
# 7. 生成详细报告
# ============================================================================
puts "\n\[7/7\] 生成详细报告..."

# 时钟报告
if {[catch {report_clocks -file reports/clock_report.txt}]} {
    puts "  WARNING: 无法生成时钟报告"
} else {
    puts "  ✓ 时钟报告: reports/clock_report.txt"
}

# 时钟网络报告
if {[catch {report_clock_networks -file reports/clock_networks.txt}]} {
    puts "  WARNING: 无法生成时钟网络报告"
} else {
    puts "  ✓ 时钟网络报告: reports/clock_networks.txt"
}

# 时序摘要
if {[catch {report_timing_summary -file reports/timing_summary.txt}]} {
    puts "  WARNING: 无法生成时序摘要"
} else {
    puts "  ✓ 时序摘要: reports/timing_summary.txt"
}

# 约束检查
if {[catch {check_timing -file reports/check_timing.txt}]} {
    puts "  WARNING: 无法生成约束检查报告"
} else {
    puts "  ✓ 约束检查: reports/check_timing.txt"
}

# ============================================================================
# 总结
# ============================================================================
puts "\n=========================================="
puts "验证完成!"
puts "=========================================="
puts "详细报告保存在 build/reports/ 目录"
puts ""
puts "下一步:"
puts "  1. 查看报告文件"
puts "  2. 修复任何警告或错误"
puts "  3. 重新运行综合/实现"
puts "=========================================="
