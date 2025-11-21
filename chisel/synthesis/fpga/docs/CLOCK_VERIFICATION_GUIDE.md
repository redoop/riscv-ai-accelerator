# 时钟约束验证指南

## 验证流程概览

```
1. 仿真验证 (Chisel/Verilog)
   ↓
2. 综合后验证 (Vivado Synthesis)
   ↓
3. 实现后验证 (Vivado Implementation)
   ↓
4. 硬件验证 (FPGA 实测)
```

## 1. 仿真验证

### 1.1 Chisel 仿真验证 SPI 时钟频率

```bash
cd chisel
sbt "testOnly riscv.ai.TFTLCDTest"
```

**验证内容**:
- SPI 时钟分频是否正确
- SPI 时钟占空比
- SPI 数据与时钟的相位关系

**查看波形**:
```bash
# 生成 VCD 文件
sbt "testOnly riscv.ai.TFTLCDTest -- -DwriteVcd=1"

# 使用 GTKWave 查看
gtkwave test_run_dir/TFTLCD_should_send_SPI_command/TFTLCD.vcd
```

**关键信号**:
- `clock` - 主时钟 (50 MHz)
- `lcd.spiClkReg` - SPI 时钟寄存器
- `lcd.spiCounter` - 分频计数器
- `io_lcd_spi_clk` - SPI 时钟输出

**预期结果**:
```
主时钟周期: 20 ns
SPI 时钟周期: ~120 ns (6 个主时钟周期)
分频比: 6
频率: 50MHz / 6 ≈ 8.33 MHz
```

### 1.2 创建验证测试


```scala
// chisel/src/test/scala/ClockVerificationTest.scala
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import riscv.ai.peripherals.TFTLCD

class ClockVerificationTest extends AnyFlatSpec with ChiselScalatestTester {
  
  "SPI Clock" should "have correct frequency" in {
    test(new TFTLCD(clockFreq = 50000000, spiFreq = 10000000))
      .withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      
      dut.clock.setTimeout(0)
      
      // 测量 SPI 时钟周期
      var spiClkCycles = 0
      var mainClkCycles = 0
      var lastSpiClk = false
      
      // 等待 SPI 时钟翻转
      while (spiClkCycles < 10) {
        val currentSpiClk = dut.io.spi_clk.peek().litToBoolean
        
        if (currentSpiClk && !lastSpiClk) {
          println(s"SPI clock rising edge at main clock cycle $mainClkCycles")
          spiClkCycles += 1
        }
        
        lastSpiClk = currentSpiClk
        mainClkCycles += 1
        dut.clock.step(1)
      }
      
      val avgPeriod = mainClkCycles.toDouble / spiClkCycles
      val frequency = 50000000.0 / avgPeriod
      
      println(f"Average SPI clock period: $avgPeriod%.2f main clock cycles")
      println(f"Measured SPI frequency: ${frequency/1000000}%.2f MHz")
      
      // 验证频率在合理范围内 (8-10 MHz)
      assert(frequency >= 8000000 && frequency <= 10000000, 
        s"SPI frequency $frequency Hz out of range")
    }
  }
  
  "SPI Clock" should "have ~50% duty cycle" in {
    test(new TFTLCD(clockFreq = 50000000, spiFreq = 10000000)) { dut =>
      
      dut.clock.setTimeout(0)
      
      var highCycles = 0
      var lowCycles = 0
      var totalCycles = 0
      val maxCycles = 1000
      
      while (totalCycles < maxCycles) {
        if (dut.io.spi_clk.peek().litToBoolean) {
          highCycles += 1
        } else {
          lowCycles += 1
        }
        totalCycles += 1
        dut.clock.step(1)
      }
      
      val dutyCycle = highCycles.toDouble / totalCycles * 100
      println(f"SPI clock duty cycle: $dutyCycle%.2f%%")
      
      // 验证占空比在 45-55% 范围内
      assert(dutyCycle >= 45 && dutyCycle <= 55,
        s"Duty cycle $dutyCycle% out of range")
    }
  }
}
```

运行验证测试：
```bash
cd chisel
sbt "testOnly riscv.ai.ClockVerificationTest"
```

## 2. Vivado 综合后验证

### 2.1 检查时钟报告

在 Vivado TCL Console 中运行：

```tcl
# 打开综合后的设计
open_run synth_1

# 1. 检查所有时钟
report_clocks

# 预期输出:
# Clock Name    Period(ns)  Frequency(MHz)
# sys_clk       20.000      50.000
# spi_clk       120.000     8.333

# 2. 检查时钟网络
report_clock_networks

# 3. 检查生成时钟
get_generated_clocks

# 4. 检查时钟树
report_clock_utilization

# 5. 检查未约束的路径
report_timing -unconstrained

# 预期: 应该没有未约束的路径

# 6. 检查时钟域交互
report_clock_interaction

# 7. 检查时序摘要
report_timing_summary
```

### 2.2 验证 SPI 时钟路径

```tcl
# 查找 SPI 时钟生成器
get_cells -hierarchical -filter {NAME =~ *spiClkReg*}

# 预期输出类似:
# lcd/lcd/spiClkReg_reg

# 检查 SPI 时钟扇出
report_clock_networks -name spi_clk

# 查看 SPI 时钟驱动的寄存器
get_cells -of [get_pins -leaf -of [get_nets -of [get_pins lcd/lcd/spiClkReg_reg/Q]]]

# 检查 SPI 相关的时序路径
report_timing -from [get_clocks sys_clk] -to [get_ports io_lcd_spi_*]
```

### 2.3 创建验证 TCL 脚本

```tcl
# chisel/synthesis/fpga/scripts/verify_clocks.tcl

puts "=========================================="
puts "时钟约束验证脚本"
puts "=========================================="

# 1. 检查主时钟
puts "\n[1] 检查主时钟..."
set main_clk [get_clocks -quiet sys_clk]
if {[llength $main_clk] == 0} {
    puts "ERROR: 主时钟 sys_clk 未定义!"
    exit 1
} else {
    set period [get_property PERIOD $main_clk]
    set freq [expr {1000.0 / $period}]
    puts "✓ 主时钟: sys_clk"
    puts "  周期: $period ns"
    puts "  频率: [format %.2f $freq] MHz"
    
    if {abs($period - 20.0) > 0.1} {
        puts "WARNING: 主时钟周期应该是 20 ns (50 MHz)"
    }
}

# 2. 检查 SPI 生成时钟
puts "\n[2] 检查 SPI 生成时钟..."
set spi_clk [get_clocks -quiet spi_clk]
if {[llength $spi_clk] == 0} {
    puts "WARNING: SPI 时钟 spi_clk 未定义"
    puts "  这可能是正常的，如果 SPI 时钟是软件生成的"
} else {
    set period [get_property PERIOD $spi_clk]
    set freq [expr {1000.0 / $period}]
    puts "✓ SPI 时钟: spi_clk"
    puts "  周期: $period ns"
    puts "  频率: [format %.2f $freq] MHz"
}

# 3. 检查 SPI 时钟生成器
puts "\n[3] 检查 SPI 时钟生成器..."
set spi_clk_reg [get_cells -quiet -hierarchical -filter {NAME =~ *spiClkReg*}]
if {[llength $spi_clk_reg] == 0} {
    puts "ERROR: 未找到 SPI 时钟寄存器!"
} else {
    puts "✓ 找到 SPI 时钟寄存器:"
    foreach cell $spi_clk_reg {
        puts "  - [get_property NAME $cell]"
    }
}

# 4. 检查未约束的路径
puts "\n[4] 检查未约束的路径..."
set unconstrained [get_timing_paths -max_paths 10 -filter {STARTPOINT_CLOCK == "" || ENDPOINT_CLOCK == ""}]
if {[llength $unconstrained] > 0} {
    puts "WARNING: 发现 [llength $unconstrained] 条未约束的路径"
    report_timing -of $unconstrained -max_paths 5
} else {
    puts "✓ 所有路径都已约束"
}

# 5. 检查时序违例
puts "\n[5] 检查时序违例..."
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -hold]]

puts "  Setup WNS: [format %.3f $wns] ns"
puts "  Hold WHS: [format %.3f $whs] ns"

if {$wns < 0} {
    puts "ERROR: Setup 时序违例!"
}
if {$whs < 0} {
    puts "ERROR: Hold 时序违例!"
}

# 6. 检查 SPI 输出端口
puts "\n[6] 检查 SPI 输出端口..."
set spi_ports [get_ports -quiet io_lcd_spi_*]
if {[llength $spi_ports] == 0} {
    puts "WARNING: 未找到 SPI 端口"
} else {
    puts "✓ 找到 [llength $spi_ports] 个 SPI 端口:"
    foreach port $spi_ports {
        puts "  - [get_property NAME $port]"
        
        # 检查输出延迟约束
        set delays [get_property OUTPUT_DELAY $port]
        if {[llength $delays] == 0} {
            puts "    WARNING: 没有输出延迟约束"
        }
    }
}

# 7. 生成报告
puts "\n[7] 生成详细报告..."
report_clocks -file reports/clock_report.txt
report_timing_summary -file reports/timing_summary.txt
report_clock_networks -file reports/clock_networks.txt

puts "\n=========================================="
puts "验证完成!"
puts "详细报告保存在 reports/ 目录"
puts "=========================================="
```

运行验证脚本：
```bash
cd chisel/synthesis/fpga
vivado -mode batch -source scripts/verify_clocks.tcl
```

## 3. 实现后验证

### 3.1 检查实现后时序

```tcl
# 打开实现后的设计
open_run impl_1

# 1. 时序摘要
report_timing_summary -delay_type min_max -report_unconstrained \
  -check_timing_verbose -max_paths 10 \
  -input_pins -routable_nets \
  -file reports/timing_summary_impl.txt

# 2. 检查最差路径
report_timi_paths 10 -nworst 1 -delay_type max \
  -sort_by slack -file reports/worst_setup_paths.txt

report_timing -max_paths 10 -ndelay_type min \
  -sort_by slack -file reports/worst_hold_paths.txt

# 3. 检查 SPI 相关路径
report_timing -from [get_clocks sys_clk] \
  -to [get_ports io_lcd_spi_*] \  -/spi_timing.txt钟偏斜
report_clock_utilization -file reports/clock_utilization.txt

# 5. 检查功耗
report_power -file reports/power.txt
```

### 3.2 验证检查清单

创建自动化检查脚本：

```tcl
# chisel/syng {} {
    puts "检查时序..."
    
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
    set whs [get_property SLACK [get_timing_paths -max_paths 1 -hold]]
    
    set pass 1
    
    if {$wns < 0} {
        puts "✗ Setup 时序违例: WNS = $wns ns"
        set pass 0
    } else {
        puts "✓ Setup 时序满足: WNS = $wns ns"
    }
    
    if {$whs < 0} {
        puts "✗ Hold 时序违例: WHS = $whs ns"
        set pass 0
    } else {
        puts "✓ Hold 时序满足: WHS = $whs ns"
    }
    
    return $pass
}

proc check_clocks {} {
    puts "检查时钟..."
    
    set clocks [get_clocks]
    puts "找到 [llength $clocks] 个时钟:"
    
    foreach clk $clocks {
        set name [get_property NAME $clk]
        set period [get_property PERIOD $clk]
        set freq [expr {1000.0 / $period}]
        puts "  - $name: [format %.2f $freq] MHz"
    }
    
    return 1
}

proc check_constraints {} {
    盖率..."
    
    set unconstrained [get_timing_paths -max_paths 1 \
      -filter {STARTPOINT_CLOCK == "" || ENDPOINT_CLOCK == ""}]
    
    if {[llength $unconstrained] > 0} {
        puts "✗ 存在未约束的路径"
        return 0
    } else {
        puts "✓ 所有路径都已约束"
        return 1
    }
}

# 主验证流程
puts "=========================================="
puts "实现后验证"
puts "=========================================="

set all_pass 1

if {![check_clocks]} { set all_pass 0 }
if {![check_timing]} { set all_pass 0 }
if {![check_constraints]} { set all_pass 0 }

puts "=========================================="
if {$all_pass} {
    puts "✓ 所有检查通过!"
    exit 0
} else {
    puts "✗ 部分检查失败"
    exit 1
}
```

## 4. 硬件验证

### 4.1 使用逻辑分析仪 (ILA)

在设计中插入 ILA 来测量实际时钟：

```tcl
# 在 Vivado 中插入 ILA
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 4ebug_cores u_ila_0]
set_property C_TRIGG_OUT_WIDTH 1 [get_debug_cores u_ila_0]
set_property C_INPUT_PIPE_STAGES 0 [get_debug_cores u_ila_0]

# 连接探针
set_property port_width 1 [get_debug_ports u_ila_0/clk]
connect_debug_port u_ila_0/clk [get_nets clock]

set_property port_width 1 [get_debug_ports u_ila_0/probe0]
connect_debug_port u_ila_0/probe0 [get_nets io_lcd_spi_clk]

set_property port_width 1 [get_debug_ports u_ila_0/probe1]
connect_debug_port u_ila_0/probe1 [get_nets io_lcd_spi_mosi]

# 实现并生成比特流
implement_debug_core [get_debug_cores u_ila_0]
```

### 4.2 示波器测量

使用示波器测量实际输出：

**测量项目**:
1. **主时钟频率** (如果可访问)
   - 预期: 50 MHz ± 100 ppm
   
2. **SPI 时钟频率**
   - 预期: 8.33 MHz ± 5%
   - 测量: `io_lcd_spi_clk` 引脚
   
3. **SPI 时钟占空比**
   - 预期: 50% ± 5%
   
4. **SPI 数据建立/保持时间**
   - 测量 `io_lcd_spi_mosi` 相对于 `io_lcd_spi_clk` 的时序
   - 建立时间: > 10 ns
   - 保持时间: > 10 ns

### 4.3 功能测试

编写测试程序验证 SPI 通信：

```c
// 测试 SPI 时钟
void test_spi_clock() {
    // 发送测试模式
    lcd_send_command(0x00);  // NOP
    
    // 使用逻辑分析仪或示波器观察:
    // 1. SPI 时钟频率
    // 2. 数据与时钟的相位关系
    // 3. 片选信号时序
}
```

## 5. 验证检查清单

### 仿真阶段
- [ ] SPI 时钟频率正确 (8-10 MHz)
- [ ] SPI 时钟占空比 ~50%
- [ ] 分频计数器工作正常
- [ ] 数据与时钟相位关系正确

### 综合阶段
- [ ] 主时钟约束正确 (20 ns)
- [ ] SPI 生成时钟定义正确
- [ ] 所有端口都有延迟约束
- [ ] 没有未约束的路径
- [ ] 时钟网络报告正常

### 实现阶段
- [ ] Setup 时序满足 (WNS ≥ 0)
- [ ] Hold 时序满足 (WHS ≥ 0)
- [ ] 时钟偏斜在合理范围
- [ ] 功耗在预期范围
- [ ] DCP 文件生成成功

### 硬件阶段
- [ ] 实际 SPI 时钟频率符合预期
- [ ] SPI 通信功能正常
- [ ] LCD 显示正常
- [ ] 无时序相关的功能异常

## 6. 常见问题排查

### 问题 1: 时序违例
```
症状: WNS < 0
原因: 时钟周期太短或路径延迟太大
解决:
1. 检查时钟约束是否正确
2. 增加流水线级数
3. 降低时钟频率
4. 优化关键路径
```

### 问题 2: SPI 时钟频率不对
```
症状: 实测频率与预期不符
原因: 分频逻辑错误或主时钟频率错误
解决:
1. 检查 clockFreq 参数
2. 检查 spiFreq 参数
3. 验证分频计数器逻辑
4. 检查主时钟实际频率
```

### 问题 3: 未约束的路径
```
症状: report_timing -unconstrained 有输出
原因: 某些路径没有时钟约束
解决:
1. 检查所有输入/输出端口
2. 添加缺失的约束
3. 标记假路径（如果适用）
```

## 7. 自动化验证脚本

完整的自动化验证流程：

```bash
#!/bin/bash
# chisel/synthesis/fpga/scripts/verify_all.sh

echo "=========================================="
echo "完整验证流程"
echo "=========================================="

# 1. Chisel 仿真
echo "[1/4] Chisel 仿真验证..."
cd chisel
sbt "testOnly riscv.ai.ClockVerificationTest" || exit 1

# 2. 生成 Verilog
echo "[2/4] 生成 Verilog..."
sbt 'runMain riscv.ai.SimpleEdgeAiSoCMain' || exit 1

# 3. Vivado 综合验证
echo "[3/4] Vivado 综合验证..."
cd ../synthesis/fpga
vivado -mode batch -source scripts/verify_clocks.tcl || exit 1

# 4. 检查报告
echo "[4/4] 检查报告..."
if grep -q "ERROR" reports/*.txt; then
    echo "✗ 发现错误，请检查报告"
    exit 1
fi

echo "=========================================="
echo "✓ 所有验证通过!"
echo "=========================================="
```

运行完整验证：
```bash
chmod +x chisel/synthesis/fpga/scripts/verify_all.sh
./chisel/synthesis/fpga/scripts/verify_all.sh
```

---

**总结**: 验证应该从仿真开始，逐步到综合、实现，最后到硬件测试。每个阶段都有特定的检查项目和工具。
