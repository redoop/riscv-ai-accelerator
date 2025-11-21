# 时钟约束数据 - 快速参考

## 主时钟参数

| 参数 | 值 |
|------|-----|
| **时钟名称** | sys_clk |
| **端口名称** | clock |
| **频率** | 100 MHz |
| **周期** | 10 ns |
| **占空比** | 50% |
| **Setup 不确定性** | 0.5 ns |
| **Hold 不确定性** | 0.3 ns |

### 主时钟约束
```tcl
create_clock -period 10.000 -name sys_clk [get_ports clock]
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]
```

## SPI 时钟参数

| 参数 | 值 |
|------|-----|
| **时钟名称** | spi_clk |
| **信号名称** | io_lcd_spi_clk |
| **目标频率** | 10 MHz |
| **实际频率** | 10.0 MHz |
| **周期** | 100 ns |
| **分频比** | 10 (从 100MHz) |
| **生成方式** | 软件分频（计数器） |
| **占空比** | ~50% |

### SPI 时钟约束
```tcl
# 生成时钟定义
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 10 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]

# SPI 输出延迟
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_mosi]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_mosi]
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_cs]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_cs]
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_dc]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_dc]
```

## SPI 相关信号

| 信号名称 | 方向 | 时钟域 | 说明 |
|---------|------|--------|------|
| io_lcd_spi_clk | Output | spi_clk | SPI 时钟 |
| io_lcd_spi_mosi | Output | spi_clk | SPI 数据输出 |
| io_lcd_spi_cs | Output | spi_clk | SPI 片选 |
| io_lcd_spi_dc | Output | spi_clk | 数据/命令选择 |
| io_lcd_spi_rst | Output | sys_clk | LCD 复位 |
| io_lcd_backlight | Output | sys_clk | 背光控制 |

## 时序要求

### 输入延迟（相对于 sys_clk）
```tcl
set_input_delay -clock sys_clk -max 2.0 [get_ports ...]
set_input_delay -clock sys_clk -min 0.5 [get_ports ...]
```

### 输出延迟
**系统时钟域 (sys_clk)**:
```tcl
set_output_delay -clock sys_clk -max 2.0 [get_ports ...]
set_output_delay -clock sys_clk -min 0.5 [get_ports ...]
```

**SPI 时钟域 (spi_clk)**:
```tcl
set_output_delay -clock spi_clk -max 5.0 [get_ports ...]
set_output_delay -clock spi_clk -min 1.0 [get_ports ...]
```

## 关键路径

### SPI 时钟生成路径
```
sys_clk (50MHz)
  → spiCounter (8-bit 计数器)
  → spiClkReg (触发器)
  → io_lcd_spi_clk (输出端口)
```

### 分频逻辑
```scala
val spiDivider = (50000000 / 10000000 / 2).U = 2.U
// 实际: 50MHz / (2 * 2.5) ≈ 10MHz
// 但由于整数运算: 50MHz / (2 * 3) = 8.33MHz
```

## 完整约束文件位置

- **详细文档**: `chisel/synthesis/fpga/docs/CLOCK_CONSTRAINTS_SPEC.md`
- **完整约束**: `chisel/synthesis/fpga/constraints/timing_complete.xdc`
- **当前约束**: `chisel/synthesis/fpga/constraints/timing.xdc` (需要更新)

## 注意事项

1. ⚠️ **当前 timing.xdc 中主时钟周期错误**
   - 当前: 10 ns (100 MHz) ❌
   - 应该: 20 ns (50 MHz) ✅

2. ⚠️ **SPI 时钟是软件生成的**
   - 不是硬件 PLL
   - 抖动较大
   - 需要宽松的时序约束

3. ⚠️ **层次路径需要验证**
   - `lcd/lcd/spiClkReg_reg` 路径需要根据实际综合结果调整
   - 建议在综合后检查实际层次结构

## 快速检查命令

### Vivado TCL 命令
```tcl
# 检查时钟
report_clocks
get_clocks

# 检查生成时钟
report_clock_networks

# 检查时序
report_timing_summary

# 检查约束覆盖率
report_timing -unconstrained
```

## 联系方式

如有疑问，请联系设计负责人。
