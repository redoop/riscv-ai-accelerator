# 时钟约束规格说明

## 主时钟配置

### 系统主时钟
- **时钟名称**: `sys_clk` / `clock`
- **频率**: 50 MHz
- **周期**: 20 ns (10.000 ns 在当前约束中需要更新)
- **占空比**: 50%
- **来源**: 外部晶振或 PLL

### 当前约束（需要更新）
```tcl
# 当前配置（错误 - 100MHz）
create_clock -period 10.000 -name sys_clk [get_ports clock]

# 应该修改为（50MHz）
create_clock -period 20.000 -name sys_clk [get_ports clock]
```

## SPI 时钟配置

### SPI 时钟特性
- **模块**: TFTLCD (LCD 控制器)
- **主时钟**: 50 MHz
- **SPI 目标频率**: 10 MHz
- **分频比**: 5 (50MHz / 10MHz)
- **实现方式**: 软件分频（计数器）

### SPI 时钟生成逻辑
```scala
// Chisel 代码
val spiDivider = (clockFreq / spiFreq / 2).U(8.W)
// spiDivider = (50000000 / 10000000 / 2) = 2.5 ≈ 2

val spiCounter = RegInit(0.U(8.W))
val spiClkReg = RegInit(false.B)

when(spiCounter >= spiDivider - 1.U) {
  spiCounter := 0.U
  spiClkReg := !spiClkReg  // 翻转时钟
}.otherwise {
  spiCounter := spiCounter + 1.U
}
```

### SPI 时钟参数
- **信号名称**: `lcd_spi_clk` / `io_lcd_spi_clk`
- **实际频率**: ~8.33 MHz (由于整数分频)
- **周期**: ~120 ns
- **占空比**: 50%
- **类型**: 生成时钟（Generated Clock）

## 完整时钟约束文件

### 推荐的 timing.xdc
```tcl
# ============================================================================
# 主时钟约束
# ============================================================================

# 系统主时钟 - 50MHz
create_clock -period 20.000 -name sys_clk [get_ports clock]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clock]

# 时钟不确定性
set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold 0.3 [get_clocks sys_clk]

# ============================================================================
# 生成时钟约束 - SPI
# ============================================================================

# SPI 时钟 - 从主时钟分频生成
# 注意：这是一个软件生成的时钟，不是硬件 PLL
# 实际频率约 8.33 MHz (50MHz / 6)
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 6 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]

# SPI 时钟输出延迟
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_clk]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_clk]

# SPI 数据输出延迟（相对于 SPI 时钟）
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_mosi]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_mosi]

# SPI 片选输出延迟
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_cs]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_cs]

# SPI DC（数据/命令）输出延迟
set_output_delay -clock spi_clk -max 5.0 [get_ports io_lcd_spi_dc]
set_output_delay -clock spi_clk -min 1.0 [get_ports io_lcd_spi_dc]

# ============================================================================
# 输入延迟约束
# ============================================================================

# UART RX
set_input_delay -clock sys_clk -max 2.0 [get_ports io_uart_rx]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_uart_rx]

# 复位信号
set_input_delay -clock sys_clk -max 2.0 [get_ports reset]
set_input_delay -clock sys_clk -min 0.5 [get_ports reset]

# GPIO 输入
set_input_delay -clock sys_clk -max 2.0 [get_ports io_gpio_in*]
set_input_delay -clock sys_clk -min 0.5 [get_ports io_gpio_in*]

# ============================================================================
# 输出延迟约束
# ============================================================================

# UART TX
set_output_delay -clock sys_clk -max 2.0 [get_ports io_uart_tx]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_uart_tx]

# GPIO 输出
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_out*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_gpio_out*]

# GPIO 输出使能
set_output_delay -clock sys_clk -max 2.0 [get_ports io_gpio_oe*]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_gpio_oe*]

# LCD 背光
set_output_delay -clock sys_clk -max 2.0 [get_ports io_lcd_backlight]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_lcd_backlight]

# LCD 复位
set_output_delay -clock sys_clk -max 2.0 [get_ports io_lcd_spi_rst]
set_output_delay -clock sys_clk -min 0.5 [get_ports io_lcd_spi_rst]

# ============================================================================
# 假路径约束
# ============================================================================

# 复位信号是异步的
set_false_path -from [get_ports reset]

# 跨时钟域路径（如果有）
# set_false_path -from [get_clocks sys_clk] -to [get_clocks spi_clk]

# ============================================================================
# 多周期路径约束
# ============================================================================

# SPI 控制器状态机可以使用多周期
set_multicycle_path -setup 2 -from [get_cells lcd/lcd/state_reg*] \
  -to [get_cells lcd/lcd/spiShiftReg_reg*]
set_multicycle_path -hold 1 -from [get_cells lcd/lcd/state_reg*] \
  -to [get_cells lcd/lcd/spiShiftReg_reg*]

# ============================================================================
# 最大延迟约束
# ============================================================================

# 组合逻辑最大延迟
set_max_delay 10.0 -from [all_inputs] -to [all_outputs]

# ============================================================================
# 时钟组约束
# ============================================================================

# 定义同步时钟组
set_clock_groups -physically_exclusive \
  -group [get_clocks sys_clk] \
  -group [get_clocks spi_clk]
```

## 时钟域说明

### 主时钟域 (sys_clk)
- **频率**: 50 MHz
- **用途**: 
  - CPU 核心
  - 内存控制器
  - UART 控制器
  - GPIO 控制器
  - SPI 控制器状态机

### SPI 时钟域 (spi_clk)
- **频率**: ~8.33 MHz (实际)
- **用途**:
  - SPI 数据输出
  - SPI 时钟输出
  - LCD 接口

### 时钟域交互
- **sys_clk → spi_clk**: 同步（通过寄存器）
- **跨域处理**: 使用双触发器同步

## 关键时序参数

### Setup/Hold 时间
```tcl
# 主时钟
Setup Time: 0.5 ns
Hold Time: 0.3 ns

# SPI 时钟
Setup Time: 5.0 ns (宽松，因为频率低)
Hold Time: 1.0 ns
```

### 输入/输出延迟
```tcl
# 输入延迟（相对于时钟边沿）
Max: 2.0 ns
Min: 0.5 ns

# 输出延迟（相对于时钟边沿）
Max: 2.0 ns (sys_clk)
Max: 5.0 ns (spi_clk)
Min: 0.5 ns (sys_clk)
Min: 1.0 ns (spi_clk)
```

## 验证检查清单

- [ ] 主时钟周期正确（20 ns = 50 MHz）
- [ ] SPI 生成时钟定义正确
- [ ] 所有输入端口有输入延迟约束
- [ ] 所有输出端口有输出延迟约束
- [ ] 跨时钟域路径已标记为假路径或多周期
- [ ] 时钟不确定性已设置
- [ ] 异步信号（如 reset）已标记为假路径
- [ ] 时序报告无违例

## 注意事项

1. **SPI 时钟是软件生成的**，不是硬件 PLL，因此：
   - 抖动较大
   - 占空比可能不精确
   - 需要宽松的时序约束

2. **实际 SPI 频率**：
   - 目标：10 MHz
   - 实际：~8.33 MHz（由于整数分频）
   - 分频比：6（不是 5）

3. **时钟路径**：
   ```
   sys_clk (50MHz) 
     → spiCounter (计数器)
     → spiClkReg (触发器)
     → io_lcd_spi_clk (输出)
   ```

4. **建议**：如果需要精确的 10 MHz SPI 时钟，考虑：
   - 使用 PLL 生成
   - 或调整主时钟频率为 60 MHz

## 联系信息

如有疑问，请联系：
- 设计负责人：童老师
- 后端负责人：[您的名字]
