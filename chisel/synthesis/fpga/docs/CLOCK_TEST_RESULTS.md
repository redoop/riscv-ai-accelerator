# 时钟验证测试结果分析

## 测试执行时间
**日期**: 2025-11-19  
**测试工具**: ChiselTest + ScalaTest

## 测试结果总结

### ❌ 测试 1: SPI 时钟频率验证 - **失败**

**测量数据**:
```
主时钟频率: 50 MHz
目标 SPI 频率: 10 MHz
SPI 时钟周期数: 20
主时钟周期数: 79
平均 SPI 周期: 3.95 个主时钟周期
测量频率: 12.658 MHz
误差: +26.58%
```

**问题分析**:
1. **预期**: SPI 时钟应该是 8-10 MHz
2. **实际**: 测量到 12.658 MHz
3. **原因**: 分频比实际是 ~4，而不是预期的 5-6

**SPI 时钟上升沿位置**:
```
上升沿 #0: 主时钟周期 2
上升沿 #1: 主时钟周期 6   (间隔 4)
上升沿 #2: 主时钟周期 10  (间隔 4)
上升沿 #3: 主时钟周期 14  (间隔 4)
上升沿 #4: 主时钟周期 18  (间隔 4)
```

**结论**: SPI 时钟每 4 个主时钟周期翻转一次，导致频率为 50MHz / 4 = 12.5 MHz

### ✅ 测试 2: SPI 时钟占空比验证 - **通过**

**测量数据**:
```
高电平周期: 1000
低电平周期: 1000
总周期: 2000
占空比: 50.00%
偏差: 0.00%
```

**结论**: 占空比完美，正好 50%

## 根本原因分析

### 分频逻辑检查

查看 `TFTLCD.scala` 中的分频逻辑：

```scala
val spiDivider = (clockFreq / spiFreq / 2).U(8.W)
// spiDivider = (50000000 / 10000000 / 2) = 2.5 ≈ 2 (整数截断)

when(spiCounter >= spiDivider - 1.U) {
  spiCounter := 0.U
  spiClkReg := !spiClkReg  // 每 2 个周期翻转一次
}.otherwise {
  spiCounter := spiCounter + 1.U
}
```

**问题**:
1. `spiDivider` 计算结果是 2.5，但 Chisel 会截断为 2
2. 计数器从 0 计数到 1（2 个周期），然后翻转时钟
3. 因此 SPI 时钟每 2 个主时钟周期翻转一次
4. 完整的 SPI 时钟周期 = 2 × 2 = 4 个主时钟周期
5. SPI 频率 = 50 MHz / 4 = 12.5 MHz

### 正确的分频计算

要得到 10 MHz 的 SPI 时钟：
- 需要的分频比 = 50 MHz / 10 MHz = 5
- 每个半周期 = 5 / 2 = 2.5 个主时钟周期
- 但由于整数限制，无法精确实现 10 MHz

**可能的解决方案**:

#### 方案 1: 接受 8.33 MHz（推荐）
```scala
val spiDivider = 3.U  // 每 3 个周期翻转
// SPI 周期 = 3 × 2 = 6 个主时钟周期
// SPI 频率 = 50 MHz / 6 = 8.33 MHz
```

#### 方案 2: 使用 60 MHz 主时钟
```scala
clockFreq = 60000000
spiFreq = 10000000
spiDivider = (60000000 / 10000000 / 2) = 3
// SPI 周期 = 3 × 2 = 6 个主时钟周期
// SPI 频率 = 60 MHz / 6 = 10 MHz (精确)
```

#### 方案 3: 使用小数分频器（复杂）
使用累加器实现小数分频，但会增加设计复杂度。

## 修复建议

### 立即修复（推荐）

修改 `TFTLCD.scala`:

```scala
// 修改前
val spiDivider = (clockFreq / spiFreq / 2).U(8.W)

// 修改后 - 明确使用 3 以获得 8.33 MHz
val spiDivider = 3.U(8.W)  // 50MHz / (3*2) = 8.33 MHz
```

或者更灵活的方式：

```scala
// 计算并向上取整
val divRatio = (clockFreq + spiFreq - 1) / spiFreq  // 向上取整
val spiDivider = ((divRatio + 1) / 2).U(8.W)
```

### 更新测试预期

如果接受 8.33 MHz 作为目标频率，更新测试：

```scala
// 修改测试范围
assert(frequency >= 8000000 && frequency <= 9000000,  // 改为 8-9 MHz
  f"SPI 频率 ${frequency/1000000}%.3f MHz 超出范围 [8.0, 9.0] MHz")
```

### 更新文档

更新所有文档中的 SPI 频率说明：
- 目标频率：10 MHz → 8.33 MHz
- 实际频率：~8.33 MHz（50 MHz / 6）

## 时钟约束更新

### 当前约束（需要更新）

```tcl
# 当前（错误）
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 6 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]
```

### 实际情况

根据测试结果，实际分频比是 4：

```tcl
# 实际（基于测试）
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 4 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]
```

### 修复后（推荐）

修复代码后，分频比应该是 6：

```tcl
# 修复后（推荐）
create_generated_clock -name spi_clk \
  -source [get_pins {lcd/lcd/spiClkReg_reg/C}] \
  -divide_by 6 \
  [get_pins {lcd/lcd/spiClkReg_reg/Q}]
```

## 行动项

### 高优先级
- [ ] 修复 `TFTLCD.scala` 中的分频计算
- [ ] 重新运行测试验证修复
- [ ] 更新时钟约束文件

### 中优先级
- [ ] 更新所有文档中的频率说明
- [ ] 更新测试预期值
- [ ] 添加分频计算的注释

### 低优先级
- [ ] 考虑使用 60 MHz 主时钟以获得精确 10 MHz
- [ ] 评估是否需要小数分频器

## 总结

**当前状态**: 
- ❌ SPI 时钟频率不符合预期（12.5 MHz vs 10 MHz）
- ✅ SPI 时钟占空比完美（50%）

**根本原因**: 
- 整数除法截断导致分频比错误

**推荐方案**: 
- 修改代码使用分频比 3，接受 8.33 MHz 作为目标频率
- 或者使用 60 MHz 主时钟以获得精确 10 MHz

**影响评估**:
- 8.33 MHz 对于大多数 LCD 来说仍然是可接受的
- 需要更新文档和约束文件
- 不影响功能，只是频率略低于目标

## 参考

- 测试代码: `chisel/src/test/scala/ClockVerificationTest.scala`
- 源代码: `chisel/src/main/scala/peripherals/TFTLCD.scala`
- 约束文件: `chisel/synthesis/fpga/constraints/timing_complete.xdc`
