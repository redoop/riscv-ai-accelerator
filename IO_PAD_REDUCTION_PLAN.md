# IO Pad 减少方案

**目标**: 从 97 个 IO 减少到 81 个 (减少 16 个)  
**日期**: 2025-12-03  
**版本**: v0.4 优化方案

---

## 当前 IO 统计 (97 个)

### 基础信号 (3 个)
- `clock` (1 input)
- `reset` (1 input, 隐含)
- `trap` (1 output)

### UART (2 个)
- `uart_tx` (1 output)
- `uart_rx` (1 input)

### LCD SPI (6 个)
- `lcd_spi_clk` (1 output)
- `lcd_spi_mosi` (1 output)
- `lcd_spi_cs` (1 output)
- `lcd_spi_dc` (1 output)
- `lcd_spi_rst` (1 output)
- `lcd_backlight` (1 output)

### GPIO (64 个)
- `gpio_out[31:0]` (32 output)
- `gpio_in[31:0]` (32 input)

### 中断信号 (4 个)
- `compact_irq` (1 output)
- `bitnet_irq` (1 output)
- `uart_tx_irq` (1 output)
- `uart_rx_irq` (1 output)

### Flash SPI (4 个)
- `flash_spi_clk` (1 output)
- `flash_spi_mosi` (1 output)
- `flash_spi_miso` (1 input)
- `flash_spi_cs` (1 output)

### PSRAM Quad SPI (10 个)
- `psram_spi_clk` (1 output)
- `psram_spi_cs` (1 output)
- `psram_spi_mosi` (1 output)
- `psram_spi_miso` (1 input)
- `psram_spi_sio2_out` (1 output)
- `psram_spi_sio2_oe` (1 output)
- `psram_spi_sio2_in` (1 input)
- `psram_spi_sio3_out` (1 output)
- `psram_spi_sio3_oe` (1 output)
- `psram_spi_sio3_in` (1 input)

**总计**: 3 + 2 + 6 + 64 + 4 + 4 + 10 = **93 个** (不含隐含 reset)

---

## 优化方案 (减少 16 个)

### 方案 1: GPIO 减半 (推荐) ⭐

**减少 32 个 GPIO → 节省 32 个 IO**

```scala
// 修改前: 64 个 GPIO
val gpio_out = Output(UInt(32.W))  // 32 个
val gpio_in = Input(UInt(32.W))    // 32 个

// 修改后: 32 个 GPIO (16-bit)
val gpio_out = Output(UInt(16.W))  // 16 个
val gpio_in = Input(UInt(16.W))    // 16 个
```

**影响评估**:
- ✅ 对核心功能无影响 (AI 加速器、UART、LCD、Flash、PSRAM 不受影响)
- ✅ 16 个 GPIO 足够大多数应用 (LED、按钮、传感器)
- ✅ 软件兼容性好 (只需修改 HAL 层宏定义)
- ✅ 实现简单，风险低

**最终 IO 数**: 93 - 32 = **61 个** ✅ (远低于 81 个限制)

---

### 方案 2: 移除调试中断信号 (可选)

**减少 4 个中断输出 → 节省 4 个 IO**

```scala
// 移除这些调试用中断信号
// val compact_irq = Output(Bool())
// val bitnet_irq = Output(Bool())
// val uart_tx_irq = Output(Bool())
// val uart_rx_irq = Output(Bool())
```

**影响评估**:
- ⚠️ 中断功能仍可通过软件轮询实现
- ⚠️ 对性能有轻微影响 (需要 CPU 轮询)
- ✅ 调试阶段可通过仿真验证
- ✅ 生产版本可选择性恢复

**最终 IO 数**: 61 - 4 = **57 个** ✅

---

### 方案 3: PSRAM 降级为标准 SPI (备选)

**移除 Quad SPI 支持 → 节省 6 个 IO**

```scala
// 修改前: Quad SPI (10 个 IO)
val psram_spi_clk = Output(Bool())
val psram_spi_cs = Output(Bool())
val psram_spi_mosi = Output(Bool())
val psram_spi_miso = Input(Bool())
val psram_spi_sio2_out = Output(Bool())
val psram_spi_sio2_oe = Output(Bool())
val psram_spi_sio2_in = Input(Bool())
val psram_spi_sio3_out = Output(Bool())
val psram_spi_sio3_oe = Output(Bool())
val psram_spi_sio3_in = Input(Bool())

// 修改后: 标准 SPI (4 个 IO)
val psram_spi_clk = Output(Bool())
val psram_spi_cs = Output(Bool())
val psram_spi_mosi = Output(Bool())
val psram_spi_miso = Input(Bool())
```

**影响评估**:
- ⚠️ PSRAM 带宽降低 75% (25 MB/s → 6.25 MB/s)
- ⚠️ 延迟增加 4× (0.5 μs → 2 μs)
- ✅ 功能完整保留
- ⚠️ 需要修改 PSRAM 控制器代码

**最终 IO 数**: 93 - 6 = **87 个** (仍超出 81 个限制)

---

### 方案 4: 共享 Flash 和 PSRAM SPI 总线 (高级)

**复用 SPI 信号 → 节省 3 个 IO**

```scala
// 共享 CLK, MOSI, MISO
val storage_spi_clk = Output(Bool())   // 共享
val storage_spi_mosi = Output(Bool())  // 共享
val storage_spi_miso = Input(Bool())   // 共享
val flash_spi_cs = Output(Bool())      // Flash 片选
val psram_spi_cs = Output(Bool())      // PSRAM 片选
// PSRAM Quad 信号保持独立
val psram_spi_sio2_out = Output(Bool())
val psram_spi_sio2_oe = Output(Bool())
val psram_spi_sio2_in = Input(Bool())
val psram_spi_sio3_out = Output(Bool())
val psram_spi_sio3_oe = Output(Bool())
val psram_spi_sio3_in = Input(Bool())
```

**影响评估**:
- ⚠️ Flash 和 PSRAM 不能同时访问
- ⚠️ 需要总线仲裁逻辑
- ⚠️ 实现复杂度高
- ✅ 性能影响小 (很少同时访问)

**最终 IO 数**: 93 - 3 = **90 个** (仍超出 81 个限制)

---

## 推荐实施方案

### 阶段 1: GPIO 减半 (立即实施) ⭐⭐⭐⭐⭐

**优先级**: 最高  
**风险**: 极低  
**收益**: 节省 32 个 IO

```scala
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // 基础信号 (3)
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    val trap = Output(Bool())
    
    // LCD SPI (6)
    val lcd_spi_clk = Output(Bool())
    val lcd_spi_mosi = Output(Bool())
    val lcd_spi_cs = Output(Bool())
    val lcd_spi_dc = Output(Bool())
    val lcd_spi_rst = Output(Bool())
    val lcd_backlight = Output(Bool())
    
    // GPIO 减半 (32 → 16)
    val gpio_out = Output(UInt(16.W))  // 16 个
    val gpio_in = Input(UInt(16.W))    // 16 个
    
    // 中断信号 (4)
    val compact_irq = Output(Bool())
    val bitnet_irq = Output(Bool())
    val uart_tx_irq = Output(Bool())
    val uart_rx_irq = Output(Bool())
    
    // Flash SPI (4)
    val flash_spi_clk = Output(Bool())
    val flash_spi_mosi = Output(Bool())
    val flash_spi_miso = Input(Bool())
    val flash_spi_cs = Output(Bool())
    
    // PSRAM Quad SPI (10)
    val psram_spi_clk = Output(Bool())
    val psram_spi_cs = Output(Bool())
    val psram_spi_mosi = Output(Bool())
    val psram_spi_miso = Input(Bool())
    val psram_spi_sio2_out = Output(Bool())
    val psram_spi_sio2_oe = Output(Bool())
    val psram_spi_sio2_in = Input(Bool())
    val psram_spi_sio3_out = Output(Bool())
    val psram_spi_sio3_oe = Output(Bool())
    val psram_spi_sio3_in = Input(Bool())
  })
}
```

**总 IO 数**: 3 + 6 + 32 + 4 + 4 + 10 = **59 个** (含 clock)  
**裕量**: 81 - 59 = **22 个** ✅

---

### 阶段 2: 移除调试中断 (可选)

如果需要更多裕量，可移除 4 个调试中断信号:

```scala
// 移除这些信号
// val compact_irq = Output(Bool())
// val bitnet_irq = Output(Bool())
// val uart_tx_irq = Output(Bool())
// val uart_rx_irq = Output(Bool())
```

**总 IO 数**: 59 - 4 = **55 个**  
**裕量**: 81 - 55 = **26 个** ✅

---

## 实施步骤

### Step 1: 修改硬件 (EdgeAiSoCSimple.scala)

```bash
# 修改 GPIO 位宽
vim chisel/src/main/scala/EdgeAiSoCSimple.scala

# 修改位置:
# 1. SimpleGPIO 类 (line 475)
# 2. SimpleEdgeAiSoC 类 (line 679)
```

### Step 2: 修改软件 HAL (hal.h)

```c
// 修改前
#define GPIO_OUT_REG  (*(volatile uint32_t*)0x20020000)
#define GPIO_IN_REG   (*(volatile uint32_t*)0x20020004)

// 修改后
#define GPIO_OUT_REG  (*(volatile uint16_t*)0x20020000)
#define GPIO_IN_REG   (*(volatile uint16_t*)0x20020004)
#define GPIO_MASK     0xFFFF  // 16-bit mask
```

### Step 3: 重新生成 Verilog

```bash
cd chisel
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
```

### Step 4: 运行测试

```bash
# 运行所有测试
sbt test

# 重点测试 GPIO
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
```

### Step 5: 重新综合

```bash
cd chisel/synthesis
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

---

## IO 对比表

| 模块 | v0.4 原始 | 方案 1 (GPIO 减半) | 方案 1+2 (移除中断) |
|------|-----------|-------------------|-------------------|
| 基础信号 | 3 | 3 | 3 |
| UART | 2 | 2 | 2 |
| LCD | 6 | 6 | 6 |
| GPIO | 64 | 32 | 32 |
| 中断 | 4 | 4 | 0 |
| Flash | 4 | 4 | 4 |
| PSRAM | 10 | 10 | 10 |
| **总计** | **93** | **61** ✅ | **57** ✅ |
| **裕量** | -12 ❌ | +20 ✅ | +24 ✅ |

---

## 风险评估

| 方案 | 风险等级 | 实施难度 | 功能影响 | 推荐度 |
|------|---------|---------|---------|--------|
| GPIO 减半 | 🟢 低 | 🟢 简单 | 🟢 无 | ⭐⭐⭐⭐⭐ |
| 移除中断 | 🟡 中 | 🟢 简单 | 🟡 轻微 | ⭐⭐⭐⭐ |
| PSRAM 降级 | 🟡 中 | 🟡 中等 | 🔴 显著 | ⭐⭐ |
| SPI 总线共享 | 🔴 高 | 🔴 复杂 | 🟡 轻微 | ⭐⭐ |

---

## 结论

**推荐方案**: GPIO 减半 (32-bit → 16-bit)

**理由**:
1. ✅ 节省 32 个 IO，远超需求 (61 vs 81)
2. ✅ 实施简单，风险极低
3. ✅ 对核心功能无影响
4. ✅ 16 个 GPIO 足够大多数应用
5. ✅ 保留所有高性能特性 (Quad SPI、中断等)

**预期结果**:
- 最终 IO 数: **61 个** (含 clock)
- 裕量: **20 个** (24.7%)
- 实施时间: **1-2 小时**
- 测试时间: **1 小时**

---

**创建日期**: 2025-12-03  
**状态**: 待实施  
**优先级**: 高
