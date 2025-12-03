# SimpleEdgeAiSoC AI 加速器接入 ysyxSoC 指南

**日期**: 2025-12-03  
**版本**: v0.4.1-ai  
**状态**: ✅ 设计完成

---

## 📋 集成概述

将 SimpleEdgeAiSoC 的 **完整核心组件**接入 ysyxSoC 平台：
- ✅ PicoRV32 核心（RISC-V RV32I CPU）
- ✅ CompactAccel（8x8 矩阵加速器，1.6 GOPS）
- ✅ BitNetAccel（16x16 BitNet 加速器，4.8 GOPS）
- ✅ SPI Flash 控制器（16 MB @ 25 MHz）
- ✅ PSRAM 控制器（8 MB @ 50 MHz, Quad SPI）
- ✅ 中断支持（AI 加速器完成中断）

---

## 🏗️ 架构设计

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ysyx_26000001 Wrapper                    │
│                                                             │
│  ┌──────────────┐                                          │
│  │  PicoRV32    │                                          │
│  │   CPU Core   │                                          │
│  └──────┬───────┘                                          │
│         │ mem_* (PicoRV32 interface)                       │
│         │                                                   │
│  ┌──────▼────────────────────────────────────────┐         │
│  │         Address Decoder                       │         │
│  │  - CompactAccel:  0x2000_0000 - 0x2000_0FFF  │         │
│  │  - BitNetAccel:   0x2000_1000 - 0x2000_1FFF  │         │
│  │  - Flash Ctrl:    0x2000_2000 - 0x2000_2FFF  │         │
│  │  - PSRAM Ctrl:    0x2000_3000 - 0x2000_3FFF  │         │
│  │  - Flash Memory:  0x3000_0000 - 0x30FF_FFFF  │         │
│  │  - PSRAM Memory:  0x0400_0000 - 0x047F_FFFF  │         │
│  │  - Bus (others):  All other addresses        │         │
│  └──────┬────────────────────────────────────────┘         │
│         │                                                   │
│    ┌────┴────┬──────────┬──────────┬──────────┐           │
│    │         │          │          │          │           │
│  ┌─▼──────┐ ┌▼────────┐ ┌▼───────┐ ┌▼───────┐ │           │
│  │Compact │ │ BitNet  │ │ Flash  │ │ PSRAM  │ │           │
│  │ Accel  │ │ Accel   │ │  Ctrl  │ │  Ctrl  │ │           │
│  │1.6GOPS │ │4.8GOPS  │ │ 16 MB  │ │  8 MB  │ │           │
│  └─┬──────┘ └┬────────┘ └────────┘ └────────┘ │           │
│    │ irq     │ irq                              │           │
│    └─────────┴──────────────────────────────────┘           │
│              │                          │                  │
│         ┌────▼────┐              ┌─────▼─────┐            │
│         │ IFU/LSU │              │  IFU/LSU  │            │
│         │ (Local) │              │   (Bus)   │            │
│         └─────────┘              └─────┬─────┘            │
│                                        │                  │
│                                  ┌─────▼─────┐            │
│                                  │ SimpleBus │            │
│                                  │ Interface │            │
│                                  └───────────┘            │
└─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                          ysyxSoC Platform
                    (UART, SPI, SDRAM, etc.)
```

### 内存映射

| 地址范围 | 设备 | 大小 | 说明 |
|----------|------|------|------|
| `0x0000_0000 - 0x0FFF_FFFF` | PSRAM (ysyxSoC) | 256 MB | 外部 PSRAM（大容量） |
| `0x0400_0000 - 0x047F_FFFF` | PSRAM (SimpleEdgeAiSoC) | 8 MB | **内置 PSRAM 控制器** |
| `0x1000_0000 - 0x1FFF_FFFF` | SDRAM (ysyxSoC) | 256 MB | 外部 SDRAM |
| `0x2000_0000 - 0x2000_0FFF` | **CompactAccel** | 4 KB | **AI 加速器 1** |
| `0x2000_1000 - 0x2000_1FFF` | **BitNetAccel** | 4 KB | **AI 加速器 2** |
| `0x2000_2000 - 0x2000_2FFF` | **Flash Controller** | 4 KB | **SPI Flash 控制器** |
| `0x2000_3000 - 0x2000_3FFF` | **PSRAM Controller** | 4 KB | **PSRAM 控制器** |
| `0x2000_4000 - 0x2FFF_FFFF` | Peripherals (ysyxSoC) | ~256 MB | UART, SPI, GPIO 等 |
| `0x3000_0000 - 0x30FF_FFFF` | **Flash Memory** | 16 MB | **SPI Flash 存储** |

---

## 🔧 关键设计

### 1. 地址解码

```verilog
// 加速器地址范围
localparam COMPACT_BASE = 32'h20000000;
localparam COMPACT_SIZE = 32'h00001000;
localparam BITNET_BASE  = 32'h20001000;
localparam BITNET_SIZE  = 32'h00001000;

// 地址解码
wire compact_sel = (mem_addr >= COMPACT_BASE) && 
                   (mem_addr < COMPACT_BASE + COMPACT_SIZE);
wire bitnet_sel  = (mem_addr >= BITNET_BASE) && 
                   (mem_addr < BITNET_BASE + BITNET_SIZE);
wire accel_sel   = compact_sel | bitnet_sel;
wire bus_sel     = ~accel_sel & ~mem_instr;
```

### 2. 接口转换

**PicoRV32 → AI 加速器**:
```verilog
// CompactAccel
assign compact_addr  = mem_addr - COMPACT_BASE;
assign compact_wdata = mem_wdata;
assign compact_wen   = compact_sel & mem_valid & (|mem_wstrb);
assign compact_ren   = compact_sel & mem_valid & (~|mem_wstrb);
assign compact_valid = compact_sel & mem_valid;

// BitNetAccel
assign bitnet_addr   = mem_addr - BITNET_BASE;
assign bitnet_wdata  = mem_wdata;
assign bitnet_wen    = bitnet_sel & mem_valid & (|mem_wstrb);
assign bitnet_ren    = bitnet_sel & mem_valid & (~|mem_wstrb);
assign bitnet_valid  = bitnet_sel & mem_valid;
```

**PicoRV32 → SimpleBus**:
```verilog
// IFU: 取指总是走 SimpleBus
assign io_ifu_reqValid = mem_valid & mem_instr;
assign io_ifu_addr = mem_addr;

// LSU: 只有非加速器访问走 SimpleBus
assign io_lsu_reqValid = mem_valid & bus_sel;
assign io_lsu_addr = mem_addr;
assign io_lsu_wen = |mem_wstrb;
```

### 3. 响应多路复用

```verilog
// 响应来源
wire ifu_resp     = io_ifu_respValid & mem_instr;
wire lsu_resp     = io_lsu_respValid & bus_sel;
wire compact_resp = compact_ready & compact_sel;
wire bitnet_resp  = bitnet_ready & bitnet_sel;

assign mem_ready = ifu_resp | lsu_resp | compact_resp | bitnet_resp;

// 数据多路复用
assign mem_rdata = mem_instr ? io_ifu_rdata :    // 指令
                   accel_sel ? accel_rdata :      // 加速器
                   io_lsu_rdata;                  // 总线
```

### 4. 中断路由

```verilog
assign cpu_irq = {
  12'h0,           // [31:20] Reserved
  2'h0,            // [19:18] Reserved
  bitnet_irq,      // [17] BitNet 加速器完成
  compact_irq,     // [16] Compact 加速器完成
  16'h0            // [15:0] Reserved
};
```

---

## 📝 实施步骤

### Step 1: 生成 AI 加速器 Verilog

```bash
cd /opt/github/riscv-ai-accelerator/chisel

# 生成完整 SoC Verilog（包含加速器）
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# 生成的文件：
# - generated/simple_edgeaisoc/SimpleEdgeAiSoC.v
# - generated/simple_edgeaisoc/SimpleCompactAccel.v
# - generated/simple_edgeaisoc/SimpleBitNetAccel.v
```

### Step 2: 使用新的 Wrapper

```bash
# 使用包含 AI 加速器的 wrapper
cp chisel/generated/simple_edgeaisoc/ysyx_26000001_with_ai.v \
   chisel/generated/simple_edgeaisoc/ysyx_26000001.v
```

### Step 3: 修改 ysyxSoCFull.v

在 `ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v` 中保持不变：

```verilog
ysyx_26000001 cpu (
  .clock            (clock),
  .reset            (reset),
  .io_ifu_addr      (_cpu_io_ifu_addr),
  .io_ifu_reqValid  (_cpu_io_ifu_reqValid),
  .io_ifu_rdata     (_bridge_io_ifu_rdata),
  .io_ifu_respValid (_bridge_io_ifu_respValid),
  .io_lsu_addr      (_cpu_io_lsu_addr),
  .io_lsu_reqValid  (_cpu_io_lsu_reqValid),
  .io_lsu_rdata     (_bridge_io_lsu_rdata),
  .io_lsu_respValid (_bridge_io_lsu_respValid),
  .io_lsu_size      (_cpu_io_lsu_size),
  .io_lsu_wen       (_cpu_io_lsu_wen),
  .io_lsu_wdata     (_cpu_io_lsu_wdata),
  .io_lsu_wmask     (_cpu_io_lsu_wmask)
);
```

### Step 4: Verilator 编译

添加 AI 加速器模块到编译列表：

```bash
verilator --cc --exe --build \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED \
  --top-module ysyxSoCTop \
  -I perip/uart16550/rtl \
  -I perip/spi/rtl \
  ysyxSoCFull.v \
  ysyx_26000001.v \
  picorv32.v \
  SimpleCompactAccel.v \
  SimpleBitNetAccel.v \
  perip/**/*.v \
  sim_main.cpp
```

### Step 5: 编译和仿真

```bash
cd ecos/ysyxSoC/ready-to-run/D-stage
./build_sim.sh
./obj_dir/VysyxSoCTop
```

---

## 🧪 测试程序

### C 代码示例

```c
#include <stdint.h>

// AI 加速器基地址
#define COMPACT_BASE 0x20000000
#define BITNET_BASE  0x20001000

// 寄存器偏移
#define REG_CTRL   0x00
#define REG_STATUS 0x04
#define REG_SIZE   0x08
#define REG_PERF   0x0C
#define REG_MATRIX_A 0x100
#define REG_MATRIX_B 0x200
#define REG_MATRIX_C 0x300

// 写寄存器
static inline void write_reg(uint32_t base, uint32_t offset, uint32_t value) {
  *(volatile uint32_t*)(base + offset) = value;
}

// 读寄存器
static inline uint32_t read_reg(uint32_t base, uint32_t offset) {
  return *(volatile uint32_t*)(base + offset);
}

// CompactAccel 测试
void test_compact_accel() {
  // 1. 设置矩阵大小
  write_reg(COMPACT_BASE, REG_SIZE, 4);  // 4x4 矩阵
  
  // 2. 写入矩阵 A 和 B
  for (int i = 0; i < 16; i++) {
    write_reg(COMPACT_BASE, REG_MATRIX_A + i*4, i);
    write_reg(COMPACT_BASE, REG_MATRIX_B + i*4, i);
  }
  
  // 3. 启动计算
  write_reg(COMPACT_BASE, REG_CTRL, 1);
  
  // 4. 等待完成
  while ((read_reg(COMPACT_BASE, REG_STATUS) & 0x2) == 0);
  
  // 5. 读取结果
  for (int i = 0; i < 16; i++) {
    uint32_t result = read_reg(COMPACT_BASE, REG_MATRIX_C + i*4);
    // 处理结果...
  }
  
  // 6. 读取性能计数
  uint32_t cycles = read_reg(COMPACT_BASE, REG_PERF);
}

// BitNetAccel 测试
void test_bitnet_accel() {
  // 类似 CompactAccel，但使用 BITNET_BASE
  // ...
}

int main() {
  test_compact_accel();
  test_bitnet_accel();
  return 0;
}
```

---

## 📊 性能指标

### AI 加速器性能

| 加速器 | 矩阵大小 | 性能 | 延迟 |
|--------|---------|------|------|
| **CompactAccel** | 8x8 | 1.6 GOPS @ 100MHz | ~512 cycles |
| **BitNetAccel** | 16x16 | 4.8 GOPS @ 100MHz | ~256 cycles |

### 资源占用（估算）

| 模块 | 标准单元 | 面积 (µm²) |
|------|---------|-----------|
| PicoRV32 | ~15,000 | ~50,000 |
| CompactAccel | ~5,000 | ~15,000 |
| BitNetAccel | ~8,000 | ~25,000 |
| Wrapper 逻辑 | ~1,000 | ~3,000 |
| **总计** | ~29,000 | ~93,000 |

---

## ⚠️ 注意事项

### 1. 地址冲突

确保 AI 加速器地址不与 ysyxSoC 外设冲突：
- ✅ 0x2000_0000 - 0x2000_1FFF: AI 加速器（8 KB）
- ✅ 0x2000_2000 及以上: ysyxSoC 外设

### 2. 中断优先级

```c
// IRQ 位分配
#define IRQ_COMPACT  16  // CompactAccel
#define IRQ_BITNET   17  // BitNetAccel
```

### 3. 时序约束

AI 加速器需要与 CPU 同频：
- 主时钟: 100 MHz
- 确保加速器满足时序要求

### 4. 仿真验证

在 Verilator 仿真中验证：
- ✅ 地址解码正确
- ✅ 数据读写正确
- ✅ 中断触发正确
- ✅ 计算结果正确

---

## 📚 文件清单

### 源文件

| 文件 | 说明 | 位置 |
|------|------|------|
| `ysyx_26000001_with_ai.v` | 完整 wrapper | chisel/generated/simple_edgeaisoc/ |
| `SimpleCompactAccel.v` | CompactAccel 模块 | chisel/generated/simple_edgeaisoc/ |
| `SimpleBitNetAccel.v` | BitNetAccel 模块 | chisel/generated/simple_edgeaisoc/ |
| `picorv32.v` | PicoRV32 核心 | chisel/src/main/resources/rtl/ |

### 文档

| 文件 | 说明 |
|------|------|
| `YSYXSOC_AI_INTEGRATION.md` | 本文档 |
| `YSYXSOC_INTEGRATION.md` | 基础集成指南 |
| `YSYXSOC_SIMULATION_REPORT.md` | 仿真报告 |

---

## ✅ 验证清单

- [ ] 生成 AI 加速器 Verilog
- [ ] 复制新 wrapper 到正确位置
- [ ] 添加加速器模块到编译列表
- [ ] Verilator 编译通过
- [ ] 仿真运行稳定
- [ ] 地址解码验证
- [ ] CompactAccel 读写测试
- [ ] BitNetAccel 读写测试
- [ ] 中断功能测试
- [ ] 计算结果验证
- [ ] 性能测试

---

## 🚀 下一步

### 短期

1. **生成 Verilog**
   ```bash
   cd chisel
   sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
   ```

2. **编译仿真**
   ```bash
   cd ecos/ysyxSoC/ready-to-run/D-stage
   ./build_sim.sh
   ```

3. **功能测试**
   - 编写 C 测试程序
   - 验证加速器功能
   - 测试中断机制

### 中期

4. **性能优化**
   - 优化地址解码逻辑
   - 减少响应延迟
   - 提高时钟频率

5. **完整验证**
   - 运行 AI 推理程序
   - 测试大规模矩阵
   - 压力测试

### 长期

6. **物理设计**
   - ASIC 综合
   - 布局布线
   - 时序收敛

---

**创建日期**: 2025-12-03  
**状态**: ✅ 设计完成，待验证  
**作者**: SimpleEdgeAiSoC Team  
**版本**: v0.4.1-ai
