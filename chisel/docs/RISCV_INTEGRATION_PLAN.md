# RISC-V + AI 加速器整合方案

## 🎯 整合目标

将 BitNetScaleAiChip 和 CompactScaleAiChip 作为协处理器整合到 RISC-V 核心中，形成完整的边缘 AI SoC。

## 🏗️ 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    RISC-V AI SoC                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────────────────────┐    │
│  │  RISC-V Core │◄───────►│    System Bus (AXI4)         │    │
│  │  (RV32IMC)   │         │                              │    │
│  └──────────────┘         └──────────────────────────────┘    │
│         │                           │                          │
│         │                           │                          │
│         ▼                           ▼                          │
│  ┌──────────────┐         ┌──────────────────────────────┐    │
│  │   L1 Cache   │         │    Memory Controller         │    │
│  │   32KB I/D   │         │    (DDR3/DDR4)              │    │
│  └──────────────┘         └──────────────────────────────┘    │
│                                     │                          │
│                           ┌─────────┴─────────┐                │
│                           │                   │                │
│                           ▼                   ▼                │
│              ┌────────────────────┐  ┌────────────────────┐   │
│              │ CompactScaleAiChip │  │ BitNetScaleAiChip  │   │
│              │ (传统模型加速器)    │  │ (BitNet加速器)     │   │
│              │                    │  │                    │   │
│              │ • 16个 MAC 单元    │  │ • 16个 BitNet 单元 │   │
│              │ • 1个 8×8 矩阵     │  │ • 2个 16×16 矩阵   │   │
│              │ • AXI4-Lite 接口   │  │ • AXI4-Lite 接口   │   │
│              └────────────────────┘  └────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              外设控制器                               │     │
│  │  • UART  • SPI  • I2C  • GPIO  • Timer              │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 方案对比

### 方案 A: 内存映射 I/O (MMIO) - 推荐 ✅

**架构**:
```
RISC-V Core
    │
    ├─ 0x0000_0000 - 0x0FFF_FFFF: RAM (256 MB)
    ├─ 0x1000_0000 - 0x1000_0FFF: CompactScale (4 KB)
    ├─ 0x1000_1000 - 0x1000_1FFF: BitNetScale (4 KB)
    ├─ 0x2000_0000 - 0x2FFF_FFFF: 外设
    └─ 0x8000_0000 - 0xFFFF_FFFF: Flash ROM
```

**优点**:
- ✅ 简单直接
- ✅ 标准 RISC-V 内存映射
- ✅ 易于软件开发
- ✅ 低延迟访问

**缺点**:
- ⚠️ 占用地址空间
- ⚠️ 需要轮询或中断

### 方案 B: 自定义指令扩展

**架构**:
```
RISC-V Core + 自定义指令
    │
    ├─ ai.matmul rd, rs1, rs2  // 矩阵乘法
    ├─ ai.load rd, addr        // 加载矩阵
    ├─ ai.store rs, addr       // 存储结果
    └─ ai.config imm           // 配置加速器
```

**优点**:
- ✅ 更高效
- ✅ 类似 CPU 指令
- ✅ 编译器可优化

**缺点**:
- ❌ 需要修改 RISC-V 核心
- ❌ 工具链需要更新
- ❌ 复杂度高

### 方案 C: DMA + 中断

**架构**:
```
RISC-V Core
    │
    ├─ DMA Controller
    │   ├─ Channel 0: CompactScale
    │   └─ Channel 1: BitNetScale
    │
    └─ Interrupt Controller
        ├─ IRQ 16: CompactScale Done
        └─ IRQ 17: BitNetScale Done
```

**优点**:
- ✅ 高吞吐量
- ✅ CPU 可以做其他事
- ✅ 适合大数据传输

**缺点**:
- ⚠️ 复杂度中等
- ⚠️ 需要 DMA 控制器

## 🎯 推荐方案：MMIO + DMA + 中断

结合三种方案的优点，采用混合架构：

### 详细设计

#### 1. 地址映射

```
地址范围                    | 功能                  | 大小
---------------------------|----------------------|-------
0x0000_0000 - 0x0FFF_FFFF | RAM                  | 256 MB
0x1000_0000 - 0x1000_0FFF | CompactScale 寄存器   | 4 KB
0x1000_1000 - 0x1000_1FFF | BitNetScale 寄存器    | 4 KB
0x1000_2000 - 0x1000_2FFF | DMA 控制器           | 4 KB
0x2000_0000 - 0x2000_FFFF | UART                 | 64 KB
0x2001_0000 - 0x2001_FFFF | SPI                  | 64 KB
0x2002_0000 - 0x2002_FFFF | I2C                  | 64 KB
0x2003_0000 - 0x2003_FFFF | GPIO                 | 64 KB
0x8000_0000 - 0x8FFF_FFFF | Flash ROM            | 256 MB
```

#### 2. CompactScale 寄存器映射

```
偏移地址  | 寄存器名称        | 读/写 | 说明
---------|------------------|------|------------------
0x000    | CTRL             | R/W  | 控制寄存器
0x004    | STATUS           | R    | 状态寄存器
0x008    | INT_EN           | R/W  | 中断使能
0x00C    | INT_STATUS       | R/W  | 中断状态
0x010    | DMA_SRC          | R/W  | DMA 源地址
0x014    | DMA_DST          | R/W  | DMA 目标地址
0x018    | DMA_LEN          | R/W  | DMA 传输长度
0x01C    | MATRIX_SIZE      | R/W  | 矩阵大小
0x020    | PERF_CYCLES      | R    | 性能计数器
0x024    | PERF_OPS         | R    | 运算计数器
0x100    | MATRIX_A[0]      | R/W  | 矩阵 A 数据
...      | ...              | ...  | ...
0x300    | MATRIX_B[0]      | R/W  | 矩阵 B 数据
...      | ...              | ...  | ...
0x500    | MATRIX_C[0]      | R    | 矩阵 C 结果
```

#### 3. BitNetScale 寄存器映射

```
偏移地址  | 寄存器名称        | 读/写 | 说明
---------|------------------|------|------------------
0x000    | CTRL             | R/W  | 控制寄存器
0x004    | STATUS           | R    | 状态寄存器
0x008    | INT_EN           | R/W  | 中断使能
0x00C    | INT_STATUS       | R/W  | 中断状态
0x010    | DMA_SRC          | R/W  | DMA 源地址
0x014    | DMA_DST          | R/W  | DMA 目标地址
0x018    | DMA_LEN          | R/W  | DMA 传输长度
0x01C    | MATRIX_SIZE      | R/W  | 矩阵大小
0x020    | CONFIG           | R/W  | BitNet 配置
0x024    | SPARSITY_EN      | R/W  | 稀疏性优化使能
0x028    | PERF_CYCLES      | R    | 性能计数器
0x02C    | PERF_OPS         | R    | 运算计数器
0x100    | ACTIVATION[0]    | R/W  | 激活值数据
...      | ...              | ...  | ...
0x300    | WEIGHT[0]        | R/W  | 权重数据 (2-bit)
...      | ...              | ...  | ...
0x500    | RESULT[0]        | R    | 结果数据
```

#### 4. 中断控制

```
中断号 | 中断源                    | 优先级
------|--------------------------|-------
16    | CompactScale 计算完成     | 高
17    | BitNetScale 计算完成      | 高
18    | CompactScale DMA 完成     | 中
19    | BitNetScale DMA 完成      | 中
20    | CompactScale 错误         | 高
21    | BitNetScale 错误          | 高
```

## 💻 软件接口设计

### 1. 驱动程序 API

```c
// ai_accelerator.h

#ifndef AI_ACCELERATOR_H
#define AI_ACCELERATOR_H

#include <stdint.h>

// 加速器类型
typedef enum {
    AI_ACCEL_COMPACT,  // CompactScale (传统模型)
    AI_ACCEL_BITNET    // BitNetScale (BitNet 模型)
} ai_accel_type_t;

// 矩阵数据结构
typedef struct {
    void *data;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
} ai_matrix_t;

// 初始化加速器
int ai_accel_init(ai_accel_type_t type);

// 矩阵乘法: C = A × B
int ai_matmul(
    ai_accel_type_t type,
    const ai_matrix_t *A,
    const ai_matrix_t *B,
    ai_matrix_t *C
);

// 异步矩阵乘法（使用 DMA + 中断）
int ai_matmul_async(
    ai_accel_type_t type,
    const ai_matrix_t *A,
    const ai_matrix_t *B,
    ai_matrix_t *C,
    void (*callback)(void*)
);

// 等待计算完成
int ai_wait(ai_accel_type_t type);

// 获取性能统计
typedef struct {
    uint32_t cycles;
    uint32_t ops;
    float throughput;  // ops/cycle
} ai_perf_stats_t;

int ai_get_perf(ai_accel_type_t type, ai_perf_stats_t *stats);

// 配置 BitNet 加速器
int ai_bitnet_config(
    bool sparsity_enable,
    uint8_t activation_bits
);

#endif // AI_ACCELERATOR_H
```

### 2. 驱动程序实现示例

```c
// ai_accelerator.c

#include "ai_accelerator.h"
#include <string.h>

// 寄存器基地址
#define COMPACT_BASE  0x10000000
#define BITNET_BASE   0x10001000

// 寄存器偏移
#define REG_CTRL       0x000
#define REG_STATUS     0x004
#define REG_INT_EN     0x008
#define REG_DMA_SRC    0x010
#define REG_DMA_DST    0x014
#define REG_DMA_LEN    0x018
#define REG_MATRIX_SIZE 0x01C

// 控制位
#define CTRL_START     (1 << 0)
#define CTRL_RESET     (1 << 1)
#define CTRL_DMA_EN    (1 << 2)

// 状态位
#define STATUS_BUSY    (1 << 0)
#define STATUS_DONE    (1 << 1)
#define STATUS_ERROR   (1 << 2)

// 寄存器访问宏
#define REG_WRITE(base, offset, value) \
    (*(volatile uint32_t*)((base) + (offset)) = (value))

#define REG_READ(base, offset) \
    (*(volatile uint32_t*)((base) + (offset)))

// 初始化加速器
int ai_accel_init(ai_accel_type_t type) {
    uint32_t base = (type == AI_ACCEL_COMPACT) ? COMPACT_BASE : BITNET_BASE;
    
    // 复位加速器
    REG_WRITE(base, REG_CTRL, CTRL_RESET);
    
    // 等待复位完成
    while (REG_READ(base, REG_STATUS) & STATUS_BUSY);
    
    // 使能中断
    REG_WRITE(base, REG_INT_EN, 0x1);
    
    return 0;
}

// 同步矩阵乘法
int ai_matmul(
    ai_accel_type_t type,
    const ai_matrix_t *A,
    const ai_matrix_t *B,
    ai_matrix_t *C
) {
    uint32_t base = (type == AI_ACCEL_COMPACT) ? COMPACT_BASE : BITNET_BASE;
    
    // 检查矩阵大小
    if (A->cols != B->rows) {
        return -1;  // 维度不匹配
    }
    
    // 配置矩阵大小
    uint32_t size = (A->rows << 16) | (B->cols << 8) | A->cols;
    REG_WRITE(base, REG_MATRIX_SIZE, size);
    
    // 配置 DMA
    REG_WRITE(base, REG_DMA_SRC, (uint32_t)A->data);
    REG_WRITE(base, REG_DMA_DST, (uint32_t)C->data);
    REG_WRITE(base, REG_DMA_LEN, A->rows * B->cols * sizeof(int32_t));
    
    // 启动计算
    REG_WRITE(base, REG_CTRL, CTRL_START | CTRL_DMA_EN);
    
    // 等待完成
    while (!(REG_READ(base, REG_STATUS) & STATUS_DONE)) {
        // 可以在这里让出 CPU
    }
    
    // 检查错误
    if (REG_READ(base, REG_STATUS) & STATUS_ERROR) {
        return -2;  // 计算错误
    }
    
    return 0;
}

// 异步矩阵乘法
int ai_matmul_async(
    ai_accel_type_t type,
    const ai_matrix_t *A,
    const ai_matrix_t *B,
    ai_matrix_t *C,
    void (*callback)(void*)
) {
    // 注册回调函数
    // 配置中断处理
    // 启动计算
    // 立即返回
    
    // 实现略...
    return 0;
}
```

### 3. 应用程序示例

```c
// example.c - BitNet-3B 推理示例

#include "ai_accelerator.h"
#include <stdio.h>

#define HIDDEN_SIZE 2048
#define NUM_LAYERS 26

// BitNet-3B 单层推理
void bitnet_layer_inference(
    const ai_matrix_t *input,    // [seq_len, hidden_size]
    const ai_matrix_t *weight,   // [hidden_size, hidden_size]
    ai_matrix_t *output          // [seq_len, hidden_size]
) {
    // 使用 BitNet 加速器
    ai_matmul(AI_ACCEL_BITNET, input, weight, output);
}

// BitNet-3B 完整推理
void bitnet_3b_inference(
    const ai_matrix_t *input,
    ai_matrix_t *output
) {
    ai_matrix_t layer_input = *input;
    ai_matrix_t layer_output;
    
    // 初始化 BitNet 加速器
    ai_accel_init(AI_ACCEL_BITNET);
    
    // 配置 BitNet 参数
    ai_bitnet_config(true, 16);  // 使能稀疏性，16-bit 激活值
    
    // 26 层推理
    for (int i = 0; i < NUM_LAYERS; i++) {
        printf("Layer %d...\n", i);
        
        // 加载权重（从 Flash 或 RAM）
        ai_matrix_t weight;
        load_layer_weight(i, &weight);
        
        // 矩阵乘法
        bitnet_layer_inference(&layer_input, &weight, &layer_output);
        
        // 下一层的输入
        layer_input = layer_output;
    }
    
    *output = layer_output;
    
    // 获取性能统计
    ai_perf_stats_t stats;
    ai_get_perf(AI_ACCEL_BITNET, &stats);
    printf("Performance: %u cycles, %u ops, %.2f ops/cycle\n",
           stats.cycles, stats.ops, stats.throughput);
}

int main() {
    // 输入数据
    int32_t input_data[HIDDEN_SIZE];
    int32_t output_data[HIDDEN_SIZE];
    
    ai_matrix_t input = {
        .data = input_data,
        .rows = 1,
        .cols = HIDDEN_SIZE,
        .stride = HIDDEN_SIZE
    };
    
    ai_matrix_t output = {
        .data = output_data,
        .rows = 1,
        .cols = HIDDEN_SIZE,
        .stride = HIDDEN_SIZE
    };
    
    // 运行推理
    printf("Starting BitNet-3B inference...\n");
    bitnet_3b_inference(&input, &output);
    printf("Inference complete!\n");
    
    return 0;
}
```

## 🔧 硬件实现

### 1. 顶层模块

```scala
// RiscvAiSoC.scala

package riscv.ai

import chisel3._
import chisel3.util._

class RiscvAiSoC extends Module {
  val io = IO(new Bundle {
    // 外部接口
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    val spi_sck = Output(Bool())
    val spi_mosi = Output(Bool())
    val spi_miso = Input(Bool())
    val gpio = Output(UInt(32.W))
  })
  
  // RISC-V 核心（使用 Rocket Chip 或 BOOM）
  val riscv = Module(new RocketCore())
  
  // 系统总线（AXI4）
  val systemBus = Module(new AXI4Crossbar())
  
  // AI 加速器
  val compactScale = Module(new CompactScaleAiChip())
  val bitnetScale = Module(new BitNetScaleAiChip())
  
  // DMA 控制器
  val dma = Module(new DMAController())
  
  // 内存控制器
  val memCtrl = Module(new MemoryController())
  
  // 外设
  val uart = Module(new UART())
  val spi = Module(new SPI())
  val gpio_ctrl = Module(new GPIO())
  
  // 中断控制器
  val intCtrl = Module(new InterruptController())
  
  // 连接系统总线
  systemBus.io.master <> riscv.io.mem
  systemBus.io.slaves(0) <> memCtrl.io.axi
  systemBus.io.slaves(1) <> compactScale.io.axi
  systemBus.io.slaves(2) <> bitnetScale.io.axi
  systemBus.io.slaves(3) <> dma.io.axi
  systemBus.io.slaves(4) <> uart.io.axi
  systemBus.io.slaves(5) <> spi.io.axi
  systemBus.io.slaves(6) <> gpio_ctrl.io.axi
  
  // 连接中断
  intCtrl.io.irq(16) := compactScale.io.status.done
  intCtrl.io.irq(17) := bitnetScale.io.status.done
  riscv.io.interrupts := intCtrl.io.cpu_irq
  
  // 连接外设
  io.uart_tx := uart.io.tx
  uart.io.rx := io.uart_rx
  io.spi_sck := spi.io.sck
  io.spi_mosi := spi.io.mosi
  spi.io.miso := io.spi_miso
  io.gpio := gpio_ctrl.io.out
}
```

### 2. DMA 控制器

```scala
// DMAController.scala

class DMAController extends Module {
  val io = IO(new Bundle {
    val axi = new AXI4LiteIO()
    val mem = new AXI4MasterIO()
    val done = Output(Bool())
  })
  
  // DMA 寄存器
  val srcAddr = RegInit(0.U(32.W))
  val dstAddr = RegInit(0.U(32.W))
  val length = RegInit(0.U(32.W))
  val ctrl = RegInit(0.U(32.W))
  
  // DMA 状态机
  val sIdle :: sRead :: sWrite :: sDone :: Nil = Enum(4)
  val state = RegInit(sIdle)
  
  // DMA 逻辑
  switch(state) {
    is(sIdle) {
      when(ctrl(0)) {
        state := sRead
      }
    }
    is(sRead) {
      // 从源地址读取数据
      // ...
      state := sWrite
    }
    is(sWrite) {
      // 写入目标地址
      // ...
      state := sDone
    }
    is(sDone) {
      io.done := true.B
      state := sIdle
    }
  }
}
```

## 📊 性能预估

### 系统性能

| 组件 | 频率 | 性能 |
|------|------|------|
| RISC-V Core | 100 MHz | 100 MIPS |
| CompactScale | 100 MHz | 1.6 GOPS |
| BitNetScale | 100 MHz | 4.8 GOPS |
| 系统总线 | 100 MHz | 400 MB/s |
| DDR3 | 800 MHz | 6.4 GB/s |

### 应用性能

| 应用 | 性能 | 说明 |
|------|------|------|
| BitNet-1B | 2,632 tok/s | 实时推理 |
| BitNet-3B | 893 tok/s | 实时推理 |
| TinyBERT | 50 infer/s | 文本分类 |
| 图像分类 | 30 fps | MobileNet |

## 💰 成本估算

### 芯片面积

| 组件 | 面积 (mm²) | 占比 |
|------|-----------|------|
| RISC-V Core | 0.5 | 10% |
| CompactScale | 1.0 | 20% |
| BitNetScale | 0.8 | 16% |
| 内存 (128KB) | 1.5 | 30% |
| 外设 | 0.5 | 10% |
| 其他 | 0.7 | 14% |
| **总计** | **5.0** | **100%** |

### 流片成本 (40nm)

| 项目 | 成本 |
|------|------|
| NRE (一次性) | $50K |
| 掩膜 | $30K |
| 测试 | $20K |
| 单片成本 (1K) | $8 |
| 单片成本 (10K) | $5 |
| 单片成本 (100K) | $3 |

## 🚀 开发路线图

### 阶段 1: 仿真验证 (1-2 个月) ✅

- [x] CompactScale 设计完成
- [x] BitNetScale 设计完成
- [x] 基础功能测试通过
- [ ] RISC-V 集成设计
- [ ] 系统级仿真

### 阶段 2: FPGA 原型 (2-3 个月)

- [ ] 选择 FPGA 平台 (Xilinx ZCU102)
- [ ] 综合和布局布线
- [ ] FPGA 验证
- [ ] 软件驱动开发
- [ ] 性能测试

### 阶段 3: 流片准备 (3-4 个月)

- [ ] 后端设计
- [ ] 时序收敛
- [ ] 功耗分析
- [ ] DFT 插入
- [ ] 流片

### 阶段 4: 量产 (6-12 个月)

- [ ] 测试芯片验证
- [ ] 量产工艺优化
- [ ] 软件生态建设
- [ ] 市场推广

## 📝 总结

### 推荐方案

**MMIO + DMA + 中断混合架构**

**优点**:
- ✅ 简单易用
- ✅ 高性能
- ✅ 标准接口
- ✅ 易于软件开发

**关键特性**:
- RISC-V RV32IMC 核心
- 双 AI 加速器（CompactScale + BitNetScale）
- AXI4 系统总线
- DMA 支持
- 中断驱动
- 完整的软件栈

**性能目标**:
- BitNet-3B: 893 tokens/秒
- 功耗: <200mW
- 成本: <$5 (量产)

**市场定位**:
- 边缘 AI 推理
- IoT 智能设备
- 移动 AI 应用
- 低功耗数据中心

---

**文档版本**: v1.0
**创建时间**: 2025-11-13
**状态**: 设计完成，待实现
