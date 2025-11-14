# RISC-V AI 加速器系统集成文档

## 概述

本项目将 PicoRV32 RISC-V 处理器与自定义 AI 加速器通过 PCPI (Pico Co-Processor Interface) 接口集成，形成完整的 RISC-V AI 加速器系统。

## 架构设计

### 系统组成

```
┌─────────────────────────────────────────────────────────┐
│                  RiscvAiChip (顶层)                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │            RiscvAiSystem (集成层)                  │  │
│  │  ┌──────────────────┐    ┌──────────────────┐    │  │
│  │  │  PicoRV32 CPU    │    │  AI Accelerator  │    │  │
│  │  │  (BlackBox)      │◄──►│  (Chisel)        │    │  │
│  │  │                  │PCPI│                  │    │  │
│  │  │  - RV32I Core    │    │  - 16 MAC Units  │    │  │
│  │  │  - Memory I/F    │    │  - Matrix Mult   │    │  │
│  │  │  - IRQ Support   │    │  - AXI-Lite I/F  │    │  │
│  │  └──────────────────┘    └──────────────────┘    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 关键接口

#### 1. PCPI 接口 (CPU ↔ AI 加速器)

PCPI (Pico Co-Processor Interface) 是 PicoRV32 的协处理器接口，用于连接外部加速器。

**信号定义:**
- `pcpi_valid`: CPU 发起协处理器请求
- `pcpi_insn`: 当前指令
- `pcpi_rs1/rs2`: 源操作数
- `pcpi_wr`: 协处理器写回使能
- `pcpi_rd`: 协处理器结果
- `pcpi_wait`: 协处理器忙
- `pcpi_ready`: 协处理器完成

**地址映射:**
```
0x80000000 - 0x8000FFFF: AI 加速器地址空间
  0x80000000 - 0x800000FF: 矩阵 A (256 bytes)
  0x80000100 - 0x800001FF: 矩阵 B (256 bytes)
  0x80000200 - 0x800002FF: 结果矩阵 C (256 bytes)
  0x80000300 - 0x800003FF: 控制寄存器
  0x80000400+: 内部存储器
```

#### 2. 内存接口 (CPU ↔ 外部存储)

标准的 PicoRV32 内存接口:
- `mem_valid`: 内存请求有效
- `mem_ready`: 内存响应就绪
- `mem_addr`: 内存地址
- `mem_wdata/wstrb`: 写数据和字节使能
- `mem_rdata`: 读数据

#### 3. AXI-Lite 接口 (内部: PCPI ↔ AI 加速器)

AI 加速器使用简化的 AXI-Lite 接口:
- 写通道: `awaddr`, `awvalid`, `awready`, `wdata`, `wvalid`, `wready`
- 读通道: `araddr`, `arvalid`, `arready`, `rdata`, `rvalid`, `rready`

## 文件结构

```
chisel/
├── src/main/
│   ├── scala/
│   │   ├── RiscvAiIntegration.scala    # 主集成文件
│   │   ├── CompactScaleDesign.scala    # AI 加速器
│   │   ├── MacUnit.scala               # MAC 单元
│   │   └── MatrixMultiplier.scala      # 矩阵乘法器
│   └── rtl/
│       └── picorv32.v                  # PicoRV32 Verilog 源码
├── src/test/scala/
│   └── RiscvAiIntegrationTest.scala    # 集成测试
└── docs/
    └── INTEGRATION.md                  # 本文档
```

## 核心模块说明

### 1. PicoRV32BlackBox

```scala
class PicoRV32BlackBox extends BlackBox with HasBlackBoxResource
```

将 Verilog 实现的 PicoRV32 封装为 Chisel BlackBox，使其可以在 Chisel 设计中实例化。

**特性:**
- RV32I 指令集
- 可配置的性能选项
- PCPI 协处理器接口
- 中断支持
- Trace 接口

### 2. RiscvAiSystem

```scala
class RiscvAiSystem extends Module
```

核心集成模块，连接 CPU 和 AI 加速器。

**功能:**
- 实例化 PicoRV32 和 AI 加速器
- PCPI 到 AXI-Lite 协议转换
- 地址解码和路由
- 状态管理

**PCPI 状态机:**
```
IDLE → READ/WRITE → DONE → IDLE
```

### 3. RiscvAiChip

```scala
class RiscvAiChip extends Module
```

顶层封装模块，提供简化的外部接口。

## 使用方法

### 1. 编译生成 Verilog

```bash
cd chisel
sbt "runMain riscv.ai.RiscvAiChipMain"
```

生成的 Verilog 文件位于 `generated/` 目录。

### 2. 运行测试

```bash
sbt test
```

### 3. 软件编程接口

在 RISC-V 软件中访问 AI 加速器:

```c
// AI 加速器基地址
#define AI_BASE 0x80000000

// 寄存器偏移
#define MATRIX_A_BASE   (AI_BASE + 0x000)
#define MATRIX_B_BASE   (AI_BASE + 0x100)
#define RESULT_BASE     (AI_BASE + 0x200)
#define CTRL_REG        (AI_BASE + 0x300)
#define STATUS_REG      (AI_BASE + 0x304)

// 写入矩阵数据
void write_matrix_a(int row, int col, int value) {
    volatile int *addr = (int*)(MATRIX_A_BASE + (row * 8 + col) * 4);
    *addr = value;
}

// 启动矩阵乘法
void start_matmul() {
    volatile int *ctrl = (int*)CTRL_REG;
    *ctrl = 1;  // 设置启动位
}

// 等待完成
void wait_done() {
    volatile int *status = (int*)STATUS_REG;
    while ((*status & 0x2) == 0);  // 等待 done 位
}

// 读取结果
int read_result(int row, int col) {
    volatile int *addr = (int*)(RESULT_BASE + (row * 8 + col) * 4);
    return *addr;
}
```

## 性能特性

### AI 加速器配置

- **MAC 单元**: 16 个并行 MAC 单元
- **矩阵规模**: 8x8 矩阵乘法
- **数据宽度**: 32 位
- **内部存储**: 512 深度

### 预期性能

- **矩阵乘法延迟**: ~64 周期 (8x8)
- **MAC 吞吐量**: 16 ops/cycle
- **峰值性能**: 16 GOPS @ 1GHz

## 设计考虑

### 1. 时钟域

当前设计使用单一时钟域。如需要，可以添加:
- 异步 FIFO 用于跨时钟域
- 时钟门控用于功耗优化

### 2. 复位策略

- CPU 使用低电平有效复位 (`resetn`)
- AI 加速器使用高电平有效复位 (`reset`)
- 在 BlackBox 连接处进行极性转换

### 3. 地址空间

AI 加速器占用 64KB 地址空间 (0x80000000-0x8000FFFF)，不与标准 RISC-V 内存映射冲突。

### 4. 错误处理

- 非法地址访问返回 0
- PCPI 超时机制 (可选)
- 状态寄存器指示错误

## 扩展方向

### 1. 添加 DMA 支持

```scala
class DmaController extends Module {
  // 自动传输数据到 AI 加速器
}
```

### 2. 多加速器支持

```scala
val aiAccels = Seq.fill(4)(Module(new CompactScaleAiChip))
// 地址解码选择不同加速器
```

### 3. 中断驱动

```scala
// AI 加速器完成时产生中断
when(aiAccel.io.status.done) {
  irq_pending := true.B
}
```

### 4. 缓存一致性

添加缓存一致性协议，确保 CPU 和加速器数据一致。

## 调试支持

### 1. Trace 接口

PicoRV32 提供 trace 接口，可用于:
- 指令跟踪
- 分支跟踪
- 性能分析

### 2. 性能计数器

AI 加速器提供 4 个性能计数器:
- Counter 0: 忙周期数
- Counter 1: 完成操作数
- Counter 2: 活跃 MAC 单元数
- Counter 3: 工作计数器

### 3. 仿真支持

使用 Verilator 或 VCS 进行仿真:

```bash
# 生成 Verilog
sbt "runMain riscv.ai.RiscvAiChipMain"

# Verilator 仿真
verilator --cc generated/RiscvAiChip.v --exe sim_main.cpp
make -C obj_dir -f VRiscvAiChip.mk
./obj_dir/VRiscvAiChip
```

## 参考资料

- [PicoRV32 GitHub](https://github.com/YosysHQ/picorv32)
- [RISC-V Spec](https://riscv.org/technical/specifications/)
- [Chisel Documentation](https://www.chisel-lang.org/)
- [AXI Protocol Spec](https://developer.arm.com/documentation/ihi0022/latest/)

## 许可证

- PicoRV32: ISC License
- AI 加速器: 根据项目许可证

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
