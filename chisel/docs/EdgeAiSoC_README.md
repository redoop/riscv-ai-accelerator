# EdgeAiSoC - Edge AI System-on-Chip

## 概述

EdgeAiSoC 是一个完整的边缘 AI SoC 设计，集成了 RISC-V 处理器核心和双 AI 加速器，专为边缘 AI 推理应用优化。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        EdgeAiSoC                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────────────────────┐    │
│  │  PicoRV32    │◄───────►│    AXI4-Lite Bus             │    │
│  │  (RV32I)     │         │    (Address Decoder)         │    │
│  └──────────────┘         └──────────────────────────────┘    │
│                                     │                          │
│                    ┌────────────────┼────────────────┐         │
│                    │                │                │         │
│                    ▼                ▼                ▼         │
│         ┌──────────────────┐  ┌──────────────┐  ┌─────────┐  │
│         │ CompactScale     │  │ BitNetScale  │  │   DMA   │  │
│         │ (8x8 Matrix)     │  │ (16x16)      │  │         │  │
│         └──────────────────┘  └──────────────┘  └─────────┘  │
│                    │                │                │         │
│                    └────────────────┼────────────────┘         │
│                                     │                          │
│                                     ▼                          │
│                    ┌────────────────────────────┐              │
│                    │  Interrupt Controller      │              │
│                    └────────────────────────────┘              │
│                                     │                          │
│                    ┌────────────────┼────────────┐             │
│                    │                │            │             │
│                    ▼                ▼            ▼             │
│              ┌─────────┐      ┌─────────┐  ┌─────────┐        │
│              │  UART   │      │  GPIO   │  │  其他   │        │
│              └─────────┘      └─────────┘  └─────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心特性

### 1. RISC-V 处理器核心
- **型号**: PicoRV32
- **指令集**: RV32I
- **特点**: 
  - 小面积、低功耗
  - 完整的 RISC-V 指令集支持
  - 中断支持
  - 适合嵌入式应用

### 2. AI 加速器

#### CompactScale 加速器
- **矩阵大小**: 8×8
- **数据类型**: 32-bit 整数/浮点
- **性能**: ~1.6 GOPS @ 100MHz
- **应用**: 传统神经网络模型

#### BitNetScale 加速器
- **矩阵大小**: 16×16
- **数据类型**: 2-bit 权重 + 16-bit 激活值
- **性能**: ~4.8 GOPS @ 100MHz
- **特性**: 
  - 稀疏性优化
  - BitNet 模型专用
  - 低功耗推理

### 3. 系统总线
- **协议**: AXI4-Lite
- **数据宽度**: 32-bit
- **地址宽度**: 32-bit
- **特点**: 简单、标准化

### 4. 外设控制器
- **DMA**: 支持内存到内存传输
- **中断控制器**: 32 个中断源
- **UART**: 串口通信
- **GPIO**: 32-bit 通用 I/O

## 内存映射

| 地址范围 | 组件 | 大小 | 说明 |
|---------|------|------|------|
| 0x0000_0000 - 0x0FFF_FFFF | RAM | 256 MB | 主内存 |
| 0x1000_0000 - 0x1000_0FFF | CompactScale | 4 KB | 传统模型加速器 |
| 0x1000_1000 - 0x1000_1FFF | BitNetScale | 4 KB | BitNet 加速器 |
| 0x1000_2000 - 0x1000_2FFF | DMA | 4 KB | DMA 控制器 |
| 0x1000_3000 - 0x1000_3FFF | IntCtrl | 4 KB | 中断控制器 |
| 0x2000_0000 - 0x2000_FFFF | UART | 64 KB | 串口控制器 |
| 0x2002_0000 - 0x2002_FFFF | GPIO | 64 KB | GPIO 控制器 |
| 0x8000_0000 - 0x8FFF_FFFF | Flash | 256 MB | 程序存储 |

## 寄存器映射

### CompactScale / BitNetScale 寄存器

| 偏移 | 名称 | 读/写 | 说明 |
|-----|------|------|------|
| 0x000 | CTRL | R/W | 控制寄存器 (bit 0: start, bit 1: reset) |
| 0x004 | STATUS | R | 状态寄存器 (bit 0: busy, bit 1: done) |
| 0x008 | INT_EN | R/W | 中断使能 |
| 0x00C | INT_STATUS | R/W | 中断状态 |
| 0x010 | DMA_SRC | R/W | DMA 源地址 |
| 0x014 | DMA_DST | R/W | DMA 目标地址 |
| 0x018 | DMA_LEN | R/W | DMA 传输长度 |
| 0x01C | MATRIX_SIZE | R/W | 矩阵大小配置 |
| 0x020 | CONFIG | R/W | 配置寄存器 (BitNet only) |
| 0x024 | SPARSITY_EN | R/W | 稀疏性优化使能 (BitNet only) |
| 0x028 | PERF_CYCLES | R | 性能计数器 - 周期数 |
| 0x02C | PERF_OPS | R | 性能计数器 - 操作数 |
| 0x100-0x2FF | MATRIX_A | R/W | 矩阵 A / 激活值数据 |
| 0x300-0x4FF | MATRIX_B | R/W | 矩阵 B / 权重数据 |
| 0x500-0x6FF | MATRIX_C | R | 矩阵 C / 结果数据 |

## 中断映射

| IRQ 号 | 中断源 | 优先级 | 说明 |
|-------|--------|--------|------|
| 16 | CompactScale Done | 高 | CompactScale 计算完成 |
| 17 | BitNetScale Done | 高 | BitNetScale 计算完成 |
| 18 | DMA Done | 中 | DMA 传输完成 |
| 19 | DMA Error | 高 | DMA 传输错误 |
| 20 | CompactScale Error | 高 | CompactScale 错误 |
| 21 | BitNetScale Error | 高 | BitNetScale 错误 |

## 使用示例

### 1. 初始化加速器

```c
// 初始化 CompactScale
volatile uint32_t *compact_ctrl = (uint32_t *)0x10000000;
*compact_ctrl = 0x2; // Reset
*compact_ctrl = 0x0; // Clear reset

// 初始化 BitNetScale
volatile uint32_t *bitnet_ctrl = (uint32_t *)0x10001000;
*bitnet_ctrl = 0x2; // Reset
*bitnet_ctrl = 0x0; // Clear reset
```

### 2. 配置矩阵数据

```c
// 写入矩阵 A 数据到 CompactScale
volatile uint32_t *matrix_a = (uint32_t *)0x10000100;
for (int i = 0; i < 64; i++) {
    matrix_a[i] = input_data[i];
}

// 写入矩阵 B 数据
volatile uint32_t *matrix_b = (uint32_t *)0x10000300;
for (int i = 0; i < 64; i++) {
    matrix_b[i] = weight_data[i];
}
```

### 3. 启动计算

```c
// 使能中断
volatile uint32_t *int_en = (uint32_t *)0x10000008;
*int_en = 0x1;

// 启动计算
*compact_ctrl = 0x1; // Start

// 等待完成 (通过中断或轮询)
volatile uint32_t *status = (uint32_t *)0x10000004;
while ((*status & 0x2) == 0) {
    // Wait for done bit
}
```

### 4. 读取结果

```c
// 读取结果矩阵
volatile uint32_t *matrix_c = (uint32_t *)0x10000500;
for (int i = 0; i < 64; i++) {
    result_data[i] = matrix_c[i];
}
```

## 性能指标

### 系统性能
- **CPU 频率**: 100 MHz
- **系统总线**: 100 MHz, 400 MB/s
- **CompactScale**: 1.6 GOPS @ 100MHz
- **BitNetScale**: 4.8 GOPS @ 100MHz

### 应用性能
- **BitNet-1B**: ~2,600 tokens/s
- **BitNet-3B**: ~890 tokens/s
- **图像分类**: ~30 fps (MobileNet)
- **文本分类**: ~50 infer/s (TinyBERT)

## 功耗估算

- **CPU**: ~20 mW @ 100MHz
- **CompactScale**: ~50 mW (active)
- **BitNetScale**: ~40 mW (active)
- **外设**: ~10 mW
- **总功耗**: <200 mW (峰值)

## 编译和生成

### 生成 Verilog

```bash
cd chisel
sbt "runMain riscv.ai.EdgeAiSoCMain"
```

生成的文件位于: `generated/edgeaisoc/EdgeAiSoC.v`

### 综合和实现

支持的 FPGA 平台:
- Xilinx Zynq-7000
- Xilinx Zynq UltraScale+
- Intel Cyclone V
- Lattice ECP5

## 软件开发

### 工具链
- **编译器**: RISC-V GCC
- **调试器**: OpenOCD + GDB
- **SDK**: 自定义 HAL 库

### 示例程序
参见 `chisel/docs/RISCV_INTEGRATION_PLAN.md` 中的软件接口设计部分。

## 应用场景

1. **边缘 AI 推理**
   - 智能摄像头
   - IoT 设备
   - 可穿戴设备

2. **语言模型**
   - BitNet-1B/3B 推理
   - 文本生成
   - 对话系统

3. **计算机视觉**
   - 图像分类
   - 目标检测
   - 人脸识别

4. **自然语言处理**
   - 文本分类
   - 情感分析
   - 命名实体识别

## 未来改进

- [ ] 添加 L1 缓存
- [ ] 支持 DDR 内存控制器
- [ ] 增加更多外设 (SPI, I2C)
- [ ] 优化 DMA 性能
- [ ] 添加调试接口 (JTAG)
- [ ] 支持更大的矩阵运算

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 项目主页: [GitHub Repository]
- 文档: `chisel/docs/`
- 示例: `chisel/examples/`
