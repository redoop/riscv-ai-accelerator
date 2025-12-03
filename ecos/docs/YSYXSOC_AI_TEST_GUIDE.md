# ysyxSoC AI 加速器测试指南

**日期**: 2025-12-03  
**状态**: ✅ 测试程序就绪

---

## 📋 测试方案

### 方案 1: 独立测试程序（推荐）

使用专门为 ysyxSoC 编写的测试程序。

**优点**:
- ✅ 简单直接
- ✅ 无依赖
- ✅ 易于调试

**位置**: `/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage/test_ai_accel.c`

### 方案 2: 使用 chisel/software

复用现有的测试程序，需要适配地址映射。

**优点**:
- ✅ 功能完整
- ✅ 已验证
- ✅ 包含图形界面

**位置**: `/opt/github/riscv-ai-accelerator/chisel/software/examples/`

---

## 🚀 方案 1: 独立测试（推荐）

### 测试内容

| 测试项 | 说明 |
|--------|------|
| **CompactAccel** | 4x4 矩阵乘法 |
| **BitNetAccel** | 4x4 BitNet 计算 |
| **寄存器读写** | 验证所有寄存器 |
| **性能计数** | 测量计算周期 |
| **结果验证** | 检查计算结果 |

### 快速开始

```bash
cd /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage

# 1. 编译测试程序（需要 RISC-V 工具链）
chmod +x compile_test.sh
./compile_test.sh

# 2. 将二进制文件复制为启动文件
cp test_ai_accel.bin hello-minirv-ysyxsoc.bin

# 3. 运行仿真
./obj_dir/VysyxSoCTop
```

### 预期输出

```
========================================
AI Accelerator Test for ysyxSoC
========================================

=== Testing CompactAccel ===
Matrix size: 4x4
Writing test matrices...
Starting computation...
Computation complete!
Cycles: 0x00000040
Result[0]: 0x00000078
CompactAccel test PASSED

=== Testing BitNetAccel ===
Matrix size: 4x4
Writing test data...
Starting computation...
Computation complete!
Cycles: 0x00000030
Result[0]: 0xFFFFFFE8
BitNetAccel test PASSED

========================================
All tests PASSED!
========================================
```

---

## 🔧 方案 2: 使用 chisel/software

### 步骤 1: 适配地址映射

创建 `hal_ysyxsoc.h`:

```c
// hal_ysyxsoc.h - HAL for ysyxSoC platform

#ifndef HAL_YSYXSOC_H
#define HAL_YSYXSOC_H

#include <stdint.h>

// AI Accelerators (ysyxSoC addresses)
#define COMPACT_BASE 0x20000000
#define BITNET_BASE  0x20001000

// Flash and PSRAM (ysyxSoC addresses)
#define FLASH_CTRL_BASE 0x20002000
#define PSRAM_CTRL_BASE 0x20003000
#define FLASH_MEM_BASE  0x30000000
#define PSRAM_MEM_BASE  0x04000000

// UART (ysyxSoC address - 需要确认)
#define UART_BASE 0x10000000

// LCD - 不可用（ysyxSoC 没有）
// GPIO - 使用 ysyxSoC 的 GPIO

#endif
```

### 步骤 2: 修改测试程序

```bash
cd /opt/github/riscv-ai-accelerator/chisel/software

# 复制并修改 ai_demo.c
cp examples/ai_demo.c examples/ai_demo_ysyxsoc.c

# 编辑文件，替换：
# #include "../lib/hal.h" 
# 为：
# #include "hal_ysyxsoc.h"

# 移除 LCD 相关代码（ysyxSoC 没有 LCD）
```

### 步骤 3: 编译

```bash
# 使用 chisel/software 的 Makefile
make PROG=ai_demo_ysyxsoc

# 或手动编译
riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 \
  -nostdlib -nostartfiles \
  -T linker.ld \
  -o ai_demo_ysyxsoc.elf \
  examples/ai_demo_ysyxsoc.c lib/hal.c

riscv64-unknown-elf-objcopy -O binary \
  ai_demo_ysyxsoc.elf ai_demo_ysyxsoc.bin
```

### 步骤 4: 运行

```bash
cd /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage

# 复制二进制文件
cp ../../chisel/software/build/ai_demo_ysyxsoc.bin hello-minirv-ysyxsoc.bin

# 运行仿真
./obj_dir/VysyxSoCTop
```

---

## 📊 测试矩阵

### CompactAccel 测试

| 测试 | 矩阵大小 | 输入 | 预期输出 | 状态 |
|------|---------|------|---------|------|
| 基础 | 2x2 | A=I, B=I | C=I | ⏳ |
| 标准 | 4x4 | A=[0..15], B=[1..16] | C=计算值 | ⏳ |
| 大矩阵 | 8x8 | 随机 | 验证 | ⏳ |

### BitNetAccel 测试

| 测试 | 矩阵大小 | 权重编码 | 预期 | 状态 |
|------|---------|---------|------|------|
| 全零 | 4x4 | 00 (零) | 0 | ⏳ |
| 全正 | 4x4 | 01 (+1) | 累加 | ⏳ |
| 全负 | 4x4 | 10 (-1) | 负累加 | ⏳ |
| 混合 | 4x4 | 混合 | 验证 | ⏳ |

---

## 🐛 调试技巧

### 1. 检查地址映射

```c
// 验证地址是否正确
volatile uint32_t* test_addr = (volatile uint32_t*)0x20000000;
*test_addr = 0x12345678;
uint32_t readback = *test_addr;
// 应该读回 0x12345678
```

### 2. 单步测试

```c
// 逐个寄存器测试
write_reg(COMPACT_BASE, REG_SIZE, 4);
uint32_t size = read_reg(COMPACT_BASE, REG_SIZE);
// 验证 size == 4
```

### 3. 超时检测

```c
uint32_t timeout = 100000;
while (timeout-- > 0) {
    if (done) break;
}
if (timeout == 0) {
    // 超时错误
}
```

### 4. 波形查看

```bash
# 如果仿真支持波形输出
./obj_dir/VysyxSoCTop --trace
gtkwave dump.vcd
```

---

## ⚠️ 注意事项

### 地址映射差异

| 组件 | SimpleEdgeAiSoC | ysyxSoC |
|------|----------------|---------|
| CompactAccel | 0x10000000 | 0x20000000 |
| BitNetAccel | 0x10001000 | 0x20001000 |
| Flash | 0x30000000 | 0x30000000 ✅ |
| PSRAM | 0x04000000 | 0x04000000 ✅ |

### 不可用的外设

ysyxSoC 平台**不包含**以下 SimpleEdgeAiSoC 外设：
- ❌ LCD (TFTLCD)
- ❌ SimpleEdgeAiSoC 的 UART
- ❌ SimpleEdgeAiSoC 的 GPIO

需要使用 ysyxSoC 的对应外设。

### UART 地址

ysyxSoC 的 UART 地址需要确认（可能是 0x10000000 或其他）。

---

## 📚 相关文件

### 测试程序

| 文件 | 说明 |
|------|------|
| `test_ai_accel.c` | 独立测试程序 |
| `compile_test.sh` | 编译脚本 |
| `linker.ld` | Linker script |

### 原始测试程序

| 文件 | 说明 |
|------|------|
| `chisel/software/examples/ai_demo.c` | AI 推理演示 |
| `chisel/software/examples/benchmark.c` | 性能测试 |
| `chisel/software/examples/flash_test.c` | Flash 测试 |
| `chisel/software/examples/psram_test.c` | PSRAM 测试 |

---

## ✅ 验证清单

- [ ] 编译测试程序
- [ ] 生成二进制文件
- [ ] 加载到仿真器
- [ ] CompactAccel 寄存器读写
- [ ] CompactAccel 计算测试
- [ ] BitNetAccel 寄存器读写
- [ ] BitNetAccel 计算测试
- [ ] 性能计数器验证
- [ ] 结果正确性验证
- [ ] 中断功能测试

---

## 🎯 下一步

1. **编译测试程序**
   ```bash
   cd /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage
   ./compile_test.sh
   ```

2. **运行测试**
   ```bash
   cp test_ai_accel.bin hello-minirv-ysyxsoc.bin
   ./obj_dir/VysyxSoCTop
   ```

3. **查看结果**
   - 检查 UART 输出
   - 验证测试通过
   - 记录性能数据

---

**创建日期**: 2025-12-03  
**状态**: ✅ 测试程序就绪  
**推荐**: 先使用方案 1（独立测试）
