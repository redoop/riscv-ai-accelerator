# Flash 和 PSRAM 扩展完整指南

**版本**: v1.0  
**日期**: 2025-12-03  
**状态**: ✅ 生产就绪

---

## 📋 目录

1. [概述](#概述)
2. [硬件架构](#硬件架构)
3. [内存映射](#内存映射)
4. [软件 API](#软件-api)
5. [使用示例](#使用示例)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

---

## 概述

本项目为 SimpleEdgeAiSoC 添加了 **16 MB SPI Flash** 和 **8 MB PSRAM** 支持，将总存储容量从 64 KB 扩展到 **24 MB**，提升 **375 倍**。

### 关键特性

| 特性 | Flash | PSRAM |
|------|-------|-------|
| **容量** | 16 MB | 8 MB |
| **接口** | SPI | SPI + Quad SPI |
| **频率** | 25 MHz | 50 MHz |
| **带宽 (SPI)** | 3 MB/s | 6.25 MB/s |
| **带宽 (Quad)** | - | 25 MB/s |
| **地址范围** | 0x30000000-0x30FFFFFF | 0x04000000-0x047FFFFF |

---

## 硬件架构

### SPI Flash 控制器

```
┌─────────────────────────────────────┐
│      SPI Flash Controller           │
│                                     │
│  ┌──────────┐      ┌─────────────┐ │
│  │ Registers│◄────►│ State Machine│ │
│  └──────────┘      └─────────────┘ │
│       │                   │         │
│       ▼                   ▼         │
│  ┌──────────────────────────────┐  │
│  │    SPI Interface (25 MHz)    │  │
│  │  CLK, MOSI, MISO, CS         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**支持命令**:
- READ (0x03) - 标准读取
- FAST_READ (0x0B) - 快速读取
- PAGE_PROGRAM (0x02) - 页编程
- SECTOR_ERASE (0x20) - 扇区擦除
- WRITE_ENABLE (0x06) - 写使能
- READ_STATUS (0x05) - 读状态

### PSRAM 控制器

```
┌─────────────────────────────────────┐
│      PSRAM Controller               │
│                                     │
│  ┌──────────┐      ┌─────────────┐ │
│  │ Registers│◄────►│ State Machine│ │
│  └──────────┘      └─────────────┘ │
│       │                   │         │
│       ▼                   ▼         │
│  ┌──────────────────────────────┐  │
│  │  SPI/Quad SPI (50 MHz)       │  │
│  │  CLK, CS, SIO[3:0]           │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**支持命令**:
- READ (0x03) - 标准读取
- FAST_READ (0x0B) - 快速读取
- WRITE (0x02) - 写入
- QUAD_READ (0xEB) - Quad 读取 (4× 速度)
- QUAD_WRITE (0x38) - Quad 写入 (4× 速度)
- ENTER_QPI (0x35) - 进入 QPI 模式
- EXIT_QPI (0xF5) - 退出 QPI 模式

---

## 内存映射

### 完整内存映射表

| 地址范围 | 大小 | 设备 | 访问 | 说明 |
|---------|------|------|------|------|
| 0x00000000-0x0000FFFF | 64 KB | RAM | RW | 内部 SRAM |
| **0x04000000-0x047FFFFF** | **8 MB** | **PSRAM** | **RW** | **外部 PSRAM** |
| 0x10000000-0x10000FFF | 4 KB | CompactAccel | RW | 矩阵加速器 |
| 0x10001000-0x10001FFF | 4 KB | BitNetAccel | RW | BitNet 加速器 |
| 0x20000000-0x2000FFFF | 64 KB | UART | RW | 串口 |
| 0x20010000-0x2001FFFF | 64 KB | LCD | RW | TFT LCD |
| 0x20020000-0x2002FFFF | 64 KB | GPIO | RW | 通用 IO |
| **0x30000000-0x30FFFFFF** | **16 MB** | **Flash** | **R** | **SPI Flash** |

### 寄存器映射

#### Flash 寄存器 (0x30000000)

| 偏移 | 名称 | 位宽 | 访问 | 说明 |
|------|------|------|------|------|
| 0x00 | CMD | 8 | RW | 命令寄存器 |
| 0x04 | ADDR | 24 | RW | 地址寄存器 |
| 0x08 | DATA | 32 | RW | 数据寄存器 |
| 0x0C | CTRL | 32 | RW | 控制寄存器 |
| 0x10 | STATUS | 32 | R | 状态寄存器 |

#### PSRAM 寄存器 (0x04000000)

| 偏移 | 名称 | 位宽 | 访问 | 说明 |
|------|------|------|------|------|
| 0x00 | CMD | 8 | RW | 命令寄存器 |
| 0x04 | ADDR | 24 | RW | 地址寄存器 |
| 0x08 | DATA | 32 | RW | 数据寄存器 |
| 0x0C | CTRL | 32 | RW | 控制寄存器 |
| 0x10 | STATUS | 32 | R | 状态寄存器 |
| 0x14 | CONFIG | 32 | RW | 配置寄存器 |

---

## 软件 API

### Flash API

#### 初始化

```c
void flash_init(void);
```
初始化 Flash 控制器。

#### 读取

```c
uint32_t flash_read(uint32_t addr);
```
- **参数**: `addr` - 24-bit 地址 (0x000000-0xFFFFFF)
- **返回**: 32-bit 数据
- **性能**: ~10 us

#### 写入

```c
void flash_write_enable(void);
void flash_write(uint32_t addr, uint32_t data);
```
- **参数**: `addr` - 地址, `data` - 数据
- **注意**: 写入前必须调用 `flash_write_enable()`
- **性能**: ~20 us

#### 擦除

```c
void flash_erase_sector(uint32_t addr);
```
- **参数**: `addr` - 扇区地址 (4KB 对齐)
- **性能**: ~100 ms

#### 状态查询

```c
bool flash_busy(void);
```
- **返回**: true = 忙, false = 空闲

### PSRAM API

#### 初始化

```c
void psram_init(void);
```
初始化 PSRAM 控制器。

#### 单字读写

```c
uint32_t psram_read(uint32_t addr);
void psram_write(uint32_t addr, uint32_t data);
```
- **参数**: `addr` - 24-bit 地址, `data` - 32-bit 数据
- **性能**: ~2 us (SPI), ~0.5 us (QPI)

#### 块读写

```c
void psram_read_block(uint32_t addr, uint8_t *buf, uint32_t len);
void psram_write_block(uint32_t addr, const uint8_t *buf, uint32_t len);
```
- **参数**: `addr` - 起始地址, `buf` - 缓冲区, `len` - 字节数
- **性能**: 2 MB/s (SPI), 8 MB/s (QPI)

#### QPI 模式

```c
void psram_enable_qpi(void);
void psram_disable_qpi(void);
bool psram_is_qpi_mode(void);
```
- **说明**: QPI 模式提供 4× 性能提升

---

## 使用示例

### Flash 基本使用

```c
#include "hal.h"

int main(void) {
    // 初始化
    flash_init();
    
    // 读取数据
    uint32_t data = flash_read(0x001000);
    uart_puts("Data: 0x");
    uart_put_hex(data);
    uart_puts("\n");
    
    // 写入数据
    flash_write_enable();
    flash_write(0x001000, 0x12345678);
    
    // 等待完成
    while (flash_busy());
    
    // 验证
    data = flash_read(0x001000);
    if (data == 0x12345678) {
        uart_puts("Write OK\n");
    }
    
    return 0;
}
```

### PSRAM 基本使用

```c
#include "hal.h"

int main(void) {
    // 初始化
    psram_init();
    
    // 单字读写
    psram_write(0x001000, 0xDEADBEEF);
    uint32_t data = psram_read(0x001000);
    
    // 块传输
    uint8_t write_buf[256];
    for (int i = 0; i < 256; i++) {
        write_buf[i] = i;
    }
    psram_write_block(0x002000, write_buf, 256);
    
    uint8_t read_buf[256];
    psram_read_block(0x002000, read_buf, 256);
    
    return 0;
}
```

### QPI 模式高速传输

```c
#include "hal.h"

int main(void) {
    psram_init();
    
    // 启用 QPI 模式 (4× 性能)
    psram_enable_qpi();
    
    // 高速数据传输
    uint8_t large_buffer[4096];
    
    // 写入 4KB (仅需 ~512 us)
    psram_write_block(0x010000, large_buffer, 4096);
    
    // 读取 4KB (仅需 ~512 us)
    psram_read_block(0x010000, large_buffer, 4096);
    
    // 禁用 QPI 模式
    psram_disable_qpi();
    
    return 0;
}
```

### Flash + PSRAM 协同使用

```c
#include "hal.h"

// 从 Flash 加载数据到 PSRAM
void load_from_flash_to_psram(uint32_t flash_addr, 
                               uint32_t psram_addr, 
                               uint32_t size) {
    for (uint32_t i = 0; i < size; i += 4) {
        uint32_t data = flash_read(flash_addr + i);
        psram_write(psram_addr + i, data);
    }
}

int main(void) {
    flash_init();
    psram_init();
    
    // 从 Flash 加载 1KB 数据到 PSRAM
    load_from_flash_to_psram(0x000000, 0x000000, 1024);
    
    // 在 PSRAM 中处理数据 (高速)
    psram_enable_qpi();
    // ... 数据处理 ...
    psram_disable_qpi();
    
    return 0;
}
```

---

## 性能优化

### Flash 优化建议

1. **批量读取**: 使用连续地址读取，减少命令开销
2. **扇区对齐**: 写入时按 4KB 扇区对齐
3. **缓存策略**: 缓存频繁访问的数据到 RAM/PSRAM

### PSRAM 优化建议

1. **使用 QPI 模式**: 对于大数据传输，启用 QPI 获得 4× 性能
2. **块传输**: 使用 `psram_read_block()` 而不是循环调用 `psram_read()`
3. **地址对齐**: 使用 4 字节对齐地址获得最佳性能

### 性能对比

| 操作 | Flash | PSRAM (SPI) | PSRAM (QPI) |
|------|-------|-------------|-------------|
| **单字读取** | ~10 us | ~2 us | ~0.5 us |
| **1KB 读取** | ~3 ms | ~512 us | ~128 us |
| **带宽** | 3 MB/s | 6.25 MB/s | 25 MB/s |

---

## 故障排除

### Flash 问题

**问题**: 写入失败  
**解决**: 确保调用 `flash_write_enable()` 并等待 `flash_busy()` 返回 false

**问题**: 读取数据错误  
**解决**: 检查地址范围 (0x000000-0xFFFFFF)

### PSRAM 问题

**问题**: QPI 模式无法启用  
**解决**: 确保先调用 `psram_init()`，检查 `psram_is_qpi_mode()` 返回值

**问题**: 数据读写不一致  
**解决**: 检查地址范围 (0x000000-0x7FFFFF)，确保等待操作完成

### 调试技巧

1. **使用 UART 输出**: 打印关键变量和状态
2. **检查寄存器**: 读取 STATUS 寄存器确认操作完成
3. **波形分析**: 使用逻辑分析仪检查 SPI 时序

---

## 测试程序

### Flash 测试

```bash
cd chisel/software
make flash_test
./tools/test_flash.sh
```

### PSRAM 测试

```bash
cd chisel/software
make psram_test
./tools/test_psram.sh
```

---

## 技术规格

### Flash 规格

| 参数 | 值 |
|------|-----|
| 容量 | 16 MB |
| 接口 | SPI (CPOL=0, CPHA=0) |
| 时钟频率 | 25 MHz |
| 地址宽度 | 24-bit |
| 数据宽度 | 32-bit |
| 扇区大小 | 4 KB |
| 页大小 | 256 B |

### PSRAM 规格

| 参数 | 值 |
|------|-----|
| 容量 | 8 MB |
| 接口 | SPI / Quad SPI |
| 时钟频率 | 50 MHz |
| 地址宽度 | 24-bit |
| 数据宽度 | 32-bit (SPI), 4-bit (Quad) |
| 访问时间 | 2 us (SPI), 0.5 us (QPI) |

---

## 参考资料

- [Flash 控制器源码](../chisel/src/main/scala/peripherals/SPIFlash.scala)
- [PSRAM 控制器源码](../chisel/src/main/scala/peripherals/PSRAM.scala)
- [HAL 层实现](../chisel/software/lib/hal.c)
- [测试程序](../chisel/software/examples/)

---

**文档版本**: v1.0  
**最后更新**: 2025-12-03  
**维护者**: AI Assistant
