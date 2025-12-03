# 所有 Verilog 文件（合并版）

**日期**: 2025-12-03  
**文件数**: 34 个  
**总大小**: 536 KB

---

## 📁 说明

此目录包含运行 ysyxSoC AI 加速器所需的**所有** Verilog 文件，已从 core/ 和 perip/ 合并到一个文件夹。

---

## 🚀 使用方法

### Verilator 编译

```bash
verilator --cc --exe --build \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-UNDRIVEN \
  -Wno-PINCONNECTEMPTY -Wno-PINMISSING -Wno-COMBDLY \
  -Wno-TIMESCALEMOD -Wno-MULTIDRIVEN -Wno-CASEINCOMPLETE \
  -Wno-BLKANDNBLK \
  --top-module ysyxSoCTop \
  *.v *.sv \
  sim_main.cpp
```

### Vivado 综合

```tcl
add_files [glob *.v]
add_files [glob *.sv]
set_property top ysyxSoCTop [current_fileset]
```

---

## 📊 文件列表

### 核心文件 (4 个)
- SimpleEdgeAiSoC.sv (149 KB) - AI 模块 + PicoRV32
- ysyx_26000001_with_ai.v (12 KB) - Wrapper
- ysyxSoCFull.v (85 KB) - ysyxSoC 顶层
- flash_fixed.v (2.7 KB) - Flash 模块

### 外设文件 (30 个)
- UART: uart_*.v (9 个)
- SPI: spi_*.v (5 个)
- SDRAM: sdram_*.v (6 个)
- 其他: gpio, ps2, psram, vga 等 (10 个)

---

## ✅ 优点

- ✅ 所有文件在一个目录
- ✅ 编译命令简单（*.v *.sv）
- ✅ 易于打包和分发
- ✅ 无需复杂的路径配置

---

**创建日期**: 2025-12-03  
**用途**: 简化编译和分发
