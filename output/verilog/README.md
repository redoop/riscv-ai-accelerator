# ysyxSoC AI 加速器 Verilog 文件包

**日期**: 2025-12-03  
**版本**: v0.4.1-ai  
**大小**: ~1 MB

---

## 📁 目录结构

```
verilog/
├── core/                           # 核心文件 (5 个)
│   ├── ysyx_26000001_with_ai.v    # AI 加速器 Wrapper (12 KB)
│   ├── SimpleEdgeAiSoC.sv         # AI 模块 (149 KB)
│   ├── picorv32.v                 # RISC-V CPU (93 KB)
│   ├── ysyxSoCFull.v              # ysyxSoC 顶层 (85 KB)
│   └── flash_fixed.v              # Flash 模块 (2.7 KB)
│
└── perip/                          # 外设文件 (38 个)
    ├── uart16550/rtl/             # UART (9 个)
    ├── spi/rtl/                   # SPI (5 个)
    ├── sdram/                     # SDRAM (6 个)
    ├── amba/                      # AMBA 总线 (2 个)
    ├── bitrev/                    # 位反转 (1 个)
    ├── gpio/                      # GPIO (1 个)
    ├── ps2/                       # PS/2 (1 个)
    ├── psram/                     # PSRAM (4 个)
    └── vga/                       # VGA (1 个)
```

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
  -Iperip/uart16550/rtl \
  -Iperip/spi/rtl \
  core/ysyxSoCFull.v \
  core/ysyx_26000001_with_ai.v \
  core/picorv32.v \
  core/SimpleEdgeAiSoC.sv \
  core/flash_fixed.v \
  perip/uart16550/rtl/*.v \
  perip/spi/rtl/*.v \
  perip/sdram/*.v \
  perip/sdram/core_sdram_axi4/*.v \
  perip/amba/*.v \
  perip/bitrev/*.v \
  perip/gpio/*.v \
  perip/ps2/*.v \
  perip/psram/*.v \
  perip/vga/*.v \
  sim_main.cpp
```

### Vivado 综合

```tcl
# 添加核心文件
add_files core/ysyx_26000001_with_ai.v
add_files core/SimpleEdgeAiSoC.sv
add_files core/picorv32.v
add_files core/ysyxSoCFull.v
add_files core/flash_fixed.v

# 添加外设文件
add_files [glob perip/uart16550/rtl/*.v]
add_files [glob perip/spi/rtl/*.v]
add_files [glob perip/sdram/*.v]
add_files [glob perip/sdram/core_sdram_axi4/*.v]
add_files [glob perip/amba/*.v]
add_files [glob perip/bitrev/*.v]
add_files [glob perip/gpio/*.v]
add_files [glob perip/ps2/*.v]
add_files [glob perip/psram/*.v]
add_files [glob perip/vga/*.v]

# 设置顶层
set_property top ysyxSoCTop [current_fileset]
```

---

## 📊 文件统计

| 类别 | 文件数 | 大小 |
|------|--------|------|
| 核心文件 | 5 | ~342 KB |
| UART | 9 | ~50 KB |
| SPI | 5 | ~10 KB |
| SDRAM | 6 | ~80 KB |
| 其他外设 | 13 | ~50 KB |
| **总计** | **38** | **~532 KB** |

---

## 🎯 核心模块说明

### ysyx_26000001_with_ai.v
- **功能**: AI 加速器集成 Wrapper
- **包含**:
  - PicoRV32 CPU 实例化
  - CompactAccel (0x20000000)
  - BitNetAccel (0x20001000)
  - Flash Controller (0x20002000)
  - PSRAM Controller (0x20003000)
  - SimpleBus 接口转换

### SimpleEdgeAiSoC.sv
- **功能**: 所有 AI 加速器模块
- **语言**: SystemVerilog
- **模块**:
  - ip1_SimpleCompactAccel - 8x8 矩阵加速器
  - ip1_SimpleBitNetAccel - 16x16 BitNet 加速器
  - ip1_SPIFlash - Flash 控制器
  - ip1_PSRAM - PSRAM 控制器

### picorv32.v
- **功能**: RISC-V RV32I CPU
- **特性**: 乘法、除法、中断
- **来源**: YosysHQ

---

## ⚠️ 注意事项

1. **Include 路径**: 需要添加 `-Iperip/uart16550/rtl` 和 `-Iperip/spi/rtl`
2. **SystemVerilog**: SimpleEdgeAiSoC.sv 需要 SV 支持
3. **警告抑制**: 建议使用提供的 `-Wno-*` 选项
4. **顶层模块**: ysyxSoCTop

---

## 📚 相关文档

- `YSYXSOC_AI_INTEGRATION.md` - 集成指南
- `YSYXSOC_VERILOG_FILES.md` - 详细文件说明
- `VERILOG_FILES_LIST.txt` - 文件清单

---

**创建日期**: 2025-12-03  
**状态**: ✅ 完整  
**用途**: ysyxSoC AI 加速器仿真和综合
