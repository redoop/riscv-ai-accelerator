# ysyx_26000001_with_ai.v 相关 Verilog 文件清单

**日期**: 2025-12-03  
**用途**: 运行 AI 加速器集成的 ysyxSoC

---

## 📁 核心文件（必需）

### 1. AI 加速器 Wrapper
```
/opt/github/riscv-ai-accelerator/chisel/generated/simple_edgeaisoc/ysyx_26000001_with_ai.v
```
- **大小**: ~13 KB
- **说明**: 包含 PicoRV32 + AI 加速器的完整 wrapper
- **包含**:
  - PicoRV32 CPU 实例化
  - CompactAccel 接口
  - BitNetAccel 接口
  - Flash Controller 接口
  - PSRAM Controller 接口
  - 地址解码器
  - SimpleBus 接口转换

### 2. AI 加速器模块（SystemVerilog）
```
/opt/github/riscv-ai-accelerator/chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```
- **大小**: ~150 KB
- **说明**: Chisel 生成的所有 AI 模块
- **包含模块**:
  - `ip1_SimpleCompactAccel` - 8x8 矩阵加速器
  - `ip1_SimpleBitNetAccel` - 16x16 BitNet 加速器
  - `ip1_SPIFlash` - Flash 控制器
  - `ip1_PSRAM` - PSRAM 控制器
  - `ip1_matrix_64x32` - 矩阵存储
  - `ip1_activation_256x32` - 激活值存储
  - `ip1_weight_256x2` - 权重存储
  - `ip1_result_256x32` - 结果存储
  - 其他辅助模块

### 3. RISC-V CPU 核心
```
/opt/github/riscv-ai-accelerator/chisel/src/main/resources/rtl/picorv32.v
```
- **大小**: ~120 KB
- **说明**: PicoRV32 RISC-V RV32I 核心
- **包含模块**:
  - `picorv32` - 主 CPU 模块
  - `picorv32_regs` - 寄存器文件
  - `picorv32_pcpi_mul` - 乘法器
  - `picorv32_pcpi_div` - 除法器
  - `picorv32_axi` - AXI 接口（未使用）
  - `picorv32_wb` - Wishbone 接口（未使用）

---

## 📁 ysyxSoC 平台文件

### 4. ysyxSoC 顶层
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v
```
- **大小**: ~100 KB
- **说明**: ysyxSoC 完整系统顶层
- **包含**:
  - CPU 接口（使用 ysyx_26000001）
  - SimpleBus 总线
  - 外设桥接
  - 地址映射

### 5. Flash 模块（修复版）
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/ready-to-run/D-stage/flash_fixed.v
```
- **大小**: ~5 KB
- **说明**: 修复了 Verilator 兼容性的 Flash 模块

---

## 📁 ysyxSoC 外设文件

### 6. UART (16550)
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/uart16550/rtl/
├── uart_top.v
├── uart_receiver.v
├── uart_regs.v
├── uart_rfifo.v
├── uart_sync_flops.v
├── uart_tfifo.v
├── uart_transmitter.v
├── uart_wb.v
├── uart_defines.v
└── raminfr.v
```

### 7. SPI
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/spi/rtl/
├── spi_top.v
├── spi_clgen.v
└── spi_shift.v
```

### 8. SDRAM
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/sdram/
├── sdram.v
├── sdram_top_apb.v
├── sdram_top_axi.v
└── core_sdram_axi4/
    ├── sdram_axi.v
    ├── sdram_axi_core.v
    └── sdram_axi_pmem.v
```

### 9. 其他外设
```
/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/
├── amba/*.v          # AMBA 总线
├── bitrev/*.v        # 位反转
├── gpio/*.v          # GPIO
├── ps2/*.v           # PS/2 接口
├── psram/*.v         # PSRAM（ysyxSoC 的）
└── vga/*.v           # VGA 控制器
```

---

## 📊 文件统计

| 类别 | 文件数 | 总大小 | 说明 |
|------|--------|--------|------|
| **AI 加速器** | 2 | ~163 KB | Wrapper + 模块 |
| **CPU 核心** | 1 | ~120 KB | PicoRV32 |
| **ysyxSoC 核心** | 2 | ~105 KB | 顶层 + Flash |
| **UART** | 10 | ~50 KB | 串口控制器 |
| **SPI** | 3 | ~10 KB | SPI 控制器 |
| **SDRAM** | 6 | ~80 KB | SDRAM 控制器 |
| **其他外设** | ~30 | ~100 KB | GPIO, VGA 等 |
| **总计** | ~54 | ~628 KB | 所有 Verilog |

---

## 🔧 编译命令

完整的 Verilator 编译命令：

```bash
verilator --cc --exe --build \
  -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-UNDRIVEN \
  -Wno-PINCONNECTEMPTY -Wno-PINMISSING -Wno-COMBDLY \
  -Wno-TIMESCALEMOD -Wno-MULTIDRIVEN -Wno-CASEINCOMPLETE \
  -Wno-BLKANDNBLK \
  --top-module ysyxSoCTop \
  -I/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/uart16550/rtl \
  -I/opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/spi/rtl \
  ysyxSoCFull.v \
  ysyx_26000001.v \
  picorv32.v \
  SimpleEdgeAiSoC.sv \
  flash_fixed.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/uart16550/rtl/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/spi/rtl/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/sdram/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/sdram/core_sdram_axi4/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/amba/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/bitrev/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/gpio/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/ps2/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/psram/*.v \
  /opt/github/riscv-ai-accelerator/ecos/ysyxSoC/perip/vga/*.v \
  sim_main.cpp
```

---

## 📦 打包清单

如果需要打包所有文件：

```bash
cd /opt/github/riscv-ai-accelerator

# 创建打包目录
mkdir -p ysyxsoc_ai_package/verilog

# 复制核心文件
cp chisel/generated/simple_edgeaisoc/ysyx_26000001_with_ai.v \
   ysyxsoc_ai_package/verilog/
cp chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv \
   ysyxsoc_ai_package/verilog/
cp chisel/src/main/resources/rtl/picorv32.v \
   ysyxsoc_ai_package/verilog/

# 复制 ysyxSoC 文件
cp ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v \
   ysyxsoc_ai_package/verilog/
cp ecos/ysyxSoC/ready-to-run/D-stage/flash_fixed.v \
   ysyxsoc_ai_package/verilog/

# 复制外设
cp -r ecos/ysyxSoC/perip ysyxsoc_ai_package/verilog/

# 打包
tar czf ysyxsoc_ai_verilog.tar.gz ysyxsoc_ai_package/
```

---

## 🔍 模块依赖关系

```
ysyxSoCTop (顶层)
├── ysyx_26000001 (Wrapper)
│   ├── picorv32 (CPU)
│   ├── ip1_SimpleCompactAccel (AI)
│   ├── ip1_SimpleBitNetAccel (AI)
│   ├── ip1_SPIFlash (存储)
│   └── ip1_PSRAM (存储)
├── SimpleBus (总线)
├── uart_top (UART)
├── spi_top (SPI)
├── sdram_top_apb (SDRAM)
└── 其他外设
```

---

## 📝 关键文件说明

### ysyx_26000001_with_ai.v
- **作用**: 将 SimpleEdgeAiSoC 集成到 ysyxSoC
- **接口**: SimpleBus (IFU/LSU)
- **地址映射**:
  - 0x20000000: CompactAccel
  - 0x20001000: BitNetAccel
  - 0x20002000: Flash Controller
  - 0x20003000: PSRAM Controller

### SimpleEdgeAiSoC.sv
- **作用**: 所有 AI 加速器模块
- **语言**: SystemVerilog
- **模块数**: ~18 个
- **特点**: Chisel 自动生成

### picorv32.v
- **作用**: RISC-V CPU 核心
- **架构**: RV32I
- **特性**: 乘法、除法、中断
- **来源**: YosysHQ

---

## ✅ 验证清单

使用此清单验证所有文件是否存在：

```bash
# 核心文件
[ -f chisel/generated/simple_edgeaisoc/ysyx_26000001_with_ai.v ] && echo "✓ Wrapper"
[ -f chisel/generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv ] && echo "✓ AI Modules"
[ -f chisel/src/main/resources/rtl/picorv32.v ] && echo "✓ CPU"

# ysyxSoC 文件
[ -f ecos/ysyxSoC/ready-to-run/D-stage/ysyxSoCFull.v ] && echo "✓ SoC Top"
[ -f ecos/ysyxSoC/ready-to-run/D-stage/flash_fixed.v ] && echo "✓ Flash"

# 外设
[ -d ecos/ysyxSoC/perip/uart16550 ] && echo "✓ UART"
[ -d ecos/ysyxSoC/perip/spi ] && echo "✓ SPI"
[ -d ecos/ysyxSoC/perip/sdram ] && echo "✓ SDRAM"
```

---

**创建日期**: 2025-12-03  
**文件总数**: ~54 个  
**总大小**: ~628 KB  
**状态**: ✅ 完整
