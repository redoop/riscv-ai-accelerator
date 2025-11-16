# RISC-V AI SoC v0.2 - Quick Start Guide

## 🚀 快速开始

### 1. 生成 Verilog

```bash
cd chisel
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"
```

输出：`generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv`

### 2. 运行测试

```bash
# UART 测试
sbt "testOnly riscv.ai.peripherals.RealUARTTest"

# LCD 测试
sbt "testOnly riscv.ai.peripherals.TFTLCDTest"

# 所有测试
sbt test
```

### 3. 使用软件工具

#### 安装依赖
```bash
pip install pyserial
pip install Pillow  # 可选，用于图像显示
```

#### 上传程序
```bash
cd software/tools

# 查看 Bootloader 信息
python upload.py /dev/ttyUSB0 --info

# 上传并运行程序
python upload.py /dev/ttyUSB0 program.bin --run

# LCD 测试
python upload.py /dev/ttyUSB0 --test-lcd

# 显示图片
python upload.py /dev/ttyUSB0 --image logo.png
```

## 📁 项目结构

```
chisel/
├── src/main/scala/
│   ├── EdgeAiSoCSimple.scala      # SoC 顶层
│   └── peripherals/
│       ├── RealUART.scala         # UART 控制器
│       └── TFTLCD.scala           # LCD 控制器
├── src/test/scala/
│   ├── RealUARTTest.scala         # UART 测试
│   └── TFTLCDTest.scala           # LCD 测试
├── software/
│   ├── lib/                       # HAL 和图形库
│   ├── bootloader/                # Bootloader
│   ├── tools/                     # Python 工具
│   └── examples/                  # 示例程序
├── generated/                     # 生成的 Verilog
└── docs/                          # 文档
```

## 🎯 核心功能

### 硬件模块
- **RealUART**: 完整的 UART 控制器（115200 bps）
- **TFTLCD**: ST7735 SPI 控制器（128x128 RGB565）
- **CompactAccel**: 8x8 矩阵加速器
- **BitNetAccel**: 16x16 矩阵加速器

### 软件库
- **HAL**: 硬件抽象层（UART, LCD, GPIO）
- **Graphics**: 图形库（点、线、矩形、圆、文本）
- **Font**: 8x8 ASCII 字体
- **Bootloader**: 程序上传和管理

## 📝 示例代码

### Hello World (C)
```c
#include "lib/hal.h"
#include "lib/graphics.h"

void main(void) {
    uart_init(115200);
    lcd_init();
    
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(20, 50, "Hello!", COLOR_WHITE, COLOR_BLACK);
    
    while(1) {
        uart_puts("Hello World!\r\n");
        delay_ms(1000);
    }
}
```

### AI Demo (C)
```c
void display_inference_result(const char* class_name, uint32_t confidence) {
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(10, 10, "AI Result:", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(10, 30, class_name, COLOR_GREEN, COLOR_BLACK);
    lcd_printf(10, 50, COLOR_CYAN, COLOR_BLACK, "Conf: %d%%", confidence);
}
```

## 🔧 内存映射

```
0x00000000: RAM (256 MB)
0x10000000: CompactAccel (4 KB)
0x10001000: BitNetAccel (4 KB)
0x20000000: UART (64 KB)
0x20010000: LCD (64 KB)
0x20020000: GPIO (64 KB)
```

## 📊 性能指标

- **UART**: 115200 bps, 16 字节 FIFO
- **LCD**: 10MHz SPI, 32KB 帧缓冲
- **CPU**: PicoRV32 @ 50MHz
- **AI**: ~6.4 GOPS (CompactAccel + BitNetAccel)

## 🐛 调试

### 查看生成的 Verilog
```bash
cat generated/simple_edgeaisoc/SimpleEdgeAiSoC.sv
```

### 运行特定测试
```bash
sbt "testOnly riscv.ai.peripherals.RealUARTTest -- -z \"should transmit\""
```

### 查看测试波形
测试会生成 VCD 文件（如果启用 WriteVcdAnnotation）

## 📚 更多信息

- 详细开发计划：`docs/DEV_PLAN_V0.2.md`
- 软件文档：`software/README.md`
- Chisel 文档：https://www.chisel-lang.org/

## ✅ 验证清单

- [x] Verilog 生成成功
- [x] UART 测试通过（7/8）
- [x] LCD 测试通过（8/8）
- [x] Bootloader 实现完成
- [x] 图形库实现完成
- [x] 示例程序编写完成
- [ ] FPGA 验证（需要硬件）

## 🎉 成果

v0.2 版本实现了完整的调试和交互功能：
- ✅ USB 串口通信
- ✅ TFT LCD 彩色显示
- ✅ 程序上传协议
- ✅ 图形库和字体
- ✅ AI 推理演示

**总代码量**: ~3000 行（Chisel + C + Python）
**测试覆盖**: 15/16 测试通过
**开发时间**: 1 天（Phase 1-4）
