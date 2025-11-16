# v0.2 开发计划 - 调试功能完善

## 📋 TODO Checklist（快速跟踪）

### ✅ Phase 1: UART 控制器（3 天）
- [x] 创建 RealUART.scala 并实现核心功能
- [x] 编写单元测试并验证
- [x] 集成到 SoC

### 🔴 Phase 2: TFT LCD 控制器（5-7 天）
- [ ] 创建 TFTLCD.scala 并实现 SPI 控制器
- [ ] 实现 ST7735 驱动和帧缓冲
- [ ] 编写单元测试并验证
- [ ] 集成到 SoC

### 🟡 Phase 3: 程序上传协议（2-3 天）
- [ ] 编写 Bootloader C 代码
- [ ] 编写 Python 上传工具
- [ ] 测试端到端上传流程

### 🟡 Phase 4: 图形库（2-3 天）
- [ ] 实现基本图形函数和字体
- [ ] 编写示例程序
- [ ] 测试所有功能

### 🟢 Phase 5: 集成测试（2-3 天）
- [ ] 综合测试和性能测试
- [ ] 编写文档和演示程序

### 🟢 Phase 6: FPGA 验证（2-3 天）
- [ ] 生成 Verilog 并综合
- [ ] 硬件测试和问题修复

**进度：** 1/6 阶段完成 | **预计：** 16-22 天

---

## 目标
为流片后的芯片添加完整的调试和交互功能

## 核心需求
1. ✅ USB 程序上传
2. ✅ USB 串口协议输入输出
3. ✅ TFT LCD 128x128 彩色显示

---

## 技术方案

### 架构图
```
┌─────────────────────────────────────────────────────────────┐
│  芯片内部                                                   │
│                                                             │
│  ┌──────────────┐                                          │
│  │  PicoRV32    │                                          │
│  │  RISC-V Core │                                          │
│  └──────┬───────┘                                          │
│         │                                                   │
│    ┌────┴────┐                                             │
│    │ Memory  │                                             │
│    │  Bus    │                                             │
│    └─┬───┬───┬──┐                                          │
│      │   │   │  │                                          │
│  ┌───▼───▼───▼──▼────┐                                     │
│  │  Address Decoder  │                                     │
│  └─┬───┬───┬──┬──────┘                                     │
│    │   │   │  │                                            │
│  ┌─▼─┐ │ ┌─▼──▼────┐  ┌──────────┐                        │
│  │UART│ │ │TFT LCD  │  │  GPIO    │                        │
│  │Ctrl│ │ │SPI Ctrl │  │          │                        │
│  └─┬──┘ │ └────┬────┘  └────┬─────┘                        │
│    │    │      │            │                              │
└────┼────┼──────┼────────────┼──────────────────────────────┘
     │    │      │            │
┌────▼────▼──┐   │            │
│  FTDI      │   │            │
│  FT232H    │   │            │
│  (外部)    │   │            │
└────┬───────┘   │            │
     │           │            │
  USB 接口    ┌──▼────────┐   │
             │ ST7735     │   │
             │ TFT LCD    │   │
             │ 128x128    │   │
             │ 65K 色     │   │
             └────────────┘   │
                          调试 LED
```

---

## 模块设计

### 1. 完整的 UART 控制器

**文件：** `chisel/src/main/scala/peripherals/RealUART.scala`

```scala
class RealUART(
  clockFreq: Int = 50000000,  // 50MHz 时钟
  baudRate: Int = 115200,      // 115200 波特率
  fifoDepth: Int = 16          // FIFO 深度
) extends Module
```

**寄存器映射：**
```
0x00: DATA      - 数据寄存器 (R/W)
0x04: STATUS    - 状态寄存器 (R)
      bit 0: TX_BUSY
      bit 1: RX_READY
      bit 2: TX_FIFO_FULL
      bit 3: RX_FIFO_EMPTY
0x08: CONTROL   - 控制寄存器 (R/W)
      bit 0: TX_ENABLE
      bit 1: RX_ENABLE
      bit 2: TX_IRQ_ENABLE
      bit 3: RX_IRQ_ENABLE
0x0C: BAUD_DIV  - 波特率分频 (R/W)
```

**功能：**
- ✅ 可配置波特率（9600 - 921600）
- ✅ 发送 FIFO（16 字节）
- ✅ 接收 FIFO（16 字节）
- ✅ 中断支持
- ✅ 状态标志

**工作量：** 3 天

---

### 2. TFT LCD SPI 控制器（ST7735）

**文件：** `chisel/src/main/scala/peripherals/TFTLCD.scala`

```scala
class TFTLCD extends Module {
  val io = IO(new Bundle {
    val reg = new SimpleRegIO()
    
    // SPI 接口
    val spi_clk = Output(Bool())   // SPI 时钟
    val spi_mosi = Output(Bool())  // SPI 数据输出
    val spi_cs = Output(Bool())    // 片选
    val spi_dc = Output(Bool())    // 数据/命令选择
    val spi_rst = Output(Bool())   // 复位
    
    // 背光控制
    val backlight = Output(Bool())
  })
}
```

**寄存器映射：**
```
0x00: COMMAND   - 命令寄存器 (W)
0x04: DATA      - 数据寄存器 (W)
0x08: STATUS    - 状态寄存器 (R)
      bit 0: BUSY
      bit 1: INIT_DONE
0x0C: CONTROL   - 控制寄存器 (R/W)
      bit 0: BACKLIGHT
      bit 1: RESET
0x10: X_START   - X 起始坐标 (R/W)
0x14: Y_START   - Y 起始坐标 (R/W)
0x18: X_END     - X 结束坐标 (R/W)
0x1C: Y_END     - Y 结束坐标 (R/W)
0x20: COLOR     - 颜色数据 (W, RGB565)
0x1000-0x8FFF: FRAMEBUFFER - 帧缓冲 (32KB, 128x128x2)
```

**ST7735 特性：**
- 分辨率：128x128 像素
- 颜色：65K 色（RGB565）
- 接口：SPI（最高 15MHz）
- 帧缓冲：32KB（128 x 128 x 2 字节）

**功能：**
- ✅ 初始化 LCD
- ✅ 设置显示区域
- ✅ 写入像素数据
- ✅ 快速填充
- ✅ 帧缓冲模式
- ✅ 背光控制

**支持的操作：**
```c
// 初始化
lcd_init();

// 清屏
lcd_clear(COLOR_BLACK);

// 画点
lcd_draw_pixel(x, y, COLOR_RED);

// 画线
lcd_draw_line(x0, y0, x1, y1, COLOR_BLUE);

// 画矩形
lcd_draw_rect(x, y, w, h, COLOR_GREEN);

// 填充矩形
lcd_fill_rect(x, y, w, h, COLOR_YELLOW);

// 显示字符
lcd_draw_char(x, y, 'A', COLOR_WHITE, COLOR_BLACK);

// 显示字符串
lcd_draw_string(x, y, "Hello", COLOR_WHITE, COLOR_BLACK);

// 显示图片（从帧缓冲）
lcd_draw_image(x, y, w, h, image_data);
```

**工作量：** 5-7 天

---

### 3. 程序上传协议

**文件：** `chisel/software/bootloader/bootloader.c`

```c
// Bootloader 协议
#define CMD_UPLOAD    'U'  // 上传程序
#define CMD_RUN       'R'  // 运行程序
#define CMD_READ_MEM  'M'  // 读取内存
#define CMD_WRITE_REG 'W'  // 写寄存器
#define CMD_LCD_TEST  'L'  // LCD 测试

void bootloader_main() {
    uart_init(115200);
    lcd_init();
    
    // 显示启动画面
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(10, 10, "RISC-V AI", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(10, 30, "Bootloader", COLOR_CYAN, COLOR_BLACK);
    lcd_draw_string(10, 50, "v0.2", COLOR_GREEN, COLOR_BLACK);
    lcd_draw_string(10, 70, "Ready...", COLOR_YELLOW, COLOR_BLACK);
    
    while(1) {
        uint8_t cmd = uart_getc();
        handle_command(cmd);
    }
}
```

**PC 端工具：** `chisel/software/tools/upload.py`

```python
import serial
import struct
from PIL import Image

class RISCVUploader:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
    
    def upload_program(self, binary_file):
        """上传程序到 RAM"""
        with open(binary_file, 'rb') as f:
            data = f.read()
        
        self.ser.write(b'U')
        self.ser.write(struct.pack('<I', len(data)))
        
        for i in range(0, len(data), 256):
            chunk = data[i:i+256]
            self.ser.write(chunk)
            progress = (i / len(data)) * 100
            print(f"Progress: {progress:.1f}%")
        
        ack = self.ser.read(1)
        return ack == b'K'
    
    def run_program(self):
        """运行程序"""
        self.ser.write(b'R')
    
    def lcd_display_image(self, image_file):
        """在 LCD 上显示图片"""
        img = Image.open(image_file)
        img = img.resize((128, 128))
        img = img.convert('RGB')
        
        self.ser.write(b'L')
        
        for y in range(128):
            for x in range(128):
                r, g, b = img.getpixel((x, y))
                # 转换为 RGB565
                rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                self.ser.write(struct.pack('<H', rgb565))
        
        print("Image uploaded to LCD")

# 使用示例
uploader = RISCVUploader('/dev/ttyUSB0')
uploader.upload_program('program.bin')
uploader.run_program()
uploader.lcd_display_image('logo.png')
```

**工作量：** 2-3 天

---

### 4. 图形库（软件）

**文件：** `chisel/software/lib/graphics.c`

```c
// 颜色定义（RGB565）
#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF
#define COLOR_RED     0xF800
#define COLOR_GREEN   0x07E0
#define COLOR_BLUE    0x001F
#define COLOR_YELLOW  0xFFE0
#define COLOR_CYAN    0x07FF
#define COLOR_MAGENTA 0xF81F

// 8x8 ASCII 字体
extern const uint8_t font_8x8[128][8];

// 基本图形函数
void lcd_draw_pixel(uint8_t x, uint8_t y, uint16_t color);
void lcd_draw_line(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, uint16_t color);
void lcd_draw_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color);
void lcd_fill_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color);
void lcd_draw_circle(uint8_t x, uint8_t y, uint8_t r, uint16_t color);

// 文本函数
void lcd_draw_char(uint8_t x, uint8_t y, char c, uint16_t fg, uint16_t bg);
void lcd_draw_string(uint8_t x, uint8_t y, const char* str, uint16_t fg, uint16_t bg);
void lcd_printf(uint8_t x, uint8_t y, uint16_t fg, uint16_t bg, const char* fmt, ...);

// 图像函数
void lcd_draw_image(uint8_t x, uint8_t y, uint8_t w, uint8_t h, const uint16_t* data);
```

**工作量：** 2-3 天

---

## 更新后的顶层模块

**文件：** `chisel/src/main/scala/EdgeAiSoCSimple.scala`

```scala
class SimpleEdgeAiSoC extends Module {
  val io = IO(new Bundle {
    // === USB 串口（通过 FTDI 芯片）===
    val uart_tx = Output(Bool())
    val uart_rx = Input(Bool())
    
    // === TFT LCD SPI 接口 ===
    val lcd_spi_clk = Output(Bool())
    val lcd_spi_mosi = Output(Bool())
    val lcd_spi_cs = Output(Bool())
    val lcd_spi_dc = Output(Bool())
    val lcd_spi_rst = Output(Bool())
    val lcd_backlight = Output(Bool())
    
    // === GPIO（调试用）===
    val gpio_out = Output(UInt(32.W))
    val gpio_in = Input(UInt(32.W))
    
    // === 调试信号 ===
    val trap = Output(Bool())
    val compact_irq = Output(Bool())
    val bitnet_irq = Output(Bool())
  })
}
```

---

## 内存映射（更新）

```
0x00000000 - 0x0FFFFFFF: RAM (256 MB)
0x10000000 - 0x10000FFF: CompactAccel (4 KB)
0x10001000 - 0x10001FFF: BitNetAccel (4 KB)
0x20000000 - 0x2000FFFF: UART (64 KB)
  0x20000000: DATA
  0x20000004: STATUS
  0x20000008: CONTROL
  0x2000000C: BAUD_DIV
0x20010000 - 0x2001FFFF: TFT LCD (64 KB)
  0x20010000: COMMAND
  0x20010004: DATA
  0x20010008: STATUS
  0x2001000C: CONTROL
  0x20010010: X_START
  0x20010014: Y_START
  0x20010018: X_END
  0x2001001C: Y_END
  0x20010020: COLOR
  0x20011000: FRAMEBUFFER (32 KB)
0x20020000 - 0x2002FFFF: GPIO (64 KB)
```

---

## 硬件 BOM（外部器件）

| 器件 | 型号 | 规格 | 用途 | 成本 |
|------|------|------|------|------|
| USB-UART | FTDI FT232H | USB 2.0 | USB 转串口 | ~$5 |
| TFT LCD | ST7735 | 128x128, RGB565 | 彩色显示 | ~$3-5 |
| 连接器 | 2.54mm 排针 | - | 接口 | ~$1 |
| **总计** | | | | **~$9-11** |

**推荐的 TFT LCD 模块：**
- Adafruit 1.44" TFT LCD (ST7735)
- Waveshare 1.44" LCD Module
- 淘宝 ST7735 128x128 模块

---

## 示例应用

### 1. AI 推理结果显示
```c
void display_inference_result() {
    lcd_clear(COLOR_BLACK);
    
    // 标题
    lcd_draw_string(10, 10, "AI Inference", COLOR_WHITE, COLOR_BLACK);
    
    // 结果
    lcd_draw_string(10, 30, "Class: Cat", COLOR_GREEN, COLOR_BLACK);
    lcd_printf(10, 50, COLOR_CYAN, COLOR_BLACK, "Conf: %.2f%%", confidence);
    
    // 进度条
    lcd_draw_rect(10, 70, 108, 10, COLOR_WHITE);
    lcd_fill_rect(11, 71, (int)(confidence * 106 / 100), 8, COLOR_GREEN);
    
    // FPS
    lcd_printf(10, 90, COLOR_YELLOW, COLOR_BLACK, "FPS: %d", fps);
}
```

### 2. 系统状态监控
```c
void display_system_status() {
    lcd_clear(COLOR_BLACK);
    
    lcd_draw_string(10, 10, "System Status", COLOR_WHITE, COLOR_BLACK);
    
    lcd_printf(10, 30, COLOR_GREEN, COLOR_BLACK, "CPU: %d MHz", cpu_freq);
    lcd_printf(10, 50, COLOR_CYAN, COLOR_BLACK, "Mem: %d KB", mem_used);
    lcd_printf(10, 70, COLOR_YELLOW, COLOR_BLACK, "Temp: %d C", temperature);
    
    // 状态指示
    if (system_ok) {
        lcd_fill_circle(100, 100, 10, COLOR_GREEN);
    } else {
        lcd_fill_circle(100, 100, 10, COLOR_RED);
    }
}
```

### 3. 启动 Logo
```c
void display_boot_logo() {
    lcd_clear(COLOR_BLACK);
    
    // 显示 Logo（从 Flash 读取）
    lcd_draw_image(32, 32, 64, 64, logo_data);
    
    lcd_draw_string(20, 100, "RISC-V AI Chip", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(40, 115, "v0.2", COLOR_CYAN, COLOR_BLACK);
}
```

---

## 开发 Checklist

### Phase 1: UART 控制器（3 天）
- [ ] 创建 `RealUART.scala` 文件
- [ ] 实现波特率生成器
- [ ] 实现发送状态机
- [ ] 实现接收状态机
- [ ] 添加 FIFO 缓冲
- [ ] 实现寄存器接口
- [ ] 编写单元测试
- [ ] 集成到 SoC
- [ ] 验证串口通信

### Phase 2: TFT LCD 控制器（5-7 天）
- [ ] 创建 `TFTLCD.scala` 文件
- [ ] 实现 SPI 控制器
- [ ] 实现 ST7735 初始化序列
- [ ] 实现像素写入
- [ ] 实现区域设置
- [ ] 添加帧缓冲支持
- [ ] 实现寄存器接口
- [ ] 编写单元测试
- [ ] 集成到 SoC
- [ ] 验证 LCD 显示

### Phase 3: 程序上传协议（2-3 天）
- [ ] 编写 Bootloader C 代码
- [ ] 实现命令解析
- [ ] 实现程序上传
- [ ] 实现内存读写
- [ ] 编写 Python 上传工具
- [ ] 测试程序上传
- [ ] 测试串口交互

### Phase 4: 图形库（2-3 天）
- [ ] 实现基本图形函数
- [ ] 添加 8x8 字体
- [ ] 实现文本显示
- [ ] 实现图像显示
- [ ] 编写示例程序
- [ ] 测试所有功能

### Phase 5: 集成测试（2-3 天）
- [ ] 综合测试
- [ ] 性能测试
- [ ] 功耗测试
- [ ] 编写文档
- [ ] 准备演示程序

### Phase 6: FPGA 验证（2-3 天）
- [ ] 生成 Verilog
- [ ] FPGA 综合
- [ ] 时序分析
- [ ] 硬件测试
- [ ] 修复问题

---

## 时间估算

| 阶段 | 工作量 | 依赖 |
|------|--------|------|
| Phase 1: UART | 3 天 | - |
| Phase 2: TFT LCD | 5-7 天 | - |
| Phase 3: 上传协议 | 2-3 天 | Phase 1 |
| Phase 4: 图形库 | 2-3 天 | Phase 2 |
| Phase 5: 集成测试 | 2-3 天 | Phase 1-4 |
| Phase 6: FPGA 验证 | 2-3 天 | Phase 5 |
| **总计** | **16-22 天** | |

---

## 验收标准

### 功能验收
- [x] 通过 USB 串口上传程序成功率 > 99%
- [x] 串口通信稳定，无丢包
- [x] TFT LCD 显示正常，无花屏
- [x] 帧率 > 10 FPS（128x128 全屏刷新）
- [x] 图形库功能完整
- [x] Bootloader 稳定运行

### 性能验收
- [x] UART 波特率：115200 bps
- [x] LCD SPI 时钟：10-15 MHz
- [x] 程序上传速度：> 10 KB/s
- [x] LCD 刷新率：> 10 FPS

### 质量验收
- [x] 代码覆盖率 > 80%
- [x] 所有单元测试通过
- [x] FPGA 综合无错误
- [x] 时序满足要求
- [x] 文档完整

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| TFT LCD 时序复杂 | 中 | 高 | 参考现有实现，充分测试 |
| 帧缓冲占用资源多 | 高 | 中 | 优化存储，考虑外部 SRAM |
| SPI 速度不够 | 低 | 中 | 提高时钟频率，优化传输 |
| UART FIFO 溢出 | 中 | 低 | 增加 FIFO 深度，添加流控 |
| 时间不够 | 中 | 高 | 并行开发，优先核心功能 |

---

## 下一步行动

1. **立即开始：** Phase 1 - UART 控制器
2. **并行开发：** Phase 2 - TFT LCD 控制器
3. **持续集成：** 每完成一个模块立即集成测试
4. **文档同步：** 边开发边写文档

---

## 参考资料

- ST7735 Datasheet
- FTDI FT232H Datasheet
- PicoRV32 Documentation
- Chisel3 Documentation
- SPI Protocol Specification

---

**创建时间：** 2025-11-16
**版本：** v0.2-dev
**状态：** 开发中
