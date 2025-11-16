# RISC-V AI SoC Software

Software components for the RISC-V AI SoC v0.2

## Directory Structure

```
software/
├── lib/              # Hardware abstraction layer and libraries
│   ├── hal.h         # Hardware definitions
│   ├── hal.c         # HAL implementation
│   ├── graphics.h    # Graphics library
│   ├── graphics.c    # Graphics implementation
│   └── font_8x8.c    # 8x8 ASCII font
├── bootloader/       # Bootloader
│   └── bootloader.c  # Main bootloader code
├── tools/            # PC-side tools
│   └── upload.py     # Program upload tool
└── examples/         # Example programs
    ├── hello_lcd.c   # Hello World LCD example
    └── ai_demo.c     # AI inference demo
```

## Hardware Abstraction Layer (HAL)

### UART Functions
- `uart_init(baudrate)` - Initialize UART
- `uart_putc(c)` - Send character
- `uart_getc()` - Receive character
- `uart_puts(str)` - Send string
- `uart_rx_ready()` - Check if data available
- `uart_tx_ready()` - Check if TX ready

### LCD Functions
- `lcd_init()` - Initialize LCD
- `lcd_clear(color)` - Clear screen
- `lcd_draw_pixel(x, y, color)` - Draw pixel
- `lcd_fill_rect(x, y, w, h, color)` - Fill rectangle
- `lcd_backlight(on)` - Control backlight

### Graphics Library
- `lcd_draw_line(x0, y0, x1, y1, color)` - Draw line
- `lcd_draw_rect(x, y, w, h, color)` - Draw rectangle
- `lcd_draw_circle(x, y, r, color)` - Draw circle
- `lcd_fill_circle(x, y, r, color)` - Fill circle
- `lcd_draw_char(x, y, c, fg, bg)` - Draw character
- `lcd_draw_string(x, y, str, fg, bg)` - Draw string
- `lcd_printf(x, y, fg, bg, fmt, ...)` - Formatted text
- `lcd_draw_image(x, y, w, h, data)` - Draw image

### Color Definitions (RGB565)
- `COLOR_BLACK`, `COLOR_WHITE`
- `COLOR_RED`, `COLOR_GREEN`, `COLOR_BLUE`
- `COLOR_YELLOW`, `COLOR_CYAN`, `COLOR_MAGENTA`
- `COLOR_ORANGE`, `COLOR_PURPLE`, `COLOR_GRAY`
- `RGB565(r, g, b)` - Convert RGB to RGB565

## Bootloader

The bootloader provides a simple protocol for uploading and running programs.

### Commands
- `U` - Upload program
- `R` - Run program
- `M` - Read memory
- `W` - Write register
- `L` - LCD test
- `P` - Ping
- `I` - Get info

### Upload Protocol
1. Send 'U' command
2. Send program size (4 bytes, little endian)
3. Send program data
4. Wait for 'K' acknowledgment

## PC Tools

### upload.py

Python tool for uploading programs to the SoC.

**Requirements:**
```bash
pip install pyserial
pip install Pillow  # Optional, for image display
```

**Usage:**
```bash
# Get bootloader info
python upload.py /dev/ttyUSB0 --info

# Upload and run program
python upload.py /dev/ttyUSB0 program.bin --run

# Run LCD test
python upload.py /dev/ttyUSB0 --test-lcd

# Display image on LCD
python upload.py /dev/ttyUSB0 --image logo.png
```

## Examples

### hello_lcd.c
Simple "Hello World" example that displays text and shapes on the LCD with animation.

### ai_demo.c
AI inference demo that shows classification results with confidence bars and FPS counter.

### benchmark.c
Performance benchmark program that tests UART, LCD, graphics, and AI accelerator performance.

### system_monitor.c
Real-time system monitor that displays CPU usage, uptime, UART stats, and AI accelerator status.

## Building Programs

### Prerequisites

Install RISC-V GCC toolchain:

```bash
# Ubuntu/Debian:
sudo apt-get install gcc-riscv64-unknown-elf

# macOS:
brew tap riscv/riscv
brew install riscv-tools
```

### Using Makefile (Recommended)

```bash
# Build all examples and bootloader
make all

# Build specific example
make hello_lcd
make ai_demo
make benchmark
make system_monitor

# Build bootloader
make bootloader

# Upload and run program
make run PROG=hello_lcd PORT=/dev/ttyUSB0

# Run LCD test
make test-lcd PORT=/dev/ttyUSB0

# Get bootloader info
make info PORT=/dev/ttyUSB0

# Clean build
make clean

# Show help
make help
```

### Manual Compilation

```bash
riscv32-unknown-elf-gcc -march=rv32i -mabi=ilp32 \
    -nostdlib -nostartfiles \
    -T linker.ld \
    -o program.elf \
    lib/start.S lib/hal.c lib/graphics.c lib/font_8x8.c examples/hello_lcd.c

riscv32-unknown-elf-objcopy -O binary program.elf program.bin
```

## Memory Map

```
0x00000000 - 0x0FFFFFFF: RAM (256 MB)
0x10000000 - 0x10000FFF: CompactAccel (4 KB)
0x10001000 - 0x10001FFF: BitNetAccel (4 KB)
0x20000000 - 0x2000FFFF: UART (64 KB)
0x20010000 - 0x2001FFFF: TFT LCD (64 KB)
0x20020000 - 0x2002FFFF: GPIO (64 KB)
```

## Notes

- Default UART baudrate: 115200 bps
- LCD resolution: 128x128 pixels
- LCD color format: RGB565 (16-bit)
- System clock: 50 MHz
- Program load address: 0x00010000
