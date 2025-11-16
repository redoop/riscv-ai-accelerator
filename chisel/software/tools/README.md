# RISC-V AI SoC Upload Tools

## Tools

### upload.py
Python script for uploading programs to real hardware via USB-UART.

**Requirements:**
```bash
pip install pyserial Pillow
```

**Usage:**
```bash
# Upload and run program
python upload.py /dev/ttyUSB0 ../build/hello_lcd.bin --run

# Get bootloader info
python upload.py /dev/ttyUSB0 --info

# Test LCD
python upload.py /dev/ttyUSB0 --test-lcd

# Display image
python upload.py /dev/ttyUSB0 --image logo.png
```

### test_upload.sh
Simulator for testing upload process without hardware.

**Usage:**
```bash
# Test hello_lcd upload
./test_upload.sh hello_lcd

# Test benchmark upload
./test_upload.sh benchmark

# Test any program
./test_upload.sh <program_name>
```

## Hardware Setup

### Required Hardware
1. RISC-V AI SoC board
2. USB-UART adapter (FTDI FT232H or similar)
3. TFT LCD display (ST7735, 128x128)
4. USB cable

### Connections

```
FTDI FT232H          RISC-V AI SoC
-----------          -------------
TX        ---------> RX (UART)
RX        <--------- TX (UART)
GND       ---------> GND

ST7735 LCD           RISC-V AI SoC
----------           -------------
SCK       <--------- SPI_CLK
MOSI      <--------- SPI_MOSI
CS        <--------- SPI_CS
DC        <--------- SPI_DC
RST       <--------- SPI_RST
VCC       ---------> 3.3V
GND       ---------> GND
LED       <--------- BACKLIGHT
```

### Finding Serial Port

**macOS:**
```bash
ls /dev/tty.*
# Look for /dev/tty.usbserial-* or similar
```

**Linux:**
```bash
ls /dev/ttyUSB*
# Usually /dev/ttyUSB0
```

**Windows:**
```bash
# Check Device Manager
# Usually COM3, COM4, etc.
```

## Workflow

### 1. Build Programs
```bash
cd chisel/software
make all
```

### 2. Test Locally (Simulator)
```bash
./tools/test_upload.sh hello_lcd
```

### 3. Upload to Hardware
```bash
# Find your serial port
ls /dev/tty*

# Upload bootloader first
make run PROG=bootloader PORT=/dev/ttyUSB0

# Upload and run example
make run PROG=hello_lcd PORT=/dev/ttyUSB0
```

### 4. Monitor Output
```bash
# Use screen or minicom
screen /dev/ttyUSB0 115200

# Or use Python
python -m serial.tools.miniterm /dev/ttyUSB0 115200
```

## Troubleshooting

### "Permission denied" on serial port
```bash
# Linux
sudo chmod 666 /dev/ttyUSB0
# Or add user to dialout group
sudo usermod -a -G dialout $USER

# macOS
sudo chmod 666 /dev/tty.usbserial-*
```

### "Device not found"
1. Check USB connection
2. Check if FTDI drivers are installed
3. Try different USB port
4. Check `dmesg` (Linux) or Console.app (macOS) for errors

### "Upload failed"
1. Make sure bootloader is running
2. Check baud rate (should be 115200)
3. Try resetting the board
4. Check serial port permissions

### "No output on LCD"
1. Check LCD connections
2. Verify LCD power (3.3V)
3. Check SPI signals with logic analyzer
4. Try LCD test: `make test-lcd PORT=/dev/ttyUSB0`

## Examples

### Upload Hello World
```bash
make run PROG=hello_lcd PORT=/dev/ttyUSB0
```

### Upload AI Demo
```bash
make run PROG=ai_demo PORT=/dev/ttyUSB0
```

### Run Benchmark
```bash
make run PROG=benchmark PORT=/dev/ttyUSB0
```

### System Monitor
```bash
make run PROG=system_monitor PORT=/dev/ttyUSB0
```

## Advanced Usage

### Manual Upload with Python
```python
from upload import RISCVUploader

uploader = RISCVUploader('/dev/ttyUSB0')
uploader.upload_program('build/hello_lcd.bin')
uploader.run_program()
```

### Display Custom Image
```python
uploader.lcd_display_image('my_image.png')
```

### Read Memory
```python
data = uploader.read_memory(0x00010000, 256)
print(data.hex())
```

## Tips

1. Always upload bootloader first
2. Use simulator for quick testing
3. Monitor UART output for debugging
4. Keep programs small (< 64KB)
5. Use LCD for visual feedback
6. Check power supply (3.3V, stable)

## Support

For issues or questions:
1. Check documentation in `chisel/docs/`
2. Review example programs in `chisel/software/examples/`
3. Check build logs in `chisel/software/build/`
