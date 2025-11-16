# Installation Guide

## Installing RISC-V GCC Toolchain

### macOS

```bash
# Using Homebrew
brew tap riscv/riscv
brew install riscv-tools

# Or build from source
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv32i --with-abi=ilp32
make
```

### Ubuntu/Debian

```bash
# Install pre-built toolchain
sudo apt-get update
sudo apt-get install gcc-riscv64-unknown-elf

# Or use the 32-bit version
sudo apt-get install gcc-riscv64-linux-gnu
```

### Building from Source (All Platforms)

```bash
# Clone the toolchain
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain

# Configure for RV32I
./configure --prefix=/opt/riscv --with-arch=rv32i --with-abi=ilp32

# Build (this takes a while)
make

# Add to PATH
export PATH=/opt/riscv/bin:$PATH
```

## Verifying Installation

```bash
# Check if the toolchain is installed
riscv32-unknown-elf-gcc --version

# Should output something like:
# riscv32-unknown-elf-gcc (GCC) 12.2.0
```

## Installing Python Dependencies

```bash
# For upload tool
pip install pyserial

# For image display (optional)
pip install Pillow
```

## Quick Test

```bash
cd chisel/software

# Build all examples
make all

# If successful, you should see:
# - build/hello_lcd.bin
# - build/ai_demo.bin
# - build/benchmark.bin
# - build/system_monitor.bin
# - build/bootloader.bin
```

## Troubleshooting

### "riscv32-unknown-elf-gcc: command not found"

Make sure the toolchain is in your PATH:
```bash
export PATH=/opt/riscv/bin:$PATH
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### "cannot find -lgcc"

The toolchain might not be properly configured. Try rebuilding with:
```bash
./configure --prefix=/opt/riscv --with-arch=rv32i --with-abi=ilp32 --enable-multilib
make clean
make
```

### Linker errors

Make sure you're using the correct architecture flags:
```bash
-march=rv32i -mabi=ilp32
```

## Alternative: Docker

If you don't want to install the toolchain locally, you can use Docker:

```bash
# Pull RISC-V toolchain image
docker pull riscv/riscv-gnu-toolchain

# Run build in container
docker run -v $(pwd):/work -w /work riscv/riscv-gnu-toolchain make all
```

## Next Steps

Once the toolchain is installed:

1. Build examples: `make all`
2. Connect your device via USB-UART
3. Upload a program: `make run PROG=hello_lcd PORT=/dev/ttyUSB0`
4. Enjoy!
