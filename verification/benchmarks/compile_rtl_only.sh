#!/bin/bash
# Compile RTL-only version without UVM for Icarus Verilog

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create work directory
mkdir -p work logs

echo "Compiling RTL-only version with Icarus Verilog..."
echo "Working directory: $(pwd)"

# First, let's check what files actually exist
echo "Checking available RTL files..."
find ../../rtl -name "*.sv" | head -10

# Compile basic RTL files only (no UVM)
iverilog -g2012 \
    -I../../rtl \
    -I../../rtl/config \
    -I../../rtl/interfaces \
    -I../../rtl/core \
    -I../../rtl/accelerators \
    -I../../rtl/top \
    -o work/rtl_test \
    ../../rtl/config/chip_config_pkg.sv \
    ../../rtl/interfaces/system_interfaces.sv \
    ../../rtl/core/riscv_core.sv \
    ../../rtl/accelerators/tpu.sv \
    ../../rtl/accelerators/vpu.sv \
    ../../rtl/top/riscv_ai_chip.sv \
    2>&1 | tee logs/compile_rtl_only.log

if [ $? -eq 0 ]; then
    echo "RTL compilation successful!"
    echo "Run with: vvp work/rtl_test"
else
    echo "RTL compilation failed. Check logs/compile_rtl_only.log for details."
    exit 1
fi