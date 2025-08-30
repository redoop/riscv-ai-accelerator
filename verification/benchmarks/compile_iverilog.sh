#!/bin/bash
# Compile AI benchmark framework using Icarus Verilog

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create work directory
mkdir -p work logs

echo "Compiling AI benchmark framework with Icarus Verilog..."
echo "Working directory: $(pwd)"

# Compile with iverilog
iverilog -g2012 \
    -I../uvm \
    -I. \
    -I../../rtl \
    -I../../rtl/config \
    -I../../rtl/interfaces \
    -I../../rtl/core \
    -I../../rtl/accelerators \
    -I../../rtl/top \
    -o work/ai_benchmark \
    ../../rtl/config/chip_config_pkg.sv \
    ../../rtl/interfaces/system_interfaces.sv \
    ../../rtl/core/riscv_core.sv \
    ../../rtl/accelerators/tpu.sv \
    ../../rtl/accelerators/vpu.sv \
    ../../rtl/top/riscv_ai_chip.sv \
    ../uvm/riscv_ai_interface.sv \
    ../uvm/riscv_ai_pkg.sv \
    ./ai_benchmark_pkg.sv \
    ./tb_ai_benchmarks.sv \
    2>&1 | tee logs/compile_iverilog.log

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: vvp work/ai_benchmark"
else
    echo "Compilation failed. Check logs/compile_iverilog.log for details."
    exit 1
fi