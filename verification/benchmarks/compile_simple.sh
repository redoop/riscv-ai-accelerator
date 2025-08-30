#!/bin/bash
# Compile simple testbench with Icarus Verilog

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create work directory
mkdir -p work logs

echo "Compiling simple testbench with Icarus Verilog..."
echo "Working directory: $(pwd)"

# Compile simple testbench
iverilog -g2012 \
    -o work/simple_test \
    simple_tb.sv \
    2>&1 | tee logs/compile_simple.log

if [ $? -eq 0 ]; then
    echo "Simple testbench compilation successful!"
    echo "Run with: vvp work/simple_test"
    echo ""
    echo "Running test..."
    vvp work/simple_test
else
    echo "Compilation failed. Check logs/compile_simple.log for details."
    exit 1
fi