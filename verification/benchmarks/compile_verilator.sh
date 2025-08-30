#!/bin/bash
# Compile with Verilator (better SystemVerilog support)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create work directory
mkdir -p work logs

echo "Compiling with Verilator..."
echo "Working directory: $(pwd)"

# Verilator compilation (lint mode first to check syntax)
verilator --lint-only \
    -I../../rtl \
    -I../../rtl/config \
    -I../../rtl/interfaces \
    -I../../rtl/core \
    -I../../rtl/accelerators \
    -I../../rtl/top \
    --top-module simple_tb \
    simple_tb.sv \
    2>&1 | tee logs/verilator_lint.log

if [ $? -eq 0 ]; then
    echo "Verilator lint passed!"
    
    # Full compilation
    verilator --cc --exe \
        -I../../rtl \
        -I../../rtl/config \
        -I../../rtl/interfaces \
        -I../../rtl/core \
        -I../../rtl/accelerators \
        -I../../rtl/top \
        --top-module simple_tb \
        simple_tb.sv \
        --exe-name simple_test \
        -o work/simple_test \
        2>&1 | tee logs/verilator_compile.log
        
    if [ $? -eq 0 ]; then
        echo "Verilator compilation successful!"
        echo "Build executable with: make -C obj_dir -f Vsimple_tb.mk"
    else
        echo "Verilator compilation failed. Check logs/verilator_compile.log"
    fi
else
    echo "Verilator lint failed. Check logs/verilator_lint.log for details."
    exit 1
fi