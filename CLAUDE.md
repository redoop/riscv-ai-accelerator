# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RISC-V AI accelerator SoC designed for edge AI inference, featuring BitNet multiplier-free architecture with 2-bit weight encoding {-1, 0, +1}. The project uses Chisel HDL (Scala) for RTL design and supports both Chinese iEDA and international OpenROAD toolchains.

## Key Directories

- `chisel/` - Main Chisel RTL design and software stack
  - `src/main/scala/` - Core hardware modules
  - `src/test/scala/` - Test benches
  - `software/` - C software stack for the SoC
  - `synthesis/` - Synthesis and simulation scripts
  - `generated/` - Generated Verilog files
- `ecos/` - ECOS ASIC synthesis files
- `docs/` - Documentation

## Build and Test Commands

### Hardware Development (Chisel)

```bash
cd chisel

# Run all tests (convenient script)
./test.sh all

# Run specific test categories
./test.sh uart        # UART controller tests
./test.sh lcd         # TFT LCD controller tests
./test.sh ai          # AI accelerator tests
./test.sh soc         # Complete SoC tests
./test.sh cpu         # PicoRV32 CPU tests
./test.sh quick       # Quick tests (UART + LCD)

# Alternative: use run.sh for common workflows
./run.sh test         # Run all tests (default)
./run.sh soc          # Run SimpleEdgeAiSoC tests
./run.sh generate     # Generate SimpleEdgeAiSoC SystemVerilog
./run.sh full         # Complete workflow (compile+test+generate)

# Direct sbt commands
sbt test              # Run all tests
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"  # Run specific test
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"   # Generate Verilog
```

### Software Development (C)

```bash
cd chisel/software

# Build all programs
make all

# Build specific program
make hello_lcd
make ai_demo
make benchmark
make system_monitor
make bootloader

# Test upload simulator (no hardware needed)
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh ai_demo
```

### Synthesis and Simulation

```bash
cd chisel/synthesis

# ICS55 55nm PDK synthesis
./run_ics55_synthesis.sh

# IHP 130nm PDK synthesis
./run_ihp_synthesis.sh

# Post-synthesis simulation
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# View waveforms
cd waves
./view_wave.sh
```

### Static Timing Analysis (STA)

```bash
cd chisel/synthesis/sta

# Run STA for ICS55
./run_sta.sh

# Note: Requires ICS55 liberty files (.lib) for full analysis
```

### ysyxSoC Integration

```bash
cd chisel/synthesis/ysyxSoc
bash ../run_ysyxsoc_synthesis.sh

# For full ysyxSoC integration
cd chisel/synthesis
bash run_ysyxsocfull_synthesis.sh
```

## Architecture Overview

### Core Components

1. **Processor**: PicoRV32 RISC-V RV32I core
2. **AI Accelerators**:
   - `SimpleCompactAccel`: 8x8 matrix accelerator (1.6 GOPS @ 100MHz)
   - `SimpleBitNetAccel`: 16x16 multiplier-free accelerator (4.8 GOPS @ 100MHz)
3. **Peripherals**:
   - `RealUART`: Full UART controller with 16B FIFO
   - `TFTLCD`: ST7735 SPI LCD controller (128x128 RGB565)
   - `SPIFlash`: 16MB external SPI flash
   - `PSRAM`: 8MB external PSRAM
   - `GPIO`: 32-bit general purpose I/O

### Memory Map

- `0x00000000 - 0x0FFFFFFF`: RAM (256 MB)
- `0x04000000 - 0x047FFFFF`: PSRAM (8 MB)
- `0x10000000 - 0x10000FFF`: CompactAccel (4 KB)
- `0x10001000 - 0x10001FFF`: BitNetAccel (4 KB)
- `0x20000000 - 0x2000FFFF`: UART (64 KB)
- `0x20010000 - 0x2001FFFF`: LCD (64 KB)
- `0x20020000 - 0x2002FFFF`: GPIO (64 KB)
- `0x30000000 - 0x30FFFFFF`: Flash (16 MB)

### Interface Architecture

The SoC uses a **simple register interface** (`SimpleRegIO`) instead of AXI4-Lite to avoid interface direction issues. Key files:

- `EdgeAiSoCSimple.scala`: Main SoC implementation
- `SimpleEdgeAiSoCMain.scala`: Verilog generator
- `SimpleMemoryMap.scala`: Memory map configuration

### BitNet Innovation

The AI accelerators use 2-bit weight encoding:
- `00` = 0 (skip computation, sparsity optimization)
- `01` = +1 (addition only)
- `10` = -1 (subtraction only)

This eliminates multipliers, reducing area by 50% and power by 60%.

## Development Workflow

1. **Hardware Changes**: Modify Scala files in `chisel/src/main/scala/`
2. **Run Tests**: `./test.sh all` or specific category
3. **Generate Verilog**: `./run.sh generate`
4. **Synthesis**: Use appropriate PDK script in `chisel/synthesis/`
5. **Software Development**: Modify C files in `chisel/software/`
6. **Build Software**: `make all` in `chisel/software/`

## Tool Dependencies

- **Java 11+**: Required for sbt/Chisel
- **sbt**: Scala build tool
- **RISC-V GCC toolchain**: For software compilation
- **Verilator/Icarus Verilog**: For simulation
- **Yosys**: For synthesis
- **OpenSTA**: For static timing analysis

## Important Notes

- The project supports multiple PDKs: ICS55 (55nm), IHP (130nm), and generic
- Clock verification system verifies 100MHz main clock and 10MHz SPI clock
- ECOS ASIC synthesis generates 623,516-line netlist with 96,087 standard cells
- ysyxSoC integration provides complete platform wrapper for educational use
- Full STA requires ICS55 liberty files (.lib) which may not be included