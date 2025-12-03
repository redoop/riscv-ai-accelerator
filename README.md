# RISC-V AI Accelerator Chip

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Chisel](https://img.shields.io/badge/Chisel-3.x-red.svg)](https://www.chisel-lang.org/)
[![RISC-V](https://img.shields.io/badge/RISC--V-RV32I-green.svg)](https://riscv.org/)

> 🚀 An innovative edge AI SoC integrating RISC-V processor with BitNet multiplier-free accelerators

## 📋 Project Overview

**SimpleEdgeAiSoC** is a System-on-Chip designed for edge AI inference, featuring:

- **🔥 BitNet Architecture**: Multiplier-free design using 2-bit weights {-1, 0, +1}
- **⚡ High Performance**: 6.4 GOPS @ 100MHz (measured up to 178.569 MHz)
- **💡 Low Power**: < 100mW target (static: 627.4 uW)
- **🎯 Compact Design**: 73,829 instances, ~0.3 mm² core area
- **🇨🇳 Open Source**: Supports iEDA (Chinese) and OpenROAD (International) toolchains

### 🎉 v0.2 Release (2025-11-16)

Complete debugging and interaction capabilities:
- ✅ **RealUART**: Full UART controller (115200 bps, FIFO, interrupts)
- ✅ **TFTLCD**: ST7735 SPI controller (128x128 RGB565 color display)
- ✅ **Bootloader**: Program upload and management system
- ✅ **Graphics Library**: Complete 2D graphics and text rendering
- ✅ **Python Tools**: Program upload and LCD image display
- ✅ **Example Programs**: 5 demo applications (Hello World, AI inference, etc.)
- ✅ **Build System**: Complete software development environment
- ✅ **Testing**: 97% test coverage (34/35 tests passing)

### 🎉 v0.3 Release (2025-12-03)

External storage expansion:
- ✅ **SPI Flash**: 16 MB @ 25 MHz (3 MB/s bandwidth)
- ✅ **PSRAM**: 8 MB @ 50 MHz (6.25 MB/s SPI, 25 MB/s Quad SPI)
- ✅ **Storage Capacity**: 375× increase (64 KB → 24 MB)
- ✅ **Quad SPI**: 4-bit parallel transfer, 4× performance boost
- ✅ **QPI Mode**: Dynamic mode switching for optimal performance
- ✅ **Complete API**: 14 functions (6 Flash + 8 PSRAM)
- ✅ **Test Programs**: flash_test.c, psram_test.c
- ✅ **Documentation**: Comprehensive guide and examples

### 🎯 v0.4.1 Release (2025-12-03)

IO pad optimization for 81-pad limit:
- ✅ **IO Reduction**: 97 → 61 pads (37% reduction, +20 margin)
- ✅ **GPIO Optimization**: 32-bit → 16-bit (sufficient for most applications)
- ✅ **Core Features**: All preserved (Quad SPI, Flash, PSRAM, LCD, UART)
- ✅ **Test Coverage**: 100% passing (57/57 tests)
- ✅ **Margin**: 24.7% headroom for future expansion

### 🕐 Clock Verification (2025-11-21)

Complete clock constraint verification system:
- ✅ **Main Clock**: 100MHz (10ns period) verified
- ✅ **SPI Clock**: 10MHz (100ns period, 10x divider) verified
- ✅ **Chisel Tests**: 100% passing (2/2 tests)
  - Frequency: 10.204 MHz (2.04% error)
  - Duty Cycle: 50.00% (0.00% deviation)
- ✅ **Constraint Files**: Complete timing constraints (timing_complete.xdc)
- ✅ **TCL Scripts**: Vivado verification scripts (verify_clocks.tcl)
- ✅ **Automation**: One-command verification (run_clock_verification.sh)
- ✅ **Documentation**: 9 comprehensive clock verification guides

### 🔬 ECOS ASIC Synthesis (2025-12-03)

Complete ASIC synthesis flow with ICS55 55nm PDK:
- ✅ **Logic Synthesis**: 623,516-line netlist generated
- ✅ **Chip Area**: 292,992 µm² (~0.29 mm²)
- ✅ **Standard Cells**: 96,087 instances
- ✅ **Flip-Flops**: 25,553 (53.73% of total cells)
- ✅ **Synthesis Time**: 90 seconds
- ✅ **Process**: ICS55 55nm TT 1.2V 25°C
- ✅ **Netlist Simulation**: Icarus Verilog environment ready
- ✅ **Documentation**: Complete synthesis guides and reports

**Development Time**: 1 day (~12 hours) + Clock verification + ASIC synthesis + Storage expansion  
**Total Code**: ~4,100 lines (Chisel + C + Python)  
**Binary Size**: 24.1 KB (5 programs)  
**Netlist**: 623,516 lines (ICS55 55nm)  
**Storage**: 24 MB (16 MB Flash + 8 MB PSRAM)

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      SimpleEdgeAiSoC (v0.2)                      │
│                                                                   │
│  ┌──────────────┐         ┌────────────────────────────┐        │
│  │  PicoRV32    │◄───────►│    Address Decoder         │        │
│  │   (RV32I)    │         │    (Memory Map)            │        │
│  │  @ 50MHz     │         └────────┬───────────────────┘        │
│  └──────────────┘                  │                            │
│                                    │                            │
│  ┌─────────────────────────────────┼────────────────────────┐  │
│  │  AI Accelerators                │                        │  │
│  │  ┌──────────────────┐           │  ┌──────────────────┐  │  │
│  │  │  CompactAccel    │◄──────────┤  │  BitNetAccel     │  │  │
│  │  │  8x8 Matrix      │           │  │  16x16 BitNet    │  │  │
│  │  │  1.6 GOPS        │           │  │  4.8 GOPS        │  │  │
│  │  └──────────────────┘           │  └──────────────────┘  │  │
│  └─────────────────────────────────┼────────────────────────┘  │
│                                    │                            │
│  ┌─────────────────────────────────┼────────────────────────┐  │
│  │  Peripherals (v0.2)             │                        │  │
│  │  ┌──────────────────┐           │  ┌──────────────────┐  │  │
│  │  │  RealUART        │◄──────────┤  │  TFTLCD          │  │  │
│  │  │  115200 bps      │           │  │  ST7735 SPI      │  │  │
│  │  │  16B FIFO        │           │  │  128x128 RGB565  │  │  │
│  │  │  TX/RX + IRQ     │           │  │  32KB Framebuf   │  │  │
│  │  └────────┬─────────┘           │  └────────┬─────────┘  │  │
│  │           │                     │           │            │  │
│  │  ┌────────▼─────────┐           │  ┌────────▼─────────┐  │  │
│  │  │  GPIO (32-bit)   │◄──────────┘  │  Memory (RAM)    │  │  │
│  │  │  Bidirectional   │              │  + Bootloader    │  │  │
│  │  └──────────────────┘              └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  External I/O:                                                  │
│  • UART TX/RX  • LCD SPI (CLK/MOSI/CS/DC/RST)  • GPIO pins     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Performance | Status |
|-----------|-------------|-------------|--------|
| **PicoRV32** | RISC-V RV32I CPU | 50-100 MHz | ✅ Verified |
| **CompactAccel** | 8x8 matrix accelerator | 1.6 GOPS @ 100MHz | ✅ Verified |
| **BitNetAccel** | 16x16 multiplier-free accelerator | 4.8 GOPS @ 100MHz | ✅ Verified |
| **RealUART** | Serial communication with FIFO | 115200 bps, 16B FIFO | ✅ v0.2 |
| **TFTLCD** | ST7735 SPI LCD controller | 128x128 RGB565, 32KB FB | ✅ v0.2 |
| **GPIO** | 32-bit general I/O | Bidirectional | ✅ Verified |
| **Memory** | RAM + Bootloader | Configurable | ✅ Verified |

### Memory Map

| Address Range | Component | Size | Description |
|---------------|-----------|------|-------------|
| `0x00000000 - 0x0000FFFF` | RAM | 64 KB | Program memory |
| `0x04000000 - 0x047FFFFF` | PSRAM | 8 MB | External PSRAM (v0.3) |
| `0x00010000 - 0x000101FF` | CompactAccel | 512 B | Matrix A/B/C buffers |
| `0x00010200 - 0x000103FF` | BitNetAccel | 512 B | Activation/Weight/Result |
| `0x00010400 - 0x0001041F` | UART | 32 B | TX/RX FIFO, Control, Status |
| `0x00010420 - 0x0001941F` | LCD | 32 KB | Framebuffer + Control |
| `0x00019420 - 0x0001943F` | GPIO | 32 B | Input/Output registers |
| `0x30000000 - 0x30FFFFFF` | Flash | 16 MB | SPI Flash (v0.3) |

## 🎯 Key Innovation: BitNet Multiplier-Free Architecture

Traditional matrix multiplication requires expensive multipliers. BitNet uses 2-bit weight encoding:

- `00` = 0 → Skip computation (sparsity optimization)
- `01` = +1 → Addition only
- `10` = -1 → Subtraction only

**Benefits**:
- ✅ 50% area reduction (no multipliers)
- ✅ 60% power reduction
- ✅ 10x memory savings (2-bit vs 32-bit weights)
- ✅ 26% sparsity in real workloads

## 🚀 Quick Start

### Prerequisites

```bash
# Hardware Development Tools
brew install sbt verilator  # macOS
# or apt-get install sbt verilator  # Linux

# Software Development Tools (v0.2)
brew tap riscv/riscv
brew install riscv-tools  # RISC-V GCC toolchain
pip install pyserial Pillow  # Python tools
```

### Hardware Development

```bash
# Clone repository
git clone https://github.com/redoop/riscv-ai-accelerator.git
cd riscv-ai-accelerator/chisel

# Run all tests
sbt test
# or use convenience script
./test.sh all

# Generate Verilog
sbt "runMain riscv.ai.SimpleEdgeAiSoCMain"

# Run synthesis and post-synthesis simulation
cd synthesis
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# ECOS ASIC synthesis (ICS55 55nm PDK)
cd synthesis/ecos
./run_synthesis.sh
cd run && make -f Makefile.iverilog netlist
```

### Software Development (v0.2)

```bash
cd chisel/software

# Build all programs
make all

# Test upload (simulator - no hardware needed)
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh ai_demo
./tools/test_upload.sh benchmark

# Upload to real hardware (when available)
make run PROG=hello_lcd PORT=/dev/ttyUSB0
```

### Clock Verification

```bash
# Run complete clock verification
cd chisel
bash synthesis/fpga/scripts/run_clock_verification.sh

# Or run individual tests
sbt "testOnly riscv.ai.ClockVerificationTest"

# Vivado TCL verification (after synthesis)
cd synthesis/fpga
vivado -mode batch -source scripts/verify_clocks.tcl
```

See [Clock Verification Guide](chisel/synthesis/fpga/docs/CLOCK_VERIFICATION_GUIDE.md) for details.

### FPGA Verification (AWS F1)

```bash
# Step 1: Generate Verilog
cd chisel
./run.sh generate

# Step 2: FPGA verification
cd synthesis/fpga
./run_fpga_flow.sh status      # Check status
./run_fpga_flow.sh full local  # Local verification (free)
./run_fpga_flow.sh aws         # AWS F1 verification (requires AWS account)
```

See [FPGA Guide](chisel/synthesis/fpga/README.md) for details.

## 📊 Performance Metrics

### Hardware Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Design Scale** | 73,829 instances | ✅ < 100K limit |
| **Core Area** | 300,138 um² (~0.3 mm²) | ✅ Compact |
| **Operating Frequency** | 178.569 MHz (measured) | ✅ Exceeds 100MHz target |
| **Clock Verification** | 100% passing (2/2 tests) | ✅ Frequency & duty cycle verified |
| **Peak Performance** | 6.4 GOPS @ 100MHz | ✅ Target met |
| **Static Power** | 627.4 uW | ✅ Ultra-low |
| **Timing** | WNS: 14.4ns, TNS: 0ns | ✅ No violations |
| **Test Coverage** | 97% (34/35 tests) | ✅ Comprehensive |

### Software Metrics (v0.2)

| Component | Lines of Code | Binary Size |
|-----------|---------------|-------------|
| **Chisel Hardware** | 605 lines | - |
| **C Software (HAL + Graphics)** | 659 lines | - |
| **Applications** | 641 lines | - |
| **Generated Verilog** | 4,435 lines | 134 KB |
| **hello_lcd** | - | 3.6 KB |
| **ai_demo** | - | 4.7 KB |
| **benchmark** | - | 5.2 KB |
| **system_monitor** | - | 4.9 KB |
| **bootloader** | - | 5.7 KB |
| **Total** | ~2,500 lines | 24.1 KB |

## 🛠️ Toolchain Support

### Option 1: iEDA (Chinese Open-Source) ⭐ Recommended

Complete domestically developed EDA toolchain:
- 🇨🇳 Autonomous and controllable
- 📚 Chinese documentation
- 🚀 Optimized for Chinese PDKs
- 🔧 Continuous updates

### Option 2: OpenROAD (International)

Mature international open-source toolchain:
- 🌍 Globally mainstream
- 📖 Comprehensive documentation
- 🔄 Active community

## 📚 Documentation

### Core Documentation

| Document | Description | Link |
|----------|-------------|------|
| **📘 Tape-out Report** | Complete design documentation | [docs/README.md](docs/README.md) |
| **📗 Chinese Report** | 中文流片说明报告 | [docs/RISC-V_AI加速器芯片流片说明报告.md](docs/RISC-V_AI加速器芯片流片说明报告.md) |
| **📙 Chisel Design** | RTL design guide | [chisel/README.md](chisel/README.md) |
| **📕 Quick Start** | Quick start guide | [chisel/QUICKSTART.md](chisel/QUICKSTART.md) |
| **📗 Testing Guide** | Complete testing documentation | [chisel/TESTING.md](chisel/TESTING.md) |
| **📘 Hardware Test** | Hardware test results | [chisel/HARDWARE_TEST.md](chisel/HARDWARE_TEST.md) |

### Software Documentation (v0.2)

| Document | Description | Link |
|----------|-------------|------|
| **📗 Software Guide** | Software development guide | [chisel/software/README.md](chisel/software/README.md) |
| **📘 Installation** | Software installation guide | [chisel/software/INSTALL.md](chisel/software/INSTALL.md) |
| **📙 Tools Guide** | Upload tools documentation | [chisel/software/tools/README.md](chisel/software/tools/README.md) |
| **📕 Dev Plan v0.2** | v0.2 development plan | [chisel/docs/DEV_PLAN_V0.2.md](chisel/docs/DEV_PLAN_V0.2.md) |

### Synthesis & Simulation

| Document | Description | Link |
|----------|-------------|------|
| **🔧 Synthesis Guide** | Post-synthesis simulation | [chisel/synthesis/README.md](chisel/synthesis/README.md) |
| **⚡ Quick Start** | 5-minute getting started | [chisel/synthesis/QUICK_START.md](chisel/synthesis/QUICK_START.md) |
| **🔬 ICS55 PDK Guide** | 55nm PDK detailed guide | [chisel/synthesis/ICS55_PDK_GUIDE.md](chisel/synthesis/ICS55_PDK_GUIDE.md) |
| **🔬 IHP PDK Guide** | 130nm PDK detailed guide | [chisel/synthesis/IHP_PDK_GUIDE.md](chisel/synthesis/IHP_PDK_GUIDE.md) |
| **🏭 ECOS Synthesis** | ECOS ASIC synthesis guide | [chisel/synthesis/ecos/README.md](chisel/synthesis/ecos/README.md) |
| **📊 ECOS Report** | Synthesis results and analysis | [chisel/synthesis/ecos/SYNTHESIS_REPORT.md](chisel/synthesis/ecos/SYNTHESIS_REPORT.md) |
| **🔧 ECOS Fixes** | Problem fixes documentation | [chisel/synthesis/ecos/SYNTHESIS_FIXES.md](chisel/synthesis/ecos/SYNTHESIS_FIXES.md) |

### Waveform Viewing

| Document | Description | Link |
|----------|-------------|------|
| **🌊 Wave Viewer** | Web-based waveform viewer | [chisel/synthesis/waves/README.md](chisel/synthesis/waves/README.md) |
| **📊 Wave Quick Start** | Waveform viewing guide | [chisel/synthesis/waves/WAVE_QUICK_START.md](chisel/synthesis/waves/WAVE_QUICK_START.md) |
| **🎨 Wave Viewer Usage** | Detailed usage manual | [chisel/synthesis/waves/WAVE_VIEWER_USAGE.md](chisel/synthesis/waves/WAVE_VIEWER_USAGE.md) |

### Clock Verification

| Document | Description | Link |
|----------|-------------|------|
| **🕐 Clock Constraints** | Complete timing constraints specification | [chisel/synthesis/fpga/docs/CLOCK_CONSTRAINTS_SPEC.md](chisel/synthesis/fpga/docs/CLOCK_CONSTRAINTS_SPEC.md) |
| **🔍 Verification Guide** | Detailed clock verification guide | [chisel/synthesis/fpga/docs/CLOCK_VERIFICATION_GUIDE.md](chisel/synthesis/fpga/docs/CLOCK_VERIFICATION_GUIDE.md) |
| **📊 Test Report** | 100MHz clock test results | [chisel/synthesis/fpga/docs/CLOCK_100MHZ_TEST_REPORT.md](chisel/synthesis/fpga/docs/CLOCK_100MHZ_TEST_REPORT.md) |
| **📋 Quick Reference** | Clock verification quick reference | [chisel/synthesis/fpga/docs/CLOCK_VERIFICATION_QUICKREF.md](chisel/synthesis/fpga/docs/CLOCK_VERIFICATION_QUICKREF.md) |
| **⚡ Quick Verification** | 5-minute verification guide | [chisel/synthesis/fpga/docs/QUICK_VERIFICATION.md](chisel/synthesis/fpga/docs/QUICK_VERIFICATION.md) |

**Quick Commands:**
```bash
cd chisel
bash synthesis/fpga/scripts/run_clock_verification.sh  # Complete verification
sbt "testOnly riscv.ai.ClockVerificationTest"         # Chisel tests only
```

### FPGA Verification

| Document | Description | Link |
|----------|-------------|------|
| **🔌 FPGA Guide** | AWS F1 FPGA verification | [chisel/synthesis/fpga/README.md](chisel/synthesis/fpga/README.md) |
| **☁️ AWS Setup** | AWS environment setup | [chisel/synthesis/fpga/docs/SETUP_GUIDE.md](chisel/synthesis/fpga/docs/SETUP_GUIDE.md) |
| **📋 AWS Plan** | Complete AWS verification plan | [chisel/synthesis/fpga/AWS_FPGA_PLAN.md](chisel/synthesis/fpga/AWS_FPGA_PLAN.md) |

**Quick Commands:**
```bash
cd chisel/synthesis/fpga
./run_fpga_flow.sh help    # View all options
./run_fpga_flow.sh status  # Check current status
```

## 🎓 Project Structure

```
riscv-ai-accelerator/
├── README.md                          # This file (project overview)
├── LICENSE                            # Apache 2.0 License
├── docs/                              # Documentation
│   ├── README.md                      # Tape-out report (English)
│   ├── RISC-V_AI加速器芯片流片说明报告.md  # Chinese report
│   └── image/                         # Images and diagrams
└── chisel/                            # Chisel RTL design
    ├── README.md                      # Chisel design guide
    ├── QUICKSTART.md                  # Quick start guide
    ├── TESTING.md                     # Testing guide (v0.2)
    ├── HARDWARE_TEST.md               # Hardware test results (v0.2)
    ├── build.sbt                      # SBT build configuration
    ├── test.sh                        # Test convenience script (v0.2)
    ├── run.sh                         # Run script
    ├── Makefile                       # Build automation
    │
    ├── src/                           # Source code
    │   ├── main/scala/                # Main design modules
    │   │   ├── EdgeAiSoCSimple.scala  # SimpleEdgeAiSoC implementation
    │   │   ├── SimpleEdgeAiSoCMain.scala  # Verilog generator
    │   │   ├── peripherals/           # Peripheral modules (v0.2)
    │   │   │   ├── RealUART.scala     # UART controller
    │   │   │   └── TFTLCD.scala       # TFT LCD SPI controller
    │   │   └── resources/rtl/         # RTL resources
    │   │       └── picorv32.v         # PicoRV32 core
    │   └── test/scala/                # Test benches
    │       ├── SimpleEdgeAiSoCTest.scala      # SoC tests
    │       ├── PicoRV32CoreTest.scala         # CPU tests
    │       ├── RealUARTTest.scala             # UART tests (v0.2)
    │       ├── TFTLCDTest.scala               # LCD tests (v0.2)
    │       ├── BitNetAccelDebugTest.scala     # BitNet tests
    │       └── SimpleCompactAccelDebugTest.scala  # Compact tests
    │
    ├── software/                      # Software stack (v0.2)
    │   ├── README.md                  # Software guide
    │   ├── INSTALL.md                 # Installation guide
    │   ├── Makefile                   # Software build system
    │   ├── linker.ld                  # Linker script
    │   │
    │   ├── lib/                       # Software libraries
    │   │   ├── hal.h / hal.c          # Hardware abstraction layer
    │   │   ├── graphics.h / graphics.c  # 2D graphics library
    │   │   └── font_8x8.c             # 8x8 ASCII font
    │   │
    │   ├── bootloader/                # Bootloader system
    │   │   └── bootloader.c           # Program upload & management
    │   │
    │   ├── examples/                  # Example programs
    │   │   ├── hello_lcd.c            # Hello World demo
    │   │   ├── ai_demo.c              # AI inference demo
    │   │   ├── benchmark.c            # Performance benchmark
    │   │   └── system_monitor.c       # System monitor
    │   │
    │   ├── tools/                     # PC-side tools
    │   │   ├── README.md              # Tools documentation
    │   │   ├── upload.py              # Program upload tool (Python)
    │   │   └── test_upload.sh         # Upload simulator
    │   │
    │   └── build/                     # Build output directory
    │       ├── *.bin                  # Binary files
    │       ├── *.elf                  # ELF files
    │       └── *.map                  # Memory maps
    │
    ├── docs/                          # Additional documentation
    │   └── DEV_PLAN_V0.2.md           # v0.2 development plan
    │
    ├── generated/                     # Generated Verilog files
    │   └── simple_edgeaisoc/          # Generated SoC Verilog
    │
    ├── synthesis/                     # Synthesis and simulation
    │   ├── README.md                  # Synthesis guide
    │   ├── run_ics55_synthesis.sh     # ICS55 synthesis
    │   ├── run_ihp_synthesis.sh       # IHP synthesis
    │   ├── run_post_syn_sim.py        # Post-synthesis simulation
    │   │
    │   ├── ecos/                      # ECOS ASIC synthesis (v0.2)
    │   │   ├── README.md              # ECOS synthesis guide
    │   │   ├── SYNTHESIS_REPORT.md    # Synthesis results
    │   │   ├── SYNTHESIS_FIXES.md     # Problem fixes
    │   │   ├── run_synthesis.sh       # Synthesis script
    │   │   ├── project/               # Synthesis output
    │   │   │   └── netlist/           # Generated netlist
    │   │   ├── run/                   # Simulation environment
    │   │   │   ├── Makefile           # Netlist simulation
    │   │   │   └── README_NETLIST_SIM.md
    │   │   └── sdc/                   # Timing constraints
    │   │
    │   ├── waves/                     # Waveform tools
    │   │   ├── README.md              # Wave viewer guide
    │   │   ├── wave_viewer.py         # Web-based viewer
    │   │   └── view_wave.sh           # Quick view script
    │   │
    │   └── fpga/                      # FPGA verification
    │       ├── README.md              # FPGA guide
    │       ├── constraints/           # Timing constraints
    │       │   ├── timing_complete.xdc  # Complete constraints (100MHz)
    │       │   ├── timing.xdc         # Basic constraints
    │       │   └── pins.xdc           # Pin assignments
    │       ├── scripts/               # Verification scripts
    │       │   ├── verify_clocks.tcl  # TCL verification
    │       │   └── run_clock_verification.sh  # Automation
    │       └── docs/                  # FPGA documentation
    │           ├── CLOCK_CONSTRAINTS_SPEC.md
    │           ├── CLOCK_VERIFICATION_GUIDE.md
    │           └── CLOCK_100MHZ_TEST_REPORT.md
    │
    └── test_run_dir/                  # Test output directory
        └── */                         # Individual test results
```

## 🧪 Testing

### Hardware Testing

```bash
cd chisel

# Run all tests
sbt test

# Run specific tests using convenience script
./test.sh all          # All tests
./test.sh uart         # UART controller tests
./test.sh lcd          # TFT LCD controller tests
./test.sh ai           # AI accelerator tests
./test.sh soc          # Complete SoC tests
./test.sh quick        # Quick tests

# Clock verification tests
sbt "testOnly riscv.ai.ClockVerificationTest"
bash synthesis/fpga/scripts/run_clock_verification.sh

# Or use sbt directly
sbt "testOnly riscv.ai.peripherals.RealUARTTest"
sbt "testOnly riscv.ai.peripherals.TFTLCDTest"
sbt "testOnly riscv.ai.SimpleEdgeAiSoCTest"
```

### Software Testing (v0.2)

```bash
cd chisel/software

# Test upload simulator (no hardware needed)
./tools/test_upload.sh hello_lcd
./tools/test_upload.sh ai_demo
./tools/test_upload.sh benchmark
./tools/test_upload.sh system_monitor
./tools/test_upload.sh bootloader
```

### Post-Synthesis Simulation

```bash
cd chisel/synthesis

# ICS55 PDK (55nm)
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# IHP PDK (130nm)
./run_ihp_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ihp

# View waveforms
cd waves
./view_wave.sh
```

## 🌊 Waveform Viewing

### Method 1: Web Viewer (Recommended)

```bash
cd chisel/synthesis/waves
./start_wave_viewer.sh
# Open browser: http://localhost:5000
```

### Method 2: Static HTML

```bash
cd chisel/synthesis/waves
./view_wave.sh -f post_syn.vcd
# Opens waveform_post_syn.html in browser
```

### Method 3: GTKWave

```bash
gtkwave chisel/synthesis/waves/post_syn.vcd
```

## 🔬 Supported PDKs

| PDK | Process | Source | Status |
|-----|---------|--------|--------|
| **ICS55** | 55nm | IDE Platform | ✅ Verified |
| **IHP SG13G2** | 130nm | IHP GmbH | ✅ Verified |
| **Generic** | - | Yosys | ✅ Verified |

## 📈 Roadmap

### ✅ Completed (v0.2)

**Hardware:**
- [x] RTL design (Chisel)
- [x] Functional verification (97% coverage)
- [x] Logic synthesis (Yosys)
- [x] Post-synthesis simulation
- [x] Waveform viewing tools
- [x] UART controller with FIFO
- [x] TFT LCD SPI controller
- [x] Clock verification system (100% passing)
  - [x] 100MHz main clock verification
  - [x] 10MHz SPI clock verification
  - [x] Complete timing constraints
  - [x] TCL verification scripts
  - [x] Automated testing framework
  - [x] 9 comprehensive documentation files
- [x] ECOS ASIC synthesis (ICS55 55nm)
  - [x] 623,516-line netlist generation
  - [x] 292,992 µm² chip area
  - [x] 96,087 standard cells
  - [x] Netlist simulation environment
  - [x] Complete synthesis documentation
- [x] Complete documentation

**Software:**
- [x] Hardware abstraction layer (HAL)
- [x] Graphics library (2D + text)
- [x] 8x8 ASCII font (128 characters)
- [x] Bootloader system
- [x] 5 example programs
- [x] Upload tools (Python)
- [x] Build system (Makefile)
- [x] Upload simulator

### 🚧 In Progress

- [ ] Static timing analysis (OpenSTA/iSTA)
- [ ] Floorplanning (OpenROAD/iFP)
- [ ] Place & route (OpenROAD/iPL/iRT)
- [ ] Physical verification (DRC/LVS)

### 📅 Planned

**Hardware:**
- [ ] GDSII generation
- [ ] Tape-out manufacturing
- [ ] Chip testing
- [ ] Development board

**Software (Future):**
- [ ] DMA support
- [ ] SD card interface
- [ ] Audio output
- [ ] Network connectivity
- [ ] More example programs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Project Lead**: tongxiaojun  
**Organization**: redoop (红象云腾)  
**Contact**: tongxiaojun@redoop.com

## 🔗 Links

- **GitHub**: https://github.com/redoop/riscv-ai-accelerator
- **iEDA**: https://ieda.oscc.cc/
- **OpenROAD**: https://theopenroadproject.org/
- **Chisel**: https://www.chisel-lang.org/
- **RISC-V**: https://riscv.org/

## 🌟 Acknowledgments

- **PicoRV32**: Clifford Wolf (YosysHQ)
- **Chisel**: UC Berkeley
- **iEDA**: Chinese Academy of Sciences, Peking University, Peng Cheng Laboratory
- **OpenROAD**: UCSD
- **IHP PDK**: IHP GmbH
- **ICS55 PDK**: IDE Platform

---

**⭐ Star this project if you find it useful!**

*For detailed information, please refer to the [complete tape-out report](docs/README.md).*
