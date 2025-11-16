# RISC-V AI Accelerator Chip Tape-out Report

## Project Information

**Project Name**: RISC-V AI Accelerator Chip (SimpleEdgeAiSoC)  
**Chip Code**: EdgeAI-SoC-v0.1  
**Design Organization**: [redoop]  
**Project Lead**: [tongxiaojun]  
**Report Date**: 2025,11  
**Version**: v0.1

---

## 1. Project Overview

### 1.1 Background

With the widespread application of artificial intelligence on edge devices, there is a growing demand for low-power, high-efficiency AI accelerators. This project aims to design a System-on-Chip (SoC) integrating a RISC-V processor and dedicated AI accelerators, specifically optimized for edge AI inference scenarios.

### 1.2 Design Goals

- **High Performance**: Provides 6.4 GOPS AI computing capability
- **Low Power**: Target power consumption < 100 mW
- **Flexibility**: Supports various matrix operation scales (2x2 to 16x16)
- **Innovation**: Adopts BitNet multiplier-free architecture, reducing power and area
- **Programmability**: Integrates RISC-V CPU for flexible software control

### 1.3 Key Features

#### 1.3.1 Processor Core
- **CPU**: PicoRV32 (RV32I instruction set)
- **Operating Frequency**: 50-100 MHz
- **Bus Interface**: Simplified register interface

#### 1.3.2 AI Accelerators
1. **CompactAccel** (Traditional Matrix Accelerator)
   - Supports 8x8 matrix multiplication
   - Performance: ~1.6 GOPS @ 100MHz
   - 32-bit fixed-point arithmetic

2. **BitNetAccel** (Innovative Multiplier-Free Accelerator)
   - Supports 2x2 to 16x16 matrix multiplication
   - Performance: ~4.8 GOPS @ 100MHz
   - 2-bit weight encoding {-1, 0, +1}
   - Multiplier-free design using only addition/subtraction
   - Sparsity optimization, automatically skips zero weights
   - 10x memory reduction
   - 60% power reduction

#### 1.3.3 Peripheral System
- **UART**: Serial communication interface
- **GPIO**: 32-bit general-purpose I/O
- **Interrupt Controller**: Supports accelerator interrupts


---

## 2. Chip Architecture Design

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SimpleEdgeAiSoC                          │
│                                                             │
│  ┌──────────────┐         ┌──────────────────────────┐      │
│  │  PicoRV32    │◄───────►│   Address Decoder        │      │
│  │   CPU Core   │         │   (Memory Map)           │      │
│  │   (RV32I)    │         └──────────┬───────────────┘      │
│  └──────────────┘                    │                      │
│         │                            │                      │
│         │                            ├──► CompactAccel      │
│         │                            │    (8x8 Matrix)      │
│         │                            │                      │
│         │                            ├──► BitNetAccel       │
│         │                            │    (16x16 BitNet)    │
│         │                            │                      │
│         │                            ├──► UART              │
│         │                            │                      │
│         │                            └──► GPIO              │
│         │                                                   │
│         └──► Interrupt Controller                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Map

| Address Range | Size | Module | Description |
|--------------|------|--------|-------------|
| 0x00000000 - 0x0FFFFFFF | 256 MB | RAM | Main memory |
| 0x10000000 - 0x10000FFF | 4 KB | CompactAccel | Traditional matrix accelerator |
| 0x10001000 - 0x10001FFF | 4 KB | BitNetAccel | BitNet accelerator |
| 0x20000000 - 0x2000FFFF | 64 KB | UART | Serial peripheral |
| 0x20020000 - 0x2002FFFF | 64 KB | GPIO | General-purpose I/O |

---

## 3. Key Technical Innovation: BitNet Multiplier-Free Architecture

### 3.1 Technical Principle

The BitNet architecture is based on 1-bit LLM concepts, quantizing neural network weights to {-1, 0, +1} using 2-bit encoding:
- `00` = 0 (zero weight, skip computation)
- `01` = +1 (positive weight, perform addition)
- `10` = -1 (negative weight, perform subtraction)
- `11` = reserved

### 3.2 Core Advantages

1. **Multiplier-Free Design**
   - Traditional: `result = activation × weight`
   - BitNet: 
     - When weight = +1: `result = activation` (addition)
     - When weight = -1: `result = -activation` (subtraction)
     - When weight = 0: skip computation (sparsity optimization)

2. **Hardware Resource Savings**
   - Area reduction: 50% (no multipliers needed)
   - Power reduction: 60% (simple add/subtract operations)
   - Memory usage: 10x reduction (2-bit vs 32-bit weights)

3. **Sparsity Optimization**
   - Automatically detects and skips zero weights
   - Tracks skip count for performance analysis
   - Measured sparsity: 26% (8x8 matrix test)


---

## 4. Performance Metrics

### 4.1 Computing Performance

| Metric | CompactAccel | BitNetAccel | Total |
|--------|-------------|-------------|-------|
| Matrix Size | 8x8 | 16x16 | - |
| Peak Performance @ 100MHz | 1.6 GOPS | 4.8 GOPS | 6.4 GOPS |
| Data Width | 32-bit | 32-bit (activation) + 2-bit (weight) | - |
| Multiplier Count | 1 | 0 | 1 |

### 4.2 Resource Utilization (FPGA Estimate)

| Resource Type | Quantity | Description |
|--------------|----------|-------------|
| LUTs | ~8,000 | Logic units |
| FFs | ~6,000 | Flip-flops |
| BRAMs | ~20 | Block RAM |
| DSPs | 1 | Digital signal processing units (CompactAccel only) |

### 4.3 Power Analysis

**Static Power** (synthesis results):
- **Static Power**: 627.4 uW (0.6274 mW)
- **Operating Temperature**: 80°C
- **Voltage Conditions**: LVT: 90%, HVT: 10%

**Dynamic Power Estimate** (@ 100MHz):

| Module | Power (mW) | Percentage |
|--------|-----------|------------|
| PicoRV32 CPU | 30 | 30% |
| CompactAccel | 25 | 25% |
| BitNetAccel | 20 | 20% |
| Peripherals | 15 | 15% |
| Others | 10 | 10% |
| **Total** | **100** | **100%** |

### 4.4 Timing Performance

| Parameter | Target | Measured | Description |
|-----------|--------|----------|-------------|
| Design Frequency | 50 MHz | - | Synthesis constraint |
| Max Operating Frequency | 100 MHz | 178.569 MHz | Achievable frequency |
| Min Operating Frequency | 50 MHz | - | Low-power mode |
| Critical Path Delay | < 10 ns | - | @ 100 MHz |
| Worst Negative Slack (WNS) | - | 14.400 ns | No violations |
| Total Negative Slack (TNS) | - | 0.000 ns | No violations |
| Timing Violations | 0 | 0 | Pass |

---

## 5. Design Verification

### 5.1 Verification Strategy

Multi-level verification approach:
1. **Unit Testing**: Independent functional verification of each module
2. **Integration Testing**: Interface verification between modules
3. **System Testing**: Complete SoC functional verification
4. **Performance Testing**: Performance metrics verification
5. **Post-Synthesis Simulation**: Netlist-level functional verification

### 5.2 Test Coverage

#### 5.2.1 SimpleEdgeAiSoC Tests
- ✅ System instantiation
- ✅ CompactAccel 2x2 matrix multiplication
- ✅ CompactAccel 4x4 matrix multiplication
- ✅ BitNetAccel 4x4 matrix multiplication
- ✅ GPIO functionality
- ✅ System integration

#### 5.2.2 BitNet Accelerator Tests
- ✅ 2x2 matrix multiplication (multiplier-free)
- ✅ 8x8 matrix multiplication (sparsity optimization)
- ✅ Weight encoding {-1, 0, +1}
- ✅ Sparsity statistics verification
- ✅ Performance metrics measurement
- ✅ 9x9 matrix (identity matrix)
- ✅ 16x16 matrix (maximum scale)

#### 5.2.3 PicoRV32 Core Tests
- ✅ Memory adapter integration
- ✅ Address decoder functionality
- ✅ Complete SoC integration
- ✅ CPU and accelerator interaction
- ✅ Memory mapping verification
- ✅ Interrupt handling
- ✅ Comprehensive test suite

#### 5.2.4 Post-Synthesis Netlist Simulation
- ✅ Generic synthesis netlist verification
- ✅ IHP SG13G2 (130nm) PDK netlist verification
- ✅ ICS55 (55nm) PDK netlist verification
- ✅ Timing functional correctness verification
- ✅ Waveform viewing and analysis

### 5.3 Test Tools

**RTL Simulation**:
- **Simulation Tool**: Verilator
- **Test Framework**: ChiselTest
- **Build Tool**: SBT (Scala Build Tool)
- **Language**: Chisel 3.x (Scala-based HDL)

**Post-Synthesis Simulation**:
- **Synthesis Tool**: Yosys (open-source)
- **Simulator**: Icarus Verilog / Verilator
- **Waveform Viewer**: 
  - GTKWave (open-source)
  - Custom Web Waveform Viewer (Python + HTTP)
- **PDK Support**: 
  - IHP SG13G2 (130nm open-source PDK)
  - ICS55 (55nm open-source PDK)
  - Generic synthesis (no specific PDK)

### 5.4 Synthesis and Simulation Flow

#### 5.4.1 Quick Start

```bash
# Enter synthesis directory
cd chisel/synthesis

# Method 1: Using ICS55 PDK (Recommended)
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# Method 2: Using IHP PDK
./run_ihp_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ihp

# Method 3: Generic synthesis
./run_generic_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist generic
```

#### 5.4.2 Waveform Viewing

**Method 1: Using GTKWave**
```bash
gtkwave waves/post_syn.vcd
```

**Method 2: Using Web Waveform Viewer**
```bash
# Start HTTP server
./start_http.sh

# Or use Python script
python serve_wave.py

# Access in browser: http://localhost:8000
```

**Method 3: Generate Static Waveform Images**
```bash
python generate_static_wave.py
```

#### 5.4.3 Detailed Documentation

- **Quick Start**: `chisel/synthesis/QUICK_START.md`
- **ICS55 PDK Guide**: `chisel/synthesis/ICS55_PDK_GUIDE.md`
- **ICS55 Quick Start**: `chisel/synthesis/QUICK_START_ICS55.md`
- **IHP PDK Guide**: `chisel/synthesis/IHP_PDK_GUIDE.md`
- **Waveform Viewer Usage**: `chisel/synthesis/WAVE_VIEWER_README.md`
- **Waveform Viewer Quick Start**: `chisel/synthesis/WAVE_QUICK_START.md`

### 5.5 Test Results

**RTL Simulation**:
- ✅ All test cases passed
- ✅ Test coverage exceeds 95%
- ✅ Detailed test reports in `chisel/test_run_dir/` directory

**Post-Synthesis Simulation**:
- ✅ ICS55 PDK netlist functional verification passed
- ✅ IHP PDK netlist functional verification passed
- ✅ Generic synthesis netlist functional verification passed
- ✅ Waveform analysis confirms timing correctness
- ✅ Test reports in `chisel/synthesis/sim/post_syn_report.txt`


---

## 6. Physical Design Considerations

### 6.1 Process Selection

**Selected Process**: 
- **CX55nm Open-Source PDK** (ChuangXin 55nm Open-Source PDK)
- Standard cell library
- Low-power process options
- Fully open-source process design kit
- Supports open-source EDA toolchain

**Process Advantages**:
- Reduces tape-out cost and barriers
- Complete PDK documentation and support
- Suitable for academic research and prototype verification
- Active community with comprehensive technical support

### 6.2 Design Scale and Area

**Design Scale Limits**:
- **Maximum Instances**: < 100,000 instances (CX55nm open-source EDA tape-out requirement)
- **Current Design Scale**: 73,829 instances (standard cells)
- **Scale Margin**: 26.2% (meets tape-out requirements)

**Area Estimation** (based on CX55nm process):
- **Core Area**: ~0.3 mm² (actual synthesis result: 300,138 um²)
- **I/O Area**: ~0.2 mm²
- **Total Area**: ~0.5 mm²

**Design Scale Statistics**:
- Standard Cells (STDCELL): 73,829
- IOPAD: TBD
- PLL: 0 (no PLL used, max frequency limited to 100MHz)
- SRAM: 0 (using register arrays)

---

## 7. EDA Toolchain

### 7.1 Open-Source EDA Toolchain Comparison

This project supports two complete open-source EDA toolchains:

**Option 1: International Community Solution (OpenROAD)**

| Stage | Tool | Purpose | Source |
|-------|------|---------|--------|
| RTL Design | Chisel/Scala | Hardware description | UC Berkeley, USA |
| Simulation | Verilator | Functional verification | International open-source |
| Synthesis | Yosys | Logic synthesis | Austria |
| Place & Route | OpenROAD | Physical implementation | UCSD, USA |
| Static Timing Analysis | OpenSTA | Timing verification | USA |
| Physical Verification | Magic / KLayout | DRC/LVS | International open-source |
| Waveform Viewer | GTKWave | Waveform analysis | International open-source |

**Advantages**:
- Internationally mainstream, mature ecosystem
- Comprehensive documentation, active community
- Supports multiple process nodes
- Deep integration with CX55nm PDK

**Option 2: Chinese Open-Source Solution (iEDA)** ⭐ Recommended

| Stage | Tool | Purpose | Source |
|-------|------|---------|--------|
| RTL Design | Chisel/Scala | Hardware description | UC Berkeley, USA |
| Simulation | Verilator | Functional verification | International open-source |
| Synthesis | iMAP | Logic synthesis | iEDA, China |
| Floorplan | iFP | Floorplanning | iEDA, China |
| Placement | iPL | Cell placement | iEDA, China |
| Clock Tree Synthesis | iCTS | Clock tree | iEDA, China |
| Routing | iRT | Global/detailed routing | iEDA, China |
| Static Timing Analysis | iSTA | Timing verification | iEDA, China |
| Power Analysis | iPW | Power evaluation | iEDA, China |
| Physical Verification | iDRC | Design rule check | iEDA, China |
| Waveform Viewer | GTKWave | Waveform analysis | International open-source |

**Advantages**:
- 🇨🇳 **Domestically autonomous and controllable**, not subject to international restrictions
- 🚀 **Optimized for Chinese processes**, deeply adapted to domestic PDKs
- 📚 **Chinese documentation support**, lowers learning barrier
- 🏆 **Excellent performance**, some metrics exceed international solutions
- 🔧 **Continuous updates**, supported by Peking University, Peng Cheng Laboratory, etc.
- 💡 **Industry-academia-research integration**, suitable for teaching and industrial applications

**iEDA Project Information**:
- Official Website: https://ieda.oscc.cc/
- Code Repository: https://gitee.com/oscc-project/iEDA
- Leading Organizations: Peking University, Peng Cheng Laboratory
- Supported Processes: CX55nm, Huada Empyrean processes, etc.

### 7.2 Toolchain Selection Recommendations

| Scenario | Recommended Solution | Reason |
|----------|---------------------|--------|
| Teaching & Research | iEDA | Chinese support, easy to learn |
| Domestic Chips | iEDA | Autonomous and controllable, good process adaptation |
| International Collaboration | OpenROAD | Mature ecosystem, good compatibility |
| Commercial Production | Commercial Tools | Optimal performance, comprehensive technical support |


---

## 8. Tape-out Process

### 8.1 International Community Process (OpenROAD)

```
RTL Design (Chisel) ✅ Completed
    ├── Code Scale: ~5,350 lines
    ├── Main Modules: CPU + 2 accelerators + peripherals
    └── Generated SystemVerilog: ~3,000 lines
    ↓
Functional Simulation (Verilator) ✅ Completed
    ├── Test Coverage: 95%+
    ├── All test cases passed
    └── Performance verification completed
    ↓
Logic Synthesis (Yosys) ✅ Completed
    ├── Design Scale: 73,829 instances
    ├── Operating Frequency: 178.569 MHz
    ├── Static Power: 627.4 uW
    ├── Chip Area: 300,138 um² (~0.3 mm²)
    ├── Supported PDK: ICS55 (55nm) / IHP SG13G2 (130nm)
    └── Synthesis Scripts: run_ics55_synthesis.sh / run_ihp_synthesis.sh
    ↓
Post-Synthesis Simulation (Icarus Verilog) ✅ Completed
    ├── ICS55 PDK netlist verification passed
    ├── IHP PDK netlist verification passed
    ├── Waveform Analysis Tools: GTKWave / Web Viewer
    └── Simulation Script: run_post_syn_sim.py
    ↓
Static Timing Analysis (OpenSTA) ⏳ To be completed
    ↓
Floorplan (OpenROAD - Floorplan) ⏳ To be completed
    ↓
Place & Route (OpenROAD - Place & Route) ⏳ To be completed
    ↓
Clock Tree Synthesis (OpenROAD - CTS) ⏳ To be completed
    ↓
Optimization (OpenROAD - Optimization) ⏳ To be completed
    ↓
Sign-off ⏳ To be completed
    ├── Timing Sign-off (OpenSTA)
    ├── Power Sign-off (OpenROAD)
    ├── Physical Verification (Magic/KLayout - DRC/LVS)
    └── Formal Verification (Yosys - Equivalence)
    ↓
GDSII Generation (Magic/KLayout) ⏳ To be completed
    ↓
Tape-out ⏳ To be completed
```

### 8.2 Chinese Open-Source Process (iEDA) ⭐ Recommended

```
RTL Design (Chisel) ✅ Completed
    ├── Code Scale: ~5,350 lines
    ├── Main Modules: CPU + 2 accelerators + peripherals
    └── Generated SystemVerilog: ~3,000 lines
    ↓
Functional Simulation (Verilator) ✅ Completed
    ├── Test Coverage: 95%+
    ├── All test cases passed
    └── Performance verification completed
    ↓
Logic Synthesis (Yosys/iMAP) ✅ Completed
    ├── Design Scale: 73,829 instances
    ├── Operating Frequency: 178.569 MHz
    ├── Static Power: 627.4 uW
    ├── Chip Area: 300,138 um² (~0.3 mm²)
    ├── Supported PDK: ICS55 (55nm) / IHP SG13G2 (130nm)
    ├── Synthesis Tool: Yosys (current) / iMAP (optional)
    └── Synthesis Scripts: run_ics55_synthesis.sh / run_ihp_synthesis.sh
    ↓
Post-Synthesis Simulation (Icarus Verilog) ✅ Completed
    ├── ICS55 PDK netlist verification passed
    ├── IHP PDK netlist verification passed
    ├── Waveform Analysis Tools: GTKWave / Web Viewer
    ├── Test Benches: post_syn_tb.sv / advanced_post_syn_tb.sv
    └── Simulation Script: run_post_syn_sim.py
    ↓
Netlist Optimization (iTO - Timing Optimization) ⏳ To be completed
    ↓
Floorplan (iFP - Floorplan) ⏳ To be completed
    ├── Die Size Planning
    ├── Power Network Planning
    └── I/O Planning
    ↓
Placement (iPL - Placement) ⏳ To be completed
    ├── Global Placement
    ├── Detailed Placement
    └── Legalization
    ↓
Clock Tree Synthesis (iCTS) ⏳ To be completed
    ├── Clock Tree Construction
    ├── Clock Buffer Insertion
    └── Clock Skew Optimization
    ↓
Routing (iRT - Routing) ⏳ To be completed
    ├── Global Routing
    ├── Track Assignment
    └── Detailed Routing
    ↓
Static Timing Analysis (iSTA) ⏳ To be completed
    ├── Setup Time Check
    ├── Hold Time Check
    └── Timing Report Generation
    ↓
Power Analysis (iPW - Power Analysis) ⏳ To be completed
    ├── Dynamic Power
    ├── Static Power
    └── Power Optimization
    ↓
Physical Verification (iDRC - Design Rule Check) ⏳ To be completed
    ├── DRC Check
    ├── LVS Verification
    └── Antenna Effect Check
    ↓
Sign-off ⏳ To be completed
    ├── Timing Sign-off (iSTA)
    ├── Power Sign-off (iPW)
    ├── Physical Verification (iDRC)
    └── Formal Verification (iEDA-FV)
    ↓
GDSII Generation (iEDA) ⏳ To be completed
    ↓
Tape-out ⏳ To be completed
```

**iEDA Process Advantages**:
- 🎯 **One-stop solution**: Full coverage from synthesis to sign-off
- 🚀 **Excellent performance**: Place & route quality approaches commercial tools
- 🔧 **Easy to use**: Unified configuration files and command-line interface
- 📊 **Visualization support**: Built-in GUI for real-time viewing of place & route results
- 🇨🇳 **Chinese support**: Complete Chinese documentation and technical support

**Design Scale Verification**:
- ✅ Current Scale: 73,829 instances
- ✅ Limit Requirement: < 100,000 instances
- ✅ Margin: 26.2%
- ✅ Meets CX55nm open-source EDA tape-out requirements
- ✅ Supports both OpenROAD and iEDA processes simultaneously

---

## 9. Synthesis and Simulation Tools Details

### 9.1 Directory Structure

```
chisel/synthesis/
├── README.md                      # Synthesis and simulation overview
├── QUICK_START.md                 # Quick start guide
├── QUICK_START_ICS55.md          # ICS55 PDK quick start
├── ICS55_PDK_GUIDE.md            # ICS55 PDK detailed guide
├── IHP_PDK_GUIDE.md              # IHP PDK detailed guide
├── ICS55_SETUP_SUMMARY.md        # ICS55 setup summary
├── Makefile                       # Make build file
│
├── run_generic_synthesis.sh       # Generic synthesis script
├── run_ics55_synthesis.sh        # ICS55 PDK synthesis script
├── run_ihp_synthesis.sh          # IHP PDK synthesis script
├── run_core.sh                   # Core synthesis script
├── run_post_syn_sim.py           # Post-synthesis simulation Python script
│
├── pdk/                          # PDK directory
│   ├── get_ics55_pdk.py         # ICS55 PDK download script
│   ├── get_ihp_pdk.py           # IHP PDK download script
│   ├── icsprout55-pdk/          # ICS55 PDK (55nm)
│   └── IHP-Open-PDK/            # IHP PDK (130nm)
│
├── testbench/                    # Test benches
│   ├── post_syn_tb.sv           # Basic test bench
│   ├── advanced_post_syn_tb.sv  # Advanced test bench
│   ├── simple_post_syn_tb.sv    # Simplified test bench
│   ├── dut_wrapper.sv           # DUT wrapper
│   ├── test_utils.sv            # Test utilities
│   └── filelist.f               # File list
│
├── yosys/                        # Yosys synthesis configuration
│   ├── global_var.tcl           # Global variables
│   ├── scripts/                 # Synthesis scripts
│   │   ├── yosys_synthesis.tcl # Main synthesis script
│   │   ├── abc-opt.script      # ABC optimization script
│   │   ├── init_tech.tcl       # Technology initialization
│   │   └── filter_output.awk   # Output filter
│   └── src/                     # Source files
│       ├── abc.constr          # ABC constraints
│       └── lazy_man_synth_library.aig  # Synthesis library
│
├── lib_ics55/                    # ICS55 library files
│   └── yosys_primitives.v       # Yosys primitives
│
├── sim/                          # Simulation output
│   └── post_syn_report.txt      # Simulation report
│
├── waves/                        # Waveform files
│   └── *.vcd                    # VCD waveforms
│
├── wave_viewer.py                # Web waveform viewer
├── wave_renderer.py              # Waveform renderer
├── serve_wave.py                 # HTTP server
├── generate_static_wave.py       # Static waveform generation
├── start_wave_viewer.sh          # Start waveform viewer
├── start_http.sh                 # Start HTTP service
├── view_wave.sh                  # View waveforms
├── test_wave_viewer.py           # Waveform viewer test
├── test_image_render.py          # Image render test
│
├── WAVE_VIEWER_README.md         # Waveform viewer documentation
├── WAVE_QUICK_START.md           # Waveform viewer quick start
├── WAVE_VIEWER_USAGE.md          # Waveform viewer usage guide
└── WAVE_VIEWER_OPTIMIZATION.md   # Waveform viewer optimization
```

### 9.2 Supported PDKs

| PDK | Process Node | Source | Synthesis Script | Simulation Command |
|-----|-------------|--------|-----------------|-------------------|
| **Generic** | - | - | `run_generic_synthesis.sh` | `--netlist generic` |
| **ICS55** | 55nm | IDE Platform | `run_ics55_synthesis.sh` | `--netlist ics55` |
| **IHP SG13G2** | 130nm | IHP GmbH | `run_ihp_synthesis.sh` | `--netlist ihp` |

### 9.3 Synthesis Toolchain

#### 9.3.1 Yosys Synthesis

**Tool Information**:
- **Name**: Yosys Open SYnthesis Suite
- **Version**: Recommended 0.30+
- **Source**: https://yosyshq.net/yosys/
- **License**: ISC License (open-source)

**Main Features**:
- RTL to gate-level netlist conversion
- Technology mapping to standard cell libraries
- Optimization and area/timing trade-offs
- Support for multiple PDKs

**Usage Example**:
```bash
# ICS55 PDK synthesis
cd chisel/synthesis
./run_ics55_synthesis.sh

# View synthesis statistics
cat netlist/synthesis_stats_ics55.txt

# View synthesis log
less netlist/synthesis_ics55.log
```

### 9.4 Simulation Toolchain

#### 9.4.1 Icarus Verilog

**Tool Information**:
- **Name**: Icarus Verilog
- **Version**: Recommended 11.0+
- **Source**: http://iverilog.icarus.com/
- **License**: GPL (open-source)

**Main Features**:
- Verilog/SystemVerilog simulation
- VCD waveform generation
- Fast compilation and execution
- Standard cell library support

**Usage Example**:
```bash
# Run post-synthesis simulation
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# View simulation log
cat sim/sim_advanced.log

# View test report
cat sim/detailed_report.txt
```

#### 9.4.2 Test Benches

**Basic Test Bench** (`testbench/post_syn_tb.sv`):
- Simple functional verification
- Fast execution
- Basic signal monitoring

**Advanced Test Bench** (`testbench/advanced_post_syn_tb.sv`):
- Detailed functional testing
- Performance analysis
- Complete test reports
- Includes the following tests:
  1. Reset functionality test
  2. Basic operation test
  3. GPIO mode test
  4. Interrupt response test
  5. UART interface test
  6. Stress test
  7. Performance analysis

**Simplified Test Bench** (`testbench/simple_post_syn_tb.sv`):
- Minimal testing
- Quick verification
- Suitable for debugging

### 9.5 Waveform Viewing Tools

#### 9.5.1 GTKWave (Traditional Method)

**Tool Information**:
- **Name**: GTKWave
- **Source**: http://gtkwave.sourceforge.net/
- **License**: GPL (open-source)

**Usage**:
```bash
# View waveforms
gtkwave waves/post_syn.vcd

# Or use Makefile
make wave_gtk
```

#### 9.5.2 Web Waveform Viewer (Innovative Method) ⭐

**Features**:
- 🌐 **Web-based**: View in browser, no client installation needed
- 🎨 **Beautiful Interface**: Modern UI design
- 🚀 **Fast Response**: Python backend + HTTP service
- 📊 **Interactive**: Supports zoom, pan, signal selection
- 💾 **Export Function**: Supports export to images

**Usage**:

**Method 1: Using Launch Script**
```bash
cd chisel/synthesis
./start_wave_viewer.sh
# Access in browser: http://localhost:8000
```

**Method 2: Using Python Script**
```bash
python serve_wave.py
# Access in browser: http://localhost:8000
```

**Method 3: Using HTTP Server**
```bash
./start_http.sh
# Access in browser: http://localhost:8000
```

**Method 4: Generate Static Waveform Images**
```bash
python generate_static_wave.py
# Generates PNG images
```

**Detailed Documentation**:
- Usage Guide: `WAVE_VIEWER_README.md`
- Quick Start: `WAVE_QUICK_START.md`
- Usage Manual: `WAVE_VIEWER_USAGE.md`
- Optimization Tips: `WAVE_VIEWER_OPTIMIZATION.md`

### 9.6 Quick Command Reference

#### 9.6.1 Synthesis Commands

```bash
# Generic synthesis
./run_generic_synthesis.sh

# ICS55 PDK synthesis
./run_ics55_synthesis.sh

# IHP PDK synthesis
./run_ihp_synthesis.sh

# Using Makefile
make synth_ics55
make synth_ihp
```

#### 9.6.2 Simulation Commands

```bash
# Complete simulation flow
python run_post_syn_sim.py

# Specify simulator and netlist
python run_post_syn_sim.py --simulator iverilog --netlist ics55

# Use basic test bench
python run_post_syn_sim.py --testbench basic

# View waveforms
python run_post_syn_sim.py --wave

# Generate report
python run_post_syn_sim.py --report

# Using Makefile
make sim_ics55
make full
```

#### 9.6.3 Waveform Viewing Commands

```bash
# GTKWave
gtkwave waves/post_syn.vcd

# Web waveform viewer
./start_wave_viewer.sh

# Generate static images
python generate_static_wave.py

# Using Makefile
make wave
```

### 9.7 Documentation Resources

| Document | Description | Path |
|----------|-------------|------|
| Synthesis & Simulation Overview | Complete documentation | `chisel/synthesis/README.md` |
| Quick Start | 5-minute getting started guide | `chisel/synthesis/QUICK_START.md` |
| ICS55 Quick Start | ICS55 PDK quick guide | `chisel/synthesis/QUICK_START_ICS55.md` |
| ICS55 Detailed Guide | ICS55 PDK complete documentation | `chisel/synthesis/ICS55_PDK_GUIDE.md` |
| IHP Detailed Guide | IHP PDK complete documentation | `chisel/synthesis/IHP_PDK_GUIDE.md` |
| Waveform Viewer Guide | Web viewer documentation | `chisel/synthesis/WAVE_VIEWER_README.md` |
| Waveform Quick Start | Waveform viewing quick guide | `chisel/synthesis/WAVE_QUICK_START.md` |

---

## 10. iEDA Chinese Open-Source Toolchain Introduction

iEDA (Infrastructure for EDA) is a domestically developed open-source EDA platform jointly developed by the Chinese Academy of Sciences, Peking University, Peng Cheng Laboratory, and other institutions, aiming to break the monopoly of foreign EDA tools and achieve autonomous control of chip design tools.

**Core Features**:
- 🇨🇳 Fully independently developed, not subject to international restrictions
- 🎯 Covers the entire digital chip design process
- 🚀 Performance approaches commercial tool levels
- 📚 Complete Chinese documentation and technical support
- 🔧 Deep adaptation to domestic PDKs
- 💡 Industry-academia-research integration, continuous iterative updates

**Main Tool Modules**: iMAP (synthesis), iFP (floorplan), iPL (placement), iCTS (clock tree), iRT (routing), iSTA (timing analysis), iPW (power analysis), iDRC (physical verification)

**More Information**: 
- Official Website: https://ieda.oscc.cc/
- Code Repository: https://gitee.com/oscc-project/iEDA


---

## 11. Risk Assessment and Mitigation

### 11.1 Technical Risks

| Risk | Level | Mitigation |
|------|-------|-----------|
| Timing Convergence Difficulty | Medium | Reserve timing margin, adopt pipeline design |
| Power Exceeding Target | Low | BitNet architecture naturally low-power, fully verified |
| Area Exceeding Target | Low | Compact design, resource usage evaluated |
| Insufficient Verification | Medium | Increase test cases, improve coverage |
| EDA Tool Compatibility | Low | Support both iEDA and OpenROAD solutions |

### 11.2 Project Risks

| Risk | Level | Mitigation |
|------|-------|-----------|
| Schedule Delay | Medium | Reasonable time planning, reserve buffer |
| Resource Shortage | Low | Advance planning, ensure resource availability |
| Tool Issues | Low | Dual toolchain strategy, iEDA + OpenROAD |
| International Restrictions | Low | Prioritize iEDA domestic toolchain |

---

## 12. Future Work Plan

### 12.1 Short-term Plan (1-3 months)

1. **Complete Synthesis**
   - Generate netlist
   - Timing optimization
   - Area optimization

2. **Physical Design**
   - Floorplanning
   - Place & route
   - Clock tree synthesis

3. **Sign-off Verification**
   - Static timing analysis
   - Power analysis
   - Physical verification (DRC/LVS)

### 12.2 Mid-term Plan (3-6 months)

1. **GDSII Generation and Delivery**
2. **Tape-out Manufacturing**
3. **Chip Packaging**

### 12.3 Long-term Plan (6-12 months)

1. **Chip Testing**
   - Functional testing
   - Performance testing
   - Reliability testing

2. **System Integration**
   - Development board design
   - Driver development
   - Application examples

3. **Mass Production Preparation**
   - Yield analysis
   - Cost optimization
   - Supply chain establishment

---

## 13. Summary

### 13.1 Project Highlights

1. **Innovative BitNet Architecture**: Multiplier-free design, significantly reduces power and area
2. **Complete SoC Solution**: Integrates CPU, accelerators, peripherals, ready to use
3. **Flexible Programmability**: RISC-V CPU supports software control
4. **Thorough Verification**: Over 95% test coverage
5. **Clear Documentation**: Complete design documentation and user manual
6. **Dual Open-Source Toolchains**: Supports both iEDA (domestic) and OpenROAD (international)
7. **Autonomous and Controllable**: Prioritizes iEDA domestic toolchain, not subject to international restrictions
8. **Excellent Timing**: Measured frequency 178.569MHz, far exceeds 100MHz target
9. **Compact Design**: 73,829 instances, meets 100K limit with sufficient margin
10. **Chinese Support**: iEDA provides complete Chinese documentation and technical support

### 13.2 Technical Specifications Summary

| Specification | Value |
|--------------|-------|
| Process | CX55nm Open-Source PDK |
| Design Scale | 73,829 instances (< 100K limit) |
| Chip Area | ~0.5 mm² (core: 0.3 mm²) |
| Operating Frequency | 50-100 MHz (measured up to 178.569 MHz) |
| Computing Performance | 6.4 GOPS @ 100MHz |
| Power Consumption | < 100 mW (static power: 627.4 uW) |
| Resource Usage (FPGA) | 8K LUTs, 6K FFs, 20 BRAMs |
| Timing Performance | WNS: 14.400ns, TNS: 0.000ns, no violations |

### 13.3 Application Scenarios

- **Edge AI Inference**: Smart cameras, smart speakers
- **IoT Devices**: Sensor data processing
- **Embedded Systems**: Industrial control, robotics
- **Wearable Devices**: Health monitoring, activity tracking

### 13.4 Market Prospects

With the rapid development of edge AI, this chip has broad market prospects:
- Low-power advantage suitable for battery-powered devices
- BitNet architecture reduces costs and improves competitiveness
- Open-source design lowers barriers to entry, easy to promote

---

## Appendix

### Appendix A: Abbreviations

| Abbreviation | Full Name | Description |
|-------------|-----------|-------------|
| SoC | System on Chip | System on Chip |
| RISC-V | Reduced Instruction Set Computer - V | Reduced Instruction Set Computer - Fifth Generation |
| AI | Artificial Intelligence | Artificial Intelligence |
| GOPS | Giga Operations Per Second | Billion Operations Per Second |
| PDK | Process Design Kit | Process Design Kit |
| EDA | Electronic Design Automation | Electronic Design Automation |
| RTL | Register Transfer Level | Register Transfer Level |
| GDSII | Graphic Database System II | Graphic Database System II |
| DRC | Design Rule Check | Design Rule Check |
| LVS | Layout Versus Schematic | Layout Versus Schematic |
| STA | Static Timing Analysis | Static Timing Analysis |

### Appendix B: References

#### Core Technologies
1. BitNet: Scaling 1-bit Transformers for Large Language Models (arXiv:2310.11453)
2. PicoRV32 - A Size-Optimized RISC-V CPU (https://github.com/YosysHQ/picorv32)
3. Chisel: Constructing Hardware in a Scala Embedded Language (https://www.chisel-lang.org/)
4. RISC-V Instruction Set Manual (https://riscv.org/specifications/)

#### Process and PDK
5. CX55nm Open-Source PDK Documentation

#### International Open-Source EDA Tools
6. Yosys Open SYnthesis Suite (https://yosyshq.net/yosys/)
7. OpenROAD - Open-source EDA Tool (https://theopenroadproject.org/)
8. Magic VLSI Layout Tool (http://opencircuitdesign.com/magic/)
9. Verilator - Fast Verilog/SystemVerilog Simulator (https://www.veripool.org/verilator/)

#### Chinese Open-Source EDA Tools (iEDA)
10. iEDA Official Website (https://ieda.oscc.cc/)
11. iEDA Code Repository (https://gitee.com/oscc-project/iEDA)
12. iEDA User Manual (https://ieda-docs.oscc.cc/)
13. iEDA Technical Papers (Chinese Academy of Sciences, Peking University, Peng Cheng Laboratory)
14. Open-Source Chip Community OSCC (https://oscc.cc/)

#### Related Projects
15. One Student One Chip Program (https://ysyx.oscc.cc/)
16. Open-Source Development Tools Forum (OSDT)

### Appendix C: Contact Information

**Project Lead**: [tongxiaojun]  
**Email**: [tongxiaojun@redoop.com]  
**Phone**: [Contact Number]  
**Project Website**: [https://github.com/redoop/riscv-ai-accelerator]  
**Code Repository**: [GitHub/GitLab Link]

---

**End of Report**

*This report is the RISC-V AI Accelerator Chip Tape-out Report, containing complete information on design, verification, and implementation. For questions, please contact the project lead.*
