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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  SimpleEdgeAiSoC                        │
│  ┌──────────────┐      ┌──────────────────────────┐    │
│  │  PicoRV32    │◄────►│   Address Decoder        │    │
│  │   (RV32I)    │      │   (Memory Map)           │    │
│  └──────────────┘      └──────┬───────────────────┘    │
│                                │                        │
│                                ├──► CompactAccel (8x8)  │
│                                ├──► BitNetAccel (16x16) │
│                                ├──► UART                │
│                                └──► GPIO                │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | Performance |
|-----------|-------------|-------------|
| **PicoRV32** | RISC-V RV32I CPU | 50-100 MHz |
| **CompactAccel** | 8x8 matrix accelerator | 1.6 GOPS @ 100MHz |
| **BitNetAccel** | 16x16 multiplier-free accelerator | 4.8 GOPS @ 100MHz |
| **UART** | Serial communication | Configurable baud rate |
| **GPIO** | 32-bit general I/O | Bidirectional |

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
# Install Scala Build Tool (SBT)
brew install sbt  # macOS
# or apt-get install sbt  # Linux

# Install Verilator (for simulation)
brew install verilator  # macOS
# or apt-get install verilator  # Linux
```

### Build and Test

```bash
# Clone repository
git clone https://github.com/redoop/riscv-ai-accelerator.git
cd riscv-ai-accelerator/chisel

# Run all tests
make test

# Generate Verilog
make verilog

# Run synthesis and post-synthesis simulation
cd synthesis
./run_ics55_synthesis.sh
python run_post_syn_sim.py --simulator iverilog --netlist ics55
```

## 📊 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Design Scale** | 73,829 instances | ✅ < 100K limit |
| **Core Area** | 300,138 um² (~0.3 mm²) | ✅ Compact |
| **Operating Frequency** | 178.569 MHz (measured) | ✅ Exceeds 100MHz target |
| **Peak Performance** | 6.4 GOPS @ 100MHz | ✅ Target met |
| **Static Power** | 627.4 uW | ✅ Ultra-low |
| **Timing** | WNS: 14.4ns, TNS: 0ns | ✅ No violations |
| **Test Coverage** | > 95% | ✅ Comprehensive |

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

### Synthesis & Simulation

| Document | Description | Link |
|----------|-------------|------|
| **🔧 Synthesis Guide** | Post-synthesis simulation | [chisel/synthesis/README.md](chisel/synthesis/README.md) |
| **⚡ Quick Start** | 5-minute getting started | [chisel/synthesis/QUICK_START.md](chisel/synthesis/QUICK_START.md) |
| **🔬 ICS55 PDK Guide** | 55nm PDK detailed guide | [chisel/synthesis/ICS55_PDK_GUIDE.md](chisel/synthesis/ICS55_PDK_GUIDE.md) |
| **🔬 IHP PDK Guide** | 130nm PDK detailed guide | [chisel/synthesis/IHP_PDK_GUIDE.md](chisel/synthesis/IHP_PDK_GUIDE.md) |

### Waveform Viewing

| Document | Description | Link |
|----------|-------------|------|
| **🌊 Wave Viewer** | Web-based waveform viewer | [chisel/synthesis/waves/README.md](chisel/synthesis/waves/README.md) |
| **📊 Wave Quick Start** | Waveform viewing guide | [chisel/synthesis/waves/WAVE_QUICK_START.md](chisel/synthesis/waves/WAVE_QUICK_START.md) |
| **🎨 Wave Viewer Usage** | Detailed usage manual | [chisel/synthesis/waves/WAVE_VIEWER_USAGE.md](chisel/synthesis/waves/WAVE_VIEWER_USAGE.md) |

### FPGA Verification

| Document | Description | Link |
|----------|-------------|------|
| **🔌 FPGA Guide** | AWS F1 FPGA verification | [chisel/synthesis/fpga/README.md](chisel/synthesis/fpga/README.md) |
| **☁️ AWS Setup** | AWS environment setup | [chisel/synthesis/fpga/docs/SETUP_GUIDE.md](chisel/synthesis/fpga/docs/SETUP_GUIDE.md) |
| **📋 AWS Plan** | Complete AWS verification plan | [chisel/synthesis/fpga/docs/AWS_FPGA_PLAN.md](chisel/synthesis/fpga/docs/AWS_FPGA_PLAN.md) |

## 🎓 Project Structure

```
riscv-ai-accelerator/
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── README.md                      # Tape-out report (English)
│   ├── RISC-V_AI加速器芯片流片说明报告.md  # Chinese report
│   └── image/                         # Images and diagrams
├── chisel/                            # Chisel RTL design
│   ├── README.md                      # Chisel design guide
│   ├── src/                           # Source code
│   │   └── main/scala/edgeai/        # Main design modules
│   ├── synthesis/                     # Synthesis and simulation
│   │   ├── README.md                  # Synthesis guide
│   │   ├── run_ics55_synthesis.sh    # ICS55 synthesis
│   │   ├── run_ihp_synthesis.sh      # IHP synthesis
│   │   ├── run_post_syn_sim.py       # Post-syn simulation
│   │   ├── waves/                     # Waveform tools
│   │   │   ├── README.md             # Wave viewer guide
│   │   │   ├── wave_viewer.py        # Web viewer
│   │   │   └── view_wave.sh          # Quick view script
│   │   └── fpga/                      # FPGA verification
│   │       ├── README.md             # FPGA guide
│   │       └── docs/                 # FPGA documentation
│   └── test/                          # Test benches
└── LICENSE                            # Apache 2.0 License
```

## 🧪 Testing

### RTL Simulation

```bash
cd chisel

# Run all tests
make test

# Run specific test
sbt "testOnly edgeai.SimpleEdgeAiSoCTest"
sbt "testOnly edgeai.BitNetAccelTest"
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

### ✅ Completed

- [x] RTL design (Chisel)
- [x] Functional verification (95%+ coverage)
- [x] Logic synthesis (Yosys)
- [x] Post-synthesis simulation
- [x] Waveform viewing tools
- [x] Documentation

### 🚧 In Progress

- [ ] Static timing analysis (OpenSTA/iSTA)
- [ ] Floorplanning (OpenROAD/iFP)
- [ ] Place & route (OpenROAD/iPL/iRT)
- [ ] Physical verification (DRC/LVS)

### 📅 Planned

- [ ] GDSII generation
- [ ] Tape-out manufacturing
- [ ] Chip testing
- [ ] Development board

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
