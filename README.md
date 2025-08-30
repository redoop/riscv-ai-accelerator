# RISC-V AI Accelerator Chip

A high-performance AI accelerator chip based on the RISC-V instruction set architecture, designed for machine learning and deep learning workloads with comprehensive PyTorch integration and macOS simulator support.

## Overview

This project implements a complete RISC-V-based AI accelerator chip featuring:

- **RISC-V cores** with RV64IMAFDV instruction set support
- **Tensor Processing Units (TPUs)** with systolic arrays for matrix operations
- **Vector Processing Units (VPUs)** for SIMD computations
- **Custom AI instruction extensions** for neural network operations
- **Advanced memory hierarchy** with multi-level caches
- **Network-on-Chip (NoC)** for efficient inter-core communication
- **Comprehensive power management** with DVFS and thermal control
- **PyTorch backend integration** for seamless AI framework support
- **macOS simulator** for development and testing on Apple Silicon

## Architecture Highlights

### Processing Elements
- **RISC-V Cores**: 64-bit cores with 6-stage pipeline, 32KB L1 caches
- **TPU**: 64x64 MAC array, 256 TOPS (INT8), 64 TFLOPS (FP16)
- **VPU**: 16-lane vector processing, RVV extension support
- **Memory**: 2MB L2 + 8MB L3 cache, 1MB scratchpad, HBM2E interface

### AI Instructions
- Matrix multiplication (`ai.matmul`)
- 2D convolution (`ai.conv2d`)
- Activation functions (`ai.relu`, `ai.sigmoid`, `ai.tanh`)
- Pooling operations (`ai.maxpool`, `ai.avgpool`)
- Batch normalization (`ai.batchnorm`)

### Connectivity
- PCIe 4.0 x16 for high-speed I/O
- Gigabit Ethernet for networking
- USB 3.1 for device connectivity
- SATA/NVMe for storage interfaces

## Project Structure

```
├── rtl/                    # RTL source code
│   ├── core/              # RISC-V processor cores
│   ├── accelerators/      # TPU and VPU implementations
│   ├── memory/            # Cache controllers and memory subsystem
│   ├── noc/               # Network-on-Chip implementation
│   ├── peripherals/       # I/O controllers (PCIe, Ethernet, etc.)
│   ├── power/             # Power management unit
│   ├── interfaces/        # SystemVerilog interfaces
│   ├── config/            # Configuration parameters
│   └── top/               # Top-level chip integration
├── verification/          # Verification environment
│   ├── testbench/         # SystemVerilog testbenches
│   ├── unit_tests/        # Component-level tests
│   └── integration_tests/ # System-level tests
├── software/              # Software stack
│   ├── drivers/           # Device drivers
│   ├── compiler/          # Compiler intrinsics and support
│   ├── runtime/           # AI runtime libraries
│   └── tests/             # Software test suite
├── docs/                  # Documentation
├── scripts/               # Build and analysis scripts
└── Makefile              # Build system
```

## Getting Started

### Prerequisites

- **RTL Simulation**: Verilator, Icarus Verilog, or ModelSim
- **Synthesis**: Yosys or commercial synthesis tools
- **Software**: RISC-V GCC toolchain (optional for cross-compilation)
- **Python 3**: Required for PyTorch integration and testing
- **PyTorch**: For AI framework integration and testing
- **NumPy**: For numerical computations
- **GTKWave**: For waveform viewing (optional)

### Building the Project

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd riscv-ai-accelerator
   ```

2. **Install Python dependencies**:
   ```bash
   make install-deps
   ```

3. **Check system compatibility**:
   ```bash
   make info
   make check-hardware-quick
   ```

4. **Run RTL simulation**:
   ```bash
   make sim
   ```

5. **Run PyTorch tests**:
   ```bash
   make test-simple
   ```

### Quick Start Commands

```bash
# Install dependencies and check system
make install-deps
make info

# Build and run RTL simulation
make sim

# Run with waveform viewer
make sim-gui

# Run PyTorch integration tests
make test-simple          # Simple CPU benchmark
make test-comprehensive   # Full AI accelerator tests
make test-quick          # Quick validation

# macOS-specific commands
make install-simulator   # Install macOS simulator
make demo-simulator     # Run simulator demo
make test-macos         # Complete macOS test suite

# Hardware and performance
make check-hardware     # Detailed hardware check
make benchmark         # Performance benchmarking

# Development tools
make synth             # Synthesize the design
make software          # Build software components
make docs              # View documentation
make clean             # Clean build artifacts
```

## Development Workflow

### RTL Development
1. Modify RTL sources in `rtl/` directory
2. Run `make lint-rtl` to check for issues
3. Run `make sim` to verify functionality
4. Run `make test-rtl` for comprehensive testing

### PyTorch Integration Development
1. Install dependencies: `make install-deps`
2. Test basic functionality: `make test-simple`
3. Run comprehensive tests: `make test-comprehensive`
4. Check performance: `make benchmark`
5. View results in `test_results/` and `logs/`

### macOS Development
1. Install simulator: `make install-simulator`
2. Test simulator: `make test-simulator`
3. Run demo: `make demo-simulator`
4. Full macOS testing: `make test-macos`

### Software Development
1. Modify software sources in `software/` directory
2. Run `make lint-sw` to check code quality
3. Run `make software` to compile
4. Run `make test-sw` to execute tests

### Adding New Features
1. Update configuration in `rtl/config/chip_config_pkg.sv`
2. Implement RTL modules with proper interfaces
3. Add corresponding software support
4. Create comprehensive test cases
5. Update documentation

## AI Instruction Usage

### C/C++ Intrinsics Example

```c
#include "riscv_ai_intrinsics.h"

// Matrix multiplication using TPU
float A[64][64], B[64][64], C[64][64];
__builtin_riscv_ai_matmul_f32(&A[0][0], &B[0][0], &C[0][0], 64, 64, 64);

// ReLU activation
float input[1024], output[1024];
__builtin_riscv_ai_relu_f32(input, output, 1024);

// 2D Convolution
float feature_map[224][224][3], kernel[3][3][3], result[222][222][1];
__builtin_riscv_ai_conv2d_f32(&feature_map[0][0][0], &kernel[0][0][0], 
                              &result[0][0][0], 224, 224, 3, 222, 222, 1, 
                              3, 3, 1, 1, 0, 0);
```

### Driver API Example

```c
#include "ai_accel_driver.h"

// Initialize AI accelerator
ai_driver_init();

// Create tensor descriptors
ai_tensor_t input_tensor, weight_tensor, output_tensor;
ai_create_tensor(&input_tensor, AI_DTYPE_FP32, 2, (uint32_t[]){64, 64}, input_data);
ai_create_tensor(&weight_tensor, AI_DTYPE_FP32, 2, (uint32_t[]){64, 64}, weight_data);
ai_create_tensor(&output_tensor, AI_DTYPE_FP32, 2, (uint32_t[]){64, 64}, output_data);

// Submit matrix multiplication task
ai_task_t task = {
    .task_id = 1,
    .operation = AI_OP_MATMUL,
    .accel_type = AI_ACCEL_TPU,
    .accel_id = 0,
    .input_tensors = {input_tensor, weight_tensor},
    .output_tensors = {output_tensor},
    .num_inputs = 2,
    .num_outputs = 1
};

ai_submit_task(&task);
ai_wait_task(1, 1000);  // Wait up to 1 second
```

## Performance Characteristics

### Peak Performance
- **Scalar**: 8 GOPS (4 cores × 2 GHz)
- **Vector**: 64 GFLOPS (2 VPUs × 16 lanes × 2 GHz)
- **AI**: 512 TOPS INT8, 128 TFLOPS FP16
- **Memory**: 1.6 TB/s bandwidth (HBM2E)

### Power Efficiency
- **Process**: 7nm technology
- **Power**: ~150W typical, 200W peak
- **Efficiency**: >3 TOPS/W (INT8)

## Testing and Verification

### Test Categories
- **RTL Tests**: Hardware simulation and verification
- **PyTorch Tests**: AI framework integration testing
- **Unit Tests**: Individual component verification
- **Integration Tests**: Multi-component system tests
- **Performance Tests**: Benchmark and performance validation
- **Simulator Tests**: macOS simulator functionality

### Running Tests
```bash
# Complete test suite
make test                # All tests (RTL + SW + PyTorch)
make test-all           # All PyTorch tests

# PyTorch-specific tests
make test-simple        # Simple CPU benchmark tests
make test-comprehensive # Full AI accelerator tests
make test-quick         # Quick validation tests
make benchmark          # Performance benchmarking

# Platform-specific tests
make test-macos         # macOS complete test suite
make test-simulator     # Simulator functionality tests

# Traditional tests
make test-rtl           # RTL simulation tests
make test-sw            # Software compilation tests
make test-unit          # Unit tests
make test-integration   # Integration tests

# Hardware validation
make check-hardware     # Detailed hardware check
make check-hardware-quick # Quick hardware status
```

### Test Output and Logs
- **Results**: Saved to `test_results/` directory
- **Logs**: Detailed logs in `logs/` directory
- **Performance**: JSON reports with timing and accuracy metrics

## Documentation

### Available Documentation
- [Architecture Overview](docs/architecture_overview.md) - System architecture and design
- [L1 Cache Implementation](docs/l1_cache_implementation.md) - L1 cache design details
- [L2/L3 Cache Implementation](docs/l2_l3_cache_implementation.md) - Multi-level cache hierarchy
- [Memory Controller Implementation](docs/memory_controller_implementation.md) - Memory subsystem
- [AI Instruction Implementation](docs/ai_instruction_implementation.md) - Custom AI instructions
- [RTL Execution Guide](docs/RTL_EXECUTION_GUIDE.md) - RTL simulation and execution
- [macOS RTL Device Guide](docs/MACOS_RTL_DEVICE_GUIDE.md) - macOS development setup
- [macOS Simulator Guide](docs/MACOS_SIMULATOR_GUIDE.md) - Simulator usage
- [JS Waveform Viewer Guide](docs/JS_WAVEFORM_VIEWER_GUIDE.md) - Waveform analysis
- [Usage Guide](docs/USAGE_GUIDE.md) - General usage instructions

### Generating Documentation
```bash
make docs  # View available documentation
```

For HTML documentation generation, install Doxygen and run:
```bash
doxygen Doxyfile
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style
- RTL: Follow SystemVerilog coding standards
- Software: Follow Linux kernel coding style
- Documentation: Use Markdown format

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review existing test cases for examples

## Platform Support

### macOS Support
This project includes comprehensive macOS support with:
- **Native simulator**: RISC-V AI accelerator simulation on Apple Silicon
- **PyTorch integration**: Seamless AI framework compatibility
- **Development tools**: Complete toolchain for macOS development
- **Performance testing**: Benchmarking and validation tools

### Linux Support
- **Hardware acceleration**: Direct hardware access on compatible systems
- **Device drivers**: Kernel-level AI accelerator drivers
- **Performance optimization**: Hardware-specific optimizations

### Cross-Platform Features
- **PyTorch backend**: Unified AI framework interface
- **RTL simulation**: Hardware simulation on any platform
- **Testing framework**: Comprehensive validation suite

## Roadmap

### Current Status
- ✅ Basic architecture and interfaces defined
- ✅ Core RTL structure implemented
- ✅ Software driver framework established
- ✅ PyTorch integration framework
- ✅ macOS simulator implementation
- ✅ Comprehensive testing infrastructure
- 🚧 Individual component implementation (in progress)

### Upcoming Features
- Complete TPU and VPU implementations
- Advanced cache coherency protocols
- Enhanced power management features
- Hardware-accelerated PyTorch operations
- Performance optimization and tuning
- Extended AI instruction set
