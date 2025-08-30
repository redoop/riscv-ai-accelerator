# RISC-V AI Accelerator UVM Verification Environment

This directory contains a comprehensive UVM (Universal Verification Methodology) verification environment for the RISC-V AI Accelerator chip. The environment provides systematic verification of AI-specific instructions, accelerator units, and system-level functionality.

## Overview

The UVM environment includes:
- **Comprehensive Transaction Model**: Support for all AI operations (MATMUL, CONV2D, activations, pooling)
- **Randomized Testing**: Constrained random test generation with intelligent coverage
- **Directed Testing**: Focused tests for specific functionality
- **Performance Analysis**: Throughput, latency, and power consumption monitoring
- **Protocol Checking**: Automated verification of interface protocols
- **Coverage Collection**: Functional and code coverage with detailed reporting

## Architecture

```
verification/uvm/
├── riscv_ai_pkg.sv              # Main UVM package
├── riscv_ai_interface.sv        # SystemVerilog interface
├── riscv_ai_transaction.sv      # Base transaction class
├── riscv_ai_sequence_item.sv    # Extended sequence item
├── riscv_ai_*_sequence.sv       # Test sequences (random, directed)
├── riscv_ai_driver.sv           # UVM driver
├── riscv_ai_monitor.sv          # UVM monitor
├── riscv_ai_agent.sv            # UVM agent
├── riscv_ai_scoreboard.sv       # Result checking
├── riscv_ai_coverage.sv         # Coverage collection
├── riscv_ai_env.sv              # Test environment
├── riscv_ai_base_test.sv        # Base test classes
├── riscv_ai_utils.sv            # Utility functions
├── tb_riscv_ai_uvm.sv           # Top-level testbench
├── Makefile                     # Build system
├── run_tests.sh                 # Test runner script
└── README.md                    # This file
```

## Supported Operations

The verification environment supports all AI accelerator operations:

### Memory Operations
- **READ_OP**: Memory read with alignment checking
- **WRITE_OP**: Memory write with alignment checking

### AI Compute Operations
- **AI_MATMUL_OP**: Matrix multiplication (supports INT8, FP16, FP32)
- **AI_CONV2D_OP**: 2D convolution with configurable parameters

### AI Activation Operations
- **AI_RELU_OP**: ReLU activation function
- **AI_SIGMOID_OP**: Sigmoid activation function

### AI Pooling Operations
- **AI_MAXPOOL_OP**: Max pooling operation
- **AI_AVGPOOL_OP**: Average pooling operation

### AI Normalization Operations
- **AI_BATCHNORM_OP**: Batch normalization

## Test Categories

### Smoke Tests
Quick functionality verification:
- `riscv_ai_smoke_test`: Basic connectivity and functionality

### Basic Tests
Core functionality verification:
- `riscv_ai_random_test`: Comprehensive randomized testing
- `riscv_ai_matmul_test`: Matrix multiplication focused testing
- `riscv_ai_conv2d_test`: Convolution operation testing
- `riscv_ai_activation_test`: Activation function testing
- `riscv_ai_memory_test`: Memory access pattern testing

### Advanced Tests
Stress and corner case testing:
- `riscv_ai_stress_test`: High-load stress testing
- `riscv_ai_power_test`: Power consumption analysis
- `riscv_ai_error_test`: Error injection and recovery testing

## Quick Start

### Prerequisites
- SystemVerilog simulator (Questa/ModelSim, VCS, or Xcelium)
- UVM library (typically included with simulator)
- Make utility
- Bash shell (for test runner script)

### Basic Usage

1. **Compile the design:**
   ```bash
   make compile
   ```

2. **Run a single test:**
   ```bash
   make run TEST=riscv_ai_smoke_test
   ```

3. **Run all tests:**
   ```bash
   make regression
   ```

4. **Generate coverage report:**
   ```bash
   make coverage
   ```

### Using the Test Runner Script

The `run_tests.sh` script provides advanced test execution capabilities:

```bash
# Run smoke tests
./run_tests.sh smoke

# Run all tests with VCS simulator
./run_tests.sh -s vcs all

# Run specific test with waves enabled
./run_tests.sh -w 1 riscv_ai_matmul_test

# Run tests in parallel
./run_tests.sh -j 8 basic
```

## Configuration Options

### Makefile Variables
- `SIMULATOR`: Choose simulator (questa|vcs|xcelium)
- `TEST`: Specify test name
- `WAVES`: Enable waveform generation (0|1)
- `COVERAGE`: Enable coverage collection (0|1)
- `VERBOSITY`: Set UVM verbosity level

### Test Runner Options
- `-s, --simulator`: Simulator selection
- `-v, --verbosity`: UVM verbosity level
- `-w, --waves`: Waveform generation
- `-c, --coverage`: Coverage collection
- `-j, --jobs`: Parallel execution jobs
- `-t, --timeout`: Test timeout in seconds

## Coverage Goals

The verification environment tracks multiple coverage metrics:

### Functional Coverage
- **Operation Coverage**: All AI operations and data types
- **Matrix Coverage**: Various matrix dimensions and shapes
- **Convolution Coverage**: Different kernel sizes, strides, padding
- **Memory Coverage**: Address patterns and access sizes
- **Performance Coverage**: Latency and throughput ranges
- **Error Coverage**: Error conditions and recovery

### Coverage Goals
- **Total Coverage**: 95%+ for production readiness
- **Operation Coverage**: 100% (all operations must be tested)
- **Cross Coverage**: 90%+ for operation/data type combinations

## Performance Monitoring

The environment includes comprehensive performance analysis:

### Metrics Tracked
- **Throughput**: Operations per second, MB/s bandwidth
- **Latency**: Transaction completion time
- **Power Consumption**: Estimated power usage
- **Resource Utilization**: TPU/VPU usage patterns

### Performance Reports
- Real-time performance monitoring during simulation
- Post-simulation performance analysis
- Comparison against performance targets

## Debugging Features

### Waveform Analysis
- Automatic waveform generation for failed tests
- Protocol violation detection and reporting
- Performance bottleneck identification

### Logging and Reporting
- Hierarchical logging with configurable verbosity
- Detailed transaction logging
- Comprehensive test reports with pass/fail analysis

### Error Analysis
- Automatic error categorization
- Error injection for robustness testing
- Recovery mechanism verification

## Advanced Features

### Constrained Random Testing
- Intelligent constraint solving for realistic test scenarios
- Weighted randomization for operation type selection
- Automatic test case generation

### Protocol Checking
- Automated interface protocol verification
- Timing constraint checking
- Data integrity verification

### Multi-Level Testing
- Unit-level component testing
- System-level integration testing
- Cross-subsystem interaction testing

## Extending the Environment

### Adding New Tests
1. Create new test class extending `riscv_ai_base_test`
2. Implement `run_test()` method with test-specific sequences
3. Add test to Makefile test list
4. Update test runner script if needed

### Adding New Operations
1. Extend `operation_type_e` enumeration
2. Update transaction classes with new parameters
3. Modify driver and monitor for new operation
4. Add coverage points for new operation
5. Update scoreboard for result checking

### Custom Sequences
1. Extend `riscv_ai_base_sequence` or `riscv_ai_directed_sequence`
2. Implement `body()` task with sequence logic
3. Add constraints for realistic parameter generation
4. Register sequence with test classes

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Check RTL file paths in Makefile
   - Verify UVM library availability
   - Ensure SystemVerilog compiler compatibility

2. **Test Failures**
   - Check test logs in `logs/` directory
   - Verify DUT connectivity in testbench
   - Review scoreboard error messages

3. **Coverage Issues**
   - Ensure coverage compilation flags are set
   - Check coverage database generation
   - Verify coverage report generation tools

4. **Performance Issues**
   - Reduce test transaction count for faster runs
   - Disable waves for compilation-only runs
   - Use parallel test execution

### Debug Commands

```bash
# Compile with debug information
make compile SIMULATOR=questa WAVES=1

# Run test with maximum verbosity
make run TEST=riscv_ai_smoke_test VERBOSITY=UVM_FULL

# Generate detailed coverage report
make coverage

# Open waveforms for analysis
make waves TEST=riscv_ai_matmul_test
```

## Contributing

When contributing to the verification environment:

1. Follow UVM coding guidelines
2. Add appropriate coverage points for new features
3. Include both positive and negative test cases
4. Update documentation for new functionality
5. Ensure all tests pass before submitting changes

## Support

For questions or issues with the verification environment:
- Check the troubleshooting section above
- Review test logs for detailed error information
- Consult UVM documentation for methodology questions
- Contact the verification team for environment-specific issues