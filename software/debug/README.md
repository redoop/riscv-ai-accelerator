# RISC-V AI Accelerator Debug Tools

This directory contains comprehensive debugging and analysis tools for the RISC-V AI accelerator, including JTAG debugging interface, GDB server integration, and performance analysis capabilities.

## Components

### JTAG Interface (`jtag_interface.h/.c`)
- Low-level JTAG TAP controller implementation
- Debug module interface for RISC-V debug specification
- Hardware abstraction for different JTAG adapters
- Support for multi-hart debugging

### GDB Server (`gdb_server.h/.c`)
- GDB remote debugging protocol implementation
- TCP server for GDB client connections
- Breakpoint and watchpoint management
- Memory and register access through JTAG

### Performance Analyzer (`performance_analyzer.h`)
- Hardware performance counter interface
- Profiling and sampling capabilities
- AI accelerator specific monitoring
- Report generation and visualization

### Hardware Counters (`hardware_counters.c`)
- Memory-mapped register access for performance counters
- TPU, VPU, and NoC monitoring
- Power consumption tracking
- Real-time performance metrics

## Features

### JTAG Debugging
- **IEEE 1149.1 JTAG Support**: Full JTAG TAP state machine implementation
- **RISC-V Debug Module**: Compatible with RISC-V debug specification v0.13
- **Multi-Hart Support**: Debug multiple processor cores simultaneously
- **Hardware Breakpoints**: Set and manage hardware breakpoints
- **Memory Access**: Read/write memory through debug module
- **Register Access**: Access all processor registers including CSRs

### GDB Integration
- **Remote Debugging**: Standard GDB remote protocol support
- **Multi-Client**: Support multiple concurrent GDB connections
- **Packet Protocol**: Full GDB packet protocol implementation
- **Target Control**: Start, stop, step, and continue execution
- **Symbol Support**: Integration with symbol tables and debug info

### Performance Analysis
- **Hardware Counters**: Access to 32+ hardware performance counters
- **AI Accelerator Metrics**: TPU and VPU utilization monitoring
- **Memory Performance**: Cache hit rates, memory bandwidth
- **Power Monitoring**: Real-time power consumption tracking
- **Profiling**: Statistical profiling with configurable sampling
- **Report Generation**: Comprehensive performance reports

## Building

### Prerequisites
```bash
# Install development tools
sudo apt-get install build-essential gcc make

# Install optional dependencies
sudo apt-get install doxygen graphviz  # For documentation
sudo apt-get install cppcheck clang    # For static analysis
```

### Build Commands
```bash
# Build all components
make all

# Build specific targets
make libdebug.a      # Debug library
make gdb_server      # GDB server executable
make perf_analyzer   # Performance analyzer tool
make test_debug_tools # Test suite

# Build with debug symbols
make debug

# Build optimized release
make release
```

## Usage

### GDB Server
```bash
# Start GDB server on default port (3333)
./gdb_server

# Start on custom port
./gdb_server 1234

# Connect with GDB
riscv64-unknown-elf-gdb
(gdb) target remote localhost:3333
(gdb) load program.elf
(gdb) break main
(gdb) continue
```

### Performance Analyzer
```bash
# Run basic performance analysis
./perf_analyzer -s my_session -t 10 -o report.txt

# Enable profiling
./perf_analyzer -s profiling_session -t 30 -p -o profile_report.txt

# Add specific counters
./perf_analyzer -c 0 -c 1 -c 2 -t 5
```

### Programming Interface

#### JTAG Interface
```c
#include "jtag_interface.h"

jtag_interface_t jtag;
debug_target_t target;

// Initialize JTAG interface
jtag_init(&jtag, tck_pin, tms_pin, tdi_pin, tdo_pin, trst_pin);

// Initialize debug target
debug_init(&jtag, &target);

// Halt processor
debug_halt_hart(&jtag, 0);

// Read register
uint64_t reg_value;
debug_read_register(&jtag, 1, &reg_value);  // Read x1

// Set breakpoint
uint32_t bp_id;
debug_set_breakpoint(&jtag, 0x80000000, &bp_id);

// Resume execution
debug_resume_hart(&jtag, 0);
```

#### Performance Monitoring
```c
#include "performance_analyzer.h"

// Initialize performance counters
perf_init_counters();

// Create monitoring session
perf_session_t session;
perf_session_create(&session, "my_session");

// Add counters
perf_session_add_counter(&session, PERF_COUNTER_CYCLES);
perf_session_add_counter(&session, PERF_COUNTER_INSTRUCTIONS);
perf_session_add_counter(&session, PERF_COUNTER_TPU_OPERATIONS);

// Start monitoring
perf_session_start(&session);

// ... run workload ...

// Stop monitoring
perf_session_stop(&session);

// Generate report
perf_report_t report;
perf_analyze_session(&session, &report);
perf_generate_report(&report, "performance_report.txt");
```

#### Profiling
```c
#include "performance_analyzer.h"

profiling_data_t profiling;
memset(&profiling, 0, sizeof(profiling));
profiling.max_samples = 10000;

// Start profiling with 1000-cycle sampling interval
profiling_start(&profiling, 1000);

// ... run workload ...

// Stop profiling and analyze
profiling_stop(&profiling);
profiling_analyze(&profiling);

// Export results
perf_export_csv(&profiling, "profile_data.csv");
```

## Hardware Integration

### JTAG Connection
The JTAG interface requires connection to the following signals:
- **TCK**: Test Clock
- **TMS**: Test Mode Select  
- **TDI**: Test Data In
- **TDO**: Test Data Out
- **TRST**: Test Reset (optional)

### Performance Counters
Performance counters are accessed through memory-mapped registers:
- **Base Address**: 0x10000000 (CPU counters)
- **AI Accelerator**: 0x20000000 (TPU/VPU counters)
- **NoC Monitor**: 0x30000000 (Network-on-Chip counters)
- **Power Monitor**: 0x40000000 (Power consumption counters)

## Available Counters

### CPU Performance Counters
- `PERF_COUNTER_CYCLES` - CPU cycles
- `PERF_COUNTER_INSTRUCTIONS` - Instructions executed
- `PERF_COUNTER_CACHE_MISSES` - Cache misses
- `PERF_COUNTER_CACHE_HITS` - Cache hits
- `PERF_COUNTER_BRANCH_MISSES` - Branch mispredictions
- `PERF_COUNTER_TLB_MISSES` - TLB misses

### AI Accelerator Counters
- `PERF_COUNTER_TPU_CYCLES` - TPU active cycles
- `PERF_COUNTER_TPU_OPERATIONS` - TPU operations executed
- `PERF_COUNTER_VPU_CYCLES` - VPU active cycles
- `PERF_COUNTER_VPU_OPERATIONS` - VPU operations executed

### System Counters
- `PERF_COUNTER_NOC_PACKETS` - Network packets transmitted
- `PERF_COUNTER_NOC_STALLS` - Network stall cycles
- `PERF_COUNTER_POWER_EVENTS` - Power management events

## Testing

### Run Test Suite
```bash
# Build and run all tests
make test

# Run specific test categories
./test_debug_tools
```

### Test Coverage
The test suite covers:
- JTAG interface functionality
- Debug module operations
- GDB server protocol handling
- Performance counter access
- Profiling and analysis
- End-to-end debugging scenarios

### Benchmarking
```bash
# Run performance benchmarks
make benchmark

# Custom benchmark
./perf_analyzer -s benchmark -t 60 -p
```

## Troubleshooting

### Common Issues

1. **JTAG Connection Failed**
   - Check physical connections
   - Verify GPIO pin assignments
   - Ensure proper voltage levels

2. **GDB Server Connection Refused**
   - Check if port is already in use
   - Verify firewall settings
   - Ensure JTAG interface is working

3. **Performance Counters Not Available**
   - Check if running with sufficient privileges
   - Verify memory-mapped register access
   - Ensure hardware support is enabled

### Debug Mode
Enable debug logging:
```c
#define DEBUG 1
// Recompile with debug symbols
make debug
```

### Verbose Output
```bash
# Enable verbose GDB server
./gdb_server -v

# Enable verbose performance analyzer
./perf_analyzer -v -s debug_session
```

## Development

### Code Style
- Follow Linux kernel coding style
- Use `make format` to format code
- Run `make analyze` for static analysis

### Adding New Features
1. Update header files with new function declarations
2. Implement functionality in corresponding .c files
3. Add tests to `test_debug_tools.c`
4. Update documentation

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## Documentation

### Generate API Documentation
```bash
make docs
# Open docs/html/index.html in browser
```

### Manual Pages
```bash
man gdb_server
man perf_analyzer
```

## License

This project is licensed under the Apache License 2.0. See the main project LICENSE file for details.

## References

- [RISC-V Debug Specification](https://github.com/riscv/riscv-debug-spec)
- [GDB Remote Serial Protocol](https://sourceware.org/gdb/onlinedocs/gdb/Remote-Protocol.html)
- [IEEE 1149.1 JTAG Standard](https://standards.ieee.org/standard/1149_1-2013.html)
- [RISC-V Instruction Set Manual](https://riscv.org/specifications/)