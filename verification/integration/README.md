# System Integration Testing Framework

This directory contains a comprehensive system-level integration testing framework for the RISC-V AI Accelerator chip. The framework tests the interaction and coordination between multiple system components under realistic workload conditions.

## Overview

The integration testing framework provides:
- **Multi-Core Coordination**: Tests synchronization and load balancing across CPU cores
- **Cache Coherency**: Validates cache coherency protocols and data consistency
- **NoC Communication**: Tests network-on-chip communication and routing
- **Power & Thermal Management**: Validates power management and thermal control
- **Stress & Endurance Testing**: Long-running stability and reliability tests
- **Mixed Workload Testing**: Real-world scenario simulation
- **System-Level Analysis**: Performance, scalability, and efficiency analysis

## Architecture

```
verification/integration/
├── system_integration_pkg.sv     # Main integration package
├── system_integration_base.sv    # Base integration test class
├── multi_core_test.sv            # Multi-core coordination tests
├── cache_coherency_test.sv       # Cache coherency validation
├── noc_communication_test.sv     # NoC communication tests
├── power_thermal_test.sv         # Power and thermal management
├── stress_endurance_test.sv      # Stress and endurance testing
├── mixed_workload_test.sv        # Mixed workload scenarios
├── system_monitor.sv             # System state monitoring
├── integration_analyzer.sv       # Results analysis and reporting
├── tb_system_integration.sv      # Top-level testbench
├── Makefile                      # Build and execution system
└── README.md                     # This file
```

## Integration Test Types

### Multi-Core Coordination Test
- **Purpose**: Validate coordination between multiple CPU cores
- **Features**:
  - Parallel task execution and synchronization
  - Load balancing across cores
  - Work stealing mechanisms
  - Barrier synchronization
  - Inter-core communication

### Cache Coherency Test
- **Purpose**: Ensure data consistency across cache hierarchy
- **Features**:
  - MESI/MOESI/MSI protocol validation
  - False sharing detection and mitigation
  - True sharing scenarios
  - Cache invalidation and writeback testing
  - Producer-consumer patterns

### NoC Communication Test
- **Purpose**: Validate network-on-chip communication
- **Features**:
  - Mesh topology routing
  - Deadlock avoidance
  - Quality of Service (QoS)
  - Congestion control
  - Bandwidth utilization

### Power & Thermal Management Test
- **Purpose**: Validate power and thermal control systems
- **Features**:
  - Dynamic Voltage and Frequency Scaling (DVFS)
  - Power gating and clock gating
  - Thermal throttling
  - Power budget management
  - Temperature monitoring

### Stress & Endurance Test
- **Purpose**: Long-term stability and reliability validation
- **Features**:
  - Extended runtime testing
  - High-load scenarios
  - Error injection and recovery
  - Memory stress testing
  - Thermal cycling

### Mixed Workload Test
- **Purpose**: Real-world scenario simulation
- **Features**:
  - AI + CPU workload mixing
  - Dynamic workload switching
  - Resource contention scenarios
  - Priority-based scheduling
  - Performance isolation

## Quick Start

### Prerequisites
- SystemVerilog simulator (Questa/ModelSim or VCS)
- UVM library
- Make utility
- Python 3.x (for analysis scripts)

### Basic Usage

1. **Compile the framework:**
   ```bash
   make compile
   ```

2. **Run multi-core coordination test:**
   ```bash
   make multi_core
   ```

3. **Run cache coherency test:**
   ```bash
   make cache_coherency
   ```

4. **Run all integration tests:**
   ```bash
   make regression
   ```

### Individual Test Execution

```bash
# Multi-core coordination
make multi_core

# Cache coherency validation
make cache_coherency

# NoC communication testing
make noc_comm

# Power and thermal management
make power_thermal

# Stress testing
make stress

# Mixed workload scenarios
make mixed_workload
```

## Configuration Options

### Test Complexity Levels
- **BASIC_INTEGRATION**: Simple scenarios with minimal components
- **INTERMEDIATE_INTEGRATION**: Moderate complexity with multiple components
- **ADVANCED_INTEGRATION**: Complex scenarios with full system
- **COMPREHENSIVE_INTEGRATION**: Exhaustive testing with all features

### System Components
Tests can be configured to include specific components:
- CPU cores (1-4 cores)
- TPU units (0-2 units)
- VPU units (0-2 units)
- Cache hierarchy (L1/L2/L3)
- Memory controller
- NoC routers
- Power management
- Thermal controller

### Workload Patterns
- **mixed**: Balanced CPU and AI workloads
- **cpu_intensive**: CPU-heavy workloads
- **ai_intensive**: AI accelerator-heavy workloads
- **memory_intensive**: Memory bandwidth-heavy workloads
- **cache_intensive**: Cache hierarchy stress testing

## Performance Metrics

### System-Level Metrics
- **Overall Throughput**: Operations per second
- **Latency Percentiles**: P50, P95, P99 response times
- **Scalability Factor**: Multi-core scaling efficiency
- **Reliability Score**: Error rate and system stability

### Component-Specific Metrics
- **CPU Utilization**: Per-core utilization rates
- **Cache Hit Rates**: L1/L2/L3 cache effectiveness
- **Memory Bandwidth**: Memory subsystem utilization
- **NoC Bandwidth**: Network utilization and congestion
- **Power Efficiency**: Operations per watt
- **Thermal Efficiency**: Operations per degree above baseline

## Analysis and Reporting

### Automated Analysis
The framework provides comprehensive analysis:
- **Performance Analysis**: Throughput, latency, and utilization
- **Scalability Analysis**: Multi-core scaling effectiveness
- **Reliability Analysis**: Error rates and system stability
- **Efficiency Analysis**: Power and thermal efficiency

### Report Generation
Multiple report formats are supported:
- **Text Reports**: Detailed analysis with recommendations
- **CSV Data**: Raw metrics for spreadsheet analysis
- **HTML Reports**: Interactive web-based reports

### Scoring System
Tests are scored on a 0-100 scale:
- **80-100**: Excellent system integration
- **60-79**: Good system integration
- **40-59**: Acceptable with improvements needed
- **0-39**: Poor integration requiring significant work

## Advanced Features

### Error Injection
- Hardware error simulation
- Transient fault injection
- Recovery mechanism testing
- Fault isolation validation

### Workload Mixing
- Dynamic workload switching
- Resource contention simulation
- Priority-based scheduling
- Performance isolation testing

### Long-Term Testing
- Extended runtime scenarios
- Thermal cycling simulation
- Wear-out mechanism testing
- Reliability prediction

## Extending the Framework

### Adding New Integration Tests

1. **Create test class** extending `system_integration_base`:
```systemverilog
class my_integration_test extends system_integration_base;
    virtual function void configure_test();
        super.configure_test();
        config.test_type = MY_TEST_TYPE;
        // Configure test-specific parameters
    endfunction
    
    virtual task run_integration_test();
        // Implement test-specific logic
    endfunction
endclass
```

2. **Register test** in the package and Makefile

### Custom System Components
Add new components by extending the `system_component_e` enumeration and implementing corresponding initialization and monitoring logic.

### Custom Metrics
Extend the `system_state_t` and `integration_performance_t` structures to include additional metrics specific to your system.

## Troubleshooting

### Common Issues

1. **Long Test Runtime**
   - Reduce test duration in configuration
   - Use BASIC_INTEGRATION complexity
   - Disable detailed monitoring

2. **Memory Usage**
   - Limit state history collection
   - Reduce number of concurrent transactions
   - Use smaller workload patterns

3. **Simulation Performance**
   - Disable waveform generation for production runs
   - Use optimized simulator settings
   - Run tests in parallel where possible

### Debug Commands

```bash
# Run with waveforms and high verbosity
make debug INTEGRATION_TEST=multi_core_coordination_test

# Quick smoke test
make smoke

# Performance-focused test
make perf
```

## Validation Methodology

### Test Coverage
- All major system components
- Critical interaction scenarios
- Error and corner cases
- Performance and scalability limits

### Verification Approach
- Bottom-up component validation
- Top-down system integration
- Cross-cutting concern validation
- Real-world scenario simulation

### Success Criteria
- All integration tests pass
- Performance targets met
- Reliability requirements satisfied
- Power and thermal limits respected

## Support and Resources

### Documentation
- [UVM User Guide](https://www.accellera.org/downloads/standards/uvm)
- [SystemVerilog LRM](https://ieeexplore.ieee.org/document/8299595)
- [Cache Coherency Protocols](https://en.wikipedia.org/wiki/Cache_coherence)

### Best Practices
- Start with simple integration scenarios
- Build complexity incrementally
- Monitor system health continuously
- Analyze results thoroughly
- Document test configurations and results

For questions or issues with the integration testing framework, consult the troubleshooting section or contact the verification team.