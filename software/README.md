# TPU Programming Interface

This directory contains the complete software stack for the RISC-V AI Accelerator's Tensor Processing Unit (TPU) programming interface.

## Overview

The TPU programming interface provides a comprehensive software stack for utilizing the AI accelerator's tensor processing capabilities:

- **Low-level driver interface** (`drivers/`) - Direct hardware access and control
- **High-level library** (`lib/`) - Easy-to-use programming interface
- **Integration tests** (`tests/`) - Comprehensive test suite
- **Examples** (`examples/`) - Demonstration programs

## Architecture

```
┌─────────────────────────────────────────┐
│           User Applications             │
├─────────────────────────────────────────┤
│         High-Level Library              │
│            (libtpu.h/c)                 │
├─────────────────────────────────────────┤
│        TPU Interface Layer              │
│        (tpu_interface.h/c)              │
├─────────────────────────────────────────┤
│       Base AI Driver Layer              │
│      (ai_accel_driver.h/c)              │
├─────────────────────────────────────────┤
│          Hardware Layer                 │
│      (TPU RTL Implementation)           │
└─────────────────────────────────────────┘
```

## Components

### 1. Base AI Driver (`drivers/ai_accel_driver.*`)

Provides the foundational interface for all AI accelerators:

- Device initialization and management
- Memory allocation and management
- Basic task submission and synchronization
- Error handling and status reporting

**Key Features:**
- Unified interface for TPU and VPU accelerators
- Memory-mapped I/O (MMIO) access
- DMA-coherent memory allocation
- Interrupt-driven completion notification

### 2. TPU Interface (`drivers/tpu_interface.*`)

Specialized interface for TPU operations:

- TPU-specific task creation and management
- Matrix operation parameter handling
- Performance counter access
- Hardware-specific optimizations

**Supported Operations:**
- Matrix multiplication (GEMM)
- 2D Convolution
- Batch matrix multiplication
- Transpose operations

**Data Types:**
- INT8 (8-bit integer)
- FP16 (16-bit floating point)
- FP32 (32-bit floating point)

### 3. High-Level Library (`lib/libtpu.*`)

User-friendly programming interface:

- Context-based programming model
- Automatic memory management
- Simplified matrix operations
- Performance profiling integration

**Programming Model:**
```c
// Create TPU context
tpu_context_t ctx;
tpu_create_context(&ctx, 0);

// Create matrices
tpu_matrix_t A, B, C;
tpu_matrix_create(&A, 128, 128, AI_DTYPE_FP32, false);
tpu_matrix_create(&B, 128, 128, AI_DTYPE_FP32, false);
tpu_matrix_create(&C, 128, 128, AI_DTYPE_FP32, false);

// Perform matrix multiplication: C = A * B
tpu_matrix_multiply(ctx, &A, &B, &C, false, false);

// Cleanup
tpu_matrix_destroy(&A);
tpu_matrix_destroy(&B);
tpu_matrix_destroy(&C);
tpu_destroy_context(ctx);
```

## Building

### Prerequisites

- GCC compiler with C99 support
- Make build system
- POSIX-compliant system (Linux/Unix)

### Build Instructions

```bash
# Build and run integration tests
cd tests/
make all
make test

# Build and run examples
cd examples/
make all
make run

# Clean all build artifacts
make clean
```

### Dependencies

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install build-essential valgrind

# Or use the provided target
cd tests/
make install-deps
```

## Usage Examples

### Basic Matrix Multiplication

```c
#include "lib/libtpu.h"

int main() {
    // Initialize TPU
    tpu_init();
    
    // Create context
    tpu_context_t ctx;
    tpu_create_context(&ctx, 0);
    
    // Create 64x64 matrices
    tpu_matrix_t A, B, C;
    tpu_matrix_create(&A, 64, 64, AI_DTYPE_FP32, false);
    tpu_matrix_create(&B, 64, 64, AI_DTYPE_FP32, false);
    tpu_matrix_create(&C, 64, 64, AI_DTYPE_FP32, false);
    
    // Fill matrices with data
    // ... (fill A and B with your data)
    
    // Perform C = A * B
    tpu_matrix_multiply(ctx, &A, &B, &C, false, false);
    
    // Use result in C
    // ...
    
    // Cleanup
    tpu_matrix_destroy(&A);
    tpu_matrix_destroy(&B);
    tpu_matrix_destroy(&C);
    tpu_destroy_context(ctx);
    tpu_cleanup();
    
    return 0;
}
```

### Asynchronous Execution

```c
// Set asynchronous execution mode
tpu_set_execution_mode(ctx, TPU_EXEC_ASYNC);

// Submit multiple operations
tpu_matrix_multiply(ctx, &A1, &B1, &C1, false, false);
tpu_matrix_multiply(ctx, &A2, &B2, &C2, false, false);
tpu_matrix_multiply(ctx, &A3, &B3, &C3, false, false);

// Wait for all operations to complete
tpu_synchronize(ctx);
```

### Performance Monitoring

```c
// Enable profiling
tpu_set_profiling(ctx, true);
tpu_reset_performance_stats(ctx);

// Perform operations
tpu_matrix_multiply(ctx, &A, &B, &C, false, false);

// Get performance statistics
tpu_performance_counters_t counters;
tpu_get_performance_stats(ctx, &counters);

printf("Throughput: %.2f GOPS\n", counters.throughput_gops);
printf("Utilization: %.1f%%\n", counters.utilization);

// Or print detailed summary
tpu_print_performance_summary(ctx);
```

### Multi-TPU Usage

```c
// Use multiple TPUs concurrently
tpu_context_t ctx1, ctx2;
tpu_create_context(&ctx1, 0);  // TPU 0
tpu_create_context(&ctx2, 1);  // TPU 1

// Set async mode
tpu_set_execution_mode(ctx1, TPU_EXEC_ASYNC);
tpu_set_execution_mode(ctx2, TPU_EXEC_ASYNC);

// Submit to both TPUs
tpu_matrix_multiply(ctx1, &A1, &B1, &C1, false, false);
tpu_matrix_multiply(ctx2, &A2, &B2, &C2, false, false);

// Wait for both
tpu_synchronize(ctx1);
tpu_synchronize(ctx2);
```

## Testing

### Integration Tests

The test suite provides comprehensive validation:

```bash
cd tests/
make test
```

**Test Coverage:**
- TPU initialization and device detection
- Context management and lifecycle
- Memory allocation and data transfer
- Matrix operations (various sizes and types)
- Performance monitoring
- Error handling and edge cases
- Multi-TPU concurrent operations

### Memory Testing

```bash
# Run tests with memory leak detection
make test-valgrind
```

### Static Analysis

```bash
# Run static code analysis
make analyze
```

## Performance Characteristics

### Theoretical Performance

- **Peak Throughput**: 256 TOPS (INT8), 64 TFLOPS (FP16)
- **Memory Bandwidth**: 1.6 TB/s (HBM2E)
- **Matrix Array**: 64x64 MAC units per TPU
- **Cache Hierarchy**: 512KB weight cache, 256KB activation cache

### Optimization Guidelines

1. **Data Types**: Use INT8 for maximum throughput, FP16 for balanced performance/accuracy
2. **Matrix Sizes**: Prefer multiples of 64 for optimal tile utilization
3. **Memory Layout**: Use row-major layout for best cache performance
4. **Batch Processing**: Use batch operations for improved throughput
5. **Asynchronous Execution**: Overlap computation with data transfer

### Performance Tuning

```c
// Get optimal tile sizes for your matrices
uint32_t tile_m, tile_n, tile_k;
tpu_get_optimal_tile_size(M, N, K, AI_DTYPE_FP16, &tile_m, &tile_n, &tile_k);

// Use the recommended tile sizes for manual tiling
```

## Error Handling

The interface provides comprehensive error reporting:

```c
ai_status_t status = tpu_matrix_multiply(ctx, &A, &B, &C, false, false);

switch (status) {
    case AI_STATUS_SUCCESS:
        // Operation completed successfully
        break;
    case AI_STATUS_INVALID_PARAM:
        // Invalid parameters (dimension mismatch, null pointers, etc.)
        break;
    case AI_STATUS_BUSY:
        // TPU is busy, try again later
        break;
    case AI_STATUS_TIMEOUT:
        // Operation timed out
        break;
    case AI_STATUS_DEVICE_ERROR:
        // Hardware error occurred
        break;
    case AI_STATUS_NO_MEMORY:
        // Memory allocation failed
        break;
}
```

## Hardware Requirements

### Minimum System Requirements

- RISC-V AI Accelerator chip with TPU support
- Linux kernel with AI accelerator driver support
- Minimum 4GB system memory
- PCIe interface for host communication

### Supported Configurations

- **Single TPU**: Basic AI acceleration
- **Dual TPU**: Parallel processing support
- **Multi-chip**: Distributed processing (future)

## Troubleshooting

### Common Issues

1. **Device Not Found**
   ```bash
   # Check if device file exists
   ls -l /dev/ai_accel
   
   # Check kernel module
   lsmod | grep ai_accel
   ```

2. **Permission Denied**
   ```bash
   # Add user to ai_accel group
   sudo usermod -a -G ai_accel $USER
   
   # Or run with sudo (not recommended for production)
   sudo ./your_program
   ```

3. **Memory Allocation Failures**
   - Reduce matrix sizes
   - Check available system memory
   - Verify DMA-coherent memory limits

4. **Performance Issues**
   - Use optimal data types (INT8/FP16)
   - Align matrix dimensions to hardware boundaries
   - Enable performance monitoring to identify bottlenecks

### Debug Mode

```c
// Enable debug output (if compiled with debug support)
#define TPU_DEBUG 1
#include "lib/libtpu.h"
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `tpu_init()` | Initialize TPU subsystem |
| `tpu_cleanup()` | Cleanup TPU subsystem |
| `tpu_create_context()` | Create TPU context |
| `tpu_destroy_context()` | Destroy TPU context |
| `tpu_matrix_multiply()` | Perform matrix multiplication |
| `tpu_get_performance_stats()` | Get performance counters |

### Data Structures

| Structure | Description |
|-----------|-------------|
| `tpu_context_t` | TPU context handle |
| `tpu_matrix_t` | Matrix descriptor |
| `tpu_performance_counters_t` | Performance statistics |
| `ai_tensor_t` | Generic tensor descriptor |

## Contributing

### Code Style

- Follow C99 standard
- Use consistent indentation (4 spaces)
- Add comprehensive comments
- Include error checking for all operations

### Testing

- Add unit tests for new functionality
- Ensure all tests pass before submitting
- Include performance benchmarks for optimizations

### Documentation

- Update API documentation for interface changes
- Add examples for new features
- Update this README for significant changes

## License

This software is part of the RISC-V AI Accelerator project and is subject to the project's licensing terms.

## Support

For technical support and questions:

1. Check the troubleshooting section above
2. Review the integration tests for usage examples
3. Consult the hardware documentation for low-level details
4. Submit issues through the project's issue tracking system