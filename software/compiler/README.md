# RISC-V AI Compiler Support

This directory contains GCC compiler support for the RISC-V AI extension, including intrinsics, patches, and comprehensive testing.

## Overview

The RISC-V AI compiler support provides:

- **GCC Backend Integration**: Patches to add AI instruction support to GCC
- **Intrinsic Functions**: C/C++ intrinsics for AI instructions
- **Optimization Passes**: Compiler optimizations for AI workloads
- **Comprehensive Testing**: Test suite for validation and performance
- **Documentation**: Complete API reference and usage examples

## Features

### Supported AI Instructions

| Instruction | Description | Data Types | Performance |
|-------------|-------------|------------|-------------|
| `ai.matmul` | Matrix multiplication | FP32, FP16, INT8 | 256 TOPS (INT8) |
| `ai.conv2d` | 2D Convolution | FP32, FP16, INT8 | 128 TOPS (INT8) |
| `ai.relu` | ReLU activation | FP32, FP16 | 1 cycle/element |
| `ai.sigmoid` | Sigmoid activation | FP32, FP16 | 2 cycles/element |
| `ai.tanh` | Tanh activation | FP32, FP16 | 2 cycles/element |
| `ai.maxpool` | Max pooling | FP32, FP16 | 1 cycle/element |
| `ai.avgpool` | Average pooling | FP32, FP16 | 2 cycles/element |
| `ai.batchnorm` | Batch normalization | FP32, FP16 | 4 cycles/element |

### Compiler Optimizations

- **Instruction Fusion**: Automatic fusion of compatible operations
- **Memory Layout**: Optimal data layout for cache performance  
- **Vectorization**: Automatic use of vector instructions
- **Loop Unrolling**: Optimized loops for AI kernels
- **Constant Folding**: Compile-time optimization of AI parameters

## Quick Start

### Prerequisites

```bash
# Install RISC-V GCC toolchain
sudo apt-get install gcc-riscv64-unknown-elf

# Or build from source with AI patches applied
```

### Building Tests

```bash
# Build and run all tests
make test

# Run performance benchmarks
make benchmark

# Check compiler support
make check-compiler
```

### Basic Usage

```c
#include "riscv_ai_intrinsics.h"

// Matrix multiplication example
void matrix_multiply_example() {
    // Declare aligned matrices
    float AI_CACHE_ALIGN a[64*64], b[64*64], c[64*64];
    
    // Initialize matrices...
    
    // Perform hardware-accelerated matrix multiplication
    __builtin_riscv_ai_matmul_f32(a, b, c, 64, 64, 64);
}

// Activation function example
void activation_example() {
    float input[1024], output[1024];
    
    // Initialize input...
    
    // Apply ReLU activation
    __builtin_riscv_ai_relu_f32(input, output, 1024);
}
```

## Installation

### Installing Intrinsics Header

```bash
# Install to system include directory
make install

# Or install to custom location
make install DESTDIR=/usr/local
```

### Applying GCC Patches

```bash
# Download GCC source
wget https://gcc.gnu.org/releases/gcc-13.2.0/gcc-13.2.0.tar.gz
tar xzf gcc-13.2.0.tar.gz

# Apply AI extension patches
make patch-gcc GCC_SOURCE_DIR=./gcc-13.2.0

# Build GCC with AI support
cd gcc-13.2.0
./configure --target=riscv64-unknown-linux-gnu --enable-languages=c,c++
make -j$(nproc)
make install
```

## API Reference

### Matrix Operations

#### `__builtin_riscv_ai_matmul_f32`

Performs single-precision floating-point matrix multiplication.

```c
void __builtin_riscv_ai_matmul_f32(
    const float* a,     // Input matrix A (M×K)
    const float* b,     // Input matrix B (K×N)  
    float* c,           // Output matrix C (M×N)
    uint32_t m,         // Number of rows in A
    uint32_t n,         // Number of columns in B
    uint32_t k          // Number of columns in A
);
```

**Requirements:**
- All matrices must be 64-byte aligned
- Dimensions must be multiples of 8 for optimal performance
- Maximum dimension: 4096

**Performance:** Up to 64 TFLOPS for FP32 operations

#### `__builtin_riscv_ai_matmul_f16`

Half-precision floating-point matrix multiplication.

```c
void __builtin_riscv_ai_matmul_f16(
    const uint16_t* a,  // Input matrix A (FP16)
    const uint16_t* b,  // Input matrix B (FP16)
    uint16_t* c,        // Output matrix C (FP16)
    uint32_t m, uint32_t n, uint32_t k
);
```

**Performance:** Up to 128 TFLOPS for FP16 operations

#### `__builtin_riscv_ai_matmul_i8`

8-bit integer matrix multiplication with 32-bit accumulation.

```c
void __builtin_riscv_ai_matmul_i8(
    const int8_t* a,    // Input matrix A (INT8)
    const int8_t* b,    // Input matrix B (INT8)
    int32_t* c,         // Output matrix C (INT32)
    uint32_t m, uint32_t n, uint32_t k
);
```

**Performance:** Up to 256 TOPS for INT8 operations

### Activation Functions

#### `__builtin_riscv_ai_relu_f32`

Rectified Linear Unit activation function.

```c
void __builtin_riscv_ai_relu_f32(
    const float* input,  // Input array
    float* output,       // Output array
    uint32_t count       // Number of elements
);
```

**Formula:** `output[i] = max(0, input[i])`

**Performance:** 1 cycle per element

#### `__builtin_riscv_ai_sigmoid_f32`

Sigmoid activation function.

```c
void __builtin_riscv_ai_sigmoid_f32(
    const float* input,
    float* output,
    uint32_t count
);
```

**Formula:** `output[i] = 1 / (1 + exp(-input[i]))`

**Performance:** 2 cycles per element

### Control Functions

#### `__builtin_riscv_ai_get_status`

Read AI accelerator status.

```c
uint32_t __builtin_riscv_ai_get_status(uint32_t accel_id);
```

**Returns:** Status register value with flags:
- `AI_STATUS_READY`: Accelerator ready for new tasks
- `AI_STATUS_BUSY`: Accelerator currently processing
- `AI_STATUS_ERROR`: Error condition detected

#### `__builtin_riscv_ai_set_config`

Configure AI accelerator settings.

```c
void __builtin_riscv_ai_set_config(uint32_t accel_id, uint32_t config);
```

**Configuration flags:**
- `AI_CONFIG_ENABLE`: Enable accelerator
- `AI_CONFIG_PERF_ENABLE`: Enable performance counters
- `AI_CONFIG_PRECISION_FP32`: Use FP32 precision
- `AI_CONFIG_PRECISION_FP16`: Use FP16 precision

## Performance Optimization

### Memory Alignment

Always use 64-byte alignment for optimal performance:

```c
// Correct alignment
float AI_CACHE_ALIGN matrix[1024*1024];

// Or use aligned allocation
float* matrix = aligned_alloc(64, sizeof(float) * 1024 * 1024);
```

### Data Layout

Use row-major layout for matrices:

```c
// Access pattern: matrix[row * width + col]
for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        matrix[i * width + j] = value;
    }
}
```

### Batch Processing

Process multiple operations together:

```c
// Batch multiple matrix multiplications
for (int batch = 0; batch < num_batches; batch++) {
    __builtin_riscv_ai_matmul_f32(
        &a[batch * m * k], 
        &b[batch * k * n], 
        &c[batch * m * n], 
        m, n, k
    );
}
```

### Asynchronous Execution

Use asynchronous mode for overlapping computation:

```c
// Enable asynchronous mode
__builtin_riscv_ai_set_config(0, AI_CONFIG_ENABLE | AI_CONFIG_AUTO_SYNC);

// Submit multiple operations
__builtin_riscv_ai_matmul_f32(a1, b1, c1, m, n, k);
__builtin_riscv_ai_matmul_f32(a2, b2, c2, m, n, k);

// Wait for completion
__builtin_riscv_ai_sync(0);
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-intrinsics
make test-performance
make test-correctness

# Run with memory checking
make memcheck
```

### Test Coverage

The test suite covers:

- **Functional Tests**: Correctness of all intrinsics
- **Performance Tests**: Throughput and latency measurements
- **Edge Cases**: Boundary conditions and error handling
- **Compiler Tests**: Code generation and optimization
- **Integration Tests**: Multi-accelerator scenarios

### Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Generate detailed performance report
make benchmark > performance_report.txt
```

Example benchmark output:

```
Performance Benchmarks:
======================
MatMul 32x32: 0.125 ms, 32.8 GFLOPS
MatMul 64x64: 0.512 ms, 64.2 GFLOPS
MatMul 128x128: 2.048 ms, 128.1 GFLOPS
MatMul 256x256: 8.192 ms, 256.3 GFLOPS
ReLU (4096 elements): 4.1 us, 1000.0 M elem/s
```

## Troubleshooting

### Common Issues

#### Compiler Not Found

```bash
# Check if RISC-V GCC is installed
which riscv64-unknown-linux-gnu-gcc

# Install if missing
sudo apt-get install gcc-riscv64-unknown-elf
```

#### AI Extension Not Supported

```bash
# Check compiler flags
riscv64-unknown-linux-gnu-gcc -march=rv64imafdv -mai --help

# Verify AI extension in compiler
echo | riscv64-unknown-linux-gnu-gcc -march=rv64imafdv -mai -dM -E - | grep AI
```

#### Alignment Errors

```c
// Ensure proper alignment
float* matrix = aligned_alloc(64, size * sizeof(float));
if (!matrix) {
    fprintf(stderr, "Failed to allocate aligned memory\n");
    exit(1);
}
```

#### Performance Issues

1. **Check alignment**: Use 64-byte aligned data
2. **Optimize dimensions**: Use multiples of 8 or 16
3. **Enable optimizations**: Use `-O2` or `-O3` compiler flags
4. **Profile code**: Use performance counters to identify bottlenecks

### Debug Mode

Enable debug output for troubleshooting:

```c
#define AI_DEBUG 1
#include "riscv_ai_intrinsics.h"

// Debug information will be printed to stderr
```

### Performance Analysis

Use built-in performance counters:

```c
// Enable performance monitoring
__builtin_riscv_ai_set_config(0, AI_CONFIG_PERF_ENABLE);

// Read performance counters
uint32_t cycles = __builtin_riscv_ai_get_perf_counter(AI_PERF_CYCLES);
uint32_t ops = __builtin_riscv_ai_get_perf_counter(AI_PERF_MATMUL_OPS);

printf("Performance: %u cycles, %u operations\n", cycles, ops);
```

## Contributing

### Code Style

- Follow C99 standard for C code
- Use consistent 4-space indentation
- Add comprehensive comments for public APIs
- Include error checking for all operations

### Adding New Intrinsics

1. Add function declaration to `riscv_ai_intrinsics.h`
2. Add GCC builtin support in patch files
3. Add machine description in `riscv-ai.md`
4. Add test cases in `tests/`
5. Update documentation

### Submitting Changes

1. Run all tests: `make test`
2. Check code quality: `make lint`
3. Update documentation as needed
4. Submit pull request with detailed description

## License

This software is part of the RISC-V AI Accelerator project and is subject to the project's licensing terms.

## Support

For technical support:

1. Check this documentation
2. Review test cases for usage examples
3. Consult hardware documentation
4. Submit issues through project issue tracker