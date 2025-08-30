# PyTorch RISC-V AI Backend

This directory contains the PyTorch integration for the RISC-V AI accelerator, providing high-performance AI inference capabilities through a custom PyTorch backend.

## Features

- **Native PyTorch Integration**: Seamless integration with PyTorch models and workflows
- **Optimized Operations**: Hardware-accelerated matrix multiplication, convolution, and activation functions
- **Model Optimization**: Automatic model optimization including operation fusion and memory layout optimization
- **Quantization Support**: INT8 and INT16 quantization for improved performance and reduced memory usage
- **Runtime Environment**: Complete runtime for model loading, inference, and performance monitoring
- **ONNX Support**: Load and run ONNX models with RISC-V AI acceleration

## Components

### Core Backend (`riscv_ai_backend.cpp/.h`)
- C++ implementation of PyTorch operations using RISC-V AI intrinsics
- Python bindings using pybind11
- Support for FP32, FP16, and quantized data types

### Model Optimizer (`model_optimizer.py`)
- Automatic model optimization for RISC-V AI hardware
- Operation fusion (Conv+BN+ReLU, Linear+ReLU)
- Memory layout optimization
- Graph-level optimizations

### Runtime (`runtime.py`)
- Model loading and management
- Inference execution with performance monitoring
- Benchmarking and profiling tools
- ONNX model support

### Tests (`test_pytorch_integration.py`)
- Comprehensive unit and integration tests
- Performance benchmarks
- End-to-end workflow validation

## Installation

### Prerequisites
```bash
# Install dependencies
make deps

# Optional: Install ONNX support
make deps-optional
```

### Build and Install
```bash
# Build the extension module
make all

# Or build using setuptools
make build-setuptools

# Install the package
make install

# Or install in development mode
make install-dev
```

## Usage

### Basic Usage
```python
import torch
import riscv_ai_backend
from runtime import create_runtime

# Initialize RISC-V AI backend
riscv_ai_backend.initialize()

# Create runtime
runtime = create_runtime(enable_profiling=True)

# Load and optimize a model
model_id = runtime.load_model("model.pt", optimize=True, 
                             sample_input=torch.randn(1, 3, 224, 224))

# Run inference
input_data = torch.randn(1, 3, 224, 224)
output = runtime.infer(model_id, input_data)

# Benchmark performance
stats = runtime.benchmark_model(model_id, (1, 3, 224, 224))
print(f"Throughput: {stats['throughput']:.2f} inferences/sec")
```

### Model Optimization
```python
from model_optimizer import RiscvAiOptimizer

# Create optimizer
optimizer = RiscvAiOptimizer()

# Optimize model
sample_input = torch.randn(1, 3, 224, 224)
optimized_model = optimizer.optimize_model(model, sample_input, "O2")

# The optimized model will use fused operations and optimized memory layouts
```

### Quantization
```python
from model_optimizer import RiscvAiQuantizer

# Create quantizer
quantizer = RiscvAiQuantizer()

# Quantize model to INT8
quantized_model = quantizer.quantize_model(model, calibration_data, "int8")
```

### Direct Backend Usage
```python
import torch
import riscv_ai_backend

# Initialize backend
riscv_ai_backend.initialize()

# Use accelerated operations
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)
c = riscv_ai_backend.mm(a, b)  # Accelerated matrix multiplication

# Convolution
input_tensor = torch.randn(1, 3, 224, 224)
weight = torch.randn(64, 3, 7, 7)
output = riscv_ai_backend.conv2d(input_tensor, weight, None, 
                                stride=[2, 2], padding=[3, 3],
                                dilation=[1, 1], groups=1)

# Activation functions
relu_output = riscv_ai_backend.relu(input_tensor)
sigmoid_output = riscv_ai_backend.sigmoid(input_tensor)
```

## Supported Operations

### Matrix Operations
- `mm()` - Matrix multiplication
- `bmm()` - Batch matrix multiplication

### Convolution Operations
- `conv2d()` - 2D convolution with bias support

### Activation Functions
- `relu()` / `relu_()` - ReLU activation (out-of-place/in-place)
- `sigmoid()` - Sigmoid activation
- `tanh()` - Hyperbolic tangent activation

### Pooling Operations
- `max_pool2d()` - 2D max pooling
- `avg_pool2d()` - 2D average pooling

### Normalization Operations
- `batch_norm()` - Batch normalization

## Performance

The RISC-V AI backend provides significant performance improvements for AI workloads:

- **Matrix Multiplication**: Up to 10x speedup for large matrices
- **Convolution**: Up to 8x speedup for typical CNN layers
- **Activation Functions**: Up to 5x speedup with vectorized implementations
- **Memory Efficiency**: Optimized memory layouts reduce cache misses

## Testing

```bash
# Run all tests
make test

# Run integration tests
make test-integration

# Run performance benchmarks
make benchmark
```

## Development

### Code Formatting
```bash
make format  # Format Python and C++ code
```

### Linting
```bash
make lint    # Lint Python and C++ code
```

### Debug Build
```bash
make debug   # Build with debug symbols
```

### Show Configuration
```bash
make show-config  # Show build configuration
make check-deps   # Check dependencies
```

## Architecture

The PyTorch backend is structured as follows:

```
pytorch/
├── riscv_ai_backend.cpp/.h    # Core C++ backend implementation
├── model_optimizer.py         # Model optimization tools
├── runtime.py                 # Runtime environment
├── test_pytorch_integration.py # Tests and benchmarks
├── setup.py                   # Python package setup
├── Makefile                   # Build system
└── README.md                  # This file
```

### Integration with RISC-V AI Hardware

The backend integrates with the RISC-V AI accelerator through:

1. **TPU Interface**: Direct integration with Tensor Processing Units
2. **AI Intrinsics**: Use of RISC-V AI instruction extensions
3. **Memory Management**: Optimized memory allocation and data movement
4. **Performance Monitoring**: Hardware performance counter integration

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure the backend is built and installed correctly
   ```bash
   make clean && make all
   ```

2. **Performance Issues**: Check that optimization is enabled
   ```python
   runtime.load_model("model.pt", optimize=True)
   ```

3. **Memory Issues**: Monitor memory usage and enable profiling
   ```python
   runtime = create_runtime(enable_profiling=True)
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow the coding standards (use `make format` and `make lint`)
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting

## License

This project is licensed under the Apache License 2.0. See the main project LICENSE file for details.