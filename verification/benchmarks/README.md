# RISC-V AI Accelerator Benchmark Framework

This directory contains a comprehensive AI workload benchmarking framework for the RISC-V AI Accelerator chip. The framework implements industry-standard benchmarks including MLPerf, along with custom benchmarks for various AI workloads.

## Overview

The benchmark framework provides:
- **MLPerf Standard Benchmarks**: Industry-standard AI performance benchmarks
- **Image Classification**: ResNet, MobileNet, EfficientNet models on ImageNet
- **Object Detection**: YOLO, SSD, Faster R-CNN models on COCO dataset
- **Natural Language Processing**: BERT, GPT-2, Transformer models
- **Recommendation Systems**: Wide & Deep, DeepFM, Neural Collaborative Filtering
- **Comprehensive Analysis**: Performance, efficiency, and comparison analysis
- **Automated Reporting**: HTML, CSV, and JSON report generation

## Architecture

```
verification/benchmarks/
├── ai_benchmark_pkg.sv           # Main benchmark package
├── ai_benchmark_base.sv          # Base benchmark class
├── mlperf_benchmarks.sv          # MLPerf inference/training benchmarks
├── image_classification_benchmark.sv  # Image classification benchmarks
├── object_detection_benchmark.sv # Object detection benchmarks
├── nlp_benchmark.sv              # Natural language processing benchmarks
├── recommendation_benchmark.sv   # Recommendation system benchmarks
├── benchmark_runner.sv           # Benchmark execution orchestrator
├── benchmark_analyzer.sv         # Results analysis and comparison
├── tb_ai_benchmarks.sv           # Top-level testbench
├── Makefile                      # Build and execution system
└── README.md                     # This file
```

## Supported Benchmarks

### MLPerf Benchmarks
- **ResNet-50 Inference**: Image classification on ImageNet
- **BERT Inference**: Natural language understanding
- **SSD-MobileNet Inference**: Object detection on COCO
- **Training Benchmarks**: ResNet-50, BERT training scenarios

### Image Classification
- **ResNet-50/101**: Deep residual networks
- **VGG-16/19**: Visual geometry group networks
- **MobileNet-V1/V2**: Mobile-optimized networks
- **EfficientNet-B0/B7**: Efficient convolutional networks

### Object Detection
- **YOLOv3/v4/v5**: You Only Look Once detectors
- **SSD-MobileNet**: Single Shot MultiBox Detector
- **Faster R-CNN**: Region-based CNN detector

### Natural Language Processing
- **BERT Base/Large**: Bidirectional encoder representations
- **GPT-2 Small/Medium**: Generative pre-trained transformer
- **Transformer Base**: Attention-based sequence models
- **LSTM Small/Large**: Long short-term memory networks

### Recommendation Systems
- **Wide & Deep**: Google's recommendation model
- **DeepFM**: Factorization machine with deep learning
- **Neural Collaborative Filtering**: Deep learning for recommendations

## Quick Start

### Prerequisites
- SystemVerilog simulator (Questa/ModelSim, VCS, or Xcelium)
- UVM library
- Make utility
- Python 3.x (for analysis scripts)

### Basic Usage

1. **Compile the framework:**
   ```bash
   make compile
   ```

2. **Run MLPerf benchmarks:**
   ```bash
   make mlperf
   ```

3. **Run all benchmarks:**
   ```bash
   make all_benchmarks
   ```

4. **Generate analysis reports:**
   ```bash
   make reports
   ```

### Benchmark Suites

Run specific benchmark categories:

```bash
# Image classification benchmarks
make image_classification

# Object detection benchmarks  
make object_detection

# NLP benchmarks
make nlp

# Recommendation system benchmarks
make recommendation

# Comprehensive regression testing
make regression
```

## Configuration Options

### Makefile Variables
- `SIMULATOR`: Choose simulator (questa|vcs)
- `BENCHMARK_SUITE`: Specify benchmark suite to run
- `WAVES`: Enable waveform generation (0|1)
- `COVERAGE`: Enable coverage collection (0|1)
- `VERBOSITY`: Set UVM verbosity level

### Command Line Options
```bash
# Run with specific simulator
make mlperf SIMULATOR=vcs

# Run with waveforms for debugging
make debug BENCHMARK_SUITE=image_classification

# Run with coverage analysis
make all_benchmarks COVERAGE=1
```

## Benchmark Metrics

The framework tracks comprehensive performance metrics:

### Accuracy Metrics
- **Top-1/Top-5 Accuracy**: Classification accuracy
- **mAP (mean Average Precision)**: Object detection accuracy
- **F1 Score**: NLP task performance
- **BLEU Score**: Translation quality
- **AUC**: Recommendation system performance

### Performance Metrics
- **Latency**: Inference time per sample
- **Throughput**: Samples processed per second
- **TOPS**: Tera operations per second
- **Memory Bandwidth**: Data transfer rate
- **Cache Hit Rate**: Memory efficiency

### Efficiency Metrics
- **Power Consumption**: Energy usage in watts
- **Energy per Inference**: Millijoules per inference
- **TOPS per Watt**: Computational efficiency
- **Performance per Dollar**: Cost efficiency

## Analysis and Reporting

### Performance Analysis
The framework provides detailed performance analysis:
- Category-wise performance scoring
- Efficiency analysis across different metrics
- Comparison with baseline results
- Trend analysis for optimization insights

### Report Generation
Multiple report formats are supported:
- **HTML Reports**: Interactive web-based reports
- **CSV Reports**: Data for spreadsheet analysis
- **JSON Reports**: Machine-readable results

### Comparison Analysis
Compare current results with baselines:
```bash
# Set current results as baseline
make set_baseline

# Compare with baseline
make compare
```

## Extending the Framework

### Adding New Benchmarks

1. **Create benchmark class** extending `ai_benchmark_base`:
```systemverilog
class my_custom_benchmark extends ai_benchmark_base;
    virtual function string get_benchmark_name();
        return "MyCustomBenchmark";
    endfunction
    
    virtual function void configure_benchmark(benchmark_config_t cfg);
        // Configure benchmark parameters
    endfunction
    
    virtual task run_benchmark();
        // Implement benchmark execution
    endfunction
    
    virtual function void analyze_results();
        // Analyze and validate results
    endfunction
endclass
```

2. **Register in benchmark runner**:
```systemverilog
// Add to register_benchmarks() function
my_custom_benchmark custom_bench = my_custom_benchmark::type_id::create("custom_bench");
custom_bench.configure_benchmark(cfg);
benchmarks.push_back(custom_bench);
```

### Custom Metrics
Add custom metrics by extending the `benchmark_results_t` structure and implementing corresponding analysis functions.

### Dataset Integration
The framework supports custom datasets by implementing the `load_dataset()` function in benchmark classes.

## Performance Optimization

### Batch Size Optimization
Different models perform optimally with different batch sizes:
- **Image Classification**: 32-128 samples
- **Object Detection**: 8-32 samples  
- **NLP Models**: 4-32 samples
- **Recommendation**: 256-1024 samples

### Precision Optimization
The framework supports multiple precision modes:
- **INT8**: 4x speedup, minimal accuracy loss
- **FP16**: 2x speedup, good accuracy retention
- **FP32**: Baseline precision
- **Mixed Precision**: Optimal speed/accuracy tradeoff

### Memory Optimization
- Use appropriate batch sizes to maximize memory utilization
- Consider model pruning for memory-constrained scenarios
- Monitor cache hit rates for memory access optimization

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   - Verify UVM library installation
   - Check SystemVerilog compiler version compatibility
   - Ensure all source files are present

2. **Benchmark Failures**
   - Check simulation timeout settings
   - Verify DUT connectivity
   - Review benchmark configuration parameters

3. **Performance Issues**
   - Reduce batch sizes for faster execution
   - Disable waveform generation for production runs
   - Use parallel execution for multiple benchmarks

### Debug Commands

```bash
# Compile with debug information
make compile WAVES=1

# Run with maximum verbosity
make mlperf VERBOSITY=UVM_FULL

# Debug specific benchmark with waveforms
make debug BENCHMARK_SUITE=image_classification
```

## Validation and Verification

### Accuracy Validation
- Compare results against reference implementations
- Validate against published benchmark scores
- Cross-check with software implementations

### Performance Validation
- Verify against theoretical peak performance
- Compare with other AI accelerators
- Validate power consumption estimates

### Coverage Analysis
- Functional coverage of all AI operations
- Code coverage of benchmark framework
- Corner case coverage for edge conditions

## Contributing

When contributing to the benchmark framework:

1. Follow SystemVerilog coding standards
2. Add comprehensive test coverage for new benchmarks
3. Include both positive and negative test cases
4. Update documentation for new features
5. Validate against reference implementations

## Support and Resources

### Documentation
- [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies)
- [MLPerf Training Rules](https://github.com/mlcommons/training_policies)
- [UVM User Guide](https://www.accellera.org/downloads/standards/uvm)

### Reference Implementations
- [MLPerf Reference Models](https://github.com/mlcommons/inference)
- [TensorFlow Model Garden](https://github.com/tensorflow/models)
- [PyTorch Examples](https://github.com/pytorch/examples)

### Performance Baselines
- Industry AI accelerator benchmarks
- Cloud AI service performance data
- Academic research benchmarks

For questions or issues with the benchmark framework, consult the troubleshooting section or contact the verification team.