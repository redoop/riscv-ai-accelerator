// AI Benchmark Package
// Comprehensive AI workload benchmarking framework

`ifndef AI_BENCHMARK_PKG_SV
`define AI_BENCHMARK_PKG_SV

package ai_benchmark_pkg;

    import uvm_pkg::*;
    `include "uvm_macros.svh"
    
    // Benchmark types
    typedef enum {
        MLPERF_INFERENCE,
        MLPERF_TRAINING,
        IMAGE_CLASSIFICATION,
        OBJECT_DETECTION,
        SEMANTIC_SEGMENTATION,
        NATURAL_LANGUAGE_PROCESSING,
        RECOMMENDATION_SYSTEM,
        SPEECH_RECOGNITION,
        MACHINE_TRANSLATION,
        CUSTOM_BENCHMARK
    } benchmark_type_e;
    
    // Model types
    typedef enum {
        RESNET50,
        RESNET101,
        VGG16,
        VGG19,
        MOBILENET_V1,
        MOBILENET_V2,
        EFFICIENTNET_B0,
        EFFICIENTNET_B7,
        YOLO_V3,
        YOLO_V4,
        YOLO_V5,
        SSD_MOBILENET,
        FASTER_RCNN,
        BERT_BASE,
        BERT_LARGE,
        GPT2_SMALL,
        GPT2_MEDIUM,
        TRANSFORMER_BASE,
        LSTM_SMALL,
        LSTM_LARGE,
        WIDE_DEEP,
        DEEP_FM,
        NEURAL_COLLABORATIVE_FILTERING
    } model_type_e;
    
    // Data types for benchmarks
    typedef enum {
        INT8_QUANT,
        INT16_QUANT,
        FP16_HALF,
        FP32_SINGLE,
        FP64_DOUBLE,
        MIXED_PRECISION
    } precision_type_e;
    
    // Benchmark configuration
    typedef struct {
        benchmark_type_e benchmark_type;
        model_type_e model_type;
        precision_type_e precision;
        int batch_size;
        int sequence_length;  // For NLP models
        int input_height;
        int input_width;
        int input_channels;
        int num_classes;
        string dataset_name;
        int num_samples;
        real target_accuracy;
        real target_latency_ms;
        real target_throughput_fps;
        real target_power_watts;
    } benchmark_config_t;
    
    // Performance metrics
    typedef struct {
        real accuracy_top1;
        real accuracy_top5;
        real latency_ms;
        real throughput_fps;
        real power_consumption_watts;
        real energy_per_inference_mj;
        longint total_operations;
        real tops_achieved;
        real memory_bandwidth_gbps;
        real cache_hit_rate;
        int total_samples_processed;
        time total_execution_time;
    } benchmark_results_t;
    
    // Include benchmark classes
    `include "ai_benchmark_base.sv"
    `include "mlperf_benchmarks.sv"
    `include "image_classification_benchmark.sv"
    `include "object_detection_benchmark.sv"
    `include "nlp_benchmark.sv"
    `include "recommendation_benchmark.sv"
    `include "benchmark_runner.sv"
    `include "benchmark_analyzer.sv"

endpackage : ai_benchmark_pkg

`endif // AI_BENCHMARK_PKG_SV