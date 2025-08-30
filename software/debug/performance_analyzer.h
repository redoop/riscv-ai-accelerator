/*
 * Performance Analyzer for RISC-V AI Accelerator
 * Provides performance monitoring, profiling, and analysis tools
 */

#ifndef PERFORMANCE_ANALYZER_H
#define PERFORMANCE_ANALYZER_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hardware performance counter definitions
#define PERF_COUNTER_CYCLES             0x00
#define PERF_COUNTER_INSTRUCTIONS       0x01
#define PERF_COUNTER_CACHE_MISSES       0x02
#define PERF_COUNTER_CACHE_HITS         0x03
#define PERF_COUNTER_BRANCH_MISSES      0x04
#define PERF_COUNTER_BRANCH_TAKEN       0x05
#define PERF_COUNTER_TLB_MISSES         0x06
#define PERF_COUNTER_MEMORY_ACCESSES    0x07

// AI accelerator specific counters
#define PERF_COUNTER_TPU_CYCLES         0x10
#define PERF_COUNTER_TPU_OPERATIONS     0x11
#define PERF_COUNTER_TPU_CACHE_MISSES   0x12
#define PERF_COUNTER_VPU_CYCLES         0x13
#define PERF_COUNTER_VPU_OPERATIONS     0x14
#define PERF_COUNTER_NOC_PACKETS        0x15
#define PERF_COUNTER_NOC_STALLS         0x16
#define PERF_COUNTER_POWER_EVENTS       0x17

#define MAX_PERF_COUNTERS               32
#define MAX_PROFILING_SAMPLES           10000

// Performance counter configuration
typedef struct {
    uint32_t counter_id;        // Counter identifier
    bool enabled;               // Counter enabled flag
    uint64_t initial_value;     // Initial counter value
    uint64_t current_value;     // Current counter value
    const char* name;           // Counter name
    const char* description;    // Counter description
} perf_counter_t;

// Performance monitoring session
typedef struct {
    perf_counter_t counters[MAX_PERF_COUNTERS];
    int counter_count;          // Number of active counters
    
    struct timespec start_time; // Session start time
    struct timespec end_time;   // Session end time
    bool active;                // Session active flag
    
    uint64_t sample_interval;   // Sampling interval (cycles)
    int sample_count;           // Number of samples collected
    
    char session_name[256];     // Session name
    char output_file[512];      // Output file path
} perf_session_t;

// Profiling sample
typedef struct {
    uint64_t timestamp;         // Sample timestamp
    uint64_t pc;                // Program counter
    uint64_t counters[MAX_PERF_COUNTERS]; // Counter values
    uint32_t hart_id;           // Hart identifier
    uint32_t context_id;        // Context identifier
} perf_sample_t;

// Profiling data
typedef struct {
    perf_sample_t samples[MAX_PROFILING_SAMPLES];
    int sample_count;           // Number of samples
    int max_samples;            // Maximum samples
    
    // Analysis results
    uint64_t total_cycles;      // Total execution cycles
    uint64_t total_instructions; // Total instructions executed
    double ipc;                 // Instructions per cycle
    double cache_hit_rate;      // Cache hit rate
    double branch_prediction_rate; // Branch prediction accuracy
    
    // Hot spots
    uint64_t* hot_addresses;    // Hot spot addresses
    uint64_t* hot_counts;       // Hot spot execution counts
    int hot_spot_count;         // Number of hot spots
} profiling_data_t;

// Performance analysis report
typedef struct {
    char title[256];            // Report title
    struct timespec generation_time; // Report generation time
    
    // Overall statistics
    uint64_t execution_time_ns; // Total execution time (nanoseconds)
    uint64_t total_cycles;      // Total CPU cycles
    uint64_t total_instructions; // Total instructions
    double average_ipc;         // Average IPC
    
    // Cache statistics
    uint64_t l1_cache_hits;     // L1 cache hits
    uint64_t l1_cache_misses;   // L1 cache misses
    uint64_t l2_cache_hits;     // L2 cache hits
    uint64_t l2_cache_misses;   // L2 cache misses
    double l1_hit_rate;         // L1 cache hit rate
    double l2_hit_rate;         // L2 cache hit rate
    
    // AI accelerator statistics
    uint64_t tpu_operations;    // TPU operations executed
    uint64_t vpu_operations;    // VPU operations executed
    double tpu_utilization;     // TPU utilization percentage
    double vpu_utilization;     // VPU utilization percentage
    
    // Memory statistics
    uint64_t memory_reads;      // Memory read operations
    uint64_t memory_writes;     // Memory write operations
    uint64_t memory_bandwidth;  // Memory bandwidth (bytes/sec)
    
    // Power statistics
    double average_power;       // Average power consumption (watts)
    double peak_power;          // Peak power consumption (watts)
    double energy_consumed;     // Total energy consumed (joules)
    
    // Hot spots
    int hot_spot_count;         // Number of hot spots
    uint64_t* hot_addresses;    // Hot spot addresses
    double* hot_percentages;    // Hot spot percentages
} perf_report_t;

// Function prototypes

// Performance counter management
int perf_init_counters(void);
int perf_enable_counter(uint32_t counter_id);
int perf_disable_counter(uint32_t counter_id);
int perf_read_counter(uint32_t counter_id, uint64_t* value);
int perf_reset_counter(uint32_t counter_id);
int perf_reset_all_counters(void);

// Performance monitoring session
int perf_session_create(perf_session_t* session, const char* name);
int perf_session_start(perf_session_t* session);
int perf_session_stop(perf_session_t* session);
int perf_session_add_counter(perf_session_t* session, uint32_t counter_id);
int perf_session_remove_counter(perf_session_t* session, uint32_t counter_id);
void perf_session_destroy(perf_session_t* session);

// Profiling
int profiling_start(profiling_data_t* data, uint64_t sample_interval);
int profiling_stop(profiling_data_t* data);
int profiling_collect_sample(profiling_data_t* data);
int profiling_analyze(profiling_data_t* data);
void profiling_cleanup(profiling_data_t* data);

// Analysis and reporting
int perf_analyze_session(perf_session_t* session, perf_report_t* report);
int perf_generate_report(perf_report_t* report, const char* output_file);
int perf_export_csv(profiling_data_t* data, const char* filename);
int perf_export_json(perf_report_t* report, const char* filename);

// Visualization helpers
int perf_generate_timeline(profiling_data_t* data, const char* output_file);
int perf_generate_heatmap(profiling_data_t* data, const char* output_file);
int perf_generate_callgraph(profiling_data_t* data, const char* output_file);

// Utility functions
const char* perf_counter_name(uint32_t counter_id);
const char* perf_counter_description(uint32_t counter_id);
double perf_calculate_ipc(uint64_t instructions, uint64_t cycles);
double perf_calculate_cache_hit_rate(uint64_t hits, uint64_t misses);
uint64_t perf_cycles_to_nanoseconds(uint64_t cycles, uint64_t frequency);
double perf_calculate_bandwidth(uint64_t bytes, uint64_t time_ns);

// Hardware abstraction
int perf_read_hardware_counter(uint32_t counter_id, uint64_t* value);
int perf_write_hardware_counter(uint32_t counter_id, uint64_t value);
int perf_configure_sampling(uint64_t interval, bool enable_interrupts);

// AI accelerator specific functions
int perf_monitor_tpu_utilization(uint32_t tpu_id, double* utilization);
int perf_monitor_vpu_utilization(uint32_t vpu_id, double* utilization);
int perf_monitor_noc_traffic(uint64_t* packets_sent, uint64_t* packets_received);
int perf_monitor_power_consumption(double* current_power, double* average_power);

// Benchmarking utilities
int perf_benchmark_matrix_multiply(int size, double* gflops);
int perf_benchmark_convolution(int channels, int height, int width, double* gflops);
int perf_benchmark_memory_bandwidth(size_t buffer_size, double* bandwidth);
int perf_benchmark_cache_latency(int cache_level, double* latency_ns);

// Error codes
#define PERF_SUCCESS                0
#define PERF_ERROR_INIT            -1
#define PERF_ERROR_INVALID_COUNTER -2
#define PERF_ERROR_NOT_SUPPORTED   -3
#define PERF_ERROR_PERMISSION      -4
#define PERF_ERROR_OVERFLOW        -5
#define PERF_ERROR_IO              -6
#define PERF_ERROR_MEMORY          -7

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_ANALYZER_H