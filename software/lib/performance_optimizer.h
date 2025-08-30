/*
 * Performance Optimizer Library Header
 * 
 * Provides adaptive performance optimization and real-time tuning
 * for the RISC-V AI accelerator.
 */

#ifndef PERFORMANCE_OPTIMIZER_H
#define PERFORMANCE_OPTIMIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits
#define MAX_WORKLOAD_PROFILES       16
#define MAX_OPTIMIZATION_HISTORY    100
#define MAX_PERFORMANCE_SAMPLES     1000

// Workload types
typedef enum {
    WORKLOAD_TYPE_CPU_INTENSIVE = 0,
    WORKLOAD_TYPE_MEMORY_INTENSIVE = 1,
    WORKLOAD_TYPE_AI_INTENSIVE = 2,
    WORKLOAD_TYPE_MIXED = 3,
    WORKLOAD_TYPE_REAL_TIME = 4,
    WORKLOAD_TYPE_BATCH = 5,
    WORKLOAD_TYPE_INTERACTIVE = 6
} workload_type_t;

// Performance metrics structure
typedef struct {
    struct timeval timestamp;
    
    // CPU metrics
    uint64_t cpu_cycles;
    uint64_t cpu_instructions;
    double ipc;
    double cpu_utilization;
    double cache_hit_rate;
    
    // AI accelerator metrics
    uint64_t tpu_cycles;
    uint64_t tpu_operations;
    uint64_t vpu_cycles;
    uint64_t vpu_operations;
    double tpu_utilization;
    double vpu_utilization;
    
    // Memory metrics
    uint64_t memory_accesses;
    double memory_bandwidth;
    double memory_latency;
    
    // NoC metrics
    uint64_t noc_packets;
    double noc_utilization;
    double noc_latency;
    
    // Power and thermal metrics
    double power_consumption;
    double average_power;
    double temperature;
    
    // Overall metrics
    double performance_score;
    bool anomaly_detected;
} performance_metrics_t;

// Workload profile structure
typedef struct {
    int profile_id;
    bool valid;
    struct timeval timestamp;
    
    workload_type_t workload_type;
    double target_performance;
    double cpu_intensity;
    double memory_intensity;
    double ai_workload_percentage;
    
    // Performance history for this workload
    performance_metrics_t performance_history[MAX_PERFORMANCE_SAMPLES];
    int performance_history_count;
    
    // Workload characteristics
    double average_ipc;
    double average_cache_hit_rate;
    double average_power;
    double performance_variance;
} workload_profile_t;

// DVFS optimization changes
typedef struct {
    bool increase_frequency;
    bool decrease_frequency;
    int frequency_step;
    
    bool increase_voltage;
    bool decrease_voltage;
    int voltage_step;
} dvfs_changes_t;

// Cache optimization changes
typedef struct {
    bool increase_prefetch_aggressiveness;
    bool decrease_prefetch_aggressiveness;
    int prefetch_step;
    
    bool enable_cache_partitioning;
    int cache_partition_policy;
} cache_changes_t;

// Memory optimization changes
typedef struct {
    bool prioritize_cpu_requests;
    bool prioritize_ai_requests;
    bool prioritize_memory_requests;
    int memory_scheduler_policy;
    
    bool adjust_memory_bandwidth;
    double memory_bandwidth_allocation;
} memory_changes_t;

// Resource allocation changes
typedef struct {
    bool enable_more_cores;
    bool disable_cores;
    int core_count_change;
    
    bool enable_more_ai_units;
    bool disable_ai_units;
    int ai_unit_count_change;
} resource_changes_t;

// Power management changes
typedef struct {
    bool enable_core_power_gating;
    bool enable_ai_power_gating;
    int cores_to_gate;
    int ai_units_to_gate;
    
    bool adjust_power_budget;
    double power_budget_change;
} power_changes_t;

// NoC optimization changes
typedef struct {
    int routing_policy;
    bool adjust_bandwidth_allocation;
    double bandwidth_allocation;
} noc_changes_t;

// Optimization strategy
typedef struct {
    struct timeval timestamp;
    
    dvfs_changes_t dvfs_changes;
    cache_changes_t cache_changes;
    memory_changes_t memory_changes;
    resource_changes_t resource_changes;
    power_changes_t power_changes;
    noc_changes_t noc_changes;
    
    double confidence_level;
    double expected_performance_gain;
    double expected_power_change;
} optimization_strategy_t;

// Performance prediction
typedef struct {
    struct timeval timestamp;
    
    double expected_performance_gain;
    double expected_power_change;
    double expected_temperature_change;
    
    double confidence;
    bool power_constraint_violated;
    bool thermal_risk;
    bool resource_constraint_violated;
} performance_prediction_t;

// Optimization results
typedef struct {
    struct timeval timestamp;
    
    double performance_improvement;
    double power_change;
    double temperature_change;
    
    bool successful;
    double confidence;
    
    performance_metrics_t before_metrics;
    performance_metrics_t after_metrics;
} optimization_results_t;

// Optimization history entry
typedef struct {
    struct timeval timestamp;
    optimization_strategy_t strategy;
    optimization_results_t results;
} optimization_history_entry_t;

// Optimization recommendations
typedef struct {
    struct timeval timestamp;
    
    int recommended_frequency_level;
    int recommended_voltage_level;
    int cache_prefetch_aggressiveness;
    int memory_scheduler_policy;
    int noc_routing_policy;
    
    bool enable_core_power_gating;
    bool enable_ai_power_gating;
    
    double confidence_score;
    char recommendation_reason[256];
} optimization_recommendations_t;

// Optimizer configuration
typedef struct {
    uint32_t optimization_interval_ms;
    double adaptation_aggressiveness;
    double power_budget_watts;
    double thermal_limit_celsius;
    
    bool enable_predictive_optimization;
    bool enable_workload_profiling;
    bool enable_power_optimization;
    bool enable_thermal_optimization;
    bool enable_qos_optimization;
    
    double performance_target;
    double power_efficiency_target;
    double thermal_efficiency_target;
} perf_optimizer_config_t;

// Main optimizer state
typedef struct {
    bool initialized;
    bool force_optimization;
    
    perf_optimizer_config_t config;
    
    workload_profile_t workload_profiles[MAX_WORKLOAD_PROFILES];
    optimization_history_entry_t optimization_history[MAX_OPTIMIZATION_HISTORY];
    int optimization_history_count;
    
    performance_metrics_t current_metrics;
    optimization_recommendations_t current_recommendations;
    
    // Statistics
    uint64_t total_optimizations;
    uint64_t successful_optimizations;
    double average_performance_gain;
    double average_power_savings;
} perf_optimizer_t;

// Function prototypes

// Initialization and control
int perf_optimizer_init(perf_optimizer_config_t* config);
int perf_optimizer_start(void);
int perf_optimizer_stop(void);
void perf_optimizer_cleanup(void);

// Workload management
int perf_optimizer_register_workload(workload_profile_t* profile);
int perf_optimizer_unregister_workload(int profile_id);
int perf_optimizer_update_workload(int profile_id, workload_profile_t* profile);

// Performance monitoring and optimization
int collect_performance_metrics(performance_metrics_t* metrics);
int generate_optimization_strategy(performance_metrics_t* metrics, 
                                 workload_profile_t* profile,
                                 optimization_strategy_t* strategy);
int predict_optimization_impact(optimization_strategy_t* strategy,
                              performance_prediction_t* prediction);
int apply_optimization_strategy(optimization_strategy_t* strategy);
int validate_optimization_results(optimization_results_t* results);

// Recommendations and reporting
int perf_optimizer_get_recommendations(optimization_recommendations_t* recommendations);
int perf_optimizer_get_metrics(performance_metrics_t* metrics);
int perf_optimizer_get_history(optimization_history_entry_t* history, int max_entries);
int perf_optimizer_force_optimization(void);

// Analysis and reporting
int update_recommendations(optimization_strategy_t* strategy, 
                         optimization_results_t* results);
int generate_performance_report(const char* output_file);
int export_optimization_history(const char* output_file);

// Utility functions
double calculate_performance_score(performance_metrics_t* metrics);
double calculate_power_efficiency(performance_metrics_t* metrics);
double calculate_thermal_efficiency(performance_metrics_t* metrics);
int detect_performance_anomalies(performance_metrics_t* metrics);

// Workload analysis
int analyze_workload_pattern(performance_metrics_t* samples, int sample_count,
                           workload_profile_t* profile);
int predict_workload_behavior(workload_profile_t* profile, 
                            performance_metrics_t* prediction);
int classify_workload_type(performance_metrics_t* metrics, workload_type_t* type);

// Advanced optimization algorithms
int genetic_algorithm_optimization(performance_metrics_t* target_metrics,
                                 optimization_strategy_t* best_strategy);
int reinforcement_learning_optimization(performance_metrics_t* current_metrics,
                                       optimization_strategy_t* strategy);
int multi_objective_optimization(performance_metrics_t* metrics,
                               double performance_weight,
                               double power_weight,
                               double thermal_weight,
                               optimization_strategy_t* strategy);

// Machine learning integration
int train_performance_model(performance_metrics_t* training_data, int data_count);
int predict_performance_ml(optimization_strategy_t* strategy, 
                         performance_prediction_t* prediction);
int update_ml_model(optimization_results_t* results);

// Error codes
#define PERF_OPT_SUCCESS                0
#define PERF_OPT_ERROR_INIT            -1
#define PERF_OPT_ERROR_INVALID_PARAM   -2
#define PERF_OPT_ERROR_NO_SPACE        -3
#define PERF_OPT_ERROR_NOT_FOUND       -4
#define PERF_OPT_ERROR_THREAD          -5
#define PERF_OPT_ERROR_COLLECTION      -6
#define PERF_OPT_ERROR_PREDICTION      -7
#define PERF_OPT_ERROR_APPLICATION     -8
#define PERF_OPT_ERROR_VALIDATION      -9
#define PERF_OPT_ERROR_IO             -10

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_OPTIMIZER_H