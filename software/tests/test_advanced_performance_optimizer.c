/*
 * Advanced Performance Optimizer Test Suite
 * 
 * Comprehensive tests for enhanced performance optimization algorithms,
 * machine learning-inspired optimization, and advanced workload analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#include "../lib/performance_optimizer.h"

// Test configuration
#define TEST_TIMEOUT_SECONDS    60
#define TEST_OPTIMIZATION_CYCLES 20
#define TEST_WORKLOAD_PROFILES   8

// Test state
static int tests_passed = 0;
static int tests_failed = 0;
static bool verbose_output = false;

// Test helper functions
static void test_assert(bool condition, const char* test_name, const char* message);
static void print_test_header(const char* test_name);
static void print_test_result(const char* test_name, bool passed);
static double random_double(double min, double max);
static void simulate_advanced_workload_metrics(performance_metrics_t* metrics, 
                                              workload_type_t type, double intensity);

// Advanced test functions
static void test_enhanced_workload_analysis(void);
static void test_advanced_performance_prediction(void);
static void test_genetic_algorithm_optimization(void);
static void test_reinforcement_learning_optimization(void);
static void test_multi_objective_optimization_detailed(void);
static void test_thermal_aware_optimization_advanced(void);
static void test_performance_report_generation(void);

int main(int argc, char* argv[]) {
    printf("=== Advanced RISC-V AI Accelerator Performance Optimizer Test Suite ===\n\n");
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose_output = true;
        }
    }
    
    // Set up test timeout
    alarm(TEST_TIMEOUT_SECONDS);
    
    // Initialize random seed
    srand(time(NULL));
    
    // Initialize optimizer with advanced configuration
    perf_optimizer_config_t advanced_config = {
        .optimization_interval_ms = 200,
        .adaptation_aggressiveness = 0.8,
        .power_budget_watts = 18.0,
        .thermal_limit_celsius = 85.0,
        .enable_predictive_optimization = true,
        .enable_workload_profiling = true,
        .enable_power_optimization = true,
        .enable_thermal_optimization = true,
        .enable_qos_optimization = true,
        .performance_target = 0.85,
        .power_efficiency_target = 0.8,
        .thermal_efficiency_target = 0.9
    };
    
    if (perf_optimizer_init(&advanced_config) != PERF_OPT_SUCCESS) {
        printf("✗ Failed to initialize performance optimizer\n");
        return 1;
    }
    
    if (perf_optimizer_start() != PERF_OPT_SUCCESS) {
        printf("✗ Failed to start performance optimizer\n");
        return 1;
    }
    
    // Give optimizer time to start
    usleep(100000);
    
    // Run advanced test suite
    test_enhanced_workload_analysis();
    test_advanced_performance_prediction();
    test_genetic_algorithm_optimization();
    test_reinforcement_learning_optimization();
    test_multi_objective_optimization_detailed();
    test_thermal_aware_optimization_advanced();
    test_performance_report_generation();
    
    // Stop optimizer
    perf_optimizer_stop();
    
    // Print final results
    printf("\n=== Advanced Test Results ===\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Total Tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("✓ All advanced tests PASSED!\n");
        return 0;
    } else {
        printf("✗ %d advanced tests FAILED!\n", tests_failed);
        return 1;
    }
}

static void test_enhanced_workload_analysis(void) {
    print_test_header("Enhanced Workload Analysis");
    
    // Test comprehensive workload characterization
    workload_type_t test_workloads[] = {
        WORKLOAD_TYPE_CPU_INTENSIVE,
        WORKLOAD_TYPE_MEMORY_INTENSIVE,
        WORKLOAD_TYPE_AI_INTENSIVE,
        WORKLOAD_TYPE_MIXED
    };
    
    for (int i = 0; i < 4; i++) {
        performance_metrics_t metrics;
        simulate_advanced_workload_metrics(&metrics, test_workloads[i], 0.8);
        
        workload_profile_t profile;
        int result = analyze_workload_pattern(&metrics, 1, &profile);
        
        test_assert(result == PERF_OPT_SUCCESS, "Workload Analysis", 
                    "Advanced workload analysis should succeed");
        
        if (verbose_output) {
            printf("  Workload %d: CPU=%.3f, Memory=%.3f, AI=%.3f, Target=%.3f\n",
                   test_workloads[i], profile.cpu_intensity, profile.memory_intensity,
                   profile.ai_workload_percentage, profile.target_performance);
        }
    }
    
    print_test_result("Enhanced Workload Analysis", tests_failed == 0);
}

static void test_advanced_performance_prediction(void) {
    print_test_header("Advanced Performance Prediction");
    
    performance_metrics_t metrics;
    simulate_advanced_workload_metrics(&metrics, WORKLOAD_TYPE_CPU_INTENSIVE, 0.7);
    
    optimization_strategy_t strategy = {
        .dvfs_changes = {.increase_frequency = true, .frequency_step = 2},
        .confidence_level = 0.8
    };
    
    performance_prediction_t prediction;
    int result = predict_optimization_impact(&strategy, &prediction);
    
    test_assert(result == PERF_OPT_SUCCESS, "Prediction Generation", 
                "Performance prediction should succeed");
    
    test_assert(prediction.confidence >= 0.1 && prediction.confidence <= 1.0, 
                "Prediction Confidence", 
                "Prediction confidence should be within valid range");
    
    test_assert(prediction.expected_performance_gain > 0.0, 
                "Performance Gain", 
                "Frequency increase should predict performance gain");
    
    print_test_result("Advanced Performance Prediction", tests_failed == 0);
}

static void test_genetic_algorithm_optimization(void) {
    print_test_header("Genetic Algorithm Optimization");
    
    performance_metrics_t target_metrics;
    simulate_advanced_workload_metrics(&target_metrics, WORKLOAD_TYPE_MIXED, 0.6);
    target_metrics.performance_score = 0.5; // Low performance requiring optimization
    
    optimization_strategy_t best_strategy;
    int result = genetic_algorithm_optimization(&target_metrics, &best_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "GA Optimization", 
                "Genetic algorithm optimization should succeed");
    
    test_assert(best_strategy.confidence_level > 0.0, "GA Strategy Confidence", 
                "GA should produce strategy with confidence");
    
    bool has_optimization = best_strategy.dvfs_changes.increase_frequency ||
                           best_strategy.dvfs_changes.increase_voltage ||
                           best_strategy.cache_changes.increase_prefetch_aggressiveness ||
                           best_strategy.resource_changes.enable_more_ai_units;
    
    test_assert(has_optimization, "GA Strategy Content", 
                "GA should produce strategy with optimization actions");
    
    print_test_result("Genetic Algorithm Optimization", tests_failed == 0);
}

static void test_reinforcement_learning_optimization(void) {
    print_test_header("Reinforcement Learning Optimization");
    
    performance_metrics_t rl_metrics = {
        .performance_score = 0.4, 
        .power_consumption = 8.0, 
        .cpu_utilization = 0.3
    };
    
    optimization_strategy_t rl_strategy;
    int result = reinforcement_learning_optimization(&rl_metrics, &rl_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "RL Optimization", 
                "Reinforcement learning optimization should succeed");
    
    test_assert(rl_strategy.confidence_level >= 0.1 && rl_strategy.confidence_level <= 1.0, 
                "RL Confidence", 
                "RL strategy should have valid confidence");
    
    print_test_result("Reinforcement Learning Optimization", tests_failed == 0);
}

static void test_multi_objective_optimization_detailed(void) {
    print_test_header("Multi-Objective Optimization");
    
    performance_metrics_t test_metrics;
    simulate_advanced_workload_metrics(&test_metrics, WORKLOAD_TYPE_MIXED, 0.7);
    test_metrics.performance_score = 0.6;
    test_metrics.power_consumption = 14.0;
    
    optimization_strategy_t mo_strategy;
    int result = multi_objective_optimization(&test_metrics, 0.4, 0.3, 0.3, &mo_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "MO Optimization", 
                "Multi-objective optimization should succeed");
    
    test_assert(mo_strategy.confidence_level > 0.0, "MO Confidence", 
                "MO strategy should have confidence");
    
    print_test_result("Multi-Objective Optimization", tests_failed == 0);
}

static void test_thermal_aware_optimization_advanced(void) {
    print_test_header("Advanced Thermal-Aware Optimization");
    
    performance_metrics_t thermal_metrics;
    simulate_advanced_workload_metrics(&thermal_metrics, WORKLOAD_TYPE_AI_INTENSIVE, 0.8);
    thermal_metrics.temperature = 85.0; // High temperature
    thermal_metrics.power_consumption = 16.0;
    
    workload_profile_t thermal_profile = {
        .workload_type = WORKLOAD_TYPE_AI_INTENSIVE,
        .target_performance = 0.8
    };
    
    optimization_strategy_t thermal_strategy;
    int result = generate_optimization_strategy(&thermal_metrics, &thermal_profile, 
                                              &thermal_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Thermal Strategy Generation", 
                "Thermal optimization strategy should be generated");
    
    bool thermal_protection = thermal_strategy.dvfs_changes.decrease_frequency ||
                            thermal_strategy.dvfs_changes.decrease_voltage ||
                            thermal_strategy.power_changes.enable_core_power_gating ||
                            thermal_strategy.power_changes.enable_ai_power_gating;
    
    test_assert(thermal_protection, "Thermal Protection", 
                "High temperature should trigger thermal protection");
    
    print_test_result("Advanced Thermal-Aware Optimization", tests_failed == 0);
}

static void test_performance_report_generation(void) {
    print_test_header("Performance Report Generation");
    
    const char* report_file = "/tmp/perf_optimizer_test_report.txt";
    int result = generate_performance_report(report_file);
    
    test_assert(result == PERF_OPT_SUCCESS, "Report Generation", 
                "Performance report generation should succeed");
    
    FILE* fp = fopen(report_file, "r");
    test_assert(fp != NULL, "Report File Creation", 
                "Report file should be created");
    
    if (fp) {
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        test_assert(file_size > 100, "Report Content", 
                    "Report should have substantial content");
        fclose(fp);
        unlink(report_file);
    }
    
    print_test_result("Performance Report Generation", tests_failed == 0);
}

// Helper function implementations
static void test_assert(bool condition, const char* test_name, const char* message) {
    if (condition) {
        tests_passed++;
        if (verbose_output) {
            printf("  ✓ %s: %s\n", test_name, message);
        }
    } else {
        tests_failed++;
        printf("  ✗ %s: %s\n", test_name, message);
    }
}

static void print_test_header(const char* test_name) {
    printf("\n--- %s ---\n", test_name);
}

static void print_test_result(const char* test_name, bool passed) {
    if (passed) {
        printf("✓ %s completed successfully\n", test_name);
    } else {
        printf("✗ %s failed\n", test_name);
    }
}

static double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static void simulate_advanced_workload_metrics(performance_metrics_t* metrics, 
                                              workload_type_t type, double intensity) {
    memset(metrics, 0, sizeof(performance_metrics_t));
    gettimeofday(&metrics->timestamp, NULL);
    
    // Base metrics
    metrics->performance_score = 0.5 + intensity * 0.4;
    metrics->power_consumption = 8.0 + intensity * 8.0;
    
    // Type-specific metrics
    switch (type) {
        case WORKLOAD_TYPE_CPU_INTENSIVE:
            metrics->ipc = 1.0 + intensity * 0.8;
            metrics->cpu_utilization = 0.6 + intensity * 0.35;
            metrics->cache_hit_rate = 0.85 + intensity * 0.1;
            metrics->tpu_utilization = 0.1 + intensity * 0.2;
            metrics->vpu_utilization = 0.1 + intensity * 0.2;
            metrics->memory_bandwidth = 8e9 + intensity * 7e9;
            break;
            
        case WORKLOAD_TYPE_MEMORY_INTENSIVE:
            metrics->ipc = 0.6 + intensity * 0.4;
            metrics->cpu_utilization = 0.4 + intensity * 0.3;
            metrics->cache_hit_rate = 0.6 + intensity * 0.25;
            metrics->tpu_utilization = 0.1 + intensity * 0.3;
            metrics->vpu_utilization = 0.1 + intensity * 0.3;
            metrics->memory_bandwidth = 15e9 + intensity * 10e9;
            metrics->memory_latency = 50.0 + intensity * 30.0;
            break;
            
        case WORKLOAD_TYPE_AI_INTENSIVE:
            metrics->ipc = 0.8 + intensity * 0.4;
            metrics->cpu_utilization = 0.3 + intensity * 0.4;
            metrics->cache_hit_rate = 0.8 + intensity * 0.15;
            metrics->tpu_utilization = 0.5 + intensity * 0.45;
            metrics->vpu_utilization = 0.4 + intensity * 0.5;
            metrics->memory_bandwidth = 12e9 + intensity * 8e9;
            metrics->tpu_operations = 1e9 + intensity * 4e9;
            metrics->vpu_operations = 0.8e9 + intensity * 3e9;
            break;
            
        default: // MIXED
            metrics->ipc = 0.8 + intensity * 0.5;
            metrics->cpu_utilization = 0.5 + intensity * 0.4;
            metrics->cache_hit_rate = 0.8 + intensity * 0.15;
            metrics->tpu_utilization = 0.3 + intensity * 0.4;
            metrics->vpu_utilization = 0.3 + intensity * 0.4;
            metrics->memory_bandwidth = 10e9 + intensity * 8e9;
            metrics->tpu_operations = 0.5e9 + intensity * 2e9;
            metrics->vpu_operations = 0.4e9 + intensity * 2e9;
            break;
    }
    
    // Common derived metrics
    metrics->memory_accesses = (uint64_t)(metrics->memory_bandwidth / 64.0 * 1000000);
    metrics->noc_packets = metrics->memory_accesses / 8;
    metrics->average_power = metrics->power_consumption * (0.9 + random_double(0.0, 0.2));
    
    // Add realistic noise
    metrics->performance_score += random_double(-0.05, 0.05);
    metrics->power_consumption += random_double(-0.5, 0.5);
    
    // Ensure bounds
    if (metrics->performance_score > 1.0) metrics->performance_score = 1.0;
    if (metrics->performance_score < 0.0) metrics->performance_score = 0.0;
    if (metrics->power_consumption < 1.0) metrics->power_consumption = 1.0;
}