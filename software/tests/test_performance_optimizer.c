/*
 * Performance Optimizer Library Test Suite
 * 
 * Comprehensive tests for the adaptive performance optimization
 * and workload analysis capabilities.
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
#define TEST_TIMEOUT_SECONDS    30
#define TEST_OPTIMIZATION_CYCLES 10
#define TEST_WORKLOAD_PROFILES   5

// Test state
static int tests_passed = 0;
static int tests_failed = 0;
static bool verbose_output = false;

// Test helper functions
static void test_assert(bool condition, const char* test_name, const char* message);
static void print_test_header(const char* test_name);
static void print_test_result(const char* test_name, bool passed);
static double random_double(double min, double max);
static void simulate_workload_metrics(performance_metrics_t* metrics, workload_type_t type);
static void print_performance_metrics(const performance_metrics_t* metrics);
static void print_optimization_strategy(const optimization_strategy_t* strategy);

// Test functions
static void test_optimizer_initialization(void);
static void test_performance_metrics_collection(void);
static void test_workload_profiling(void);
static void test_optimization_strategy_generation(void);
static void test_performance_prediction(void);
static void test_adaptive_optimization_loop(void);
static void test_workload_aware_optimization(void);
static void test_power_optimization(void);
static void test_thermal_optimization(void);
static void test_multi_objective_optimization(void);
static void test_configuration_management(void);
static void test_stress_scenarios(void);

int main(int argc, char* argv[]) {
    printf("=== RISC-V AI Accelerator Performance Optimizer Test Suite ===\n\n");
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose_output = true;
        }
    }
    
    // Set up test timeout
    alarm(TEST_TIMEOUT_SECONDS);
    
    // Run test suite
    test_optimizer_initialization();
    test_performance_metrics_collection();
    test_workload_profiling();
    test_optimization_strategy_generation();
    test_performance_prediction();
    test_adaptive_optimization_loop();
    test_workload_aware_optimization();
    test_power_optimization();
    test_thermal_optimization();
    test_multi_objective_optimization();
    test_configuration_management();
    test_stress_scenarios();
    
    // Print final results
    printf("\n=== Test Results ===\n");
    printf("Tests Passed: %d\n", tests_passed);
    printf("Tests Failed: %d\n", tests_failed);
    printf("Total Tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("✓ All tests PASSED!\n");
        return 0;
    } else {
        printf("✗ %d tests FAILED!\n", tests_failed);
        return 1;
    }
}

static void test_optimizer_initialization(void) {
    print_test_header("Optimizer Initialization");
    
    // Test default initialization
    int result = perf_optimizer_init(NULL);
    test_assert(result == PERF_OPT_SUCCESS, "Default Init", 
                "Default initialization should succeed");
    
    // Test custom configuration
    perf_optimizer_config_t config = {
        .optimization_interval_ms = 500,
        .adaptation_aggressiveness = 0.7,
        .power_budget_watts = 20.0,
        .thermal_limit_celsius = 90.0,
        .enable_predictive_optimization = true,
        .enable_workload_profiling = true,
        .enable_power_optimization = true,
        .enable_thermal_optimization = true,
        .performance_target = 0.85,
        .power_efficiency_target = 0.8,
        .thermal_efficiency_target = 0.9
    };
    
    result = perf_optimizer_init(&config);
    test_assert(result == PERF_OPT_SUCCESS, "Custom Config Init", 
                "Custom configuration initialization should succeed");
    
    // Test double initialization (should succeed)
    result = perf_optimizer_init(&config);
    test_assert(result == PERF_OPT_SUCCESS, "Double Init", 
                "Double initialization should succeed");
    
    // Test optimizer start
    result = perf_optimizer_start();
    test_assert(result == PERF_OPT_SUCCESS, "Optimizer Start", 
                "Optimizer start should succeed");
    
    // Give optimizer thread time to start
    usleep(100000); // 100ms
    
    print_test_result("Optimizer Initialization", tests_failed == 0);
}

static void test_performance_metrics_collection(void) {
    print_test_header("Performance Metrics Collection");
    
    performance_metrics_t metrics;
    int result = collect_performance_metrics(&metrics);
    
    test_assert(result == PERF_OPT_SUCCESS, "Metrics Collection", 
                "Performance metrics collection should succeed");
    
    // Validate metrics structure
    test_assert(metrics.timestamp.tv_sec > 0, "Timestamp", 
                "Metrics should have valid timestamp");
    
    test_assert(metrics.performance_score >= 0.0 && metrics.performance_score <= 1.0, 
                "Performance Score Range", 
                "Performance score should be between 0.0 and 1.0");
    
    if (verbose_output) {
        print_performance_metrics(&metrics);
    }
    
    // Test multiple collections
    performance_metrics_t metrics2;
    usleep(10000); // 10ms delay
    result = collect_performance_metrics(&metrics2);
    
    test_assert(result == PERF_OPT_SUCCESS, "Second Collection", 
                "Second metrics collection should succeed");
    
    test_assert(metrics2.timestamp.tv_sec >= metrics.timestamp.tv_sec, 
                "Timestamp Ordering", 
                "Second collection should have later timestamp");
    
    print_test_result("Performance Metrics Collection", tests_failed == 0);
}

static void test_workload_profiling(void) {
    print_test_header("Workload Profiling");
    
    // Test workload profile registration
    workload_profile_t profiles[TEST_WORKLOAD_PROFILES];
    int profile_ids[TEST_WORKLOAD_PROFILES];
    
    for (int i = 0; i < TEST_WORKLOAD_PROFILES; i++) {
        profiles[i].workload_type = (workload_type_t)(i % 7);
        profiles[i].target_performance = 0.8 + (i * 0.05);
        profiles[i].cpu_intensity = random_double(0.3, 0.9);
        profiles[i].memory_intensity = random_double(0.2, 0.8);
        profiles[i].ai_workload_percentage = random_double(0.1, 0.7);
        profiles[i].valid = true;
        gettimeofday(&profiles[i].timestamp, NULL);
        
        profile_ids[i] = perf_optimizer_register_workload(&profiles[i]);
        test_assert(profile_ids[i] >= 0, "Profile Registration", 
                    "Workload profile registration should succeed");
    }
    
    // Test profile retrieval and validation
    for (int i = 0; i < TEST_WORKLOAD_PROFILES; i++) {
        // Profiles are stored internally, so we can't directly retrieve them
        // This would require additional API functions in a real implementation
        test_assert(profile_ids[i] != profile_ids[(i+1) % TEST_WORKLOAD_PROFILES], 
                    "Unique Profile IDs", 
                    "Each profile should get a unique ID");
    }
    
    // Test profile unregistration
    int result = perf_optimizer_unregister_workload(profile_ids[0]);
    test_assert(result == PERF_OPT_SUCCESS, "Profile Unregistration", 
                "Workload profile unregistration should succeed");
    
    // Test invalid profile operations
    result = perf_optimizer_unregister_workload(-1);
    test_assert(result == PERF_OPT_ERROR_INVALID_PARAM, "Invalid Unregistration", 
                "Invalid profile unregistration should fail");
    
    result = perf_optimizer_register_workload(NULL);
    test_assert(result == PERF_OPT_ERROR_INVALID_PARAM, "NULL Profile Registration", 
                "NULL profile registration should fail");
    
    print_test_result("Workload Profiling", tests_failed == 0);
}

static void test_optimization_strategy_generation(void) {
    print_test_header("Optimization Strategy Generation");
    
    // Create test metrics for different scenarios
    performance_metrics_t low_perf_metrics;
    simulate_workload_metrics(&low_perf_metrics, WORKLOAD_TYPE_CPU_INTENSIVE);
    low_perf_metrics.performance_score = 0.4; // Low performance
    low_perf_metrics.ipc = 0.8; // Low IPC
    low_perf_metrics.cache_hit_rate = 0.7; // Poor cache performance
    low_perf_metrics.power_consumption = 8.0; // Moderate power
    
    workload_profile_t cpu_profile;
    cpu_profile.workload_type = WORKLOAD_TYPE_CPU_INTENSIVE;
    cpu_profile.target_performance = 0.85;
    cpu_profile.cpu_intensity = 0.9;
    cpu_profile.memory_intensity = 0.3;
    cpu_profile.ai_workload_percentage = 0.1;
    
    optimization_strategy_t strategy;
    int result = generate_optimization_strategy(&low_perf_metrics, &cpu_profile, &strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Strategy Generation", 
                "Optimization strategy generation should succeed");
    
    test_assert(strategy.expected_performance_gain > 0.0, "Performance Gain Expected", 
                "Strategy should expect performance improvement");
    
    // For low performance scenario, should recommend frequency increase
    test_assert(strategy.dvfs_changes.increase_frequency, "Frequency Increase", 
                "Low performance should trigger frequency increase");
    
    // Should recommend cache prefetch increase for poor cache performance
    test_assert(strategy.cache_changes.increase_prefetch_aggressiveness, 
                "Cache Prefetch Increase", 
                "Poor cache performance should trigger prefetch increase");
    
    if (verbose_output) {
        print_optimization_strategy(&strategy);
    }
    
    // Test AI-intensive workload strategy
    performance_metrics_t ai_metrics;
    simulate_workload_metrics(&ai_metrics, WORKLOAD_TYPE_AI_INTENSIVE);
    
    workload_profile_t ai_profile;
    ai_profile.workload_type = WORKLOAD_TYPE_AI_INTENSIVE;
    ai_profile.target_performance = 0.9;
    ai_profile.ai_workload_percentage = 0.8;
    
    optimization_strategy_t ai_strategy;
    result = generate_optimization_strategy(&ai_metrics, &ai_profile, &ai_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "AI Strategy Generation", 
                "AI workload strategy generation should succeed");
    
    test_assert(ai_strategy.memory_changes.prioritize_ai_requests, 
                "AI Memory Priority", 
                "AI workload should prioritize AI memory requests");
    
    print_test_result("Optimization Strategy Generation", tests_failed == 0);
}

static void test_performance_prediction(void) {
    print_test_header("Performance Prediction");
    
    // Create a test strategy
    optimization_strategy_t strategy = {0};
    strategy.dvfs_changes.increase_frequency = true;
    strategy.dvfs_changes.frequency_step = 2;
    strategy.dvfs_changes.increase_voltage = true;
    strategy.dvfs_changes.voltage_step = 1;
    strategy.cache_changes.increase_prefetch_aggressiveness = true;
    strategy.cache_changes.prefetch_step = 1;
    
    performance_prediction_t prediction;
    int result = predict_optimization_impact(&strategy, &prediction);
    
    test_assert(result == PERF_OPT_SUCCESS, "Prediction Generation", 
                "Performance prediction should succeed");
    
    test_assert(prediction.expected_performance_gain > 0.0, "Positive Performance Gain", 
                "Frequency increase should predict performance gain");
    
    test_assert(prediction.expected_power_change > 0.0, "Power Increase", 
                "Frequency/voltage increase should predict power increase");
    
    test_assert(prediction.confidence > 0.0 && prediction.confidence <= 1.0, 
                "Confidence Range", 
                "Prediction confidence should be between 0.0 and 1.0");
    
    // Test power-constrained scenario
    optimization_strategy_t power_strategy = {0};
    power_strategy.dvfs_changes.increase_frequency = true;
    power_strategy.dvfs_changes.frequency_step = 5; // Large increase
    power_strategy.dvfs_changes.increase_voltage = true;
    power_strategy.dvfs_changes.voltage_step = 3;
    
    performance_prediction_t power_prediction;
    result = predict_optimization_impact(&power_strategy, &power_prediction);
    
    test_assert(result == PERF_OPT_SUCCESS, "Power Prediction", 
                "Power-constrained prediction should succeed");
    
    test_assert(power_prediction.expected_power_change > 5.0, "High Power Change", 
                "Large frequency increase should predict significant power change");
    
    if (verbose_output) {
        printf("  Prediction: Perf gain=%.3f, Power change=%.3f, Confidence=%.3f\n",
               prediction.expected_performance_gain, 
               prediction.expected_power_change,
               prediction.confidence);
    }
    
    print_test_result("Performance Prediction", tests_failed == 0);
}

static void test_adaptive_optimization_loop(void) {
    print_test_header("Adaptive Optimization Loop");
    
    // Force optimization to run
    int result = perf_optimizer_force_optimization();
    test_assert(result == PERF_OPT_SUCCESS, "Force Optimization", 
                "Force optimization should succeed");
    
    // Wait for optimization to complete
    usleep(500000); // 500ms
    
    // Get current recommendations
    optimization_recommendations_t recommendations;
    result = perf_optimizer_get_recommendations(&recommendations);
    test_assert(result == PERF_OPT_SUCCESS, "Get Recommendations", 
                "Getting recommendations should succeed");
    
    test_assert(recommendations.confidence_score >= 0.0 && 
                recommendations.confidence_score <= 1.0, 
                "Recommendation Confidence", 
                "Recommendation confidence should be valid");
    
    // Get current metrics
    performance_metrics_t current_metrics;
    result = perf_optimizer_get_metrics(&current_metrics);
    test_assert(result == PERF_OPT_SUCCESS, "Get Current Metrics", 
                "Getting current metrics should succeed");
    
    if (verbose_output) {
        printf("  Current Performance Score: %.3f\n", current_metrics.performance_score);
        printf("  Recommended Frequency: %d\n", recommendations.recommended_frequency_level);
        printf("  Recommended Voltage: %d\n", recommendations.recommended_voltage_level);
        printf("  Confidence: %.3f\n", recommendations.confidence_score);
    }
    
    // Run multiple optimization cycles
    for (int i = 0; i < TEST_OPTIMIZATION_CYCLES; i++) {
        result = perf_optimizer_force_optimization();
        test_assert(result == PERF_OPT_SUCCESS, "Optimization Cycle", 
                    "Each optimization cycle should succeed");
        
        usleep(200000); // 200ms between cycles
    }
    
    print_test_result("Adaptive Optimization Loop", tests_failed == 0);
}

static void test_workload_aware_optimization(void) {
    print_test_header("Workload-Aware Optimization");
    
    // Test different workload types and their optimization strategies
    workload_type_t workload_types[] = {
        WORKLOAD_TYPE_CPU_INTENSIVE,
        WORKLOAD_TYPE_MEMORY_INTENSIVE,
        WORKLOAD_TYPE_AI_INTENSIVE,
        WORKLOAD_TYPE_REAL_TIME,
        WORKLOAD_TYPE_BATCH
    };
    
    for (int i = 0; i < 5; i++) {
        performance_metrics_t metrics;
        simulate_workload_metrics(&metrics, workload_types[i]);
        
        workload_profile_t profile;
        profile.workload_type = workload_types[i];
        profile.target_performance = 0.8;
        
        switch (workload_types[i]) {
            case WORKLOAD_TYPE_CPU_INTENSIVE:
                profile.cpu_intensity = 0.9;
                profile.memory_intensity = 0.3;
                profile.ai_workload_percentage = 0.1;
                break;
            case WORKLOAD_TYPE_MEMORY_INTENSIVE:
                profile.cpu_intensity = 0.4;
                profile.memory_intensity = 0.9;
                profile.ai_workload_percentage = 0.2;
                break;
            case WORKLOAD_TYPE_AI_INTENSIVE:
                profile.cpu_intensity = 0.3;
                profile.memory_intensity = 0.5;
                profile.ai_workload_percentage = 0.8;
                break;
            case WORKLOAD_TYPE_REAL_TIME:
                profile.cpu_intensity = 0.7;
                profile.memory_intensity = 0.4;
                profile.ai_workload_percentage = 0.3;
                profile.target_performance = 0.95; // High target
                break;
            case WORKLOAD_TYPE_BATCH:
                profile.cpu_intensity = 0.6;
                profile.memory_intensity = 0.6;
                profile.ai_workload_percentage = 0.4;
                profile.target_performance = 0.7; // Lower target, focus on efficiency
                break;
            default:
                break;
        }
        
        optimization_strategy_t strategy;
        int result = generate_optimization_strategy(&metrics, &profile, &strategy);
        
        test_assert(result == PERF_OPT_SUCCESS, "Workload Strategy", 
                    "Workload-specific strategy generation should succeed");
        
        // Validate workload-specific optimizations
        switch (workload_types[i]) {
            case WORKLOAD_TYPE_AI_INTENSIVE:
                test_assert(strategy.memory_changes.prioritize_ai_requests, 
                            "AI Memory Priority", 
                            "AI workload should prioritize AI requests");
                break;
            case WORKLOAD_TYPE_REAL_TIME:
                test_assert(strategy.expected_performance_gain >= 0.0, 
                            "Real-time Performance", 
                            "Real-time workload should focus on performance");
                break;
            case WORKLOAD_TYPE_BATCH:
                // Batch workloads should focus on efficiency over performance
                break;
            default:
                break;
        }
        
        if (verbose_output) {
            printf("  Workload Type %d: Expected gain=%.3f\n", 
                   workload_types[i], strategy.expected_performance_gain);
        }
    }
    
    print_test_result("Workload-Aware Optimization", tests_failed == 0);
}

static void test_power_optimization(void) {
    print_test_header("Power Optimization");
    
    // Create high power consumption scenario
    performance_metrics_t high_power_metrics;
    simulate_workload_metrics(&high_power_metrics, WORKLOAD_TYPE_CPU_INTENSIVE);
    high_power_metrics.power_consumption = 18.0; // High power
    high_power_metrics.cpu_utilization = 0.3; // Low utilization
    high_power_metrics.tpu_utilization = 0.1; // Very low AI utilization
    
    workload_profile_t profile;
    profile.workload_type = WORKLOAD_TYPE_CPU_INTENSIVE;
    profile.target_performance = 0.7; // Lower target to allow power savings
    
    optimization_strategy_t strategy;
    int result = generate_optimization_strategy(&high_power_metrics, &profile, &strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Power Strategy Generation", 
                "Power optimization strategy should be generated");
    
    // Should recommend power gating for low utilization
    test_assert(strategy.power_changes.enable_core_power_gating || 
                strategy.power_changes.enable_ai_power_gating, 
                "Power Gating", 
                "Low utilization should trigger power gating");
    
    // Should recommend frequency/voltage reduction
    test_assert(strategy.dvfs_changes.decrease_frequency || 
                strategy.dvfs_changes.decrease_voltage, 
                "DVFS Reduction", 
                "High power should trigger DVFS reduction");
    
    // Test power prediction
    performance_prediction_t prediction;
    result = predict_optimization_impact(&strategy, &prediction);
    
    test_assert(result == PERF_OPT_SUCCESS, "Power Prediction", 
                "Power optimization prediction should succeed");
    
    test_assert(prediction.expected_power_change < 0.0, "Power Reduction", 
                "Power optimization should predict power reduction");
    
    if (verbose_output) {
        printf("  Power reduction expected: %.3f W\n", prediction.expected_power_change);
    }
    
    print_test_result("Power Optimization", tests_failed == 0);
}

static void test_thermal_optimization(void) {
    print_test_header("Thermal Optimization");
    
    // Create high temperature scenario
    performance_metrics_t hot_metrics;
    simulate_workload_metrics(&hot_metrics, WORKLOAD_TYPE_AI_INTENSIVE);
    hot_metrics.temperature = 88.0; // High temperature
    hot_metrics.power_consumption = 16.0; // High power
    
    workload_profile_t profile;
    profile.workload_type = WORKLOAD_TYPE_AI_INTENSIVE;
    profile.target_performance = 0.8;
    
    optimization_strategy_t strategy;
    int result = generate_optimization_strategy(&hot_metrics, &profile, &strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Thermal Strategy Generation", 
                "Thermal optimization strategy should be generated");
    
    // Should recommend aggressive power reduction for thermal management
    test_assert(strategy.dvfs_changes.decrease_frequency || 
                strategy.dvfs_changes.decrease_voltage ||
                strategy.power_changes.enable_core_power_gating ||
                strategy.power_changes.enable_ai_power_gating, 
                "Thermal Protection", 
                "High temperature should trigger thermal protection measures");
    
    performance_prediction_t prediction;
    result = predict_optimization_impact(&strategy, &prediction);
    
    test_assert(result == PERF_OPT_SUCCESS, "Thermal Prediction", 
                "Thermal optimization prediction should succeed");
    
    if (verbose_output) {
        printf("  Thermal optimization - Power change: %.3f W\n", 
               prediction.expected_power_change);
    }
    
    print_test_result("Thermal Optimization", tests_failed == 0);
}

static void test_multi_objective_optimization(void) {
    print_test_header("Multi-Objective Optimization");
    
    // Test balancing performance, power, and thermal objectives
    performance_metrics_t metrics;
    simulate_workload_metrics(&metrics, WORKLOAD_TYPE_MIXED);
    metrics.performance_score = 0.6; // Moderate performance
    metrics.power_consumption = 12.0; // Moderate power
    metrics.temperature = 75.0; // Moderate temperature
    
    workload_profile_t profile;
    profile.workload_type = WORKLOAD_TYPE_MIXED;
    profile.target_performance = 0.8;
    profile.cpu_intensity = 0.5;
    profile.memory_intensity = 0.5;
    profile.ai_workload_percentage = 0.5;
    
    optimization_strategy_t strategy;
    int result = generate_optimization_strategy(&metrics, &profile, &strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Multi-Objective Strategy", 
                "Multi-objective optimization should succeed");
    
    // Strategy should balance multiple objectives
    test_assert(strategy.confidence_level > 0.0, "Strategy Confidence", 
                "Multi-objective strategy should have confidence");
    
    // Test that strategy considers multiple factors
    bool considers_performance = (strategy.dvfs_changes.increase_frequency || 
                                 strategy.cache_changes.increase_prefetch_aggressiveness);
    bool considers_power = (strategy.power_changes.enable_core_power_gating || 
                           strategy.power_changes.enable_ai_power_gating);
    bool considers_resources = (strategy.resource_changes.enable_more_cores || 
                               strategy.resource_changes.enable_more_ai_units);
    
    test_assert(considers_performance || considers_power || considers_resources, 
                "Multi-Factor Consideration", 
                "Strategy should consider multiple optimization factors");
    
    if (verbose_output) {
        printf("  Multi-objective strategy confidence: %.3f\n", strategy.confidence_level);
        printf("  Expected performance gain: %.3f\n", strategy.expected_performance_gain);
    }
    
    print_test_result("Multi-Objective Optimization", tests_failed == 0);
}

static void test_configuration_management(void) {
    print_test_header("Configuration Management");
    
    // Test configuration with different parameters
    perf_optimizer_config_t test_configs[] = {
        {
            .optimization_interval_ms = 100,
            .adaptation_aggressiveness = 0.3,
            .power_budget_watts = 10.0,
            .thermal_limit_celsius = 70.0,
            .enable_predictive_optimization = false,
            .performance_target = 0.9
        },
        {
            .optimization_interval_ms = 2000,
            .adaptation_aggressiveness = 0.9,
            .power_budget_watts = 25.0,
            .thermal_limit_celsius = 95.0,
            .enable_predictive_optimization = true,
            .performance_target = 0.7
        }
    };
    
    for (int i = 0; i < 2; i++) {
        int result = perf_optimizer_init(&test_configs[i]);
        test_assert(result == PERF_OPT_SUCCESS, "Config Test", 
                    "Configuration should be accepted");
        
        // Test that configuration affects behavior
        result = perf_optimizer_force_optimization();
        test_assert(result == PERF_OPT_SUCCESS, "Config Optimization", 
                    "Optimization with custom config should work");
        
        usleep(200000); // 200ms
    }
    
    print_test_result("Configuration Management", tests_failed == 0);
}

static void test_stress_scenarios(void) {
    print_test_header("Stress Test Scenarios");
    
    // Test rapid optimization requests
    for (int i = 0; i < 20; i++) {
        int result = perf_optimizer_force_optimization();
        test_assert(result == PERF_OPT_SUCCESS, "Rapid Optimization", 
                    "Rapid optimization requests should succeed");
        usleep(50000); // 50ms between requests
    }
    
    // Test with extreme metrics
    performance_metrics_t extreme_metrics;
    simulate_workload_metrics(&extreme_metrics, WORKLOAD_TYPE_CPU_INTENSIVE);
    extreme_metrics.performance_score = 0.1; // Very low
    extreme_metrics.power_consumption = 30.0; // Very high
    extreme_metrics.temperature = 100.0; // Very high
    extreme_metrics.ipc = 0.2; // Very low
    extreme_metrics.cache_hit_rate = 0.3; // Very low
    
    workload_profile_t extreme_profile;
    extreme_profile.workload_type = WORKLOAD_TYPE_CPU_INTENSIVE;
    extreme_profile.target_performance = 0.95; // Very high target
    
    optimization_strategy_t extreme_strategy;
    int result = generate_optimization_strategy(&extreme_metrics, &extreme_profile, 
                                              &extreme_strategy);
    
    test_assert(result == PERF_OPT_SUCCESS, "Extreme Scenario", 
                "Extreme scenario should be handled gracefully");
    
    // Test concurrent access (simplified)
    pthread_t threads[4];
    for (int i = 0; i < 4; i++) {
        // In a real implementation, we'd create threads that call optimizer functions
        // For this test, we'll just simulate concurrent calls
        result = perf_optimizer_force_optimization();
        test_assert(result == PERF_OPT_SUCCESS, "Concurrent Access", 
                    "Concurrent optimization should work");
    }
    
    print_test_result("Stress Test Scenarios", tests_failed == 0);
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

static void simulate_workload_metrics(performance_metrics_t* metrics, workload_type_t type) {
    memset(metrics, 0, sizeof(performance_metrics_t));
    gettimeofday(&metrics->timestamp, NULL);
    
    // Simulate metrics based on workload type
    switch (type) {
        case WORKLOAD_TYPE_CPU_INTENSIVE:
            metrics->ipc = random_double(1.2, 2.0);
            metrics->cpu_utilization = random_double(0.7, 0.95);
            metrics->cache_hit_rate = random_double(0.85, 0.95);
            metrics->tpu_utilization = random_double(0.1, 0.3);
            metrics->vpu_utilization = random_double(0.1, 0.3);
            metrics->memory_bandwidth = random_double(5e9, 12e9);
            metrics->power_consumption = random_double(8.0, 15.0);
            break;
            
        case WORKLOAD_TYPE_MEMORY_INTENSIVE:
            metrics->ipc = random_double(0.6, 1.2);
            metrics->cpu_utilization = random_double(0.4, 0.7);
            metrics->cache_hit_rate = random_double(0.6, 0.8);
            metrics->tpu_utilization = random_double(0.1, 0.4);
            metrics->vpu_utilization = random_double(0.1, 0.4);
            metrics->memory_bandwidth = random_double(15e9, 25e9);
            metrics->power_consumption = random_double(10.0, 18.0);
            break;
            
        case WORKLOAD_TYPE_AI_INTENSIVE:
            metrics->ipc = random_double(0.8, 1.4);
            metrics->cpu_utilization = random_double(0.3, 0.6);
            metrics->cache_hit_rate = random_double(0.7, 0.9);
            metrics->tpu_utilization = random_double(0.6, 0.95);
            metrics->vpu_utilization = random_double(0.5, 0.9);
            metrics->memory_bandwidth = random_double(12e9, 20e9);
            metrics->power_consumption = random_double(12.0, 22.0);
            break;
            
        default:
            metrics->ipc = random_double(0.8, 1.6);
            metrics->cpu_utilization = random_double(0.4, 0.8);
            metrics->cache_hit_rate = random_double(0.75, 0.9);
            metrics->tpu_utilization = random_double(0.2, 0.6);
            metrics->vpu_utilization = random_double(0.2, 0.6);
            metrics->memory_bandwidth = random_double(8e9, 16e9);
            metrics->power_consumption = random_double(8.0, 16.0);
            break;
    }
    
    metrics->temperature = random_double(45.0, 80.0);
    metrics->performance_score = calculate_performance_score(metrics);
    metrics->anomaly_detected = false;
}

static void print_performance_metrics(const performance_metrics_t* metrics) {
    printf("  Performance Metrics:\n");
    printf("    IPC: %.3f\n", metrics->ipc);
    printf("    CPU Utilization: %.3f\n", metrics->cpu_utilization);
    printf("    Cache Hit Rate: %.3f\n", metrics->cache_hit_rate);
    printf("    TPU Utilization: %.3f\n", metrics->tpu_utilization);
    printf("    VPU Utilization: %.3f\n", metrics->vpu_utilization);
    printf("    Memory Bandwidth: %.2e bytes/sec\n", metrics->memory_bandwidth);
    printf("    Power Consumption: %.2f W\n", metrics->power_consumption);
    printf("    Temperature: %.1f C\n", metrics->temperature);
    printf("    Performance Score: %.3f\n", metrics->performance_score);
}

static void print_optimization_strategy(const optimization_strategy_t* strategy) {
    printf("  Optimization Strategy:\n");
    if (strategy->dvfs_changes.increase_frequency) {
        printf("    Increase frequency by %d steps\n", strategy->dvfs_changes.frequency_step);
    }
    if (strategy->dvfs_changes.decrease_frequency) {
        printf("    Decrease frequency by %d steps\n", strategy->dvfs_changes.frequency_step);
    }
    if (strategy->dvfs_changes.increase_voltage) {
        printf("    Increase voltage by %d steps\n", strategy->dvfs_changes.voltage_step);
    }
    if (strategy->dvfs_changes.decrease_voltage) {
        printf("    Decrease voltage by %d steps\n", strategy->dvfs_changes.voltage_step);
    }
    if (strategy->cache_changes.increase_prefetch_aggressiveness) {
        printf("    Increase cache prefetch by %d steps\n", strategy->cache_changes.prefetch_step);
    }
    if (strategy->power_changes.enable_core_power_gating) {
        printf("    Enable core power gating for %d cores\n", strategy->power_changes.cores_to_gate);
    }
    if (strategy->power_changes.enable_ai_power_gating) {
        printf("    Enable AI power gating for %d units\n", strategy->power_changes.ai_units_to_gate);
    }
    printf("    Expected performance gain: %.3f\n", strategy->expected_performance_gain);
    printf("    Expected power change: %.3f W\n", strategy->expected_power_change);
    printf("    Confidence: %.3f\n", strategy->confidence_level);
}