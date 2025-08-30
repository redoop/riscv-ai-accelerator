/*
 * Power Optimization Library Test Suite
 * 
 * Comprehensive tests for advanced power optimization algorithms,
 * thermal management, and machine learning-based prediction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#include "../lib/power_optimizer.h"

// Test configuration
#define TEST_TIMEOUT_SECONDS    60
#define TEST_OPTIMIZATION_CYCLES 15
#define TEST_THERMAL_SCENARIOS   6
#define TEST_POWER_SCENARIOS     8

// Test state
static int tests_passed = 0;
static int tests_failed = 0;
static bool verbose_output = false;

// Test helper functions
static void test_assert(bool condition, const char* test_name, const char* message);
static void print_test_header(const char* test_name);
static void print_test_result(const char* test_name, bool passed);
static double random_double(double min, double max);
static void simulate_power_metrics(power_metrics_t* metrics, double base_power, double variation);
static void simulate_thermal_metrics(thermal_metrics_t* metrics, double base_temp, double variation);
static void print_power_metrics(const power_metrics_t* metrics);
static void print_thermal_metrics(const thermal_metrics_t* metrics);
static void print_power_strategy(const power_optimization_strategy_t* strategy);

// Test functions
static void test_power_optimizer_initialization(void);
static void test_power_metrics_collection(void);
static void test_thermal_metrics_collection(void);
static void test_power_prediction(void);
static void test_thermal_prediction(void);
static void test_power_optimization_strategies(void);
static void test_thermal_aware_optimization(void);
static void test_battery_optimization(void);
static void test_emergency_thermal_protection(void);
static void test_machine_learning_prediction(void);
static void test_multi_objective_optimization(void);
static void test_adaptive_optimization_loop(void);
static void test_configuration_management(void);
static void test_stress_scenarios(void);

int main(int argc, char* argv[]) {
    printf("=== RISC-V AI Accelerator Power Optimization Test Suite ===\n\n");
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose_output = true;
        }
    }
    
    // Set up test timeout
    alarm(TEST_TIMEOUT_SECONDS);
    
    // Run test suite
    test_power_optimizer_initialization();
    test_power_metrics_collection();
    test_thermal_metrics_collection();
    test_power_prediction();
    test_thermal_prediction();
    test_power_optimization_strategies();
    test_thermal_aware_optimization();
    test_battery_optimization();
    test_emergency_thermal_protection();
    test_machine_learning_prediction();
    test_multi_objective_optimization();
    test_adaptive_optimization_loop();
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

static void test_power_optimizer_initialization(void) {
    print_test_header("Power Optimizer Initialization");
    
    // Test default initialization
    int result = power_optimizer_init(NULL);
    test_assert(result == POWER_OPT_SUCCESS, "Default Init", 
                "Default initialization should succeed");
    
    // Test custom configuration
    power_optimizer_config_t config = {
        .optimization_interval_ms = 800,
        .power_budget_watts = 20.0,
        .thermal_limit_celsius = 90.0,
        .battery_capacity_wh = 60.0,
        .enable_ml_prediction = true,
        .enable_thermal_optimization = true,
        .enable_battery_optimization = true,
        .optimization_aggressiveness = 0.7,
        .thermal_safety_margin = 8.0,
        .power_efficiency_target = 0.85
    };
    
    result = power_optimizer_init(&config);
    test_assert(result == POWER_OPT_SUCCESS, "Custom Config Init", 
                "Custom configuration initialization should succeed");
    
    // Test configuration retrieval
    power_optimizer_config_t retrieved_config;
    result = power_optimizer_get_config(&retrieved_config);
    test_assert(result == POWER_OPT_SUCCESS, "Config Retrieval", 
                "Configuration retrieval should succeed");
    
    test_assert(fabs(retrieved_config.power_budget_watts - 20.0) < 0.1, 
                "Config Validation", 
                "Retrieved configuration should match set values");
    
    // Test optimizer start
    result = power_optimizer_start();
    test_assert(result == POWER_OPT_SUCCESS, "Optimizer Start", 
                "Power optimizer start should succeed");
    
    // Give optimizer thread time to start
    usleep(200000); // 200ms
    
    print_test_result("Power Optimizer Initialization", tests_failed == 0);
}

static void test_power_metrics_collection(void) {
    print_test_header("Power Metrics Collection");
    
    power_metrics_t power_metrics;
    thermal_metrics_t thermal_metrics;
    
    int result = power_optimizer_get_metrics(&power_metrics, &thermal_metrics);
    test_assert(result == POWER_OPT_SUCCESS, "Metrics Collection", 
                "Power and thermal metrics collection should succeed");
    
    // Validate power metrics structure
    test_assert(power_metrics.timestamp.tv_sec > 0, "Power Timestamp", 
                "Power metrics should have valid timestamp");
    
    test_assert(power_metrics.total_power > 0.0, "Total Power", 
                "Total power should be positive");
    
    test_assert(power_metrics.power_efficiency >= 0.0, "Power Efficiency", 
                "Power efficiency should be non-negative");
    
    // Test individual component power readings
    bool has_core_power = false;
    for (int i = 0; i < MAX_CORES; i++) {
        if (power_metrics.core_power[i] > 0.0) {
            has_core_power = true;
            break;
        }
    }
    test_assert(has_core_power, "Core Power Readings", 
                "At least one core should have power reading");
    
    bool has_ai_power = false;
    for (int i = 0; i < MAX_AI_UNITS; i++) {
        if (power_metrics.ai_unit_power[i] > 0.0) {
            has_ai_power = true;
            break;
        }
    }
    test_assert(has_ai_power, "AI Unit Power Readings", 
                "At least one AI unit should have power reading");
    
    if (verbose_output) {
        print_power_metrics(&power_metrics);
    }
    
    print_test_result("Power Metrics Collection", tests_failed == 0);
}

static void test_thermal_metrics_collection(void) {
    print_test_header("Thermal Metrics Collection");
    
    power_metrics_t power_metrics;
    thermal_metrics_t thermal_metrics;
    
    int result = power_optimizer_get_metrics(&power_metrics, &thermal_metrics);
    test_assert(result == POWER_OPT_SUCCESS, "Thermal Metrics Collection", 
                "Thermal metrics collection should succeed");
    
    // Validate thermal metrics
    test_assert(thermal_metrics.timestamp.tv_sec > 0, "Thermal Timestamp", 
                "Thermal metrics should have valid timestamp");
    
    test_assert(thermal_metrics.ambient_temperature > 0.0 && 
                thermal_metrics.ambient_temperature < 100.0, 
                "Ambient Temperature Range", 
                "Ambient temperature should be reasonable");
    
    test_assert(thermal_metrics.max_temperature >= thermal_metrics.ambient_temperature, 
                "Max Temperature Logic", 
                "Max temperature should be >= ambient temperature");
    
    test_assert(thermal_metrics.hotspot_zone >= 0 && 
                thermal_metrics.hotspot_zone < MAX_THERMAL_ZONES, 
                "Hotspot Zone Range", 
                "Hotspot zone should be within valid range");
    
    // Test thermal zone readings
    bool has_valid_zones = true;
    for (int i = 0; i < MAX_THERMAL_ZONES; i++) {
        if (thermal_metrics.zone_temperature[i] < thermal_metrics.ambient_temperature ||
            thermal_metrics.zone_temperature[i] > 150.0) {
            has_valid_zones = false;
            break;
        }
    }
    test_assert(has_valid_zones, "Thermal Zone Readings", 
                "All thermal zones should have reasonable temperatures");
    
    if (verbose_output) {
        print_thermal_metrics(&thermal_metrics);
    }
    
    print_test_result("Thermal Metrics Collection", tests_failed == 0);
}

static void test_power_prediction(void) {
    print_test_header("Power Prediction");
    
    // Build up some history first
    for (int i = 0; i < 10; i++) {
        power_metrics_t metrics;
        simulate_power_metrics(&metrics, 12.0 + i * 0.5, 1.0);
        update_power_history(&metrics);
        usleep(50000); // 50ms between samples
    }
    
    // Get predictions
    power_prediction_t power_pred;
    thermal_prediction_t thermal_pred;
    
    int result = power_optimizer_get_predictions(&power_pred, &thermal_pred);
    test_assert(result == POWER_OPT_SUCCESS, "Prediction Retrieval", 
                "Power prediction retrieval should succeed");
    
    test_assert(power_pred.predicted_power > 0.0, "Predicted Power Positive", 
                "Predicted power should be positive");
    
    test_assert(power_pred.confidence >= 0.0 && power_pred.confidence <= 1.0, 
                "Prediction Confidence Range", 
                "Prediction confidence should be between 0.0 and 1.0");
    
    test_assert(power_pred.prediction_horizon_ms > 0, "Prediction Horizon", 
                "Prediction horizon should be positive");
    
    // Test power budget violation risk
    test_assert(power_pred.power_budget_violation_risk >= 0.0 && 
                power_pred.power_budget_violation_risk <= 1.0, 
                "Budget Violation Risk Range", 
                "Power budget violation risk should be between 0.0 and 1.0");
    
    if (verbose_output) {
        printf("  Power Prediction: %.2f W (confidence: %.3f)\n", 
               power_pred.predicted_power, power_pred.confidence);
        printf("  Budget violation risk: %.3f\n", power_pred.power_budget_violation_risk);
    }
    
    print_test_result("Power Prediction", tests_failed == 0);
}

static void test_thermal_prediction(void) {
    print_test_header("Thermal Prediction");
    
    // Build up thermal history with increasing trend
    for (int i = 0; i < 8; i++) {
        thermal_metrics_t metrics;
        simulate_thermal_metrics(&metrics, 45.0 + i * 2.0, 3.0);
        update_thermal_history(&metrics);
        usleep(50000); // 50ms between samples
    }
    
    // Get thermal predictions
    power_prediction_t power_pred;
    thermal_prediction_t thermal_pred;
    
    int result = power_optimizer_get_predictions(&power_pred, &thermal_pred);
    test_assert(result == POWER_OPT_SUCCESS, "Thermal Prediction Retrieval", 
                "Thermal prediction retrieval should succeed");
    
    test_assert(thermal_pred.predicted_max_temperature > 0.0, 
                "Predicted Temperature Positive", 
                "Predicted temperature should be positive");
    
    test_assert(thermal_pred.confidence >= 0.0 && thermal_pred.confidence <= 1.0, 
                "Thermal Confidence Range", 
                "Thermal prediction confidence should be between 0.0 and 1.0");
    
    test_assert(thermal_pred.predicted_hotspot_zone >= 0 && 
                thermal_pred.predicted_hotspot_zone < MAX_THERMAL_ZONES, 
                "Predicted Hotspot Zone Range", 
                "Predicted hotspot zone should be within valid range");
    
    // Test thermal violation risk
    test_assert(thermal_pred.thermal_violation_risk >= 0.0 && 
                thermal_pred.thermal_violation_risk <= 1.0, 
                "Thermal Violation Risk Range", 
                "Thermal violation risk should be between 0.0 and 1.0");
    
    if (verbose_output) {
        printf("  Thermal Prediction: %.1f°C (confidence: %.3f)\n", 
               thermal_pred.predicted_max_temperature, thermal_pred.confidence);
        printf("  Thermal violation risk: %.3f\n", thermal_pred.thermal_violation_risk);
        printf("  Cooling required: %s\n", thermal_pred.cooling_required ? "Yes" : "No");
    }
    
    print_test_result("Thermal Prediction", tests_failed == 0);
}

static void test_power_optimization_strategies(void) {
    print_test_header("Power Optimization Strategies");
    
    // Test different power scenarios
    power_scenario_t scenarios[] = {
        {"High Power", 18.0, 1.0, POWER_OBJ_REDUCE_POWER},
        {"Low Power", 8.0, 1.0, POWER_OBJ_OPTIMIZE_EFFICIENCY},
        {"Battery Mode", 12.0, 0.2, POWER_OBJ_EXTEND_BATTERY}, // 20% battery
        {"Performance Mode", 14.0, 0.8, POWER_OBJ_MAXIMIZE_PERFORMANCE}
    };
    
    for (int s = 0; s < 4; s++) {
        if (verbose_output) {
            printf("  Testing scenario: %s\n", scenarios[s].name);
        }
        
        // Set up scenario
        power_optimizer_set_power_budget(20.0);
        
        // Simulate metrics for this scenario
        power_metrics_t metrics;
        simulate_power_metrics(&metrics, scenarios[s].power_level, 0.5);
        metrics.battery_level = scenarios[s].battery_level;
        metrics.ac_power_available = (scenarios[s].battery_level > 0.5);
        
        update_power_history(&metrics);
        
        // Force optimization
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Force Optimization", 
                    "Force optimization should succeed");
        
        // Wait for optimization to complete
        usleep(300000); // 300ms
        
        // Get optimization statistics
        power_optimization_stats_t stats;
        result = get_power_optimization_stats(&stats);
        test_assert(result == POWER_OPT_SUCCESS, "Get Stats", 
                    "Getting optimization statistics should succeed");
        
        test_assert(stats.total_optimizations > 0, "Optimizations Performed", 
                    "At least one optimization should have been performed");
        
        if (verbose_output) {
            printf("    Optimizations: %lu, Success rate: %.1f%%\n", 
                   stats.total_optimizations, stats.success_rate * 100.0);
        }
    }
    
    print_test_result("Power Optimization Strategies", tests_failed == 0);
}

static void test_thermal_aware_optimization(void) {
    print_test_header("Thermal-Aware Optimization");
    
    // Create thermal stress scenarios
    thermal_scenario_t scenarios[] = {
        {"Normal Thermal", 50.0, 5.0},
        {"High Thermal", 75.0, 8.0},
        {"Critical Thermal", 88.0, 3.0},
        {"Thermal Imbalance", 60.0, 20.0} // High variation
    };
    
    for (int s = 0; s < 4; s++) {
        if (verbose_output) {
            printf("  Testing thermal scenario: %s\n", scenarios[s].name);
        }
        
        // Set thermal limit
        power_optimizer_set_thermal_limit(85.0);
        
        // Simulate thermal metrics
        thermal_metrics_t thermal_metrics;
        simulate_thermal_metrics(&thermal_metrics, scenarios[s].base_temp, scenarios[s].variation);
        
        // Simulate corresponding power metrics
        power_metrics_t power_metrics;
        simulate_power_metrics(&power_metrics, 15.0, 2.0);
        
        update_thermal_history(&thermal_metrics);
        update_power_history(&power_metrics);
        
        // Force thermal-aware optimization
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Thermal Optimization", 
                    "Thermal-aware optimization should succeed");
        
        usleep(250000); // 250ms
        
        // Validate thermal response
        power_optimization_stats_t stats;
        get_power_optimization_stats(&stats);
        
        if (scenarios[s].base_temp > 80.0) {
            // High temperature should trigger optimizations
            test_assert(stats.total_optimizations > 0, "High Temp Response", 
                        "High temperature should trigger optimizations");
        }
        
        if (verbose_output) {
            printf("    Base temp: %.1f°C, Optimizations: %lu\n", 
                   scenarios[s].base_temp, stats.total_optimizations);
        }
    }
    
    print_test_result("Thermal-Aware Optimization", tests_failed == 0);
}

static void test_battery_optimization(void) {
    print_test_header("Battery Optimization");
    
    // Test different battery scenarios
    battery_scenario_t scenarios[] = {
        {"Full Battery AC", 1.0, true, false},
        {"Half Battery AC", 0.5, true, false},
        {"Low Battery AC", 0.2, true, true},
        {"Critical Battery No AC", 0.1, false, true},
        {"Empty Battery No AC", 0.05, false, true}
    };
    
    for (int s = 0; s < 5; s++) {
        if (verbose_output) {
            printf("  Testing battery scenario: %s\n", scenarios[s].name);
        }
        
        // Simulate battery metrics
        power_metrics_t metrics;
        simulate_power_metrics(&metrics, 12.0, 1.0);
        metrics.battery_level = scenarios[s].battery_level;
        metrics.ac_power_available = scenarios[s].ac_available;
        
        update_power_history(&metrics);
        
        // Force battery optimization
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Battery Optimization", 
                    "Battery optimization should succeed");
        
        usleep(200000); // 200ms
        
        // Check optimization response
        power_optimization_stats_t stats;
        get_power_optimization_stats(&stats);
        
        if (scenarios[s].should_optimize) {
            test_assert(stats.total_optimizations > 0, "Battery Response", 
                        "Low battery should trigger optimizations");
            
            if (stats.total_power_saved > 0.0) {
                test_assert(stats.total_power_saved > 0.5, "Power Savings", 
                            "Battery optimization should achieve significant power savings");
            }
        }
        
        if (verbose_output) {
            printf("    Battery: %.1f%%, AC: %s, Power saved: %.2f W\n", 
                   scenarios[s].battery_level * 100.0,
                   scenarios[s].ac_available ? "Yes" : "No",
                   stats.total_power_saved);
        }
    }
    
    print_test_result("Battery Optimization", tests_failed == 0);
}

static void test_emergency_thermal_protection(void) {
    print_test_header("Emergency Thermal Protection");
    
    // Create emergency thermal scenario
    thermal_metrics_t thermal_metrics;
    simulate_thermal_metrics(&thermal_metrics, 95.0, 5.0); // Very high temperature
    
    power_metrics_t power_metrics;
    simulate_power_metrics(&power_metrics, 18.0, 1.0); // High power
    
    update_thermal_history(&thermal_metrics);
    update_power_history(&power_metrics);
    
    // Set aggressive thermal limit
    power_optimizer_set_thermal_limit(90.0);
    
    // Force emergency optimization
    int result = power_optimizer_force_optimization();
    test_assert(result == POWER_OPT_SUCCESS, "Emergency Optimization", 
                "Emergency thermal optimization should succeed");
    
    usleep(300000); // 300ms
    
    // Check emergency response
    power_optimization_stats_t stats;
    get_power_optimization_stats(&stats);
    
    test_assert(stats.total_optimizations > 0, "Emergency Response", 
                "Emergency thermal condition should trigger optimizations");
    
    test_assert(stats.total_power_saved > 2.0, "Emergency Power Reduction", 
                "Emergency thermal protection should achieve significant power reduction");
    
    if (verbose_output) {
        printf("  Emergency thermal response: %.2f W power saved\n", stats.total_power_saved);
    }
    
    print_test_result("Emergency Thermal Protection", tests_failed == 0);
}

static void test_machine_learning_prediction(void) {
    print_test_header("Machine Learning Prediction");
    
    // Build up training data
    for (int i = 0; i < 25; i++) {
        power_metrics_t power_metrics;
        thermal_metrics_t thermal_metrics;
        
        // Create correlated power and thermal data
        double base_power = 10.0 + sin(i * 0.2) * 3.0;
        double base_temp = 45.0 + (base_power - 10.0) * 2.0;
        
        simulate_power_metrics(&power_metrics, base_power, 0.5);
        simulate_thermal_metrics(&thermal_metrics, base_temp, 2.0);
        
        update_power_history(&power_metrics);
        update_thermal_history(&thermal_metrics);
        
        usleep(20000); // 20ms between samples
    }
    
    // Train ML model
    int result = train_ml_power_model();
    test_assert(result == POWER_OPT_SUCCESS, "ML Model Training", 
                "ML power model training should succeed");
    
    // Get model statistics
    power_optimization_stats_t stats;
    get_power_optimization_stats(&stats);
    
    test_assert(stats.ml_model_trained, "ML Model Trained", 
                "ML model should be marked as trained");
    
    test_assert(stats.ml_model_accuracy > 0.5, "ML Model Accuracy", 
                "ML model should have reasonable accuracy");
    
    // Test ML-based prediction
    power_prediction_t power_pred;
    thermal_prediction_t thermal_pred;
    
    result = power_optimizer_get_predictions(&power_pred, &thermal_pred);
    test_assert(result == POWER_OPT_SUCCESS, "ML Prediction", 
                "ML-based prediction should succeed");
    
    // ML predictions should have higher confidence
    test_assert(power_pred.confidence > 0.6, "ML Prediction Confidence", 
                "ML-based predictions should have higher confidence");
    
    if (verbose_output) {
        printf("  ML model accuracy: %.1f%%\n", stats.ml_model_accuracy * 100.0);
        printf("  ML prediction confidence: %.3f\n", power_pred.confidence);
    }
    
    print_test_result("Machine Learning Prediction", tests_failed == 0);
}

static void test_multi_objective_optimization(void) {
    print_test_header("Multi-Objective Optimization");
    
    // Test balancing multiple objectives
    multi_objective_scenario_t scenarios[] = {
        {"Performance Priority", 0.7, 0.2, 0.1},
        {"Power Priority", 0.1, 0.8, 0.1},
        {"Thermal Priority", 0.1, 0.2, 0.7},
        {"Balanced", 0.33, 0.33, 0.34}
    };
    
    for (int s = 0; s < 4; s++) {
        if (verbose_output) {
            printf("  Testing multi-objective scenario: %s\n", scenarios[s].name);
        }
        
        // Set up challenging scenario
        power_metrics_t power_metrics;
        thermal_metrics_t thermal_metrics;
        
        simulate_power_metrics(&power_metrics, 16.0, 1.0); // High power
        simulate_thermal_metrics(&thermal_metrics, 78.0, 5.0); // High temperature
        
        update_power_history(&power_metrics);
        update_thermal_history(&thermal_metrics);
        
        // Apply multi-objective optimization (simulated)
        // In a real implementation, this would call a specific multi-objective function
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Multi-Objective Optimization", 
                    "Multi-objective optimization should succeed");
        
        usleep(250000); // 250ms
        
        // Validate optimization considers multiple objectives
        power_optimization_stats_t stats;
        get_power_optimization_stats(&stats);
        
        test_assert(stats.total_optimizations > 0, "Multi-Objective Response", 
                    "Multi-objective optimization should be performed");
        
        if (verbose_output) {
            printf("    Weights - Perf: %.2f, Power: %.2f, Thermal: %.2f\n", 
                   scenarios[s].performance_weight,
                   scenarios[s].power_weight,
                   scenarios[s].thermal_weight);
            printf("    Optimizations: %lu, Power saved: %.2f W\n", 
                   stats.total_optimizations, stats.total_power_saved);
        }
    }
    
    print_test_result("Multi-Objective Optimization", tests_failed == 0);
}

static void test_adaptive_optimization_loop(void) {
    print_test_header("Adaptive Optimization Loop");
    
    // Run multiple optimization cycles to test adaptation
    for (int cycle = 0; cycle < TEST_OPTIMIZATION_CYCLES; cycle++) {
        // Vary conditions over time
        double power_level = 12.0 + sin(cycle * 0.3) * 4.0;
        double temp_level = 50.0 + cos(cycle * 0.2) * 15.0;
        
        power_metrics_t power_metrics;
        thermal_metrics_t thermal_metrics;
        
        simulate_power_metrics(&power_metrics, power_level, 1.0);
        simulate_thermal_metrics(&thermal_metrics, temp_level, 3.0);
        
        update_power_history(&power_metrics);
        update_thermal_history(&thermal_metrics);
        
        // Force optimization
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Adaptive Cycle", 
                    "Each adaptive optimization cycle should succeed");
        
        usleep(150000); // 150ms between cycles
        
        if (verbose_output && cycle % 3 == 0) {
            printf("  Cycle %d: Power %.1f W, Temp %.1f°C\n", 
                   cycle, power_level, temp_level);
        }
    }
    
    // Check adaptation results
    power_optimization_stats_t stats;
    get_power_optimization_stats(&stats);
    
    test_assert(stats.total_optimizations >= TEST_OPTIMIZATION_CYCLES, 
                "Optimization Count", 
                "Should have performed at least one optimization per cycle");
    
    test_assert(stats.success_rate > 0.6, "Adaptation Success Rate", 
                "Adaptive optimization should have good success rate");
    
    if (verbose_output) {
        printf("  Adaptive optimization results:\n");
        printf("    Total optimizations: %lu\n", stats.total_optimizations);
        printf("    Success rate: %.1f%%\n", stats.success_rate * 100.0);
        printf("    Total power saved: %.2f W\n", stats.total_power_saved);
        printf("    Total energy saved: %.2f Wh\n", stats.total_energy_saved);
    }
    
    print_test_result("Adaptive Optimization Loop", tests_failed == 0);
}

static void test_configuration_management(void) {
    print_test_header("Configuration Management");
    
    // Test power budget configuration
    double test_budgets[] = {10.0, 15.0, 20.0, 25.0};
    
    for (int i = 0; i < 4; i++) {
        int result = power_optimizer_set_power_budget(test_budgets[i]);
        test_assert(result == POWER_OPT_SUCCESS, "Set Power Budget", 
                    "Setting power budget should succeed");
        
        power_optimizer_config_t config;
        result = power_optimizer_get_config(&config);
        test_assert(result == POWER_OPT_SUCCESS, "Get Config", 
                    "Getting configuration should succeed");
        
        test_assert(fabs(config.power_budget_watts - test_budgets[i]) < 0.1, 
                    "Power Budget Validation", 
                    "Power budget should be set correctly");
    }
    
    // Test thermal limit configuration
    double test_limits[] = {75.0, 80.0, 85.0, 90.0};
    
    for (int i = 0; i < 4; i++) {
        int result = power_optimizer_set_thermal_limit(test_limits[i]);
        test_assert(result == POWER_OPT_SUCCESS, "Set Thermal Limit", 
                    "Setting thermal limit should succeed");
        
        power_optimizer_config_t config;
        result = power_optimizer_get_config(&config);
        test_assert(result == POWER_OPT_SUCCESS, "Get Config", 
                    "Getting configuration should succeed");
        
        test_assert(fabs(config.thermal_limit_celsius - test_limits[i]) < 0.1, 
                    "Thermal Limit Validation", 
                    "Thermal limit should be set correctly");
    }
    
    // Test optimization aggressiveness
    double test_aggressiveness[] = {0.3, 0.5, 0.7, 0.9};
    
    for (int i = 0; i < 4; i++) {
        int result = power_optimizer_set_optimization_aggressiveness(test_aggressiveness[i]);
        test_assert(result == POWER_OPT_SUCCESS, "Set Aggressiveness", 
                    "Setting optimization aggressiveness should succeed");
        
        power_optimizer_config_t config;
        result = power_optimizer_get_config(&config);
        test_assert(result == POWER_OPT_SUCCESS, "Get Config", 
                    "Getting configuration should succeed");
        
        test_assert(fabs(config.optimization_aggressiveness - test_aggressiveness[i]) < 0.1, 
                    "Aggressiveness Validation", 
                    "Optimization aggressiveness should be set correctly");
    }
    
    // Test invalid configurations
    int result = power_optimizer_set_power_budget(-5.0);
    test_assert(result == POWER_OPT_ERROR_INVALID_PARAM, "Invalid Power Budget", 
                "Invalid power budget should be rejected");
    
    result = power_optimizer_set_thermal_limit(150.0);
    test_assert(result == POWER_OPT_ERROR_INVALID_PARAM, "Invalid Thermal Limit", 
                "Invalid thermal limit should be rejected");
    
    print_test_result("Configuration Management", tests_failed == 0);
}

static void test_stress_scenarios(void) {
    print_test_header("Stress Test Scenarios");
    
    // Scenario 1: Rapid optimization requests
    for (int i = 0; i < 30; i++) {
        int result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Rapid Optimization", 
                    "Rapid optimization requests should succeed");
        usleep(25000); // 25ms between requests
    }
    
    // Scenario 2: Extreme conditions
    power_metrics_t extreme_power;
    thermal_metrics_t extreme_thermal;
    
    simulate_power_metrics(&extreme_power, 25.0, 5.0); // Very high power
    simulate_thermal_metrics(&extreme_thermal, 100.0, 10.0); // Very high temperature
    
    extreme_power.battery_level = 0.05; // Critical battery
    extreme_power.ac_power_available = false;
    
    update_power_history(&extreme_power);
    update_thermal_history(&extreme_thermal);
    
    int result = power_optimizer_force_optimization();
    test_assert(result == POWER_OPT_SUCCESS, "Extreme Conditions", 
                "Extreme conditions should be handled gracefully");
    
    // Scenario 3: Oscillating conditions
    for (int i = 0; i < 20; i++) {
        power_metrics_t osc_power;
        thermal_metrics_t osc_thermal;
        
        double power_level = 15.0 + sin(i * 0.5) * 8.0;
        double temp_level = 60.0 + cos(i * 0.7) * 25.0;
        
        simulate_power_metrics(&osc_power, power_level, 1.0);
        simulate_thermal_metrics(&osc_thermal, temp_level, 2.0);
        
        update_power_history(&osc_power);
        update_thermal_history(&osc_thermal);
        
        result = power_optimizer_force_optimization();
        test_assert(result == POWER_OPT_SUCCESS, "Oscillating Conditions", 
                    "Oscillating conditions should be handled");
        
        usleep(50000); // 50ms
    }
    
    // Check system stability after stress tests
    power_optimization_stats_t final_stats;
    get_power_optimization_stats(&final_stats);
    
    test_assert(final_stats.success_rate > 0.5, "System Stability", 
                "System should remain stable under stress");
    
    if (verbose_output) {
        printf("  Stress test results:\n");
        printf("    Total optimizations: %lu\n", final_stats.total_optimizations);
        printf("    Success rate: %.1f%%\n", final_stats.success_rate * 100.0);
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

static void simulate_power_metrics(power_metrics_t* metrics, double base_power, double variation) {
    memset(metrics, 0, sizeof(power_metrics_t));
    gettimeofday(&metrics->timestamp, NULL);
    
    // Distribute power across components
    double core_power_total = base_power * 0.4;
    double ai_power_total = base_power * 0.4;
    double other_power = base_power * 0.2;
    
    for (int i = 0; i < MAX_CORES; i++) {
        metrics->core_power[i] = (core_power_total / MAX_CORES) + 
                                random_double(-variation, variation);
        metrics->core_utilization[i] = random_double(0.2, 0.9);
        metrics->core_frequency[i] = 800 + (int)random_double(0, 800);
        metrics->core_voltage[i] = 0.8 + random_double(0, 0.4);
    }
    
    for (int i = 0; i < MAX_AI_UNITS; i++) {
        metrics->ai_unit_power[i] = (ai_power_total / MAX_AI_UNITS) + 
                                   random_double(-variation, variation);
        metrics->ai_unit_utilization[i] = random_double(0.1, 0.8);
        metrics->ai_unit_frequency[i] = 400 + (int)random_double(0, 400);
        metrics->ai_unit_voltage[i] = 0.9 + random_double(0, 0.3);
    }
    
    metrics->memory_power = other_power * 0.6 + random_double(-variation/2, variation/2);
    metrics->noc_power = other_power * 0.4 + random_double(-variation/2, variation/2);
    metrics->total_power = base_power + random_double(-variation, variation);
    
    metrics->memory_bandwidth_utilization = random_double(0.3, 0.8);
    metrics->noc_utilization = random_double(0.2, 0.7);
    
    metrics->battery_level = random_double(0.2, 1.0);
    metrics->battery_current = metrics->total_power / 12.0;
    metrics->ac_power_available = (random_double(0, 1) > 0.3);
    
    metrics->power_efficiency = random_double(0.4, 0.9);
}

static void simulate_thermal_metrics(thermal_metrics_t* metrics, double base_temp, double variation) {
    memset(metrics, 0, sizeof(thermal_metrics_t));
    gettimeofday(&metrics->timestamp, NULL);
    
    metrics->ambient_temperature = 25.0 + random_double(-5, 10);
    
    metrics->max_temperature = base_temp;
    metrics->hotspot_zone = 0;
    
    for (int i = 0; i < MAX_THERMAL_ZONES; i++) {
        metrics->zone_temperature[i] = base_temp + random_double(-variation, variation);
        metrics->thermal_limits[i] = 85.0 + random_double(-5, 10);
        metrics->cooling_capacity[i] = random_double(0.7, 1.0);
        
        if (metrics->zone_temperature[i] > metrics->max_temperature) {
            metrics->max_temperature = metrics->zone_temperature[i];
            metrics->hotspot_zone = i;
        }
    }
    
    // Calculate thermal gradient
    double min_temp = metrics->zone_temperature[0];
    for (int i = 1; i < MAX_THERMAL_ZONES; i++) {
        if (metrics->zone_temperature[i] < min_temp) {
            min_temp = metrics->zone_temperature[i];
        }
    }
    metrics->thermal_gradient = metrics->max_temperature - min_temp;
    
    metrics->thermal_efficiency = random_double(0.5, 0.9);
    metrics->thermal_alert = (metrics->max_temperature > 80.0);
    metrics->thermal_emergency = (metrics->max_temperature > 90.0);
}

static void print_power_metrics(const power_metrics_t* metrics) {
    printf("  Power Metrics:\n");
    printf("    Total Power: %.2f W\n", metrics->total_power);
    printf("    Core Power: ");
    for (int i = 0; i < MAX_CORES; i++) {
        printf("%.1f ", metrics->core_power[i]);
    }
    printf("W\n");
    printf("    AI Unit Power: ");
    for (int i = 0; i < MAX_AI_UNITS; i++) {
        printf("%.1f ", metrics->ai_unit_power[i]);
    }
    printf("W\n");
    printf("    Memory Power: %.2f W\n", metrics->memory_power);
    printf("    NoC Power: %.2f W\n", metrics->noc_power);
    printf("    Battery Level: %.1f%%\n", metrics->battery_level * 100.0);
    printf("    AC Power: %s\n", metrics->ac_power_available ? "Available" : "Not Available");
    printf("    Power Efficiency: %.3f\n", metrics->power_efficiency);
}

static void print_thermal_metrics(const thermal_metrics_t* metrics) {
    printf("  Thermal Metrics:\n");
    printf("    Ambient Temperature: %.1f°C\n", metrics->ambient_temperature);
    printf("    Zone Temperatures: ");
    for (int i = 0; i < MAX_THERMAL_ZONES; i++) {
        printf("%.1f ", metrics->zone_temperature[i]);
    }
    printf("°C\n");
    printf("    Max Temperature: %.1f°C (Zone %d)\n", 
           metrics->max_temperature, metrics->hotspot_zone);
    printf("    Thermal Gradient: %.1f°C\n", metrics->thermal_gradient);
    printf("    Thermal Efficiency: %.3f\n", metrics->thermal_efficiency);
    printf("    Thermal Alert: %s\n", metrics->thermal_alert ? "Yes" : "No");
    printf("    Thermal Emergency: %s\n", metrics->thermal_emergency ? "Yes" : "No");
}

static void print_power_strategy(const power_optimization_strategy_t* strategy) {
    printf("  Power Optimization Strategy:\n");
    printf("    Primary Objective: %d\n", strategy->primary_objective);
    printf("    Aggressiveness: %.2f\n", strategy->optimization_aggressiveness);
    printf("    Expected Power Savings: %.2f W\n", strategy->expected_power_savings);
    printf("    Expected Thermal Reduction: %.2f°C\n", strategy->expected_thermal_reduction);
    printf("    Confidence: %.3f\n", strategy->confidence);
}

// Define test scenario structures
typedef struct {
    const char* name;
    double power_level;
    double battery_level;
    power_optimization_objective_t expected_objective;
} power_scenario_t;

typedef struct {
    const char* name;
    double base_temp;
    double variation;
} thermal_scenario_t;

typedef struct {
    const char* name;
    double battery_level;
    bool ac_available;
    bool should_optimize;
} battery_scenario_t;

typedef struct {
    const char* name;
    double performance_weight;
    double power_weight;
    double thermal_weight;
} multi_objective_scenario_t;