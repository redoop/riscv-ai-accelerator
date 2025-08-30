/*
 * Comprehensive Power and Thermal Management Test Suite
 * 
 * Tests all aspects of the power optimization library including:
 * - Intelligent power management strategies
 * - Low-power modes and standby functionality
 * - Battery-aware optimization
 * - Thermal management integration
 * - Machine learning-based predictions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include "../lib/power_optimizer.h"

// Test configuration
#define TEST_DURATION_SEC       10
#define TEST_ITERATIONS         1000
#define POWER_BUDGET_WATTS      150.0f
#define THERMAL_LIMIT_CELSIUS   85.0f
#define BATTERY_CAPACITY_WH     100.0f

// Test state
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
    double total_power_saved;
    double total_energy_saved;
    int thermal_violations;
    int battery_cycles;
} test_results_t;

static test_results_t g_test_results = {0};

// Test function prototypes
static int test_power_optimizer_initialization(void);
static int test_normal_power_management(void);
static int test_thermal_emergency_handling(void);
static int test_battery_mode_optimization(void);
static int test_low_power_modes(void);
static int test_ml_power_prediction(void);
static int test_workload_optimization(void);
static int test_configuration_management(void);
static int test_stress_scenarios(void);
static int test_performance_metrics(void);

// Helper functions
static void simulate_system_state(system_state_t* state, int scenario);
static void simulate_workload(workload_info_t* workload, int type);
static void simulate_battery_info(battery_info_t* battery, int scenario);
static int validate_power_recommendation(const power_recommendation_t* rec);
static void print_test_result(const char* test_name, int result);
static void print_power_metrics(const power_metrics_t* metrics);

int main(void) {
    printf("=== Comprehensive Power and Thermal Management Test Suite ===\n\n");
    
    // Initialize test results
    memset(&g_test_results, 0, sizeof(test_results_t));
    
    // Run test suite
    printf("Running power optimization tests...\n");
    
    // Test 1: Initialization
    print_test_result("Power Optimizer Initialization", 
                     test_power_optimizer_initialization());
    
    // Test 2: Normal power management
    print_test_result("Normal Power Management", 
                     test_normal_power_management());
    
    // Test 3: Thermal emergency handling
    print_test_result("Thermal Emergency Handling", 
                     test_thermal_emergency_handling());
    
    // Test 4: Battery mode optimization
    print_test_result("Battery Mode Optimization", 
                     test_battery_mode_optimization());
    
    // Test 5: Low-power modes
    print_test_result("Low-Power Modes", 
                     test_low_power_modes());
    
    // Test 6: ML power prediction
    print_test_result("ML Power Prediction", 
                     test_ml_power_prediction());
    
    // Test 7: Workload optimization
    print_test_result("Workload Optimization", 
                     test_workload_optimization());
    
    // Test 8: Configuration management
    print_test_result("Configuration Management", 
                     test_configuration_management());
    
    // Test 9: Stress scenarios
    print_test_result("Stress Scenarios", 
                     test_stress_scenarios());
    
    // Test 10: Performance metrics
    print_test_result("Performance Metrics", 
                     test_performance_metrics());
    
    // Print final results
    printf("\n=== Test Results Summary ===\n");
    printf("Tests run: %d\n", g_test_results.tests_run);
    printf("Tests passed: %d\n", g_test_results.tests_passed);
    printf("Tests failed: %d\n", g_test_results.tests_failed);
    printf("Success rate: %.1f%%\n", 
           (float)g_test_results.tests_passed / g_test_results.tests_run * 100.0f);
    printf("Total power saved: %.2f W\n", g_test_results.total_power_saved);
    printf("Total energy saved: %.2f Wh\n", g_test_results.total_energy_saved);
    printf("Thermal violations: %d\n", g_test_results.thermal_violations);
    printf("Battery cycles: %d\n", g_test_results.battery_cycles);
    
    // Cleanup
    power_optimizer_cleanup();
    
    if (g_test_results.tests_failed == 0) {
        printf("\n*** ALL TESTS PASSED ***\n");
        return 0;
    } else {
        printf("\n*** %d TESTS FAILED ***\n", g_test_results.tests_failed);
        return 1;
    }
}

static int test_power_optimizer_initialization(void) {
    power_config_t config = {
        .dvfs_enable = true,
        .power_gating_enable = true,
        .thermal_management_enable = true,
        .ml_prediction_enable = true,
        .aggressive_optimization = false,
        .power_budget_watts = POWER_BUDGET_WATTS,
        .thermal_limit_celsius = THERMAL_LIMIT_CELSIUS,
        .battery_mode = false,
        .performance_priority = POWER_PRIORITY_BALANCED,
        .low_power_modes_enable = true,
        .standby_timeout_ms = 5000,
        .deep_sleep_timeout_ms = 30000,
        .hibernate_timeout_ms = 300000,
        .battery_low_threshold = 20.0f,
        .battery_critical_threshold = 5.0f
    };
    
    // Test initialization
    int result = power_optimizer_init(&config);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Power optimizer initialization failed: %d\n", result);
        return 0;
    }
    
    // Test double initialization (should succeed)
    result = power_optimizer_init(&config);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Double initialization failed: %d\n", result);
        return 0;
    }
    
    printf("Power optimizer initialized successfully\n");
    return 1;
}

static int test_normal_power_management(void) {
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    // Simulate normal operating conditions
    simulate_system_state(&system_state, 0); // Normal scenario
    simulate_workload(&workload, WORKLOAD_CPU_INTENSIVE);
    
    // Update system state
    int result = power_optimizer_update(&system_state);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: System state update failed: %d\n", result);
        return 0;
    }
    
    // Get optimization recommendation
    result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Power optimization failed: %d\n", result);
        return 0;
    }
    
    // Validate recommendation
    if (!validate_power_recommendation(&recommendation)) {
        printf("ERROR: Invalid power recommendation\n");
        return 0;
    }
    
    printf("Normal power management: V=%d, F=%d, Mode=%d, Savings=%.2fW\n",
           recommendation.voltage_level, recommendation.frequency_level,
           recommendation.power_mode, recommendation.power_savings);
    
    g_test_results.total_power_saved += recommendation.power_savings;
    
    return 1;
}

static int test_thermal_emergency_handling(void) {
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    // Simulate thermal emergency
    simulate_system_state(&system_state, 1); // Thermal emergency scenario
    system_state.max_temperature = 90.0f; // Above thermal limit
    
    simulate_workload(&workload, WORKLOAD_AI_TRAINING);
    
    // Update system state
    int result = power_optimizer_update(&system_state);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Thermal emergency update failed: %d\n", result);
        return 0;
    }
    
    // Get optimization recommendation
    result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Thermal emergency optimization failed: %d\n", result);
        return 0;
    }
    
    // Check thermal throttling is enabled
    if (!recommendation.thermal_throttle) {
        printf("ERROR: Thermal throttling not enabled in emergency\n");
        return 0;
    }
    
    // Check power mode is thermal throttle
    if (recommendation.power_mode != POWER_MODE_THERMAL_THROTTLE) {
        printf("ERROR: Power mode not set to thermal throttle\n");
        return 0;
    }
    
    // Check voltage/frequency reduction
    if (recommendation.voltage_level >= VOLTAGE_NOMINAL) {
        printf("ERROR: Voltage not reduced in thermal emergency\n");
        return 0;
    }
    
    printf("Thermal emergency handled: V=%d, F=%d, Throttle=%s\n",
           recommendation.voltage_level, recommendation.frequency_level,
           recommendation.thermal_throttle ? "Yes" : "No");
    
    g_test_results.thermal_violations++;
    
    return 1;
}

static int test_battery_mode_optimization(void) {
    battery_info_t battery_info;
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    // Test normal battery operation
    simulate_battery_info(&battery_info, 0); // Normal battery
    battery_info.ac_connected = false;
    battery_info.level_percent = 75.0f;
    
    int result = power_optimizer_set_battery_state(&battery_info);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Battery state update failed: %d\n", result);
        return 0;
    }
    
    simulate_system_state(&system_state, 0);
    simulate_workload(&workload, WORKLOAD_CPU_INTENSIVE);
    
    power_optimizer_update(&system_state);
    result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Battery optimization failed: %d\n", result);
        return 0;
    }
    
    printf("Normal battery mode: Level=%.1f%%, Runtime=%.1fh\n",
           recommendation.battery_level, recommendation.estimated_runtime_hours);
    
    // Test low battery scenario
    battery_info.level_percent = 15.0f; // Below low threshold
    power_optimizer_set_battery_state(&battery_info);
    power_optimizer_update(&system_state);
    result = power_optimizer_optimize(&workload, &recommendation);
    
    if (recommendation.power_mode != POWER_MODE_BATTERY_SAVE) {
        printf("ERROR: Battery save mode not activated for low battery\n");
        return 0;
    }
    
    printf("Low battery mode: Level=%.1f%%, Mode=%d\n",
           recommendation.battery_level, recommendation.power_mode);
    
    // Test critical battery scenario
    battery_info.level_percent = 3.0f; // Below critical threshold
    power_optimizer_set_battery_state(&battery_info);
    power_optimizer_update(&system_state);
    result = power_optimizer_optimize(&workload, &recommendation);
    
    if (recommendation.power_mode != POWER_MODE_BATTERY_CRITICAL) {
        printf("ERROR: Battery critical mode not activated\n");
        return 0;
    }
    
    printf("Critical battery mode: Level=%.1f%%, Mode=%d\n",
           recommendation.battery_level, recommendation.power_mode);
    
    g_test_results.battery_cycles++;
    
    return 1;
}

static int test_low_power_modes(void) {
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    // Test standby mode
    simulate_system_state(&system_state, 2); // Low activity scenario
    system_state.cpu_utilization = 2.0f; // Very low utilization
    system_state.ai_utilization = 0.0f;
    
    simulate_workload(&workload, WORKLOAD_IDLE);
    
    power_optimizer_update(&system_state);
    int result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Low power optimization failed: %d\n", result);
        return 0;
    }
    
    // Check that power saving mode is recommended
    if (recommendation.power_mode != POWER_MODE_POWER_SAVE && 
        recommendation.power_mode != POWER_MODE_STANDBY) {
        printf("WARNING: Power save mode not recommended for idle workload\n");
    }
    
    printf("Idle workload optimization: Mode=%d, V=%d, F=%d\n",
           recommendation.power_mode, recommendation.voltage_level, 
           recommendation.frequency_level);
    
    // Test sleep mode entry
    result = power_optimizer_enter_sleep(POWER_MODE_STANDBY);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Sleep mode entry failed: %d\n", result);
        return 0;
    }
    
    printf("Entered standby mode\n");
    
    // Test wake up
    result = power_optimizer_wake_up();
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Wake up failed: %d\n", result);
        return 0;
    }
    
    printf("Woke up from standby mode\n");
    
    // Test deep sleep
    result = power_optimizer_enter_sleep(POWER_MODE_DEEP_SLEEP);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Deep sleep entry failed: %d\n", result);
        return 0;
    }
    
    printf("Entered deep sleep mode\n");
    
    power_optimizer_wake_up();
    
    // Test hibernate
    result = power_optimizer_enter_sleep(POWER_MODE_HIBERNATE);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Hibernate entry failed: %d\n", result);
        return 0;
    }
    
    printf("Entered hibernate mode\n");
    
    power_optimizer_wake_up();
    
    return 1;
}

static int test_ml_power_prediction(void) {
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    // Generate training data by running various scenarios
    for (int i = 0; i < 50; i++) {
        simulate_system_state(&system_state, i % 3);
        simulate_workload(&workload, i % 5);
        
        power_optimizer_update(&system_state);
        power_optimizer_optimize(&workload, &recommendation);
        
        // Simulate some delay for training
        usleep(1000); // 1ms
    }
    
    // Test prediction accuracy
    simulate_system_state(&system_state, 0);
    simulate_workload(&workload, WORKLOAD_AI_INFERENCE);
    
    power_optimizer_update(&system_state);
    int result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: ML prediction optimization failed: %d\n", result);
        return 0;
    }
    
    // Check that prediction is reasonable
    if (recommendation.predicted_power <= 0 || 
        recommendation.predicted_power > POWER_BUDGET_WATTS * 2) {
        printf("ERROR: ML prediction out of range: %.2fW\n", 
               recommendation.predicted_power);
        return 0;
    }
    
    printf("ML prediction: %.2fW, Efficiency: %.3f\n",
           recommendation.predicted_power, recommendation.efficiency_score);
    
    return 1;
}

static int test_workload_optimization(void) {
    system_state_t system_state;
    power_recommendation_t recommendation;
    
    simulate_system_state(&system_state, 0);
    power_optimizer_update(&system_state);
    
    // Test different workload types
    workload_info_t workloads[] = {
        {WORKLOAD_CPU_INTENSIVE, WORKLOAD_PRIORITY_HIGH, 1000, 50.0f},
        {WORKLOAD_AI_INFERENCE, WORKLOAD_PRIORITY_MEDIUM, 2000, 80.0f},
        {WORKLOAD_AI_TRAINING, WORKLOAD_PRIORITY_HIGH, 5000, 120.0f},
        {WORKLOAD_MEMORY_INTENSIVE, WORKLOAD_PRIORITY_LOW, 3000, 40.0f},
        {WORKLOAD_IDLE, WORKLOAD_PRIORITY_LOW, 0, 10.0f}
    };
    
    for (int i = 0; i < 5; i++) {
        int result = power_optimizer_optimize(&workloads[i], &recommendation);
        if (result != POWER_OPT_SUCCESS) {
            printf("ERROR: Workload %d optimization failed: %d\n", i, result);
            return 0;
        }
        
        printf("Workload %d: Type=%d, V=%d, F=%d, AI=%s\n",
               i, workloads[i].type, recommendation.voltage_level,
               recommendation.frequency_level,
               recommendation.ai_unit_enable ? "On" : "Off");
        
        // Validate workload-specific optimizations
        switch (workloads[i].type) {
            case WORKLOAD_AI_INFERENCE:
            case WORKLOAD_AI_TRAINING:
                if (!recommendation.ai_unit_enable) {
                    printf("WARNING: AI unit not enabled for AI workload\n");
                }
                break;
                
            case WORKLOAD_MEMORY_INTENSIVE:
                if (!recommendation.memory_freq_boost) {
                    printf("WARNING: Memory frequency not boosted for memory workload\n");
                }
                break;
                
            case WORKLOAD_IDLE:
                if (recommendation.voltage_level >= VOLTAGE_NOMINAL) {
                    printf("WARNING: Voltage not reduced for idle workload\n");
                }
                break;
        }
    }
    
    return 1;
}

static int test_configuration_management(void) {
    power_config_t config;
    power_config_t new_config = {
        .dvfs_enable = true,
        .power_gating_enable = true,
        .thermal_management_enable = true,
        .ml_prediction_enable = false, // Disable ML
        .aggressive_optimization = true,
        .power_budget_watts = 100.0f, // Reduced budget
        .thermal_limit_celsius = 75.0f, // Lower thermal limit
        .battery_mode = true,
        .performance_priority = POWER_PRIORITY_POWER_SAVE,
        .low_power_modes_enable = true,
        .standby_timeout_ms = 2000, // Shorter timeout
        .deep_sleep_timeout_ms = 10000,
        .hibernate_timeout_ms = 60000,
        .battery_low_threshold = 30.0f,
        .battery_critical_threshold = 10.0f
    };
    
    // Test configuration update
    int result = power_optimizer_set_config(&new_config);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Configuration update failed: %d\n", result);
        return 0;
    }
    
    printf("Configuration updated successfully\n");
    
    // Test with new configuration
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    simulate_system_state(&system_state, 0);
    system_state.total_power = 110.0f; // Above new budget
    
    simulate_workload(&workload, WORKLOAD_CPU_INTENSIVE);
    
    power_optimizer_update(&system_state);
    result = power_optimizer_optimize(&workload, &recommendation);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Optimization with new config failed: %d\n", result);
        return 0;
    }
    
    // Check that power budget violation is handled
    if (recommendation.voltage_level >= VOLTAGE_NOMINAL) {
        printf("ERROR: Power budget violation not handled\n");
        return 0;
    }
    
    printf("New config optimization: Budget=%.1fW, V=%d, F=%d\n",
           new_config.power_budget_watts, recommendation.voltage_level,
           recommendation.frequency_level);
    
    return 1;
}

static int test_stress_scenarios(void) {
    system_state_t system_state;
    workload_info_t workload;
    power_recommendation_t recommendation;
    
    printf("Running stress test scenarios...\n");
    
    // Stress test with rapid changes
    for (int i = 0; i < 100; i++) {
        // Randomly vary system conditions
        simulate_system_state(&system_state, rand() % 3);
        system_state.total_power = 50.0f + (rand() % 100);
        system_state.max_temperature = 40.0f + (rand() % 50);
        system_state.cpu_utilization = (rand() % 100);
        system_state.ai_utilization = (rand() % 100);
        
        simulate_workload(&workload, rand() % 5);
        
        int result = power_optimizer_update(&system_state);
        if (result != POWER_OPT_SUCCESS) {
            printf("ERROR: Stress test update %d failed: %d\n", i, result);
            return 0;
        }
        
        result = power_optimizer_optimize(&workload, &recommendation);
        if (result != POWER_OPT_SUCCESS) {
            printf("ERROR: Stress test optimization %d failed: %d\n", i, result);
            return 0;
        }
        
        // Validate recommendation
        if (!validate_power_recommendation(&recommendation)) {
            printf("ERROR: Invalid recommendation in stress test %d\n", i);
            return 0;
        }
        
        if (i % 20 == 0) {
            printf("  Stress test %d: Power=%.1fW, Temp=%.1f°C, Mode=%d\n",
                   i, system_state.total_power, system_state.max_temperature,
                   recommendation.power_mode);
        }
    }
    
    printf("Stress test completed successfully\n");
    return 1;
}

static int test_performance_metrics(void) {
    power_metrics_t metrics;
    battery_info_t battery_info;
    
    // Get performance metrics
    int result = power_optimizer_get_metrics(&metrics);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Failed to get metrics: %d\n", result);
        return 0;
    }
    
    print_power_metrics(&metrics);
    
    // Get battery info
    result = power_optimizer_get_battery_info(&battery_info);
    if (result != POWER_OPT_SUCCESS) {
        printf("ERROR: Failed to get battery info: %d\n", result);
        return 0;
    }
    
    printf("Battery: %.1f%%, %.2fV, %s, Runtime: %.1fh\n",
           battery_info.level_percent, battery_info.voltage,
           battery_info.ac_connected ? "AC" : "Battery",
           battery_info.estimated_runtime_hours);
    
    // Update global test results
    g_test_results.total_power_saved += metrics.power_savings_total;
    g_test_results.total_energy_saved += metrics.power_savings_total * 0.001f; // Simplified
    
    return 1;
}

// Helper function implementations

static void simulate_system_state(system_state_t* state, int scenario) {
    memset(state, 0, sizeof(system_state_t));
    
    switch (scenario) {
        case 0: // Normal operation
            state->total_power = 80.0f;
            state->max_temperature = 55.0f;
            state->cpu_utilization = 40.0f;
            state->ai_utilization = 20.0f;
            break;
            
        case 1: // High load / thermal stress
            state->total_power = 140.0f;
            state->max_temperature = 85.0f;
            state->cpu_utilization = 90.0f;
            state->ai_utilization = 80.0f;
            break;
            
        case 2: // Low activity
            state->total_power = 30.0f;
            state->max_temperature = 45.0f;
            state->cpu_utilization = 5.0f;
            state->ai_utilization = 0.0f;
            break;
    }
}

static void simulate_workload(workload_info_t* workload, int type) {
    memset(workload, 0, sizeof(workload_info_t));
    
    workload->type = type;
    workload->priority = WORKLOAD_PRIORITY_MEDIUM;
    workload->deadline_ms = 1000;
    
    switch (type) {
        case WORKLOAD_CPU_INTENSIVE:
            workload->estimated_power = 60.0f;
            break;
        case WORKLOAD_AI_INFERENCE:
            workload->estimated_power = 80.0f;
            break;
        case WORKLOAD_AI_TRAINING:
            workload->estimated_power = 120.0f;
            break;
        case WORKLOAD_MEMORY_INTENSIVE:
            workload->estimated_power = 45.0f;
            break;
        case WORKLOAD_IDLE:
            workload->estimated_power = 15.0f;
            break;
    }
}

static void simulate_battery_info(battery_info_t* battery, int scenario) {
    memset(battery, 0, sizeof(battery_info_t));
    
    switch (scenario) {
        case 0: // Normal battery
            battery->level_percent = 75.0f;
            battery->voltage = 3.8f;
            battery->current_ma = 1500.0f;
            battery->temperature = 25.0f;
            battery->charging = false;
            battery->ac_connected = false;
            break;
            
        case 1: // Low battery
            battery->level_percent = 15.0f;
            battery->voltage = 3.4f;
            battery->current_ma = 800.0f;
            battery->temperature = 30.0f;
            battery->charging = false;
            battery->ac_connected = false;
            break;
            
        case 2: // Charging
            battery->level_percent = 60.0f;
            battery->voltage = 4.1f;
            battery->current_ma = -2000.0f; // Negative for charging
            battery->temperature = 35.0f;
            battery->charging = true;
            battery->ac_connected = true;
            break;
    }
}

static int validate_power_recommendation(const power_recommendation_t* rec) {
    // Check voltage level is valid
    if (rec->voltage_level > VOLTAGE_ULTRA_HIGH) {
        return 0;
    }
    
    // Check frequency level is valid
    if (rec->frequency_level > FREQ_ULTRA_HIGH) {
        return 0;
    }
    
    // Check power mode is valid
    if (rec->power_mode > POWER_MODE_BATTERY_CRITICAL) {
        return 0;
    }
    
    // Check efficiency score is reasonable
    if (rec->efficiency_score < 0.0f || rec->efficiency_score > 10.0f) {
        return 0;
    }
    
    return 1;
}

static void print_test_result(const char* test_name, int result) {
    g_test_results.tests_run++;
    
    if (result) {
        g_test_results.tests_passed++;
        printf("✓ %s: PASSED\n", test_name);
    } else {
        g_test_results.tests_failed++;
        printf("✗ %s: FAILED\n", test_name);
    }
}

static void print_power_metrics(const power_metrics_t* metrics) {
    printf("Power Metrics:\n");
    printf("  Optimization cycles: %lu\n", metrics->optimization_cycles);
    printf("  Power savings total: %.2f W-cycles\n", metrics->power_savings_total);
    printf("  Thermal violations: %lu\n", metrics->thermal_violations);
    printf("  Efficiency score: %.3f\n", metrics->efficiency_score);
    printf("  Sleep time total: %lu ms\n", metrics->sleep_time_total);
    printf("  Wake events total: %lu\n", metrics->wake_events_total);
    printf("  Battery cycles: %lu\n", metrics->battery_cycles);
}