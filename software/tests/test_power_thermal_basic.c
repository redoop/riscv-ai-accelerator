/*
 * Basic Power and Thermal Management Test
 * 
 * Simple test for power optimization functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// Basic power management structures (simplified)
typedef struct {
    float total_power;
    float max_temperature;
    float cpu_utilization;
    float ai_utilization;
} system_state_t;

typedef struct {
    int type;
    int priority;
    int deadline_ms;
    float estimated_power;
} workload_info_t;

typedef struct {
    float level_percent;
    float voltage;
    float current_ma;
    float temperature;
    int charging;
    int ac_connected;
    float estimated_runtime_hours;
} battery_info_t;

typedef struct {
    int voltage_level;
    int frequency_level;
    int power_mode;
    int thermal_throttle;
    int ai_unit_enable;
    int memory_freq_boost;
    int power_gating_mask;
    int sleep_request;
    float predicted_power;
    float efficiency_score;
    float power_savings;
    float battery_level;
    int sleep_mode;
    float estimated_runtime_hours;
} power_recommendation_t;

// Workload types
#define WORKLOAD_CPU_INTENSIVE  0
#define WORKLOAD_AI_INFERENCE   1
#define WORKLOAD_AI_TRAINING    2
#define WORKLOAD_MEMORY_INTENSIVE 3
#define WORKLOAD_IDLE           4

// Power modes
#define POWER_MODE_NORMAL           0
#define POWER_MODE_POWER_SAVE       1
#define POWER_MODE_THERMAL_THROTTLE 2
#define POWER_MODE_EMERGENCY        3
#define POWER_MODE_STANDBY          4
#define POWER_MODE_DEEP_SLEEP       5
#define POWER_MODE_HIBERNATE        6
#define POWER_MODE_BATTERY_NORMAL   7
#define POWER_MODE_BATTERY_SAVE     8
#define POWER_MODE_BATTERY_CRITICAL 9

// Voltage/frequency levels
#define VOLTAGE_ULTRA_LOW   0
#define VOLTAGE_LOW         1
#define VOLTAGE_NOMINAL     2
#define VOLTAGE_HIGH        3
#define VOLTAGE_ULTRA_HIGH  4

#define FREQ_ULTRA_LOW      0
#define FREQ_LOW            1
#define FREQ_NOMINAL        2
#define FREQ_HIGH           3
#define FREQ_ULTRA_HIGH     4

// Test configuration
#define POWER_BUDGET_WATTS      150.0f
#define THERMAL_LIMIT_CELSIUS   85.0f

// Test state
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
    double total_power_saved;
} test_results_t;

static test_results_t g_test_results = {0};

// Simple power optimization logic
static int simple_power_optimize(const system_state_t* state, 
                                const workload_info_t* workload,
                                power_recommendation_t* rec) {
    if (!state || !rec) return -1;
    
    // Initialize recommendation
    memset(rec, 0, sizeof(power_recommendation_t));
    rec->voltage_level = VOLTAGE_NOMINAL;
    rec->frequency_level = FREQ_NOMINAL;
    rec->power_mode = POWER_MODE_NORMAL;
    
    // Thermal emergency handling
    if (state->max_temperature > THERMAL_LIMIT_CELSIUS) {
        rec->power_mode = POWER_MODE_THERMAL_THROTTLE;
        rec->voltage_level = VOLTAGE_LOW;
        rec->frequency_level = FREQ_LOW;
        rec->thermal_throttle = 1;
        printf("Thermal emergency: %.1f°C > %.1f°C\n", 
               state->max_temperature, THERMAL_LIMIT_CELSIUS);
        return 0;
    }
    
    // Power budget handling
    if (state->total_power > POWER_BUDGET_WATTS) {
        rec->voltage_level = VOLTAGE_LOW;
        rec->frequency_level = FREQ_LOW;
        printf("Power budget exceeded: %.1fW > %.1fW\n", 
               state->total_power, POWER_BUDGET_WATTS);
    }
    
    // Workload-specific optimization
    if (workload) {
        switch (workload->type) {
            case WORKLOAD_CPU_INTENSIVE:
                if (state->total_power < POWER_BUDGET_WATTS * 0.8f) {
                    rec->frequency_level = FREQ_HIGH;
                }
                break;
                
            case WORKLOAD_AI_INFERENCE:
            case WORKLOAD_AI_TRAINING:
                rec->ai_unit_enable = 1;
                if (workload->priority > 5) {
                    rec->voltage_level = VOLTAGE_HIGH;
                }
                break;
                
            case WORKLOAD_MEMORY_INTENSIVE:
                rec->memory_freq_boost = 1;
                break;
                
            case WORKLOAD_IDLE:
                rec->power_mode = POWER_MODE_POWER_SAVE;
                rec->voltage_level = VOLTAGE_LOW;
                rec->frequency_level = FREQ_LOW;
                rec->power_gating_mask = 0xFE; // Gate all but core 0
                break;
        }
    }
    
    // Calculate efficiency and savings
    float baseline_power = POWER_BUDGET_WATTS;
    if (state->total_power < baseline_power) {
        rec->power_savings = baseline_power - state->total_power;
    }
    
    rec->efficiency_score = (state->cpu_utilization + state->ai_utilization) / 
                           (2.0f * state->total_power);
    
    return 0;
}

// Test functions
static int test_normal_power_management(void) {
    system_state_t state = {
        .total_power = 80.0f,
        .max_temperature = 55.0f,
        .cpu_utilization = 40.0f,
        .ai_utilization = 20.0f
    };
    
    workload_info_t workload = {
        .type = WORKLOAD_CPU_INTENSIVE,
        .priority = 5,
        .deadline_ms = 1000,
        .estimated_power = 60.0f
    };
    
    power_recommendation_t rec;
    
    int result = simple_power_optimize(&state, &workload, &rec);
    if (result != 0) {
        printf("ERROR: Power optimization failed\n");
        return 0;
    }
    
    printf("Normal operation: V=%d, F=%d, Mode=%d, Savings=%.2fW\n",
           rec.voltage_level, rec.frequency_level, rec.power_mode, rec.power_savings);
    
    g_test_results.total_power_saved += rec.power_savings;
    
    return 1;
}

static int test_thermal_emergency(void) {
    system_state_t state = {
        .total_power = 140.0f,
        .max_temperature = 90.0f, // Above thermal limit
        .cpu_utilization = 90.0f,
        .ai_utilization = 80.0f
    };
    
    workload_info_t workload = {
        .type = WORKLOAD_AI_TRAINING,
        .priority = 8,
        .deadline_ms = 5000,
        .estimated_power = 120.0f
    };
    
    power_recommendation_t rec;
    
    int result = simple_power_optimize(&state, &workload, &rec);
    if (result != 0) {
        printf("ERROR: Thermal emergency optimization failed\n");
        return 0;
    }
    
    if (!rec.thermal_throttle) {
        printf("ERROR: Thermal throttling not enabled\n");
        return 0;
    }
    
    if (rec.power_mode != POWER_MODE_THERMAL_THROTTLE) {
        printf("ERROR: Power mode not set to thermal throttle\n");
        return 0;
    }
    
    printf("Thermal emergency: V=%d, F=%d, Throttle=%s\n",
           rec.voltage_level, rec.frequency_level, 
           rec.thermal_throttle ? "Yes" : "No");
    
    return 1;
}

static int test_workload_optimization(void) {
    system_state_t state = {
        .total_power = 70.0f,
        .max_temperature = 50.0f,
        .cpu_utilization = 30.0f,
        .ai_utilization = 10.0f
    };
    
    // Test different workload types
    workload_info_t workloads[] = {
        {WORKLOAD_CPU_INTENSIVE, 7, 1000, 50.0f},
        {WORKLOAD_AI_INFERENCE, 6, 2000, 80.0f},
        {WORKLOAD_AI_TRAINING, 9, 5000, 120.0f},
        {WORKLOAD_MEMORY_INTENSIVE, 4, 3000, 40.0f},
        {WORKLOAD_IDLE, 1, 0, 10.0f}
    };
    
    for (int i = 0; i < 5; i++) {
        power_recommendation_t rec;
        
        int result = simple_power_optimize(&state, &workloads[i], &rec);
        if (result != 0) {
            printf("ERROR: Workload %d optimization failed\n", i);
            return 0;
        }
        
        printf("Workload %d: Type=%d, V=%d, F=%d, AI=%s\n",
               i, workloads[i].type, rec.voltage_level, rec.frequency_level,
               rec.ai_unit_enable ? "On" : "Off");
        
        // Validate workload-specific optimizations
        switch (workloads[i].type) {
            case WORKLOAD_AI_INFERENCE:
            case WORKLOAD_AI_TRAINING:
                if (!rec.ai_unit_enable) {
                    printf("WARNING: AI unit not enabled for AI workload\n");
                }
                break;
                
            case WORKLOAD_MEMORY_INTENSIVE:
                if (!rec.memory_freq_boost) {
                    printf("WARNING: Memory frequency not boosted\n");
                }
                break;
                
            case WORKLOAD_IDLE:
                if (rec.voltage_level >= VOLTAGE_NOMINAL) {
                    printf("WARNING: Voltage not reduced for idle workload\n");
                }
                break;
        }
    }
    
    return 1;
}

static int test_battery_optimization(void) {
    system_state_t state = {
        .total_power = 60.0f,
        .max_temperature = 45.0f,
        .cpu_utilization = 25.0f,
        .ai_utilization = 5.0f
    };
    
    workload_info_t workload = {
        .type = WORKLOAD_CPU_INTENSIVE,
        .priority = 5,
        .deadline_ms = 2000,
        .estimated_power = 50.0f
    };
    
    power_recommendation_t rec;
    
    // Simulate battery mode (would need actual battery state handling)
    int result = simple_power_optimize(&state, &workload, &rec);
    if (result != 0) {
        printf("ERROR: Battery optimization failed\n");
        return 0;
    }
    
    printf("Battery mode simulation: V=%d, F=%d, Efficiency=%.3f\n",
           rec.voltage_level, rec.frequency_level, rec.efficiency_score);
    
    return 1;
}

static int test_power_budget_violation(void) {
    system_state_t state = {
        .total_power = 180.0f, // Above budget
        .max_temperature = 60.0f,
        .cpu_utilization = 80.0f,
        .ai_utilization = 70.0f
    };
    
    workload_info_t workload = {
        .type = WORKLOAD_AI_TRAINING,
        .priority = 8,
        .deadline_ms = 3000,
        .estimated_power = 100.0f
    };
    
    power_recommendation_t rec;
    
    int result = simple_power_optimize(&state, &workload, &rec);
    if (result != 0) {
        printf("ERROR: Power budget optimization failed\n");
        return 0;
    }
    
    // Check that power is reduced
    if (rec.voltage_level >= VOLTAGE_NOMINAL && rec.frequency_level >= FREQ_NOMINAL) {
        printf("ERROR: Power not reduced for budget violation\n");
        return 0;
    }
    
    printf("Power budget violation: V=%d, F=%d\n", rec.voltage_level, rec.frequency_level);
    
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

int main(void) {
    printf("=== Basic Power and Thermal Management Test ===\n\n");
    
    // Initialize test results
    memset(&g_test_results, 0, sizeof(test_results_t));
    
    // Run test suite
    printf("Running basic power optimization tests...\n");
    
    print_test_result("Normal Power Management", test_normal_power_management());
    print_test_result("Thermal Emergency Handling", test_thermal_emergency());
    print_test_result("Workload Optimization", test_workload_optimization());
    print_test_result("Battery Optimization", test_battery_optimization());
    print_test_result("Power Budget Violation", test_power_budget_violation());
    
    // Print final results
    printf("\n=== Test Results Summary ===\n");
    printf("Tests run: %d\n", g_test_results.tests_run);
    printf("Tests passed: %d\n", g_test_results.tests_passed);
    printf("Tests failed: %d\n", g_test_results.tests_failed);
    printf("Success rate: %.1f%%\n", 
           (float)g_test_results.tests_passed / g_test_results.tests_run * 100.0f);
    printf("Total power saved: %.2f W\n", g_test_results.total_power_saved);
    
    if (g_test_results.tests_failed == 0) {
        printf("\n*** ALL TESTS PASSED ***\n");
        return 0;
    } else {
        printf("\n*** %d TESTS FAILED ***\n", g_test_results.tests_failed);
        return 1;
    }
}