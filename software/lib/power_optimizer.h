/*
 * Advanced Power Optimization Library Header
 * 
 * Provides sophisticated power optimization algorithms including
 * machine learning-based prediction, thermal-aware optimization,
 * and adaptive power management strategies.
 */

#ifndef POWER_OPTIMIZER_H
#define POWER_OPTIMIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum limits
#define MAX_CORES               8
#define MAX_AI_UNITS           4
#define MAX_THERMAL_ZONES      8
#define MAX_POWER_HISTORY      100
#define MAX_THERMAL_HISTORY    100
#define MAX_OPTIMIZATION_HISTORY 50

// Power optimization objectives
typedef enum {
    POWER_OBJ_REDUCE_POWER = 0,
    POWER_OBJ_REDUCE_THERMAL = 1,
    POWER_OBJ_EXTEND_BATTERY = 2,
    POWER_OBJ_OPTIMIZE_EFFICIENCY = 3,
    POWER_OBJ_MAXIMIZE_PERFORMANCE = 4
} power_optimization_objective_t;

// Power metrics structure
typedef struct {
    struct timeval timestamp;
    
    // Core power metrics
    double core_power[MAX_CORES];
    double core_utilization[MAX_CORES];
    int core_frequency[MAX_CORES];      // MHz
    double core_voltage[MAX_CORES];     // Volts
    
    // AI unit power metrics
    double ai_unit_power[MAX_AI_UNITS];
    double ai_unit_utilization[MAX_AI_UNITS];
    int ai_unit_frequency[MAX_AI_UNITS]; // MHz
    double ai_unit_voltage[MAX_AI_UNITS]; // Volts
    
    // System power metrics
    double memory_power;
    double noc_power;
    double total_power;
    double memory_bandwidth_utilization;
    double noc_utilization;
    
    // Battery metrics
    double battery_level;               // 0.0 to 1.0
    double battery_current;             // Amperes
    bool ac_power_available;
    
    // Efficiency metrics
    double power_efficiency;            // Performance per watt
} power_metrics_t;

// Thermal metrics structure
typedef struct {
    struct timeval timestamp;
    
    // Temperature readings
    double zone_temperature[MAX_THERMAL_ZONES];
    double thermal_limits[MAX_THERMAL_ZONES];
    double ambient_temperature;
    double max_temperature;
    int hotspot_zone;
    
    // Thermal characteristics
    double thermal_gradient;            // Max - min temperature
    double cooling_capacity[MAX_THERMAL_ZONES];
    double thermal_efficiency;
    
    // Thermal alerts
    bool thermal_alert;
    bool thermal_emergency;
} thermal_metrics_t;

// Power prediction structure
typedef struct {
    struct timeval timestamp;
    
    // Predicted power consumption
    double predicted_power;
    double predicted_core_power[MAX_CORES];
    double predicted_ai_unit_power[MAX_AI_UNITS];
    
    // Prediction metadata
    double confidence;                  // 0.0 to 1.0
    int prediction_horizon_ms;
    
    // Risk assessment
    double power_budget_violation_risk; // 0.0 to 1.0
    double battery_depletion_time_hours;
} power_prediction_t;

// Thermal prediction structure
typedef struct {
    struct timeval timestamp;
    
    // Predicted temperatures
    double predicted_temperature[MAX_THERMAL_ZONES];
    double predicted_max_temperature;
    int predicted_hotspot_zone;
    
    // Prediction metadata
    double confidence;
    int prediction_horizon_ms;
    
    // Thermal management requirements
    bool cooling_required;
    double required_cooling_capacity;
    double thermal_violation_risk;      // 0.0 to 1.0
} thermal_prediction_t;

// DVFS optimization strategy
typedef struct {
    double core_voltage_change[MAX_CORES];      // Voltage delta in volts
    int core_frequency_change[MAX_CORES];       // Frequency delta in MHz
    double ai_unit_voltage_change[MAX_AI_UNITS];
    int ai_unit_frequency_change[MAX_AI_UNITS];
} dvfs_strategy_t;

// Power gating strategy
typedef struct {
    bool gate_core[MAX_CORES];
    bool gate_ai_unit[MAX_AI_UNITS];
    bool gate_memory_controller;
    bool gate_noc_routers;
} power_gating_strategy_t;

// Thermal management strategy
typedef struct {
    bool increase_cooling;
    int cooling_level_change;           // Delta in cooling level
    bool enable_thermal_throttling;
    double throttling_factor;           // 0.0 to 1.0
    bool migrate_hot_tasks;
    int preferred_cool_zones[MAX_THERMAL_ZONES];
} thermal_strategy_t;

// Power optimization strategy
typedef struct {
    struct timeval timestamp;
    
    power_optimization_objective_t primary_objective;
    double optimization_aggressiveness; // 0.0 to 1.0
    
    dvfs_strategy_t dvfs_strategy;
    power_gating_strategy_t power_gating_strategy;
    thermal_strategy_t thermal_strategy;
    
    // Expected results
    double expected_power_savings;      // Watts
    double expected_thermal_reduction;  // Celsius
    double expected_efficiency_gain;
    double confidence;                  // 0.0 to 1.0
} power_optimization_strategy_t;

// Power optimization results
typedef struct {
    struct timeval timestamp;
    
    // Actual results
    double power_savings;               // Watts
    double thermal_reduction;           // Celsius
    double efficiency_improvement;
    double energy_savings;              // Wh
    
    // Validation
    bool optimization_successful;
    double confidence;
    
    // Performance impact
    double performance_impact;          // -1.0 to 1.0 (negative = degradation)
} power_optimization_results_t;

// Optimization history entry
typedef struct {
    struct timeval timestamp;
    power_optimization_strategy_t strategy;
    power_optimization_results_t results;
} power_optimization_history_t;

// Machine learning model state
typedef struct {
    bool model_trained;
    double learning_rate;
    double prediction_accuracy;
    int training_samples;
    
    // Model parameters (simplified)
    double power_coefficients[10];
    double thermal_coefficients[10];
} ml_model_t;

// Power optimizer configuration
typedef struct {
    int optimization_interval_ms;
    double power_budget_watts;
    double thermal_limit_celsius;
    double battery_capacity_wh;
    
    bool enable_ml_prediction;
    bool enable_thermal_optimization;
    bool enable_battery_optimization;
    bool enable_performance_scaling;
    
    double optimization_aggressiveness;  // 0.0 to 1.0
    double thermal_safety_margin;        // Celsius
    double power_efficiency_target;      // 0.0 to 1.0
    
    // Advanced settings
    double ml_learning_rate;
    int prediction_window_size;
    double thermal_prediction_weight;
    double power_prediction_weight;
} power_optimizer_config_t;

// Power optimization statistics
typedef struct {
    uint64_t total_optimizations;
    uint64_t successful_optimizations;
    double success_rate;
    
    double total_power_saved;           // Watts
    double total_energy_saved;          // Wh
    double average_power_savings;
    
    bool ml_model_trained;
    double ml_model_accuracy;
    
    struct timeval last_optimization;
    struct timeval total_runtime;
} power_optimization_stats_t;

// Main power optimizer state
typedef struct {
    bool initialized;
    bool force_optimization;
    
    power_optimizer_config_t config;
    
    // Current metrics
    power_metrics_t current_power_metrics;
    thermal_metrics_t current_thermal_metrics;
    
    // Predictions
    power_prediction_t power_prediction;
    thermal_prediction_t thermal_prediction;
    
    // History
    power_metrics_t power_history[MAX_POWER_HISTORY];
    int power_history_count;
    thermal_metrics_t thermal_history[MAX_THERMAL_HISTORY];
    int thermal_history_count;
    
    // Optimization history
    power_optimization_history_t optimization_history[MAX_OPTIMIZATION_HISTORY];
    int optimization_history_count;
    
    // Machine learning model
    ml_model_t ml_model;
    
    // Statistics
    uint64_t total_optimizations;
    uint64_t successful_optimizations;
    double total_power_saved;
    double total_energy_saved;
} power_optimizer_t;

// Function prototypes

// Initialization and control
int power_optimizer_init(power_optimizer_config_t* config);
int power_optimizer_start(void);
int power_optimizer_stop(void);
void power_optimizer_cleanup(void);

// Configuration management
int power_optimizer_set_power_budget(double power_budget_watts);
int power_optimizer_set_thermal_limit(double thermal_limit_celsius);
int power_optimizer_set_optimization_aggressiveness(double aggressiveness);
int power_optimizer_get_config(power_optimizer_config_t* config);

// Metrics and monitoring
int power_optimizer_get_metrics(power_metrics_t* power_metrics, thermal_metrics_t* thermal_metrics);
int power_optimizer_get_predictions(power_prediction_t* power_pred, thermal_prediction_t* thermal_pred);
int power_optimizer_get_history(power_optimization_history_t* history, int max_entries);

// Manual optimization control
int power_optimizer_force_optimization(void);
int power_optimizer_apply_strategy(power_optimization_strategy_t* strategy);
int power_optimizer_validate_strategy(power_optimization_strategy_t* strategy);

// History management
int update_power_history(power_metrics_t* metrics);
int update_thermal_history(thermal_metrics_t* metrics);
int clear_optimization_history(void);

// Machine learning functions
int train_ml_power_model(void);
int predict_power_ml(power_prediction_t* prediction);
int predict_thermal_ml(thermal_prediction_t* prediction);
int update_ml_model(power_optimization_results_t* results);

// Advanced optimization algorithms
int genetic_algorithm_power_optimization(power_optimization_strategy_t* strategy);
int simulated_annealing_optimization(power_optimization_strategy_t* strategy);
int multi_objective_power_optimization(power_optimization_strategy_t* strategy,
                                      double power_weight,
                                      double thermal_weight,
                                      double performance_weight);

// Thermal-aware optimization
int thermal_aware_power_optimization(power_optimization_strategy_t* strategy);
int hotspot_mitigation_strategy(power_optimization_strategy_t* strategy);
int thermal_balancing_optimization(power_optimization_strategy_t* strategy);

// Battery optimization
int battery_life_optimization(power_optimization_strategy_t* strategy);
int adaptive_battery_management(power_optimization_strategy_t* strategy);
int power_profile_optimization(power_optimization_strategy_t* strategy);

// Performance analysis
int analyze_power_efficiency(power_metrics_t* metrics, double* efficiency_score);
int analyze_thermal_efficiency(thermal_metrics_t* metrics, double* efficiency_score);
int calculate_optimization_roi(power_optimization_results_t* results, double* roi);

// Reporting and statistics
int get_power_optimization_stats(power_optimization_stats_t* stats);
int generate_power_report(const char* output_file);
int export_power_history(const char* output_file);
int export_thermal_history(const char* output_file);

// Utility functions
double calculate_power_efficiency_score(power_metrics_t* metrics);
double calculate_thermal_efficiency_score(thermal_metrics_t* metrics);
double calculate_battery_life_estimate(power_metrics_t* metrics, double battery_capacity_wh);
int detect_power_anomalies(power_metrics_t* metrics);
int detect_thermal_anomalies(thermal_metrics_t* metrics);

// Hardware abstraction layer
int read_power_registers(power_metrics_t* metrics);
int read_thermal_registers(thermal_metrics_t* metrics);
int write_dvfs_settings(dvfs_strategy_t* dvfs);
int write_power_gating_settings(power_gating_strategy_t* gating);
int write_thermal_settings(thermal_strategy_t* thermal);

// Calibration and tuning
int calibrate_power_model(void);
int calibrate_thermal_model(void);
int tune_optimization_parameters(void);
int validate_hardware_capabilities(void);

// Error codes
#define POWER_OPT_SUCCESS                0
#define POWER_OPT_ERROR_INIT            -1
#define POWER_OPT_ERROR_INVALID_PARAM   -2
#define POWER_OPT_ERROR_NOT_SUPPORTED   -3
#define POWER_OPT_ERROR_HARDWARE        -4
#define POWER_OPT_ERROR_THREAD          -5
#define POWER_OPT_ERROR_COLLECTION      -6
#define POWER_OPT_ERROR_PREDICTION      -7
#define POWER_OPT_ERROR_OPTIMIZATION    -8
#define POWER_OPT_ERROR_VALIDATION      -9
#define POWER_OPT_ERROR_INSUFFICIENT_DATA -10
#define POWER_OPT_ERROR_MODEL_NOT_TRAINED -11
#define POWER_OPT_ERROR_IO              -12

#ifdef __cplusplus
}
#endif

#endif // POWER_OPTIMIZER_H