/*
 * Performance Optimizer Library for RISC-V AI Accelerator
 * 
 * This library provides adaptive performance optimization algorithms
 * and real-time tuning capabilities for AI workloads.
 */

#include "performance_optimizer.h"
#include "performance_analyzer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>

// Global optimizer state
static perf_optimizer_t g_optimizer = {0};
static pthread_mutex_t g_optimizer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_optimizer_thread;
static volatile bool g_optimizer_running = false;

// Hardware register access (platform-specific)
#define PERF_MONITOR_BASE       0x50000000
#define RESOURCE_SCHEDULER_BASE 0x51000000

// Performance optimization thread
static void* performance_optimization_thread(void* arg);

// Helper functions
static int analyze_workload_characteristics(workload_profile_t* profile);
static int predict_performance_impact(optimization_strategy_t* strategy, 
                                    performance_prediction_t* prediction);
static int apply_optimization_strategy(optimization_strategy_t* strategy);
static int validate_optimization_results(optimization_results_t* results);
static double calculate_performance_score(performance_metrics_t* metrics);
static int detect_performance_anomalies(performance_metrics_t* metrics);

int perf_optimizer_init(perf_optimizer_config_t* config) {
    pthread_mutex_lock(&g_optimizer_mutex);
    
    if (g_optimizer.initialized) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_SUCCESS;
    }
    
    // Initialize performance counters
    if (perf_init_counters() != PERF_SUCCESS) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_INIT;
    }
    
    // Copy configuration
    if (config) {
        memcpy(&g_optimizer.config, config, sizeof(perf_optimizer_config_t));
    } else {
        // Use default configuration
        g_optimizer.config.optimization_interval_ms = 1000;
        g_optimizer.config.adaptation_aggressiveness = 0.5;
        g_optimizer.config.power_budget_watts = 15.0;
        g_optimizer.config.thermal_limit_celsius = 85.0;
        g_optimizer.config.enable_predictive_optimization = true;
        g_optimizer.config.enable_workload_profiling = true;
        g_optimizer.config.enable_power_optimization = true;
        g_optimizer.config.enable_thermal_optimization = true;
    }
    
    // Initialize workload profiles
    for (int i = 0; i < MAX_WORKLOAD_PROFILES; i++) {
        g_optimizer.workload_profiles[i].profile_id = i;
        g_optimizer.workload_profiles[i].valid = false;
    }
    
    // Initialize optimization history
    g_optimizer.optimization_history_count = 0;
    
    // Initialize performance metrics
    memset(&g_optimizer.current_metrics, 0, sizeof(performance_metrics_t));
    
    g_optimizer.initialized = true;
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_start(void) {
    pthread_mutex_lock(&g_optimizer_mutex);
    
    if (!g_optimizer.initialized) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_INIT;
    }
    
    if (g_optimizer_running) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_SUCCESS;
    }
    
    g_optimizer_running = true;
    
    // Create optimization thread
    if (pthread_create(&g_optimizer_thread, NULL, performance_optimization_thread, NULL) != 0) {
        g_optimizer_running = false;
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_THREAD;
    }
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_stop(void) {
    pthread_mutex_lock(&g_optimizer_mutex);
    
    if (!g_optimizer_running) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_SUCCESS;
    }
    
    g_optimizer_running = false;
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    // Wait for optimization thread to finish
    pthread_join(g_optimizer_thread, NULL);
    
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_register_workload(workload_profile_t* profile) {
    if (!profile) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_optimizer_mutex);
    
    // Find available slot
    int slot = -1;
    for (int i = 0; i < MAX_WORKLOAD_PROFILES; i++) {
        if (!g_optimizer.workload_profiles[i].valid) {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_NO_SPACE;
    }
    
    // Copy workload profile
    memcpy(&g_optimizer.workload_profiles[slot], profile, sizeof(workload_profile_t));
    g_optimizer.workload_profiles[slot].valid = true;
    g_optimizer.workload_profiles[slot].profile_id = slot;
    
    // Initialize performance history for this workload
    g_optimizer.workload_profiles[slot].performance_history_count = 0;
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return slot; // Return profile ID
}

int perf_optimizer_unregister_workload(int profile_id) {
    if (profile_id < 0 || profile_id >= MAX_WORKLOAD_PROFILES) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_optimizer_mutex);
    
    g_optimizer.workload_profiles[profile_id].valid = false;
    memset(&g_optimizer.workload_profiles[profile_id], 0, sizeof(workload_profile_t));
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_get_recommendations(optimization_recommendations_t* recommendations) {
    if (!recommendations) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_optimizer_mutex);
    
    if (!g_optimizer.initialized) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_INIT;
    }
    
    // Copy current recommendations
    memcpy(recommendations, &g_optimizer.current_recommendations, 
           sizeof(optimization_recommendations_t));
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_get_metrics(performance_metrics_t* metrics) {
    if (!metrics) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_optimizer_mutex);
    
    // Copy current metrics
    memcpy(metrics, &g_optimizer.current_metrics, sizeof(performance_metrics_t));
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return PERF_OPT_SUCCESS;
}

int perf_optimizer_force_optimization(void) {
    pthread_mutex_lock(&g_optimizer_mutex);
    
    if (!g_optimizer.initialized) {
        pthread_mutex_unlock(&g_optimizer_mutex);
        return PERF_OPT_ERROR_INIT;
    }
    
    g_optimizer.force_optimization = true;
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return PERF_OPT_SUCCESS;
}

// Performance optimization thread implementation
static void* performance_optimization_thread(void* arg) {
    struct timespec sleep_time;
    sleep_time.tv_sec = g_optimizer.config.optimization_interval_ms / 1000;
    sleep_time.tv_nsec = (g_optimizer.config.optimization_interval_ms % 1000) * 1000000;
    
    while (g_optimizer_running) {
        pthread_mutex_lock(&g_optimizer_mutex);
        
        bool should_optimize = g_optimizer.force_optimization;
        g_optimizer.force_optimization = false;
        
        pthread_mutex_unlock(&g_optimizer_mutex);
        
        if (should_optimize || true) { // Always run optimization cycle
            // Collect current performance metrics
            performance_metrics_t current_metrics;
            if (collect_performance_metrics(&current_metrics) == PERF_OPT_SUCCESS) {
                
                pthread_mutex_lock(&g_optimizer_mutex);
                memcpy(&g_optimizer.current_metrics, &current_metrics, 
                       sizeof(performance_metrics_t));
                pthread_mutex_unlock(&g_optimizer_mutex);
                
                // Analyze workload characteristics
                workload_profile_t current_profile;
                if (analyze_workload_characteristics(&current_profile) == PERF_OPT_SUCCESS) {
                    
                    // Generate optimization strategy
                    optimization_strategy_t strategy;
                    if (generate_optimization_strategy(&current_metrics, &current_profile, 
                                                     &strategy) == PERF_OPT_SUCCESS) {
                        
                        // Predict performance impact
                        performance_prediction_t prediction;
                        if (predict_performance_impact(&strategy, &prediction) == PERF_OPT_SUCCESS) {
                            
                            // Apply optimization if beneficial
                            if (prediction.expected_performance_gain > 0.05) { // 5% threshold
                                optimization_results_t results;
                                if (apply_optimization_strategy(&strategy) == PERF_OPT_SUCCESS) {
                                    
                                    // Wait for optimization to take effect
                                    usleep(100000); // 100ms
                                    
                                    // Validate results
                                    if (validate_optimization_results(&results) == PERF_OPT_SUCCESS) {
                                        
                                        pthread_mutex_lock(&g_optimizer_mutex);
                                        
                                        // Update optimization history
                                        if (g_optimizer.optimization_history_count < MAX_OPTIMIZATION_HISTORY) {
                                            int idx = g_optimizer.optimization_history_count++;
                                            memcpy(&g_optimizer.optimization_history[idx].strategy, 
                                                   &strategy, sizeof(optimization_strategy_t));
                                            memcpy(&g_optimizer.optimization_history[idx].results, 
                                                   &results, sizeof(optimization_results_t));
                                            gettimeofday(&g_optimizer.optimization_history[idx].timestamp, NULL);
                                        }
                                        
                                        // Update recommendations
                                        update_recommendations(&strategy, &results);
                                        
                                        pthread_mutex_unlock(&g_optimizer_mutex);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        nanosleep(&sleep_time, NULL);
    }
    
    return NULL;
}

int collect_performance_metrics(performance_metrics_t* metrics) {
    if (!metrics) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    memset(metrics, 0, sizeof(performance_metrics_t));
    
    // Collect CPU performance metrics
    uint64_t cycles, instructions, cache_hits, cache_misses;
    
    if (perf_read_counter(PERF_COUNTER_CYCLES, &cycles) == PERF_SUCCESS) {
        metrics->cpu_cycles = cycles;
    }
    
    if (perf_read_counter(PERF_COUNTER_INSTRUCTIONS, &instructions) == PERF_SUCCESS) {
        metrics->cpu_instructions = instructions;
    }
    
    if (perf_read_counter(PERF_COUNTER_CACHE_HITS, &cache_hits) == PERF_SUCCESS &&
        perf_read_counter(PERF_COUNTER_CACHE_MISSES, &cache_misses) == PERF_SUCCESS) {
        uint64_t total_accesses = cache_hits + cache_misses;
        if (total_accesses > 0) {
            metrics->cache_hit_rate = (double)cache_hits / (double)total_accesses;
        }
    }
    
    // Calculate IPC
    if (cycles > 0) {
        metrics->ipc = (double)instructions / (double)cycles;
    }
    
    // Collect AI accelerator metrics
    uint64_t tpu_cycles, tpu_ops, vpu_cycles, vpu_ops;
    
    if (perf_read_counter(PERF_COUNTER_TPU_CYCLES, &tpu_cycles) == PERF_SUCCESS) {
        metrics->tpu_cycles = tpu_cycles;
    }
    
    if (perf_read_counter(PERF_COUNTER_TPU_OPERATIONS, &tpu_ops) == PERF_SUCCESS) {
        metrics->tpu_operations = tpu_ops;
    }
    
    if (perf_read_counter(PERF_COUNTER_VPU_CYCLES, &vpu_cycles) == PERF_SUCCESS) {
        metrics->vpu_cycles = vpu_cycles;
    }
    
    if (perf_read_counter(PERF_COUNTER_VPU_OPERATIONS, &vpu_ops) == PERF_SUCCESS) {
        metrics->vpu_operations = vpu_ops;
    }
    
    // Calculate AI utilization
    if (cycles > 0) {
        metrics->tpu_utilization = (double)tpu_cycles / (double)cycles;
        metrics->vpu_utilization = (double)vpu_cycles / (double)cycles;
    }
    
    // Collect power and thermal metrics
    double current_power, average_power;
    if (perf_monitor_power_consumption(&current_power, &average_power) == PERF_SUCCESS) {
        metrics->power_consumption = current_power;
        metrics->average_power = average_power;
    }
    
    // Collect memory metrics
    uint64_t memory_accesses;
    if (perf_read_counter(PERF_COUNTER_MEMORY_ACCESSES, &memory_accesses) == PERF_SUCCESS) {
        metrics->memory_accesses = memory_accesses;
        
        // Estimate memory bandwidth (simplified)
        if (cycles > 0) {
            // Assume 64-byte cache lines and 1GHz clock
            metrics->memory_bandwidth = (double)(memory_accesses * 64) / 
                                       ((double)cycles / 1000000000.0);
        }
    }
    
    // Collect NoC metrics
    uint64_t noc_packets;
    if (perf_monitor_noc_traffic(&noc_packets, NULL) == PERF_SUCCESS) {
        metrics->noc_packets = noc_packets;
    }
    
    // Calculate overall performance score
    metrics->performance_score = calculate_performance_score(metrics);
    
    // Detect anomalies
    metrics->anomaly_detected = (detect_performance_anomalies(metrics) != 0);
    
    // Set timestamp
    gettimeofday(&metrics->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

int generate_optimization_strategy(performance_metrics_t* metrics, 
                                 workload_profile_t* profile,
                                 optimization_strategy_t* strategy) {
    if (!metrics || !profile || !strategy) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    memset(strategy, 0, sizeof(optimization_strategy_t));
    
    // Analyze current performance vs targets
    double performance_gap = profile->target_performance - metrics->performance_score;
    double power_headroom = g_optimizer.config.power_budget_watts - metrics->power_consumption;
    
    // DVFS optimization
    if (performance_gap > 0.1 && power_headroom > 2.0) {
        // Performance is below target and we have power headroom
        strategy->dvfs_changes.increase_frequency = true;
        strategy->dvfs_changes.frequency_step = (int)(performance_gap * 10);
        if (strategy->dvfs_changes.frequency_step > 3) {
            strategy->dvfs_changes.frequency_step = 3;
        }
        
        strategy->dvfs_changes.increase_voltage = true;
        strategy->dvfs_changes.voltage_step = strategy->dvfs_changes.frequency_step;
        
    } else if (performance_gap < -0.1 || power_headroom < 1.0) {
        // Performance is above target or power is constrained
        strategy->dvfs_changes.decrease_frequency = true;
        strategy->dvfs_changes.frequency_step = (int)(fabs(performance_gap) * 5);
        if (strategy->dvfs_changes.frequency_step > 2) {
            strategy->dvfs_changes.frequency_step = 2;
        }
        
        strategy->dvfs_changes.decrease_voltage = true;
        strategy->dvfs_changes.voltage_step = strategy->dvfs_changes.frequency_step;
    }
    
    // Cache optimization
    if (metrics->cache_hit_rate < 0.85) {
        strategy->cache_changes.increase_prefetch_aggressiveness = true;
        strategy->cache_changes.prefetch_step = (int)((0.85 - metrics->cache_hit_rate) * 20);
        if (strategy->cache_changes.prefetch_step > 4) {
            strategy->cache_changes.prefetch_step = 4;
        }
    } else if (metrics->cache_hit_rate > 0.95) {
        strategy->cache_changes.decrease_prefetch_aggressiveness = true;
        strategy->cache_changes.prefetch_step = 1;
    }
    
    // Memory scheduling optimization
    if (profile->workload_type == WORKLOAD_TYPE_MEMORY_INTENSIVE) {
        strategy->memory_changes.prioritize_memory_requests = true;
        strategy->memory_changes.memory_scheduler_policy = 3; // Memory-intensive policy
    } else if (profile->workload_type == WORKLOAD_TYPE_AI_INTENSIVE) {
        strategy->memory_changes.prioritize_ai_requests = true;
        strategy->memory_changes.memory_scheduler_policy = 4; // AI-intensive policy
    } else {
        strategy->memory_changes.memory_scheduler_policy = 1; // Balanced policy
    }
    
    // Resource allocation optimization
    if (metrics->tpu_utilization < 0.5 && profile->ai_workload_percentage > 0.7) {
        strategy->resource_changes.enable_more_ai_units = true;
        strategy->resource_changes.ai_unit_count_change = 1;
    } else if (metrics->tpu_utilization > 0.9) {
        strategy->resource_changes.enable_more_ai_units = true;
        strategy->resource_changes.ai_unit_count_change = 1;
    }
    
    // Power gating optimization
    if (metrics->cpu_utilization < 0.3) {
        strategy->power_changes.enable_core_power_gating = true;
        strategy->power_changes.cores_to_gate = (int)((0.3 - metrics->cpu_utilization) * 8);
        if (strategy->power_changes.cores_to_gate > 2) {
            strategy->power_changes.cores_to_gate = 2;
        }
    }
    
    if (metrics->tpu_utilization < 0.2) {
        strategy->power_changes.enable_ai_power_gating = true;
        strategy->power_changes.ai_units_to_gate = 1;
    }
    
    // NoC optimization
    if (profile->workload_type == WORKLOAD_TYPE_AI_INTENSIVE) {
        strategy->noc_changes.routing_policy = 2; // High-bandwidth routing
    } else if (profile->workload_type == WORKLOAD_TYPE_REAL_TIME) {
        strategy->noc_changes.routing_policy = 1; // Low-latency routing
    } else {
        strategy->noc_changes.routing_policy = 0; // Adaptive routing
    }
    
    // Set strategy metadata
    strategy->confidence_level = 0.8; // Default confidence
    strategy->expected_performance_gain = performance_gap * 0.5; // Conservative estimate
    strategy->expected_power_change = 0.0; // Will be calculated by prediction
    
    gettimeofday(&strategy->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

int update_recommendations(optimization_strategy_t* strategy, 
                         optimization_results_t* results) {
    if (!strategy || !results) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Update current recommendations based on successful optimizations
    optimization_recommendations_t* rec = &g_optimizer.current_recommendations;
    
    if (results->performance_improvement > 0.05) {
        // Successful optimization - update recommendations
        
        if (strategy->dvfs_changes.increase_frequency) {
            rec->recommended_frequency_level += strategy->dvfs_changes.frequency_step;
            if (rec->recommended_frequency_level > 7) {
                rec->recommended_frequency_level = 7;
            }
        } else if (strategy->dvfs_changes.decrease_frequency) {
            rec->recommended_frequency_level -= strategy->dvfs_changes.frequency_step;
            if (rec->recommended_frequency_level < 0) {
                rec->recommended_frequency_level = 0;
            }
        }
        
        if (strategy->dvfs_changes.increase_voltage) {
            rec->recommended_voltage_level += strategy->dvfs_changes.voltage_step;
            if (rec->recommended_voltage_level > 7) {
                rec->recommended_voltage_level = 7;
            }
        } else if (strategy->dvfs_changes.decrease_voltage) {
            rec->recommended_voltage_level -= strategy->dvfs_changes.voltage_step;
            if (rec->recommended_voltage_level < 0) {
                rec->recommended_voltage_level = 0;
            }
        }
        
        if (strategy->cache_changes.increase_prefetch_aggressiveness) {
            rec->cache_prefetch_aggressiveness += strategy->cache_changes.prefetch_step;
            if (rec->cache_prefetch_aggressiveness > 7) {
                rec->cache_prefetch_aggressiveness = 7;
            }
        } else if (strategy->cache_changes.decrease_prefetch_aggressiveness) {
            rec->cache_prefetch_aggressiveness -= strategy->cache_changes.prefetch_step;
            if (rec->cache_prefetch_aggressiveness < 0) {
                rec->cache_prefetch_aggressiveness = 0;
            }
        }
        
        rec->memory_scheduler_policy = strategy->memory_changes.memory_scheduler_policy;
        rec->noc_routing_policy = strategy->noc_changes.routing_policy;
        
        rec->confidence_score = (rec->confidence_score * 0.9) + (results->confidence * 0.1);
        
    } else {
        // Unsuccessful optimization - reduce confidence
        rec->confidence_score *= 0.95;
    }
    
    // Update recommendation timestamp
    gettimeofday(&rec->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

// Helper function implementations
static int analyze_workload_characteristics(workload_profile_t* profile) {
    if (!profile) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    memset(profile, 0, sizeof(workload_profile_t));
    
    // Collect current performance metrics
    performance_metrics_t metrics;
    if (collect_performance_metrics(&metrics) != PERF_OPT_SUCCESS) {
        return PERF_OPT_ERROR_COLLECTION;
    }
    
    // Advanced workload characterization with multiple dimensions
    double cpu_score = 0.0;
    double memory_score = 0.0;
    double ai_score = 0.0;
    double interactive_score = 0.0;
    double batch_score = 0.0;
    
    // CPU intensity analysis
    cpu_score = (metrics.ipc / 2.0) * 0.4 + 
                (metrics.cpu_utilization) * 0.4 + 
                (1.0 - metrics.cache_hit_rate) * 0.2; // Cache misses indicate CPU work
    
    // Memory intensity analysis
    memory_score = (metrics.memory_bandwidth / 20e9) * 0.5 + 
                   (1.0 - metrics.cache_hit_rate) * 0.3 + 
                   (metrics.memory_accesses / 1e9) * 0.2; // Normalize to 1B accesses
    
    // AI workload analysis
    ai_score = (metrics.tpu_utilization + metrics.vpu_utilization) / 2.0 * 0.6 + 
               (metrics.tpu_operations / 1e9) * 0.2 + // Normalize to 1B ops
               (metrics.vpu_operations / 1e9) * 0.2;
    
    // Interactive vs batch workload detection
    // Interactive workloads typically have:
    // - Variable performance patterns
    // - Lower sustained utilization
    // - More frequent context switches (approximated by cache behavior)
    if (metrics.performance_score > 0.0) {
        // Calculate workload variability (simplified)
        double utilization_avg = (metrics.cpu_utilization + metrics.tpu_utilization + 
                                 metrics.vpu_utilization) / 3.0;
        
        if (utilization_avg < 0.6 && metrics.cache_hit_rate < 0.9) {
            interactive_score = 0.8; // Likely interactive
            batch_score = 0.2;
        } else if (utilization_avg > 0.8 && metrics.cache_hit_rate > 0.85) {
            batch_score = 0.9; // Likely batch
            interactive_score = 0.1;
        } else {
            interactive_score = 0.5; // Mixed or unclear
            batch_score = 0.5;
        }
    }
    
    // Determine primary workload type based on highest score
    double max_score = cpu_score;
    profile->workload_type = WORKLOAD_TYPE_CPU_INTENSIVE;
    
    if (memory_score > max_score) {
        max_score = memory_score;
        profile->workload_type = WORKLOAD_TYPE_MEMORY_INTENSIVE;
    }
    
    if (ai_score > max_score) {
        max_score = ai_score;
        profile->workload_type = WORKLOAD_TYPE_AI_INTENSIVE;
    }
    
    // Refine classification based on interactive vs batch
    if (interactive_score > 0.7) {
        profile->workload_type = WORKLOAD_TYPE_INTERACTIVE;
    } else if (batch_score > 0.8) {
        profile->workload_type = WORKLOAD_TYPE_BATCH;
    }
    
    // Check for real-time characteristics
    // Real-time workloads have consistent performance requirements
    if (metrics.performance_score > 0.85 && cpu_score > 0.6 && interactive_score > 0.6) {
        profile->workload_type = WORKLOAD_TYPE_REAL_TIME;
    }
    
    // Mixed workload detection
    if (max_score < 0.6 || (cpu_score > 0.4 && memory_score > 0.4 && ai_score > 0.3)) {
        profile->workload_type = WORKLOAD_TYPE_MIXED;
    }
    
    // Set intensity values
    profile->cpu_intensity = cpu_score > 1.0 ? 1.0 : cpu_score;
    profile->memory_intensity = memory_score > 1.0 ? 1.0 : memory_score;
    profile->ai_workload_percentage = ai_score > 1.0 ? 1.0 : ai_score;
    
    // Calculate performance variance for stability assessment
    // This would ideally use historical data, but we'll approximate
    profile->performance_variance = fabs(metrics.performance_score - 0.8) / 0.8;
    
    // Set workload-specific targets and characteristics
    switch (profile->workload_type) {
        case WORKLOAD_TYPE_AI_INTENSIVE:
            profile->target_performance = 0.9;
            profile->average_power = 15.0; // AI workloads typically use more power
            break;
        case WORKLOAD_TYPE_REAL_TIME:
            profile->target_performance = 0.95;
            profile->average_power = 12.0; // Consistent power for predictability
            break;
        case WORKLOAD_TYPE_BATCH:
            profile->target_performance = 0.7; // Focus on efficiency
            profile->average_power = 10.0; // Lower power for efficiency
            break;
        case WORKLOAD_TYPE_INTERACTIVE:
            profile->target_performance = 0.85; // Good responsiveness
            profile->average_power = 8.0; // Variable power usage
            break;
        case WORKLOAD_TYPE_MEMORY_INTENSIVE:
            profile->target_performance = 0.75; // Memory bound
            profile->average_power = 11.0; // Memory access power
            break;
        case WORKLOAD_TYPE_CPU_INTENSIVE:
            profile->target_performance = 0.88; // CPU performance critical
            profile->average_power = 13.0; // High CPU power
            break;
        default: // MIXED
            profile->target_performance = 0.8;
            profile->average_power = 11.0;
            break;
    }
    
    // Calculate additional metrics
    profile->average_ipc = metrics.ipc;
    profile->average_cache_hit_rate = metrics.cache_hit_rate;
    
    // Workload stability assessment
    if (profile->performance_variance < 0.1) {
        // Stable workload - can be more aggressive with optimization
        profile->target_performance += 0.05;
    } else if (profile->performance_variance > 0.3) {
        // Unstable workload - be more conservative
        profile->target_performance -= 0.05;
    }
    
    // Ensure target performance is within bounds
    if (profile->target_performance > 0.98) profile->target_performance = 0.98;
    if (profile->target_performance < 0.5) profile->target_performance = 0.5;
    
    profile->valid = true;
    gettimeofday(&profile->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

static int predict_performance_impact(optimization_strategy_t* strategy, 
                                    performance_prediction_t* prediction) {
    if (!strategy || !prediction) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    memset(prediction, 0, sizeof(performance_prediction_t));
    
    // Advanced performance prediction model with workload awareness
    double performance_change = 0.0;
    double power_change = 0.0;
    double temperature_change = 0.0;
    double confidence = 0.8; // Start with high confidence
    
    // Get current workload characteristics for better prediction
    double cpu_intensity = 0.5; // Default values
    double ai_intensity = 0.3;
    double memory_intensity = 0.4;
    
    // Analyze current metrics to determine workload characteristics
    if (g_optimizer.current_metrics.cpu_utilization > 0.7) {
        cpu_intensity = g_optimizer.current_metrics.cpu_utilization;
    }
    if (g_optimizer.current_metrics.tpu_utilization > 0.3 || 
        g_optimizer.current_metrics.vpu_utilization > 0.3) {
        ai_intensity = (g_optimizer.current_metrics.tpu_utilization + 
                       g_optimizer.current_metrics.vpu_utilization) / 2.0;
    }
    if (g_optimizer.current_metrics.memory_bandwidth > 10e9) {
        memory_intensity = g_optimizer.current_metrics.memory_bandwidth / 20e9;
        if (memory_intensity > 1.0) memory_intensity = 1.0;
    }
    
    // DVFS impact prediction with workload-specific scaling
    if (strategy->dvfs_changes.increase_frequency) {
        double freq_impact = strategy->dvfs_changes.frequency_step * 0.15;
        
        // Scale based on workload type
        if (cpu_intensity > 0.7) {
            freq_impact *= 1.3; // CPU-intensive workloads benefit more
        } else if (ai_intensity > 0.6) {
            freq_impact *= 1.1; // AI workloads benefit moderately
        } else if (memory_intensity > 0.7) {
            freq_impact *= 0.8; // Memory-bound workloads benefit less
        }
        
        performance_change += freq_impact;
        power_change += strategy->dvfs_changes.frequency_step * 2.0;
        temperature_change += strategy->dvfs_changes.frequency_step * 1.5; // 1.5C per step
        
    } else if (strategy->dvfs_changes.decrease_frequency) {
        double freq_impact = strategy->dvfs_changes.frequency_step * 0.12;
        
        // Workload-specific impact
        if (cpu_intensity > 0.7) {
            freq_impact *= 1.4; // CPU workloads hurt more by freq reduction
        } else if (memory_intensity > 0.7) {
            freq_impact *= 0.7; // Memory workloads less affected
        }
        
        performance_change -= freq_impact;
        power_change -= strategy->dvfs_changes.frequency_step * 1.8;
        temperature_change -= strategy->dvfs_changes.frequency_step * 1.2;
    }
    
    if (strategy->dvfs_changes.increase_voltage) {
        power_change += strategy->dvfs_changes.voltage_step * 1.5;
        temperature_change += strategy->dvfs_changes.voltage_step * 0.8;
    } else if (strategy->dvfs_changes.decrease_voltage) {
        power_change -= strategy->dvfs_changes.voltage_step * 1.3;
        temperature_change -= strategy->dvfs_changes.voltage_step * 0.7;
        
        // Voltage reduction may impact performance at high frequencies
        if (g_optimizer.current_metrics.cpu_utilization > 0.8) {
            performance_change -= strategy->dvfs_changes.voltage_step * 0.02;
        }
    }
    
    // Cache optimization impact with workload consideration
    if (strategy->cache_changes.increase_prefetch_aggressiveness) {
        double cache_impact = strategy->cache_changes.prefetch_step * 0.05;
        
        // Memory-intensive workloads benefit more from prefetching
        if (memory_intensity > 0.6) {
            cache_impact *= 1.5;
        } else if (cpu_intensity > 0.8) {
            cache_impact *= 1.2; // CPU workloads also benefit
        }
        
        performance_change += cache_impact;
        power_change += strategy->cache_changes.prefetch_step * 0.5;
        
    } else if (strategy->cache_changes.decrease_prefetch_aggressiveness) {
        double cache_impact = strategy->cache_changes.prefetch_step * 0.03;
        
        if (memory_intensity > 0.6) {
            cache_impact *= 1.8; // Memory workloads hurt more
        }
        
        performance_change -= cache_impact;
        power_change -= strategy->cache_changes.prefetch_step * 0.3;
    }
    
    // Resource allocation impact
    if (strategy->resource_changes.enable_more_ai_units) {
        double ai_impact = strategy->resource_changes.ai_unit_count_change * 0.3;
        
        // Scale based on AI workload intensity
        ai_impact *= (0.5 + ai_intensity); // 0.5x to 1.5x scaling
        
        performance_change += ai_impact;
        power_change += strategy->resource_changes.ai_unit_count_change * 5.0;
        temperature_change += strategy->resource_changes.ai_unit_count_change * 3.0;
    }
    
    if (strategy->resource_changes.enable_more_cores) {
        double core_impact = strategy->resource_changes.core_count_change * 0.25;
        
        // Scale based on CPU workload intensity and parallelizability
        if (cpu_intensity > 0.8) {
            core_impact *= 1.2; // High CPU utilization benefits from more cores
        } else if (cpu_intensity < 0.4) {
            core_impact *= 0.6; // Low utilization doesn't benefit much
        }
        
        performance_change += core_impact;
        power_change += strategy->resource_changes.core_count_change * 2.5;
        temperature_change += strategy->resource_changes.core_count_change * 2.0;
    }
    
    // Power gating impact
    if (strategy->power_changes.enable_core_power_gating) {
        double gating_impact = strategy->power_changes.cores_to_gate * 0.1;
        
        // Impact depends on current utilization
        if (cpu_intensity < 0.3) {
            gating_impact *= 0.5; // Low impact if already underutilized
            confidence *= 1.1; // Higher confidence for obvious decision
        } else if (cpu_intensity > 0.7) {
            gating_impact *= 1.5; // Higher impact if currently well-utilized
            confidence *= 0.8; // Lower confidence for aggressive gating
        }
        
        performance_change -= gating_impact;
        power_change -= strategy->power_changes.cores_to_gate * 2.5;
        temperature_change -= strategy->power_changes.cores_to_gate * 1.8;
    }
    
    if (strategy->power_changes.enable_ai_power_gating) {
        double ai_gating_impact = strategy->power_changes.ai_units_to_gate * 0.2;
        
        // Scale based on AI utilization
        if (ai_intensity < 0.2) {
            ai_gating_impact *= 0.4;
            confidence *= 1.2;
        } else if (ai_intensity > 0.6) {
            ai_gating_impact *= 1.6;
            confidence *= 0.7;
        }
        
        performance_change -= ai_gating_impact;
        power_change -= strategy->power_changes.ai_units_to_gate * 4.0;
        temperature_change -= strategy->power_changes.ai_units_to_gate * 3.2;
    }
    
    // Memory and NoC optimization impact
    if (strategy->memory_changes.memory_scheduler_policy != 1) {
        // Non-default memory policy
        if (memory_intensity > 0.6) {
            performance_change += 0.08; // 8% improvement for memory-intensive
            power_change += 0.5; // Slight power increase
        } else {
            performance_change += 0.03; // 3% improvement for others
        }
    }
    
    if (strategy->noc_changes.routing_policy != 0) {
        // Non-default NoC routing
        if (ai_intensity > 0.5) {
            performance_change += 0.05; // 5% improvement for AI workloads
            power_change += 0.3;
        }
    }
    
    // Apply workload-specific confidence adjustments
    if (cpu_intensity > 0.8 || ai_intensity > 0.7 || memory_intensity > 0.8) {
        confidence *= 1.1; // Higher confidence for clear workload patterns
    } else if (cpu_intensity < 0.3 && ai_intensity < 0.3 && memory_intensity < 0.3) {
        confidence *= 0.8; // Lower confidence for idle/unclear workloads
    }
    
    // Historical accuracy adjustment (simplified)
    if (g_optimizer.successful_optimizations > 10) {
        double success_rate = (double)g_optimizer.successful_optimizations / 
                             (double)g_optimizer.total_optimizations;
        confidence *= (0.5 + success_rate); // Scale confidence by historical success
    }
    
    prediction->expected_performance_gain = performance_change;
    prediction->expected_power_change = power_change;
    prediction->expected_temperature_change = temperature_change;
    prediction->confidence = confidence > 1.0 ? 1.0 : (confidence < 0.1 ? 0.1 : confidence);
    
    // Constraint checking with enhanced logic
    if (g_optimizer.current_metrics.power_consumption + power_change > 
        g_optimizer.config.power_budget_watts) {
        prediction->power_constraint_violated = true;
        prediction->confidence *= 0.4; // Significantly reduce confidence
    }
    
    if (power_change > 3.0) {
        prediction->thermal_risk = true;
        prediction->confidence *= 0.7;
    }
    
    // Temperature constraint checking
    double current_temp = 70.0; // Default assumption
    if (g_optimizer.current_metrics.timestamp.tv_sec > 0) {
        // Use actual temperature if available (would need hardware interface)
        current_temp = 70.0 + (g_optimizer.current_metrics.power_consumption - 10.0) * 2.0;
    }
    
    if (current_temp + temperature_change > g_optimizer.config.thermal_limit_celsius) {
        prediction->thermal_risk = true;
        prediction->confidence *= 0.5;
    }
    
    // Resource constraint checking
    if (strategy->resource_changes.enable_more_cores || 
        strategy->resource_changes.enable_more_ai_units) {
        // Check if resources are actually available (simplified)
        if (g_optimizer.current_metrics.cpu_utilization > 0.9 || 
            g_optimizer.current_metrics.tpu_utilization > 0.9) {
            prediction->resource_constraint_violated = true;
            prediction->confidence *= 0.6;
        }
    }
    
    gettimeofday(&prediction->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

static int apply_optimization_strategy(optimization_strategy_t* strategy) {
    if (!strategy) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Apply DVFS changes (simplified - would write to hardware registers)
    if (strategy->dvfs_changes.increase_frequency || strategy->dvfs_changes.decrease_frequency) {
        // Write to frequency control register
        printf("Applying frequency change: %s %d steps\n", 
               strategy->dvfs_changes.increase_frequency ? "increase" : "decrease",
               strategy->dvfs_changes.frequency_step);
    }
    
    if (strategy->dvfs_changes.increase_voltage || strategy->dvfs_changes.decrease_voltage) {
        // Write to voltage control register
        printf("Applying voltage change: %s %d steps\n", 
               strategy->dvfs_changes.increase_voltage ? "increase" : "decrease",
               strategy->dvfs_changes.voltage_step);
    }
    
    // Apply cache changes
    if (strategy->cache_changes.increase_prefetch_aggressiveness || 
        strategy->cache_changes.decrease_prefetch_aggressiveness) {
        printf("Applying cache prefetch change: %s %d steps\n",
               strategy->cache_changes.increase_prefetch_aggressiveness ? "increase" : "decrease",
               strategy->cache_changes.prefetch_step);
    }
    
    // Apply memory scheduler changes
    if (strategy->memory_changes.memory_scheduler_policy != 0) {
        printf("Applying memory scheduler policy: %d\n", 
               strategy->memory_changes.memory_scheduler_policy);
    }
    
    // Apply resource allocation changes
    if (strategy->resource_changes.enable_more_ai_units) {
        printf("Enabling %d additional AI units\n", 
               strategy->resource_changes.ai_unit_count_change);
    }
    
    // Apply power gating changes
    if (strategy->power_changes.enable_core_power_gating) {
        printf("Power gating %d CPU cores\n", strategy->power_changes.cores_to_gate);
    }
    
    if (strategy->power_changes.enable_ai_power_gating) {
        printf("Power gating %d AI units\n", strategy->power_changes.ai_units_to_gate);
    }
    
    // Apply NoC changes
    if (strategy->noc_changes.routing_policy != 0) {
        printf("Applying NoC routing policy: %d\n", strategy->noc_changes.routing_policy);
    }
    
    return PERF_OPT_SUCCESS;
}

static int validate_optimization_results(optimization_results_t* results) {
    if (!results) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    memset(results, 0, sizeof(optimization_results_t));
    
    // Collect metrics before and after optimization
    performance_metrics_t after_metrics;
    if (collect_performance_metrics(&after_metrics) != PERF_OPT_SUCCESS) {
        return PERF_OPT_ERROR_COLLECTION;
    }
    
    // Calculate performance improvement
    double before_score = g_optimizer.current_metrics.performance_score;
    double after_score = after_metrics.performance_score;
    
    results->performance_improvement = after_score - before_score;
    results->power_change = after_metrics.power_consumption - 
                           g_optimizer.current_metrics.power_consumption;
    
    // Calculate confidence based on measurement stability
    results->confidence = 0.8; // Default confidence
    
    if (fabs(results->performance_improvement) > 0.1) {
        results->confidence = 0.9; // High confidence for significant changes
    } else if (fabs(results->performance_improvement) < 0.02) {
        results->confidence = 0.6; // Lower confidence for small changes
    }
    
    results->successful = (results->performance_improvement > 0.01); // 1% threshold
    
    gettimeofday(&results->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

static double calculate_performance_score(performance_metrics_t* metrics) {
    if (!metrics) {
        return 0.0;
    }
    
    // Weighted performance score calculation
    double ipc_score = metrics->ipc / 2.0; // Normalize to 2.0 IPC max
    double cache_score = metrics->cache_hit_rate;
    double ai_score = (metrics->tpu_utilization + metrics->vpu_utilization) / 2.0;
    double power_efficiency = 0.5; // Default
    
    if (metrics->power_consumption > 0) {
        power_efficiency = (ipc_score + ai_score) / (metrics->power_consumption / 10.0);
        if (power_efficiency > 1.0) power_efficiency = 1.0;
    }
    
    // Weighted combination
    double score = (ipc_score * 0.3 + cache_score * 0.2 + ai_score * 0.3 + power_efficiency * 0.2);
    
    if (score > 1.0) score = 1.0;
    if (score < 0.0) score = 0.0;
    
    return score;
}

static int detect_performance_anomalies(performance_metrics_t* metrics) {
    if (!metrics) {
        return 0;
    }
    
    int anomaly_count = 0;
    
    // Check for performance anomalies
    if (metrics->ipc < 0.5) anomaly_count++; // Very low IPC
    if (metrics->cache_hit_rate < 0.7) anomaly_count++; // Low cache hit rate
    if (metrics->power_consumption > g_optimizer.config.power_budget_watts * 1.1) {
        anomaly_count++; // Power budget exceeded
    }
    if (metrics->tpu_utilization > 0.95 && metrics->performance_score < 0.7) {
        anomaly_count++; // High utilization but low performance
    }
    
    return anomaly_count;
}
// A
dvanced optimization algorithms implementation

int genetic_algorithm_optimization(performance_metrics_t* target_metrics,
                                 optimization_strategy_t* best_strategy) {
    if (!target_metrics || !best_strategy) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Simplified genetic algorithm for optimization strategy evolution
    const int POPULATION_SIZE = 20;
    const int GENERATIONS = 10;
    const double MUTATION_RATE = 0.1;
    
    typedef struct {
        optimization_strategy_t strategy;
        double fitness;
    } individual_t;
    
    individual_t population[POPULATION_SIZE];
    individual_t new_population[POPULATION_SIZE];
    
    // Initialize population with random strategies
    for (int i = 0; i < POPULATION_SIZE; i++) {
        memset(&population[i].strategy, 0, sizeof(optimization_strategy_t));
        
        // Random DVFS settings
        population[i].strategy.dvfs_changes.frequency_step = rand() % 4;
        population[i].strategy.dvfs_changes.voltage_step = rand() % 3;
        population[i].strategy.dvfs_changes.increase_frequency = (rand() % 2) == 1;
        population[i].strategy.dvfs_changes.increase_voltage = (rand() % 2) == 1;
        
        // Random cache settings
        population[i].strategy.cache_changes.prefetch_step = rand() % 5;
        population[i].strategy.cache_changes.increase_prefetch_aggressiveness = (rand() % 2) == 1;
        
        // Random resource settings
        population[i].strategy.resource_changes.ai_unit_count_change = rand() % 3;
        population[i].strategy.resource_changes.enable_more_ai_units = (rand() % 2) == 1;
        
        // Calculate fitness (simplified)
        performance_prediction_t prediction;
        if (predict_performance_impact(&population[i].strategy, &prediction) == PERF_OPT_SUCCESS) {
            population[i].fitness = prediction.expected_performance_gain * prediction.confidence;
            
            // Penalize constraint violations
            if (prediction.power_constraint_violated) population[i].fitness -= 0.5;
            if (prediction.thermal_risk) population[i].fitness -= 0.3;
            if (prediction.resource_constraint_violated) population[i].fitness -= 0.2;
        } else {
            population[i].fitness = -1.0; // Invalid strategy
        }
    }
    
    // Evolution loop
    for (int gen = 0; gen < GENERATIONS; gen++) {
        // Selection and crossover (simplified tournament selection)
        for (int i = 0; i < POPULATION_SIZE; i += 2) {
            // Select parents
            int parent1 = 0, parent2 = 1;
            for (int j = 2; j < POPULATION_SIZE; j++) {
                if (population[j].fitness > population[parent1].fitness) parent1 = j;
                if (population[j].fitness > population[parent2].fitness && j != parent1) parent2 = j;
            }
            
            // Crossover
            new_population[i] = population[parent1];
            new_population[i+1] = population[parent2];
            
            // Simple crossover - swap some parameters
            if (rand() % 2) {
                new_population[i].strategy.dvfs_changes = population[parent2].strategy.dvfs_changes;
                new_population[i+1].strategy.dvfs_changes = population[parent1].strategy.dvfs_changes;
            }
            
            // Mutation
            if ((double)rand() / RAND_MAX < MUTATION_RATE) {
                new_population[i].strategy.dvfs_changes.frequency_step = rand() % 4;
            }
            if ((double)rand() / RAND_MAX < MUTATION_RATE) {
                new_population[i+1].strategy.cache_changes.prefetch_step = rand() % 5;
            }
        }
        
        // Evaluate new population
        for (int i = 0; i < POPULATION_SIZE; i++) {
            performance_prediction_t prediction;
            if (predict_performance_impact(&new_population[i].strategy, &prediction) == PERF_OPT_SUCCESS) {
                new_population[i].fitness = prediction.expected_performance_gain * prediction.confidence;
                
                if (prediction.power_constraint_violated) new_population[i].fitness -= 0.5;
                if (prediction.thermal_risk) new_population[i].fitness -= 0.3;
                if (prediction.resource_constraint_violated) new_population[i].fitness -= 0.2;
            } else {
                new_population[i].fitness = -1.0;
            }
        }
        
        // Replace population
        memcpy(population, new_population, sizeof(population));
    }
    
    // Find best individual
    int best_idx = 0;
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (population[i].fitness > population[best_idx].fitness) {
            best_idx = i;
        }
    }
    
    memcpy(best_strategy, &population[best_idx].strategy, sizeof(optimization_strategy_t));
    best_strategy->confidence_level = population[best_idx].fitness > 0 ? 
                                     (population[best_idx].fitness > 1.0 ? 1.0 : population[best_idx].fitness) : 0.1;
    
    return PERF_OPT_SUCCESS;
}

int reinforcement_learning_optimization(performance_metrics_t* current_metrics,
                                       optimization_strategy_t* strategy) {
    if (!current_metrics || !strategy) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Simplified Q-learning inspired optimization
    // In a real implementation, this would maintain Q-tables and use exploration/exploitation
    
    static double q_table[8][8][4]; // State x Action x Value (simplified)
    static bool q_table_initialized = false;
    static double learning_rate = 0.1;
    static double discount_factor = 0.9;
    static double epsilon = 0.1; // Exploration rate
    
    if (!q_table_initialized) {
        // Initialize Q-table with small random values
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 4; k++) {
                    q_table[i][j][k] = ((double)rand() / RAND_MAX) * 0.1;
                }
            }
        }
        q_table_initialized = true;
    }
    
    // Discretize current state (simplified)
    int performance_state = (int)(current_metrics->performance_score * 7);
    int power_state = (int)((current_metrics->power_consumption / 20.0) * 7);
    if (performance_state > 7) performance_state = 7;
    if (power_state > 7) power_state = 7;
    
    // Choose action (epsilon-greedy)
    int action = 0;
    if ((double)rand() / RAND_MAX < epsilon) {
        // Explore - random action
        action = rand() % 4;
    } else {
        // Exploit - best known action
        double best_q = q_table[performance_state][power_state][0];
        for (int i = 1; i < 4; i++) {
            if (q_table[performance_state][power_state][i] > best_q) {
                best_q = q_table[performance_state][power_state][i];
                action = i;
            }
        }
    }
    
    // Convert action to strategy
    memset(strategy, 0, sizeof(optimization_strategy_t));
    
    switch (action) {
        case 0: // Increase performance
            strategy->dvfs_changes.increase_frequency = true;
            strategy->dvfs_changes.frequency_step = 1;
            strategy->cache_changes.increase_prefetch_aggressiveness = true;
            strategy->cache_changes.prefetch_step = 1;
            break;
        case 1: // Decrease power
            strategy->dvfs_changes.decrease_frequency = true;
            strategy->dvfs_changes.frequency_step = 1;
            strategy->power_changes.enable_core_power_gating = true;
            strategy->power_changes.cores_to_gate = 1;
            break;
        case 2: // Balance performance and power
            if (current_metrics->performance_score < 0.7) {
                strategy->dvfs_changes.increase_frequency = true;
                strategy->dvfs_changes.frequency_step = 1;
            } else if (current_metrics->power_consumption > 15.0) {
                strategy->dvfs_changes.decrease_voltage = true;
                strategy->dvfs_changes.voltage_step = 1;
            }
            break;
        case 3: // Optimize for workload
            if (current_metrics->tpu_utilization > 0.5) {
                strategy->resource_changes.enable_more_ai_units = true;
                strategy->resource_changes.ai_unit_count_change = 1;
                strategy->memory_changes.prioritize_ai_requests = true;
            } else {
                strategy->cache_changes.increase_prefetch_aggressiveness = true;
                strategy->cache_changes.prefetch_step = 2;
            }
            break;
    }
    
    // Set confidence based on Q-value
    strategy->confidence_level = q_table[performance_state][power_state][action];
    if (strategy->confidence_level > 1.0) strategy->confidence_level = 1.0;
    if (strategy->confidence_level < 0.1) strategy->confidence_level = 0.1;
    
    gettimeofday(&strategy->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

int multi_objective_optimization(performance_metrics_t* metrics,
                               double performance_weight,
                               double power_weight,
                               double thermal_weight,
                               optimization_strategy_t* strategy) {
    if (!metrics || !strategy) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Normalize weights
    double total_weight = performance_weight + power_weight + thermal_weight;
    if (total_weight <= 0.0) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    performance_weight /= total_weight;
    power_weight /= total_weight;
    thermal_weight /= total_weight;
    
    memset(strategy, 0, sizeof(optimization_strategy_t));
    
    // Multi-objective optimization using weighted sum approach
    double current_performance = metrics->performance_score;
    double current_power = metrics->power_consumption;
    double current_temp = 70.0 + (current_power - 10.0) * 2.0; // Estimate temperature
    
    // Performance optimization component
    if (performance_weight > 0.3 && current_performance < 0.8) {
        strategy->dvfs_changes.increase_frequency = true;
        strategy->dvfs_changes.frequency_step = (int)((0.8 - current_performance) * 5);
        if (strategy->dvfs_changes.frequency_step > 3) strategy->dvfs_changes.frequency_step = 3;
        
        strategy->cache_changes.increase_prefetch_aggressiveness = true;
        strategy->cache_changes.prefetch_step = 2;
        
        if (metrics->tpu_utilization > 0.6 || metrics->vpu_utilization > 0.6) {
            strategy->resource_changes.enable_more_ai_units = true;
            strategy->resource_changes.ai_unit_count_change = 1;
        }
    }
    
    // Power optimization component
    if (power_weight > 0.3 && current_power > 12.0) {
        if (metrics->cpu_utilization < 0.5) {
            strategy->power_changes.enable_core_power_gating = true;
            strategy->power_changes.cores_to_gate = (int)((0.5 - metrics->cpu_utilization) * 4);
            if (strategy->power_changes.cores_to_gate > 2) strategy->power_changes.cores_to_gate = 2;
        }
        
        if (metrics->tpu_utilization < 0.3) {
            strategy->power_changes.enable_ai_power_gating = true;
            strategy->power_changes.ai_units_to_gate = 1;
        }
        
        strategy->dvfs_changes.decrease_voltage = true;
        strategy->dvfs_changes.voltage_step = 1;
    }
    
    // Thermal optimization component
    if (thermal_weight > 0.3 && current_temp > 75.0) {
        strategy->dvfs_changes.decrease_frequency = true;
        strategy->dvfs_changes.frequency_step = (int)((current_temp - 75.0) / 5.0);
        if (strategy->dvfs_changes.frequency_step > 2) strategy->dvfs_changes.frequency_step = 2;
        
        strategy->power_changes.enable_core_power_gating = true;
        strategy->power_changes.cores_to_gate = 1;
    }
    
    // Resolve conflicts between objectives
    if (strategy->dvfs_changes.increase_frequency && strategy->dvfs_changes.decrease_frequency) {
        // Conflict resolution based on weights
        if (performance_weight > power_weight + thermal_weight) {
            strategy->dvfs_changes.decrease_frequency = false;
        } else {
            strategy->dvfs_changes.increase_frequency = false;
        }
    }
    
    if (strategy->dvfs_changes.increase_voltage && strategy->dvfs_changes.decrease_voltage) {
        if (performance_weight > power_weight) {
            strategy->dvfs_changes.decrease_voltage = false;
        } else {
            strategy->dvfs_changes.increase_voltage = false;
        }
    }
    
    // Calculate expected gains for each objective
    performance_prediction_t prediction;
    if (predict_performance_impact(strategy, &prediction) == PERF_OPT_SUCCESS) {
        double perf_gain = prediction.expected_performance_gain * performance_weight;
        double power_gain = -prediction.expected_power_change * power_weight; // Negative power change is good
        double thermal_gain = -prediction.expected_temperature_change * thermal_weight;
        
        strategy->expected_performance_gain = perf_gain + power_gain + thermal_gain;
        strategy->confidence_level = prediction.confidence;
    } else {
        strategy->expected_performance_gain = 0.0;
        strategy->confidence_level = 0.1;
    }
    
    gettimeofday(&strategy->timestamp, NULL);
    
    return PERF_OPT_SUCCESS;
}

// Machine learning model training (simplified)
int train_performance_model(performance_metrics_t* training_data, int data_count) {
    if (!training_data || data_count <= 0) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Simplified model training - in practice this would use more sophisticated ML
    // For now, just calculate some basic statistics for future predictions
    
    static struct {
        double avg_ipc;
        double avg_cache_hit_rate;
        double avg_power;
        double avg_performance_score;
        int sample_count;
        bool initialized;
    } model_data = {0};
    
    if (!model_data.initialized) {
        model_data.avg_ipc = 0.0;
        model_data.avg_cache_hit_rate = 0.0;
        model_data.avg_power = 0.0;
        model_data.avg_performance_score = 0.0;
        model_data.sample_count = 0;
        model_data.initialized = true;
    }
    
    // Update running averages
    for (int i = 0; i < data_count; i++) {
        model_data.avg_ipc = (model_data.avg_ipc * model_data.sample_count + 
                             training_data[i].ipc) / (model_data.sample_count + 1);
        model_data.avg_cache_hit_rate = (model_data.avg_cache_hit_rate * model_data.sample_count + 
                                        training_data[i].cache_hit_rate) / (model_data.sample_count + 1);
        model_data.avg_power = (model_data.avg_power * model_data.sample_count + 
                               training_data[i].power_consumption) / (model_data.sample_count + 1);
        model_data.avg_performance_score = (model_data.avg_performance_score * model_data.sample_count + 
                                           training_data[i].performance_score) / (model_data.sample_count + 1);
        model_data.sample_count++;
    }
    
    return PERF_OPT_SUCCESS;
}

int predict_performance_ml(optimization_strategy_t* strategy, 
                         performance_prediction_t* prediction) {
    if (!strategy || !prediction) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Use the trained model for prediction (simplified)
    // In practice, this would use the trained ML model
    
    // For now, fall back to the enhanced physics-based prediction
    return predict_performance_impact(strategy, prediction);
}

int update_ml_model(optimization_results_t* results) {
    if (!results) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    // Update the ML model with new results (simplified)
    // In practice, this would update model weights based on prediction accuracy
    
    // For now, just use the results to update our simple statistics
    performance_metrics_t training_sample = results->after_metrics;
    return train_performance_model(&training_sample, 1);
}

// Utility function to generate performance reports
int generate_performance_report(const char* output_file) {
    if (!output_file) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        return PERF_OPT_ERROR_IO;
    }
    
    fprintf(fp, "=== RISC-V AI Accelerator Performance Report ===\n\n");
    
    // Current metrics
    performance_metrics_t current_metrics;
    if (perf_optimizer_get_metrics(&current_metrics) == PERF_OPT_SUCCESS) {
        fprintf(fp, "Current Performance Metrics:\n");
        fprintf(fp, "  Performance Score: %.3f\n", current_metrics.performance_score);
        fprintf(fp, "  IPC: %.3f\n", current_metrics.ipc);
        fprintf(fp, "  Cache Hit Rate: %.3f\n", current_metrics.cache_hit_rate);
        fprintf(fp, "  CPU Utilization: %.3f\n", current_metrics.cpu_utilization);
        fprintf(fp, "  TPU Utilization: %.3f\n", current_metrics.tpu_utilization);
        fprintf(fp, "  VPU Utilization: %.3f\n", current_metrics.vpu_utilization);
        fprintf(fp, "  Power Consumption: %.2f W\n", current_metrics.power_consumption);
        fprintf(fp, "  Memory Bandwidth: %.2f GB/s\n", current_metrics.memory_bandwidth / 1e9);
        fprintf(fp, "\n");
    }
    
    // Current recommendations
    optimization_recommendations_t recommendations;
    if (perf_optimizer_get_recommendations(&recommendations) == PERF_OPT_SUCCESS) {
        fprintf(fp, "Current Recommendations:\n");
        fprintf(fp, "  Frequency Level: %d\n", recommendations.recommended_frequency_level);
        fprintf(fp, "  Voltage Level: %d\n", recommendations.recommended_voltage_level);
        fprintf(fp, "  Cache Prefetch: %d\n", recommendations.cache_prefetch_aggressiveness);
        fprintf(fp, "  Memory Policy: %d\n", recommendations.memory_scheduler_policy);
        fprintf(fp, "  NoC Routing: %d\n", recommendations.noc_routing_policy);
        fprintf(fp, "  Confidence: %.3f\n", recommendations.confidence_score);
        fprintf(fp, "\n");
    }
    
    // Optimization statistics
    pthread_mutex_lock(&g_optimizer_mutex);
    fprintf(fp, "Optimization Statistics:\n");
    fprintf(fp, "  Total Optimizations: %lu\n", g_optimizer.total_optimizations);
    fprintf(fp, "  Successful Optimizations: %lu\n", g_optimizer.successful_optimizations);
    if (g_optimizer.total_optimizations > 0) {
        fprintf(fp, "  Success Rate: %.1f%%\n", 
                (double)g_optimizer.successful_optimizations / g_optimizer.total_optimizations * 100.0);
    }
    fprintf(fp, "  Average Performance Gain: %.3f\n", g_optimizer.average_performance_gain);
    fprintf(fp, "  Average Power Savings: %.3f W\n", g_optimizer.average_power_savings);
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    fclose(fp);
    return PERF_OPT_SUCCESS;
}

int export_optimization_history(const char* output_file) {
    if (!output_file) {
        return PERF_OPT_ERROR_INVALID_PARAM;
    }
    
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        return PERF_OPT_ERROR_IO;
    }
    
    fprintf(fp, "timestamp,performance_gain,power_change,confidence,successful\n");
    
    pthread_mutex_lock(&g_optimizer_mutex);
    for (int i = 0; i < g_optimizer.optimization_history_count; i++) {
        optimization_history_entry_t* entry = &g_optimizer.optimization_history[i];
        fprintf(fp, "%ld.%06ld,%.6f,%.6f,%.6f,%d\n",
                entry->timestamp.tv_sec, entry->timestamp.tv_usec,
                entry->results.performance_improvement,
                entry->results.power_change,
                entry->results.confidence,
                entry->results.successful ? 1 : 0);
    }
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    fclose(fp);
    return PERF_OPT_SUCCESS;
}