// TPU Programming Interface Implementation
// Provides specialized interface for Tensor Processing Unit operations

#include "tpu_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <math.h>

// TPU hardware register offsets
#define TPU_BASE_OFFSET         0x1000
#define TPU_STRIDE              0x400   // 1KB per TPU

// TPU register offsets (relative to TPU base)
#define TPU_REG_CONTROL         0x000
#define TPU_REG_STATUS          0x004
#define TPU_REG_OPERATION       0x008
#define TPU_REG_DATA_TYPE       0x00C
#define TPU_REG_MATRIX_M        0x010
#define TPU_REG_MATRIX_N        0x014
#define TPU_REG_MATRIX_K        0x018
#define TPU_REG_ADDR_A_LOW      0x020
#define TPU_REG_ADDR_A_HIGH     0x024
#define TPU_REG_ADDR_B_LOW      0x028
#define TPU_REG_ADDR_B_HIGH     0x02C
#define TPU_REG_ADDR_C_LOW      0x030
#define TPU_REG_ADDR_C_HIGH     0x034
#define TPU_REG_TASK_ID         0x038
#define TPU_REG_ERROR_CODE      0x03C

// Performance counter registers
#define TPU_REG_PERF_CTRL       0x100
#define TPU_REG_CYCLE_COUNT_LOW 0x104
#define TPU_REG_CYCLE_COUNT_HIGH 0x108
#define TPU_REG_OP_COUNT_LOW    0x10C
#define TPU_REG_OP_COUNT_HIGH   0x110
#define TPU_REG_CACHE_HITS      0x114
#define TPU_REG_CACHE_MISSES    0x118
#define TPU_REG_MEM_READS       0x11C
#define TPU_REG_MEM_WRITES      0x120
#define TPU_REG_ERROR_COUNT     0x124
#define TPU_REG_TEMPERATURE     0x128
#define TPU_REG_POWER           0x12C

// Control register bits
#define TPU_CTRL_ENABLE         (1 << 0)
#define TPU_CTRL_START          (1 << 1)
#define TPU_CTRL_RESET          (1 << 2)
#define TPU_CTRL_INT_EN         (1 << 3)

// Status register bits
#define TPU_STATUS_READY        (1 << 0)
#define TPU_STATUS_BUSY         (1 << 1)
#define TPU_STATUS_DONE         (1 << 2)
#define TPU_STATUS_ERROR        (1 << 3)

// Performance control bits
#define TPU_PERF_ENABLE         (1 << 0)
#define TPU_PERF_RESET          (1 << 1)

// TPU driver state
typedef struct {
    int device_fd;
    void* mmio_base;
    size_t mmio_size;
    int initialized;
    uint32_t num_tpus;
    uint32_t next_task_id;
} tpu_driver_state_t;

static tpu_driver_state_t g_tpu_state = {0};

// Helper functions for register access
static inline uint32_t tpu_read_reg(uint32_t tpu_id, uint32_t offset) {
    if (!g_tpu_state.mmio_base || tpu_id >= g_tpu_state.num_tpus) return 0;
    uint32_t addr = TPU_BASE_OFFSET + (tpu_id * TPU_STRIDE) + offset;
    return *((volatile uint32_t*)((char*)g_tpu_state.mmio_base + addr));
}

static inline void tpu_write_reg(uint32_t tpu_id, uint32_t offset, uint32_t value) {
    if (!g_tpu_state.mmio_base || tpu_id >= g_tpu_state.num_tpus) return;
    uint32_t addr = TPU_BASE_OFFSET + (tpu_id * TPU_STRIDE) + offset;
    *((volatile uint32_t*)((char*)g_tpu_state.mmio_base + addr)) = value;
}

// ========================================
// TPU Interface Implementation
// ========================================

ai_status_t tpu_init(void) {
    if (g_tpu_state.initialized) {
        return AI_STATUS_SUCCESS;
    }
    
    // Initialize base AI driver first
    ai_status_t status = ai_driver_init();
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Get TPU count from base driver
    int tpu_count = ai_get_accelerator_count(AI_ACCEL_TPU);
    if (tpu_count <= 0) {
        printf("No TPUs found in system\n");
        return AI_STATUS_DEVICE_ERROR;
    }
    
    g_tpu_state.num_tpus = tpu_count;
    g_tpu_state.next_task_id = 1;
    g_tpu_state.initialized = 1;
    
    // Initialize each TPU
    for (uint32_t i = 0; i < g_tpu_state.num_tpus; i++) {
        // Reset TPU
        tpu_write_reg(i, TPU_REG_CONTROL, TPU_CTRL_RESET);
        usleep(1000);  // Wait 1ms
        tpu_write_reg(i, TPU_REG_CONTROL, TPU_CTRL_ENABLE);
        
        // Enable performance monitoring
        tpu_write_reg(i, TPU_REG_PERF_CTRL, TPU_PERF_ENABLE | TPU_PERF_RESET);
        
        // Check if TPU is ready
        uint32_t status_reg = tpu_read_reg(i, TPU_REG_STATUS);
        if (!(status_reg & TPU_STATUS_READY)) {
            printf("TPU %u not ready after initialization\n", i);
            return AI_STATUS_DEVICE_ERROR;
        }
    }
    
    printf("TPU subsystem initialized with %u TPUs\n", g_tpu_state.num_tpus);
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_cleanup(void) {
    if (!g_tpu_state.initialized) {
        return AI_STATUS_SUCCESS;
    }
    
    // Reset all TPUs
    for (uint32_t i = 0; i < g_tpu_state.num_tpus; i++) {
        tpu_write_reg(i, TPU_REG_CONTROL, TPU_CTRL_RESET);
    }
    
    memset(&g_tpu_state, 0, sizeof(g_tpu_state));
    
    // Cleanup base AI driver
    return ai_driver_cleanup();
}

int tpu_get_count(void) {
    if (!g_tpu_state.initialized) {
        return -1;
    }
    return g_tpu_state.num_tpus;
}

ai_status_t tpu_get_status(uint32_t tpu_id, tpu_status_t* status) {
    if (!g_tpu_state.initialized || !status || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    uint32_t status_reg = tpu_read_reg(tpu_id, TPU_REG_STATUS);
    
    status->is_available = (status_reg & TPU_STATUS_READY) != 0;
    status->is_busy = (status_reg & TPU_STATUS_BUSY) != 0;
    status->has_error = (status_reg & TPU_STATUS_ERROR) != 0;
    status->current_task_id = tpu_read_reg(tpu_id, TPU_REG_TASK_ID);
    status->queue_depth = 0;  // Simplified - no queue in this implementation
    
    // Read temperature and power (convert from fixed-point)
    uint32_t temp_raw = tpu_read_reg(tpu_id, TPU_REG_TEMPERATURE);
    uint32_t power_raw = tpu_read_reg(tpu_id, TPU_REG_POWER);
    
    status->temperature_celsius = (float)temp_raw / 256.0f;  // 8.8 fixed point
    status->power_watts = (float)power_raw / 1000.0f;       // mW to W
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_submit_task(const tpu_task_t* task) {
    if (!g_tpu_state.initialized || !task || task->tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Validate task parameters
    ai_status_t status = tpu_validate_task(task);
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    uint32_t tpu_id = task->tpu_id;
    
    // Check if TPU is available
    uint32_t status_reg = tpu_read_reg(tpu_id, TPU_REG_STATUS);
    if (status_reg & TPU_STATUS_BUSY) {
        return AI_STATUS_BUSY;
    }
    
    if (!(status_reg & TPU_STATUS_READY)) {
        return AI_STATUS_DEVICE_ERROR;
    }
    
    // Assign task ID if not provided
    uint32_t task_id = task->task_id;
    if (task_id == 0) {
        task_id = g_tpu_state.next_task_id++;
    }
    
    // Configure TPU registers
    tpu_write_reg(tpu_id, TPU_REG_TASK_ID, task_id);
    tpu_write_reg(tpu_id, TPU_REG_OPERATION, task->operation);
    tpu_write_reg(tpu_id, TPU_REG_DATA_TYPE, task->data_type);
    
    // Set matrix dimensions based on operation
    if (task->operation == TPU_OP_MATMUL || task->operation == TPU_OP_BATCH_MATMUL) {
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_M, task->params.matmul.m);
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_N, task->params.matmul.n);
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_K, task->params.matmul.k);
    } else if (task->operation == TPU_OP_CONV2D) {
        // For convolution, use output dimensions
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_M, task->params.conv2d.output_height);
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_N, task->params.conv2d.output_width);
        tpu_write_reg(tpu_id, TPU_REG_MATRIX_K, task->params.conv2d.output_channels);
    }
    
    // Set memory addresses
    uint64_t addr_a = (uint64_t)task->input_a.data_ptr;
    uint64_t addr_b = (uint64_t)task->input_b.data_ptr;
    uint64_t addr_c = (uint64_t)task->output.data_ptr;
    
    tpu_write_reg(tpu_id, TPU_REG_ADDR_A_LOW, addr_a & 0xFFFFFFFF);
    tpu_write_reg(tpu_id, TPU_REG_ADDR_A_HIGH, (addr_a >> 32) & 0xFFFFFFFF);
    tpu_write_reg(tpu_id, TPU_REG_ADDR_B_LOW, addr_b & 0xFFFFFFFF);
    tpu_write_reg(tpu_id, TPU_REG_ADDR_B_HIGH, (addr_b >> 32) & 0xFFFFFFFF);
    tpu_write_reg(tpu_id, TPU_REG_ADDR_C_LOW, addr_c & 0xFFFFFFFF);
    tpu_write_reg(tpu_id, TPU_REG_ADDR_C_HIGH, (addr_c >> 32) & 0xFFFFFFFF);
    
    // Start execution
    uint32_t control = TPU_CTRL_ENABLE | TPU_CTRL_START;
    if (!task->async_execution) {
        control |= TPU_CTRL_INT_EN;  // Enable interrupt for synchronous execution
    }
    tpu_write_reg(tpu_id, TPU_REG_CONTROL, control);
    
    printf("TPU %u: Task %u submitted (operation=%u)\n", tpu_id, task_id, task->operation);
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_wait_task(uint32_t task_id, uint32_t timeout_ms) {
    if (!g_tpu_state.initialized) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Find which TPU is running this task
    uint32_t tpu_id = 0;
    bool found = false;
    
    for (uint32_t i = 0; i < g_tpu_state.num_tpus; i++) {
        if (tpu_read_reg(i, TPU_REG_TASK_ID) == task_id) {
            tpu_id = i;
            found = true;
            break;
        }
    }
    
    if (!found) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    uint32_t elapsed_ms = 0;
    const uint32_t poll_interval_ms = 1;
    
    while (elapsed_ms < timeout_ms || timeout_ms == 0) {
        uint32_t status_reg = tpu_read_reg(tpu_id, TPU_REG_STATUS);
        
        if (status_reg & TPU_STATUS_ERROR) {
            uint32_t error_code = tpu_read_reg(tpu_id, TPU_REG_ERROR_CODE);
            printf("TPU %u: Task %u failed with error code: 0x%x\n", tpu_id, task_id, error_code);
            return AI_STATUS_ERROR;
        }
        
        if (status_reg & TPU_STATUS_DONE) {
            printf("TPU %u: Task %u completed successfully\n", tpu_id, task_id);
            return AI_STATUS_SUCCESS;
        }
        
        if (!(status_reg & TPU_STATUS_BUSY)) {
            return AI_STATUS_ERROR;  // Task not running
        }
        
        usleep(poll_interval_ms * 1000);
        elapsed_ms += poll_interval_ms;
    }
    
    printf("TPU %u: Task %u timed out after %u ms\n", tpu_id, task_id, timeout_ms);
    return AI_STATUS_TIMEOUT;
}

ai_status_t tpu_check_task_status(uint32_t task_id) {
    if (!g_tpu_state.initialized) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Find which TPU is running this task
    for (uint32_t i = 0; i < g_tpu_state.num_tpus; i++) {
        if (tpu_read_reg(i, TPU_REG_TASK_ID) == task_id) {
            uint32_t status_reg = tpu_read_reg(i, TPU_REG_STATUS);
            
            if (status_reg & TPU_STATUS_ERROR) {
                return AI_STATUS_ERROR;
            } else if (status_reg & TPU_STATUS_DONE) {
                return AI_STATUS_SUCCESS;
            } else if (status_reg & TPU_STATUS_BUSY) {
                return AI_STATUS_BUSY;
            }
        }
    }
    
    return AI_STATUS_INVALID_PARAM;  // Task not found
}

ai_status_t tpu_cancel_task(uint32_t task_id) {
    if (!g_tpu_state.initialized) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Find which TPU is running this task
    for (uint32_t i = 0; i < g_tpu_state.num_tpus; i++) {
        if (tpu_read_reg(i, TPU_REG_TASK_ID) == task_id) {
            tpu_write_reg(i, TPU_REG_CONTROL, TPU_CTRL_RESET);
            usleep(1000);  // Wait 1ms
            tpu_write_reg(i, TPU_REG_CONTROL, TPU_CTRL_ENABLE);
            
            printf("TPU %u: Task %u cancelled\n", i, task_id);
            return AI_STATUS_SUCCESS;
        }
    }
    
    return AI_STATUS_INVALID_PARAM;  // Task not found
}

ai_status_t tpu_reset(uint32_t tpu_id) {
    if (!g_tpu_state.initialized || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_write_reg(tpu_id, TPU_REG_CONTROL, TPU_CTRL_RESET);
    usleep(1000);  // Wait 1ms
    tpu_write_reg(tpu_id, TPU_REG_CONTROL, TPU_CTRL_ENABLE);
    
    // Re-enable performance monitoring
    tpu_write_reg(tpu_id, TPU_REG_PERF_CTRL, TPU_PERF_ENABLE | TPU_PERF_RESET);
    
    printf("TPU %u reset successfully\n", tpu_id);
    return AI_STATUS_SUCCESS;
}// ===
=====================================
// Performance Monitoring Implementation
// ========================================

ai_status_t tpu_get_performance_counters(uint32_t tpu_id, tpu_performance_counters_t* counters) {
    if (!g_tpu_state.initialized || !counters || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Read 64-bit cycle counters
    uint32_t cycles_low = tpu_read_reg(tpu_id, TPU_REG_CYCLE_COUNT_LOW);
    uint32_t cycles_high = tpu_read_reg(tpu_id, TPU_REG_CYCLE_COUNT_HIGH);
    counters->total_cycles = ((uint64_t)cycles_high << 32) | cycles_low;
    
    uint32_t ops_low = tpu_read_reg(tpu_id, TPU_REG_OP_COUNT_LOW);
    uint32_t ops_high = tpu_read_reg(tpu_id, TPU_REG_OP_COUNT_HIGH);
    counters->operations_count = ((uint64_t)ops_high << 32) | ops_low;
    
    // Read other counters
    counters->cache_hits = tpu_read_reg(tpu_id, TPU_REG_CACHE_HITS);
    counters->cache_misses = tpu_read_reg(tpu_id, TPU_REG_CACHE_MISSES);
    counters->memory_reads = tpu_read_reg(tpu_id, TPU_REG_MEM_READS);
    counters->memory_writes = tpu_read_reg(tpu_id, TPU_REG_MEM_WRITES);
    counters->error_count = tpu_read_reg(tpu_id, TPU_REG_ERROR_COUNT);
    
    // Calculate derived metrics
    if (counters->total_cycles > 0) {
        // Assume 1 GHz clock for calculations
        const float clock_freq_ghz = 1.0f;
        float time_seconds = counters->total_cycles / (clock_freq_ghz * 1e9f);
        
        counters->throughput_gops = counters->operations_count / (time_seconds * 1e9f);
        counters->utilization = (float)counters->operations_count / counters->total_cycles * 100.0f;
        
        // Estimate memory bandwidth (assuming 64-byte cache lines)
        uint64_t total_memory_ops = counters->memory_reads + counters->memory_writes;
        counters->memory_bandwidth_gb = (total_memory_ops * 64) / (time_seconds * 1e9f);
    } else {
        counters->throughput_gops = 0.0f;
        counters->utilization = 0.0f;
        counters->memory_bandwidth_gb = 0.0f;
    }
    
    // For now, set compute/memory/idle cycles to simplified estimates
    counters->compute_cycles = counters->operations_count;  // 1 cycle per op
    counters->memory_cycles = counters->memory_reads + counters->memory_writes;
    counters->idle_cycles = counters->total_cycles - counters->compute_cycles - counters->memory_cycles;
    
    counters->mac_operations = counters->operations_count;  // Most ops are MAC
    counters->overflow_count = 0;   // Would need additional hardware support
    counters->underflow_count = 0;  // Would need additional hardware support
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_reset_performance_counters(uint32_t tpu_id) {
    if (!g_tpu_state.initialized || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Reset performance counters
    tpu_write_reg(tpu_id, TPU_REG_PERF_CTRL, TPU_PERF_ENABLE | TPU_PERF_RESET);
    usleep(100);  // Wait for reset to complete
    tpu_write_reg(tpu_id, TPU_REG_PERF_CTRL, TPU_PERF_ENABLE);
    
    printf("TPU %u: Performance counters reset\n", tpu_id);
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_set_performance_monitoring(uint32_t tpu_id, bool enable) {
    if (!g_tpu_state.initialized || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    uint32_t ctrl_value = enable ? TPU_PERF_ENABLE : 0;
    tpu_write_reg(tpu_id, TPU_REG_PERF_CTRL, ctrl_value);
    
    printf("TPU %u: Performance monitoring %s\n", tpu_id, enable ? "enabled" : "disabled");
    return AI_STATUS_SUCCESS;
}

// ========================================
// Utility Functions Implementation
// ========================================

ai_status_t tpu_create_matmul_task(tpu_task_t* task, uint32_t tpu_id,
                                  uint32_t m, uint32_t n, uint32_t k,
                                  const ai_tensor_t* input_a,
                                  const ai_tensor_t* input_b,
                                  const ai_tensor_t* output,
                                  const tpu_matmul_params_t* params) {
    if (!task || !input_a || !input_b || !output || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    memset(task, 0, sizeof(tpu_task_t));
    
    task->task_id = 0;  // Will be assigned automatically
    task->tpu_id = tpu_id;
    task->operation = TPU_OP_MATMUL;
    task->data_type = input_a->dtype;
    
    // Copy tensor descriptors
    task->input_a = *input_a;
    task->input_b = *input_b;
    task->output = *output;
    
    // Set matrix multiplication parameters
    if (params) {
        task->params.matmul = *params;
    } else {
        // Default parameters
        task->params.matmul.m = m;
        task->params.matmul.n = n;
        task->params.matmul.k = k;
        task->params.matmul.transpose_a = false;
        task->params.matmul.transpose_b = false;
        task->params.matmul.accumulate = false;
        task->params.matmul.alpha = 1.0f;
        task->params.matmul.beta = 0.0f;
    }
    
    // Set default execution options
    task->priority = 4;  // Medium priority
    task->async_execution = true;
    task->timeout_ms = 5000;  // 5 second timeout
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_create_conv2d_task(tpu_task_t* task, uint32_t tpu_id,
                                  const ai_tensor_t* input,
                                  const ai_tensor_t* weights,
                                  const ai_tensor_t* bias,
                                  const ai_tensor_t* output,
                                  const tpu_conv2d_params_t* params) {
    if (!task || !input || !weights || !output || !params || tpu_id >= g_tpu_state.num_tpus) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    memset(task, 0, sizeof(tpu_task_t));
    
    task->task_id = 0;  // Will be assigned automatically
    task->tpu_id = tpu_id;
    task->operation = TPU_OP_CONV2D;
    task->data_type = input->dtype;
    
    // Copy tensor descriptors
    task->input_a = *input;    // Input activations
    task->input_b = *weights;  // Convolution weights
    task->output = *output;
    
    if (bias) {
        task->input_c = *bias;  // Bias tensor
    }
    
    // Set convolution parameters
    task->params.conv2d = *params;
    
    // Set default execution options
    task->priority = 4;  // Medium priority
    task->async_execution = true;
    task->timeout_ms = 10000;  // 10 second timeout for convolution
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_validate_task(const tpu_task_t* task) {
    if (!task) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Check TPU ID
    if (task->tpu_id >= g_tpu_state.num_tpus) {
        printf("Invalid TPU ID: %u (max: %u)\n", task->tpu_id, g_tpu_state.num_tpus - 1);
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Check data type consistency
    if (task->input_a.dtype != task->input_b.dtype || 
        task->input_a.dtype != task->output.dtype) {
        printf("Data type mismatch between tensors\n");
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Validate operation-specific parameters
    switch (task->operation) {
        case TPU_OP_MATMUL:
        case TPU_OP_BATCH_MATMUL: {
            const tpu_matmul_params_t* params = &task->params.matmul;
            
            // Check matrix dimensions
            if (params->m == 0 || params->n == 0 || params->k == 0) {
                printf("Invalid matrix dimensions: %ux%ux%u\n", params->m, params->n, params->k);
                return AI_STATUS_INVALID_PARAM;
            }
            
            // Check tensor shapes match matrix dimensions
            uint32_t expected_a_size = params->m * params->k;
            uint32_t expected_b_size = params->k * params->n;
            uint32_t expected_c_size = params->m * params->n;
            
            size_t dtype_size = ai_dtype_size(task->data_type);
            
            if (task->input_a.size_bytes < expected_a_size * dtype_size ||
                task->input_b.size_bytes < expected_b_size * dtype_size ||
                task->output.size_bytes < expected_c_size * dtype_size) {
                printf("Tensor sizes don't match matrix dimensions\n");
                return AI_STATUS_INVALID_PARAM;
            }
            break;
        }
        
        case TPU_OP_CONV2D: {
            const tpu_conv2d_params_t* params = &task->params.conv2d;
            
            // Check basic dimensions
            if (params->input_height == 0 || params->input_width == 0 || 
                params->input_channels == 0 || params->output_channels == 0 ||
                params->kernel_height == 0 || params->kernel_width == 0) {
                printf("Invalid convolution dimensions\n");
                return AI_STATUS_INVALID_PARAM;
            }
            
            // Check stride values
            if (params->stride_h == 0 || params->stride_w == 0) {
                printf("Invalid stride values\n");
                return AI_STATUS_INVALID_PARAM;
            }
            
            break;
        }
        
        default:
            printf("Unsupported operation: %u\n", task->operation);
            return AI_STATUS_INVALID_PARAM;
    }
    
    // Check memory pointers
    if (!task->input_a.data_ptr || !task->input_b.data_ptr || !task->output.data_ptr) {
        printf("Null data pointers in tensors\n");
        return AI_STATUS_INVALID_PARAM;
    }
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_calculate_optimal_tiling(uint32_t m, uint32_t n, uint32_t k,
                                        ai_data_type_t data_type,
                                        uint32_t* tile_m, uint32_t* tile_n, uint32_t* tile_k) {
    if (!tile_m || !tile_n || !tile_k) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // TPU hardware constraints
    const uint32_t MAX_TILE_SIZE = 64;  // Maximum tile dimension
    const uint32_t CACHE_SIZE_KB = 512; // TPU cache size
    
    size_t dtype_size = ai_dtype_size(data_type);
    if (dtype_size == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Calculate optimal tile sizes based on cache capacity
    uint32_t cache_elements = (CACHE_SIZE_KB * 1024) / dtype_size;
    
    // Start with maximum tile size and reduce if necessary
    *tile_m = (m < MAX_TILE_SIZE) ? m : MAX_TILE_SIZE;
    *tile_n = (n < MAX_TILE_SIZE) ? n : MAX_TILE_SIZE;
    *tile_k = (k < MAX_TILE_SIZE) ? k : MAX_TILE_SIZE;
    
    // Ensure tiles fit in cache (A + B + C tiles)
    while ((*tile_m * *tile_k) + (*tile_k * *tile_n) + (*tile_m * *tile_n) > cache_elements) {
        if (*tile_k > 1) *tile_k /= 2;
        else if (*tile_m > 1) *tile_m /= 2;
        else if (*tile_n > 1) *tile_n /= 2;
        else break;
    }
    
    // Ensure tiles are at least 1
    if (*tile_m == 0) *tile_m = 1;
    if (*tile_n == 0) *tile_n = 1;
    if (*tile_k == 0) *tile_k = 1;
    
    printf("Optimal tiling for %ux%ux%u: %ux%ux%u\n", 
           m, n, k, *tile_m, *tile_n, *tile_k);
    
    return AI_STATUS_SUCCESS;
}