// TPU Programming Interface Header
// Provides specialized interface for Tensor Processing Unit operations
// Supports matrix operations, convolutions, and neural network inference

#ifndef TPU_INTERFACE_H
#define TPU_INTERFACE_H

#include "ai_accel_driver.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// TPU-Specific Data Structures
// ========================================

// TPU operation types
typedef enum {
    TPU_OP_MATMUL = 0,
    TPU_OP_CONV2D = 1,
    TPU_OP_DEPTHWISE_CONV = 2,
    TPU_OP_TRANSPOSE = 3,
    TPU_OP_BATCH_MATMUL = 4
} tpu_operation_t;

// TPU data layout formats
typedef enum {
    TPU_LAYOUT_NCHW = 0,  // Batch, Channel, Height, Width
    TPU_LAYOUT_NHWC = 1,  // Batch, Height, Width, Channel
    TPU_LAYOUT_ROW_MAJOR = 2,
    TPU_LAYOUT_COL_MAJOR = 3
} tpu_data_layout_t;

// Matrix multiplication parameters
typedef struct {
    uint32_t m, n, k;           // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
    bool transpose_a;           // Transpose matrix A
    bool transpose_b;           // Transpose matrix B
    bool accumulate;            // Accumulate with existing output
    float alpha, beta;          // C = alpha * A * B + beta * C
} tpu_matmul_params_t;

// Convolution parameters
typedef struct {
    uint32_t batch_size;
    uint32_t input_height, input_width, input_channels;
    uint32_t output_height, output_width, output_channels;
    uint32_t kernel_height, kernel_width;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    uint32_t dilation_h, dilation_w;
    bool use_bias;
    tpu_data_layout_t input_layout;
    tpu_data_layout_t output_layout;
} tpu_conv2d_params_t;

// TPU task descriptor
typedef struct {
    uint32_t task_id;
    uint32_t tpu_id;            // Which TPU to use (0 or 1)
    tpu_operation_t operation;
    ai_data_type_t data_type;
    
    // Input/output tensors
    ai_tensor_t input_a;        // Primary input (activations)
    ai_tensor_t input_b;        // Secondary input (weights)
    ai_tensor_t input_c;        // Bias or accumulation input (optional)
    ai_tensor_t output;         // Output tensor
    
    // Operation parameters
    union {
        tpu_matmul_params_t matmul;
        tpu_conv2d_params_t conv2d;
    } params;
    
    // Execution options
    uint32_t priority;          // Task priority (0-7, higher = more priority)
    bool async_execution;       // Execute asynchronously
    uint32_t timeout_ms;        // Execution timeout
} tpu_task_t;

// TPU performance counters
typedef struct {
    uint64_t total_cycles;      // Total clock cycles
    uint64_t compute_cycles;    // Cycles spent computing
    uint64_t memory_cycles;     // Cycles spent on memory operations
    uint64_t idle_cycles;       // Idle cycles
    
    uint64_t operations_count;  // Total operations executed
    uint64_t mac_operations;    // Multiply-accumulate operations
    uint64_t memory_reads;      // Memory read transactions
    uint64_t memory_writes;     // Memory write transactions
    
    uint64_t cache_hits;        // Cache hit count
    uint64_t cache_misses;      // Cache miss count
    
    float utilization;          // Compute utilization percentage
    float throughput_gops;      // Throughput in GOPS
    float memory_bandwidth_gb;  // Memory bandwidth in GB/s
    
    uint32_t error_count;       // Number of errors detected
    uint32_t overflow_count;    // Arithmetic overflow count
    uint32_t underflow_count;   // Arithmetic underflow count
} tpu_performance_counters_t;

// TPU status information
typedef struct {
    bool is_available;          // TPU is available for use
    bool is_busy;              // TPU is currently executing a task
    bool has_error;            // TPU has encountered an error
    uint32_t current_task_id;  // ID of currently executing task
    uint32_t queue_depth;      // Number of queued tasks
    float temperature_celsius; // Current temperature
    float power_watts;         // Current power consumption
} tpu_status_t;

// ========================================
// TPU Interface Functions
// ========================================

/**
 * Initialize TPU subsystem
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_init(void);

/**
 * Cleanup TPU subsystem
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_cleanup(void);

/**
 * Get number of available TPUs
 * @return Number of TPUs, or -1 on error
 */
int tpu_get_count(void);

/**
 * Get TPU status
 * @param tpu_id TPU identifier (0-based)
 * @param status Pointer to status structure
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_get_status(uint32_t tpu_id, tpu_status_t* status);

/**
 * Submit a task to TPU
 * @param task Pointer to TPU task descriptor
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_submit_task(const tpu_task_t* task);

/**
 * Wait for TPU task completion
 * @param task_id Task ID to wait for
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @return AI_STATUS_SUCCESS on completion, error code otherwise
 */
ai_status_t tpu_wait_task(uint32_t task_id, uint32_t timeout_ms);

/**
 * Check TPU task status without blocking
 * @param task_id Task ID to check
 * @return Task status
 */
ai_status_t tpu_check_task_status(uint32_t task_id);

/**
 * Cancel a TPU task
 * @param task_id Task ID to cancel
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_cancel_task(uint32_t task_id);

/**
 * Reset TPU to idle state
 * @param tpu_id TPU identifier
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_reset(uint32_t tpu_id);

// ========================================
// Performance Monitoring Functions
// ========================================

/**
 * Get TPU performance counters
 * @param tpu_id TPU identifier
 * @param counters Pointer to performance counters structure
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_get_performance_counters(uint32_t tpu_id, tpu_performance_counters_t* counters);

/**
 * Reset TPU performance counters
 * @param tpu_id TPU identifier
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_reset_performance_counters(uint32_t tpu_id);

/**
 * Enable/disable performance monitoring
 * @param tpu_id TPU identifier
 * @param enable True to enable, false to disable
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_set_performance_monitoring(uint32_t tpu_id, bool enable);

// ========================================
// Utility Functions
// ========================================

/**
 * Create a matrix multiplication task
 * @param task Pointer to task structure to initialize
 * @param tpu_id TPU identifier
 * @param m, n, k Matrix dimensions
 * @param input_a Input matrix A
 * @param input_b Input matrix B
 * @param output Output matrix C
 * @param params Matrix multiplication parameters
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_create_matmul_task(tpu_task_t* task, uint32_t tpu_id,
                                  uint32_t m, uint32_t n, uint32_t k,
                                  const ai_tensor_t* input_a,
                                  const ai_tensor_t* input_b,
                                  const ai_tensor_t* output,
                                  const tpu_matmul_params_t* params);

/**
 * Create a convolution task
 * @param task Pointer to task structure to initialize
 * @param tpu_id TPU identifier
 * @param input Input tensor (activations)
 * @param weights Weight tensor
 * @param bias Bias tensor (optional, can be NULL)
 * @param output Output tensor
 * @param params Convolution parameters
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_create_conv2d_task(tpu_task_t* task, uint32_t tpu_id,
                                  const ai_tensor_t* input,
                                  const ai_tensor_t* weights,
                                  const ai_tensor_t* bias,
                                  const ai_tensor_t* output,
                                  const tpu_conv2d_params_t* params);

/**
 * Validate TPU task parameters
 * @param task Pointer to task to validate
 * @return AI_STATUS_SUCCESS if valid, error code otherwise
 */
ai_status_t tpu_validate_task(const tpu_task_t* task);

/**
 * Calculate optimal tile size for matrix operations
 * @param m, n, k Matrix dimensions
 * @param data_type Data type
 * @param tile_m, tile_n, tile_k Optimal tile dimensions (output)
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_calculate_optimal_tiling(uint32_t m, uint32_t n, uint32_t k,
                                        ai_data_type_t data_type,
                                        uint32_t* tile_m, uint32_t* tile_n, uint32_t* tile_k);

#ifdef __cplusplus
}
#endif

#endif // TPU_INTERFACE_H