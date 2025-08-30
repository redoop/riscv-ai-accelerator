// LibTPU - High-level TPU Programming Library
// Provides easy-to-use interface for TPU operations

#ifndef LIBTPU_H
#define LIBTPU_H

#include "../drivers/tpu_interface.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// High-level API Data Structures
// ========================================

// TPU context handle
typedef struct tpu_context* tpu_context_t;

// Memory allocation flags
typedef enum {
    TPU_MEM_READ_ONLY = 1,
    TPU_MEM_WRITE_ONLY = 2,
    TPU_MEM_READ_WRITE = 3,
    TPU_MEM_HOST_CACHED = 4,
    TPU_MEM_DEVICE_LOCAL = 8
} tpu_memory_flags_t;

// Execution modes
typedef enum {
    TPU_EXEC_SYNC = 0,      // Synchronous execution
    TPU_EXEC_ASYNC = 1,     // Asynchronous execution
    TPU_EXEC_STREAM = 2     // Stream-based execution
} tpu_execution_mode_t;

// Matrix handle for simplified operations
typedef struct {
    uint32_t rows, cols;
    ai_data_type_t dtype;
    void* data;
    size_t size_bytes;
    bool device_memory;
} tpu_matrix_t;

// ========================================
// Context Management
// ========================================

/**
 * Create TPU context
 * @param ctx Pointer to context handle
 * @param tpu_id TPU device ID to use
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_create_context(tpu_context_t* ctx, uint32_t tpu_id);

/**
 * Destroy TPU context
 * @param ctx Context handle
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_destroy_context(tpu_context_t ctx);

/**
 * Set execution mode for context
 * @param ctx Context handle
 * @param mode Execution mode
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_set_execution_mode(tpu_context_t ctx, tpu_execution_mode_t mode);

// ========================================
// Memory Management
// ========================================

/**
 * Allocate TPU memory
 * @param ctx Context handle
 * @param size Size in bytes
 * @param flags Memory allocation flags
 * @return Pointer to allocated memory, or NULL on failure
 */
void* tpu_malloc(tpu_context_t ctx, size_t size, tpu_memory_flags_t flags);

/**
 * Free TPU memory
 * @param ctx Context handle
 * @param ptr Pointer to memory to free
 */
void tpu_free(tpu_context_t ctx, void* ptr);

/**
 * Copy memory between host and device
 * @param ctx Context handle
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Number of bytes to copy
 * @param to_device True if copying to device, false if copying to host
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_memcpy(tpu_context_t ctx, void* dst, const void* src, size_t size, bool to_device);

// ========================================
// Matrix Operations
// ========================================

/**
 * Create matrix handle
 * @param matrix Pointer to matrix structure
 * @param rows Number of rows
 * @param cols Number of columns
 * @param dtype Data type
 * @param device_memory True to allocate on device, false for host
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_matrix_create(tpu_matrix_t* matrix, uint32_t rows, uint32_t cols, 
                             ai_data_type_t dtype, bool device_memory);

/**
 * Destroy matrix handle
 * @param matrix Pointer to matrix structure
 */
void tpu_matrix_destroy(tpu_matrix_t* matrix);

/**
 * Matrix multiplication: C = A * B
 * @param ctx Context handle
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param transpose_a Transpose matrix A
 * @param transpose_b Transpose matrix B
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_matrix_multiply(tpu_context_t ctx, const tpu_matrix_t* A, 
                               const tpu_matrix_t* B, tpu_matrix_t* C,
                               bool transpose_a, bool transpose_b);

/**
 * Matrix multiplication with scaling: C = alpha * A * B + beta * C
 * @param ctx Context handle
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C (also input for accumulation)
 * @param alpha Scaling factor for A*B
 * @param beta Scaling factor for existing C
 * @param transpose_a Transpose matrix A
 * @param transpose_b Transpose matrix B
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_matrix_multiply_scaled(tpu_context_t ctx, const tpu_matrix_t* A,
                                      const tpu_matrix_t* B, tpu_matrix_t* C,
                                      float alpha, float beta,
                                      bool transpose_a, bool transpose_b);

// ========================================
// Neural Network Operations
// ========================================

/**
 * 2D Convolution operation
 * @param ctx Context handle
 * @param input Input tensor (NCHW or NHWC format)
 * @param weights Weight tensor
 * @param bias Bias tensor (optional, can be NULL)
 * @param output Output tensor
 * @param params Convolution parameters
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_conv2d(tpu_context_t ctx, const ai_tensor_t* input,
                      const ai_tensor_t* weights, const ai_tensor_t* bias,
                      ai_tensor_t* output, const tpu_conv2d_params_t* params);

/**
 * Batch matrix multiplication for transformer models
 * @param ctx Context handle
 * @param A Input tensor A (batch of matrices)
 * @param B Input tensor B (batch of matrices)
 * @param C Output tensor C (batch of matrices)
 * @param batch_size Number of matrices in batch
 * @param m, n, k Matrix dimensions
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_batch_matmul(tpu_context_t ctx, const ai_tensor_t* A,
                            const ai_tensor_t* B, ai_tensor_t* C,
                            uint32_t batch_size, uint32_t m, uint32_t n, uint32_t k);

// ========================================
// Synchronization and Events
// ========================================

/**
 * Wait for all pending operations to complete
 * @param ctx Context handle
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_synchronize(tpu_context_t ctx);

/**
 * Check if context has pending operations
 * @param ctx Context handle
 * @return True if operations are pending, false otherwise
 */
bool tpu_is_busy(tpu_context_t ctx);

// ========================================
// Performance and Debugging
// ========================================

/**
 * Get performance statistics for context
 * @param ctx Context handle
 * @param counters Pointer to performance counters structure
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_get_performance_stats(tpu_context_t ctx, tpu_performance_counters_t* counters);

/**
 * Reset performance counters
 * @param ctx Context handle
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_reset_performance_stats(tpu_context_t ctx);

/**
 * Enable/disable profiling for context
 * @param ctx Context handle
 * @param enable True to enable profiling, false to disable
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_set_profiling(tpu_context_t ctx, bool enable);

/**
 * Print performance summary
 * @param ctx Context handle
 */
void tpu_print_performance_summary(tpu_context_t ctx);

// ========================================
// Utility Functions
// ========================================

/**
 * Get TPU device information
 * @param tpu_id TPU device ID
 * @param info Device information string (output)
 * @param info_size Size of info buffer
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_get_device_info(uint32_t tpu_id, char* info, size_t info_size);

/**
 * Check if data type is supported by TPU
 * @param dtype Data type to check
 * @return True if supported, false otherwise
 */
bool tpu_is_dtype_supported(ai_data_type_t dtype);

/**
 * Get optimal matrix tile size for given dimensions
 * @param m, n, k Matrix dimensions
 * @param dtype Data type
 * @param tile_m, tile_n, tile_k Optimal tile dimensions (output)
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t tpu_get_optimal_tile_size(uint32_t m, uint32_t n, uint32_t k,
                                     ai_data_type_t dtype,
                                     uint32_t* tile_m, uint32_t* tile_n, uint32_t* tile_k);

#ifdef __cplusplus
}
#endif

#endif // LIBTPU_H