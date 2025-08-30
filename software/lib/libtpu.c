// LibTPU - High-level TPU Programming Library Implementation

#include "libtpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ========================================
// Internal Data Structures
// ========================================

typedef struct tpu_context {
    uint32_t tpu_id;
    tpu_execution_mode_t exec_mode;
    bool profiling_enabled;
    uint32_t active_tasks[16];  // Track active task IDs
    uint32_t num_active_tasks;
    tpu_performance_counters_t perf_baseline;
} tpu_context_impl_t;

// ========================================
// Context Management Implementation
// ========================================

ai_status_t tpu_create_context(tpu_context_t* ctx, uint32_t tpu_id) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Initialize TPU subsystem if not already done
    ai_status_t status = tpu_init();
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Check if TPU ID is valid
    int tpu_count = tpu_get_count();
    if (tpu_count <= 0 || tpu_id >= (uint32_t)tpu_count) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Allocate context
    tpu_context_impl_t* impl = malloc(sizeof(tpu_context_impl_t));
    if (!impl) {
        return AI_STATUS_NO_MEMORY;
    }
    
    memset(impl, 0, sizeof(tpu_context_impl_t));
    impl->tpu_id = tpu_id;
    impl->exec_mode = TPU_EXEC_SYNC;
    impl->profiling_enabled = false;
    
    // Get baseline performance counters
    tpu_get_performance_counters(tpu_id, &impl->perf_baseline);
    
    *ctx = (tpu_context_t)impl;
    
    printf("TPU context created for device %u\n", tpu_id);
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_destroy_context(tpu_context_t ctx) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Wait for any pending operations
    tpu_synchronize(ctx);
    
    printf("TPU context destroyed for device %u\n", impl->tpu_id);
    free(impl);
    
    return AI_STATUS_SUCCESS;
}

ai_status_t tpu_set_execution_mode(tpu_context_t ctx, tpu_execution_mode_t mode) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    impl->exec_mode = mode;
    
    return AI_STATUS_SUCCESS;
}

// ========================================
// Memory Management Implementation
// ========================================

void* tpu_malloc(tpu_context_t ctx, size_t size, tpu_memory_flags_t flags) {
    if (!ctx || size == 0) {
        return NULL;
    }
    
    // For this implementation, use device memory allocation from base driver
    size_t alignment = 64;  // 64-byte alignment for optimal performance
    return ai_alloc_device_memory(size, alignment);
}

void tpu_free(tpu_context_t ctx, void* ptr) {
    if (!ctx || !ptr) {
        return;
    }
    
    ai_free_device_memory(ptr);
}

ai_status_t tpu_memcpy(tpu_context_t ctx, void* dst, const void* src, size_t size, bool to_device) {
    if (!ctx || !dst || !src || size == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    return ai_memcpy(dst, src, size, to_device);
}

// ========================================
// Matrix Operations Implementation
// ========================================

ai_status_t tpu_matrix_create(tpu_matrix_t* matrix, uint32_t rows, uint32_t cols, 
                             ai_data_type_t dtype, bool device_memory) {
    if (!matrix || rows == 0 || cols == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    size_t dtype_size = ai_dtype_size(dtype);
    if (dtype_size == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->dtype = dtype;
    matrix->size_bytes = rows * cols * dtype_size;
    matrix->device_memory = device_memory;
    
    if (device_memory) {
        matrix->data = ai_alloc_device_memory(matrix->size_bytes, 64);
    } else {
        matrix->data = aligned_alloc(64, matrix->size_bytes);
    }
    
    if (!matrix->data) {
        return AI_STATUS_NO_MEMORY;
    }
    
    // Initialize to zero
    memset(matrix->data, 0, matrix->size_bytes);
    
    return AI_STATUS_SUCCESS;
}

void tpu_matrix_destroy(tpu_matrix_t* matrix) {
    if (!matrix || !matrix->data) {
        return;
    }
    
    if (matrix->device_memory) {
        ai_free_device_memory(matrix->data);
    } else {
        free(matrix->data);
    }
    
    memset(matrix, 0, sizeof(tpu_matrix_t));
}

ai_status_t tpu_matrix_multiply(tpu_context_t ctx, const tpu_matrix_t* A, 
                               const tpu_matrix_t* B, tpu_matrix_t* C,
                               bool transpose_a, bool transpose_b) {
    return tpu_matrix_multiply_scaled(ctx, A, B, C, 1.0f, 0.0f, transpose_a, transpose_b);
}

ai_status_t tpu_matrix_multiply_scaled(tpu_context_t ctx, const tpu_matrix_t* A,
                                      const tpu_matrix_t* B, tpu_matrix_t* C,
                                      float alpha, float beta,
                                      bool transpose_a, bool transpose_b) {
    if (!ctx || !A || !B || !C) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Check data type consistency
    if (A->dtype != B->dtype || A->dtype != C->dtype) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Determine matrix dimensions
    uint32_t m = transpose_a ? A->cols : A->rows;
    uint32_t k = transpose_a ? A->rows : A->cols;
    uint32_t n = transpose_b ? B->rows : B->cols;
    uint32_t k2 = transpose_b ? B->cols : B->rows;
    
    // Check dimension compatibility
    if (k != k2 || C->rows != m || C->cols != n) {
        printf("Matrix dimension mismatch: A=%ux%u, B=%ux%u, C=%ux%u\n",
               A->rows, A->cols, B->rows, B->cols, C->rows, C->cols);
        return AI_STATUS_INVALID_PARAM;
    }
    
    // Create tensors from matrices
    ai_tensor_t tensor_a, tensor_b, tensor_c;
    uint32_t shape_a[2] = {A->rows, A->cols};
    uint32_t shape_b[2] = {B->rows, B->cols};
    uint32_t shape_c[2] = {C->rows, C->cols};
    
    ai_create_tensor(&tensor_a, A->dtype, 2, shape_a, A->data);
    ai_create_tensor(&tensor_b, B->dtype, 2, shape_b, B->data);
    ai_create_tensor(&tensor_c, C->dtype, 2, shape_c, C->data);
    
    // Create TPU task
    tpu_task_t task;
    tpu_matmul_params_t params = {
        .m = m, .n = n, .k = k,
        .transpose_a = transpose_a,
        .transpose_b = transpose_b,
        .accumulate = (beta != 0.0f),
        .alpha = alpha,
        .beta = beta
    };
    
    ai_status_t status = tpu_create_matmul_task(&task, impl->tpu_id, m, n, k,
                                               &tensor_a, &tensor_b, &tensor_c, &params);
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Set execution mode
    task.async_execution = (impl->exec_mode != TPU_EXEC_SYNC);
    
    // Submit task
    status = tpu_submit_task(&task);
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Track task if async
    if (task.async_execution && impl->num_active_tasks < 16) {
        impl->active_tasks[impl->num_active_tasks++] = task.task_id;
    }
    
    // Wait for completion if synchronous
    if (impl->exec_mode == TPU_EXEC_SYNC) {
        status = tpu_wait_task(task.task_id, task.timeout_ms);
    }
    
    return status;
}

// ========================================
// Neural Network Operations Implementation
// ========================================

ai_status_t tpu_conv2d(tpu_context_t ctx, const ai_tensor_t* input,
                      const ai_tensor_t* weights, const ai_tensor_t* bias,
                      ai_tensor_t* output, const tpu_conv2d_params_t* params) {
    if (!ctx || !input || !weights || !output || !params) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Create TPU task
    tpu_task_t task;
    ai_status_t status = tpu_create_conv2d_task(&task, impl->tpu_id,
                                               input, weights, bias, output, params);
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Set execution mode
    task.async_execution = (impl->exec_mode != TPU_EXEC_SYNC);
    
    // Submit task
    status = tpu_submit_task(&task);
    if (status != AI_STATUS_SUCCESS) {
        return status;
    }
    
    // Track task if async
    if (task.async_execution && impl->num_active_tasks < 16) {
        impl->active_tasks[impl->num_active_tasks++] = task.task_id;
    }
    
    // Wait for completion if synchronous
    if (impl->exec_mode == TPU_EXEC_SYNC) {
        status = tpu_wait_task(task.task_id, task.timeout_ms);
    }
    
    return status;
}

ai_status_t tpu_batch_matmul(tpu_context_t ctx, const ai_tensor_t* A,
                            const ai_tensor_t* B, ai_tensor_t* C,
                            uint32_t batch_size, uint32_t m, uint32_t n, uint32_t k) {
    if (!ctx || !A || !B || !C || batch_size == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Create tensors for each batch element and process sequentially
    // In a real implementation, this could be optimized for parallel execution
    size_t dtype_size = ai_dtype_size(A->dtype);
    size_t matrix_a_size = m * k * dtype_size;
    size_t matrix_b_size = k * n * dtype_size;
    size_t matrix_c_size = m * n * dtype_size;
    
    for (uint32_t batch = 0; batch < batch_size; batch++) {
        // Create tensor views for current batch
        ai_tensor_t batch_a = *A;
        ai_tensor_t batch_b = *B;
        ai_tensor_t batch_c = *C;
        
        batch_a.data_ptr = (char*)A->data_ptr + (batch * matrix_a_size);
        batch_b.data_ptr = (char*)B->data_ptr + (batch * matrix_b_size);
        batch_c.data_ptr = (char*)C->data_ptr + (batch * matrix_c_size);
        
        batch_a.size_bytes = matrix_a_size;
        batch_b.size_bytes = matrix_b_size;
        batch_c.size_bytes = matrix_c_size;
        
        // Update tensor shapes for 2D matrices
        batch_a.ndim = 2;
        batch_a.shape[0] = m;
        batch_a.shape[1] = k;
        
        batch_b.ndim = 2;
        batch_b.shape[0] = k;
        batch_b.shape[1] = n;
        
        batch_c.ndim = 2;
        batch_c.shape[0] = m;
        batch_c.shape[1] = n;
        
        // Create and submit task
        tpu_task_t task;
        tpu_matmul_params_t params = {
            .m = m, .n = n, .k = k,
            .transpose_a = false,
            .transpose_b = false,
            .accumulate = false,
            .alpha = 1.0f,
            .beta = 0.0f
        };
        
        ai_status_t status = tpu_create_matmul_task(&task, impl->tpu_id, m, n, k,
                                                   &batch_a, &batch_b, &batch_c, &params);
        if (status != AI_STATUS_SUCCESS) {
            return status;
        }
        
        task.async_execution = false;  // Process batches sequentially for now
        
        status = tpu_submit_task(&task);
        if (status != AI_STATUS_SUCCESS) {
            return status;
        }
        
        status = tpu_wait_task(task.task_id, task.timeout_ms);
        if (status != AI_STATUS_SUCCESS) {
            return status;
        }
    }
    
    return AI_STATUS_SUCCESS;
}

// ========================================
// Synchronization and Events Implementation
// ========================================

ai_status_t tpu_synchronize(tpu_context_t ctx) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Wait for all active tasks to complete
    for (uint32_t i = 0; i < impl->num_active_tasks; i++) {
        ai_status_t status = tpu_wait_task(impl->active_tasks[i], 0);  // No timeout
        if (status != AI_STATUS_SUCCESS) {
            printf("Task %u failed during synchronization\n", impl->active_tasks[i]);
        }
    }
    
    impl->num_active_tasks = 0;  // Clear active task list
    
    return AI_STATUS_SUCCESS;
}

bool tpu_is_busy(tpu_context_t ctx) {
    if (!ctx) {
        return false;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    // Check if any tasks are still running
    for (uint32_t i = 0; i < impl->num_active_tasks; i++) {
        ai_status_t status = tpu_check_task_status(impl->active_tasks[i]);
        if (status == AI_STATUS_BUSY) {
            return true;
        }
    }
    
    return false;
}

// ========================================
// Performance and Debugging Implementation
// ========================================

ai_status_t tpu_get_performance_stats(tpu_context_t ctx, tpu_performance_counters_t* counters) {
    if (!ctx || !counters) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    return tpu_get_performance_counters(impl->tpu_id, counters);
}

ai_status_t tpu_reset_performance_stats(tpu_context_t ctx) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    
    ai_status_t status = tpu_reset_performance_counters(impl->tpu_id);
    if (status == AI_STATUS_SUCCESS) {
        // Update baseline
        tpu_get_performance_counters(impl->tpu_id, &impl->perf_baseline);
    }
    
    return status;
}

ai_status_t tpu_set_profiling(tpu_context_t ctx, bool enable) {
    if (!ctx) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    impl->profiling_enabled = enable;
    
    return tpu_set_performance_monitoring(impl->tpu_id, enable);
}

void tpu_print_performance_summary(tpu_context_t ctx) {
    if (!ctx) {
        return;
    }
    
    tpu_context_impl_t* impl = (tpu_context_impl_t*)ctx;
    tpu_performance_counters_t counters;
    
    if (tpu_get_performance_counters(impl->tpu_id, &counters) != AI_STATUS_SUCCESS) {
        printf("Failed to get performance counters\n");
        return;
    }
    
    printf("\n=== TPU %u Performance Summary ===\n", impl->tpu_id);
    printf("Total Cycles:        %lu\n", counters.total_cycles);
    printf("Compute Cycles:      %lu (%.1f%%)\n", 
           counters.compute_cycles, 
           (float)counters.compute_cycles / counters.total_cycles * 100.0f);
    printf("Memory Cycles:       %lu (%.1f%%)\n", 
           counters.memory_cycles,
           (float)counters.memory_cycles / counters.total_cycles * 100.0f);
    printf("Idle Cycles:         %lu (%.1f%%)\n", 
           counters.idle_cycles,
           (float)counters.idle_cycles / counters.total_cycles * 100.0f);
    printf("\n");
    printf("Operations Count:    %lu\n", counters.operations_count);
    printf("MAC Operations:      %lu\n", counters.mac_operations);
    printf("Memory Reads:        %lu\n", counters.memory_reads);
    printf("Memory Writes:       %lu\n", counters.memory_writes);
    printf("\n");
    printf("Cache Hits:          %lu\n", counters.cache_hits);
    printf("Cache Misses:        %lu\n", counters.cache_misses);
    printf("Cache Hit Rate:      %.1f%%\n", 
           (float)counters.cache_hits / (counters.cache_hits + counters.cache_misses) * 100.0f);
    printf("\n");
    printf("Utilization:         %.1f%%\n", counters.utilization);
    printf("Throughput:          %.2f GOPS\n", counters.throughput_gops);
    printf("Memory Bandwidth:    %.2f GB/s\n", counters.memory_bandwidth_gb);
    printf("\n");
    printf("Error Count:         %u\n", counters.error_count);
    printf("Overflow Count:      %u\n", counters.overflow_count);
    printf("Underflow Count:     %u\n", counters.underflow_count);
    printf("================================\n\n");
}

// ========================================
// Utility Functions Implementation
// ========================================

ai_status_t tpu_get_device_info(uint32_t tpu_id, char* info, size_t info_size) {
    if (!info || info_size == 0) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    int tpu_count = tpu_get_count();
    if (tpu_count <= 0 || tpu_id >= (uint32_t)tpu_count) {
        return AI_STATUS_INVALID_PARAM;
    }
    
    tpu_status_t status;
    ai_status_t result = tpu_get_status(tpu_id, &status);
    if (result != AI_STATUS_SUCCESS) {
        return result;
    }
    
    snprintf(info, info_size,
             "TPU %u: %s, Temp: %.1f°C, Power: %.1fW, %s",
             tpu_id,
             status.is_available ? "Available" : "Unavailable",
             status.temperature_celsius,
             status.power_watts,
             status.is_busy ? "Busy" : "Idle");
    
    return AI_STATUS_SUCCESS;
}

bool tpu_is_dtype_supported(ai_data_type_t dtype) {
    switch (dtype) {
        case AI_DTYPE_INT8:
        case AI_DTYPE_FP16:
        case AI_DTYPE_FP32:
            return true;
        case AI_DTYPE_INT16:
        case AI_DTYPE_INT32:
        case AI_DTYPE_FP64:
            return false;  // Not supported by current TPU implementation
        default:
            return false;
    }
}

ai_status_t tpu_get_optimal_tile_size(uint32_t m, uint32_t n, uint32_t k,
                                     ai_data_type_t dtype,
                                     uint32_t* tile_m, uint32_t* tile_n, uint32_t* tile_k) {
    return tpu_calculate_optimal_tiling(m, n, k, dtype, tile_m, tile_n, tile_k);
}