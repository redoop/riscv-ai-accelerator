// TPU Programming Interface Example
// Demonstrates basic usage of the TPU programming interface

#include "../lib/libtpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Example configuration
#define MATRIX_SIZE 128
#define NUM_ITERATIONS 10

// Helper function to fill matrix with random values
static void fill_matrix_random(float* matrix, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random [-1, 1]
    }
}

// Helper function to print matrix (first few elements)
static void print_matrix_sample(const char* name, const float* matrix, uint32_t rows, uint32_t cols) {
    printf("%s (%ux%u) sample:\n", name, rows, cols);
    uint32_t print_rows = (rows < 4) ? rows : 4;
    uint32_t print_cols = (cols < 4) ? cols : 4;
    
    for (uint32_t i = 0; i < print_rows; i++) {
        printf("  ");
        for (uint32_t j = 0; j < print_cols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        if (cols > 4) printf("...");
        printf("\n");
    }
    if (rows > 4) printf("  ...\n");
    printf("\n");
}

// Example 1: Basic matrix multiplication
static int example_basic_matmul(void) {
    printf("=== Example 1: Basic Matrix Multiplication ===\n");
    
    // Create TPU context
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create TPU context: %d\n", status);
        return -1;
    }
    
    // Enable profiling
    tpu_set_profiling(ctx, true);
    tpu_reset_performance_stats(ctx);
    
    // Create matrices
    tpu_matrix_t mat_a, mat_b, mat_c;
    
    status = tpu_matrix_create(&mat_a, MATRIX_SIZE, MATRIX_SIZE, AI_DTYPE_FP32, false);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create matrix A: %d\n", status);
        tpu_destroy_context(ctx);
        return -1;
    }
    
    status = tpu_matrix_create(&mat_b, MATRIX_SIZE, MATRIX_SIZE, AI_DTYPE_FP32, false);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create matrix B: %d\n", status);
        tpu_matrix_destroy(&mat_a);
        tpu_destroy_context(ctx);
        return -1;
    }
    
    status = tpu_matrix_create(&mat_c, MATRIX_SIZE, MATRIX_SIZE, AI_DTYPE_FP32, false);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create matrix C: %d\n", status);
        tpu_matrix_destroy(&mat_a);
        tpu_matrix_destroy(&mat_b);
        tpu_destroy_context(ctx);
        return -1;
    }
    
    // Fill matrices with random data
    srand(42);  // Fixed seed for reproducible results
    fill_matrix_random((float*)mat_a.data, MATRIX_SIZE, MATRIX_SIZE);
    fill_matrix_random((float*)mat_b.data, MATRIX_SIZE, MATRIX_SIZE);
    
    printf("Performing matrix multiplication: C = A * B\n");
    printf("Matrix dimensions: %ux%u\n", MATRIX_SIZE, MATRIX_SIZE);
    
    // Print sample of input matrices
    print_matrix_sample("Matrix A", (float*)mat_a.data, MATRIX_SIZE, MATRIX_SIZE);
    print_matrix_sample("Matrix B", (float*)mat_b.data, MATRIX_SIZE, MATRIX_SIZE);
    
    // Perform matrix multiplication
    clock_t start_time = clock();
    
    status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
    if (status != AI_STATUS_SUCCESS) {
        printf("Matrix multiplication failed: %d\n", status);
        goto cleanup;
    }
    
    clock_t end_time = clock();
    double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    printf("Matrix multiplication completed in %.3f seconds\n", cpu_time);
    
    // Print sample of result
    print_matrix_sample("Result C", (float*)mat_c.data, MATRIX_SIZE, MATRIX_SIZE);
    
    // Print performance statistics
    tpu_print_performance_summary(ctx);
    
cleanup:
    // Cleanup
    tpu_matrix_destroy(&mat_a);
    tpu_matrix_destroy(&mat_b);
    tpu_matrix_destroy(&mat_c);
    tpu_destroy_context(ctx);
    
    return (status == AI_STATUS_SUCCESS) ? 0 : -1;
}

// Example 2: Batch matrix multiplication
static int example_batch_matmul(void) {
    printf("=== Example 2: Batch Matrix Multiplication ===\n");
    
    const uint32_t batch_size = 4;
    const uint32_t matrix_size = 64;
    
    // Create TPU context
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create TPU context: %d\n", status);
        return -1;
    }
    
    // Create batch tensors
    uint32_t batch_shape[3] = {batch_size, matrix_size, matrix_size};
    size_t batch_size_bytes = batch_size * matrix_size * matrix_size * sizeof(float);
    
    ai_tensor_t tensor_a, tensor_b, tensor_c;
    
    ai_create_tensor(&tensor_a, AI_DTYPE_FP32, 3, batch_shape, 
                    ai_alloc_device_memory(batch_size_bytes, 64));
    ai_create_tensor(&tensor_b, AI_DTYPE_FP32, 3, batch_shape,
                    ai_alloc_device_memory(batch_size_bytes, 64));
    ai_create_tensor(&tensor_c, AI_DTYPE_FP32, 3, batch_shape,
                    ai_alloc_device_memory(batch_size_bytes, 64));
    
    if (!tensor_a.data_ptr || !tensor_b.data_ptr || !tensor_c.data_ptr) {
        printf("Failed to allocate batch tensors\n");
        goto batch_cleanup;
    }
    
    // Fill batch tensors with random data
    fill_matrix_random((float*)tensor_a.data_ptr, batch_size * matrix_size, matrix_size);
    fill_matrix_random((float*)tensor_b.data_ptr, batch_size * matrix_size, matrix_size);
    
    printf("Performing batch matrix multiplication\n");
    printf("Batch size: %u, Matrix size: %ux%u\n", batch_size, matrix_size, matrix_size);
    
    // Perform batch matrix multiplication
    clock_t start_time = clock();
    
    status = tpu_batch_matmul(ctx, &tensor_a, &tensor_b, &tensor_c,
                             batch_size, matrix_size, matrix_size, matrix_size);
    
    clock_t end_time = clock();
    double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    if (status == AI_STATUS_SUCCESS) {
        printf("Batch matrix multiplication completed in %.3f seconds\n", cpu_time);
        
        // Print sample from first batch
        printf("Sample from batch 0:\n");
        print_matrix_sample("Result C[0]", (float*)tensor_c.data_ptr, matrix_size, matrix_size);
    } else {
        printf("Batch matrix multiplication failed: %d\n", status);
    }
    
batch_cleanup:
    if (tensor_a.data_ptr) ai_free_device_memory(tensor_a.data_ptr);
    if (tensor_b.data_ptr) ai_free_device_memory(tensor_b.data_ptr);
    if (tensor_c.data_ptr) ai_free_device_memory(tensor_c.data_ptr);
    
    tpu_destroy_context(ctx);
    
    return (status == AI_STATUS_SUCCESS) ? 0 : -1;
}

// Example 3: Performance comparison
static int example_performance_comparison(void) {
    printf("=== Example 3: Performance Comparison ===\n");
    
    const uint32_t sizes[] = {32, 64, 128, 256};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Create TPU context
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to create TPU context: %d\n", status);
        return -1;
    }
    
    tpu_set_profiling(ctx, true);
    
    printf("Matrix Size | Time (ms) | GFLOPS | Utilization\n");
    printf("------------|-----------|--------|------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        uint32_t size = sizes[i];
        
        // Create matrices
        tpu_matrix_t mat_a, mat_b, mat_c;
        tpu_matrix_create(&mat_a, size, size, AI_DTYPE_FP32, false);
        tpu_matrix_create(&mat_b, size, size, AI_DTYPE_FP32, false);
        tpu_matrix_create(&mat_c, size, size, AI_DTYPE_FP32, false);
        
        // Fill with random data
        fill_matrix_random((float*)mat_a.data, size, size);
        fill_matrix_random((float*)mat_b.data, size, size);
        
        // Reset performance counters
        tpu_reset_performance_stats(ctx);
        
        // Measure time
        clock_t start_time = clock();
        
        // Perform multiple iterations for better measurement
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
            if (status != AI_STATUS_SUCCESS) {
                printf("Matrix multiplication failed for size %u: %d\n", size, status);
                break;
            }
        }
        
        clock_t end_time = clock();
        double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        double avg_time_ms = (total_time / NUM_ITERATIONS) * 1000.0;
        
        // Calculate theoretical GFLOPS
        double ops_per_matmul = 2.0 * size * size * size;  // 2*N^3 for N×N matrix multiply
        double total_ops = ops_per_matmul * NUM_ITERATIONS;
        double gflops = (total_ops / total_time) / 1e9;
        
        // Get TPU utilization
        tpu_performance_counters_t counters;
        tpu_get_performance_stats(ctx, &counters);
        
        printf("%8ux%-3u | %9.2f | %6.2f | %9.1f%%\n",
               size, size, avg_time_ms, gflops, counters.utilization);
        
        // Cleanup matrices
        tpu_matrix_destroy(&mat_a);
        tpu_matrix_destroy(&mat_b);
        tpu_matrix_destroy(&mat_c);
    }
    
    tpu_destroy_context(ctx);
    return 0;
}

// Example 4: Multi-TPU usage
static int example_multi_tpu(void) {
    printf("=== Example 4: Multi-TPU Usage ===\n");
    
    int tpu_count = tpu_get_count();
    printf("Available TPUs: %d\n", tpu_count);
    
    if (tpu_count < 2) {
        printf("Multi-TPU example requires at least 2 TPUs\n");
        return 0;  // Not an error, just skip
    }
    
    const uint32_t size = 64;
    
    // Create contexts for multiple TPUs
    tpu_context_t ctx1, ctx2;
    ai_status_t status1 = tpu_create_context(&ctx1, 0);
    ai_status_t status2 = tpu_create_context(&ctx2, 1);
    
    if (status1 != AI_STATUS_SUCCESS || status2 != AI_STATUS_SUCCESS) {
        printf("Failed to create TPU contexts\n");
        return -1;
    }
    
    // Set async execution mode
    tpu_set_execution_mode(ctx1, TPU_EXEC_ASYNC);
    tpu_set_execution_mode(ctx2, TPU_EXEC_ASYNC);
    
    // Create matrices for both TPUs
    tpu_matrix_t mat1_a, mat1_b, mat1_c;
    tpu_matrix_t mat2_a, mat2_b, mat2_c;
    
    tpu_matrix_create(&mat1_a, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat1_b, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat1_c, size, size, AI_DTYPE_FP32, false);
    
    tpu_matrix_create(&mat2_a, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat2_b, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat2_c, size, size, AI_DTYPE_FP32, false);
    
    // Fill with different random data
    srand(42);
    fill_matrix_random((float*)mat1_a.data, size, size);
    fill_matrix_random((float*)mat1_b.data, size, size);
    
    srand(123);
    fill_matrix_random((float*)mat2_a.data, size, size);
    fill_matrix_random((float*)mat2_b.data, size, size);
    
    printf("Submitting operations to both TPUs simultaneously...\n");
    
    // Submit operations to both TPUs
    clock_t start_time = clock();
    
    status1 = tpu_matrix_multiply(ctx1, &mat1_a, &mat1_b, &mat1_c, false, false);
    status2 = tpu_matrix_multiply(ctx2, &mat2_a, &mat2_b, &mat2_c, false, false);
    
    if (status1 != AI_STATUS_SUCCESS || status2 != AI_STATUS_SUCCESS) {
        printf("Failed to submit operations\n");
        goto multi_cleanup;
    }
    
    printf("Waiting for completion...\n");
    
    // Wait for both to complete
    status1 = tpu_synchronize(ctx1);
    status2 = tpu_synchronize(ctx2);
    
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    if (status1 == AI_STATUS_SUCCESS && status2 == AI_STATUS_SUCCESS) {
        printf("Both operations completed in %.3f seconds\n", total_time);
        
        // Print samples from both results
        print_matrix_sample("TPU 1 Result", (float*)mat1_c.data, size, size);
        print_matrix_sample("TPU 2 Result", (float*)mat2_c.data, size, size);
    } else {
        printf("One or more operations failed\n");
    }
    
multi_cleanup:
    // Cleanup
    tpu_matrix_destroy(&mat1_a);
    tpu_matrix_destroy(&mat1_b);
    tpu_matrix_destroy(&mat1_c);
    tpu_matrix_destroy(&mat2_a);
    tpu_matrix_destroy(&mat2_b);
    tpu_matrix_destroy(&mat2_c);
    
    tpu_destroy_context(ctx1);
    tpu_destroy_context(ctx2);
    
    return (status1 == AI_STATUS_SUCCESS && status2 == AI_STATUS_SUCCESS) ? 0 : -1;
}

// Main function
int main(void) {
    printf("TPU Programming Interface Examples\n");
    printf("==================================\n\n");
    
    // Initialize TPU subsystem
    ai_status_t status = tpu_init();
    if (status != AI_STATUS_SUCCESS) {
        printf("Failed to initialize TPU subsystem: %d\n", status);
        return -1;
    }
    
    // Print system information
    int tpu_count = tpu_get_count();
    printf("System Information:\n");
    printf("  Available TPUs: %d\n", tpu_count);
    
    for (int i = 0; i < tpu_count; i++) {
        char info[256];
        if (tpu_get_device_info(i, info, sizeof(info)) == AI_STATUS_SUCCESS) {
            printf("  %s\n", info);
        }
    }
    printf("\n");
    
    // Run examples
    int result = 0;
    
    result |= example_basic_matmul();
    printf("\n");
    
    result |= example_batch_matmul();
    printf("\n");
    
    result |= example_performance_comparison();
    printf("\n");
    
    result |= example_multi_tpu();
    printf("\n");
    
    // Cleanup
    tpu_cleanup();
    
    if (result == 0) {
        printf("All examples completed successfully!\n");
    } else {
        printf("Some examples failed.\n");
    }
    
    return result;
}