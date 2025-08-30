// TPU Programming Interface Integration Tests
// Tests the complete TPU software stack from high-level API to hardware interface

#include "../lib/libtpu.h"
#include "../drivers/tpu_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// Test configuration
#define TEST_MATRIX_SIZE 64
#define TEST_TOLERANCE 1e-5f
#define TEST_TIMEOUT_MS 5000

// Test result tracking
typedef struct {
    int tests_run;
    int tests_passed;
    int tests_failed;
} test_results_t;

static test_results_t g_results = {0};

// Helper macros
#define TEST_ASSERT(condition, message) do { \
    g_results.tests_run++; \
    if (condition) { \
        g_results.tests_passed++; \
        printf("  ✓ %s\n", message); \
    } else { \
        g_results.tests_failed++; \
        printf("  ✗ %s\n", message); \
    } \
} while(0)

#define TEST_ASSERT_STATUS(status, expected, message) \
    TEST_ASSERT((status) == (expected), message)

// ========================================
// Test Utility Functions
// ========================================

static void fill_matrix_random(float* matrix, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // Random [-1, 1]
    }
}

static void fill_matrix_identity(float* matrix, uint32_t size) {
    memset(matrix, 0, size * size * sizeof(float));
    for (uint32_t i = 0; i < size; i++) {
        matrix[i * size + i] = 1.0f;
    }
}

static bool matrices_equal(const float* a, const float* b, uint32_t rows, uint32_t cols, float tolerance) {
    for (uint32_t i = 0; i < rows * cols; i++) {
        if (fabsf(a[i] - b[i]) > tolerance) {
            printf("    Mismatch at [%u]: %.6f vs %.6f (diff: %.6f)\n", 
                   i, a[i], b[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

static void cpu_matrix_multiply(const float* a, const float* b, float* c, 
                               uint32_t m, uint32_t n, uint32_t k) {
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (uint32_t l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// ========================================
// Basic Interface Tests
// ========================================

static void test_tpu_initialization(void) {
    printf("\n--- Testing TPU Initialization ---\n");
    
    ai_status_t status = tpu_init();
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "TPU initialization");
    
    int tpu_count = tpu_get_count();
    TEST_ASSERT(tpu_count > 0, "TPU count > 0");
    printf("  Found %d TPUs\n", tpu_count);
    
    // Test device info
    for (int i = 0; i < tpu_count; i++) {
        char info[256];
        status = tpu_get_device_info(i, info, sizeof(info));
        TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Get device info");
        printf("  %s\n", info);
        
        tpu_status_t tpu_status;
        status = tpu_get_status(i, &tpu_status);
        TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Get TPU status");
        TEST_ASSERT(tpu_status.is_available, "TPU is available");
    }
}

static void test_context_management(void) {
    printf("\n--- Testing Context Management ---\n");
    
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create TPU context");
    
    status = tpu_set_execution_mode(ctx, TPU_EXEC_SYNC);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Set synchronous execution mode");
    
    status = tpu_set_profiling(ctx, true);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Enable profiling");
    
    bool is_busy = tpu_is_busy(ctx);
    TEST_ASSERT(!is_busy, "Context not busy initially");
    
    status = tpu_destroy_context(ctx);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Destroy TPU context");
}

static void test_memory_management(void) {
    printf("\n--- Testing Memory Management ---\n");
    
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create context for memory test");
    
    // Test device memory allocation
    size_t test_size = 1024 * 1024;  // 1MB
    void* device_mem = tpu_malloc(ctx, test_size, TPU_MEM_READ_WRITE);
    TEST_ASSERT(device_mem != NULL, "Allocate device memory");
    
    // Test host memory allocation
    void* host_mem = malloc(test_size);
    TEST_ASSERT(host_mem != NULL, "Allocate host memory");
    
    // Fill host memory with test pattern
    memset(host_mem, 0xAA, test_size);
    
    // Test memory copy to device
    status = tpu_memcpy(ctx, device_mem, host_mem, test_size, true);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Copy memory to device");
    
    // Clear host memory and copy back
    memset(host_mem, 0, test_size);
    status = tpu_memcpy(ctx, host_mem, device_mem, test_size, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Copy memory from device");
    
    // Verify data integrity
    bool data_correct = true;
    for (size_t i = 0; i < test_size; i++) {
        if (((uint8_t*)host_mem)[i] != 0xAA) {
            data_correct = false;
            break;
        }
    }
    TEST_ASSERT(data_correct, "Memory data integrity");
    
    // Cleanup
    tpu_free(ctx, device_mem);
    free(host_mem);
    tpu_destroy_context(ctx);
}

// ========================================
// Matrix Operation Tests
// ========================================

static void test_matrix_operations(void) {
    printf("\n--- Testing Matrix Operations ---\n");
    
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create context for matrix test");
    
    const uint32_t size = 32;  // Smaller size for faster testing
    
    // Create matrices
    tpu_matrix_t mat_a, mat_b, mat_c, mat_ref;
    status = tpu_matrix_create(&mat_a, size, size, AI_DTYPE_FP32, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create matrix A");
    
    status = tpu_matrix_create(&mat_b, size, size, AI_DTYPE_FP32, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create matrix B");
    
    status = tpu_matrix_create(&mat_c, size, size, AI_DTYPE_FP32, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create matrix C");
    
    status = tpu_matrix_create(&mat_ref, size, size, AI_DTYPE_FP32, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create reference matrix");
    
    // Fill matrices with test data
    srand(42);  // Fixed seed for reproducible results
    fill_matrix_random((float*)mat_a.data, size, size);
    fill_matrix_identity((float*)mat_b.data, size);
    
    // Test A * I = A (identity multiplication)
    status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Matrix multiplication A * I");
    
    bool result_correct = matrices_equal((float*)mat_a.data, (float*)mat_c.data, 
                                        size, size, TEST_TOLERANCE);
    TEST_ASSERT(result_correct, "A * I = A verification");
    
    // Test general matrix multiplication
    fill_matrix_random((float*)mat_b.data, size, size);
    
    // Compute reference result on CPU
    cpu_matrix_multiply((float*)mat_a.data, (float*)mat_b.data, (float*)mat_ref.data,
                       size, size, size);
    
    // Compute result on TPU
    status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "General matrix multiplication");
    
    result_correct = matrices_equal((float*)mat_ref.data, (float*)mat_c.data, 
                                   size, size, TEST_TOLERANCE);
    TEST_ASSERT(result_correct, "General matrix multiplication verification");
    
    // Test scaled multiplication: C = 2.0 * A * B + 0.5 * C
    memcpy(mat_c.data, mat_ref.data, mat_c.size_bytes);  // Initialize C with reference
    
    status = tpu_matrix_multiply_scaled(ctx, &mat_a, &mat_b, &mat_c, 
                                       2.0f, 0.5f, false, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Scaled matrix multiplication");
    
    // Verify: result should be 2.0 * ref + 0.5 * ref = 2.5 * ref
    float* c_data = (float*)mat_c.data;
    float* ref_data = (float*)mat_ref.data;
    bool scaled_correct = true;
    for (uint32_t i = 0; i < size * size; i++) {
        float expected = 2.5f * ref_data[i];
        if (fabsf(c_data[i] - expected) > TEST_TOLERANCE) {
            scaled_correct = false;
            break;
        }
    }
    TEST_ASSERT(scaled_correct, "Scaled multiplication verification");
    
    // Cleanup
    tpu_matrix_destroy(&mat_a);
    tpu_matrix_destroy(&mat_b);
    tpu_matrix_destroy(&mat_c);
    tpu_matrix_destroy(&mat_ref);
    tpu_destroy_context(ctx);
}

// ========================================
// Performance Tests
// ========================================

static void test_performance_monitoring(void) {
    printf("\n--- Testing Performance Monitoring ---\n");
    
    tpu_context_t ctx;
    ai_status_t status = tpu_create_context(&ctx, 0);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Create context for performance test");
    
    status = tpu_set_profiling(ctx, true);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Enable profiling");
    
    status = tpu_reset_performance_stats(ctx);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Reset performance counters");
    
    // Perform some operations
    const uint32_t size = 64;
    tpu_matrix_t mat_a, mat_b, mat_c;
    
    tpu_matrix_create(&mat_a, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat_b, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat_c, size, size, AI_DTYPE_FP32, false);
    
    fill_matrix_random((float*)mat_a.data, size, size);
    fill_matrix_random((float*)mat_b.data, size, size);
    
    clock_t start_time = clock();
    
    // Perform multiple operations
    for (int i = 0; i < 5; i++) {
        status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
        TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Performance test matrix multiply");
    }
    
    clock_t end_time = clock();
    double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    // Get performance statistics
    tpu_performance_counters_t counters;
    status = tpu_get_performance_stats(ctx, &counters);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Get performance statistics");
    
    TEST_ASSERT(counters.total_cycles > 0, "Total cycles > 0");
    TEST_ASSERT(counters.operations_count > 0, "Operations count > 0");
    TEST_ASSERT(counters.utilization >= 0.0f && counters.utilization <= 100.0f, 
                "Utilization in valid range");
    
    printf("  Operations: %lu, Cycles: %lu, Utilization: %.1f%%\n",
           counters.operations_count, counters.total_cycles, counters.utilization);
    printf("  Throughput: %.2f GOPS, CPU time: %.3f seconds\n",
           counters.throughput_gops, cpu_time);
    
    // Print detailed performance summary
    tpu_print_performance_summary(ctx);
    
    // Cleanup
    tpu_matrix_destroy(&mat_a);
    tpu_matrix_destroy(&mat_b);
    tpu_matrix_destroy(&mat_c);
    tpu_destroy_context(ctx);
}

// ========================================
// Error Handling Tests
// ========================================

static void test_error_handling(void) {
    printf("\n--- Testing Error Handling ---\n");
    
    // Test invalid parameters
    tpu_context_t ctx = NULL;
    ai_status_t status = tpu_create_context(&ctx, 999);  // Invalid TPU ID
    TEST_ASSERT_STATUS(status, AI_STATUS_INVALID_PARAM, "Invalid TPU ID rejection");
    
    status = tpu_create_context(&ctx, 0);
    TEST_ASSERT_STATUS(status, AI_STATUS_SUCCESS, "Valid context creation");
    
    // Test invalid matrix operations
    tpu_matrix_t mat_a, mat_b, mat_c;
    tpu_matrix_create(&mat_a, 10, 20, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat_b, 30, 10, AI_DTYPE_FP32, false);  // Incompatible dimensions
    tpu_matrix_create(&mat_c, 10, 10, AI_DTYPE_FP32, false);
    
    status = tpu_matrix_multiply(ctx, &mat_a, &mat_b, &mat_c, false, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_INVALID_PARAM, "Dimension mismatch detection");
    
    // Test data type mismatch
    tpu_matrix_t mat_d;
    tpu_matrix_create(&mat_d, 10, 10, AI_DTYPE_INT8, false);  // Different data type
    
    status = tpu_matrix_multiply(ctx, &mat_a, &mat_d, &mat_c, false, false);
    TEST_ASSERT_STATUS(status, AI_STATUS_INVALID_PARAM, "Data type mismatch detection");
    
    // Cleanup
    tpu_matrix_destroy(&mat_a);
    tpu_matrix_destroy(&mat_b);
    tpu_matrix_destroy(&mat_c);
    tpu_matrix_destroy(&mat_d);
    tpu_destroy_context(ctx);
}

// ========================================
// Stress Tests
// ========================================

static void test_concurrent_operations(void) {
    printf("\n--- Testing Concurrent Operations ---\n");
    
    int tpu_count = tpu_get_count();
    if (tpu_count < 2) {
        printf("  Skipping concurrent test (need at least 2 TPUs)\n");
        return;
    }
    
    tpu_context_t ctx1, ctx2;
    ai_status_t status1 = tpu_create_context(&ctx1, 0);
    ai_status_t status2 = tpu_create_context(&ctx2, 1);
    
    TEST_ASSERT_STATUS(status1, AI_STATUS_SUCCESS, "Create context 1");
    TEST_ASSERT_STATUS(status2, AI_STATUS_SUCCESS, "Create context 2");
    
    // Set async mode
    tpu_set_execution_mode(ctx1, TPU_EXEC_ASYNC);
    tpu_set_execution_mode(ctx2, TPU_EXEC_ASYNC);
    
    const uint32_t size = 32;
    
    // Create matrices for both contexts
    tpu_matrix_t mat1_a, mat1_b, mat1_c;
    tpu_matrix_t mat2_a, mat2_b, mat2_c;
    
    tpu_matrix_create(&mat1_a, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat1_b, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat1_c, size, size, AI_DTYPE_FP32, false);
    
    tpu_matrix_create(&mat2_a, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat2_b, size, size, AI_DTYPE_FP32, false);
    tpu_matrix_create(&mat2_c, size, size, AI_DTYPE_FP32, false);
    
    fill_matrix_random((float*)mat1_a.data, size, size);
    fill_matrix_random((float*)mat1_b.data, size, size);
    fill_matrix_random((float*)mat2_a.data, size, size);
    fill_matrix_random((float*)mat2_b.data, size, size);
    
    // Submit operations to both TPUs
    status1 = tpu_matrix_multiply(ctx1, &mat1_a, &mat1_b, &mat1_c, false, false);
    status2 = tpu_matrix_multiply(ctx2, &mat2_a, &mat2_b, &mat2_c, false, false);
    
    TEST_ASSERT_STATUS(status1, AI_STATUS_SUCCESS, "Submit operation to TPU 1");
    TEST_ASSERT_STATUS(status2, AI_STATUS_SUCCESS, "Submit operation to TPU 2");
    
    // Wait for completion
    status1 = tpu_synchronize(ctx1);
    status2 = tpu_synchronize(ctx2);
    
    TEST_ASSERT_STATUS(status1, AI_STATUS_SUCCESS, "TPU 1 synchronization");
    TEST_ASSERT_STATUS(status2, AI_STATUS_SUCCESS, "TPU 2 synchronization");
    
    // Cleanup
    tpu_matrix_destroy(&mat1_a);
    tpu_matrix_destroy(&mat1_b);
    tpu_matrix_destroy(&mat1_c);
    tpu_matrix_destroy(&mat2_a);
    tpu_matrix_destroy(&mat2_b);
    tpu_matrix_destroy(&mat2_c);
    
    tpu_destroy_context(ctx1);
    tpu_destroy_context(ctx2);
}

// ========================================
// Main Test Runner
// ========================================

int main(void) {
    printf("=== TPU Programming Interface Integration Tests ===\n");
    
    // Initialize test environment
    srand(time(NULL));
    
    // Run test suites
    test_tpu_initialization();
    test_context_management();
    test_memory_management();
    test_matrix_operations();
    test_performance_monitoring();
    test_error_handling();
    test_concurrent_operations();
    
    // Cleanup
    tpu_cleanup();
    
    // Print results
    printf("\n=== Test Results ===\n");
    printf("Tests Run:    %d\n", g_results.tests_run);
    printf("Tests Passed: %d\n", g_results.tests_passed);
    printf("Tests Failed: %d\n", g_results.tests_failed);
    
    if (g_results.tests_failed == 0) {
        printf("All tests PASSED! ✓\n");
        return 0;
    } else {
        printf("Some tests FAILED! ✗\n");
        return 1;
    }
}