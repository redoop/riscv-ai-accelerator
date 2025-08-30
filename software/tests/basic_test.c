// Basic AI Accelerator Test Suite
// Tests fundamental functionality of the AI accelerator driver and hardware

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "ai_accel_driver.h"
#include "riscv_ai_intrinsics.h"

// Test configuration
#define MATRIX_SIZE 64
#define VECTOR_SIZE 1024
#define TEST_ITERATIONS 10

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;

// Helper macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            printf("PASS: %s\n", message); \
            tests_passed++; \
        } else { \
            printf("FAIL: %s\n", message); \
            tests_failed++; \
        } \
    } while(0)

#define FLOAT_TOLERANCE 1e-5f

// Helper functions
static float random_float(void) {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static int float_equal(float a, float b, float tolerance) {
    return fabs(a - b) < tolerance;
}

static void init_random_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = random_float();
    }
}

static void init_random_vector(float* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = random_float();
    }
}

// ========================================
// Driver Interface Tests
// ========================================

void test_driver_init_cleanup(void) {
    printf("\n=== Testing Driver Initialization and Cleanup ===\n");
    
    ai_status_t status = ai_driver_init();
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Driver initialization");
    
    int tpu_count = ai_get_accelerator_count(AI_ACCEL_TPU);
    TEST_ASSERT(tpu_count == 2, "TPU count detection");
    
    int vpu_count = ai_get_accelerator_count(AI_ACCEL_VPU);
    TEST_ASSERT(vpu_count == 2, "VPU count detection");
    
    status = ai_driver_cleanup();
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Driver cleanup");
}

void test_memory_allocation(void) {
    printf("\n=== Testing Memory Allocation ===\n");
    
    ai_driver_init();
    
    // Test various allocation sizes
    void* ptr1 = ai_alloc_device_memory(1024, 64);
    TEST_ASSERT(ptr1 != NULL, "1KB allocation");
    
    void* ptr2 = ai_alloc_device_memory(1024 * 1024, 4096);
    TEST_ASSERT(ptr2 != NULL, "1MB allocation with 4KB alignment");
    
    // Test memory copy
    float test_data[256];
    init_random_vector(test_data, 256);
    
    ai_status_t status = ai_memcpy(ptr1, test_data, sizeof(test_data), true);
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Host to device memory copy");
    
    float verify_data[256];
    status = ai_memcpy(verify_data, ptr1, sizeof(verify_data), false);
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Device to host memory copy");
    
    int data_match = 1;
    for (int i = 0; i < 256; i++) {
        if (!float_equal(test_data[i], verify_data[i], FLOAT_TOLERANCE)) {
            data_match = 0;
            break;
        }
    }
    TEST_ASSERT(data_match, "Memory copy data integrity");
    
    ai_free_device_memory(ptr1);
    ai_free_device_memory(ptr2);
    
    ai_driver_cleanup();
}

void test_tensor_operations(void) {
    printf("\n=== Testing Tensor Operations ===\n");
    
    ai_tensor_t tensor;
    uint32_t shape[] = {4, 8, 16};
    float* data = malloc(4 * 8 * 16 * sizeof(float));
    
    ai_status_t status = ai_create_tensor(&tensor, AI_DTYPE_FP32, 3, shape, data);
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Tensor creation");
    
    TEST_ASSERT(tensor.ndim == 3, "Tensor dimension count");
    TEST_ASSERT(tensor.shape[0] == 4 && tensor.shape[1] == 8 && tensor.shape[2] == 16, 
                "Tensor shape");
    
    size_t expected_size = 4 * 8 * 16 * sizeof(float);
    TEST_ASSERT(ai_tensor_size_bytes(&tensor) == expected_size, "Tensor size calculation");
    
    TEST_ASSERT(ai_dtype_size(AI_DTYPE_FP32) == 4, "FP32 data type size");
    TEST_ASSERT(ai_dtype_size(AI_DTYPE_FP16) == 2, "FP16 data type size");
    TEST_ASSERT(ai_dtype_size(AI_DTYPE_INT8) == 1, "INT8 data type size");
    
    free(data);
}

// ========================================
// AI Instruction Tests
// ========================================

void test_matrix_multiplication(void) {
    printf("\n=== Testing Matrix Multiplication ===\n");
    
    ai_driver_init();
    
    // Allocate matrices
    float* A = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* B = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* C_hw = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* C_sw = malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    
    // Initialize input matrices
    init_random_matrix(A, MATRIX_SIZE, MATRIX_SIZE);
    init_random_matrix(B, MATRIX_SIZE, MATRIX_SIZE);
    
    // Software reference implementation
    clock_t start_sw = clock();
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }
            C_sw[i * MATRIX_SIZE + j] = sum;
        }
    }
    clock_t end_sw = clock();
    double time_sw = ((double)(end_sw - start_sw)) / CLOCKS_PER_SEC;
    
    // Hardware accelerated implementation
    clock_t start_hw = clock();
    __builtin_riscv_ai_matmul_f32(A, B, C_hw, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    clock_t end_hw = clock();
    double time_hw = ((double)(end_hw - start_hw)) / CLOCKS_PER_SEC;
    
    // Verify results
    int results_match = 1;
    float max_error = 0.0f;
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        float error = fabs(C_hw[i] - C_sw[i]);
        if (error > max_error) max_error = error;
        if (error > FLOAT_TOLERANCE) {
            results_match = 0;
        }
    }
    
    TEST_ASSERT(results_match, "Matrix multiplication correctness");
    printf("Max error: %e\n", max_error);
    printf("Software time: %.6f s, Hardware time: %.6f s\n", time_sw, time_hw);
    
    // Cleanup
    ai_free_device_memory(A);
    ai_free_device_memory(B);
    ai_free_device_memory(C_hw);
    free(C_sw);
    
    ai_driver_cleanup();
}

void test_activation_functions(void) {
    printf("\n=== Testing Activation Functions ===\n");
    
    float* input = malloc(VECTOR_SIZE * sizeof(float));
    float* output_hw = malloc(VECTOR_SIZE * sizeof(float));
    float* output_sw = malloc(VECTOR_SIZE * sizeof(float));
    
    init_random_vector(input, VECTOR_SIZE);
    
    // Test ReLU
    __builtin_riscv_ai_relu_f32(input, output_hw, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        output_sw[i] = fmaxf(0.0f, input[i]);
    }
    
    int relu_correct = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (!float_equal(output_hw[i], output_sw[i], FLOAT_TOLERANCE)) {
            relu_correct = 0;
            break;
        }
    }
    TEST_ASSERT(relu_correct, "ReLU activation function");
    
    // Test Sigmoid
    __builtin_riscv_ai_sigmoid_f32(input, output_hw, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        output_sw[i] = 1.0f / (1.0f + expf(-input[i]));
    }
    
    int sigmoid_correct = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (!float_equal(output_hw[i], output_sw[i], FLOAT_TOLERANCE * 10)) {  // Relaxed tolerance for transcendental functions
            sigmoid_correct = 0;
            break;
        }
    }
    TEST_ASSERT(sigmoid_correct, "Sigmoid activation function");
    
    // Test Tanh
    __builtin_riscv_ai_tanh_f32(input, output_hw, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        output_sw[i] = tanhf(input[i]);
    }
    
    int tanh_correct = 1;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (!float_equal(output_hw[i], output_sw[i], FLOAT_TOLERANCE * 10)) {  // Relaxed tolerance for transcendental functions
            tanh_correct = 0;
            break;
        }
    }
    TEST_ASSERT(tanh_correct, "Tanh activation function");
    
    free(input);
    free(output_hw);
    free(output_sw);
}

void test_task_submission(void) {
    printf("\n=== Testing Task Submission ===\n");
    
    ai_driver_init();
    
    // Create test tensors
    float* input_data = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* weight_data = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* output_data = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    
    init_random_matrix(input_data, MATRIX_SIZE, MATRIX_SIZE);
    init_random_matrix(weight_data, MATRIX_SIZE, MATRIX_SIZE);
    
    ai_tensor_t input_tensor, weight_tensor, output_tensor;
    uint32_t matrix_shape[] = {MATRIX_SIZE, MATRIX_SIZE};
    
    ai_create_tensor(&input_tensor, AI_DTYPE_FP32, 2, matrix_shape, input_data);
    ai_create_tensor(&weight_tensor, AI_DTYPE_FP32, 2, matrix_shape, weight_data);
    ai_create_tensor(&output_tensor, AI_DTYPE_FP32, 2, matrix_shape, output_data);
    
    // Create and submit task
    ai_task_t task = {
        .task_id = 1,
        .operation = AI_OP_MATMUL,
        .accel_type = AI_ACCEL_TPU,
        .accel_id = 0,
        .input_tensors = {input_tensor, weight_tensor},
        .output_tensors = {output_tensor},
        .num_inputs = 2,
        .num_outputs = 1,
        .params = NULL,
        .params_size = 0
    };
    
    ai_status_t status = ai_submit_task(&task);
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Task submission");
    
    status = ai_wait_task(1, 5000);  // 5 second timeout
    TEST_ASSERT(status == AI_STATUS_SUCCESS, "Task completion");
    
    // Cleanup
    ai_free_device_memory(input_data);
    ai_free_device_memory(weight_data);
    ai_free_device_memory(output_data);
    
    ai_driver_cleanup();
}

// ========================================
// Performance Tests
// ========================================

void test_performance_benchmarks(void) {
    printf("\n=== Performance Benchmarks ===\n");
    
    ai_driver_init();
    
    // Matrix multiplication benchmark
    float* A = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* B = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    float* C = ai_alloc_device_memory(MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64);
    
    init_random_matrix(A, MATRIX_SIZE, MATRIX_SIZE);
    init_random_matrix(B, MATRIX_SIZE, MATRIX_SIZE);
    
    clock_t start = clock();
    for (int i = 0; i < TEST_ITERATIONS; i++) {
        __builtin_riscv_ai_matmul_f32(A, B, C, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    }
    clock_t end = clock();
    
    double total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double avg_time = total_time / TEST_ITERATIONS;
    double ops = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;  // Multiply-accumulate operations
    double gflops = (ops / avg_time) / 1e9;
    
    printf("Matrix multiplication (%dx%d): %.6f s avg, %.2f GFLOPS\n", 
           MATRIX_SIZE, MATRIX_SIZE, avg_time, gflops);
    
    TEST_ASSERT(gflops > 1.0, "Matrix multiplication performance > 1 GFLOPS");
    
    ai_free_device_memory(A);
    ai_free_device_memory(B);
    ai_free_device_memory(C);
    
    ai_driver_cleanup();
}

// ========================================
// Main Test Runner
// ========================================

int main(int argc, char* argv[]) {
    printf("RISC-V AI Accelerator Test Suite\n");
    printf("================================\n");
    
    // Initialize random seed
    srand(time(NULL));
    
    // Run all tests
    test_driver_init_cleanup();
    test_memory_allocation();
    test_tensor_operations();
    test_matrix_multiplication();
    test_activation_functions();
    test_task_submission();
    test_performance_benchmarks();
    
    // Print summary
    printf("\n=== Test Summary ===\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("Total tests: %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}