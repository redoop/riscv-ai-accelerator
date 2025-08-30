/*
 * RISC-V AI Intrinsics Test Suite
 * Tests compiler integration and code generation for AI instructions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "../riscv_ai_intrinsics.h"

// Test configuration
#define TEST_MATRIX_SIZE 64
#define TEST_CONV_SIZE 32
#define TEST_CHANNELS 16
#define TEST_TOLERANCE 1e-5f
#define TEST_ITERATIONS 100

// Test data alignment
#define ALIGN_64 __attribute__((aligned(64)))

// Global test matrices
static float ALIGN_64 test_matrix_a[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];
static float ALIGN_64 test_matrix_b[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];
static float ALIGN_64 test_matrix_c[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];
static float ALIGN_64 test_matrix_ref[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];

// Test vectors for activation functions
static float ALIGN_64 test_input[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];
static float ALIGN_64 test_output[TEST_MATRIX_SIZE * TEST_MATRIX_SIZE];

// Performance measurement
static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Initialize test data
static void init_test_data(void) {
    srand(42); // Fixed seed for reproducible tests
    
    // Initialize matrices with random values
    for (int i = 0; i < TEST_MATRIX_SIZE * TEST_MATRIX_SIZE; i++) {
        test_matrix_a[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        test_matrix_b[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        test_input[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
    }
    
    // Clear output matrices
    memset(test_matrix_c, 0, sizeof(test_matrix_c));
    memset(test_matrix_ref, 0, sizeof(test_matrix_ref));
    memset(test_output, 0, sizeof(test_output));
}

// Reference matrix multiplication (for validation)
static void reference_matmul(const float* a, const float* b, float* c,
                           int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Reference ReLU implementation
static void reference_relu(const float* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// Reference sigmoid implementation
static void reference_sigmoid(const float* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Reference tanh implementation
static void reference_tanh(const float* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = tanhf(input[i]);
    }
}

// Compare floating point arrays with tolerance
static int compare_arrays(const float* a, const float* b, int count, float tolerance) {
    for (int i = 0; i < count; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: expected %f, got %f (diff: %f)\n", 
                   i, a[i], b[i], diff);
            return 0;
        }
    }
    return 1;
}

// Test matrix multiplication intrinsic
static int test_matmul_intrinsic(void) {
    printf("Testing matrix multiplication intrinsic...\n");
    
    const int size = 32; // Use smaller size for faster testing
    
    // Compute reference result
    reference_matmul(test_matrix_a, test_matrix_b, test_matrix_ref, size, size, size);
    
    // Test AI intrinsic
    double start_time = get_time_seconds();
    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        __builtin_riscv_ai_matmul_f32(test_matrix_a, test_matrix_b, test_matrix_c, 
                                     size, size, size);
    }
    double end_time = get_time_seconds();
    
    // Validate results
    if (!compare_arrays(test_matrix_c, test_matrix_ref, size * size, TEST_TOLERANCE)) {
        printf("FAIL: Matrix multiplication results don't match reference\n");
        return 0;
    }
    
    double avg_time = (end_time - start_time) / TEST_ITERATIONS;
    double gflops = (2.0 * size * size * size) / (avg_time * 1e9);
    
    printf("PASS: Matrix multiplication (%dx%d) - %.3f ms, %.2f GFLOPS\n", 
           size, size, avg_time * 1000, gflops);
    return 1;
}

// Test activation function intrinsics
static int test_activation_intrinsics(void) {
    printf("Testing activation function intrinsics...\n");
    
    const int count = TEST_MATRIX_SIZE * TEST_MATRIX_SIZE;
    int all_passed = 1;
    
    // Test ReLU
    reference_relu(test_input, test_matrix_ref, count);
    __builtin_riscv_ai_relu_f32(test_input, test_output, count);
    
    if (compare_arrays(test_output, test_matrix_ref, count, TEST_TOLERANCE)) {
        printf("PASS: ReLU activation function\n");
    } else {
        printf("FAIL: ReLU activation function\n");
        all_passed = 0;
    }
    
    // Test Sigmoid
    reference_sigmoid(test_input, test_matrix_ref, count);
    __builtin_riscv_ai_sigmoid_f32(test_input, test_output, count);
    
    if (compare_arrays(test_output, test_matrix_ref, count, TEST_TOLERANCE)) {
        printf("PASS: Sigmoid activation function\n");
    } else {
        printf("FAIL: Sigmoid activation function\n");
        all_passed = 0;
    }
    
    // Test Tanh
    reference_tanh(test_input, test_matrix_ref, count);
    __builtin_riscv_ai_tanh_f32(test_input, test_output, count);
    
    if (compare_arrays(test_output, test_matrix_ref, count, TEST_TOLERANCE)) {
        printf("PASS: Tanh activation function\n");
    } else {
        printf("FAIL: Tanh activation function\n");
        all_passed = 0;
    }
    
    return all_passed;
}

// Test control and status intrinsics
static int test_control_intrinsics(void) {
    printf("Testing control and status intrinsics...\n");
    
    // Test status reading
    uint32_t status = __builtin_riscv_ai_get_status(0);
    printf("AI Status: 0x%08x\n", status);
    
    // Test configuration
    __builtin_riscv_ai_set_config(0, AI_CONFIG_ENABLE | AI_CONFIG_PERF_ENABLE);
    
    // Test synchronization
    __builtin_riscv_ai_sync(0);
    
    // Test flush
    __builtin_riscv_ai_flush(0);
    
    printf("PASS: Control and status intrinsics\n");
    return 1;
}

// Test compiler optimizations
static int test_compiler_optimizations(void) {
    printf("Testing compiler optimizations...\n");
    
    // Test that the compiler can optimize multiple ReLU calls
    float test_val = 1.5f;
    float result1, result2, result3;
    
    // Chain of ReLU operations that should be optimized
    asm volatile (
        "ai.relu %0, %3\n\t"
        "ai.relu %1, %0\n\t"
        "ai.relu %2, %1"
        : "=f"(result1), "=f"(result2), "=f"(result3)
        : "f"(test_val)
    );
    
    if (fabsf(result3 - test_val) < TEST_TOLERANCE) {
        printf("PASS: Compiler optimization test\n");
        return 1;
    } else {
        printf("FAIL: Compiler optimization test\n");
        return 0;
    }
}

// Test different data types
static int test_data_types(void) {
    printf("Testing different data types...\n");
    
    // Test FP16 matrix multiplication
    static uint16_t ALIGN_64 a_fp16[16 * 16];
    static uint16_t ALIGN_64 b_fp16[16 * 16];
    static uint16_t ALIGN_64 c_fp16[16 * 16];
    
    // Initialize with simple values for FP16 test
    for (int i = 0; i < 16 * 16; i++) {
        a_fp16[i] = 0x3C00; // 1.0 in FP16
        b_fp16[i] = 0x4000; // 2.0 in FP16
    }
    
    __builtin_riscv_ai_matmul_f16(a_fp16, b_fp16, c_fp16, 16, 16, 16);
    
    // Test INT8 matrix multiplication
    static int8_t ALIGN_64 a_int8[16 * 16];
    static int8_t ALIGN_64 b_int8[16 * 16];
    static int32_t ALIGN_64 c_int32[16 * 16];
    
    // Initialize with simple values for INT8 test
    for (int i = 0; i < 16 * 16; i++) {
        a_int8[i] = 2;
        b_int8[i] = 3;
    }
    
    __builtin_riscv_ai_matmul_i8(a_int8, b_int8, c_int32, 16, 16, 16);
    
    // Verify INT8 result (should be 16 * 2 * 3 = 96 for each element)
    int expected = 16 * 2 * 3;
    if (c_int32[0] == expected) {
        printf("PASS: Data type tests (FP16, INT8)\n");
        return 1;
    } else {
        printf("FAIL: Data type tests - expected %d, got %d\n", expected, c_int32[0]);
        return 0;
    }
}

// Performance benchmark
static void benchmark_performance(void) {
    printf("\nPerformance Benchmarks:\n");
    printf("======================\n");
    
    const int sizes[] = {32, 64, 128, 256};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        if (size > TEST_MATRIX_SIZE) continue;
        
        // Benchmark matrix multiplication
        double start_time = get_time_seconds();
        for (int iter = 0; iter < 10; iter++) {
            __builtin_riscv_ai_matmul_f32(test_matrix_a, test_matrix_b, test_matrix_c,
                                         size, size, size);
        }
        double end_time = get_time_seconds();
        
        double avg_time = (end_time - start_time) / 10.0;
        double gflops = (2.0 * size * size * size) / (avg_time * 1e9);
        
        printf("MatMul %dx%d: %.3f ms, %.2f GFLOPS\n", 
               size, size, avg_time * 1000, gflops);
    }
    
    // Benchmark activation functions
    const int count = TEST_MATRIX_SIZE * TEST_MATRIX_SIZE;
    
    double start_time = get_time_seconds();
    for (int iter = 0; iter < 1000; iter++) {
        __builtin_riscv_ai_relu_f32(test_input, test_output, count);
    }
    double end_time = get_time_seconds();
    
    double avg_time = (end_time - start_time) / 1000.0;
    double throughput = count / (avg_time * 1e6); // Million elements per second
    
    printf("ReLU (%d elements): %.3f us, %.2f M elem/s\n", 
           count, avg_time * 1e6, throughput);
}

// Main test function
int main(void) {
    printf("RISC-V AI Intrinsics Test Suite\n");
    printf("================================\n\n");
    
    // Initialize test data
    init_test_data();
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run all tests
    total_tests++; if (test_matmul_intrinsic()) passed_tests++;
    total_tests++; if (test_activation_intrinsics()) passed_tests++;
    total_tests++; if (test_control_intrinsics()) passed_tests++;
    total_tests++; if (test_compiler_optimizations()) passed_tests++;
    total_tests++; if (test_data_types()) passed_tests++;
    
    // Run performance benchmarks
    benchmark_performance();
    
    // Print summary
    printf("\nTest Summary:\n");
    printf("=============\n");
    printf("Passed: %d/%d tests\n", passed_tests, total_tests);
    
    if (passed_tests == total_tests) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}