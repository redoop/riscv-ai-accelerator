// RISC-V AI Instruction Intrinsics
// Provides C/C++ intrinsics for custom AI instructions
// Optimized for GCC compiler integration

#ifndef RISCV_AI_INTRINSICS_H
#define RISCV_AI_INTRINSICS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Include AI constants and definitions
#ifndef __ASSEMBLER__
#include "riscv-ai-constants.h"
#endif

// Compiler-specific optimizations
#ifdef __GNUC__
#define AI_INLINE __attribute__((always_inline)) static inline
#define AI_PURE __attribute__((pure))
#define AI_CONST __attribute__((const))
#define AI_NOTHROW __attribute__((nothrow))
#else
#define AI_INLINE static inline
#define AI_PURE
#define AI_CONST
#define AI_NOTHROW
#endif

// Memory alignment macros
#define AI_ALIGN(x) __attribute__((aligned(x)))
#define AI_CACHE_ALIGN AI_ALIGN(64)

// Branch prediction hints
#ifdef __GNUC__
#define AI_LIKELY(x) __builtin_expect(!!(x), 1)
#define AI_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define AI_LIKELY(x) (x)
#define AI_UNLIKELY(x) (x)
#endif

// ========================================
// AI Instruction Intrinsics
// ========================================

/**
 * Matrix multiplication intrinsic (FP32)
 * Performs matrix multiplication using TPU hardware acceleration
 * C = A * B where A is MxK, B is KxN, C is MxN
 * @param a Pointer to matrix A (must be 64-byte aligned)
 * @param b Pointer to matrix B (must be 64-byte aligned)
 * @param c Pointer to result matrix C (must be 64-byte aligned)
 * @param m Number of rows in A
 * @param n Number of columns in B
 * @param k Number of columns in A (rows in B)
 */
AI_INLINE void __builtin_riscv_ai_matmul_f32(
    const float* AI_CACHE_ALIGN a, 
    const float* AI_CACHE_ALIGN b, 
    float* AI_CACHE_ALIGN c,
    uint32_t m, uint32_t n, uint32_t k) AI_NOTHROW {
    
    // Use GCC builtin if available, otherwise inline assembly
    #ifdef __has_builtin
    #if __has_builtin(__builtin_riscv_ai_matmul_f32)
        __builtin_riscv_ai_matmul_f32(a, b, c, m, n, k);
        return;
    #endif
    #endif
    
    // Encode dimensions into parameter register
    uint64_t params = AI_MATMUL_PARAMS(m, n, k);
    
    asm volatile (
        "ai.matmul %0, %1, %2, %3"
        : "=r"(c)
        : "r"(a), "r"(b), "r"(params)
        : "memory"
    );
}

/**
 * Matrix multiplication intrinsic (FP16)
 */
AI_INLINE void __builtin_riscv_ai_matmul_f16(
    const uint16_t* AI_CACHE_ALIGN a, 
    const uint16_t* AI_CACHE_ALIGN b, 
    uint16_t* AI_CACHE_ALIGN c,
    uint32_t m, uint32_t n, uint32_t k) AI_NOTHROW {
    
    uint64_t params = AI_MATMUL_PARAMS(m, n, k) | (AI_DTYPE_FP16 << 56);
    
    asm volatile (
        "ai.matmul %0, %1, %2, %3"
        : "=r"(c)
        : "r"(a), "r"(b), "r"(params)
        : "memory"
    );
}

/**
 * Matrix multiplication intrinsic (INT8)
 */
AI_INLINE void __builtin_riscv_ai_matmul_i8(
    const int8_t* AI_CACHE_ALIGN a, 
    const int8_t* AI_CACHE_ALIGN b, 
    int32_t* AI_CACHE_ALIGN c,
    uint32_t m, uint32_t n, uint32_t k) AI_NOTHROW {
    
    uint64_t params = AI_MATMUL_PARAMS(m, n, k) | (AI_DTYPE_INT8 << 56);
    
    asm volatile (
        "ai.matmul %0, %1, %2, %3"
        : "=r"(c)
        : "r"(a), "r"(b), "r"(params)
        : "memory"
    );
}

/**
 * 2D Convolution intrinsic
 * Performs 2D convolution using specialized hardware
 */
static inline void __builtin_riscv_ai_conv2d_f32(
    const float* input, const float* kernel, float* output,
    uint32_t in_h, uint32_t in_w, uint32_t in_c,
    uint32_t out_h, uint32_t out_w, uint32_t out_c,
    uint32_t kernel_h, uint32_t kernel_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {
    
    uint64_t params = ((uint64_t)in_h << 48) | ((uint64_t)in_w << 32) | 
                      ((uint64_t)kernel_h << 16) | ((uint64_t)kernel_w);
    uint64_t strides = ((uint64_t)stride_h << 16) | ((uint64_t)stride_w);
    
    asm volatile (
        "ai.conv2d %0, %1, %2, %3"
        : "=r"(output)
        : "r"(input), "r"(kernel), "r"(params)
        : "memory"
    );
}

/**
 * ReLU activation function intrinsic
 */
static inline void __builtin_riscv_ai_relu_f32(const float* input, float* output, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        asm volatile (
            "ai.relu %0, %1"
            : "=f"(output[i])
            : "f"(input[i])
        );
    }
}

/**
 * Sigmoid activation function intrinsic
 */
static inline void __builtin_riscv_ai_sigmoid_f32(const float* input, float* output, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        asm volatile (
            "ai.sigmoid %0, %1"
            : "=f"(output[i])
            : "f"(input[i])
        );
    }
}

/**
 * Tanh activation function intrinsic
 */
static inline void __builtin_riscv_ai_tanh_f32(const float* input, float* output, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        asm volatile (
            "ai.tanh %0, %1"
            : "=f"(output[i])
            : "f"(input[i])
        );
    }
}

/**
 * Max pooling intrinsic
 */
static inline void __builtin_riscv_ai_maxpool_f32(
    const float* input, float* output,
    uint32_t in_h, uint32_t in_w, uint32_t channels,
    uint32_t pool_h, uint32_t pool_w,
    uint32_t stride_h, uint32_t stride_w) {
    
    uint64_t params = ((uint64_t)pool_h << 16) | ((uint64_t)pool_w);
    uint64_t strides = ((uint64_t)stride_h << 16) | ((uint64_t)stride_w);
    
    asm volatile (
        "ai.maxpool %0, %1, %2"
        : "=r"(output)
        : "r"(input), "r"(params)
        : "memory"
    );
}

/**
 * Average pooling intrinsic
 */
static inline void __builtin_riscv_ai_avgpool_f32(
    const float* input, float* output,
    uint32_t in_h, uint32_t in_w, uint32_t channels,
    uint32_t pool_h, uint32_t pool_w,
    uint32_t stride_h, uint32_t stride_w) {
    
    uint64_t params = ((uint64_t)pool_h << 16) | ((uint64_t)pool_w);
    uint64_t strides = ((uint64_t)stride_h << 16) | ((uint64_t)stride_w);
    
    asm volatile (
        "ai.avgpool %0, %1, %2"
        : "=r"(output)
        : "r"(input), "r"(params)
        : "memory"
    );
}

/**
 * Batch normalization intrinsic
 */
static inline void __builtin_riscv_ai_batchnorm_f32(
    const float* input, float* output,
    const float* scale, const float* bias,
    const float* mean, const float* variance,
    uint32_t count, float epsilon) {
    
    asm volatile (
        "ai.batchnorm %0, %1, %2, %3"
        : "=r"(output)
        : "r"(input), "r"(scale), "r"(bias)
        : "memory"
    );
}

// ========================================
// Vector Intrinsics (RVV Extension)
// ========================================

/**
 * Vector load intrinsic
 */
#define __builtin_riscv_vload_f32(ptr, vl) \
    __builtin_riscv_vle32_v_f32m1(ptr, vl)

/**
 * Vector store intrinsic
 */
#define __builtin_riscv_vstore_f32(ptr, vec, vl) \
    __builtin_riscv_vse32_v_f32m1(ptr, vec, vl)

/**
 * Vector add intrinsic
 */
#define __builtin_riscv_vadd_f32(va, vb, vl) \
    __builtin_riscv_vfadd_vv_f32m1(va, vb, vl)

/**
 * Vector multiply intrinsic
 */
#define __builtin_riscv_vmul_f32(va, vb, vl) \
    __builtin_riscv_vfmul_vv_f32m1(va, vb, vl)

/**
 * Vector fused multiply-add intrinsic
 */
#define __builtin_riscv_vfmadd_f32(va, vb, vc, vl) \
    __builtin_riscv_vfmadd_vv_f32m1(va, vb, vc, vl)

// ========================================
// Performance and Control Intrinsics
// ========================================

/**
 * Get AI accelerator status
 */
static inline uint32_t __builtin_riscv_ai_get_status(uint32_t accel_id) {
    uint32_t status;
    asm volatile (
        "csrr %0, 0x800"  // Custom CSR for AI status
        : "=r"(status)
        :
        :
    );
    return status;
}

/**
 * Set AI accelerator configuration
 */
static inline void __builtin_riscv_ai_set_config(uint32_t accel_id, uint32_t config) {
    asm volatile (
        "csrw 0x801, %0"  // Custom CSR for AI config
        :
        : "r"(config)
        :
    );
}

/**
 * Flush AI accelerator pipeline
 */
static inline void __builtin_riscv_ai_flush(uint32_t accel_id) {
    asm volatile (
        "ai.flush %0"
        :
        : "r"(accel_id)
        : "memory"
    );
}

/**
 * Synchronize with AI accelerator
 */
static inline void __builtin_riscv_ai_sync(uint32_t accel_id) {
    asm volatile (
        "ai.sync %0"
        :
        : "r"(accel_id)
        : "memory"
    );
}

#ifdef __cplusplus
}
#endif

#endif // RISCV_AI_INTRINSICS_H