// RISC-V AI Constants Header
// Defines constants for AI instruction extensions

#ifndef RISCV_AI_CONSTANTS_H
#define RISCV_AI_CONSTANTS_H

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// AI Instruction Opcodes
// ========================================

// Matrix operations
#define AI_MATMUL_OPCODE    0x01
#define AI_CONV2D_OPCODE    0x02

// Activation functions
#define AI_RELU_OPCODE      0x10
#define AI_SIGMOID_OPCODE   0x11
#define AI_TANH_OPCODE      0x12

// Pooling operations
#define AI_MAXPOOL_OPCODE   0x20
#define AI_AVGPOOL_OPCODE   0x21

// Batch normalization
#define AI_BATCHNORM_OPCODE 0x30

// ========================================
// Data Type Constants
// ========================================

#define AI_DTYPE_INT8       0
#define AI_DTYPE_INT16      1
#define AI_DTYPE_INT32      2
#define AI_DTYPE_FP16       3
#define AI_DTYPE_FP32       4

// ========================================
// Hardware Limits
// ========================================

#define AI_MAX_MATRIX_SIZE  64
#define AI_MAX_VECTOR_SIZE  1024
#define AI_MAX_BATCH_SIZE   256

// ========================================
// Helper Macros
// ========================================

#define AI_MATMUL_PARAMS(m, n, k) ((uint64_t)(m) | ((uint64_t)(n) << 16) | ((uint64_t)(k) << 32))

#ifdef __cplusplus
}
#endif

#endif // RISCV_AI_CONSTANTS_H