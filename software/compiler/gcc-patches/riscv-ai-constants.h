/* RISC-V AI Extension Constants
   Copyright (C) 2023 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

GCC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING3.  If not see
<http://www.gnu.org/licenses/>.  */

#ifndef GCC_RISCV_AI_CONSTANTS_H
#define GCC_RISCV_AI_CONSTANTS_H

/* AI instruction UNSPEC constants */
enum {
  /* Matrix and tensor operations */
  UNSPEC_AI_MATMUL = 200,
  UNSPEC_AI_MATMUL_RELU,
  UNSPEC_AI_CONV2D,
  UNSPEC_AI_CONV2D_RELU,
  
  /* Activation functions */
  UNSPEC_AI_RELU,
  UNSPEC_AI_SIGMOID,
  UNSPEC_AI_TANH,
  UNSPEC_AI_SOFTMAX,
  UNSPEC_AI_GELU,
  
  /* Pooling operations */
  UNSPEC_AI_MAXPOOL,
  UNSPEC_AI_AVGPOOL,
  UNSPEC_AI_GLOBALPOOL,
  
  /* Normalization operations */
  UNSPEC_AI_BATCHNORM,
  UNSPEC_AI_LAYERNORM,
  UNSPEC_AI_INSTANCENORM,
  
  /* Data movement and transformation */
  UNSPEC_AI_TRANSPOSE,
  UNSPEC_AI_RESHAPE,
  UNSPEC_AI_CONCAT,
  UNSPEC_AI_SPLIT,
  
  /* Quantization operations */
  UNSPEC_AI_QUANTIZE,
  UNSPEC_AI_DEQUANTIZE,
  UNSPEC_AI_SCALE,
  
  /* Control and status operations */
  UNSPEC_AI_GET_STATUS,
  UNSPEC_AI_SET_CONFIG,
  UNSPEC_AI_FLUSH,
  UNSPEC_AI_SYNC,
  UNSPEC_AI_BARRIER,
  
  /* Performance monitoring */
  UNSPEC_AI_PERF_START,
  UNSPEC_AI_PERF_STOP,
  UNSPEC_AI_PERF_READ,
  
  /* Memory operations */
  UNSPEC_AI_PREFETCH,
  UNSPEC_AI_CACHE_FLUSH,
  UNSPEC_AI_DMA_START,
  UNSPEC_AI_DMA_WAIT
};

/* AI instruction opcodes (custom-0 space: 0x0B) */
#define AI_OPCODE_BASE          0x0B

/* AI instruction function codes */
#define AI_FUNC_MATMUL          0x00
#define AI_FUNC_CONV2D          0x01
#define AI_FUNC_RELU            0x02
#define AI_FUNC_SIGMOID         0x03
#define AI_FUNC_TANH            0x04
#define AI_FUNC_MAXPOOL         0x05
#define AI_FUNC_AVGPOOL         0x06
#define AI_FUNC_BATCHNORM       0x07
#define AI_FUNC_TRANSPOSE       0x08
#define AI_FUNC_QUANTIZE        0x09
#define AI_FUNC_DEQUANTIZE      0x0A
#define AI_FUNC_FLUSH           0x10
#define AI_FUNC_SYNC            0x11
#define AI_FUNC_BARRIER         0x12

/* AI CSR addresses */
#define AI_CSR_STATUS           0x800
#define AI_CSR_CONFIG           0x801
#define AI_CSR_PERF_COUNTER0    0x802
#define AI_CSR_PERF_COUNTER1    0x803
#define AI_CSR_PERF_COUNTER2    0x804
#define AI_CSR_PERF_COUNTER3    0x805
#define AI_CSR_ERROR_STATUS     0x806
#define AI_CSR_ERROR_ADDR       0x807

/* AI status register bits */
#define AI_STATUS_READY         (1 << 0)
#define AI_STATUS_BUSY          (1 << 1)
#define AI_STATUS_ERROR         (1 << 2)
#define AI_STATUS_OVERFLOW      (1 << 3)
#define AI_STATUS_UNDERFLOW     (1 << 4)
#define AI_STATUS_TPU0_READY    (1 << 8)
#define AI_STATUS_TPU1_READY    (1 << 9)
#define AI_STATUS_VPU0_READY    (1 << 10)
#define AI_STATUS_VPU1_READY    (1 << 11)

/* AI configuration register bits */
#define AI_CONFIG_ENABLE        (1 << 0)
#define AI_CONFIG_AUTO_SYNC     (1 << 1)
#define AI_CONFIG_PERF_ENABLE   (1 << 2)
#define AI_CONFIG_ERROR_ENABLE  (1 << 3)
#define AI_CONFIG_PRECISION_MASK (0x3 << 4)
#define AI_CONFIG_PRECISION_FP32 (0x0 << 4)
#define AI_CONFIG_PRECISION_FP16 (0x1 << 4)
#define AI_CONFIG_PRECISION_INT8 (0x2 << 4)
#define AI_CONFIG_PRECISION_INT4 (0x3 << 4)

/* AI data type encodings */
#define AI_DTYPE_FP32           0x0
#define AI_DTYPE_FP16           0x1
#define AI_DTYPE_BF16           0x2
#define AI_DTYPE_INT32          0x3
#define AI_DTYPE_INT16          0x4
#define AI_DTYPE_INT8           0x5
#define AI_DTYPE_INT4           0x6
#define AI_DTYPE_UINT32         0x7
#define AI_DTYPE_UINT16         0x8
#define AI_DTYPE_UINT8          0x9
#define AI_DTYPE_UINT4          0xA

/* AI operation modes */
#define AI_MODE_SYNC            0x0
#define AI_MODE_ASYNC           0x1
#define AI_MODE_STREAM          0x2

/* AI matrix layout formats */
#define AI_LAYOUT_ROW_MAJOR     0x0
#define AI_LAYOUT_COL_MAJOR     0x1
#define AI_LAYOUT_BLOCKED       0x2

/* AI convolution parameters */
#define AI_CONV_PADDING_VALID   0x0
#define AI_CONV_PADDING_SAME    0x1
#define AI_CONV_PADDING_FULL    0x2

/* AI pooling types */
#define AI_POOL_MAX             0x0
#define AI_POOL_AVG             0x1
#define AI_POOL_GLOBAL_MAX      0x2
#define AI_POOL_GLOBAL_AVG      0x3

/* AI activation function types */
#define AI_ACTIVATION_RELU      0x0
#define AI_ACTIVATION_SIGMOID   0x1
#define AI_ACTIVATION_TANH      0x2
#define AI_ACTIVATION_SOFTMAX   0x3
#define AI_ACTIVATION_GELU      0x4
#define AI_ACTIVATION_SWISH     0x5

/* AI error codes */
#define AI_ERROR_NONE           0x0
#define AI_ERROR_INVALID_OP     0x1
#define AI_ERROR_INVALID_PARAM  0x2
#define AI_ERROR_DIMENSION_MISMATCH 0x3
#define AI_ERROR_MEMORY_FAULT   0x4
#define AI_ERROR_TIMEOUT        0x5
#define AI_ERROR_OVERFLOW       0x6
#define AI_ERROR_UNDERFLOW      0x7

/* AI performance counter events */
#define AI_PERF_CYCLES          0x0
#define AI_PERF_INSTRUCTIONS    0x1
#define AI_PERF_MATMUL_OPS      0x2
#define AI_PERF_CONV_OPS        0x3
#define AI_PERF_CACHE_HITS      0x4
#define AI_PERF_CACHE_MISSES    0x5
#define AI_PERF_MEMORY_ACCESSES 0x6
#define AI_PERF_STALL_CYCLES    0x7

/* AI instruction format macros */
#define AI_ENCODE_R_TYPE(func, rd, rs1, rs2) \
  ((AI_OPCODE_BASE) | ((rd) << 7) | ((func) << 12) | ((rs1) << 15) | ((rs2) << 20))

#define AI_ENCODE_I_TYPE(func, rd, rs1, imm) \
  ((AI_OPCODE_BASE) | ((rd) << 7) | ((func) << 12) | ((rs1) << 15) | ((imm) << 20))

#define AI_ENCODE_S_TYPE(func, rs1, rs2, imm) \
  ((AI_OPCODE_BASE) | ((imm & 0x1F) << 7) | ((func) << 12) | ((rs1) << 15) | ((rs2) << 20) | ((imm >> 5) << 25))

/* AI instruction helper macros */
#define AI_MATMUL_PARAMS(m, n, k) \
  (((uint64_t)(m) << 32) | ((uint64_t)(n) << 16) | (uint64_t)(k))

#define AI_CONV2D_PARAMS(kh, kw, sh, sw, ph, pw) \
  (((uint64_t)(kh) << 40) | ((uint64_t)(kw) << 32) | \
   ((uint64_t)(sh) << 24) | ((uint64_t)(sw) << 16) | \
   ((uint64_t)(ph) << 8) | (uint64_t)(pw))

#define AI_POOL_PARAMS(kh, kw, sh, sw) \
  (((uint32_t)(kh) << 24) | ((uint32_t)(kw) << 16) | \
   ((uint32_t)(sh) << 8) | (uint32_t)(sw))

/* AI tensor descriptor format */
#define AI_TENSOR_DESC(dtype, ndim, layout) \
  (((uint32_t)(dtype) << 24) | ((uint32_t)(ndim) << 16) | (uint32_t)(layout))

#endif /* GCC_RISCV_AI_CONSTANTS_H */