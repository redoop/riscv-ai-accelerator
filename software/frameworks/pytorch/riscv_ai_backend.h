/*
 * PyTorch RISC-V AI Backend Header
 * Defines interfaces for RISC-V AI accelerator integration with PyTorch
 */

#ifndef PYTORCH_RISCV_AI_BACKEND_H_
#define PYTORCH_RISCV_AI_BACKEND_H_

#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <memory>
#include <unordered_map>
#include <mutex>

extern "C" {
#include "../../lib/libtpu.h"
#include "../../compiler/riscv_ai_intrinsics.h"
}

namespace at {
namespace riscv_ai {

// Device type registration
constexpr DeviceType kRiscvAI = static_cast<DeviceType>(42);

// Forward declarations
class RiscvAiAllocator;
class ModelOptimizer;
class Quantizer;

// Core tensor operations
Tensor riscv_ai_mm(const Tensor& self, const Tensor& mat2);
Tensor riscv_ai_bmm(const Tensor& self, const Tensor& mat2);
Tensor riscv_ai_conv2d(const Tensor& input, const Tensor& weight,
                       const c10::optional<Tensor>& bias,
                       IntArrayRef stride, IntArrayRef padding,
                       IntArrayRef dilation, int64_t groups);

// Activation functions
Tensor riscv_ai_relu(const Tensor& self);
Tensor& riscv_ai_relu_(Tensor& self);
Tensor riscv_ai_sigmoid(const Tensor& self);
Tensor riscv_ai_tanh(const Tensor& self);

// Pooling operations
Tensor riscv_ai_max_pool2d(const Tensor& self, IntArrayRef kernel_size,
                          IntArrayRef stride, IntArrayRef padding);
Tensor riscv_ai_avg_pool2d(const Tensor& self, IntArrayRef kernel_size,
                          IntArrayRef stride, IntArrayRef padding);

// Batch normalization
Tensor riscv_ai_batch_norm(const Tensor& input, const Tensor& weight,
                          const Tensor& bias, const Tensor& running_mean,
                          const Tensor& running_var, double eps);

// Device management
bool is_riscv_ai_available();
int64_t riscv_ai_device_count();
void riscv_ai_synchronize();
void initialize_riscv_ai();
void cleanup_riscv_ai();

// Memory management
void* riscv_ai_malloc(size_t size);
void riscv_ai_free(void* ptr);

} // namespace riscv_ai
} // namespace at

#endif // PYTORCH_RISCV_AI_BACKEND_H_