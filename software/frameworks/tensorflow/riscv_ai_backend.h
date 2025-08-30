/*
 * TensorFlow RISC-V AI Backend Header
 * Defines interfaces for RISC-V AI accelerator integration with TensorFlow
 */

#ifndef TENSORFLOW_RISCV_AI_BACKEND_H_
#define TENSORFLOW_RISCV_AI_BACKEND_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

extern "C" {
#include "../../lib/libtpu.h"
}

namespace tensorflow {
namespace riscv_ai {

// Device type constant
constexpr const char* DEVICE_RISCV_AI = "RISCV_AI";

// Forward declarations
class RiscvAiDevice;
class RiscvAiDeviceFactory;

// RISC-V AI specific tensor operations
class RiscvAiTensorOps {
public:
    // Matrix operations
    static Status MatMul(const Tensor& a, const Tensor& b, Tensor* c,
                        bool transpose_a = false, bool transpose_b = false);
    
    // Convolution operations  
    static Status Conv2D(const Tensor& input, const Tensor& filter, Tensor* output,
                         const std::vector<int32>& strides, const string& padding);
    
    // Activation functions
    static Status ReLU(const Tensor& input, Tensor* output);
    static Status Sigmoid(const Tensor& input, Tensor* output);
    static Status Tanh(const Tensor& input, Tensor* output);
    
    // Pooling operations
    static Status MaxPool(const Tensor& input, Tensor* output,
                         const std::vector<int32>& ksize,
                         const std::vector<int32>& strides);
    
    static Status AvgPool(const Tensor& input, Tensor* output,
                         const std::vector<int32>& ksize,
                         const std::vector<int32>& strides);
    
    // Normalization operations
    static Status BatchNorm(const Tensor& input, const Tensor& scale,
                           const Tensor& offset, const Tensor& mean,
                           const Tensor& variance, Tensor* output);
};

// Memory management for RISC-V AI device
class RiscvAiAllocator : public Allocator {
public:
    RiscvAiAllocator();
    ~RiscvAiAllocator() override;
    
    string Name() override { return "riscv_ai"; }
    
    void* AllocateRaw(size_t alignment, size_t num_bytes) override;
    void DeallocateRaw(void* ptr) override;
    
    bool TracksAllocationSizes() const override { return true; }
    size_t RequestedSize(const void* ptr) const override;
    size_t AllocatedSize(const void* ptr) const override;
    
private:
    std::unordered_map<void*, size_t> allocated_sizes_;
    mutable mutex mu_;
};

// Performance monitoring
class RiscvAiProfiler {
public:
    struct PerformanceStats {
        uint64_t total_ops;
        uint64_t total_cycles;
        uint64_t cache_hits;
        uint64_t cache_misses;
        double throughput_gops;
        double utilization;
    };
    
    static void StartProfiling();
    static void StopProfiling();
    static PerformanceStats GetStats();
    static void ResetStats();
};

// Utility functions
namespace util {

// Data type conversion utilities
bool IsSupportedDataType(DataType dtype);
ai_data_type_t ToAiDataType(DataType dtype);
DataType FromAiDataType(ai_data_type_t ai_dtype);

// Shape and dimension utilities
bool IsValidMatrixShape(const TensorShape& shape);
bool IsValidConvolutionShape(const TensorShape& input_shape,
                            const TensorShape& filter_shape);

// Memory layout utilities
Status ConvertToOptimalLayout(const Tensor& input, Tensor* output,
                             const string& target_layout);

// Error handling utilities
Status ConvertAiStatus(ai_status_t ai_status);
string AiStatusToString(ai_status_t ai_status);

} // namespace util

} // namespace riscv_ai
} // namespace tensorflow

#endif // TENSORFLOW_RISCV_AI_BACKEND_H_