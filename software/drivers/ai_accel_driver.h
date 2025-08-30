// AI Accelerator Driver Header
// Provides user-space interface to TPU and VPU accelerators

#ifndef AI_ACCEL_DRIVER_H
#define AI_ACCEL_DRIVER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// Data Type Definitions
// ========================================

typedef enum {
    AI_DTYPE_INT8  = 0,
    AI_DTYPE_INT16 = 1,
    AI_DTYPE_INT32 = 2,
    AI_DTYPE_FP16  = 3,
    AI_DTYPE_FP32  = 4,
    AI_DTYPE_FP64  = 5
} ai_data_type_t;

typedef enum {
    AI_OP_MATMUL    = 0,
    AI_OP_CONV2D    = 1,
    AI_OP_RELU      = 2,
    AI_OP_SIGMOID   = 3,
    AI_OP_TANH      = 4,
    AI_OP_MAXPOOL   = 5,
    AI_OP_AVGPOOL   = 6,
    AI_OP_BATCHNORM = 7
} ai_operation_t;

typedef enum {
    AI_ACCEL_TPU = 0,
    AI_ACCEL_VPU = 1
} ai_accelerator_type_t;

// Tensor descriptor structure
typedef struct {
    ai_data_type_t dtype;
    uint32_t ndim;
    uint32_t shape[8];
    uint32_t stride[8];
    void* data_ptr;
    size_t size_bytes;
} ai_tensor_t;

// Task descriptor for accelerator operations
typedef struct {
    uint32_t task_id;
    ai_operation_t operation;
    ai_accelerator_type_t accel_type;
    uint32_t accel_id;
    ai_tensor_t input_tensors[4];
    ai_tensor_t output_tensors[2];
    uint32_t num_inputs;
    uint32_t num_outputs;
    void* params;  // Operation-specific parameters
    size_t params_size;
} ai_task_t;

// Status and error codes
typedef enum {
    AI_STATUS_SUCCESS = 0,
    AI_STATUS_ERROR = 1,
    AI_STATUS_BUSY = 2,
    AI_STATUS_TIMEOUT = 3,
    AI_STATUS_INVALID_PARAM = 4,
    AI_STATUS_NO_MEMORY = 5,
    AI_STATUS_DEVICE_ERROR = 6
} ai_status_t;

// ========================================
// Driver Interface Functions
// ========================================

/**
 * Initialize the AI accelerator driver
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_driver_init(void);

/**
 * Cleanup and shutdown the AI accelerator driver
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_driver_cleanup(void);

/**
 * Get the number of available accelerators of a specific type
 * @param accel_type Type of accelerator (TPU or VPU)
 * @return Number of available accelerators, or -1 on error
 */
int ai_get_accelerator_count(ai_accelerator_type_t accel_type);

/**
 * Submit a task to an AI accelerator
 * @param task Pointer to task descriptor
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_submit_task(const ai_task_t* task);

/**
 * Wait for task completion
 * @param task_id Task ID to wait for
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @return AI_STATUS_SUCCESS on completion, error code otherwise
 */
ai_status_t ai_wait_task(uint32_t task_id, uint32_t timeout_ms);

/**
 * Check task status without blocking
 * @param task_id Task ID to check
 * @return Task status
 */
ai_status_t ai_check_task_status(uint32_t task_id);

/**
 * Cancel a pending or running task
 * @param task_id Task ID to cancel
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_cancel_task(uint32_t task_id);

/**
 * Allocate device memory for tensors
 * @param size Size in bytes to allocate
 * @param alignment Memory alignment requirement
 * @return Pointer to allocated memory, or NULL on failure
 */
void* ai_alloc_device_memory(size_t size, size_t alignment);

/**
 * Free device memory
 * @param ptr Pointer to memory to free
 */
void ai_free_device_memory(void* ptr);

/**
 * Copy data between host and device memory
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Number of bytes to copy
 * @param to_device True if copying to device, false if copying to host
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_memcpy(void* dst, const void* src, size_t size, bool to_device);

// ========================================
// Utility Functions
// ========================================

/**
 * Create a tensor descriptor
 * @param tensor Pointer to tensor structure to initialize
 * @param dtype Data type
 * @param ndim Number of dimensions
 * @param shape Array of dimension sizes
 * @param data_ptr Pointer to tensor data
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_create_tensor(ai_tensor_t* tensor, ai_data_type_t dtype, 
                            uint32_t ndim, const uint32_t* shape, void* data_ptr);

/**
 * Calculate tensor size in bytes
 * @param tensor Pointer to tensor descriptor
 * @return Size in bytes, or 0 on error
 */
size_t ai_tensor_size_bytes(const ai_tensor_t* tensor);

/**
 * Get data type size in bytes
 * @param dtype Data type
 * @return Size in bytes
 */
size_t ai_dtype_size(ai_data_type_t dtype);

/**
 * Get accelerator capabilities
 * @param accel_type Accelerator type
 * @param accel_id Accelerator ID
 * @param capabilities Pointer to capabilities structure (implementation-specific)
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_get_capabilities(ai_accelerator_type_t accel_type, uint32_t accel_id, void* capabilities);

/**
 * Get performance counters
 * @param accel_type Accelerator type
 * @param accel_id Accelerator ID
 * @param counters Pointer to counters structure (implementation-specific)
 * @return AI_STATUS_SUCCESS on success, error code otherwise
 */
ai_status_t ai_get_performance_counters(ai_accelerator_type_t accel_type, uint32_t accel_id, void* counters);

#ifdef __cplusplus
}
#endif

#endif // AI_ACCEL_DRIVER_H