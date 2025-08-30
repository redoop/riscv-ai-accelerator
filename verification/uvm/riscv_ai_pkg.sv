// RISC-V AI Accelerator UVM Package
// Comprehensive verification environment package

`ifndef RISCV_AI_PKG_SV
`define RISCV_AI_PKG_SV

package riscv_ai_pkg;

    import uvm_pkg::*;
    `include "uvm_macros.svh"
    
    // Configuration parameters
    parameter int ADDR_WIDTH = 64;
    parameter int DATA_WIDTH = 64;
    parameter int NUM_CORES = 4;
    parameter int NUM_TPU = 2;
    parameter int NUM_VPU = 2;
    parameter int HBM_CHANNELS = 4;
    
    // Data types
    typedef enum {
        READ_OP,
        WRITE_OP,
        AI_MATMUL_OP,
        AI_CONV2D_OP,
        AI_RELU_OP,
        AI_SIGMOID_OP,
        AI_MAXPOOL_OP,
        AI_AVGPOOL_OP,
        AI_BATCHNORM_OP
    } operation_type_e;
    
    typedef enum {
        INT8_TYPE,
        INT16_TYPE,
        INT32_TYPE,
        FP16_TYPE,
        FP32_TYPE,
        FP64_TYPE
    } data_type_e;
    
    // Transaction classes
    `include "riscv_ai_transaction.sv"
    `include "riscv_ai_sequence_item.sv"
    
    // Sequence classes
    `include "riscv_ai_base_sequence.sv"
    `include "riscv_ai_random_sequence.sv"
    `include "riscv_ai_directed_sequence.sv"
    
    // Driver and monitor classes
    `include "riscv_ai_driver.sv"
    `include "riscv_ai_monitor.sv"
    
    // Agent classes
    `include "riscv_ai_agent.sv"
    
    // Scoreboard and coverage
    `include "riscv_ai_scoreboard.sv"
    `include "riscv_ai_coverage.sv"
    
    // Environment and test classes
    `include "riscv_ai_env.sv"
    `include "riscv_ai_base_test.sv"
    
    // Utility classes
    `include "riscv_ai_utils.sv"

endpackage : riscv_ai_pkg

`endif // RISCV_AI_PKG_SV