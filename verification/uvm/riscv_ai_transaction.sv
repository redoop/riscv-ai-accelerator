// RISC-V AI Transaction Class
// Base transaction class for all AI accelerator operations

`ifndef RISCV_AI_TRANSACTION_SV
`define RISCV_AI_TRANSACTION_SV

class riscv_ai_transaction extends uvm_sequence_item;
    
    // Transaction fields
    rand bit [ADDR_WIDTH-1:0] addr;
    rand bit [DATA_WIDTH-1:0] data;
    rand operation_type_e op_type;
    rand data_type_e data_type;
    rand bit [31:0] size;
    rand bit [7:0] core_id;
    rand bit [7:0] tpu_id;
    rand bit [7:0] vpu_id;
    
    // AI-specific fields
    rand bit [31:0] matrix_m, matrix_n, matrix_k;  // Matrix dimensions
    rand bit [31:0] conv_height, conv_width, conv_channels;  // Convolution dimensions
    rand bit [31:0] kernel_size, stride, padding;  // Convolution parameters
    rand bit [31:0] pool_size;  // Pooling parameters
    
    // Control fields
    rand bit valid;
    rand bit ready;
    bit response;
    bit error;
    
    // Timing fields
    time start_time;
    time end_time;
    int latency;
    
    // Constraints
    constraint addr_c {
        addr inside {[64'h0000_0000_0000_0000:64'h0000_FFFF_FFFF_FFFF]};
    }
    
    constraint size_c {
        size inside {[1:4096]};
        size % 8 == 0;  // 8-byte aligned
    }
    
    constraint core_id_c {
        core_id < NUM_CORES;
    }
    
    constraint tpu_id_c {
        tpu_id < NUM_TPU;
    }
    
    constraint vpu_id_c {
        vpu_id < NUM_VPU;
    }
    
    constraint matrix_dims_c {
        matrix_m inside {[1:1024]};
        matrix_n inside {[1:1024]};
        matrix_k inside {[1:1024]};
        matrix_m % 8 == 0;
        matrix_n % 8 == 0;
        matrix_k % 8 == 0;
    }
    
    constraint conv_dims_c {
        conv_height inside {[1:512]};
        conv_width inside {[1:512]};
        conv_channels inside {[1:512]};
        kernel_size inside {[1, 3, 5, 7]};
        stride inside {[1, 2, 4]};
        padding inside {[0:3]};
    }
    
    constraint pool_size_c {
        pool_size inside {[2, 4, 8]};
    }
    
    constraint op_type_data_type_c {
        if (op_type inside {AI_MATMUL_OP, AI_CONV2D_OP}) {
            data_type inside {INT8_TYPE, FP16_TYPE, FP32_TYPE};
        }
        if (op_type inside {AI_RELU_OP, AI_SIGMOID_OP}) {
            data_type inside {FP16_TYPE, FP32_TYPE};
        }
    }
    
    // UVM automation macros
    `uvm_object_utils_begin(riscv_ai_transaction)
        `uvm_field_int(addr, UVM_ALL_ON)
        `uvm_field_int(data, UVM_ALL_ON)
        `uvm_field_enum(operation_type_e, op_type, UVM_ALL_ON)
        `uvm_field_enum(data_type_e, data_type, UVM_ALL_ON)
        `uvm_field_int(size, UVM_ALL_ON)
        `uvm_field_int(core_id, UVM_ALL_ON)
        `uvm_field_int(tpu_id, UVM_ALL_ON)
        `uvm_field_int(vpu_id, UVM_ALL_ON)
        `uvm_field_int(matrix_m, UVM_ALL_ON)
        `uvm_field_int(matrix_n, UVM_ALL_ON)
        `uvm_field_int(matrix_k, UVM_ALL_ON)
        `uvm_field_int(conv_height, UVM_ALL_ON)
        `uvm_field_int(conv_width, UVM_ALL_ON)
        `uvm_field_int(conv_channels, UVM_ALL_ON)
        `uvm_field_int(kernel_size, UVM_ALL_ON)
        `uvm_field_int(stride, UVM_ALL_ON)
        `uvm_field_int(padding, UVM_ALL_ON)
        `uvm_field_int(pool_size, UVM_ALL_ON)
        `uvm_field_int(valid, UVM_ALL_ON)
        `uvm_field_int(ready, UVM_ALL_ON)
        `uvm_field_int(response, UVM_ALL_ON)
        `uvm_field_int(error, UVM_ALL_ON)
    `uvm_object_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_transaction");
        super.new(name);
    endfunction
    
    // Calculate expected result for AI operations
    virtual function bit [DATA_WIDTH-1:0] calculate_expected_result();
        case (op_type)
            AI_RELU_OP: begin
                // ReLU: max(0, x)
                return (data[DATA_WIDTH-1] == 1'b1) ? 0 : data;
            end
            AI_SIGMOID_OP: begin
                // Simplified sigmoid approximation
                return data; // Placeholder - actual implementation would use lookup table
            end
            default: begin
                return data;
            end
        endcase
    endfunction
    
    // Validate transaction consistency
    virtual function bit is_valid();
        if (op_type inside {AI_MATMUL_OP}) begin
            return (matrix_m > 0 && matrix_n > 0 && matrix_k > 0);
        end
        if (op_type inside {AI_CONV2D_OP}) begin
            return (conv_height > 0 && conv_width > 0 && conv_channels > 0 && kernel_size > 0);
        end
        return 1;
    endfunction
    
endclass : riscv_ai_transaction

`endif // RISCV_AI_TRANSACTION_SV