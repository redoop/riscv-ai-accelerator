// RISC-V AI Sequence Item
// Extended sequence item with AI-specific functionality

`ifndef RISCV_AI_SEQUENCE_ITEM_SV
`define RISCV_AI_SEQUENCE_ITEM_SV

class riscv_ai_sequence_item extends riscv_ai_transaction;
    
    // Additional sequence-specific fields
    rand int delay_cycles;
    rand bit burst_mode;
    rand int burst_length;
    
    // Performance tracking
    int throughput_mbps;
    int power_consumption_mw;
    
    // Error injection
    rand bit inject_error;
    rand bit [3:0] error_type;
    
    // Constraints for sequence items
    constraint delay_c {
        delay_cycles inside {[0:100]};
    }
    
    constraint burst_c {
        if (burst_mode) {
            burst_length inside {[2:16]};
        } else {
            burst_length == 1;
        }
    }
    
    constraint error_injection_c {
        inject_error dist {0 := 95, 1 := 5};  // 5% error injection rate
        if (inject_error) {
            error_type inside {[1:8]};
        }
    }
    
    // UVM automation
    `uvm_object_utils_begin(riscv_ai_sequence_item)
        `uvm_field_int(delay_cycles, UVM_ALL_ON)
        `uvm_field_int(burst_mode, UVM_ALL_ON)
        `uvm_field_int(burst_length, UVM_ALL_ON)
        `uvm_field_int(throughput_mbps, UVM_ALL_ON)
        `uvm_field_int(power_consumption_mw, UVM_ALL_ON)
        `uvm_field_int(inject_error, UVM_ALL_ON)
        `uvm_field_int(error_type, UVM_ALL_ON)
    `uvm_object_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_sequence_item");
        super.new(name);
    endfunction
    
    // Pre-randomize function
    function void pre_randomize();
        super.pre_randomize();
    endfunction
    
    // Post-randomize function
    function void post_randomize();
        super.post_randomize();
        
        // Calculate performance metrics
        calculate_performance_metrics();
        
        // Validate constraints
        if (!is_valid()) begin
            `uvm_error("SEQ_ITEM", "Invalid sequence item generated")
        end
    endfunction
    
    // Calculate performance metrics
    virtual function void calculate_performance_metrics();
        // Throughput calculation based on operation type and data size
        case (op_type)
            AI_MATMUL_OP: begin
                throughput_mbps = (matrix_m * matrix_n * matrix_k * get_data_type_bytes()) / 1000;
            end
            AI_CONV2D_OP: begin
                throughput_mbps = (conv_height * conv_width * conv_channels * get_data_type_bytes()) / 1000;
            end
            default: begin
                throughput_mbps = (size * get_data_type_bytes()) / 1000;
            end
        endcase
        
        // Power consumption estimation
        case (data_type)
            INT8_TYPE: power_consumption_mw = throughput_mbps * 10;
            FP16_TYPE: power_consumption_mw = throughput_mbps * 15;
            FP32_TYPE: power_consumption_mw = throughput_mbps * 25;
            default: power_consumption_mw = throughput_mbps * 20;
        endcase
    endfunction
    
    // Get data type size in bytes
    virtual function int get_data_type_bytes();
        case (data_type)
            INT8_TYPE: return 1;
            INT16_TYPE: return 2;
            INT32_TYPE: return 4;
            FP16_TYPE: return 2;
            FP32_TYPE: return 4;
            FP64_TYPE: return 8;
            default: return 4;
        endcase
    endfunction
    
    // Create AI-specific sequence items
    static function riscv_ai_sequence_item create_matmul_item(int m, int n, int k, data_type_e dtype);
        riscv_ai_sequence_item item = riscv_ai_sequence_item::type_id::create("matmul_item");
        item.op_type = AI_MATMUL_OP;
        item.matrix_m = m;
        item.matrix_n = n;
        item.matrix_k = k;
        item.data_type = dtype;
        return item;
    endfunction
    
    static function riscv_ai_sequence_item create_conv2d_item(int h, int w, int c, int ks, data_type_e dtype);
        riscv_ai_sequence_item item = riscv_ai_sequence_item::type_id::create("conv2d_item");
        item.op_type = AI_CONV2D_OP;
        item.conv_height = h;
        item.conv_width = w;
        item.conv_channels = c;
        item.kernel_size = ks;
        item.data_type = dtype;
        return item;
    endfunction
    
endclass : riscv_ai_sequence_item

`endif // RISCV_AI_SEQUENCE_ITEM_SV