// RISC-V AI Random Sequence
// Randomized test sequence for comprehensive coverage

`ifndef RISCV_AI_RANDOM_SEQUENCE_SV
`define RISCV_AI_RANDOM_SEQUENCE_SV

class riscv_ai_random_sequence extends riscv_ai_base_sequence;
    
    // Randomization weights for different operation types
    int matmul_weight = 30;
    int conv2d_weight = 25;
    int activation_weight = 20;
    int pooling_weight = 15;
    int memory_weight = 10;
    
    // UVM automation
    `uvm_object_utils_begin(riscv_ai_random_sequence)
        `uvm_field_int(matmul_weight, UVM_ALL_ON)
        `uvm_field_int(conv2d_weight, UVM_ALL_ON)
        `uvm_field_int(activation_weight, UVM_ALL_ON)
        `uvm_field_int(pooling_weight, UVM_ALL_ON)
        `uvm_field_int(memory_weight, UVM_ALL_ON)
    `uvm_object_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_random_sequence");
        super.new(name);
    endfunction
    
    // Main sequence body
    virtual task body();
        riscv_ai_sequence_item item;
        
        for (int i = 0; i < num_transactions; i++) begin
            item = riscv_ai_sequence_item::type_id::create($sformatf("random_item_%0d", i));
            
            // Randomize operation type based on weights
            randomize_operation_type(item);
            
            // Send transaction
            send_transaction(item);
            
            // Random delay between transactions
            if (i < num_transactions - 1) begin
                wait_cycles($urandom_range(1, 10));
            end
        end
    endtask
    
    // Randomize operation type with weighted distribution
    virtual function void randomize_operation_type(riscv_ai_sequence_item item);
        int total_weight = matmul_weight + conv2d_weight + activation_weight + pooling_weight + memory_weight;
        int rand_val = $urandom_range(0, total_weight - 1);
        
        if (rand_val < matmul_weight) begin
            item.op_type = AI_MATMUL_OP;
        end else if (rand_val < matmul_weight + conv2d_weight) begin
            item.op_type = AI_CONV2D_OP;
        end else if (rand_val < matmul_weight + conv2d_weight + activation_weight) begin
            // Randomly choose activation function
            case ($urandom_range(0, 1))
                0: item.op_type = AI_RELU_OP;
                1: item.op_type = AI_SIGMOID_OP;
            endcase
        end else if (rand_val < matmul_weight + conv2d_weight + activation_weight + pooling_weight) begin
            // Randomly choose pooling operation
            case ($urandom_range(0, 1))
                0: item.op_type = AI_MAXPOOL_OP;
                1: item.op_type = AI_AVGPOOL_OP;
            endcase
        end else begin
            // Memory operations
            case ($urandom_range(0, 1))
                0: item.op_type = READ_OP;
                1: item.op_type = WRITE_OP;
            endcase
        end
    endfunction
    
endclass : riscv_ai_random_sequence

// Specialized random sequences for different scenarios

class riscv_ai_stress_sequence extends riscv_ai_random_sequence;
    
    `uvm_object_utils(riscv_ai_stress_sequence)
    
    function new(string name = "riscv_ai_stress_sequence");
        super.new(name);
        // Higher transaction count for stress testing
        num_transactions = 5000;
        // Favor compute-intensive operations
        matmul_weight = 40;
        conv2d_weight = 35;
        activation_weight = 15;
        pooling_weight = 10;
        memory_weight = 0;
    endfunction
    
    virtual task body();
        // Run multiple sequences in parallel for stress testing
        fork
            super.body();
            super.body();
            super.body();
        join
    endtask
    
endclass : riscv_ai_stress_sequence

class riscv_ai_power_sequence extends riscv_ai_random_sequence;
    
    `uvm_object_utils(riscv_ai_power_sequence)
    
    function new(string name = "riscv_ai_power_sequence");
        super.new(name);
        // Focus on power-sensitive operations
        matmul_weight = 50;  // High power consumption
        conv2d_weight = 30;  // High power consumption
        activation_weight = 10;
        pooling_weight = 5;
        memory_weight = 5;
    endfunction
    
    virtual task body();
        riscv_ai_sequence_item item;
        
        for (int i = 0; i < num_transactions; i++) begin
            item = riscv_ai_sequence_item::type_id::create($sformatf("power_item_%0d", i));
            
            randomize_operation_type(item);
            
            // Force high-power data types
            if (!item.randomize() with {
                data_type inside {FP32_TYPE, FP64_TYPE};
                if (op_type == AI_MATMUL_OP) {
                    matrix_m inside {[256:1024]};
                    matrix_n inside {[256:1024]};
                    matrix_k inside {[256:1024]};
                }
            }) begin
                `uvm_error(get_type_name(), "Failed to randomize power sequence item")
                continue;
            end
            
            send_transaction(item);
            
            // Longer delays to observe power transitions
            if (i < num_transactions - 1) begin
                wait_cycles($urandom_range(50, 200));
            end
        end
    endtask
    
endclass : riscv_ai_power_sequence

`endif // RISCV_AI_RANDOM_SEQUENCE_SV