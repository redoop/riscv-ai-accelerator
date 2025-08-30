// RISC-V AI Directed Sequence
// Directed test sequences for specific functionality verification

`ifndef RISCV_AI_DIRECTED_SEQUENCE_SV
`define RISCV_AI_DIRECTED_SEQUENCE_SV

class riscv_ai_directed_sequence extends riscv_ai_base_sequence;
    
    `uvm_object_utils(riscv_ai_directed_sequence)
    
    function new(string name = "riscv_ai_directed_sequence");
        super.new(name);
        num_transactions = 100;  // Smaller, focused tests
    endfunction
    
    virtual task body();
        `uvm_error(get_type_name(), "Base directed sequence body() called - should be overridden")
    endtask
    
endclass : riscv_ai_directed_sequence

// Matrix multiplication focused sequence
class riscv_ai_matmul_sequence extends riscv_ai_directed_sequence;
    
    // Matrix dimensions to test
    int test_dimensions[] = '{8, 16, 32, 64, 128, 256, 512, 1024};
    
    `uvm_object_utils(riscv_ai_matmul_sequence)
    
    function new(string name = "riscv_ai_matmul_sequence");
        super.new(name);
    endfunction
    
    virtual task body();
        riscv_ai_sequence_item item;
        
        // Test different matrix sizes
        foreach (test_dimensions[i]) begin
            foreach (test_dimensions[j]) begin
                foreach (test_dimensions[k]) begin
                    item = riscv_ai_sequence_item::create_matmul_item(
                        test_dimensions[i], test_dimensions[j], test_dimensions[k], FP32_TYPE);
                    
                    `uvm_info(get_type_name(), 
                             $sformatf("Testing MATMUL %0dx%0dx%0d", 
                                      test_dimensions[i], test_dimensions[j], test_dimensions[k]), UVM_MEDIUM)
                    
                    send_transaction(item);
                    wait_cycles(10);
                end
            end
        end
        
        // Test different data types
        data_type_e data_types[] = '{INT8_TYPE, FP16_TYPE, FP32_TYPE};
        foreach (data_types[dt]) begin
            item = riscv_ai_sequence_item::create_matmul_item(64, 64, 64, data_types[dt]);
            `uvm_info(get_type_name(), 
                     $sformatf("Testing MATMUL with data type %s", data_types[dt].name()), UVM_MEDIUM)
            send_transaction(item);
            wait_cycles(10);
        end
    endtask
    
endclass : riscv_ai_matmul_sequence

// Convolution focused sequence
class riscv_ai_conv2d_sequence extends riscv_ai_directed_sequence;
    
    // Common convolution configurations
    typedef struct {
        int height, width, channels, kernel_size;
    } conv_config_t;
    
    conv_config_t conv_configs[] = '{
        '{224, 224, 3, 3},    // First layer of ResNet
        '{112, 112, 64, 3},   // ResNet basic block
        '{56, 56, 128, 3},    // ResNet basic block
        '{28, 28, 256, 3},    // ResNet basic block
        '{14, 14, 512, 3},    // ResNet basic block
        '{7, 7, 512, 1}       // 1x1 convolution
    };
    
    `uvm_object_utils(riscv_ai_conv2d_sequence)
    
    function new(string name = "riscv_ai_conv2d_sequence");
        super.new(name);
    endfunction
    
    virtual task body();
        riscv_ai_sequence_item item;
        
        // Test common convolution configurations
        foreach (conv_configs[i]) begin
            item = riscv_ai_sequence_item::create_conv2d_item(
                conv_configs[i].height, conv_configs[i].width, 
                conv_configs[i].channels, conv_configs[i].kernel_size, FP32_TYPE);
            
            `uvm_info(get_type_name(), 
                     $sformatf("Testing CONV2D %0dx%0dx%0d with kernel %0dx%0d", 
                              conv_configs[i].height, conv_configs[i].width, conv_configs[i].channels,
                              conv_configs[i].kernel_size, conv_configs[i].kernel_size), UVM_MEDIUM)
            
            send_transaction(item);
            wait_cycles(20);
        end
        
        // Test different strides and padding
        int strides[] = '{1, 2, 4};
        int paddings[] = '{0, 1, 2};
        
        foreach (strides[s]) begin
            foreach (paddings[p]) begin
                item = riscv_ai_sequence_item::create_conv2d_item(32, 32, 64, 3, FP16_TYPE);
                if (!item.randomize() with {
                    stride == strides[s];
                    padding == paddings[p];
                }) begin
                    `uvm_error(get_type_name(), "Failed to randomize conv2d item")
                    continue;
                end
                
                `uvm_info(get_type_name(), 
                         $sformatf("Testing CONV2D with stride=%0d, padding=%0d", strides[s], paddings[p]), UVM_MEDIUM)
                
                send_transaction(item);
                wait_cycles(15);
            end
        end
    endtask
    
endclass : riscv_ai_conv2d_sequence

// Activation function sequence
class riscv_ai_activation_sequence extends riscv_ai_directed_sequence;
    
    operation_type_e activation_ops[] = '{AI_RELU_OP, AI_SIGMOID_OP};
    data_type_e data_types[] = '{FP16_TYPE, FP32_TYPE};
    
    `uvm_object_utils(riscv_ai_activation_sequence)
    
    function new(string name = "riscv_ai_activation_sequence");
        super.new(name);
    endfunction
    
    virtual task body();
        riscv_ai_sequence_item item;
        
        // Test all activation functions with different data types
        foreach (activation_ops[op]) begin
            foreach (data_types[dt]) begin
                for (int i = 0; i < 50; i++) begin
                    item = riscv_ai_sequence_item::type_id::create($sformatf("activation_item_%0d", i));
                    
                    if (!item.randomize() with {
                        op_type == activation_ops[op];
                        data_type == data_types[dt];
                        size inside {[64:1024]};
                    }) begin
                        `uvm_error(get_type_name(), "Failed to randomize activation item")
                        continue;
                    end
                    
                    send_transaction(item);
                    wait_cycles(2);
                end
                
                `uvm_info(get_type_name(), 
                         $sformatf("Completed %s testing with %s", 
                                  activation_ops[op].name(), data_types[dt].name()), UVM_MEDIUM)
            end
        end
    endtask
    
endclass : riscv_ai_activation_sequence

// Memory access pattern sequence
class riscv_ai_memory_sequence extends riscv_ai_directed_sequence;
    
    `uvm_object_utils(riscv_ai_memory_sequence)
    
    function new(string name = "riscv_ai_memory_sequence");
        super.new(name);
    endfunction
    
    virtual task body();
        riscv_ai_sequence_item item;
        
        // Sequential access pattern
        `uvm_info(get_type_name(), "Testing sequential memory access pattern", UVM_MEDIUM)
        for (int i = 0; i < 100; i++) begin
            item = riscv_ai_sequence_item::type_id::create($sformatf("seq_mem_item_%0d", i));
            if (!item.randomize() with {
                op_type inside {READ_OP, WRITE_OP};
                addr == 64'h1000_0000 + (i * 64);
                size == 64;
            }) begin
                `uvm_error(get_type_name(), "Failed to randomize sequential memory item")
                continue;
            end
            send_transaction(item);
        end
        
        wait_cycles(50);
        
        // Random access pattern
        `uvm_info(get_type_name(), "Testing random memory access pattern", UVM_MEDIUM)
        for (int i = 0; i < 100; i++) begin
            item = riscv_ai_sequence_item::type_id::create($sformatf("rand_mem_item_%0d", i));
            if (!item.randomize() with {
                op_type inside {READ_OP, WRITE_OP};
                addr inside {[64'h1000_0000:64'h1FFF_FFFF]};
                size inside {[8, 16, 32, 64, 128, 256]};
            }) begin
                `uvm_error(get_type_name(), "Failed to randomize random memory item")
                continue;
            end
            send_transaction(item);
        end
        
        wait_cycles(50);
        
        // Burst access pattern
        `uvm_info(get_type_name(), "Testing burst memory access pattern", UVM_MEDIUM)
        for (int i = 0; i < 20; i++) begin
            item = riscv_ai_sequence_item::type_id::create($sformatf("burst_mem_item_%0d", i));
            if (!item.randomize() with {
                op_type inside {READ_OP, WRITE_OP};
                burst_mode == 1;
                burst_length inside {[4, 8, 16]};
                addr inside {[64'h2000_0000:64'h2FFF_FFFF]};
                size == 64;
            }) begin
                `uvm_error(get_type_name(), "Failed to randomize burst memory item")
                continue;
            end
            send_transaction(item);
            wait_cycles(5);
        end
    endtask
    
endclass : riscv_ai_memory_sequence

`endif // RISCV_AI_DIRECTED_SEQUENCE_SV