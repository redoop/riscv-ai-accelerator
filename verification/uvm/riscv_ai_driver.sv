// RISC-V AI Driver
// UVM driver for AI accelerator transactions

`ifndef RISCV_AI_DRIVER_SV
`define RISCV_AI_DRIVER_SV

class riscv_ai_driver extends uvm_driver #(riscv_ai_sequence_item);
    
    // Virtual interface handle
    virtual riscv_ai_interface vif;
    
    // Driver configuration
    bit enable_error_injection = 0;
    int max_outstanding_transactions = 16;
    int current_outstanding = 0;
    
    // Performance tracking
    int total_transactions = 0;
    int total_cycles = 0;
    real average_latency = 0.0;
    
    // Transaction queue for outstanding requests
    riscv_ai_sequence_item outstanding_queue[$];
    
    `uvm_component_utils_begin(riscv_ai_driver)
        `uvm_field_int(enable_error_injection, UVM_ALL_ON)
        `uvm_field_int(max_outstanding_transactions, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_driver", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Get virtual interface from config DB
        if (!uvm_config_db#(virtual riscv_ai_interface)::get(this, "", "vif", vif)) begin
            `uvm_fatal(get_type_name(), "Virtual interface not found in config DB")
        end
    endfunction
    
    // Run phase
    virtual task run_phase(uvm_phase phase);
        fork
            drive_transactions();
            monitor_responses();
            reset_handler();
        join_none
    endtask
    
    // Main driver task
    virtual task drive_transactions();
        riscv_ai_sequence_item req;
        
        forever begin
            // Wait for reset deassertion
            wait(vif.rst_n);
            
            // Get next transaction from sequencer
            seq_item_port.get_next_item(req);
            
            // Wait for available slot if queue is full
            while (current_outstanding >= max_outstanding_transactions) begin
                @(posedge vif.clk);
            end
            
            // Drive the transaction
            drive_item(req);
            
            // Add to outstanding queue
            outstanding_queue.push_back(req);
            current_outstanding++;
            
            // Indicate item is done
            seq_item_port.item_done();
        end
    endtask
    
    // Drive individual transaction
    virtual task drive_item(riscv_ai_sequence_item item);
        `uvm_info(get_type_name(), $sformatf("Driving transaction: %s", item.convert2string()), UVM_HIGH)
        
        // Wait for ready signal
        while (!vif.ready) begin
            @(posedge vif.clk);
        end
        
        // Apply delay if specified
        if (item.delay_cycles > 0) begin
            repeat(item.delay_cycles) @(posedge vif.clk);
        end
        
        // Drive transaction based on operation type
        case (item.op_type)
            READ_OP: drive_read_transaction(item);
            WRITE_OP: drive_write_transaction(item);
            AI_MATMUL_OP: drive_matmul_transaction(item);
            AI_CONV2D_OP: drive_conv2d_transaction(item);
            AI_RELU_OP: drive_activation_transaction(item);
            AI_SIGMOID_OP: drive_activation_transaction(item);
            AI_MAXPOOL_OP: drive_pooling_transaction(item);
            AI_AVGPOOL_OP: drive_pooling_transaction(item);
            AI_BATCHNORM_OP: drive_batchnorm_transaction(item);
            default: begin
                `uvm_error(get_type_name(), $sformatf("Unsupported operation type: %s", item.op_type.name()))
            end
        endcase
        
        total_transactions++;
    endtask
    
    // Drive read transaction
    virtual task drive_read_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.addr <= item.addr;
        vif.size <= item.size;
        vif.op_type <= READ_OP;
        vif.core_id <= item.core_id;
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("Read transaction driven: addr=0x%016h, size=%0d", item.addr, item.size), UVM_HIGH)
    endtask
    
    // Drive write transaction
    virtual task drive_write_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.addr <= item.addr;
        vif.data <= item.data;
        vif.size <= item.size;
        vif.op_type <= WRITE_OP;
        vif.core_id <= item.core_id;
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("Write transaction driven: addr=0x%016h, data=0x%016h, size=%0d", 
                 item.addr, item.data, item.size), UVM_HIGH)
    endtask
    
    // Drive matrix multiplication transaction
    virtual task drive_matmul_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.op_type <= AI_MATMUL_OP;
        vif.tpu_id <= item.tpu_id;
        vif.data_type <= item.data_type;
        vif.matrix_m <= item.matrix_m;
        vif.matrix_n <= item.matrix_n;
        vif.matrix_k <= item.matrix_k;
        vif.addr <= item.addr;  // Input matrix A address
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("MATMUL transaction driven: %0dx%0dx%0d, TPU=%0d, type=%s", 
                 item.matrix_m, item.matrix_n, item.matrix_k, item.tpu_id, item.data_type.name()), UVM_HIGH)
    endtask
    
    // Drive convolution transaction
    virtual task drive_conv2d_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.op_type <= AI_CONV2D_OP;
        vif.tpu_id <= item.tpu_id;
        vif.data_type <= item.data_type;
        vif.conv_height <= item.conv_height;
        vif.conv_width <= item.conv_width;
        vif.conv_channels <= item.conv_channels;
        vif.kernel_size <= item.kernel_size;
        vif.stride <= item.stride;
        vif.padding <= item.padding;
        vif.addr <= item.addr;  // Input tensor address
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("CONV2D transaction driven: %0dx%0dx%0d, kernel=%0d, TPU=%0d", 
                 item.conv_height, item.conv_width, item.conv_channels, item.kernel_size, item.tpu_id), UVM_HIGH)
    endtask
    
    // Drive activation function transaction
    virtual task drive_activation_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.op_type <= item.op_type;
        vif.vpu_id <= item.vpu_id;
        vif.data_type <= item.data_type;
        vif.size <= item.size;
        vif.addr <= item.addr;  // Input data address
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("Activation transaction driven: %s, VPU=%0d, size=%0d", 
                 item.op_type.name(), item.vpu_id, item.size), UVM_HIGH)
    endtask
    
    // Drive pooling transaction
    virtual task drive_pooling_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.op_type <= item.op_type;
        vif.vpu_id <= item.vpu_id;
        vif.data_type <= item.data_type;
        vif.pool_size <= item.pool_size;
        vif.conv_height <= item.conv_height;
        vif.conv_width <= item.conv_width;
        vif.addr <= item.addr;  // Input tensor address
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("Pooling transaction driven: %s, pool_size=%0d, VPU=%0d", 
                 item.op_type.name(), item.pool_size, item.vpu_id), UVM_HIGH)
    endtask
    
    // Drive batch normalization transaction
    virtual task drive_batchnorm_transaction(riscv_ai_sequence_item item);
        @(posedge vif.clk);
        vif.valid <= 1'b1;
        vif.op_type <= AI_BATCHNORM_OP;
        vif.vpu_id <= item.vpu_id;
        vif.data_type <= item.data_type;
        vif.conv_channels <= item.conv_channels;
        vif.size <= item.size;
        vif.addr <= item.addr;  // Input data address
        
        @(posedge vif.clk);
        vif.valid <= 1'b0;
        
        `uvm_info(get_type_name(), $sformatf("BatchNorm transaction driven: channels=%0d, VPU=%0d", 
                 item.conv_channels, item.vpu_id), UVM_HIGH)
    endtask
    
    // Monitor responses from DUT
    virtual task monitor_responses();
        riscv_ai_sequence_item item;
        
        forever begin
            @(posedge vif.clk);
            
            if (vif.response_valid) begin
                if (outstanding_queue.size() > 0) begin
                    item = outstanding_queue.pop_front();
                    current_outstanding--;
                    
                    // Update item with response
                    item.response = vif.response_data;
                    item.error = vif.error;
                    item.end_time = $time;
                    item.latency = (item.end_time - item.start_time) / 1ns;
                    
                    // Update performance statistics
                    total_cycles += item.latency;
                    average_latency = real'(total_cycles) / real'(total_transactions);
                    
                    `uvm_info(get_type_name(), $sformatf("Response received: error=%0b, latency=%0d cycles", 
                             item.error, item.latency), UVM_HIGH)
                end else begin
                    `uvm_warning(get_type_name(), "Received response but no outstanding transactions")
                end
            end
        end
    endtask
    
    // Handle reset
    virtual task reset_handler();
        forever begin
            @(negedge vif.rst_n);
            `uvm_info(get_type_name(), "Reset detected - clearing outstanding transactions", UVM_MEDIUM)
            
            // Clear all interface signals
            vif.valid <= 1'b0;
            vif.addr <= 64'h0;
            vif.data <= 64'h0;
            vif.size <= 32'h0;
            vif.op_type <= READ_OP;
            vif.core_id <= 8'h0;
            vif.tpu_id <= 8'h0;
            vif.vpu_id <= 8'h0;
            
            // Clear outstanding queue
            outstanding_queue.delete();
            current_outstanding = 0;
            
            // Wait for reset deassertion
            @(posedge vif.rst_n);
        end
    endtask
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), $sformatf("Driver Statistics:"), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Transactions: %0d", total_transactions), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Cycles: %0d", total_cycles), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Average Latency: %.2f cycles", average_latency), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Outstanding Transactions: %0d", current_outstanding), UVM_LOW)
    endfunction
    
endclass : riscv_ai_driver

`endif // RISCV_AI_DRIVER_SV