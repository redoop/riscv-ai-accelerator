// RISC-V AI Monitor
// UVM monitor for observing AI accelerator transactions

`ifndef RISCV_AI_MONITOR_SV
`define RISCV_AI_MONITOR_SV

class riscv_ai_monitor extends uvm_monitor;
    
    // Virtual interface handle
    virtual riscv_ai_interface vif;
    
    // Analysis ports for sending transactions to scoreboard and coverage
    uvm_analysis_port #(riscv_ai_sequence_item) ap_request;
    uvm_analysis_port #(riscv_ai_sequence_item) ap_response;
    
    // Monitor configuration
    bit enable_coverage = 1;
    bit enable_protocol_checking = 1;
    bit enable_performance_monitoring = 1;
    
    // Performance monitoring
    int total_requests = 0;
    int total_responses = 0;
    int total_errors = 0;
    real total_bandwidth_mbps = 0.0;
    time last_activity_time = 0;
    
    // Protocol checking
    bit outstanding_requests[$];
    int max_outstanding = 16;
    
    // Coverage collection
    riscv_ai_sequence_item coverage_item;
    
    `uvm_component_utils_begin(riscv_ai_monitor)
        `uvm_field_int(enable_coverage, UVM_ALL_ON)
        `uvm_field_int(enable_protocol_checking, UVM_ALL_ON)
        `uvm_field_int(enable_performance_monitoring, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_monitor", uvm_component parent = null);
        super.new(name, parent);
        ap_request = new("ap_request", this);
        ap_response = new("ap_response", this);
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
            monitor_requests();
            monitor_responses();
            if (enable_protocol_checking) protocol_checker();
            if (enable_performance_monitoring) performance_monitor();
        join_none
    endtask
    
    // Monitor request transactions
    virtual task monitor_requests();
        riscv_ai_sequence_item req_item;
        
        forever begin
            @(posedge vif.clk);
            
            if (vif.rst_n && vif.valid && vif.ready) begin
                req_item = riscv_ai_sequence_item::type_id::create("monitored_request");
                
                // Capture request transaction
                capture_request(req_item);
                
                // Send to analysis port
                ap_request.write(req_item);
                
                // Update statistics
                total_requests++;
                last_activity_time = $time;
                
                // Protocol checking
                if (enable_protocol_checking) begin
                    outstanding_requests.push_back(1);
                    if (outstanding_requests.size() > max_outstanding) begin
                        `uvm_error(get_type_name(), $sformatf("Too many outstanding requests: %0d", outstanding_requests.size()))
                    end
                end
                
                `uvm_info(get_type_name(), $sformatf("Request monitored: %s", req_item.convert2string()), UVM_HIGH)
            end
        end
    endtask
    
    // Monitor response transactions
    virtual task monitor_responses();
        riscv_ai_sequence_item resp_item;
        
        forever begin
            @(posedge vif.clk);
            
            if (vif.rst_n && vif.response_valid) begin
                resp_item = riscv_ai_sequence_item::type_id::create("monitored_response");
                
                // Capture response transaction
                capture_response(resp_item);
                
                // Send to analysis port
                ap_response.write(resp_item);
                
                // Update statistics
                total_responses++;
                if (resp_item.error) total_errors++;
                last_activity_time = $time;
                
                // Protocol checking
                if (enable_protocol_checking) begin
                    if (outstanding_requests.size() > 0) begin
                        outstanding_requests.pop_front();
                    end else begin
                        `uvm_error(get_type_name(), "Received response without outstanding request")
                    end
                end
                
                `uvm_info(get_type_name(), $sformatf("Response monitored: error=%0b", resp_item.error), UVM_HIGH)
            end
        end
    endtask
    
    // Capture request transaction details
    virtual function void capture_request(riscv_ai_sequence_item item);
        item.addr = vif.addr;
        item.data = vif.data;
        item.size = vif.size;
        item.core_id = vif.core_id;
        item.tpu_id = vif.tpu_id;
        item.vpu_id = vif.vpu_id;
        item.valid = vif.valid;
        item.ready = vif.ready;
        item.start_time = $time;
        
        // Decode operation type
        case (vif.op_type)
            3'b000: item.op_type = READ_OP;
            3'b001: item.op_type = WRITE_OP;
            3'b010: item.op_type = AI_MATMUL_OP;
            3'b011: item.op_type = AI_CONV2D_OP;
            3'b100: item.op_type = AI_RELU_OP;
            3'b101: item.op_type = AI_SIGMOID_OP;
            3'b110: item.op_type = AI_MAXPOOL_OP;
            3'b111: item.op_type = AI_AVGPOOL_OP;
            default: item.op_type = READ_OP;
        endcase
        
        // Decode data type
        case (vif.data_type)
            3'b000: item.data_type = INT8_TYPE;
            3'b001: item.data_type = INT16_TYPE;
            3'b010: item.data_type = INT32_TYPE;
            3'b011: item.data_type = FP16_TYPE;
            3'b100: item.data_type = FP32_TYPE;
            3'b101: item.data_type = FP64_TYPE;
            default: item.data_type = FP32_TYPE;
        endcase
        
        // Capture AI-specific parameters
        if (item.op_type == AI_MATMUL_OP) begin
            item.matrix_m = vif.matrix_m;
            item.matrix_n = vif.matrix_n;
            item.matrix_k = vif.matrix_k;
        end
        
        if (item.op_type == AI_CONV2D_OP) begin
            item.conv_height = vif.conv_height;
            item.conv_width = vif.conv_width;
            item.conv_channels = vif.conv_channels;
            item.kernel_size = vif.kernel_size;
            item.stride = vif.stride;
            item.padding = vif.padding;
        end
        
        if (item.op_type inside {AI_MAXPOOL_OP, AI_AVGPOOL_OP}) begin
            item.pool_size = vif.pool_size;
            item.conv_height = vif.conv_height;
            item.conv_width = vif.conv_width;
        end
    endfunction
    
    // Capture response transaction details
    virtual function void capture_response(riscv_ai_sequence_item item);
        item.response = vif.response_data;
        item.error = vif.error;
        item.end_time = $time;
        
        // Calculate latency if start time is available
        if (item.start_time > 0) begin
            item.latency = (item.end_time - item.start_time) / 1ns;
        end
    endfunction
    
    // Protocol checker task
    virtual task protocol_checker();
        forever begin
            @(posedge vif.clk);
            
            if (!vif.rst_n) begin
                // Reset protocol state
                outstanding_requests.delete();
                continue;
            end
            
            // Check for protocol violations
            
            // Valid/Ready handshake checking
            if (vif.valid && !vif.ready) begin
                // Valid asserted but not ready - this is allowed, just wait
            end
            
            if (!vif.valid && vif.ready) begin
                // Ready without valid - this is normal
            end
            
            // Check for back-to-back transactions without proper handshake
            if (vif.valid && vif.ready) begin
                @(posedge vif.clk);
                if (vif.valid && !vif.ready) begin
                    `uvm_warning(get_type_name(), "Back-to-back transaction without ready deassertion")
                end
            end
            
            // Check for address alignment
            if (vif.valid && vif.ready) begin
                case (vif.size)
                    8: if (vif.addr[2:0] != 3'b000) `uvm_error(get_type_name(), "64-bit access not 8-byte aligned");
                    4: if (vif.addr[1:0] != 2'b00) `uvm_error(get_type_name(), "32-bit access not 4-byte aligned");
                    2: if (vif.addr[0] != 1'b0) `uvm_error(get_type_name(), "16-bit access not 2-byte aligned");
                endcase
            end
        end
    endtask
    
    // Performance monitoring task
    virtual task performance_monitor();
        time window_start_time;
        int window_requests = 0;
        int window_responses = 0;
        real window_bandwidth = 0.0;
        
        forever begin
            window_start_time = $time;
            window_requests = total_requests;
            window_responses = total_responses;
            
            // Wait for monitoring window (1000 clock cycles)
            repeat(1000) @(posedge vif.clk);
            
            // Calculate performance metrics for this window
            int req_delta = total_requests - window_requests;
            int resp_delta = total_responses - window_responses;
            time window_duration = $time - window_start_time;
            
            if (window_duration > 0) begin
                real req_rate = real'(req_delta) / (real'(window_duration) / 1ns) * 1e9;  // requests per second
                real resp_rate = real'(resp_delta) / (real'(window_duration) / 1ns) * 1e9; // responses per second
                
                `uvm_info(get_type_name(), $sformatf("Performance Window: Req Rate=%.2f/s, Resp Rate=%.2f/s", 
                         req_rate, resp_rate), UVM_MEDIUM)
            end
        end
    endtask
    
    // Check phase
    virtual function void check_phase(uvm_phase phase);
        super.check_phase(phase);
        
        // Check for any remaining outstanding requests
        if (outstanding_requests.size() > 0) begin
            `uvm_warning(get_type_name(), $sformatf("%0d outstanding requests at end of test", outstanding_requests.size()))
        end
        
        // Check for reasonable error rate
        if (total_requests > 0) begin
            real error_rate = real'(total_errors) / real'(total_responses) * 100.0;
            if (error_rate > 5.0) begin
                `uvm_warning(get_type_name(), $sformatf("High error rate: %.2f%%", error_rate))
            end
        end
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), $sformatf("Monitor Statistics:"), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Requests: %0d", total_requests), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Responses: %0d", total_responses), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Errors: %0d", total_errors), UVM_LOW)
        
        if (total_responses > 0) begin
            real error_rate = real'(total_errors) / real'(total_responses) * 100.0;
            `uvm_info(get_type_name(), $sformatf("  Error Rate: %.2f%%", error_rate), UVM_LOW)
        end
        
        `uvm_info(get_type_name(), $sformatf("  Outstanding Requests: %0d", outstanding_requests.size()), UVM_LOW)
    endfunction
    
endclass : riscv_ai_monitor

`endif // RISCV_AI_MONITOR_SV