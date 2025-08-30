// RISC-V AI Scoreboard
// UVM scoreboard for checking AI accelerator functionality

`ifndef RISCV_AI_SCOREBOARD_SV
`define RISCV_AI_SCOREBOARD_SV

class riscv_ai_scoreboard extends uvm_scoreboard;
    
    // Analysis imports for receiving transactions
    uvm_analysis_imp_request #(riscv_ai_sequence_item, riscv_ai_scoreboard) ap_request;
    uvm_analysis_imp_response #(riscv_ai_sequence_item, riscv_ai_scoreboard) ap_response;
    
    // Transaction queues
    riscv_ai_sequence_item request_queue[$];
    riscv_ai_sequence_item response_queue[$];
    
    // Expected results queue
    riscv_ai_sequence_item expected_queue[$];
    
    // Statistics
    int total_requests = 0;
    int total_responses = 0;
    int total_matches = 0;
    int total_mismatches = 0;
    int total_errors = 0;
    
    // Performance tracking
    real total_latency = 0.0;
    real min_latency = 999999.0;
    real max_latency = 0.0;
    real average_latency = 0.0;
    
    // Throughput tracking
    real total_throughput_mbps = 0.0;
    real peak_throughput_mbps = 0.0;
    
    // Error tracking by operation type
    int error_count_by_op[operation_type_e];
    int request_count_by_op[operation_type_e];
    
    // Configuration
    bit enable_detailed_checking = 1;
    bit enable_performance_analysis = 1;
    real latency_threshold_ns = 1000.0;  // 1us threshold
    
    `uvm_component_utils_begin(riscv_ai_scoreboard)
        `uvm_field_int(enable_detailed_checking, UVM_ALL_ON)
        `uvm_field_int(enable_performance_analysis, UVM_ALL_ON)
        `uvm_field_real(latency_threshold_ns, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_scoreboard", uvm_component parent = null);
        super.new(name, parent);
        ap_request = new("ap_request", this);
        ap_response = new("ap_response", this);
        
        // Initialize error counters
        foreach (error_count_by_op[op]) begin
            error_count_by_op[op] = 0;
            request_count_by_op[op] = 0;
        end
    endfunction
    
    // Write method for request transactions
    virtual function void write_request(riscv_ai_sequence_item req);
        riscv_ai_sequence_item expected_resp;
        
        `uvm_info(get_type_name(), $sformatf("Received request: %s", req.convert2string()), UVM_HIGH)
        
        // Add to request queue
        request_queue.push_back(req);
        total_requests++;
        request_count_by_op[req.op_type]++;
        
        // Generate expected response
        expected_resp = generate_expected_response(req);
        if (expected_resp != null) begin
            expected_queue.push_back(expected_resp);
        end
    endfunction
    
    // Write method for response transactions
    virtual function void write_response(riscv_ai_sequence_item resp);
        riscv_ai_sequence_item expected_resp;
        bit match_found = 0;
        
        `uvm_info(get_type_name(), $sformatf("Received response: error=%0b", resp.error), UVM_HIGH)
        
        // Add to response queue
        response_queue.push_back(resp);
        total_responses++;
        
        if (resp.error) begin
            total_errors++;
            // Find corresponding request to update error count
            foreach (request_queue[i]) begin
                if (request_queue[i].start_time == resp.start_time) begin
                    error_count_by_op[request_queue[i].op_type]++;
                    break;
                end
            end
        end
        
        // Find matching expected response
        foreach (expected_queue[i]) begin
            if (compare_transactions(expected_queue[i], resp)) begin
                expected_resp = expected_queue[i];
                expected_queue.delete(i);
                match_found = 1;
                break;
            end
        end
        
        if (match_found) begin
            check_response(expected_resp, resp);
        end else begin
            `uvm_warning(get_type_name(), "No matching expected response found")
            total_mismatches++;
        end
        
        // Performance analysis
        if (enable_performance_analysis) begin
            analyze_performance(resp);
        end
    endfunction
    
    // Generate expected response for a request
    virtual function riscv_ai_sequence_item generate_expected_response(riscv_ai_sequence_item req);
        riscv_ai_sequence_item expected;
        
        expected = riscv_ai_sequence_item::type_id::create("expected_response");
        expected.copy(req);
        
        // Generate expected results based on operation type
        case (req.op_type)
            READ_OP: begin
                // For read operations, we can't predict the exact data
                // but we can check that no error occurred for valid addresses
                expected.error = (req.addr inside {[64'h0:64'hFFFF_FFFF_FFFF_FFFF]}) ? 0 : 1;
            end
            
            WRITE_OP: begin
                // Write operations should succeed for valid addresses
                expected.error = (req.addr inside {[64'h0:64'hFFFF_FFFF_FFFF_FFFF]}) ? 0 : 1;
                expected.response = req.data;  // Echo back written data
            end
            
            AI_RELU_OP: begin
                // ReLU: max(0, x)
                expected.response = req.calculate_expected_result();
                expected.error = 0;
            end
            
            AI_SIGMOID_OP: begin
                // Sigmoid approximation
                expected.response = req.calculate_expected_result();
                expected.error = 0;
            end
            
            AI_MATMUL_OP: begin
                // Matrix multiplication - complex calculation
                expected.response = calculate_matmul_result(req);
                expected.error = (req.matrix_m == 0 || req.matrix_n == 0 || req.matrix_k == 0) ? 1 : 0;
            end
            
            AI_CONV2D_OP: begin
                // Convolution - complex calculation
                expected.response = calculate_conv2d_result(req);
                expected.error = (req.conv_height == 0 || req.conv_width == 0 || req.conv_channels == 0) ? 1 : 0;
            end
            
            AI_MAXPOOL_OP, AI_AVGPOOL_OP: begin
                // Pooling operations
                expected.response = calculate_pooling_result(req);
                expected.error = (req.pool_size == 0) ? 1 : 0;
            end
            
            AI_BATCHNORM_OP: begin
                // Batch normalization
                expected.response = calculate_batchnorm_result(req);
                expected.error = 0;
            end
            
            default: begin
                `uvm_warning(get_type_name(), $sformatf("Unknown operation type: %s", req.op_type.name()))
                return null;
            end
        endcase
        
        return expected;
    endfunction
    
    // Compare two transactions for matching
    virtual function bit compare_transactions(riscv_ai_sequence_item expected, riscv_ai_sequence_item actual);
        // Simple comparison based on timing or transaction ID
        // In a real implementation, this would be more sophisticated
        return (expected.start_time == actual.start_time);
    endfunction
    
    // Check response against expected
    virtual function void check_response(riscv_ai_sequence_item expected, riscv_ai_sequence_item actual);
        bit data_match = 1;
        bit error_match = 1;
        
        // Check error status
        if (expected.error != actual.error) begin
            `uvm_error(get_type_name(), $sformatf("Error mismatch: expected=%0b, actual=%0b", expected.error, actual.error))
            error_match = 0;
        end
        
        // Check data only if no error expected
        if (!expected.error && enable_detailed_checking) begin
            case (expected.op_type)
                AI_RELU_OP, AI_SIGMOID_OP: begin
                    // For activation functions, check exact match
                    if (expected.response != actual.response) begin
                        `uvm_error(get_type_name(), $sformatf("Data mismatch for %s: expected=0x%016h, actual=0x%016h", 
                                 expected.op_type.name(), expected.response, actual.response))
                        data_match = 0;
                    end
                end
                
                AI_MATMUL_OP, AI_CONV2D_OP: begin
                    // For complex operations, allow some tolerance due to floating point precision
                    real tolerance = get_tolerance_for_data_type(expected.data_type);
                    if (!within_tolerance(expected.response, actual.response, tolerance)) begin
                        `uvm_error(get_type_name(), $sformatf("Data mismatch for %s: expected=0x%016h, actual=0x%016h", 
                                 expected.op_type.name(), expected.response, actual.response))
                        data_match = 0;
                    end
                end
                
                default: begin
                    // For other operations, exact match required
                    if (expected.response != actual.response) begin
                        `uvm_warning(get_type_name(), $sformatf("Data mismatch for %s: expected=0x%016h, actual=0x%016h", 
                                   expected.op_type.name(), expected.response, actual.response))
                        data_match = 0;
                    end
                end
            endcase
        end
        
        if (data_match && error_match) begin
            total_matches++;
            `uvm_info(get_type_name(), $sformatf("Transaction check PASSED for %s", expected.op_type.name()), UVM_HIGH)
        end else begin
            total_mismatches++;
            `uvm_error(get_type_name(), $sformatf("Transaction check FAILED for %s", expected.op_type.name()))
        end
    endfunction
    
    // Performance analysis
    virtual function void analyze_performance(riscv_ai_sequence_item resp);
        real latency_ns;
        
        if (resp.latency > 0) begin
            latency_ns = real'(resp.latency);
            
            // Update latency statistics
            total_latency += latency_ns;
            if (latency_ns < min_latency) min_latency = latency_ns;
            if (latency_ns > max_latency) max_latency = latency_ns;
            average_latency = total_latency / real'(total_responses);
            
            // Check latency threshold
            if (latency_ns > latency_threshold_ns) begin
                `uvm_warning(get_type_name(), $sformatf("High latency detected: %.2f ns (threshold: %.2f ns)", 
                           latency_ns, latency_threshold_ns))
            end
        end
        
        // Update throughput statistics
        if (resp.throughput_mbps > 0) begin
            total_throughput_mbps += resp.throughput_mbps;
            if (resp.throughput_mbps > peak_throughput_mbps) begin
                peak_throughput_mbps = resp.throughput_mbps;
            end
        end
    endfunction
    
    // Calculate expected results for complex operations
    virtual function bit [63:0] calculate_matmul_result(riscv_ai_sequence_item req);
        // Simplified matrix multiplication result calculation
        // In a real implementation, this would involve actual matrix operations
        return req.data + (req.matrix_m * req.matrix_n * req.matrix_k);
    endfunction
    
    virtual function bit [63:0] calculate_conv2d_result(riscv_ai_sequence_item req);
        // Simplified convolution result calculation
        return req.data + (req.conv_height * req.conv_width * req.conv_channels);
    endfunction
    
    virtual function bit [63:0] calculate_pooling_result(riscv_ai_sequence_item req);
        // Simplified pooling result calculation
        case (req.op_type)
            AI_MAXPOOL_OP: return req.data | (64'h1 << req.pool_size);
            AI_AVGPOOL_OP: return req.data >> req.pool_size;
            default: return req.data;
        endcase
    endfunction
    
    virtual function bit [63:0] calculate_batchnorm_result(riscv_ai_sequence_item req);
        // Simplified batch normalization result
        return req.data ^ 64'hAAAA_AAAA_AAAA_AAAA;
    endfunction
    
    // Utility functions
    virtual function real get_tolerance_for_data_type(data_type_e dtype);
        case (dtype)
            FP16_TYPE: return 1e-3;
            FP32_TYPE: return 1e-6;
            FP64_TYPE: return 1e-12;
            default: return 0.0;  // Exact match for integer types
        endcase
    endfunction
    
    virtual function bit within_tolerance(bit [63:0] expected, bit [63:0] actual, real tolerance);
        if (tolerance == 0.0) begin
            return (expected == actual);
        end else begin
            // For floating point comparison, this is simplified
            // Real implementation would properly handle FP formats
            real diff = $abs($itor(expected) - $itor(actual));
            return (diff <= tolerance);
        end
    endfunction
    
    // Check phase
    virtual function void check_phase(uvm_phase phase);
        super.check_phase(phase);
        
        // Check for unmatched transactions
        if (expected_queue.size() > 0) begin
            `uvm_warning(get_type_name(), $sformatf("%0d expected responses not matched", expected_queue.size()))
        end
        
        // Check overall pass/fail ratio
        if (total_responses > 0) begin
            real pass_rate = real'(total_matches) / real'(total_responses) * 100.0;
            if (pass_rate < 95.0) begin
                `uvm_error(get_type_name(), $sformatf("Low pass rate: %.2f%% (threshold: 95%%)", pass_rate))
            end
        end
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== SCOREBOARD FINAL REPORT ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Requests: %0d", total_requests), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Responses: %0d", total_responses), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Matches: %0d", total_matches), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Mismatches: %0d", total_mismatches), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Errors: %0d", total_errors), UVM_LOW)
        
        if (total_responses > 0) begin
            real pass_rate = real'(total_matches) / real'(total_responses) * 100.0;
            real error_rate = real'(total_errors) / real'(total_responses) * 100.0;
            `uvm_info(get_type_name(), $sformatf("Pass Rate: %.2f%%", pass_rate), UVM_LOW)
            `uvm_info(get_type_name(), $sformatf("Error Rate: %.2f%%", error_rate), UVM_LOW)
        end
        
        if (enable_performance_analysis && total_responses > 0) begin
            `uvm_info(get_type_name(), "=== PERFORMANCE ANALYSIS ===", UVM_LOW)
            `uvm_info(get_type_name(), $sformatf("Average Latency: %.2f ns", average_latency), UVM_LOW)
            `uvm_info(get_type_name(), $sformatf("Min Latency: %.2f ns", min_latency), UVM_LOW)
            `uvm_info(get_type_name(), $sformatf("Max Latency: %.2f ns", max_latency), UVM_LOW)
            `uvm_info(get_type_name(), $sformatf("Peak Throughput: %.2f MB/s", peak_throughput_mbps), UVM_LOW)
        end
        
        // Report error breakdown by operation type
        `uvm_info(get_type_name(), "=== ERROR BREAKDOWN BY OPERATION ===", UVM_LOW)
        foreach (error_count_by_op[op]) begin
            if (request_count_by_op[op] > 0) begin
                real op_error_rate = real'(error_count_by_op[op]) / real'(request_count_by_op[op]) * 100.0;
                `uvm_info(get_type_name(), $sformatf("%s: %0d errors / %0d requests (%.2f%%)", 
                         op.name(), error_count_by_op[op], request_count_by_op[op], op_error_rate), UVM_LOW)
            end
        end
    endfunction
    
endclass : riscv_ai_scoreboard

`endif // RISCV_AI_SCOREBOARD_SV