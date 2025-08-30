// Testbench for TPU MAC Array
// Tests the functionality of MAC units and systolic array
// Covers INT8, FP16, and FP32 data types

`timescale 1ns/1ps

module test_tpu_mac_array;

    // Parameters
    parameter ARRAY_SIZE = 8;  // Smaller array for testing
    parameter DATA_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Test control
    logic test_enable;
    logic [1:0] test_data_type;
    
    // DUT signals
    logic enable;
    logic [1:0] data_type;
    logic load_weights;
    logic start_compute;
    logic accumulate_mode;
    
    logic [DATA_WIDTH-1:0] a_inputs [ARRAY_SIZE-1:0];
    logic [DATA_WIDTH-1:0] b_inputs [ARRAY_SIZE-1:0];
    logic [DATA_WIDTH-1:0] c_inputs [ARRAY_SIZE-1:0];
    logic [DATA_WIDTH-1:0] results [ARRAY_SIZE-1:0];
    
    logic computation_done;
    logic overflow_detected;
    logic underflow_detected;
    logic [31:0] cycles_count;
    logic [31:0] ops_count;
    
    // Test variables
    int test_case;
    int error_count;
    logic [DATA_WIDTH-1:0] expected_results [ARRAY_SIZE-1:0];
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    tpu_systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_type(data_type),
        .load_weights(load_weights),
        .start_compute(start_compute),
        .accumulate_mode(accumulate_mode),
        .a_inputs(a_inputs),
        .b_inputs(b_inputs),
        .c_inputs(c_inputs),
        .results(results),
        .computation_done(computation_done),
        .overflow_detected(overflow_detected),
        .underflow_detected(underflow_detected),
        .cycles_count(cycles_count),
        .ops_count(ops_count)
    );
    
    // Test stimulus
    initial begin
        $display("Starting TPU MAC Array Tests");
        
        // Initialize
        rst_n = 0;
        enable = 0;
        data_type = 2'b00;
        load_weights = 0;
        start_compute = 0;
        accumulate_mode = 0;
        test_case = 0;
        error_count = 0;
        
        // Clear input arrays
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = '0;
            b_inputs[i] = '0;
            c_inputs[i] = '0;
            expected_results[i] = '0;
        end
        
        // Reset sequence
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        // Test Case 1: INT8 Matrix Multiplication
        test_int8_matrix_mult();
        
        // Test Case 2: FP16 Matrix Multiplication  
        test_fp16_matrix_mult();
        
        // Test Case 3: FP32 Matrix Multiplication
        test_fp32_matrix_mult();
        
        // Test Case 4: Accumulation Mode
        test_accumulation_mode();
        
        // Test Case 5: Error Detection
        test_error_detection();
        
        // Test Case 6: Performance Counters
        test_performance_counters();
        
        // Summary
        $display("\n=== Test Summary ===");
        $display("Total test cases: %0d", test_case);
        $display("Errors found: %0d", error_count);
        
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end
        
        $finish;
    end    
   
 // Test INT8 matrix multiplication
    task test_int8_matrix_mult();
        test_case++;
        $display("\nTest Case %0d: INT8 Matrix Multiplication", test_case);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        
        // Load test weights (identity-like pattern)
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            for (int j = 0; j < ARRAY_SIZE; j++) begin
                if (i == j)
                    b_inputs[j] = 32'h01010101;  // 1 in each INT8 position
                else
                    b_inputs[j] = 32'h00000000;
            end
            @(posedge clk);
        end
        load_weights = 0;
        
        // Setup input activations
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h02020202;  // 2 in each INT8 position
            c_inputs[i] = 32'h00000000;  // No initial partial sum
            expected_results[i] = 32'h00000002;  // Expected: 2*1 = 2
        end
        
        // Start computation
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        
        // Wait for completion
        wait(computation_done);
        @(posedge clk);
        
        // Check results
        check_results("INT8 Matrix Mult");
        
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Test FP16 matrix multiplication
    task test_fp16_matrix_mult();
        test_case++;
        $display("\nTest Case %0d: FP16 Matrix Multiplication", test_case);
        
        data_type = 2'b01;  // FP16
        enable = 1;
        
        // Load test weights
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            b_inputs[i] = 32'h3C003C00;  // 1.0 in FP16 format (simplified)
            @(posedge clk);
        end
        load_weights = 0;
        
        // Setup input activations
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h40004000;  // 2.0 in FP16 format (simplified)
            c_inputs[i] = 32'h00000000;
            expected_results[i] = 32'h40004000;  // Expected: 2.0*1.0 = 2.0
        end
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        
        wait(computation_done);
        @(posedge clk);
        
        check_results("FP16 Matrix Mult");
        
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Test FP32 matrix multiplication
    task test_fp32_matrix_mult();
        test_case++;
        $display("\nTest Case %0d: FP32 Matrix Multiplication", test_case);
        
        data_type = 2'b10;  // FP32
        enable = 1;
        
        // Load test weights
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            b_inputs[i] = 32'h3F800000;  // 1.0 in FP32 format
            @(posedge clk);
        end
        load_weights = 0;
        
        // Setup input activations
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h40000000;  // 2.0 in FP32 format
            c_inputs[i] = 32'h00000000;
            expected_results[i] = 32'h40000000;  // Expected: 2.0*1.0 = 2.0
        end
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        
        wait(computation_done);
        @(posedge clk);
        
        check_results("FP32 Matrix Mult");
        
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Test accumulation mode
    task test_accumulation_mode();
        test_case++;
        $display("\nTest Case %0d: Accumulation Mode", test_case);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        accumulate_mode = 1;
        
        // Load weights
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            b_inputs[i] = 32'h01010101;
            @(posedge clk);
        end
        load_weights = 0;
        
        // First computation
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h01010101;  // 1
            c_inputs[i] = 32'h00000000;  // No initial sum
        end
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        wait(computation_done);
        
        // Second computation with accumulation
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h01010101;  // 1
            c_inputs[i] = results[i];    // Use previous results
            expected_results[i] = 32'h00000002;  // Expected: 1+1 = 2
        end
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        wait(computation_done);
        
        check_results("Accumulation Mode");
        
        accumulate_mode = 0;
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Test error detection
    task test_error_detection();
        test_case++;
        $display("\nTest Case %0d: Error Detection", test_case);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        
        // Load large weights to cause overflow
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            b_inputs[i] = 32'h7F7F7F7F;  // Maximum INT8 values
            @(posedge clk);
        end
        load_weights = 0;
        
        // Large input values
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h7F7F7F7F;  // Maximum INT8 values
            c_inputs[i] = 32'h7FFFFFFF;  // Large partial sum
        end
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        
        wait(computation_done);
        @(posedge clk);
        
        // Check for overflow detection
        if (overflow_detected) begin
            $display("  PASS: Overflow correctly detected");
        end else begin
            $display("  FAIL: Overflow not detected");
            error_count++;
        end
        
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Test performance counters
    task test_performance_counters();
        test_case++;
        $display("\nTest Case %0d: Performance Counters", test_case);
        
        data_type = 2'b00;
        enable = 1;
        
        // Reset counters
        enable = 0;
        @(posedge clk);
        enable = 1;
        
        // Simple computation
        load_weights = 1;
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            b_inputs[i] = 32'h01010101;
            @(posedge clk);
        end
        load_weights = 0;
        
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_inputs[i] = 32'h01010101;
            c_inputs[i] = 32'h00000000;
        end
        
        logic [31:0] start_cycles = cycles_count;
        logic [31:0] start_ops = ops_count;
        
        start_compute = 1;
        @(posedge clk);
        start_compute = 0;
        
        wait(computation_done);
        @(posedge clk);
        
        // Check counter increments
        if (cycles_count > start_cycles) begin
            $display("  PASS: Cycle counter incremented (%0d cycles)", cycles_count - start_cycles);
        end else begin
            $display("  FAIL: Cycle counter not incremented");
            error_count++;
        end
        
        if (ops_count > start_ops) begin
            $display("  PASS: Operations counter incremented (%0d ops)", ops_count - start_ops);
        end else begin
            $display("  FAIL: Operations counter not incremented");
            error_count++;
        end
        
        enable = 0;
        #(CLK_PERIOD * 2);
    endtask
    
    // Helper task to check results
    task check_results(string test_name);
        logic pass = 1;
        
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            if (results[i] !== expected_results[i]) begin
                $display("  FAIL: %s - Index %0d: Expected 0x%08x, Got 0x%08x", 
                        test_name, i, expected_results[i], results[i]);
                pass = 0;
                error_count++;
            end
        end
        
        if (pass) begin
            $display("  PASS: %s", test_name);
        end
    endtask

endmodule