// Simple TPU MAC Unit Test
// Basic functionality test for MAC unit

`timescale 1ns/1ps

module test_tpu_mac_simple;

    parameter DATA_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // DUT signals
    logic enable;
    logic [1:0] data_type;
    logic [DATA_WIDTH-1:0] a_in, b_in, c_in;
    logic [DATA_WIDTH-1:0] a_out, b_out, c_out;
    logic load_weight;
    logic accumulate;
    logic overflow, underflow;
    
    // Test control
    int test_count = 0;
    int pass_count = 0;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    tpu_mac_unit #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_type(data_type),
        .a_in(a_in),
        .b_in(b_in),
        .c_in(c_in),
        .a_out(a_out),
        .b_out(b_out),
        .c_out(c_out),
        .load_weight(load_weight),
        .accumulate(accumulate),
        .overflow(overflow),
        .underflow(underflow)
    );
    
    // Test sequence
    initial begin
        $display("=== Simple TPU MAC Unit Test ===");
        
        // Initialize
        rst_n = 0;
        enable = 0;
        data_type = 2'b00;
        a_in = 0;
        b_in = 0;
        c_in = 0;
        load_weight = 0;
        accumulate = 0;
        
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(4) @(posedge clk);
        
        // Test 1: Basic INT8 multiplication
        test_basic_int8();
        
        // Test 2: Weight loading
        test_weight_loading();
        
        // Test 3: Accumulation
        test_accumulation();
        
        // Summary
        $display("\n=== Test Summary ===");
        $display("Tests run: %0d", test_count);
        $display("Tests passed: %0d", pass_count);
        
        if (pass_count == test_count) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end
        
        $finish;
    end
    
    // Test basic INT8 multiplication
    task test_basic_int8();
        test_count++;
        $display("\nTest %0d: Basic INT8 multiplication", test_count);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        accumulate = 0;
        
        // Load weight
        load_weight = 1;
        b_in = 32'h05050505;  // Weight = 5
        repeat(2) @(posedge clk);
        load_weight = 0;
        
        // Multiply
        a_in = 32'h03030303;  // Input = 3
        c_in = 32'h00000000;  // No partial sum
        repeat(3) @(posedge clk);
        
        // Check result (3 * 5 = 15 = 0x0F)
        if (c_out[7:0] == 8'h0F) begin
            $display("  PASS: 3 * 5 = %0d", c_out[7:0]);
            pass_count++;
        end else begin
            $display("  FAIL: Expected 15, got %0d", c_out[7:0]);
        end
        
        enable = 0;
        repeat(2) @(posedge clk);
    endtask
    
    // Test weight loading
    task test_weight_loading();
        test_count++;
        $display("\nTest %0d: Weight loading", test_count);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        
        // Load different weight
        load_weight = 1;
        b_in = 32'h07070707;  // Weight = 7
        repeat(2) @(posedge clk);
        load_weight = 0;
        
        // Multiply with new weight
        a_in = 32'h02020202;  // Input = 2
        c_in = 32'h00000000;
        repeat(3) @(posedge clk);
        
        // Check result (2 * 7 = 14 = 0x0E)
        if (c_out[7:0] == 8'h0E) begin
            $display("  PASS: 2 * 7 = %0d", c_out[7:0]);
            pass_count++;
        end else begin
            $display("  FAIL: Expected 14, got %0d", c_out[7:0]);
        end
        
        enable = 0;
        repeat(2) @(posedge clk);
    endtask
    
    // Test accumulation
    task test_accumulation();
        test_count++;
        $display("\nTest %0d: Accumulation", test_count);
        
        data_type = 2'b00;  // INT8
        enable = 1;
        accumulate = 1;
        
        // Load weight
        load_weight = 1;
        b_in = 32'h04040404;  // Weight = 4
        repeat(2) @(posedge clk);
        load_weight = 0;
        
        // First multiplication with accumulation
        a_in = 32'h03030303;  // Input = 3
        c_in = 32'h05050505;  // Partial sum = 5
        repeat(3) @(posedge clk);
        
        // Check result (3 * 4 + 5 = 17 = 0x11)
        if (c_out[7:0] == 8'h11) begin
            $display("  PASS: 3 * 4 + 5 = %0d", c_out[7:0]);
            pass_count++;
        end else begin
            $display("  FAIL: Expected 17, got %0d", c_out[7:0]);
        end
        
        accumulate = 0;
        enable = 0;
        repeat(2) @(posedge clk);
    endtask

endmodule
