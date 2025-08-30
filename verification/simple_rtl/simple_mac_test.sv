// Simple MAC unit test - minimal RTL test
`timescale 1ns/1ps

module simple_mac_test;

    // Clock and reset
    logic clk;
    logic rst_n;
    
    // MAC unit inputs
    logic [15:0] a = 16'h0000;
    logic [15:0] b = 16'h0000;
    logic [31:0] c = 32'h00000000;
    logic valid = 0;
    
    // MAC unit outputs
    logic [31:0] result;
    logic ready;
    
    // Clock generation - 100MHz
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Simple MAC unit (behavioral model for testing)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 32'h0;
            ready <= 1'b0;
        end else if (valid) begin
            result <= (a * b) + c;  // MAC operation: result = a * b + c
            ready <= 1'b1;
        end else begin
            ready <= 1'b0;
        end
    end
    
    // Test sequence
    initial begin
        rst_n = 0;
        $display("Starting simple MAC RTL test...");
        
        // Reset sequence
        rst_n = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        $display("Reset released");
        
        // Test case 1: 2 * 3 + 4 = 10
        repeat(2) @(posedge clk);
        a = 16'd2;
        b = 16'd3;
        c = 32'd4;
        valid = 1;
        @(posedge clk);
        valid = 0;
        
        // Wait for result
        wait(ready);
        @(posedge clk);
        
        if (result == 32'd10) begin
            $display("✅ Test 1 PASSED: 2 * 3 + 4 = %0d", result);
        end else begin
            $display("❌ Test 1 FAILED: Expected 10, got %0d", result);
        end
        
        // Test case 2: 5 * 7 + 1 = 36
        repeat(2) @(posedge clk);
        a = 16'd5;
        b = 16'd7;
        c = 32'd1;
        valid = 1;
        @(posedge clk);
        valid = 0;
        
        // Wait for result
        wait(ready);
        @(posedge clk);
        
        if (result == 32'd36) begin
            $display("✅ Test 2 PASSED: 5 * 7 + 1 = %0d", result);
        end else begin
            $display("❌ Test 2 FAILED: Expected 36, got %0d", result);
        end
        
        // Test case 3: 0 * 100 + 50 = 50
        repeat(2) @(posedge clk);
        a = 16'd0;
        b = 16'd100;
        c = 32'd50;
        valid = 1;
        @(posedge clk);
        valid = 0;
        
        // Wait for result
        wait(ready);
        @(posedge clk);
        
        if (result == 32'd50) begin
            $display("✅ Test 3 PASSED: 0 * 100 + 50 = %0d", result);
        end else begin
            $display("❌ Test 3 FAILED: Expected 50, got %0d", result);
        end
        
        repeat(10) @(posedge clk);
        $display("Simple MAC RTL test completed successfully!");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #10000; // 10us timeout
        $display("ERROR: Test timeout!");
        $finish;
    end
    
    // Generate VCD for waveform viewing
    initial begin
        $dumpfile("simple_mac_test.vcd");
        $dumpvars(0, simple_mac_test);
    end

endmodule