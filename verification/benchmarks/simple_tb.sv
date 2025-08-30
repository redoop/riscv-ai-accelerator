`timescale 1ns/1ps

// Simple testbench for basic RTL verification without UVM
module simple_tb;

    // Clock and reset
    logic clk = 0;
    logic rst_n = 0;
    
    // Simple test signals
    logic [31:0] test_data = 32'h12345678;
    logic test_valid = 0;
    logic test_ready;
    
    // Clock generation - 100MHz
    always #5 clk = ~clk;
    
    // Reset sequence
    initial begin
        $display("Starting simple RTL test...");
        rst_n = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        $display("Reset released");
        
        // Simple test sequence
        repeat(5) @(posedge clk);
        test_valid = 1;
        @(posedge clk);
        test_valid = 0;
        
        repeat(20) @(posedge clk);
        $display("Test completed successfully");
        $finish;
    end
    
    // Simple monitoring
    always @(posedge clk) begin
        if (rst_n && test_valid) begin
            $display("Time %0t: Test data = 0x%h", $time, test_data);
        end
    end

endmodule