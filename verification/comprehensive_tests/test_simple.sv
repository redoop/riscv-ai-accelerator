// Simple Test for Basic Functionality
// Minimal test to verify compilation and basic operation

`timescale 1ns/1ps

module test_simple;

    // Basic signals
    logic clk, rst_n;
    logic [31:0] counter;
    logic test_pass;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end
    
    // Reset generation
    initial begin
        rst_n = 0;
        #50;
        rst_n = 1;
    end
    
    // Simple counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 32'b0;
        end else begin
            counter <= counter + 1;
        end
    end
    
    // Test logic
    initial begin
        $display("Starting simple test...");
        
        // Wait for reset
        wait(rst_n);
        repeat(10) @(posedge clk);
        
        // Check counter
        if (counter > 0) begin
            $display("✓ Counter is working: %0d", counter);
            test_pass = 1'b1;
        end else begin
            $display("✗ Counter not working");
            test_pass = 1'b0;
        end
        
        // Wait a bit more
        repeat(100) @(posedge clk);
        
        // Final check
        if (counter > 100 && test_pass) begin
            $display("🎉 SIMPLE TEST PASSED! 🎉");
            $display("Final counter value: %0d", counter);
        end else begin
            $display("❌ SIMPLE TEST FAILED");
        end
        
        $finish;
    end
    
    // Timeout protection
    initial begin
        repeat(1000) @(posedge clk);
        $display("Test timeout!");
        $finish;
    end

endmodule