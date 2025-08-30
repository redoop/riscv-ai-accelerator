// Test using real TPU MAC unit from the project
`timescale 1ns/1ps

`include "../../rtl/config/chip_config.sv"

module test_real_tpu_mac;

    // Clock and reset
    logic clk;
    logic rst_n;
    
    // TPU MAC unit interface
    logic enable;
    logic [2:0] data_type;
    logic [15:0] a_data;
    logic [15:0] b_data;
    logic [31:0] c_data;
    logic valid_in;
    
    logic [31:0] result;
    logic valid_out;
    logic ready;
    
    // Clock generation - 100MHz
    initial clk = 0;
    always #5 clk = ~clk;
    
    // Instantiate the real TPU MAC unit
    tpu_mac_unit #(
        .DATA_WIDTH(32)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_type(data_type),
        .a_data(a_data),
        .b_data(b_data),
        .c_data(c_data),
        .valid_in(valid_in),
        .result(result),
        .valid_out(valid_out),
        .ready(ready)
    );
    
    // Test sequence
    initial begin
        $display("🔬 Testing Real TPU MAC Unit from RTL Project");
        $display("==============================================");
        
        // Initialize signals
        rst_n = 0;
        enable = 0;
        data_type = 3'b000; // INT8
        a_data = 16'h0000;
        b_data = 16'h0000;
        c_data = 32'h00000000;
        valid_in = 0;
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        enable = 1;
        $display("✅ Reset released, TPU MAC unit enabled");
        
        // Test case 1: Basic MAC operation
        repeat(2) @(posedge clk);
        a_data = 16'd10;
        b_data = 16'd20;
        c_data = 32'd5;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for result
        wait(valid_out);
        @(posedge clk);
        
        $display("🧮 Test 1: 10 * 20 + 5 = %0d", result);
        if (result == 32'd205) begin
            $display("✅ Test 1 PASSED");
        end else begin
            $display("❌ Test 1 FAILED: Expected 205, got %0d", result);
        end
        
        // Test case 2: Different data type
        repeat(5) @(posedge clk);
        data_type = 3'b010; // INT32
        a_data = 16'd7;
        b_data = 16'd8;
        c_data = 32'd100;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for result
        wait(valid_out);
        @(posedge clk);
        
        $display("🧮 Test 2: 7 * 8 + 100 = %0d", result);
        if (result == 32'd156) begin
            $display("✅ Test 2 PASSED");
        end else begin
            $display("❌ Test 2 FAILED: Expected 156, got %0d", result);
        end
        
        // Test case 3: Zero multiplication
        repeat(5) @(posedge clk);
        a_data = 16'd0;
        b_data = 16'd999;
        c_data = 32'd42;
        valid_in = 1;
        @(posedge clk);
        valid_in = 0;
        
        // Wait for result
        wait(valid_out);
        @(posedge clk);
        
        $display("🧮 Test 3: 0 * 999 + 42 = %0d", result);
        if (result == 32'd42) begin
            $display("✅ Test 3 PASSED");
        end else begin
            $display("❌ Test 3 FAILED: Expected 42, got %0d", result);
        end
        
        repeat(10) @(posedge clk);
        $display("🎉 Real TPU MAC RTL test completed!");
        $display("✨ Successfully executed actual RTL hardware code!");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #50000; // 50us timeout
        $display("❌ ERROR: Test timeout!");
        $finish;
    end
    
    // Generate VCD for waveform viewing
    initial begin
        $dumpfile("test_real_tpu_mac.vcd");
        $dumpvars(0, test_real_tpu_mac);
    end

endmodule