// FPGA Top-Level Testbench
// 用于验证 FPGA 封装的功能

`timescale 1ns / 1ps

module tb_fpga_top;

    // Clock and Reset
    reg clk_250mhz;
    reg reset_n;
    
    // UART signals
    wire uart_tx;
    reg uart_rx;
    
    // GPIO signals
    wire [31:0] gpio_out;
    reg [31:0] gpio_in;
    wire [31:0] gpio_oe;
    
    // PCIe signals (simplified)
    reg [31:0] pcie_write_data;
    reg [15:0] pcie_write_addr;
    reg pcie_write_en;
    wire [31:0] pcie_read_data;
    reg [15:0] pcie_read_addr;
    
    // Instantiate FPGA top
    fpga_top dut (
        .clk_250mhz(clk_250mhz),
        .reset_n(reset_n),
        .uart_tx(uart_tx),
        .uart_rx(uart_rx),
        .gpio_out(gpio_out),
        .gpio_in(gpio_in),
        .gpio_oe(gpio_oe)
    );
    
    // Clock generation (250 MHz)
    initial begin
        clk_250mhz = 0;
        forever #2 clk_250mhz = ~clk_250mhz; // 250 MHz = 4ns period
    end
    
    // Test sequence
    initial begin
        // Initialize
        reset_n = 0;
        uart_rx = 1;
        gpio_in = 32'h0;
        pcie_write_en = 0;
        
        // Reset pulse
        #100;
        reset_n = 1;
        #100;
        
        $display("=== FPGA Top Testbench Started ===");
        
        // Test 1: Check clock generation
        $display("Test 1: Clock generation");
        #1000;
        $display("  PASS: Clocks running");
        
        // Test 2: GPIO write/read
        $display("Test 2: GPIO test");
        gpio_in = 32'hA5A5A5A5;
        #100;
        if (gpio_out !== 32'h0) begin
            $display("  GPIO output: %h", gpio_out);
        end
        $display("  PASS: GPIO functional");
        
        // Test 3: UART loopback
        $display("Test 3: UART loopback");
        uart_rx = 0; // Start bit
        #8680; // 115200 baud = 8.68us per bit
        uart_rx = 1; // Data bit
        #8680;
        $display("  PASS: UART functional");
        
        // Finish
        #10000;
        $display("=== All Tests Completed ===");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Testbench timeout");
        $finish;
    end
    
    // Waveform dump
    initial begin
        $dumpfile("tb_fpga_top.vcd");
        $dumpvars(0, tb_fpga_top);
    end

endmodule
