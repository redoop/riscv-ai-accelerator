/**
 * Testbench for Error Injector
 * Verifies error injection patterns and control mechanisms
 */

`timescale 1ns/1ps

module test_error_injector;

    // Parameters
    parameter ADDR_WIDTH = 32;
    parameter DATA_WIDTH = 64;
    parameter CLK_PERIOD = 10;

    // Signals
    logic                    clk;
    logic                    rst_n;
    logic                    inject_enable;
    logic [3:0]              inject_mode;
    logic [ADDR_WIDTH-1:0]  inject_addr;
    logic [15:0]             inject_mask;
    logic [31:0]             inject_count;
    logic                    inject_trigger;
    logic                    mem_access;
    logic [ADDR_WIDTH-1:0]  mem_addr;
    logic                    mem_we;
    logic [DATA_WIDTH-1:0]  mem_wdata;
    logic [DATA_WIDTH-1:0]  mem_wdata_out;
    logic [DATA_WIDTH-1:0]  mem_rdata;
    logic [DATA_WIDTH-1:0]  mem_rdata_out;
    logic                    single_error_inject;
    logic                    double_error_inject;
    logic                    burst_error_inject;
    logic                    address_error_inject;
    logic                    control_error_inject;
    logic [31:0]             injection_count;
    logic                    injection_active;
    logic [ADDR_WIDTH-1:0]  last_inject_addr;

    // DUT instantiation
    error_injector #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .inject_enable(inject_enable),
        .inject_mode(inject_mode),
        .inject_addr(inject_addr),
        .inject_mask(inject_mask),
        .inject_count(inject_count),
        .inject_trigger(inject_trigger),
        .mem_access(mem_access),
        .mem_addr(mem_addr),
        .mem_we(mem_we),
        .mem_wdata(mem_wdata),
        .mem_wdata_out(mem_wdata_out),
        .mem_rdata(mem_rdata),
        .mem_rdata_out(mem_rdata_out),
        .single_error_inject(single_error_inject),
        .double_error_inject(double_error_inject),
        .burst_error_inject(burst_error_inject),
        .address_error_inject(address_error_inject),
        .control_error_inject(control_error_inject),
        .injection_count(injection_count),
        .injection_active(injection_active),
        .last_inject_addr(last_inject_addr)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test variables
    int error_count = 0;
    int test_count = 0;

    // Test tasks
    task reset_system();
        rst_n = 0;
        inject_enable = 0;
        inject_mode = 0;
        inject_addr = 0;
        inject_mask = 0;
        inject_count = 0;
        inject_trigger = 0;
        mem_access = 0;
        mem_addr = 0;
        mem_we = 0;
        mem_wdata = 0;
        mem_rdata = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    task memory_access(input [ADDR_WIDTH-1:0] addr, input logic we, input [DATA_WIDTH-1:0] wdata);
        mem_access = 1;
        mem_addr = addr;
        mem_we = we;
        mem_wdata = wdata;
        @(posedge clk);
        mem_access = 0;
        @(posedge clk);
    endtask

    // Test scenarios
    initial begin
        $display("Starting Error Injector Test");
        
        reset_system();

        // Test 1: Single bit error injection
        $display("Test 1: Single bit error injection");
        inject_enable = 1;
        inject_mode = 4'b0001; // INJECT_SINGLE_BIT
        inject_mask = 16'h0001; // Flip bit 0
        inject_trigger = 1;
        
        memory_access(32'h100, 1'b1, 64'hDEADBEEFCAFEBABE);
        
        if (single_error_inject && injection_active) begin
            $display("PASS: Single bit error injection");
        end else begin
            $display("FAIL: Single bit error injection");
            error_count++;
        end
        test_count++;

        // Test 2: Targeted address injection
        $display("Test 2: Targeted address injection");
        inject_enable = 1;
        inject_mode = 4'b1000; // INJECT_TARGETED
        inject_addr = 32'h200;
        inject_mask = 16'h0002; // Flip bit 1
        
        memory_access(32'h200, 1'b0, 64'h0); // Read from target address
        
        if (injection_active && last_inject_addr == 32'h200) begin
            $display("PASS: Targeted address injection");
        end else begin
            $display("FAIL: Targeted address injection");
            error_count++;
        end
        test_count++;

        // Test summary
        $display("\n=== Error Injector Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Failed tests: %d", error_count);
        
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end

        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_error_injector.vcd");
        $dumpvars(0, test_error_injector);
    end

endmodule