/**
 * Testbench for ECC Controller
 * Verifies ECC encoding, decoding, error detection and correction
 */

`timescale 1ns/1ps

module test_ecc_controller;

    // Parameters
    parameter DATA_WIDTH = 64;
    parameter ECC_WIDTH = 8;
    parameter ADDR_WIDTH = 32;
    parameter CLK_PERIOD = 10;

    // Signals
    logic                    clk;
    logic                    rst_n;
    logic                    mem_req;
    logic                    mem_we;
    logic [ADDR_WIDTH-1:0]  mem_addr;
    logic [DATA_WIDTH-1:0]  mem_wdata;
    logic [DATA_WIDTH-1:0]  mem_rdata;
    logic                    mem_ready;
    logic                    single_error;
    logic                    double_error;
    logic [ADDR_WIDTH-1:0]  error_addr;
    logic                    error_inject_en;
    logic [1:0]              error_inject_type;
    logic                    array_req;
    logic                    array_we;
    logic [ADDR_WIDTH-1:0]  array_addr;
    logic [DATA_WIDTH+ECC_WIDTH-1:0] array_wdata;
    logic [DATA_WIDTH+ECC_WIDTH-1:0] array_rdata;
    logic                    array_ready;

    // Memory array simulation
    logic [DATA_WIDTH+ECC_WIDTH-1:0] memory_array [0:1023];
    
    // DUT instantiation
    ecc_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mem_req(mem_req),
        .mem_we(mem_we),
        .mem_addr(mem_addr),
        .mem_wdata(mem_wdata),
        .mem_rdata(mem_rdata),
        .mem_ready(mem_ready),
        .single_error(single_error),
        .double_error(double_error),
        .error_addr(error_addr),
        .error_inject_en(error_inject_en),
        .error_inject_type(error_inject_type),
        .array_req(array_req),
        .array_we(array_we),
        .array_addr(array_addr),
        .array_wdata(array_wdata),
        .array_rdata(array_rdata),
        .array_ready(array_ready)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Memory array simulation
    always_ff @(posedge clk) begin
        array_ready <= array_req;
        if (array_req && array_we) begin
            memory_array[array_addr[9:0]] <= array_wdata;
        end else if (array_req && !array_we) begin
            array_rdata <= memory_array[array_addr[9:0]];
        end
    end

    // Test variables
    logic [DATA_WIDTH-1:0] test_data;
    logic [DATA_WIDTH-1:0] read_data;
    int error_count;
    int test_count;

    // Test tasks
    task reset_system();
        rst_n = 0;
        mem_req = 0;
        mem_we = 0;
        mem_addr = 0;
        mem_wdata = 0;
        error_inject_en = 0;
        error_inject_type = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    task write_memory(input [ADDR_WIDTH-1:0] addr, input [DATA_WIDTH-1:0] data);
        @(posedge clk);
        mem_req = 1;
        mem_we = 1;
        mem_addr = addr;
        mem_wdata = data;
        @(posedge clk);
        while (!mem_ready) @(posedge clk);
        mem_req = 0;
        mem_we = 0;
        @(posedge clk);
    endtask

    task read_memory(input [ADDR_WIDTH-1:0] addr, output [DATA_WIDTH-1:0] data);
        @(posedge clk);
        mem_req = 1;
        mem_we = 0;
        mem_addr = addr;
        @(posedge clk);
        while (!mem_ready) @(posedge clk);
        data = mem_rdata;
        mem_req = 0;
        @(posedge clk);
    endtask

    task inject_error(input [1:0] error_type);
        error_inject_en = 1;
        error_inject_type = error_type;
        @(posedge clk);
        error_inject_en = 0;
    endtask

    // Test scenarios
    initial begin
        $display("Starting ECC Controller Test");
        
        // Initialize
        error_count = 0;
        test_count = 0;
        reset_system();

        // Test 1: Basic write/read without errors
        $display("Test 1: Basic write/read without errors");
        test_data = 64'hDEADBEEFCAFEBABE;
        write_memory(32'h100, test_data);
        read_memory(32'h100, read_data);
        
        if (read_data == test_data) begin
            $display("PASS: Basic write/read test");
        end else begin
            $display("FAIL: Basic write/read test - Expected: %h, Got: %h", test_data, read_data);
            error_count++;
        end
        test_count++;

        // Test 2: Single bit error injection and correction
        $display("Test 2: Single bit error injection and correction");
        test_data = 64'h123456789ABCDEF0;
        write_memory(32'h200, test_data);
        
        // Inject single bit error
        inject_error(2'b01);
        read_memory(32'h200, read_data);
        
        if (read_data == test_data && single_error) begin
            $display("PASS: Single bit error correction");
        end else begin
            $display("FAIL: Single bit error correction - Expected: %h, Got: %h, Single Error: %b", 
                    test_data, read_data, single_error);
            error_count++;
        end
        test_count++;

        // Test 3: Double bit error detection
        $display("Test 3: Double bit error detection");
        test_data = 64'hFEDCBA9876543210;
        write_memory(32'h300, test_data);
        
        // Inject double bit error
        inject_error(2'b10);
        read_memory(32'h300, read_data);
        
        if (double_error) begin
            $display("PASS: Double bit error detection");
        end else begin
            $display("FAIL: Double bit error detection - Double Error: %b", double_error);
            error_count++;
        end
        test_count++;

        // Test 4: Multiple memory locations
        $display("Test 4: Multiple memory locations");
        for (int i = 0; i < 16; i++) begin
            test_data = 64'h0123456789ABCDEF + i;
            write_memory(32'h400 + i*8, test_data);
        end
        
        for (int i = 0; i < 16; i++) begin
            read_memory(32'h400 + i*8, read_data);
            if (read_data != (64'h0123456789ABCDEF + i)) begin
                $display("FAIL: Multiple locations test at address %h", 32'h400 + i*8);
                error_count++;
            end
        end
        
        if (error_count == 0) begin
            $display("PASS: Multiple memory locations test");
        end
        test_count++;

        // Test 5: Error address reporting
        $display("Test 5: Error address reporting");
        test_data = 64'hA5A5A5A5A5A5A5A5;
        write_memory(32'h500, test_data);
        
        inject_error(2'b01);
        read_memory(32'h500, read_data);
        
        if (error_addr == 32'h500 && single_error) begin
            $display("PASS: Error address reporting");
        end else begin
            $display("FAIL: Error address reporting - Expected: %h, Got: %h", 32'h500, error_addr);
            error_count++;
        end
        test_count++;

        // Test 6: Stress test with random data
        $display("Test 6: Stress test with random data");
        for (int i = 0; i < 100; i++) begin
            test_data = $random();
            write_memory(32'h600 + i*8, test_data);
            read_memory(32'h600 + i*8, read_data);
            
            if (read_data != test_data) begin
                $display("FAIL: Stress test at iteration %d", i);
                error_count++;
                break;
            end
        end
        
        if (error_count == 0) begin
            $display("PASS: Stress test with random data");
        end
        test_count++;

        // Test summary
        $display("\n=== ECC Controller Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Failed tests: %d", error_count);
        $display("Success rate: %.1f%%", (test_count - error_count) * 100.0 / test_count);
        
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end

        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_ecc_controller.vcd");
        $dumpvars(0, test_ecc_controller);
    end

endmodule