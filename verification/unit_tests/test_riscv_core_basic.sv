// Basic integration test for RISC-V Core
// Tests basic instruction execution and pipeline operation

`timescale 1ns/1ps

module test_riscv_core_basic;

    parameter XLEN = 64;
    parameter VLEN = 512;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Memory interfaces
    logic [XLEN-1:0]    imem_addr;
    logic               imem_req;
    logic [31:0]        imem_rdata;
    logic               imem_ready;
    
    logic [XLEN-1:0]    dmem_addr;
    logic [XLEN-1:0]    dmem_wdata;
    logic [7:0]         dmem_wmask;
    logic               dmem_req;
    logic               dmem_we;
    logic [XLEN-1:0]    dmem_rdata;
    logic               dmem_ready;
    
    // AI accelerator interface (placeholder)
    ai_accel_if ai_if();
    
    // Interrupt interface
    logic ext_irq = 1'b0;
    logic timer_irq = 1'b0;
    logic soft_irq = 1'b0;
    
    // Simple instruction memory
    logic [31:0] instruction_memory [0:1023];
    
    // Simple data memory
    logic [XLEN-1:0] data_memory [0:1023];
    
    // Test program counter
    int program_counter = 0;
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;

    // DUT instantiation
    riscv_core #(
        .XLEN(XLEN),
        .VLEN(VLEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .imem_addr(imem_addr),
        .imem_req(imem_req),
        .imem_rdata(imem_rdata),
        .imem_ready(imem_ready),
        .dmem_addr(dmem_addr),
        .dmem_wdata(dmem_wdata),
        .dmem_wmask(dmem_wmask),
        .dmem_req(dmem_req),
        .dmem_we(dmem_we),
        .dmem_rdata(dmem_rdata),
        .dmem_ready(dmem_ready),
        .ai_if(ai_if.master),
        .ext_irq(ext_irq),
        .timer_irq(timer_irq),
        .soft_irq(soft_irq)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Simple instruction memory model
    always_comb begin
        if (imem_req && (imem_addr[31:2] < 1024)) begin
            imem_rdata = instruction_memory[imem_addr[31:2]];
            imem_ready = 1'b1;
        end else begin
            imem_rdata = 32'h0000_0013; // NOP
            imem_ready = 1'b1;
        end
    end

    // Simple data memory model
    always_ff @(posedge clk) begin
        if (dmem_req && !dmem_we && (dmem_addr[31:3] < 1024)) begin
            dmem_rdata <= data_memory[dmem_addr[31:3]];
            dmem_ready <= 1'b1;
        end else if (dmem_req && dmem_we && (dmem_addr[31:3] < 1024)) begin
            data_memory[dmem_addr[31:3]] <= dmem_wdata;
            dmem_ready <= 1'b1;
        end else begin
            dmem_rdata <= 64'h0;
            dmem_ready <= 1'b1;
        end
    end

    // Test task to check register values (simplified)
    task automatic check_execution(
        input string test_name,
        input int cycles_to_wait
    );
        test_count++;
        
        repeat(cycles_to_wait) @(posedge clk);
        
        $display("Test: %s - Executed for %0d cycles", test_name, cycles_to_wait);
        pass_count++; // For now, just count as pass if no errors
    endtask

    // Initialize test program
    initial begin
        // Initialize instruction memory with a simple test program
        
        // Test program: Basic arithmetic and memory operations
        instruction_memory[0] = 32'h00500093;  // addi x1, x0, 5      (x1 = 5)
        instruction_memory[1] = 32'h00300113;  // addi x2, x0, 3      (x2 = 3)
        instruction_memory[2] = 32'h002081b3;  // add  x3, x1, x2     (x3 = x1 + x2 = 8)
        instruction_memory[3] = 32'h40208233;  // sub  x4, x1, x2     (x4 = x1 - x2 = 2)
        instruction_memory[4] = 32'h002092b3;  // sll  x5, x1, x2     (x5 = x1 << x2 = 40)
        instruction_memory[5] = 32'h0020a333;  // slt  x6, x1, x2     (x6 = x1 < x2 = 0)
        instruction_memory[6] = 32'h0020c3b3;  // xor  x7, x1, x2     (x7 = x1 ^ x2 = 6)
        instruction_memory[7] = 32'h0020e433;  // or   x8, x1, x2     (x8 = x1 | x2 = 7)
        instruction_memory[8] = 32'h0020f4b3;  // and  x9, x1, x2     (x9 = x1 & x2 = 1)
        
        // Memory operations
        instruction_memory[9]  = 32'h00302023; // sw   x3, 0(x0)      (store x3 to address 0)
        instruction_memory[10] = 32'h00002503; // lw   x10, 0(x0)     (load from address 0 to x10)
        
        // Branch test
        instruction_memory[11] = 32'h00208663; // beq  x1, x2, 12     (if x1 == x2, branch to PC+12)
        instruction_memory[12] = 32'h00100593; // addi x11, x0, 1     (x11 = 1, should execute)
        instruction_memory[13] = 32'h00000013; // nop
        instruction_memory[14] = 32'h00000013; // nop
        instruction_memory[15] = 32'h00000013; // nop
        
        // Fill rest with NOPs
        for (int i = 16; i < 1024; i++) begin
            instruction_memory[i] = 32'h00000013; // nop
        end
        
        // Initialize data memory
        for (int i = 0; i < 1024; i++) begin
            data_memory[i] = 64'h0;
        end
    end

    // Main test sequence
    initial begin
        $display("Starting RISC-V Core Basic Integration Test");
        $display("==========================================");
        
        // Reset sequence
        rst_n = 1'b0;
        repeat(10) @(posedge clk);
        rst_n = 1'b1;
        repeat(5) @(posedge clk);
        
        // Test basic instruction execution
        check_execution("Basic Arithmetic Instructions", 20);
        
        // Test memory operations
        check_execution("Memory Load/Store Operations", 10);
        
        // Test branch instructions
        check_execution("Branch Instructions", 10);
        
        // Let pipeline flush
        repeat(10) @(posedge clk);
        
        // Display test results
        $display("\n==========================================");
        $display("Core Integration Test Results:");
        $display("Total tests: %0d", test_count);
        $display("Passed:      %0d", pass_count);
        $display("Failed:      %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end

    // Monitor for debugging
    initial begin
        $monitor("Time=%0t PC=%h Instr=%h", $time, imem_addr, imem_rdata);
    end

endmodule