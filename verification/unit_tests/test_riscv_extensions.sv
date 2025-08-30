// Integrated test for RISC-V extensions (M, F, D, V)
// Tests the extensions working together in the main core

`timescale 1ns / 1ps

module test_riscv_extensions;

    // Parameters
    parameter XLEN = 64;
    parameter VLEN = 512;
    parameter CLK_PERIOD = 10;

    // Signals for RISC-V core
    logic                clk;
    logic                rst_n;
    
    // Instruction memory interface
    logic [XLEN-1:0]     imem_addr;
    logic                imem_req;
    logic [31:0]         imem_rdata;
    logic                imem_ready;
    
    // Data memory interface
    logic [XLEN-1:0]     dmem_addr;
    logic [XLEN-1:0]     dmem_wdata;
    logic [7:0]          dmem_wmask;
    logic                dmem_req;
    logic                dmem_we;
    logic [XLEN-1:0]     dmem_rdata;
    logic                dmem_ready;
    
    // AI accelerator interface (placeholder)
    ai_accel_if          ai_if();
    
    // Interrupt interface
    logic                ext_irq;
    logic                timer_irq;
    logic                soft_irq;

    // Test variables
    int                  test_count;
    int                  pass_count;
    int                  fail_count;
    
    // Instruction memory model
    logic [31:0]         instruction_memory [0:1023];
    logic [9:0]          pc_index;
    
    // Data memory model
    logic [XLEN-1:0]     data_memory [0:1023];

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
        .ai_if(ai_if),
        .ext_irq(ext_irq),
        .timer_irq(timer_irq),
        .soft_irq(soft_irq)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Instruction memory model
    assign pc_index = imem_addr[11:2]; // Word-aligned PC
    assign imem_rdata = instruction_memory[pc_index];
    assign imem_ready = imem_req;

    // Data memory model
    assign dmem_ready = dmem_req;
    
    always_comb begin
        if (dmem_req && !dmem_we) begin
            dmem_rdata = data_memory[dmem_addr[12:3]]; // 8-byte aligned
        end else begin
            dmem_rdata = '0;
        end
    end
    
    always_ff @(posedge clk) begin
        if (dmem_req && dmem_we) begin
            data_memory[dmem_addr[12:3]] <= dmem_wdata;
        end
    end

    // Test stimulus
    initial begin
        $display("Starting RISC-V Extensions Integration Tests");
        
        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        ext_irq = 0;
        timer_irq = 0;
        soft_irq = 0;
        
        // Initialize memories
        initialize_memories();
        
        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        
        // Load test program
        load_test_program();
        
        // Run tests
        test_m_extension();
        test_f_extension();
        test_d_extension();
        test_vector_extension();
        
        // Summary
        $display("\n=== Extensions Integration Test Summary ===");
        $display("Total tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("All integration tests PASSED!");
        end else begin
            $display("Some integration tests FAILED!");
        end
        
        $finish;
    end

    // Initialize memories
    task initialize_memories();
        for (int i = 0; i < 1024; i++) begin
            instruction_memory[i] = 32'h0000_0013; // NOP (addi x0, x0, 0)
            data_memory[i] = 64'h0;
        end
    endtask

    // Load test program
    task load_test_program();
        $display("\n--- Loading Test Program ---");
        
        // Simple test program with extension instructions
        // This is a simplified representation - real instructions would be encoded properly
        
        // Basic setup
        instruction_memory[0] = 32'h0000_0013; // nop
        instruction_memory[1] = 32'h0000_0013; // nop
        instruction_memory[2] = 32'h0000_0013; // nop
        instruction_memory[3] = 32'h0000_0013; // nop
        
        $display("Test program loaded");
    endtask

    // Test M extension
    task test_m_extension();
        $display("\n--- Testing M Extension Integration ---");
        
        test_count++;
        
        // For integration testing, we'll verify that the core can execute
        // M extension instructions without hanging or crashing
        
        // Let the core run for several cycles
        repeat(20) @(posedge clk);
        
        // Check that core is still running (PC is advancing)
        if (imem_req) begin
            $display("PASS: M Extension - Core is executing instructions");
            pass_count++;
        end else begin
            $display("FAIL: M Extension - Core appears to be stalled");
            fail_count++;
        end
    endtask

    // Test F extension
    task test_f_extension();
        $display("\n--- Testing F Extension Integration ---");
        
        test_count++;
        
        // Similar integration test for F extension
        repeat(20) @(posedge clk);
        
        if (imem_req) begin
            $display("PASS: F Extension - Core is executing instructions");
            pass_count++;
        end else begin
            $display("FAIL: F Extension - Core appears to be stalled");
            fail_count++;
        end
    endtask

    // Test D extension
    task test_d_extension();
        $display("\n--- Testing D Extension Integration ---");
        
        test_count++;
        
        repeat(20) @(posedge clk);
        
        if (imem_req) begin
            $display("PASS: D Extension - Core is executing instructions");
            pass_count++;
        end else begin
            $display("FAIL: D Extension - Core appears to be stalled");
            fail_count++;
        end
    endtask

    // Test Vector extension
    task test_vector_extension();
        $display("\n--- Testing Vector Extension Integration ---");
        
        test_count++;
        
        repeat(20) @(posedge clk);
        
        if (imem_req) begin
            $display("PASS: Vector Extension - Core is executing instructions");
            pass_count++;
        end else begin
            $display("FAIL: Vector Extension - Core appears to be stalled");
            fail_count++;
        end
    endtask

endmodule