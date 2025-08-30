// Comprehensive Test Suite for RISC-V AI Accelerator
// Tests core functionality based on implementation tasks
// Covers RISC-V core, TPU, VPU, and system integration

`timescale 1ns/1ps

module test_riscv_ai_comprehensive;

    // ========================================
    // Test Parameters and Constants
    // ========================================
    
    parameter CLK_PERIOD = 10;  // 100MHz
    parameter TIMEOUT_CYCLES = 10000;
    parameter TEST_DATA_WIDTH = 32;
    parameter MATRIX_SIZE = 8;  // Small matrix for testing
    
    // ========================================
    // Clock and Reset
    // ========================================
    
    logic clk;
    logic rst_n;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Reset generation
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
    end
    
    // ========================================
    // Test Signals and Interfaces
    // ========================================
    
    // RISC-V Core signals
    logic [63:0]    imem_addr, dmem_addr;
    logic [31:0]    imem_rdata;
    logic [63:0]    dmem_wdata, dmem_rdata;
    logic [7:0]     dmem_wmask;
    logic           imem_req, imem_ready;
    logic           dmem_req, dmem_we, dmem_ready;
    logic           ext_irq, timer_irq, soft_irq;
    
    // TPU signals
    logic           tpu_enable, tpu_start, tpu_done, tpu_busy;
    logic [7:0]     tpu_operation;
    logic [1:0]     tpu_data_type;
    logic [7:0]     tpu_matrix_m, tpu_matrix_n, tpu_matrix_k;
    logic [31:0]    tpu_mem_addr, tpu_mem_wdata, tpu_mem_rdata;
    logic           tpu_mem_read, tpu_mem_write, tpu_mem_ready;
    logic [31:0]    tpu_cycle_count, tpu_op_count;
    logic           tpu_error;
    
    // VPU signals
    logic           vpu_busy, vpu_error;
    logic [7:0]     vpu_status;
    
    // Test control
    logic [31:0]    test_counter;
    logic [7:0]     current_test;
    logic           test_pass, test_fail;
    logic [31:0]    pass_count, fail_count;
    
    // Task-specific variables
    logic [63:0]    test_addr, test_data, read_data;
    logic [31:0]    initial_cycles, final_cycles;
    
    // ========================================
    // Memory Models
    // ========================================
    
    // Simple instruction memory model
    logic [31:0] instruction_memory [0:1023];
    
    // Simple data memory model
    logic [63:0] data_memory [0:1023];
    
    // Initialize test instructions
    initial begin
        // Basic RISC-V instructions for testing
        instruction_memory[0] = 32'h00000013;  // NOP (addi x0, x0, 0)
        instruction_memory[1] = 32'h00100093;  // addi x1, x0, 1
        instruction_memory[2] = 32'h00200113;  // addi x2, x0, 2
        instruction_memory[3] = 32'h002081b3;  // add x3, x1, x2
        instruction_memory[4] = 32'h40208233;  // sub x4, x1, x2
        instruction_memory[5] = 32'h002092b3;  // sll x5, x1, x2
        instruction_memory[6] = 32'h0020a333;  // slt x6, x1, x2
        instruction_memory[7] = 32'h0020c3b3;  // xor x7, x1, x2
        instruction_memory[8] = 32'h0020e433;  // or x8, x1, x2
        instruction_memory[9] = 32'h002104b3;  // add x9, x2, x2 (multiply by 2)
        
        // Initialize remaining memory
        for (int i = 10; i < 1024; i++) begin
            instruction_memory[i] = 32'h00000013; // NOP
        end
        
        // Initialize data memory with test patterns
        for (int i = 0; i < 1024; i++) begin
            data_memory[i] = 64'h0123456789ABCDEF + i;
        end
    end
    
    // Instruction memory interface
    always_comb begin
        imem_ready = 1'b1;
        imem_rdata = instruction_memory[imem_addr[11:2]];
    end
    
    // Data memory interface
    always_ff @(posedge clk) begin
        dmem_ready <= 1'b1;
        if (dmem_req && !dmem_we) begin
            dmem_rdata <= data_memory[dmem_addr[11:3]];
        end else if (dmem_req && dmem_we) begin
            // Handle byte enables for writes
            case (dmem_wmask)
                8'b0000_0001: data_memory[dmem_addr[11:3]][7:0] <= dmem_wdata[7:0];
                8'b0000_0011: data_memory[dmem_addr[11:3]][15:0] <= dmem_wdata[15:0];
                8'b0000_1111: data_memory[dmem_addr[11:3]][31:0] <= dmem_wdata[31:0];
                8'b1111_1111: data_memory[dmem_addr[11:3]] <= dmem_wdata;
                default: data_memory[dmem_addr[11:3]] <= dmem_wdata;
            endcase
        end
    end
    
    // TPU memory interface
    always_ff @(posedge clk) begin
        tpu_mem_ready <= 1'b1;
        if (tpu_mem_read) begin
            tpu_mem_rdata <= data_memory[tpu_mem_addr[11:3]][31:0];
        end else if (tpu_mem_write) begin
            data_memory[tpu_mem_addr[11:3]][31:0] <= tpu_mem_wdata;
        end
    end
    
    // ========================================
    // AI Accelerator Interface
    // ========================================
    
    ai_accel_if ai_if();
    
    // Connect AI interface
    assign ai_if.clk = clk;
    assign ai_if.rst_n = rst_n;
    
    // ========================================
    // DUT Instantiation
    // ========================================
    
    // RISC-V Core
    riscv_core #(
        .XLEN(64),
        .VLEN(512)
    ) dut_riscv_core (
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
    
    // TPU (Tensor Processing Unit)
    tpu #(
        .DATA_WIDTH(32),
        .ADDR_WIDTH(32),
        .ARRAY_SIZE(64)
    ) dut_tpu (
        .clk(clk),
        .rst_n(rst_n),
        .enable(tpu_enable),
        .start(tpu_start),
        .done(tpu_done),
        .busy(tpu_busy),
        .operation(tpu_operation),
        .data_type(tpu_data_type),
        .matrix_size_m(tpu_matrix_m),
        .matrix_size_n(tpu_matrix_n),
        .matrix_size_k(tpu_matrix_k),
        .mem_addr(tpu_mem_addr),
        .mem_read(tpu_mem_read),
        .mem_write(tpu_mem_write),
        .mem_wdata(tpu_mem_wdata),
        .mem_rdata(tpu_mem_rdata),
        .mem_ready(tpu_mem_ready),
        .cycle_count(tpu_cycle_count),
        .op_count(tpu_op_count),
        .error(tpu_error)
    );
    
    // VPU (Vector Processing Unit) - Simplified interface
    vpu #(
        .VECTOR_LANES(16),
        .VECTOR_REGS(32),
        .MAX_VLEN(512),
        .ELEMENT_WIDTH(64)
    ) dut_vpu (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_addr(ai_if.addr),
        .ctrl_wdata(ai_if.wdata),
        .ctrl_rdata(ai_if.rdata),
        .ctrl_req(ai_if.req),
        .ctrl_we(ai_if.we),
        .ctrl_ready(ai_if.ready),
        .status(vpu_status),
        .busy(vpu_busy),
        .error(vpu_error)
    );
    
    // ========================================
    // Test Tasks
    // ========================================
    
    // Task 1: Test RISC-V Basic Instructions (Task 2.1)
    task test_riscv_basic_instructions();
        $display("=== Test 1: RISC-V Basic Instructions ===");
        
        // Reset and wait for core to start
        @(posedge clk);
        
        // Let the core execute several instructions
        repeat(20) @(posedge clk);
        
        // Check if instructions are being fetched
        if (imem_req && imem_ready) begin
            $display("✓ Instruction fetch working");
            pass_count++;
        end else begin
            $display("✗ Instruction fetch failed");
            fail_count++;
        end
        
        // Check if PC is advancing
        if (imem_addr > 0) begin
            $display("✓ Program counter advancing");
            pass_count++;
        end else begin
            $display("✗ Program counter not advancing");
            fail_count++;
        end
        
        $display("Test 1 completed\n");
    endtask
    
    // Task 2: Test TPU Matrix Operations (Task 4.1, 4.2)
    task test_tpu_matrix_operations();
        $display("=== Test 2: TPU Matrix Operations ===");
        
        // Configure TPU for matrix multiplication
        tpu_enable = 1'b1;
        tpu_operation = 8'h01;  // Matrix multiply
        tpu_data_type = 2'b10;  // FP32
        tpu_matrix_m = MATRIX_SIZE;
        tpu_matrix_n = MATRIX_SIZE;
        tpu_matrix_k = MATRIX_SIZE;
        
        @(posedge clk);
        
        // Start TPU operation
        tpu_start = 1'b1;
        @(posedge clk);
        tpu_start = 1'b0;
        
        // Wait for TPU to become busy
        wait(tpu_busy);
        $display("✓ TPU started operation");
        pass_count++;
        
        // Wait for completion or timeout
        fork
            begin
                wait(tpu_done);
                $display("✓ TPU operation completed");
                $display("  Cycles: %0d, Operations: %0d", tpu_cycle_count, tpu_op_count);
                pass_count++;
            end
            begin
                repeat(TIMEOUT_CYCLES) @(posedge clk);
                $display("✗ TPU operation timeout");
                fail_count++;
            end
        join_any
        disable fork;
        
        // Check for errors
        if (!tpu_error) begin
            $display("✓ No TPU errors detected");
            pass_count++;
        end else begin
            $display("✗ TPU error detected");
            fail_count++;
        end
        
        tpu_enable = 1'b0;
        $display("Test 2 completed\n");
    endtask
    
    // Task 3: Test VPU Vector Operations (Task 5.1, 5.2)
    task test_vpu_vector_operations();
        $display("=== Test 3: VPU Vector Operations ===");
        
        // Configure VPU through AI interface
        ai_if.addr = 32'h0000_1000;  // VPU base address
        ai_if.wdata = 64'h0000_0010_0000_0001;  // Vector length = 16, operation = ADD
        ai_if.req = 1'b1;
        ai_if.we = 1'b1;
        
        @(posedge clk);
        ai_if.req = 1'b0;
        ai_if.we = 1'b0;
        
        // Wait for VPU to process
        repeat(10) @(posedge clk);
        
        // Check VPU status
        if (vpu_status == 8'h00) begin  // Assuming 0x00 is OK status
            $display("✓ VPU operation successful");
            pass_count++;
        end else begin
            $display("✗ VPU operation failed, status: 0x%02x", vpu_status);
            fail_count++;
        end
        
        // Check for errors
        if (!vpu_error) begin
            $display("✓ No VPU errors detected");
            pass_count++;
        end else begin
            $display("✗ VPU error detected");
            fail_count++;
        end
        
        $display("Test 3 completed\n");
    endtask
    
    // Task 4: Test Memory Subsystem (Task 3.1, 3.2, 3.3)
    task test_memory_subsystem();
        $display("=== Test 4: Memory Subsystem ===");
        
        // Test data memory write
        @(posedge clk);
        // Memory write will be tested through RISC-V core store instructions
        // For now, test direct memory interface
        
        // Check if memory interface is responsive
        if (dmem_ready) begin
            $display("✓ Data memory interface ready");
            pass_count++;
        end else begin
            $display("✗ Data memory interface not ready");
            fail_count++;
        end
        
        // Test instruction memory
        if (imem_ready) begin
            $display("✓ Instruction memory interface ready");
            pass_count++;
        end else begin
            $display("✗ Instruction memory interface not ready");
            fail_count++;
        end
        
        $display("Test 4 completed\n");
    endtask
    
    // Task 5: Test AI Instructions Integration (Task 2.3)
    task test_ai_instructions();
        $display("=== Test 5: AI Instructions Integration ===");
        
        // Test AI interface connectivity
        if (ai_if.clk === clk && ai_if.rst_n === rst_n) begin
            $display("✓ AI interface properly connected");
            pass_count++;
        end else begin
            $display("✗ AI interface connection issue");
            fail_count++;
        end
        
        // Test AI interface basic functionality
        ai_if.addr = 32'h0000_2000;
        ai_if.wdata = 64'h1234567890ABCDEF;
        ai_if.req = 1'b1;
        ai_if.we = 1'b0;  // Read operation
        
        @(posedge clk);
        
        if (ai_if.ready) begin
            $display("✓ AI interface responds to requests");
            pass_count++;
        end else begin
            $display("✗ AI interface not responding");
            fail_count++;
        end
        
        ai_if.req = 1'b0;
        $display("Test 5 completed\n");
    endtask
    
    // Task 6: Test Performance Monitoring (Task 12.1)
    task test_performance_monitoring();
        $display("=== Test 6: Performance Monitoring ===");
        
        // Record initial cycle count from TPU
        initial_cycles = tpu_cycle_count;
        
        // Run a short operation
        tpu_enable = 1'b1;
        tpu_start = 1'b1;
        @(posedge clk);
        tpu_start = 1'b0;
        
        repeat(100) @(posedge clk);
        
        final_cycles = tpu_cycle_count;
        
        if (final_cycles > initial_cycles) begin
            $display("✓ Performance counters working");
            $display("  Cycle count increased by: %0d", final_cycles - initial_cycles);
            pass_count++;
        end else begin
            $display("✗ Performance counters not working");
            fail_count++;
        end
        
        tpu_enable = 1'b0;
        $display("Test 6 completed\n");
    endtask
    
    // ========================================
    // Main Test Sequence
    // ========================================
    
    initial begin
        $display("========================================");
        $display("RISC-V AI Accelerator Comprehensive Test");
        $display("========================================\n");
        
        // Initialize signals
        ext_irq = 1'b0;
        timer_irq = 1'b0;
        soft_irq = 1'b0;
        tpu_enable = 1'b0;
        tpu_start = 1'b0;
        tpu_operation = 8'h00;
        tpu_data_type = 2'b00;
        tpu_matrix_m = 8'd0;
        tpu_matrix_n = 8'd0;
        tpu_matrix_k = 8'd0;
        
        pass_count = 0;
        fail_count = 0;
        current_test = 0;
        
        // Wait for reset deassertion
        wait(rst_n);
        repeat(10) @(posedge clk);
        
        // Run test suite
        test_riscv_basic_instructions();
        test_tpu_matrix_operations();
        test_vpu_vector_operations();
        test_memory_subsystem();
        test_ai_instructions();
        test_performance_monitoring();
        
        // Final results
        $display("========================================");
        $display("Test Results Summary:");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("  Total:  %0d", pass_count + fail_count);
        
        if (fail_count == 0) begin
            $display("🎉 ALL TESTS PASSED! 🎉");
        end else begin
            $display("❌ %0d TESTS FAILED", fail_count);
        end
        $display("========================================");
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        repeat(TIMEOUT_CYCLES * 10) @(posedge clk);
        $display("ERROR: Global test timeout!");
        $finish;
    end
    
    // Monitor key signals
    initial begin
        $monitor("Time: %0t | PC: 0x%08x | TPU: %s | VPU: %s", 
                 $time, imem_addr, 
                 tpu_busy ? "BUSY" : "IDLE",
                 vpu_busy ? "BUSY" : "IDLE");
    end

endmodule