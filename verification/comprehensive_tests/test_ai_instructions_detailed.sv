// Detailed AI Instructions Test Suite
// Tests custom AI instruction extensions (Task 2.3)
// Covers matrix operations, convolution, activation functions, etc.

`timescale 1ns/1ps

module test_ai_instructions_detailed;

    // ========================================
    // Test Parameters
    // ========================================
    
    parameter CLK_PERIOD = 10;
    parameter TIMEOUT_CYCLES = 5000;
    parameter DATA_WIDTH = 32;
    parameter MATRIX_DIM = 4;  // Small matrices for testing
    
    // AI Instruction Opcodes (Custom)
    parameter AI_MATMUL     = 7'b0001011;  // Custom opcode for matrix multiply
    parameter AI_CONV2D     = 7'b0001111;  // Custom opcode for convolution
    parameter AI_ACTIVATION = 7'b0010011;  // Custom opcode for activation functions
    parameter AI_POOLING    = 7'b0010111;  // Custom opcode for pooling
    parameter AI_BATCHNORM  = 7'b0011011;  // Custom opcode for batch normalization
    
    // Function codes for AI instructions
    parameter FUNCT3_MATMUL_INT8  = 3'b000;
    parameter FUNCT3_MATMUL_FP16  = 3'b001;
    parameter FUNCT3_MATMUL_FP32  = 3'b010;
    parameter FUNCT3_CONV2D_3X3   = 3'b000;
    parameter FUNCT3_CONV2D_5X5   = 3'b001;
    parameter FUNCT3_RELU         = 3'b000;
    parameter FUNCT3_SIGMOID      = 3'b001;
    parameter FUNCT3_TANH         = 3'b010;
    parameter FUNCT3_MAXPOOL      = 3'b000;
    parameter FUNCT3_AVGPOOL      = 3'b001;
    
    // ========================================
    // Signals
    // ========================================
    
    logic clk, rst_n;
    
    // RISC-V Core interface
    logic [63:0]    imem_addr, dmem_addr;
    logic [31:0]    imem_rdata;
    logic [63:0]    dmem_wdata, dmem_rdata;
    logic [7:0]     dmem_wmask;
    logic           imem_req, imem_ready;
    logic           dmem_req, dmem_we, dmem_ready;
    logic           ext_irq, timer_irq, soft_irq;
    
    // AI accelerator interface
    ai_accel_if ai_if();
    
    // Test control
    logic [31:0] test_counter;
    logic [31:0] pass_count, fail_count;
    
    // ========================================
    // Clock and Reset
    // ========================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 5);
        rst_n = 1;
    end
    
    // ========================================
    // Test Memory with AI Instructions
    // ========================================
    
    logic [31:0] instruction_memory [0:1023];
    logic [63:0] data_memory [0:1023];
    
    // Initialize test instructions with AI operations
    initial begin
        // Test 1: Matrix Multiplication (INT8)
        instruction_memory[0] = {7'b0000000, 5'd2, 5'd1, FUNCT3_MATMUL_INT8, 5'd3, AI_MATMUL};
        
        // Test 2: Matrix Multiplication (FP32)
        instruction_memory[1] = {7'b0000000, 5'd4, 5'd3, FUNCT3_MATMUL_FP32, 5'd5, AI_MATMUL};
        
        // Test 3: 2D Convolution (3x3 kernel)
        instruction_memory[2] = {7'b0000000, 5'd6, 5'd5, FUNCT3_CONV2D_3X3, 5'd7, AI_CONV2D};
        
        // Test 4: ReLU Activation
        instruction_memory[3] = {7'b0000000, 5'd0, 5'd7, FUNCT3_RELU, 5'd8, AI_ACTIVATION};
        
        // Test 5: Sigmoid Activation
        instruction_memory[4] = {7'b0000000, 5'd0, 5'd8, FUNCT3_SIGMOID, 5'd9, AI_ACTIVATION};
        
        // Test 6: Max Pooling
        instruction_memory[5] = {7'b0000000, 5'd0, 5'd9, FUNCT3_MAXPOOL, 5'd10, AI_POOLING};
        
        // Test 7: Batch Normalization
        instruction_memory[6] = {7'b0000000, 5'd11, 5'd10, 3'b000, 5'd12, AI_BATCHNORM};
        
        // Standard RISC-V instructions for setup
        instruction_memory[7] = 32'h00100093;   // addi x1, x0, 1
        instruction_memory[8] = 32'h00200113;   // addi x2, x0, 2
        instruction_memory[9] = 32'h00000013;   // nop
        
        // Fill remaining with NOPs
        for (int i = 10; i < 1024; i++) begin
            instruction_memory[i] = 32'h00000013; // NOP
        end
        
        // Initialize test data matrices
        // Matrix A (4x4) at address 0x1000
        data_memory[128] = 64'h0001000200030004;  // Elements [0,1,2,3]
        data_memory[129] = 64'h0005000600070008;  // Elements [4,5,6,7]
        data_memory[130] = 64'h000900010001000B;  // Elements [8,9,10,11]
        data_memory[131] = 64'h000C000D000E000F;  // Elements [12,13,14,15]
        
        // Matrix B (4x4) at address 0x1100
        data_memory[136] = 64'h0001000000000000;  // Identity matrix
        data_memory[137] = 64'h0000000100000000;
        data_memory[138] = 64'h0000000000010000;
        data_memory[139] = 64'h0000000000000001;
        
        // Convolution kernel (3x3) at address 0x1200
        data_memory[144] = 64'h00010002FFFF0000;  // [1, 2, -1, 0]
        data_memory[145] = 64'h0000FFFF00020001;  // [0, -1, 2, 1]
        data_memory[146] = 64'h0001000000000000;  // [1, 0, 0, 0] (padding)
        
        // Input feature map (8x8) at address 0x1300
        for (int i = 0; i < 16; i++) begin
            data_memory[152 + i] = 64'h0102030405060708 + (i << 8);
        end
    end
    
    // Memory interfaces
    always_comb begin
        imem_ready = 1'b1;
        imem_rdata = instruction_memory[imem_addr[11:2]];
    end
    
    always_ff @(posedge clk) begin
        dmem_ready <= 1'b1;
        if (dmem_req && !dmem_we) begin
            dmem_rdata <= data_memory[dmem_addr[11:3]];
        end else if (dmem_req && dmem_we) begin
            case (dmem_wmask)
                8'b1111_1111: data_memory[dmem_addr[11:3]] <= dmem_wdata;
                8'b0000_1111: data_memory[dmem_addr[11:3]][31:0] <= dmem_wdata[31:0];
                8'b0000_0011: data_memory[dmem_addr[11:3]][15:0] <= dmem_wdata[15:0];
                8'b0000_0001: data_memory[dmem_addr[11:3]][7:0] <= dmem_wdata[7:0];
                default: data_memory[dmem_addr[11:3]] <= dmem_wdata;
            endcase
        end
    end
    
    // ========================================
    // DUT Instantiation
    // ========================================
    
    // Connect AI interface
    assign ai_if.clk = clk;
    assign ai_if.rst_n = rst_n;
    
    // RISC-V Core with AI extensions
    riscv_core #(
        .XLEN(64),
        .VLEN(512)
    ) dut_core (
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
    
    // ========================================
    // Test Tasks
    // ========================================
    
    // Test matrix multiplication instruction
    task automatic test_matmul_instruction();
        logic [63:0] expected_result;
        logic [63:0] actual_result;
        
        $display("=== Testing Matrix Multiplication Instructions ===");
        
        // Wait for matrix multiplication instruction to execute
        wait(imem_addr == 64'h0000_0000); // First AI instruction
        repeat(10) @(posedge clk);
        
        // Check if AI interface is activated
        if (ai_if.req) begin
            $display("✓ AI interface activated for MATMUL");
            pass_count = pass_count + 1;
            
            // Check instruction encoding
            if (ai_if.addr[6:0] == AI_MATMUL) begin
                $display("✓ MATMUL opcode correctly decoded");
                pass_count = pass_count + 1;
            end else begin
                $display("✗ MATMUL opcode incorrectly decoded: 0x%02x", ai_if.addr[6:0]);
                fail_count++;
            end
            
            // Check function code
            if (ai_if.addr[14:12] == FUNCT3_MATMUL_INT8) begin
                $display("✓ MATMUL INT8 function code correct");
                pass_count++;
            end else begin
                $display("✗ MATMUL function code incorrect: 0x%01x", ai_if.addr[14:12]);
                fail_count++;
            end
            
        end else begin
            $display("✗ AI interface not activated for MATMUL");
            fail_count++;
        end
        
        $display("Matrix Multiplication test completed\n");
    endtask
    
    // Test convolution instruction
    task test_conv2d_instruction();
        $display("=== Testing 2D Convolution Instructions ===");
        
        // Wait for convolution instruction
        wait(imem_addr == 64'h0000_0008); // Third AI instruction
        repeat(10) @(posedge clk);
        
        if (ai_if.req && (ai_if.addr[6:0] == AI_CONV2D)) begin
            $display("✓ CONV2D instruction recognized");
            pass_count++;
            
            // Check kernel size
            if (ai_if.addr[14:12] == FUNCT3_CONV2D_3X3) begin
                $display("✓ 3x3 convolution kernel selected");
                pass_count++;
            end else begin
                $display("✗ Incorrect convolution kernel: 0x%01x", ai_if.addr[14:12]);
                fail_count++;
            end
            
        end else begin
            $display("✗ CONV2D instruction not recognized");
            fail_count++;
        end
        
        $display("2D Convolution test completed\n");
    endtask
    
    // Test activation function instructions
    task test_activation_instructions();
        $display("=== Testing Activation Function Instructions ===");
        
        // Test ReLU
        wait(imem_addr == 64'h0000_000C); // Fourth AI instruction
        repeat(5) @(posedge clk);
        
        if (ai_if.req && (ai_if.addr[6:0] == AI_ACTIVATION)) begin
            $display("✓ Activation instruction recognized");
            pass_count++;
            
            if (ai_if.addr[14:12] == FUNCT3_RELU) begin
                $display("✓ ReLU activation function selected");
                pass_count++;
            end else begin
                $display("✗ Incorrect activation function: 0x%01x", ai_if.addr[14:12]);
                fail_count++;
            end
        end else begin
            $display("✗ Activation instruction not recognized");
            fail_count++;
        end
        
        // Test Sigmoid
        wait(imem_addr == 64'h0000_0010); // Fifth AI instruction
        repeat(5) @(posedge clk);
        
        if (ai_if.req && (ai_if.addr[14:12] == FUNCT3_SIGMOID)) begin
            $display("✓ Sigmoid activation function selected");
            pass_count++;
        end else begin
            $display("✗ Sigmoid activation function not selected");
            fail_count++;
        end
        
        $display("Activation Functions test completed\n");
    endtask
    
    // Test pooling instructions
    task test_pooling_instructions();
        $display("=== Testing Pooling Instructions ===");
        
        // Wait for pooling instruction
        wait(imem_addr == 64'h0000_0014); // Sixth AI instruction
        repeat(10) @(posedge clk);
        
        if (ai_if.req && (ai_if.addr[6:0] == AI_POOLING)) begin
            $display("✓ Pooling instruction recognized");
            pass_count++;
            
            if (ai_if.addr[14:12] == FUNCT3_MAXPOOL) begin
                $display("✓ Max pooling operation selected");
                pass_count++;
            end else begin
                $display("✗ Incorrect pooling operation: 0x%01x", ai_if.addr[14:12]);
                fail_count++;
            end
            
        end else begin
            $display("✗ Pooling instruction not recognized");
            fail_count++;
        end
        
        $display("Pooling test completed\n");
    endtask
    
    // Test batch normalization instruction
    task test_batchnorm_instruction();
        $display("=== Testing Batch Normalization Instructions ===");
        
        // Wait for batch normalization instruction
        wait(imem_addr == 64'h0000_0018); // Seventh AI instruction
        repeat(10) @(posedge clk);
        
        if (ai_if.req && (ai_if.addr[6:0] == AI_BATCHNORM)) begin
            $display("✓ Batch normalization instruction recognized");
            pass_count++;
        end else begin
            $display("✗ Batch normalization instruction not recognized");
            fail_count++;
        end
        
        $display("Batch Normalization test completed\n");
    endtask
    
    // Test AI instruction performance
    task automatic test_ai_instruction_performance();
        logic [31:0] start_time, end_time, execution_time;
        
        $display("=== Testing AI Instruction Performance ===");
        
        start_time = $time;
        
        // Execute a series of AI instructions
        repeat(100) @(posedge clk);
        
        end_time = $time;
        execution_time = end_time - start_time;
        
        $display("AI instruction execution time: %0d ns", execution_time);
        
        if (execution_time < (TIMEOUT_CYCLES * CLK_PERIOD)) begin
            $display("✓ AI instructions execute within acceptable time");
            pass_count = pass_count + 1;
        end else begin
            $display("✗ AI instructions too slow");
            fail_count++;
        end
        
        $display("Performance test completed\n");
    endtask
    
    // Test error handling in AI instructions
    task test_ai_error_handling();
        $display("=== Testing AI Instruction Error Handling ===");
        
        // Test invalid opcode
        ai_if.addr = 32'h7F000000; // Invalid opcode
        ai_if.req = 1'b1;
        ai_if.we = 1'b1;
        
        @(posedge clk);
        
        if (ai_if.error) begin
            $display("✓ Error detected for invalid AI instruction");
            pass_count++;
        end else begin
            $display("✗ No error detected for invalid AI instruction");
            fail_count++;
        end
        
        ai_if.req = 1'b0;
        ai_if.we = 1'b0;
        
        $display("Error Handling test completed\n");
    endtask
    
    // ========================================
    // Main Test Sequence
    // ========================================
    
    initial begin
        $display("========================================");
        $display("AI Instructions Detailed Test Suite");
        $display("========================================\n");
        
        // Initialize
        ext_irq = 1'b0;
        timer_irq = 1'b0;
        soft_irq = 1'b0;
        pass_count = 0;
        fail_count = 0;
        
        // Wait for reset
        wait(rst_n);
        repeat(10) @(posedge clk);
        
        // Run tests
        fork
            begin
                test_matmul_instruction();
                test_conv2d_instruction();
                test_activation_instructions();
                test_pooling_instructions();
                test_batchnorm_instruction();
                test_ai_instruction_performance();
                test_ai_error_handling();
            end
            begin
                // Timeout watchdog
                repeat(TIMEOUT_CYCLES) @(posedge clk);
                $display("ERROR: Test timeout!");
            end
        join_any
        disable fork;
        
        // Results
        $display("========================================");
        $display("AI Instructions Test Results:");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("  Total:  %0d", pass_count + fail_count);
        
        if (fail_count == 0) begin
            $display("🎉 ALL AI INSTRUCTION TESTS PASSED! 🎉");
        end else begin
            $display("❌ %0d AI INSTRUCTION TESTS FAILED", fail_count);
        end
        $display("========================================");
        
        $finish;
    end
    
    // Signal monitoring
    always @(posedge clk) begin
        if (ai_if.req) begin
            $display("Time: %0t | AI Request: addr=0x%08x, data=0x%016x, we=%b", 
                     $time, ai_if.addr, ai_if.wdata, ai_if.we);
        end
    end

endmodule