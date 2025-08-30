// Test bench for RISC-V AI Instruction Extensions
// Tests matrix multiplication, convolution, activation functions, pooling, and batch normalization

`timescale 1ns / 1ps

module test_riscv_ai_instructions;

    // Test parameters
    parameter XLEN = 64;
    parameter DATA_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    // DUT signals
    logic                    clk;
    logic                    rst_n;
    
    // AI Unit interface
    logic                    ai_enable;
    logic [6:0]              ai_opcode;
    logic [2:0]              funct3;
    logic [6:0]              funct7;
    logic [4:0]              rs1, rs2, rs3, rd;
    logic [XLEN-1:0]         rs1_data, rs2_data, rs3_data;
    logic [XLEN-1:0]         ai_result;
    logic                    ai_ready;
    logic                    ai_valid;
    logic [4:0]              ai_flags;
    
    // Memory interface
    logic [XLEN-1:0]         ai_mem_addr;
    logic [XLEN-1:0]         ai_mem_wdata;
    logic [7:0]              ai_mem_wmask;
    logic                    ai_mem_req;
    logic                    ai_mem_we;
    logic [XLEN-1:0]         ai_mem_rdata;
    logic                    ai_mem_ready;
    
    // Test memory
    logic [DATA_WIDTH-1:0]   test_memory [0:4095];
    
    // Test constants
    localparam [6:0] OP_AI_CUSTOM = 7'b0001011;
    localparam [6:0] AI_MATMUL     = 7'b0000001;
    localparam [6:0] AI_CONV2D     = 7'b0000010;
    localparam [6:0] AI_RELU       = 7'b0000100;
    localparam [6:0] AI_SIGMOID    = 7'b0000101;
    localparam [6:0] AI_TANH       = 7'b0000110;
    localparam [6:0] AI_MAXPOOL    = 7'b0001000;
    localparam [6:0] AI_AVGPOOL    = 7'b0001001;
    localparam [6:0] AI_BATCHNORM  = 7'b0001010;
    
    localparam [2:0] AI_FP32 = 3'b101;
    localparam [2:0] AI_INT32 = 3'b010;

    // DUT instantiation
    riscv_ai_unit #(
        .XLEN(XLEN),
        .AI_DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .ai_enable(ai_enable),
        .ai_opcode(ai_opcode),
        .funct3(funct3),
        .funct7(funct7),
        .rs1(rs1),
        .rs2(rs2),
        .rs3(rs3),
        .rd(rd),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .rs3_data(rs3_data),
        .ai_result(ai_result),
        .ai_mem_addr(ai_mem_addr),
        .ai_mem_wdata(ai_mem_wdata),
        .ai_mem_wmask(ai_mem_wmask),
        .ai_mem_req(ai_mem_req),
        .ai_mem_we(ai_mem_we),
        .ai_mem_rdata(ai_mem_rdata),
        .ai_mem_ready(ai_mem_ready),
        .ai_ready(ai_ready),
        .ai_valid(ai_valid),
        .ai_flags(ai_flags)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Memory model
    always_ff @(posedge clk) begin
        if (ai_mem_req && ai_mem_ready) begin
            if (ai_mem_we) begin
                // Write to memory
                case (ai_mem_wmask)
                    8'b1111: test_memory[ai_mem_addr[15:2]] <= ai_mem_wdata[31:0];
                    8'b11111111: test_memory[ai_mem_addr[15:2]] <= ai_mem_wdata[31:0];
                    default: test_memory[ai_mem_addr[15:2]] <= ai_mem_wdata[31:0];
                endcase
            end else begin
                // Read from memory
                ai_mem_rdata <= {32'b0, test_memory[ai_mem_addr[15:2]]};
            end
        end
    end
    
    assign ai_mem_ready = ai_mem_req; // Always ready for simplicity

    // Test tasks
    task reset_dut();
        rst_n = 0;
        ai_enable = 0;
        ai_opcode = 0;
        funct3 = 0;
        funct7 = 0;
        rs1 = 0;
        rs2 = 0;
        rs3 = 0;
        rd = 0;
        rs1_data = 0;
        rs2_data = 0;
        rs3_data = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    task wait_for_completion();
        while (!ai_valid) begin
            @(posedge clk);
        end
        @(posedge clk);
    endtask

    task setup_test_data();
        // Initialize test memory with known patterns
        for (int i = 0; i < 4096; i++) begin
            test_memory[i] = i % 256; // Simple pattern
        end
        
        // Setup specific test matrices
        // 2x2 matrix A at address 0x0000
        test_memory[0] = 32'h3F800000; // 1.0 in FP32
        test_memory[1] = 32'h40000000; // 2.0 in FP32
        test_memory[2] = 32'h40400000; // 3.0 in FP32
        test_memory[3] = 32'h40800000; // 4.0 in FP32
        
        // 2x2 matrix B at address 0x0010
        test_memory[4] = 32'h40A00000; // 5.0 in FP32
        test_memory[5] = 32'h40C00000; // 6.0 in FP32
        test_memory[6] = 32'h40E00000; // 7.0 in FP32
        test_memory[7] = 32'h41000000; // 8.0 in FP32
        
        // Test activation function inputs
        test_memory[16] = 32'hC0000000; // -2.0 (should be 0 after ReLU)
        test_memory[17] = 32'h40000000; //  2.0 (should remain 2.0 after ReLU)
        test_memory[18] = 32'h00000000; //  0.0 (should remain 0.0 after ReLU)
        test_memory[19] = 32'h3F800000; //  1.0 (should remain 1.0 after ReLU)
    endtask

    // Test ReLU activation function
    task test_relu();
        $display("Testing ReLU activation function...");
        
        // Test negative input (should become 0)
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_RELU;
        funct3 = AI_FP32;
        rs1_data = 64'hC0000000; // -2.0 in FP32
        
        @(posedge clk);
        wait_for_completion();
        
        if (ai_result[31:0] == 32'h00000000) begin
            $display("✓ ReLU(-2.0) = 0.0 - PASS");
        end else begin
            $display("✗ ReLU(-2.0) = %h, expected 0.0 - FAIL", ai_result[31:0]);
        end
        
        // Test positive input (should remain unchanged)
        rs1_data = 64'h40000000; // 2.0 in FP32
        ai_enable = 0;
        @(posedge clk);
        ai_enable = 1;
        @(posedge clk);
        wait_for_completion();
        
        if (ai_result[31:0] == 32'h40000000) begin
            $display("✓ ReLU(2.0) = 2.0 - PASS");
        end else begin
            $display("✗ ReLU(2.0) = %h, expected 2.0 - FAIL", ai_result[31:0]);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Sigmoid activation function
    task test_sigmoid();
        $display("Testing Sigmoid activation function...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_SIGMOID;
        funct3 = AI_FP32;
        rs1_data = 64'h00000000; // 0.0 in FP32
        
        @(posedge clk);
        wait_for_completion();
        
        // Sigmoid(0) should be approximately 0.5
        if (ai_result[31:24] == 8'h3E || ai_result[31:24] == 8'h3F) begin // Rough check for ~0.5
            $display("✓ Sigmoid(0.0) ≈ 0.5 - PASS");
        end else begin
            $display("✗ Sigmoid(0.0) = %h, expected ≈0.5 - FAIL", ai_result[31:0]);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Tanh activation function
    task test_tanh();
        $display("Testing Tanh activation function...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_TANH;
        funct3 = AI_FP32;
        rs1_data = 64'h00000000; // 0.0 in FP32
        
        @(posedge clk);
        wait_for_completion();
        
        // Tanh(0) should be 0.0
        if (ai_result[31:0] == 32'h00000000) begin
            $display("✓ Tanh(0.0) = 0.0 - PASS");
        end else begin
            $display("✗ Tanh(0.0) = %h, expected 0.0 - FAIL", ai_result[31:0]);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Matrix Multiplication
    task test_matmul();
        $display("Testing Matrix Multiplication...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_MATMUL;
        funct3 = AI_FP32;
        rs1_data = 64'h0000; // Matrix A address
        rs2_data = 64'h0010; // Matrix B address
        rs3_data = 64'h00020202; // Dimensions: M=2, N=2, K=2
        
        @(posedge clk);
        wait_for_completion();
        
        // Check if operation completed successfully
        if (ai_valid && !ai_flags[4]) begin // No memory fault
            $display("✓ Matrix multiplication completed - PASS");
        end else begin
            $display("✗ Matrix multiplication failed, flags: %b - FAIL", ai_flags);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test 2D Convolution
    task test_conv2d();
        $display("Testing 2D Convolution...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_CONV2D;
        funct3 = AI_FP32;
        rs1_data = 64'h0100; // Input tensor address
        rs2_data = 64'h0200; // Kernel address
        rs3_data = 64'h04043311; // Parameters: in_h=4, in_w=4, ker_h=3, ker_w=3, stride=1, pad=1
        
        @(posedge clk);
        wait_for_completion();
        
        // Check if operation completed successfully
        if (ai_valid && !ai_flags[4]) begin // No memory fault
            $display("✓ 2D Convolution completed - PASS");
        end else begin
            $display("✗ 2D Convolution failed, flags: %b - FAIL", ai_flags);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Max Pooling
    task test_maxpool();
        $display("Testing Max Pooling...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_MAXPOOL;
        funct3 = AI_FP32;
        rs1_data = 64'h0300; // Input tensor address
        rs2_data = 64'h04042211; // Parameters: in_h=4, in_w=4, pool_h=2, pool_w=2, stride_h=1, stride_w=1
        
        @(posedge clk);
        wait_for_completion();
        
        // Check if operation completed successfully
        if (ai_valid && !ai_flags[4]) begin // No memory fault
            $display("✓ Max Pooling completed - PASS");
        end else begin
            $display("✗ Max Pooling failed, flags: %b - FAIL", ai_flags);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Average Pooling
    task test_avgpool();
        $display("Testing Average Pooling...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_AVGPOOL;
        funct3 = AI_FP32;
        rs1_data = 64'h0300; // Input tensor address
        rs2_data = 64'h04042211; // Parameters: in_h=4, in_w=4, pool_h=2, pool_w=2, stride_h=1, stride_w=1
        
        @(posedge clk);
        wait_for_completion();
        
        // Check if operation completed successfully
        if (ai_valid && !ai_flags[4]) begin // No memory fault
            $display("✓ Average Pooling completed - PASS");
        end else begin
            $display("✗ Average Pooling failed, flags: %b - FAIL", ai_flags);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test Batch Normalization
    task test_batchnorm();
        $display("Testing Batch Normalization...");
        
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = AI_BATCHNORM;
        funct3 = AI_FP32;
        rs1_data = 64'h0400; // Input tensor address
        rs2_data = 64'h0500; // Scale parameters address
        rs3_data = 64'h0600; // Bias parameters address
        
        @(posedge clk);
        wait_for_completion();
        
        // Check if operation completed successfully
        if (ai_valid && !ai_flags[4]) begin // No memory fault
            $display("✓ Batch Normalization completed - PASS");
        end else begin
            $display("✗ Batch Normalization failed, flags: %b - FAIL", ai_flags);
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Test error conditions
    task test_error_conditions();
        $display("Testing Error Conditions...");
        
        // Test invalid instruction
        ai_enable = 1;
        ai_opcode = OP_AI_CUSTOM;
        funct7 = 7'b1111111; // Invalid funct7
        funct3 = AI_FP32;
        
        @(posedge clk);
        wait_for_completion();
        
        // Should set invalid operation flag
        if (ai_flags[2]) begin // Invalid operation flag
            $display("✓ Invalid instruction detected - PASS");
        end else begin
            $display("✗ Invalid instruction not detected - FAIL");
        end
        
        ai_enable = 0;
        @(posedge clk);
    endtask

    // Main test sequence
    initial begin
        $display("Starting RISC-V AI Instruction Extension Tests");
        $display("================================================");
        
        reset_dut();
        setup_test_data();
        
        // Run individual tests
        test_relu();
        test_sigmoid();
        test_tanh();
        test_matmul();
        test_conv2d();
        test_maxpool();
        test_avgpool();
        test_batchnorm();
        test_error_conditions();
        
        $display("================================================");
        $display("AI Instruction Extension Tests Completed");
        
        #100;
        $finish;
    end

    // Monitor signals for debugging
    initial begin
        $monitor("Time: %0t | State: ai_enable=%b, ai_valid=%b, ai_ready=%b, ai_flags=%b", 
                 $time, ai_enable, ai_valid, ai_ready, ai_flags);
    end

endmodule