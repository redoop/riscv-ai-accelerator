// Integration test for AI instructions with RISC-V core
// Tests AI instructions through the complete processor pipeline

`timescale 1ns / 1ps

module test_ai_integration;

    // Test parameters
    parameter XLEN = 64;
    parameter CLK_PERIOD = 10;
    
    // Core interface signals
    logic                    clk;
    logic                    rst_n;
    
    // Instruction memory interface
    logic [XLEN-1:0]         imem_addr;
    logic                    imem_req;
    logic [31:0]             imem_rdata;
    logic                    imem_ready;
    
    // Data memory interface
    logic [XLEN-1:0]         dmem_addr;
    logic [XLEN-1:0]         dmem_wdata;
    logic [7:0]              dmem_wmask;
    logic                    dmem_req;
    logic                    dmem_we;
    logic [XLEN-1:0]         dmem_rdata;
    logic                    dmem_ready;
    
    // AI accelerator interface (placeholder)
    ai_accel_if              ai_if();
    
    // Interrupt interface
    logic                    ext_irq = 0;
    logic                    timer_irq = 0;
    logic                    soft_irq = 0;
    
    // Test memory arrays
    logic [31:0]             instruction_memory [0:1023];
    logic [31:0]             data_memory [0:4095];
    
    // Test instruction encodings
    // AI instruction format: funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    localparam [6:0] OP_AI_CUSTOM = 7'b0001011;
    localparam [6:0] AI_RELU      = 7'b0000100;
    localparam [6:0] AI_SIGMOID   = 7'b0000101;
    localparam [6:0] AI_TANH      = 7'b0000110;
    localparam [6:0] AI_MATMUL    = 7'b0000001;
    localparam [2:0] AI_FP32      = 3'b101;

    // DUT instantiation
    riscv_core #(
        .XLEN(XLEN),
        .VLEN(512)
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
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Instruction memory model
    always_comb begin
        if (imem_req && (imem_addr[31:2] < 1024)) begin
            imem_rdata = instruction_memory[imem_addr[31:2]];
            imem_ready = 1'b1;
        end else begin
            imem_rdata = 32'h00000013; // NOP
            imem_ready = 1'b1;
        end
    end

    // Data memory model
    always_ff @(posedge clk) begin
        if (dmem_req && dmem_ready) begin
            if (dmem_we) begin
                // Write to memory
                case (dmem_wmask)
                    8'b0001: data_memory[dmem_addr[31:2]][7:0] <= dmem_wdata[7:0];
                    8'b0011: data_memory[dmem_addr[31:2]][15:0] <= dmem_wdata[15:0];
                    8'b1111: data_memory[dmem_addr[31:2]] <= dmem_wdata[31:0];
                    8'b11111111: data_memory[dmem_addr[31:2]] <= dmem_wdata[31:0];
                    default: data_memory[dmem_addr[31:2]] <= dmem_wdata[31:0];
                endcase
            end
        end
    end
    
    always_comb begin
        if (dmem_req && !dmem_we && (dmem_addr[31:2] < 4096)) begin
            dmem_rdata = {32'b0, data_memory[dmem_addr[31:2]]};
            dmem_ready = 1'b1;
        end else if (dmem_req && dmem_we) begin
            dmem_rdata = 64'b0;
            dmem_ready = 1'b1;
        end else begin
            dmem_rdata = 64'b0;
            dmem_ready = 1'b0;
        end
    end

    // Helper function to create AI instruction
    function [31:0] create_ai_instruction(
        input [6:0] funct7,
        input [4:0] rs2,
        input [4:0] rs1,
        input [2:0] funct3,
        input [4:0] rd
    );
        return {funct7, rs2, rs1, funct3, rd, OP_AI_CUSTOM};
    endfunction

    // Helper function to create immediate instruction
    function [31:0] create_imm_instruction(
        input [11:0] imm,
        input [4:0] rs1,
        input [2:0] funct3,
        input [4:0] rd,
        input [6:0] opcode
    );
        return {imm, rs1, funct3, rd, opcode};
    endfunction

    // Test tasks
    task reset_system();
        rst_n = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
    endtask

    task load_test_program();
        // Initialize instruction memory
        for (int i = 0; i < 1024; i++) begin
            instruction_memory[i] = 32'h00000013; // NOP
        end
        
        // Initialize data memory with test values
        for (int i = 0; i < 4096; i++) begin
            data_memory[i] = 32'h00000000;
        end
        
        // Load test data for AI operations
        data_memory[0] = 32'hC0000000; // -2.0 in FP32 (for ReLU test)
        data_memory[1] = 32'h40000000; //  2.0 in FP32
        data_memory[2] = 32'h00000000; //  0.0 in FP32
        data_memory[3] = 32'h3F800000; //  1.0 in FP32
        
        // Test program: AI instruction sequence
        int pc = 0;
        
        // Load immediate values into registers
        instruction_memory[pc++] = create_imm_instruction(12'h000, 5'b00000, 3'b000, 5'b00001, 7'b0010011); // addi x1, x0, 0 (load address 0)
        instruction_memory[pc++] = create_imm_instruction(12'h004, 5'b00000, 3'b000, 5'b00010, 7'b0010011); // addi x2, x0, 4 (load address 4)
        
        // Load test value from memory
        instruction_memory[pc++] = {12'h000, 5'b00001, 3'b010, 5'b00011, 7'b0000011}; // lw x3, 0(x1) - load -2.0
        
        // Test ReLU: ai.relu x4, x3
        instruction_memory[pc++] = create_ai_instruction(AI_RELU, 5'b00000, 5'b00011, AI_FP32, 5'b00100);
        
        // Store result back to memory
        instruction_memory[pc++] = {7'b0000000, 5'b00100, 5'b00010, 3'b010, 5'b00000, 7'b0100011}; // sw x4, 0(x2)
        
        // Load positive test value
        instruction_memory[pc++] = {12'h004, 5'b00001, 3'b010, 5'b00101, 7'b0000011}; // lw x5, 4(x1) - load 2.0
        
        // Test ReLU on positive value: ai.relu x6, x5
        instruction_memory[pc++] = create_ai_instruction(AI_RELU, 5'b00000, 5'b00101, AI_FP32, 5'b00110);
        
        // Store result
        instruction_memory[pc++] = {7'b0000000, 5'b00110, 5'b00010, 3'b010, 5'b00100, 7'b0100011}; // sw x6, 4(x2)
        
        // Test Sigmoid: ai.sigmoid x7, x5
        instruction_memory[pc++] = create_ai_instruction(AI_SIGMOID, 5'b00000, 5'b00101, AI_FP32, 5'b00111);
        
        // Store sigmoid result
        instruction_memory[pc++] = {7'b0000000, 5'b00111, 5'b00010, 3'b010, 5'b01000, 7'b0100011}; // sw x7, 8(x2)
        
        // Test Tanh: ai.tanh x8, x3
        instruction_memory[pc++] = create_ai_instruction(AI_TANH, 5'b00000, 5'b00011, AI_FP32, 5'b01000);
        
        // Store tanh result
        instruction_memory[pc++] = {7'b0000000, 5'b01000, 5'b00010, 3'b010, 5'b01100, 7'b0100011}; // sw x8, 12(x2)
        
        // Infinite loop to end program
        instruction_memory[pc++] = {12'h000, 5'b00000, 3'b000, 5'b00000, 7'b1100011}; // beq x0, x0, 0 (infinite loop)
    endtask

    task run_test_program();
        int cycle_count = 0;
        int max_cycles = 1000;
        
        $display("Running AI instruction test program...");
        
        while (cycle_count < max_cycles) begin
            @(posedge clk);
            cycle_count++;
            
            // Check if we've reached the infinite loop (PC should stop incrementing)
            if (cycle_count > 100 && imem_addr == (13 * 4)) begin // PC at infinite loop
                break;
            end
        end
        
        $display("Program completed after %0d cycles", cycle_count);
    endtask

    task verify_results();
        $display("Verifying AI instruction results...");
        
        // Check ReLU(-2.0) result at address 4
        if (data_memory[1] == 32'h00000000) begin
            $display("✓ ReLU(-2.0) = 0.0 - PASS");
        end else begin
            $display("✗ ReLU(-2.0) = %h, expected 0.0 - FAIL", data_memory[1]);
        end
        
        // Check ReLU(2.0) result at address 8
        if (data_memory[2] == 32'h40000000) begin
            $display("✓ ReLU(2.0) = 2.0 - PASS");
        end else begin
            $display("✗ ReLU(2.0) = %h, expected 2.0 - FAIL", data_memory[2]);
        end
        
        // Check Sigmoid result at address 12
        $display("ℹ Sigmoid(2.0) = %h", data_memory[3]);
        
        // Check Tanh result at address 16
        $display("ℹ Tanh(-2.0) = %h", data_memory[4]);
    endtask

    // Performance monitoring
    int instruction_count = 0;
    int ai_instruction_count = 0;
    
    always @(posedge clk) begin
        if (rst_n && imem_req && imem_ready) begin
            instruction_count++;
            
            // Check if it's an AI instruction
            if (imem_rdata[6:0] == OP_AI_CUSTOM) begin
                ai_instruction_count++;
                $display("AI Instruction executed: funct7=%b, funct3=%b at cycle %0d", 
                         imem_rdata[31:25], imem_rdata[14:12], instruction_count);
            end
        end
    end

    // Main test sequence
    initial begin
        $display("Starting RISC-V AI Integration Tests");
        $display("====================================");
        
        reset_system();
        load_test_program();
        run_test_program();
        verify_results();
        
        $display("====================================");
        $display("Total instructions executed: %0d", instruction_count);
        $display("AI instructions executed: %0d", ai_instruction_count);
        $display("AI Integration Tests Completed");
        
        #100;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #50000; // 50us timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule