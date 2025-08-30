// Unit test for RISC-V Control Unit
// Tests control signal generation for all instruction types

`timescale 1ns/1ps

module test_riscv_control_unit;

    // Testbench signals
    logic [6:0]  opcode;
    logic [2:0]  funct3;
    logic [6:0]  funct7;
    logic        reg_write;
    logic        mem_read;
    logic        mem_write;
    logic        branch;
    logic        jump;
    logic        alu_src;
    logic [3:0]  alu_op;
    logic [1:0]  wb_sel;
    
    // Test counters
    int          test_count = 0;
    int          pass_count = 0;
    int          fail_count = 0;

    // DUT instantiation
    riscv_control_unit dut (
        .opcode(opcode),
        .funct3(funct3),
        .funct7(funct7),
        .reg_write(reg_write),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .branch(branch),
        .jump(jump),
        .alu_src(alu_src),
        .alu_op(alu_op),
        .wb_sel(wb_sel)
    );

    // Test task
    task automatic test_instruction(
        input [6:0] test_opcode,
        input [2:0] test_funct3,
        input [6:0] test_funct7,
        input exp_reg_write,
        input exp_mem_read,
        input exp_mem_write,
        input exp_branch,
        input exp_jump,
        input exp_alu_src,
        input [3:0] exp_alu_op,
        input [1:0] exp_wb_sel,
        input string instr_name
    );
        opcode = test_opcode;
        funct3 = test_funct3;
        funct7 = test_funct7;
        
        #1; // Wait for combinational logic
        
        test_count++;
        
        if (reg_write === exp_reg_write && 
            mem_read === exp_mem_read &&
            mem_write === exp_mem_write &&
            branch === exp_branch &&
            jump === exp_jump &&
            alu_src === exp_alu_src &&
            alu_op === exp_alu_op &&
            wb_sel === exp_wb_sel) begin
            $display("PASS: %s", instr_name);
            pass_count++;
        end else begin
            $display("FAIL: %s", instr_name);
            $display("      Expected: reg_write=%b, mem_read=%b, mem_write=%b, branch=%b, jump=%b, alu_src=%b, alu_op=%b, wb_sel=%b",
                     exp_reg_write, exp_mem_read, exp_mem_write, exp_branch, exp_jump, exp_alu_src, exp_alu_op, exp_wb_sel);
            $display("      Got:      reg_write=%b, mem_read=%b, mem_write=%b, branch=%b, jump=%b, alu_src=%b, alu_op=%b, wb_sel=%b",
                     reg_write, mem_read, mem_write, branch, jump, alu_src, alu_op, wb_sel);
            fail_count++;
        end
    endtask

    initial begin
        $display("Starting RISC-V Control Unit Tests");
        $display("==================================");
        
        // Test LUI instruction
        test_instruction(7'b0110111, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b1010, 2'b00, "LUI");
        
        // Test AUIPC instruction
        test_instruction(7'b0010111, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b1011, 2'b00, "AUIPC");
        
        // Test JAL instruction
        test_instruction(7'b1101111, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 4'b0000, 2'b10, "JAL");
        
        // Test JALR instruction
        test_instruction(7'b1100111, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b1, 1'b1, 4'b0000, 2'b10, "JALR");
        
        // Test BEQ instruction
        test_instruction(7'b1100011, 3'b000, 7'b0000000, 
                        1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 4'b0000, 2'b00, "BEQ");
        
        // Test BNE instruction
        test_instruction(7'b1100011, 3'b001, 7'b0000000, 
                        1'b0, 1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 4'b0000, 2'b00, "BNE");
        
        // Test LB instruction
        test_instruction(7'b0000011, 3'b000, 7'b0000000, 
                        1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b01, "LB");
        
        // Test LW instruction
        test_instruction(7'b0000011, 3'b010, 7'b0000000, 
                        1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b01, "LW");
        
        // Test LD instruction
        test_instruction(7'b0000011, 3'b011, 7'b0000000, 
                        1'b1, 1'b1, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b01, "LD");
        
        // Test SB instruction
        test_instruction(7'b0100011, 3'b000, 7'b0000000, 
                        1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b00, "SB");
        
        // Test SW instruction
        test_instruction(7'b0100011, 3'b010, 7'b0000000, 
                        1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b00, "SW");
        
        // Test SD instruction
        test_instruction(7'b0100011, 3'b011, 7'b0000000, 
                        1'b0, 1'b0, 1'b1, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b00, "SD");
        
        // Test ADDI instruction
        test_instruction(7'b0010011, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0000, 2'b00, "ADDI");
        
        // Test SLTI instruction
        test_instruction(7'b0010011, 3'b010, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0011, 2'b00, "SLTI");
        
        // Test XORI instruction
        test_instruction(7'b0010011, 3'b100, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0101, 2'b00, "XORI");
        
        // Test SLLI instruction
        test_instruction(7'b0010011, 3'b001, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0010, 2'b00, "SLLI");
        
        // Test SRLI instruction
        test_instruction(7'b0010011, 3'b101, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0110, 2'b00, "SRLI");
        
        // Test SRAI instruction
        test_instruction(7'b0010011, 3'b101, 7'b0100000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b1, 4'b0111, 2'b00, "SRAI");
        
        // Test ADD instruction
        test_instruction(7'b0110011, 3'b000, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b0000, 2'b00, "ADD");
        
        // Test SUB instruction
        test_instruction(7'b0110011, 3'b000, 7'b0100000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b0001, 2'b00, "SUB");
        
        // Test SLL instruction
        test_instruction(7'b0110011, 3'b001, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b0010, 2'b00, "SLL");
        
        // Test SLT instruction
        test_instruction(7'b0110011, 3'b010, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b0011, 2'b00, "SLT");
        
        // Test XOR instruction
        test_instruction(7'b0110011, 3'b100, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b0101, 2'b00, "XOR");
        
        // Test OR instruction
        test_instruction(7'b0110011, 3'b110, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b1000, 2'b00, "OR");
        
        // Test AND instruction
        test_instruction(7'b0110011, 3'b111, 7'b0000000, 
                        1'b1, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0, 4'b1001, 2'b00, "AND");
        
        // Display test results
        $display("\n==================================");
        $display("Control Unit Test Results:");
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

endmodule