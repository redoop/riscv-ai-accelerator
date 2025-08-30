// Unit test for RISC-V ALU
// Tests all arithmetic and logical operations

`timescale 1ns/1ps

module test_riscv_alu;

    parameter XLEN = 64;
    
    // Testbench signals
    logic [XLEN-1:0]    a, b;
    logic [3:0]         alu_op;
    logic [XLEN-1:0]    result;
    logic               zero;
    logic               overflow;
    
    // Test vectors
    logic [XLEN-1:0]    expected_result;
    logic               expected_zero;
    logic               expected_overflow;
    
    // Test counters
    int                 test_count = 0;
    int                 pass_count = 0;
    int                 fail_count = 0;

    // DUT instantiation
    riscv_alu #(.XLEN(XLEN)) dut (
        .a(a),
        .b(b),
        .alu_op(alu_op),
        .result(result),
        .zero(zero),
        .overflow(overflow)
    );

    // Test task
    task automatic test_operation(
        input [XLEN-1:0] test_a,
        input [XLEN-1:0] test_b,
        input [3:0] test_op,
        input [XLEN-1:0] exp_result,
        input exp_zero,
        input exp_overflow,
        input string op_name
    );
        a = test_a;
        b = test_b;
        alu_op = test_op;
        expected_result = exp_result;
        expected_zero = exp_zero;
        expected_overflow = exp_overflow;
        
        #1; // Wait for combinational logic
        
        test_count++;
        
        if (result === expected_result && zero === expected_zero && overflow === expected_overflow) begin
            $display("PASS: %s - a=%h, b=%h, result=%h, zero=%b, overflow=%b", 
                     op_name, test_a, test_b, result, zero, overflow);
            pass_count++;
        end else begin
            $display("FAIL: %s - a=%h, b=%h", op_name, test_a, test_b);
            $display("      Expected: result=%h, zero=%b, overflow=%b", 
                     expected_result, expected_zero, expected_overflow);
            $display("      Got:      result=%h, zero=%b, overflow=%b", 
                     result, zero, overflow);
            fail_count++;
        end
    endtask

    initial begin
        $display("Starting RISC-V ALU Unit Tests");
        $display("==============================");
        
        // Test ADD operation
        test_operation(64'h0000_0000_0000_0005, 64'h0000_0000_0000_0003, 4'b0000, 
                      64'h0000_0000_0000_0008, 1'b0, 1'b0, "ADD");
        test_operation(64'h7FFF_FFFF_FFFF_FFFF, 64'h0000_0000_0000_0001, 4'b0000, 
                      64'h8000_0000_0000_0000, 1'b0, 1'b1, "ADD_OVERFLOW");
        test_operation(64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 4'b0000, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "ADD_ZERO");
        
        // Test SUB operation
        test_operation(64'h0000_0000_0000_0008, 64'h0000_0000_0000_0003, 4'b0001, 
                      64'h0000_0000_0000_0005, 1'b0, 1'b0, "SUB");
        test_operation(64'h8000_0000_0000_0000, 64'h0000_0000_0000_0001, 4'b0001, 
                      64'h7FFF_FFFF_FFFF_FFFF, 1'b0, 1'b1, "SUB_OVERFLOW");
        test_operation(64'h0000_0000_0000_0005, 64'h0000_0000_0000_0005, 4'b0001, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "SUB_ZERO");
        
        // Test SLL operation
        test_operation(64'h0000_0000_0000_0001, 64'h0000_0000_0000_0004, 4'b0010, 
                      64'h0000_0000_0000_0010, 1'b0, 1'b0, "SLL");
        test_operation(64'h1234_5678_9ABC_DEF0, 64'h0000_0000_0000_0008, 4'b0010, 
                      64'h3456_789A_BCDE_F000, 1'b0, 1'b0, "SLL_LARGE");
        
        // Test SLT operation
        test_operation(64'h0000_0000_0000_0003, 64'h0000_0000_0000_0005, 4'b0011, 
                      64'h0000_0000_0000_0001, 1'b0, 1'b0, "SLT_TRUE");
        test_operation(64'h0000_0000_0000_0005, 64'h0000_0000_0000_0003, 4'b0011, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "SLT_FALSE");
        test_operation(64'hFFFF_FFFF_FFFF_FFFF, 64'h0000_0000_0000_0001, 4'b0011, 
                      64'h0000_0000_0000_0001, 1'b0, 1'b0, "SLT_SIGNED");
        
        // Test SLTU operation
        test_operation(64'h0000_0000_0000_0003, 64'h0000_0000_0000_0005, 4'b0100, 
                      64'h0000_0000_0000_0001, 1'b0, 1'b0, "SLTU_TRUE");
        test_operation(64'hFFFF_FFFF_FFFF_FFFF, 64'h0000_0000_0000_0001, 4'b0100, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "SLTU_UNSIGNED");
        
        // Test XOR operation
        test_operation(64'h1234_5678_9ABC_DEF0, 64'hFEDC_BA98_7654_3210, 4'b0101, 
                      64'hECE8_ECE0_ECE8_ECE0, 1'b0, 1'b0, "XOR");
        test_operation(64'h1234_5678_9ABC_DEF0, 64'h1234_5678_9ABC_DEF0, 4'b0101, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "XOR_ZERO");
        
        // Test SRL operation
        test_operation(64'h8000_0000_0000_0000, 64'h0000_0000_0000_0004, 4'b0110, 
                      64'h0800_0000_0000_0000, 1'b0, 1'b0, "SRL");
        test_operation(64'h1234_5678_9ABC_DEF0, 64'h0000_0000_0000_0008, 4'b0110, 
                      64'h0012_3456_789A_BCDE, 1'b0, 1'b0, "SRL_LARGE");
        
        // Test SRA operation
        test_operation(64'h8000_0000_0000_0000, 64'h0000_0000_0000_0004, 4'b0111, 
                      64'hF800_0000_0000_0000, 1'b0, 1'b0, "SRA_NEGATIVE");
        test_operation(64'h1234_5678_9ABC_DEF0, 64'h0000_0000_0000_0008, 4'b0111, 
                      64'h0012_3456_789A_BCDE, 1'b0, 1'b0, "SRA_POSITIVE");
        
        // Test OR operation
        test_operation(64'h1234_5678_0000_0000, 64'h0000_0000_9ABC_DEF0, 4'b1000, 
                      64'h1234_5678_9ABC_DEF0, 1'b0, 1'b0, "OR");
        test_operation(64'h0000_0000_0000_0000, 64'h0000_0000_0000_0000, 4'b1000, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "OR_ZERO");
        
        // Test AND operation
        test_operation(64'h1234_5678_9ABC_DEF0, 64'hFFFF_0000_FFFF_0000, 4'b1001, 
                      64'h1234_0000_9ABC_0000, 1'b0, 1'b0, "AND");
        test_operation(64'h1234_5678_9ABC_DEF0, 64'h0000_0000_0000_0000, 4'b1001, 
                      64'h0000_0000_0000_0000, 1'b1, 1'b0, "AND_ZERO");
        
        // Test LUI operation
        test_operation(64'h0000_0000_0000_0000, 64'h1234_5000_0000_0000, 4'b1010, 
                      64'h1234_5000_0000_0000, 1'b0, 1'b0, "LUI");
        
        // Test AUIPC operation (PC + immediate)
        test_operation(64'h0000_0000_1000_0000, 64'h0000_0000_0000_1000, 4'b1011, 
                      64'h0000_0000_1000_1000, 1'b0, 1'b0, "AUIPC");
        
        // Display test results
        $display("\n==============================");
        $display("ALU Test Results:");
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