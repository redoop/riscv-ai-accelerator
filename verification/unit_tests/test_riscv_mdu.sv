// Test bench for RISC-V Multiply/Divide Unit (MDU)
// Tests RV64M extension functionality

`timescale 1ns / 1ps

module test_riscv_mdu;

    // Parameters
    parameter XLEN = 64;
    parameter CLK_PERIOD = 10;

    // Signals
    logic                clk;
    logic                rst_n;
    logic                mdu_enable;
    logic [2:0]          funct3;
    logic                is_32bit;
    logic [XLEN-1:0]     rs1_data;
    logic [XLEN-1:0]     rs2_data;
    logic [XLEN-1:0]     mdu_result;
    logic                mdu_ready;
    logic                mdu_valid;

    // Test variables
    int                  test_count;
    int                  pass_count;
    int                  fail_count;

    // DUT instantiation
    riscv_mdu #(
        .XLEN(XLEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mdu_enable(mdu_enable),
        .funct3(funct3),
        .is_32bit(is_32bit),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .mdu_result(mdu_result),
        .mdu_ready(mdu_ready),
        .mdu_valid(mdu_valid)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test stimulus
    initial begin
        $display("Starting RISC-V MDU Tests");
        
        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        mdu_enable = 0;
        funct3 = 3'b0;
        is_32bit = 0;
        rs1_data = 0;
        rs2_data = 0;
        
        // Reset
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // Test MUL instruction
        test_mul_64bit();
        
        // Test MULH instruction
        test_mulh_64bit();
        
        // Test MULHU instruction
        test_mulhu_64bit();
        
        // Test DIV instruction
        test_div_64bit();
        
        // Test DIVU instruction
        test_divu_64bit();
        
        // Test REM instruction
        test_rem_64bit();
        
        // Test REMU instruction
        test_remu_64bit();
        
        // Test 32-bit operations
        test_mul_32bit();
        test_div_32bit();
        
        // Test edge cases
        test_divide_by_zero();
        test_overflow_cases();
        
        // Summary
        $display("\n=== MDU Test Summary ===");
        $display("Total tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("All MDU tests PASSED!");
        end else begin
            $display("Some MDU tests FAILED!");
        end
        
        $finish;
    end

    // Test tasks
    task test_mul_64bit();
        $display("\n--- Testing MUL (64-bit) ---");
        
        // Test case 1: 15 * 10 = 150
        execute_mdu_op(3'b000, 0, 64'd15, 64'd10, 64'd150, "MUL 15*10");
        
        // Test case 2: -5 * 7 = -35
        execute_mdu_op(3'b000, 0, -64'd5, 64'd7, -64'd35, "MUL -5*7");
        
        // Test case 3: Large numbers
        execute_mdu_op(3'b000, 0, 64'h1000_0000, 64'h2000, 64'h2000_0000_0000, "MUL large");
    endtask

    task test_mulh_64bit();
        $display("\n--- Testing MULH (64-bit) ---");
        
        // Test case: Large multiplication requiring upper bits
        logic [127:0] expected_full;
        expected_full = $signed(64'h8000_0000_0000_0000) * $signed(64'h2);
        execute_mdu_op(3'b001, 0, 64'h8000_0000_0000_0000, 64'h2, 
                      expected_full[127:64], "MULH signed");
    endtask

    task test_mulhu_64bit();
        $display("\n--- Testing MULHU (64-bit) ---");
        
        // Test case: Unsigned multiplication
        logic [127:0] expected_full;
        expected_full = 64'hFFFF_FFFF_FFFF_FFFF * 64'h2;
        execute_mdu_op(3'b011, 0, 64'hFFFF_FFFF_FFFF_FFFF, 64'h2, 
                      expected_full[127:64], "MULHU unsigned");
    endtask

    task test_div_64bit();
        $display("\n--- Testing DIV (64-bit) ---");
        
        // Test case 1: 100 / 10 = 10
        execute_mdu_op(3'b100, 0, 64'd100, 64'd10, 64'd10, "DIV 100/10");
        
        // Test case 2: -100 / 10 = -10
        execute_mdu_op(3'b100, 0, -64'd100, 64'd10, -64'd10, "DIV -100/10");
        
        // Test case 3: 100 / -10 = -10
        execute_mdu_op(3'b100, 0, 64'd100, -64'd10, -64'd10, "DIV 100/-10");
    endtask

    task test_divu_64bit();
        $display("\n--- Testing DIVU (64-bit) ---");
        
        // Test case: Unsigned division
        execute_mdu_op(3'b101, 0, 64'hFFFF_FFFF_FFFF_FFFF, 64'h2, 
                      64'h7FFF_FFFF_FFFF_FFFF, "DIVU unsigned");
    endtask

    task test_rem_64bit();
        $display("\n--- Testing REM (64-bit) ---");
        
        // Test case 1: 103 % 10 = 3
        execute_mdu_op(3'b110, 0, 64'd103, 64'd10, 64'd3, "REM 103%10");
        
        // Test case 2: -103 % 10 = -3
        execute_mdu_op(3'b110, 0, -64'd103, 64'd10, -64'd3, "REM -103%10");
    endtask

    task test_remu_64bit();
        $display("\n--- Testing REMU (64-bit) ---");
        
        // Test case: Unsigned remainder
        execute_mdu_op(3'b111, 0, 64'd103, 64'd10, 64'd3, "REMU 103%10");
    endtask

    task test_mul_32bit();
        $display("\n--- Testing MULW (32-bit) ---");
        
        // Test case: 32-bit multiplication with sign extension
        execute_mdu_op(3'b000, 1, 64'h0000_0000_8000_0000, 64'h0000_0000_0000_0002, 
                      64'hFFFF_FFFF_0000_0000, "MULW 32-bit");
    endtask

    task test_div_32bit();
        $display("\n--- Testing DIVW (32-bit) ---");
        
        // Test case: 32-bit division with sign extension
        execute_mdu_op(3'b100, 1, 64'hFFFF_FFFF_FFFF_FF9C, 64'h0000_0000_0000_000A, 
                      64'hFFFF_FFFF_FFFF_FFFF, "DIVW 32-bit");
    endtask

    task test_divide_by_zero();
        $display("\n--- Testing Divide by Zero ---");
        
        // Division by zero should return all 1s
        execute_mdu_op(3'b100, 0, 64'd100, 64'd0, 64'hFFFF_FFFF_FFFF_FFFF, "DIV by zero");
        
        // Remainder by zero should return dividend
        execute_mdu_op(3'b110, 0, 64'd100, 64'd0, 64'd100, "REM by zero");
    endtask

    task test_overflow_cases();
        $display("\n--- Testing Overflow Cases ---");
        
        // Test signed overflow case: most negative / -1
        execute_mdu_op(3'b100, 0, 64'h8000_0000_0000_0000, 64'hFFFF_FFFF_FFFF_FFFF, 
                      64'h8000_0000_0000_0000, "DIV overflow");
    endtask

    task execute_mdu_op(
        input [2:0] op,
        input is_32,
        input [XLEN-1:0] a,
        input [XLEN-1:0] b,
        input [XLEN-1:0] expected,
        input string test_name
    );
        test_count++;
        
        // Set inputs
        funct3 = op;
        is_32bit = is_32;
        rs1_data = a;
        rs2_data = b;
        mdu_enable = 1;
        
        // Wait for operation to start
        @(posedge clk);
        mdu_enable = 0;
        
        // Wait for completion
        wait(mdu_valid);
        @(posedge clk);
        
        // Check result
        if (mdu_result == expected) begin
            $display("PASS: %s - Result: 0x%016h", test_name, mdu_result);
            pass_count++;
        end else begin
            $display("FAIL: %s - Expected: 0x%016h, Got: 0x%016h", 
                    test_name, expected, mdu_result);
            fail_count++;
        end
        
        // Wait a cycle before next test
        @(posedge clk);
    endtask

endmodule