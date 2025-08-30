// Test bench for RISC-V Floating Point Unit (FPU)
// Tests RV64F and RV64D extensions

`timescale 1ns / 1ps

module test_riscv_fpu;

    // Parameters
    parameter XLEN = 64;
    parameter CLK_PERIOD = 10;

    // Signals
    logic                clk;
    logic                rst_n;
    logic                fpu_enable;
    logic [2:0]          fpu_op;
    logic [2:0]          funct3;
    logic [6:0]          funct7;
    logic                is_double;
    logic [XLEN-1:0]     rs1_data;
    logic [XLEN-1:0]     rs2_data;
    logic [XLEN-1:0]     rs3_data;
    logic [XLEN-1:0]     fpu_result;
    logic                fpu_ready;
    logic [4:0]          fpu_flags;

    // Test variables
    int                  test_count;
    int                  pass_count;
    int                  fail_count;

    // DUT instantiation
    riscv_fpu #(
        .XLEN(XLEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .fpu_enable(fpu_enable),
        .fpu_op(fpu_op),
        .funct3(funct3),
        .funct7(funct7),
        .is_double(is_double),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .rs3_data(rs3_data),
        .fpu_result(fpu_result),
        .fpu_ready(fpu_ready),
        .fpu_flags(fpu_flags)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test stimulus
    initial begin
        $display("Starting RISC-V FPU Tests");
        
        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        fpu_enable = 0;
        fpu_op = 3'b0;
        funct3 = 3'b0;
        funct7 = 7'b0;
        is_double = 0;
        rs1_data = 0;
        rs2_data = 0;
        rs3_data = 0;
        
        // Reset
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // Test single precision operations
        test_single_precision();
        
        // Test double precision operations
        test_double_precision();
        
        // Test floating point comparisons
        test_fp_comparisons();
        
        // Test floating point conversions
        test_fp_conversions();
        
        // Test special cases
        test_special_cases();
        
        // Summary
        $display("\n=== FPU Test Summary ===");
        $display("Total tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("All FPU tests PASSED!");
        end else begin
            $display("Some FPU tests FAILED!");
        end
        
        $finish;
    end

    // Test tasks
    task test_single_precision();
        $display("\n--- Testing Single Precision Operations ---");
        
        // Test FADD.S: 1.5 + 2.5 = 4.0
        execute_fpu_op(3'b000, 3'b000, 7'b0000000, 0, 
                      {32'hFFFF_FFFF, 32'h3FC0_0000}, // 1.5 (NaN-boxed)
                      {32'hFFFF_FFFF, 32'h4020_0000}, // 2.5 (NaN-boxed)
                      64'h0,
                      {32'hFFFF_FFFF, 32'h4080_0000}, // 4.0 (NaN-boxed)
                      "FADD.S 1.5+2.5");
        
        // Test FSUB.S: 4.0 - 1.5 = 2.5
        execute_fpu_op(3'b001, 3'b000, 7'b0000100, 0,
                      {32'hFFFF_FFFF, 32'h4080_0000}, // 4.0
                      {32'hFFFF_FFFF, 32'h3FC0_0000}, // 1.5
                      64'h0,
                      {32'hFFFF_FFFF, 32'h4020_0000}, // 2.5
                      "FSUB.S 4.0-1.5");
        
        // Test FMUL.S: 2.0 * 3.0 = 6.0
        execute_fpu_op(3'b010, 3'b000, 7'b0001000, 0,
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      {32'hFFFF_FFFF, 32'h4040_0000}, // 3.0
                      64'h0,
                      {32'hFFFF_FFFF, 32'h40C0_0000}, // 6.0
                      "FMUL.S 2.0*3.0");
    endtask

    task test_double_precision();
        $display("\n--- Testing Double Precision Operations ---");
        
        // Test FADD.D: 1.5 + 2.5 = 4.0
        execute_fpu_op(3'b000, 3'b000, 7'b0000001, 1,
                      64'h3FF8_0000_0000_0000, // 1.5 (double)
                      64'h4004_0000_0000_0000, // 2.5 (double)
                      64'h0,
                      64'h4010_0000_0000_0000, // 4.0 (double)
                      "FADD.D 1.5+2.5");
        
        // Test FSUB.D: 4.0 - 1.5 = 2.5
        execute_fpu_op(3'b001, 3'b000, 7'b0000101, 1,
                      64'h4010_0000_0000_0000, // 4.0
                      64'h3FF8_0000_0000_0000, // 1.5
                      64'h0,
                      64'h4004_0000_0000_0000, // 2.5
                      "FSUB.D 4.0-1.5");
    endtask

    task test_fp_comparisons();
        $display("\n--- Testing Floating Point Comparisons ---");
        
        // Test FEQ.S: 2.0 == 2.0 should be true (1)
        execute_fpu_op(3'b110, 3'b010, 7'b1010000, 0,
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      64'h0,
                      64'h0000_0000_0000_0001, // true
                      "FEQ.S 2.0==2.0");
        
        // Test FLT.S: 1.0 < 2.0 should be true (1)
        execute_fpu_op(3'b110, 3'b001, 7'b1010000, 0,
                      {32'hFFFF_FFFF, 32'h3F80_0000}, // 1.0
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      64'h0,
                      64'h0000_0000_0000_0001, // true
                      "FLT.S 1.0<2.0");
        
        // Test FLE.S: 2.0 <= 2.0 should be true (1)
        execute_fpu_op(3'b110, 3'b000, 7'b1010000, 0,
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      64'h0,
                      64'h0000_0000_0000_0001, // true
                      "FLE.S 2.0<=2.0");
    endtask

    task test_fp_conversions();
        $display("\n--- Testing Floating Point Conversions ---");
        
        // Test FCVT.S.D: Convert double to single
        execute_fpu_op(3'b111, 3'b000, 7'b0100000, 0,
                      64'h4010_0000_0000_0000, // 4.0 (double)
                      64'h0,
                      64'h0,
                      {32'hFFFF_FFFF, 32'h4080_0000}, // 4.0 (single, NaN-boxed)
                      "FCVT.S.D 4.0");
        
        // Test FCVT.D.S: Convert single to double
        execute_fpu_op(3'b111, 3'b000, 7'b0100001, 1,
                      {32'hFFFF_FFFF, 32'h4080_0000}, // 4.0 (single)
                      64'h0,
                      64'h0,
                      64'h4010_0000_0000_0000, // 4.0 (double)
                      "FCVT.D.S 4.0");
    endtask

    task test_special_cases();
        $display("\n--- Testing Special Cases ---");
        
        // Test FSGNJ.S: Sign injection
        execute_fpu_op(3'b101, 3'b000, 7'b0010000, 0,
                      {32'hFFFF_FFFF, 32'h4000_0000}, // +2.0
                      {32'hFFFF_FFFF, 32'hC000_0000}, // -2.0 (for sign)
                      64'h0,
                      {32'hFFFF_FFFF, 32'hC000_0000}, // -2.0 (result)
                      "FSGNJ.S +2.0,-2.0");
        
        // Test FMIN.S: Minimum of two values
        execute_fpu_op(3'b101, 3'b000, 7'b0010100, 0,
                      {32'hFFFF_FFFF, 32'h3F80_0000}, // 1.0
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      64'h0,
                      {32'hFFFF_FFFF, 32'h3F80_0000}, // 1.0 (minimum)
                      "FMIN.S 1.0,2.0");
        
        // Test FMAX.S: Maximum of two values
        execute_fpu_op(3'b101, 3'b001, 7'b0010100, 0,
                      {32'hFFFF_FFFF, 32'h3F80_0000}, // 1.0
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0
                      64'h0,
                      {32'hFFFF_FFFF, 32'h4000_0000}, // 2.0 (maximum)
                      "FMAX.S 1.0,2.0");
    endtask

    task execute_fpu_op(
        input [2:0] op,
        input [2:0] f3,
        input [6:0] f7,
        input is_dp,
        input [XLEN-1:0] a,
        input [XLEN-1:0] b,
        input [XLEN-1:0] c,
        input [XLEN-1:0] expected,
        input string test_name
    );
        test_count++;
        
        // Set inputs
        fpu_op = op;
        funct3 = f3;
        funct7 = f7;
        is_double = is_dp;
        rs1_data = a;
        rs2_data = b;
        rs3_data = c;
        fpu_enable = 1;
        
        // Wait for operation
        @(posedge clk);
        fpu_enable = 0;
        
        // Wait for ready (combinational for now)
        wait(fpu_ready);
        @(posedge clk);
        
        // Check result (simplified comparison for basic functionality)
        if (fpu_result == expected) begin
            $display("PASS: %s - Result: 0x%016h", test_name, fpu_result);
            pass_count++;
        end else begin
            $display("FAIL: %s - Expected: 0x%016h, Got: 0x%016h", 
                    test_name, expected, fpu_result);
            fail_count++;
        end
        
        // Wait a cycle before next test
        @(posedge clk);
    endtask

endmodule