// Test bench for Vector ALU module
// Tests arithmetic operations and data type conversions

`timescale 1ns / 1ps

`include "chip_config.sv"

module test_vector_alu;

    import chip_config_pkg::*;

    // ========================================
    // Test Parameters
    // ========================================
    
    parameter CLK_PERIOD = 10; // 10ns = 100MHz
    parameter ELEMENT_WIDTH = 64;

    // ========================================
    // DUT Signals
    // ========================================
    
    logic clk;
    logic rst_n;
    
    // Operation control
    logic [3:0] operation;
    data_type_e src_dtype;
    data_type_e dst_dtype;
    
    // Data inputs
    logic [ELEMENT_WIDTH-1:0] operand_a;
    logic [ELEMENT_WIDTH-1:0] operand_b;
    logic valid_in;
    
    // Results
    logic [ELEMENT_WIDTH-1:0] result;
    logic valid_out;
    logic overflow;
    logic underflow;

    // ========================================
    // DUT Instantiation
    // ========================================
    
    vector_alu #(
        .ELEMENT_WIDTH(ELEMENT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .operation(operation),
        .src_dtype(src_dtype),
        .dst_dtype(dst_dtype),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .valid_in(valid_in),
        .result(result),
        .valid_out(valid_out),
        .overflow(overflow),
        .underflow(underflow)
    );

    // ========================================
    // Clock Generation
    // ========================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // ========================================
    // Test Variables
    // ========================================
    
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    // ========================================
    // Test Tasks
    // ========================================
    
    // Reset task
    task reset_dut();
        begin
            rst_n = 0;
            operation = 0;
            src_dtype = DTYPE_INT32;
            dst_dtype = DTYPE_INT32;
            operand_a = 0;
            operand_b = 0;
            valid_in = 0;
            
            repeat (10) @(posedge clk);
            rst_n = 1;
            repeat (5) @(posedge clk);
        end
    endtask
    
    // Execute operation task
    task execute_operation(
        input [3:0] op,
        input [ELEMENT_WIDTH-1:0] a,
        input [ELEMENT_WIDTH-1:0] b,
        input data_type_e src_type,
        input data_type_e dst_type
    );
        begin
            @(posedge clk);
            operation = op;
            operand_a = a;
            operand_b = b;
            src_dtype = src_type;
            dst_dtype = dst_type;
            valid_in = 1;
            
            @(posedge clk);
            valid_in = 0;
            
            // Wait for result
            while (!valid_out) @(posedge clk);
            @(posedge clk);
        end
    endtask
    
    // Check result task
    task check_result(
        input string test_name,
        input [ELEMENT_WIDTH-1:0] expected,
        input logic expected_overflow,
        input logic expected_underflow
    );
        begin
            test_count++;
            if (result == expected && overflow == expected_overflow && underflow == expected_underflow) begin
                $display("[PASS] %s: Result = %h, Overflow = %b, Underflow = %b", 
                        test_name, result, overflow, underflow);
                pass_count++;
            end else begin
                $display("[FAIL] %s: Expected = %h (OF=%b, UF=%b), Actual = %h (OF=%b, UF=%b)", 
                        test_name, expected, expected_overflow, expected_underflow, 
                        result, overflow, underflow);
                fail_count++;
            end
        end
    endtask

    // ========================================
    // Test Scenarios
    // ========================================
    
    // Test addition operations
    task test_addition();
        begin
            $display("\n=== Testing Addition Operations ===");
            
            // Basic addition
            execute_operation(4'b0000, 64'h123456789ABCDEF0, 64'h0FEDCBA987654321, DTYPE_INT32, DTYPE_INT32);
            check_result("Basic Addition", 64'h2222222222222211, 1'b0, 1'b0);
            
            // Addition with overflow (using smaller values to demonstrate)
            execute_operation(4'b0000, 64'hFFFFFFFFFFFFFFFF, 64'h0000000000000001, DTYPE_INT32, DTYPE_INT32);
            check_result("Addition Overflow", 64'h0000000000000000, 1'b1, 1'b0);
            
            // Zero addition
            execute_operation(4'b0000, 64'h0000000000000000, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Zero Addition", 64'h0000000000000000, 1'b0, 1'b0);
        end
    endtask
    
    // Test subtraction operations
    task test_subtraction();
        begin
            $display("\n=== Testing Subtraction Operations ===");
            
            // Basic subtraction
            execute_operation(4'b0001, 64'h123456789ABCDEF0, 64'h0FEDCBA987654321, DTYPE_INT32, DTYPE_INT32);
            check_result("Basic Subtraction", 64'h02468ACE13579BCF, 1'b0, 1'b0);
            
            // Subtraction with underflow
            execute_operation(4'b0001, 64'h0000000000000001, 64'h0000000000000002, DTYPE_INT32, DTYPE_INT32);
            check_result("Subtraction Underflow", 64'h0000000000000000, 1'b0, 1'b1);
            
            // Self subtraction
            execute_operation(4'b0001, 64'h123456789ABCDEF0, 64'h123456789ABCDEF0, DTYPE_INT32, DTYPE_INT32);
            check_result("Self Subtraction", 64'h0000000000000000, 1'b0, 1'b0);
        end
    endtask
    
    // Test multiplication operations
    task test_multiplication();
        begin
            $display("\n=== Testing Multiplication Operations ===");
            
            // Basic multiplication (using lower 32 bits)
            execute_operation(4'b0010, 64'h0000000000000010, 64'h0000000000000020, DTYPE_INT32, DTYPE_INT32);
            check_result("Basic Multiplication", 64'h0000000000000200, 1'b0, 1'b0);
            
            // Multiplication by zero
            execute_operation(4'b0010, 64'h123456789ABCDEF0, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Multiplication by Zero", 64'h0000000000000000, 1'b0, 1'b0);
            
            // Multiplication by one
            execute_operation(4'b0010, 64'h0000000000000042, 64'h0000000000000001, DTYPE_INT32, DTYPE_INT32);
            check_result("Multiplication by One", 64'h0000000000000042, 1'b0, 1'b0);
        end
    endtask
    
    // Test division operations
    task test_division();
        begin
            $display("\n=== Testing Division Operations ===");
            
            // Basic division
            execute_operation(4'b0011, 64'h0000000000000100, 64'h0000000000000010, DTYPE_INT32, DTYPE_INT32);
            check_result("Basic Division", 64'h0000000000000010, 1'b0, 1'b0);
            
            // Division by zero
            execute_operation(4'b0011, 64'h123456789ABCDEF0, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Division by Zero", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            // Division by one
            execute_operation(4'b0011, 64'h0000000000000042, 64'h0000000000000001, DTYPE_INT32, DTYPE_INT32);
            check_result("Division by One", 64'h0000000000000042, 1'b0, 1'b0);
        end
    endtask
    
    // Test logical operations
    task test_logical_operations();
        begin
            $display("\n=== Testing Logical Operations ===");
            
            // AND operation
            execute_operation(4'b0100, 64'hAAAAAAAAAAAAAAAA, 64'h5555555555555555, DTYPE_INT32, DTYPE_INT32);
            check_result("AND Operation", 64'h0000000000000000, 1'b0, 1'b0);
            
            // OR operation
            execute_operation(4'b0101, 64'hAAAAAAAAAAAAAAAA, 64'h5555555555555555, DTYPE_INT32, DTYPE_INT32);
            check_result("OR Operation", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            // XOR operation
            execute_operation(4'b0110, 64'hAAAAAAAAAAAAAAAA, 64'h5555555555555555, DTYPE_INT32, DTYPE_INT32);
            check_result("XOR Operation", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            // XOR with self (should be zero)
            execute_operation(4'b0110, 64'h123456789ABCDEF0, 64'h123456789ABCDEF0, DTYPE_INT32, DTYPE_INT32);
            check_result("XOR Self", 64'h0000000000000000, 1'b0, 1'b0);
        end
    endtask
    
    // Test min/max operations
    task test_min_max_operations();
        begin
            $display("\n=== Testing Min/Max Operations ===");
            
            // MIN operation
            execute_operation(4'b0111, 64'h0000000000000100, 64'h0000000000000200, DTYPE_INT32, DTYPE_INT32);
            check_result("MIN Operation", 64'h0000000000000100, 1'b0, 1'b0);
            
            // MAX operation
            execute_operation(4'b1000, 64'h0000000000000100, 64'h0000000000000200, DTYPE_INT32, DTYPE_INT32);
            check_result("MAX Operation", 64'h0000000000000200, 1'b0, 1'b0);
            
            // MIN with equal values
            execute_operation(4'b0111, 64'h0000000000000100, 64'h0000000000000100, DTYPE_INT32, DTYPE_INT32);
            check_result("MIN Equal Values", 64'h0000000000000100, 1'b0, 1'b0);
            
            // MAX with equal values
            execute_operation(4'b1000, 64'h0000000000000100, 64'h0000000000000100, DTYPE_INT32, DTYPE_INT32);
            check_result("MAX Equal Values", 64'h0000000000000100, 1'b0, 1'b0);
        end
    endtask
    
    // Enhanced data type conversions test
    task test_data_type_conversions();
        begin
            $display("\n=== Testing Enhanced Data Type Conversions ===");
            
            // Test 1: INT8 to FP16 conversion (enhanced)
            execute_operation(4'b1001, 64'h0000000000000042, 64'h0000000000000000, DTYPE_INT8, DTYPE_FP16);
            // Expected: 66 decimal = 0x5420 in FP16 (sign=0, exp=10101, mantissa=0001000000)
            check_result("INT8 to FP16 Enhanced", 64'h0000000000005420, 1'b0, 1'b0);
            
            // Test 2: FP16 to INT8 conversion (enhanced)
            execute_operation(4'b1001, 64'h0000000000005420, 64'h0000000000000000, DTYPE_FP16, DTYPE_INT8);
            // Expected: Should convert back to 66 (0x42)
            check_result("FP16 to INT8 Enhanced", 64'h0000000000000042, 1'b0, 1'b0);
            
            // Test 3: Negative number conversion
            execute_operation(4'b1001, 64'h00000000000000BE, 64'h0000000000000000, DTYPE_INT8, DTYPE_FP16);
            // -66 in INT8 = 0xBE, should convert to negative FP16
            check_result("Negative INT8 to FP16", 64'h000000000000D420, 1'b0, 1'b0);
            
            // Test 4: Zero conversion
            execute_operation(4'b1001, 64'h0000000000000000, 64'h0000000000000000, DTYPE_INT8, DTYPE_FP16);
            check_result("Zero INT8 to FP16", 64'h0000000000000000, 1'b0, 1'b0);
            
            // Test 5: Maximum value conversion
            execute_operation(4'b1001, 64'h000000000000007F, 64'h0000000000000000, DTYPE_INT8, DTYPE_FP16);
            // 127 decimal should convert properly
            check_result("Max INT8 to FP16", 64'h00000000000057F0, 1'b0, 1'b0);
            
            // Test 6: Widening conversion INT16 to INT32
            execute_operation(4'b1001, 64'h0000000000001234, 64'h0000000000000000, DTYPE_INT16, DTYPE_INT32);
            // Expected: sign-extended and shifted appropriately
            check_result("INT16 to INT32 Widening", 64'h0000000012340000, 1'b0, 1'b0);
            
            // Test 7: Narrowing conversion INT32 to INT16
            execute_operation(4'b1001, 64'h0000000012345678, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT16);
            // Expected: truncated to upper 16 bits
            check_result("INT32 to INT16 Narrowing", 64'h0000000000001234, 1'b0, 1'b0);
            
            // Test 8: Same type conversion (pass-through)
            execute_operation(4'b1001, 64'h123456789ABCDEF0, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Same Type Conversion", 64'h123456789ABCDEF0, 1'b0, 1'b0);
        end
    endtask
    
    // Test pipeline behavior and timing
    task test_pipeline_behavior();
        begin
            $display("\n=== Testing Pipeline Behavior ===");
            
            // Test 1: Single cycle operations (ADD, SUB, logical)
            execute_operation(4'b0000, 64'h0000000000000010, 64'h0000000000000020, DTYPE_INT32, DTYPE_INT32);
            check_result("Single Cycle ADD", 64'h0000000000000030, 1'b0, 1'b0);
            
            // Test 2: Multi-cycle multiply operation
            execute_operation(4'b0010, 64'h0000000000000008, 64'h0000000000000004, DTYPE_INT32, DTYPE_INT32);
            check_result("Multi-cycle MUL", 64'h0000000000000020, 1'b0, 1'b0);
            
            // Test 3: Multi-cycle divide operation
            execute_operation(4'b0011, 64'h0000000000000100, 64'h0000000000000008, DTYPE_INT32, DTYPE_INT32);
            check_result("Multi-cycle DIV", 64'h0000000000000020, 1'b0, 1'b0);
            
            // Test 4: Back-to-back operations
            execute_operation(4'b0000, 64'h0000000000000001, 64'h0000000000000002, DTYPE_INT32, DTYPE_INT32);
            check_result("Back-to-back Op 1", 64'h0000000000000003, 1'b0, 1'b0);
            
            execute_operation(4'b0001, 64'h0000000000000010, 64'h0000000000000005, DTYPE_INT32, DTYPE_INT32);
            check_result("Back-to-back Op 2", 64'h000000000000000B, 1'b0, 1'b0);
        end
    endtask
    
    // Test error conditions and edge cases
    task test_error_conditions();
        begin
            $display("\n=== Testing Error Conditions ===");
            
            // Test 1: Division by zero
            execute_operation(4'b0011, 64'h0000000000000100, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Division by Zero", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            // Test 2: Overflow in addition
            execute_operation(4'b0000, 64'hFFFFFFFFFFFFFFFF, 64'h0000000000000001, DTYPE_INT32, DTYPE_INT32);
            check_result("Addition Overflow", 64'h0000000000000000, 1'b1, 1'b0);
            
            // Test 3: Underflow in subtraction
            execute_operation(4'b0001, 64'h0000000000000001, 64'h0000000000000002, DTYPE_INT32, DTYPE_INT32);
            check_result("Subtraction Underflow", 64'h0000000000000000, 1'b0, 1'b1);
            
            // Test 4: Invalid operation code
            execute_operation(4'b1111, 64'h123456789ABCDEF0, 64'h0FEDCBA987654321, DTYPE_INT32, DTYPE_INT32);
            check_result("Invalid Operation", 64'h123456789ABCDEF0, 1'b0, 1'b0);
            
            // Test 5: Maximum value operations
            execute_operation(4'b1000, 64'hFFFFFFFFFFFFFFFF, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("MAX with Zero", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            execute_operation(4'b0111, 64'hFFFFFFFFFFFFFFFF, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("MIN with Zero", 64'h0000000000000000, 1'b0, 1'b0);
        end
    endtask
    
    // Test edge cases
    task test_edge_cases();
        begin
            $display("\n=== Testing Edge Cases ===");
            
            // Maximum values
            execute_operation(4'b0000, 64'hFFFFFFFFFFFFFFFF, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Max Value Addition", 64'hFFFFFFFFFFFFFFFF, 1'b0, 1'b0);
            
            // Minimum values
            execute_operation(4'b0001, 64'h0000000000000000, 64'h0000000000000000, DTYPE_INT32, DTYPE_INT32);
            check_result("Min Value Subtraction", 64'h0000000000000000, 1'b0, 1'b0);
            
            // Invalid operation (should default to operand_a)
            execute_operation(4'b1111, 64'h123456789ABCDEF0, 64'h0FEDCBA987654321, DTYPE_INT32, DTYPE_INT32);
            check_result("Invalid Operation", 64'h123456789ABCDEF0, 1'b0, 1'b0);
        end
    endtask

    // ========================================
    // Main Test Sequence
    // ========================================
    
    initial begin
        $display("Starting Vector ALU Test");
        $display("========================");
        
        // Initialize
        reset_dut();
        
        // Run enhanced test scenarios
        test_addition();
        test_subtraction();
        test_multiplication();
        test_division();
        test_logical_operations();
        test_min_max_operations();
        test_data_type_conversions();
        test_pipeline_behavior();
        test_error_conditions();
        test_edge_cases();
        
        // Test summary
        $display("\n=== Test Summary ===");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $display("Vector ALU Test Complete");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000; // 100us timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule