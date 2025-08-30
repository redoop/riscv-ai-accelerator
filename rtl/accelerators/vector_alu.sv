// Vector Arithmetic Logic Unit (ALU)
// Implements vector arithmetic operations and data type conversions

// Import chip configuration package
import chip_config_pkg::*;

`timescale 1ns/1ps

module vector_alu #(
    parameter ELEMENT_WIDTH = 64
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Operation control
    input  logic [3:0]              operation,
    input  data_type_e              src_dtype,
    input  data_type_e              dst_dtype,
    
    // Data inputs
    input  logic [ELEMENT_WIDTH-1:0] operand_a,
    input  logic [ELEMENT_WIDTH-1:0] operand_b,
    input  logic                     valid_in,
    
    // Results
    output logic [ELEMENT_WIDTH-1:0] result,
    output logic                     valid_out,
    output logic                     overflow,
    output logic                     underflow
);

    // Operation types
    typedef enum logic [3:0] {
        ALU_ADD     = 4'b0000,
        ALU_SUB     = 4'b0001,
        ALU_MUL     = 4'b0010,
        ALU_DIV     = 4'b0011,
        ALU_AND     = 4'b0100,
        ALU_OR      = 4'b0101,
        ALU_XOR     = 4'b0110,
        ALU_MIN     = 4'b0111,
        ALU_MAX     = 4'b1000,
        ALU_CONVERT = 4'b1001
    } alu_operation_e;
    
    // Internal signals
    logic [ELEMENT_WIDTH-1:0] alu_result;
    logic                     alu_valid;
    logic                     alu_overflow;
    logic                     alu_underflow;
    
    // Pipeline registers
    logic [ELEMENT_WIDTH-1:0] op_a_reg;
    logic [ELEMENT_WIDTH-1:0] op_b_reg;
    logic [3:0]               operation_reg;
    logic                     valid_reg;
    
    // Pipeline stage 1: Register inputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_a_reg <= '0;
            op_b_reg <= '0;
            operation_reg <= '0;
            valid_reg <= 1'b0;
        end else begin
            op_a_reg <= operand_a;
            op_b_reg <= operand_b;
            operation_reg <= operation;
            valid_reg <= valid_in;
        end
    end
    
    // Pipeline stage 2: Arithmetic operations
    always_comb begin
        alu_result = '0;
        alu_overflow = 1'b0;
        alu_underflow = 1'b0;
        alu_valid = valid_reg;
        
        case (alu_operation_e'(operation_reg))
            ALU_ADD: begin
                alu_result = op_a_reg + op_b_reg;
                // Simple overflow detection
                alu_overflow = (op_a_reg[ELEMENT_WIDTH-1] == op_b_reg[ELEMENT_WIDTH-1]) && 
                              (alu_result[ELEMENT_WIDTH-1] != op_a_reg[ELEMENT_WIDTH-1]);
            end
            
            ALU_SUB: begin
                alu_result = op_a_reg - op_b_reg;
                // Simple underflow detection
                alu_underflow = (op_a_reg < op_b_reg);
            end
            
            ALU_MUL: begin
                // Simplified multiplication (lower bits only)
                alu_result = op_a_reg[ELEMENT_WIDTH/2-1:0] * op_b_reg[ELEMENT_WIDTH/2-1:0];
            end
            
            ALU_DIV: begin
                if (op_b_reg != 0) begin
                    alu_result = op_a_reg / op_b_reg;
                end else begin
                    alu_result = '1; // Division by zero result
                    alu_overflow = 1'b1;
                end
            end
            
            ALU_AND: begin
                alu_result = op_a_reg & op_b_reg;
            end
            
            ALU_OR: begin
                alu_result = op_a_reg | op_b_reg;
            end
            
            ALU_XOR: begin
                alu_result = op_a_reg ^ op_b_reg;
            end
            
            ALU_MIN: begin
                alu_result = (op_a_reg < op_b_reg) ? op_a_reg : op_b_reg;
            end
            
            ALU_MAX: begin
                alu_result = (op_a_reg > op_b_reg) ? op_a_reg : op_b_reg;
            end
            
            ALU_CONVERT: begin
                // Simplified type conversion (just pass through for now)
                alu_result = op_a_reg;
            end
            
            default: begin
                alu_result = op_a_reg;
            end
        endcase
    end
    
    // Output assignment
    assign result = alu_result;
    assign valid_out = alu_valid;
    assign overflow = alu_overflow;
    assign underflow = alu_underflow;

endmodule