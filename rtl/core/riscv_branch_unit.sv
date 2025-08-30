// RISC-V Branch Unit
// Handles branch condition evaluation and jump decisions

`timescale 1ns/1ps

module riscv_branch_unit #(
    parameter XLEN = 64
) (
    input  logic [XLEN-1:0]     rs1_data,
    input  logic [XLEN-1:0]     rs2_data,
    input  logic [2:0]          funct3,
    input  logic                branch,
    input  logic                jump,
    
    output logic                branch_taken,
    output logic                jump_taken
);

    // Branch function codes
    localparam [2:0] FUNCT3_BEQ  = 3'b000;  // Branch if Equal
    localparam [2:0] FUNCT3_BNE  = 3'b001;  // Branch if Not Equal
    localparam [2:0] FUNCT3_BLT  = 3'b100;  // Branch if Less Than
    localparam [2:0] FUNCT3_BGE  = 3'b101;  // Branch if Greater or Equal
    localparam [2:0] FUNCT3_BLTU = 3'b110;  // Branch if Less Than Unsigned
    localparam [2:0] FUNCT3_BGEU = 3'b111;  // Branch if Greater or Equal Unsigned

    logic condition_met;

    always_comb begin
        condition_met = 1'b0;
        
        case (funct3)
            FUNCT3_BEQ: begin
                condition_met = (rs1_data == rs2_data);
            end
            
            FUNCT3_BNE: begin
                condition_met = (rs1_data != rs2_data);
            end
            
            FUNCT3_BLT: begin
                condition_met = ($signed(rs1_data) < $signed(rs2_data));
            end
            
            FUNCT3_BGE: begin
                condition_met = ($signed(rs1_data) >= $signed(rs2_data));
            end
            
            FUNCT3_BLTU: begin
                condition_met = (rs1_data < rs2_data);
            end
            
            FUNCT3_BGEU: begin
                condition_met = (rs1_data >= rs2_data);
            end
            
            default: begin
                condition_met = 1'b0;
            end
        endcase
    end

    assign branch_taken = branch && condition_met;
    assign jump_taken = jump;

endmodule