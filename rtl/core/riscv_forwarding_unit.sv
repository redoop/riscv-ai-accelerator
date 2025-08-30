// RISC-V Forwarding Unit
// Handles data forwarding to resolve data hazards

`timescale 1ns/1ps

module riscv_forwarding_unit (
    input  logic [4:0]  rs1_e,
    input  logic [4:0]  rs2_e,
    input  logic [4:0]  rd_m,
    input  logic [4:0]  rd_w,
    input  logic        reg_write_m,
    input  logic        reg_write_w,
    
    output logic [1:0]  forward_a,
    output logic [1:0]  forward_b
);

    // Forwarding control values
    // 00: No forwarding (use register file)
    // 01: Forward from writeback stage
    // 10: Forward from memory stage
    // 11: Reserved

    always_comb begin
        // Default: no forwarding
        forward_a = 2'b00;
        forward_b = 2'b00;
        
        // Forward A (rs1)
        if (reg_write_m && (rd_m != 5'b0) && (rd_m == rs1_e)) begin
            forward_a = 2'b10; // Forward from memory stage
        end else if (reg_write_w && (rd_w != 5'b0) && (rd_w == rs1_e)) begin
            forward_a = 2'b01; // Forward from writeback stage
        end
        
        // Forward B (rs2)
        if (reg_write_m && (rd_m != 5'b0) && (rd_m == rs2_e)) begin
            forward_b = 2'b10; // Forward from memory stage
        end else if (reg_write_w && (rd_w != 5'b0) && (rd_w == rs2_e)) begin
            forward_b = 2'b01; // Forward from writeback stage
        end
    end

endmodule