// RISC-V Hazard Detection Unit
// Detects and handles pipeline hazards

`timescale 1ns/1ps

module riscv_hazard_unit (
    input  logic [4:0]  rs1_d,
    input  logic [4:0]  rs2_d,
    input  logic [4:0]  rd_e,
    input  logic        mem_read_e,
    input  logic        branch_taken,
    input  logic        jump_taken,
    
    output logic        stall,
    output logic        load_use_hazard
);

    // Load-use hazard detection
    // Occurs when an instruction tries to use the result of a load instruction
    // in the immediately following instruction
    always_comb begin
        load_use_hazard = 1'b0;
        
        if (mem_read_e && (rd_e != 5'b0)) begin
            if ((rd_e == rs1_d) || (rd_e == rs2_d)) begin
                load_use_hazard = 1'b1;
            end
        end
    end

    // Stall conditions
    assign stall = load_use_hazard;

endmodule