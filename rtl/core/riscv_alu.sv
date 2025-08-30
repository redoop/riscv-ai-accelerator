// RISC-V Arithmetic Logic Unit
// Performs arithmetic and logical operations

`timescale 1ns/1ps

module riscv_alu #(
    parameter XLEN = 64
) (
    input  logic [XLEN-1:0]     a,
    input  logic [XLEN-1:0]     b,
    input  logic [3:0]          alu_op,
    
    output logic [XLEN-1:0]     result,
    output logic                zero,
    output logic                overflow
);

    // ALU Operation Codes
    localparam [3:0] ALU_ADD    = 4'b0000;
    localparam [3:0] ALU_SUB    = 4'b0001;
    localparam [3:0] ALU_SLL    = 4'b0010;
    localparam [3:0] ALU_SLT    = 4'b0011;
    localparam [3:0] ALU_SLTU   = 4'b0100;
    localparam [3:0] ALU_XOR    = 4'b0101;
    localparam [3:0] ALU_SRL    = 4'b0110;
    localparam [3:0] ALU_SRA    = 4'b0111;
    localparam [3:0] ALU_OR     = 4'b1000;
    localparam [3:0] ALU_AND    = 4'b1001;
    localparam [3:0] ALU_LUI    = 4'b1010;
    localparam [3:0] ALU_AUIPC  = 4'b1011;

    logic [XLEN-1:0] add_result;
    logic [XLEN-1:0] sub_result;
    logic            add_overflow;
    logic            sub_overflow;
    logic [5:0]      shift_amount;

    // Addition and subtraction with overflow detection
    assign add_result = a + b;
    assign sub_result = a - b;
    
    // Overflow detection for signed operations
    assign add_overflow = (a[XLEN-1] == b[XLEN-1]) && (add_result[XLEN-1] != a[XLEN-1]);
    assign sub_overflow = (a[XLEN-1] != b[XLEN-1]) && (sub_result[XLEN-1] != a[XLEN-1]);
    
    // Shift amount (only lower 6 bits for 64-bit, 5 bits for 32-bit)
    assign shift_amount = b[5:0];

    always_comb begin
        result = '0;
        overflow = 1'b0;
        
        case (alu_op)
            ALU_ADD: begin
                result = add_result;
                overflow = add_overflow;
            end
            
            ALU_SUB: begin
                result = sub_result;
                overflow = sub_overflow;
            end
            
            ALU_SLL: begin
                result = a << shift_amount;
            end
            
            ALU_SLT: begin
                result = {{(XLEN-1){1'b0}}, ($signed(a) < $signed(b))};
            end
            
            ALU_SLTU: begin
                result = {{(XLEN-1){1'b0}}, (a < b)};
            end
            
            ALU_XOR: begin
                result = a ^ b;
            end
            
            ALU_SRL: begin
                result = a >> shift_amount;
            end
            
            ALU_SRA: begin
                result = $signed(a) >>> shift_amount;
            end
            
            ALU_OR: begin
                result = a | b;
            end
            
            ALU_AND: begin
                result = a & b;
            end
            
            ALU_LUI: begin
                result = b; // Immediate is already shifted in decode stage
            end
            
            ALU_AUIPC: begin
                result = a + b; // PC + immediate
            end
            
            default: begin
                result = add_result;
            end
        endcase
    end

    assign zero = (result == '0);

endmodule