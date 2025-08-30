// RISC-V Multiply/Divide Unit (MDU)
// Implements RV64M extension (multiply and divide operations)

`timescale 1ns/1ps

module riscv_mdu #(
    parameter XLEN = 64
) (
    input  logic                clk,
    input  logic                rst_n,
    
    // Control interface
    input  logic                mdu_enable,
    input  logic [2:0]          funct3,
    input  logic                is_32bit,   // 1 for 32-bit operations (W suffix)
    
    // Data interface
    input  logic [XLEN-1:0]     rs1_data,
    input  logic [XLEN-1:0]     rs2_data,
    
    output logic [XLEN-1:0]     mdu_result,
    output logic                mdu_ready,
    output logic                mdu_valid
);

    // M extension function codes
    localparam [2:0] FUNCT3_MUL    = 3'b000;  // MUL/MULW
    localparam [2:0] FUNCT3_MULH   = 3'b001;  // MULH
    localparam [2:0] FUNCT3_MULHSU = 3'b010;  // MULHSU
    localparam [2:0] FUNCT3_MULHU  = 3'b011;  // MULHU
    localparam [2:0] FUNCT3_DIV    = 3'b100;  // DIV/DIVW
    localparam [2:0] FUNCT3_DIVU   = 3'b101;  // DIVU/DIVUW
    localparam [2:0] FUNCT3_REM    = 3'b110;  // REM/REMW
    localparam [2:0] FUNCT3_REMU   = 3'b111;  // REMU/REMUW

    // Internal signals
    logic [XLEN-1:0]    operand_a, operand_b;
    logic [2*XLEN-1:0]  multiply_result;
    logic [XLEN-1:0]    divide_result;
    logic [XLEN-1:0]    remainder_result;
    logic               divide_by_zero;
    logic               operation_complete;
    
    // Pipeline registers for multi-cycle operations
    logic [2:0]         operation_reg;
    logic               is_32bit_reg;
    logic [XLEN-1:0]    operand_a_reg, operand_b_reg;
    logic [3:0]         cycle_counter;
    logic               operation_active;

    // ========================================
    // Input Processing
    // ========================================
    
    always_comb begin
        if (is_32bit) begin
            // 32-bit operations - sign extend from lower 32 bits
            operand_a = {{32{rs1_data[31]}}, rs1_data[31:0]};
            operand_b = {{32{rs2_data[31]}}, rs2_data[31:0]};
        end else begin
            // 64-bit operations
            operand_a = rs1_data;
            operand_b = rs2_data;
        end
    end

    // ========================================
    // Operation Control
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            operation_reg <= 3'b0;
            is_32bit_reg <= 1'b0;
            operand_a_reg <= '0;
            operand_b_reg <= '0;
            cycle_counter <= 4'b0;
            operation_active <= 1'b0;
        end else begin
            if (mdu_enable && !operation_active) begin
                // Start new operation
                operation_reg <= funct3;
                is_32bit_reg <= is_32bit;
                operand_a_reg <= operand_a;
                operand_b_reg <= operand_b;
                cycle_counter <= 4'b0;
                operation_active <= 1'b1;
            end else if (operation_active) begin
                // Continue operation
                cycle_counter <= cycle_counter + 1;
                if (operation_complete) begin
                    operation_active <= 1'b0;
                end
            end
        end
    end

    // ========================================
    // Multiplication Logic
    // ========================================
    
    logic [2*XLEN-1:0]  signed_multiply;
    logic [2*XLEN-1:0]  unsigned_multiply;
    logic [2*XLEN-1:0]  mixed_multiply;
    
    // Signed multiplication
    assign signed_multiply = $signed(operand_a_reg) * $signed(operand_b_reg);
    
    // Unsigned multiplication
    assign unsigned_multiply = operand_a_reg * operand_b_reg;
    
    // Mixed sign multiplication (signed * unsigned)
    assign mixed_multiply = $signed(operand_a_reg) * $signed({1'b0, operand_b_reg});
    
    always_comb begin
        case (operation_reg)
            FUNCT3_MUL: begin
                multiply_result = signed_multiply;
            end
            FUNCT3_MULH: begin
                multiply_result = signed_multiply;
            end
            FUNCT3_MULHSU: begin
                multiply_result = mixed_multiply;
            end
            FUNCT3_MULHU: begin
                multiply_result = unsigned_multiply;
            end
            default: begin
                multiply_result = signed_multiply;
            end
        endcase
    end

    // ========================================
    // Division Logic
    // ========================================
    
    logic [XLEN-1:0]    dividend, divisor;
    logic [XLEN-1:0]    quotient, remainder;
    logic               div_sign, rem_sign;
    logic [XLEN-1:0]    abs_dividend, abs_divisor;
    logic [XLEN-1:0]    unsigned_quotient, unsigned_remainder;
    
    assign dividend = operand_a_reg;
    assign divisor = operand_b_reg;
    assign divide_by_zero = (divisor == '0);
    
    // Sign handling for signed division
    assign div_sign = dividend[XLEN-1] ^ divisor[XLEN-1];
    assign rem_sign = dividend[XLEN-1];
    
    // Absolute values for signed operations
    assign abs_dividend = dividend[XLEN-1] ? (~dividend + 1) : dividend;
    assign abs_divisor = divisor[XLEN-1] ? (~divisor + 1) : divisor;
    
    // Simple division implementation (non-restoring division)
    // In a real implementation, this would be a more sophisticated algorithm
    always_comb begin
        if (divide_by_zero) begin
            unsigned_quotient = {XLEN{1'b1}};  // All 1s for divide by zero
            unsigned_remainder = dividend;
        end else begin
            case (operation_reg)
                FUNCT3_DIV, FUNCT3_REM: begin
                    // Signed division
                    unsigned_quotient = abs_dividend / abs_divisor;
                    unsigned_remainder = abs_dividend % abs_divisor;
                end
                FUNCT3_DIVU, FUNCT3_REMU: begin
                    // Unsigned division
                    unsigned_quotient = dividend / divisor;
                    unsigned_remainder = dividend % divisor;
                end
                default: begin
                    unsigned_quotient = '0;
                    unsigned_remainder = '0;
                end
            endcase
        end
    end
    
    // Apply signs for signed operations
    always_comb begin
        case (operation_reg)
            FUNCT3_DIV: begin
                quotient = div_sign ? (~unsigned_quotient + 1) : unsigned_quotient;
            end
            FUNCT3_REM: begin
                remainder = rem_sign ? (~unsigned_remainder + 1) : unsigned_remainder;
            end
            FUNCT3_DIVU: begin
                quotient = unsigned_quotient;
            end
            FUNCT3_REMU: begin
                remainder = unsigned_remainder;
            end
            default: begin
                quotient = unsigned_quotient;
                remainder = unsigned_remainder;
            end
        endcase
    end
    
    assign divide_result = quotient;
    assign remainder_result = remainder;

    // ========================================
    // Operation Timing Control
    // ========================================
    
    always_comb begin
        case (operation_reg)
            FUNCT3_MUL, FUNCT3_MULH, FUNCT3_MULHSU, FUNCT3_MULHU: begin
                // Multiplication completes in 1 cycle
                operation_complete = (cycle_counter >= 4'd0);
            end
            FUNCT3_DIV, FUNCT3_DIVU, FUNCT3_REM, FUNCT3_REMU: begin
                // Division takes multiple cycles (simplified to 4 cycles)
                operation_complete = (cycle_counter >= 4'd3);
            end
            default: begin
                operation_complete = 1'b1;
            end
        endcase
    end

    // ========================================
    // Result Selection and Output
    // ========================================
    
    always_comb begin
        case (operation_reg)
            FUNCT3_MUL: begin
                if (is_32bit_reg) begin
                    // MULW - sign extend 32-bit result
                    mdu_result = {{32{multiply_result[31]}}, multiply_result[31:0]};
                end else begin
                    // MUL - lower 64 bits
                    mdu_result = multiply_result[XLEN-1:0];
                end
            end
            
            FUNCT3_MULH: begin
                // MULH - upper 64 bits of signed multiplication
                mdu_result = multiply_result[2*XLEN-1:XLEN];
            end
            
            FUNCT3_MULHSU: begin
                // MULHSU - upper 64 bits of mixed sign multiplication
                mdu_result = multiply_result[2*XLEN-1:XLEN];
            end
            
            FUNCT3_MULHU: begin
                // MULHU - upper 64 bits of unsigned multiplication
                mdu_result = multiply_result[2*XLEN-1:XLEN];
            end
            
            FUNCT3_DIV: begin
                if (is_32bit_reg) begin
                    // DIVW - sign extend 32-bit result
                    mdu_result = {{32{divide_result[31]}}, divide_result[31:0]};
                end else begin
                    // DIV
                    mdu_result = divide_result;
                end
            end
            
            FUNCT3_DIVU: begin
                if (is_32bit_reg) begin
                    // DIVUW - sign extend 32-bit result
                    mdu_result = {{32{divide_result[31]}}, divide_result[31:0]};
                end else begin
                    // DIVU
                    mdu_result = divide_result;
                end
            end
            
            FUNCT3_REM: begin
                if (is_32bit_reg) begin
                    // REMW - sign extend 32-bit result
                    mdu_result = {{32{remainder_result[31]}}, remainder_result[31:0]};
                end else begin
                    // REM
                    mdu_result = remainder_result;
                end
            end
            
            FUNCT3_REMU: begin
                if (is_32bit_reg) begin
                    // REMUW - sign extend 32-bit result
                    mdu_result = {{32{remainder_result[31]}}, remainder_result[31:0]};
                end else begin
                    // REMU
                    mdu_result = remainder_result;
                end
            end
            
            default: begin
                mdu_result = '0;
            end
        endcase
    end
    
    assign mdu_ready = operation_complete;
    assign mdu_valid = operation_active && operation_complete;

endmodule