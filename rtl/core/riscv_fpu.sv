// RISC-V Floating Point Unit (FPU)
// Implements RV64F (single precision) and RV64D (double precision) extensions

`timescale 1ns/1ps

module riscv_fpu #(
    parameter XLEN = 64
) (
    input  logic                clk,
    input  logic                rst_n,
    
    // Control interface
    input  logic                fpu_enable,
    input  logic [2:0]          fpu_op,
    input  logic [2:0]          funct3,
    input  logic [6:0]          funct7,
    input  logic                is_double,  // 1 for double precision, 0 for single
    
    // Data interface
    input  logic [XLEN-1:0]     rs1_data,
    input  logic [XLEN-1:0]     rs2_data,
    input  logic [XLEN-1:0]     rs3_data,   // For fused operations
    
    output logic [XLEN-1:0]     fpu_result,
    output logic                fpu_ready,
    output logic [4:0]          fpu_flags   // Exception flags
);

    // FPU operation codes
    localparam [2:0] FPU_ADD    = 3'b000;
    localparam [2:0] FPU_SUB    = 3'b001;
    localparam [2:0] FPU_MUL    = 3'b010;
    localparam [2:0] FPU_DIV    = 3'b011;
    localparam [2:0] FPU_SQRT   = 3'b100;
    localparam [2:0] FPU_MISC   = 3'b101;
    localparam [2:0] FPU_CMP    = 3'b110;
    localparam [2:0] FPU_CVT    = 3'b111;

    // Exception flags
    localparam [4:0] FLAG_NV = 5'b10000;  // Invalid operation
    localparam [4:0] FLAG_DZ = 5'b01000;  // Divide by zero
    localparam [4:0] FLAG_OF = 5'b00100;  // Overflow
    localparam [4:0] FLAG_UF = 5'b00010;  // Underflow
    localparam [4:0] FLAG_NX = 5'b00001;  // Inexact

    // Internal signals
    logic [63:0]    fp_a, fp_b, fp_c;
    logic [63:0]    fp_result;
    logic [4:0]     exception_flags;
    logic           operation_valid;
    
    // Single precision extraction/packing
    logic [31:0]    sp_a, sp_b, sp_c, sp_result;
    logic [63:0]    dp_a, dp_b, dp_c, dp_result;
    
    // Rounding mode (from funct3 for most operations)
    logic [2:0]     rm;
    
    assign rm = funct3;

    // ========================================
    // Input Data Preparation
    // ========================================
    
    always_comb begin
        if (is_double) begin
            // Double precision - use full 64 bits
            fp_a = rs1_data;
            fp_b = rs2_data;
            fp_c = rs3_data;
        end else begin
            // Single precision - extract from lower 32 bits and NaN-box
            sp_a = rs1_data[31:0];
            sp_b = rs2_data[31:0];
            sp_c = rs3_data[31:0];
            
            // Convert to double precision for internal computation
            fp_a = {sp_a[31], {11{sp_a[30]}}, sp_a[29:23], sp_a[22:0], 29'b0};
            fp_b = {sp_b[31], {11{sp_b[30]}}, sp_b[29:23], sp_b[22:0], 29'b0};
            fp_c = {sp_c[31], {11{sp_c[30]}}, sp_c[29:23], sp_c[22:0], 29'b0};
        end
    end

    // ========================================
    // Floating Point Operations
    // ========================================
    
    always_comb begin
        fp_result = 64'b0;
        exception_flags = 5'b0;
        operation_valid = 1'b1;
        
        if (fpu_enable) begin
            case (fpu_op)
                FPU_ADD: begin
                    // Floating point addition
                    fp_result = fp_add(fp_a, fp_b, rm);
                end
                
                FPU_SUB: begin
                    // Floating point subtraction
                    fp_result = fp_sub(fp_a, fp_b, rm);
                end
                
                FPU_MUL: begin
                    // Floating point multiplication
                    fp_result = fp_mul(fp_a, fp_b, rm);
                end
                
                FPU_DIV: begin
                    // Floating point division
                    fp_result = fp_div(fp_a, fp_b, rm);
                end
                
                FPU_SQRT: begin
                    // Floating point square root
                    fp_result = fp_sqrt(fp_a, rm);
                end
                
                FPU_MISC: begin
                    // Miscellaneous operations (FSGNJ, FMIN, FMAX, etc.)
                    case (funct3)
                        3'b000: fp_result = {fp_b[63], fp_a[62:0]};  // FSGNJ
                        3'b001: fp_result = {~fp_b[63], fp_a[62:0]}; // FSGNJN
                        3'b010: fp_result = {fp_a[63] ^ fp_b[63], fp_a[62:0]}; // FSGNJX
                        3'b000: fp_result = fp_min(fp_a, fp_b);      // FMIN (funct7 dependent)
                        3'b001: fp_result = fp_max(fp_a, fp_b);      // FMAX (funct7 dependent)
                        default: begin
                            fp_result = 64'b0;
                            exception_flags = FLAG_NV;
                        end
                    endcase
                end
                
                FPU_CMP: begin
                    // Floating point comparison
                    case (funct3)
                        3'b010: fp_result = {63'b0, fp_eq(fp_a, fp_b)};  // FEQ
                        3'b001: fp_result = {63'b0, fp_lt(fp_a, fp_b)};  // FLT
                        3'b000: fp_result = {63'b0, fp_le(fp_a, fp_b)};  // FLE
                        default: begin
                            fp_result = 64'b0;
                            exception_flags = FLAG_NV;
                        end
                    endcase
                end
                
                FPU_CVT: begin
                    // Floating point conversion
                    case (funct7[6:2])
                        5'b11000: fp_result = fp_to_int(fp_a, rm, funct7[0]); // FCVT.W/L.S/D
                        5'b11010: fp_result = int_to_fp(fp_a, rm, funct7[0]); // FCVT.S/D.W/L
                        5'b01000: fp_result = sp_to_dp(fp_a[31:0]);           // FCVT.D.S
                        5'b01000: fp_result = dp_to_sp(fp_a);                 // FCVT.S.D
                        default: begin
                            fp_result = 64'b0;
                            exception_flags = FLAG_NV;
                        end
                    endcase
                end
                
                default: begin
                    fp_result = 64'b0;
                    exception_flags = FLAG_NV;
                    operation_valid = 1'b0;
                end
            endcase
        end
    end

    // ========================================
    // Output Data Formatting
    // ========================================
    
    always_comb begin
        if (is_double) begin
            // Double precision result
            fpu_result = fp_result;
        end else begin
            // Single precision result - NaN-box in upper 32 bits
            sp_result = fp_result[31:0];
            fpu_result = {32'hFFFF_FFFF, sp_result};
        end
    end
    
    assign fpu_flags = exception_flags;
    assign fpu_ready = 1'b1; // Combinational implementation for now

    // ========================================
    // Floating Point Arithmetic Functions
    // ========================================
    
    function automatic logic [63:0] fp_add(input logic [63:0] a, b, input logic [2:0] rm);
        // Simplified floating point addition
        // In a real implementation, this would handle special cases, normalization, etc.
        logic sign_a, sign_b, sign_result;
        logic [10:0] exp_a, exp_b, exp_result;
        logic [51:0] mant_a, mant_b, mant_result;
        
        sign_a = a[63];
        sign_b = b[63];
        exp_a = a[62:52];
        exp_b = b[62:52];
        mant_a = a[51:0];
        mant_b = b[51:0];
        
        // Simplified: assume same exponent for now
        if (exp_a == exp_b) begin
            if (sign_a == sign_b) begin
                mant_result = mant_a + mant_b;
                sign_result = sign_a;
            end else begin
                if (mant_a >= mant_b) begin
                    mant_result = mant_a - mant_b;
                    sign_result = sign_a;
                end else begin
                    mant_result = mant_b - mant_a;
                    sign_result = sign_b;
                end
            end
            exp_result = exp_a;
        end else begin
            // For now, return the larger operand
            fp_add = (exp_a > exp_b) ? a : b;
        end
        
        fp_add = {sign_result, exp_result, mant_result};
    endfunction
    
    function automatic logic [63:0] fp_sub(input logic [63:0] a, b, input logic [2:0] rm);
        // Floating point subtraction = addition with negated b
        logic [63:0] neg_b;
        neg_b = {~b[63], b[62:0]};
        fp_sub = fp_add(a, neg_b, rm);
    endfunction
    
    function automatic logic [63:0] fp_mul(input logic [63:0] a, b, input logic [2:0] rm);
        // Simplified floating point multiplication
        logic sign_result;
        logic [10:0] exp_result;
        logic [51:0] mant_result;
        
        sign_result = a[63] ^ b[63];
        exp_result = a[62:52] + b[62:52] - 11'd1023; // Subtract bias
        mant_result = a[51:0]; // Simplified
        
        fp_mul = {sign_result, exp_result, mant_result};
    endfunction
    
    function automatic logic [63:0] fp_div(input logic [63:0] a, b, input logic [2:0] rm);
        // Simplified floating point division
        logic sign_result;
        logic [10:0] exp_result;
        logic [51:0] mant_result;
        
        sign_result = a[63] ^ b[63];
        exp_result = a[62:52] - b[62:52] + 11'd1023; // Add bias
        mant_result = a[51:0]; // Simplified
        
        fp_div = {sign_result, exp_result, mant_result};
    endfunction
    
    function automatic logic [63:0] fp_sqrt(input logic [63:0] a, input logic [2:0] rm);
        // Simplified square root - return input for now
        fp_sqrt = a;
    endfunction
    
    function automatic logic [63:0] fp_min(input logic [63:0] a, b);
        // Return minimum of two floating point numbers
        if (fp_lt(a, b)) fp_min = a;
        else fp_min = b;
    endfunction
    
    function automatic logic [63:0] fp_max(input logic [63:0] a, b);
        // Return maximum of two floating point numbers
        if (fp_lt(a, b)) fp_max = b;
        else fp_max = a;
    endfunction
    
    function automatic logic fp_eq(input logic [63:0] a, b);
        // Floating point equality comparison
        fp_eq = (a == b);
    endfunction
    
    function automatic logic fp_lt(input logic [63:0] a, b);
        // Floating point less than comparison
        if (a[63] != b[63]) begin
            fp_lt = a[63]; // Different signs
        end else begin
            if (a[63]) begin
                fp_lt = (a[62:0] > b[62:0]); // Both negative
            end else begin
                fp_lt = (a[62:0] < b[62:0]); // Both positive
            end
        end
    endfunction
    
    function automatic logic fp_le(input logic [63:0] a, b);
        // Floating point less than or equal comparison
        fp_le = fp_lt(a, b) || fp_eq(a, b);
    endfunction
    
    function automatic logic [63:0] fp_to_int(input logic [63:0] a, input logic [2:0] rm, input logic is_long);
        // Convert floating point to integer
        // Simplified implementation
        fp_to_int = a[63:0]; // Pass through for now
    endfunction
    
    function automatic logic [63:0] int_to_fp(input logic [63:0] a, input logic [2:0] rm, input logic is_long);
        // Convert integer to floating point
        // Simplified implementation
        int_to_fp = a; // Pass through for now
    endfunction
    
    function automatic logic [63:0] sp_to_dp(input logic [31:0] sp);
        // Convert single precision to double precision
        logic sign;
        logic [7:0] sp_exp;
        logic [22:0] sp_mant;
        logic [10:0] dp_exp;
        logic [51:0] dp_mant;
        
        sign = sp[31];
        sp_exp = sp[30:23];
        sp_mant = sp[22:0];
        
        // Convert exponent (add bias difference)
        dp_exp = {3'b0, sp_exp} + 11'd896; // 1023 - 127 = 896
        
        // Extend mantissa
        dp_mant = {sp_mant, 29'b0};
        
        sp_to_dp = {sign, dp_exp, dp_mant};
    endfunction
    
    function automatic logic [31:0] dp_to_sp(input logic [63:0] dp);
        // Convert double precision to single precision
        logic sign;
        logic [10:0] dp_exp;
        logic [51:0] dp_mant;
        logic [7:0] sp_exp;
        logic [22:0] sp_mant;
        
        sign = dp[63];
        dp_exp = dp[62:52];
        dp_mant = dp[51:0];
        
        // Convert exponent (subtract bias difference)
        sp_exp = dp_exp[7:0] - 8'd127; // Corrected bias difference
        
        // Truncate mantissa
        sp_mant = dp_mant[51:29];
        
        dp_to_sp = {sign, sp_exp, sp_mant};
    endfunction

endmodule