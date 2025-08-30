// TPU MAC (Multiply-Accumulate) Unit
// Supports INT8, FP16, and FP32 data types
// Part of the 64x64 systolic array

`timescale 1ns/1ps

module tpu_mac_unit #(
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    enable,
    
    // Data type selection
    input  logic [1:0]              data_type,  // 00: INT8, 01: FP16, 10: FP32
    
    // Input data
    input  logic [DATA_WIDTH-1:0]   a_in,       // Input A (activation)
    input  logic [DATA_WIDTH-1:0]   b_in,       // Input B (weight)
    input  logic [DATA_WIDTH-1:0]   c_in,       // Partial sum input
    
    // Output data
    output logic [DATA_WIDTH-1:0]   a_out,      // Pass-through A
    output logic [DATA_WIDTH-1:0]   b_out,      // Pass-through B
    output logic [DATA_WIDTH-1:0]   c_out,      // Accumulated result
    
    // Control signals
    input  logic                    load_weight,
    input  logic                    accumulate,
    output logic                    overflow,
    output logic                    underflow
);

    // Internal registers
    logic [DATA_WIDTH-1:0] weight_reg;
    logic [DATA_WIDTH-1:0] mult_result;
    logic [DATA_WIDTH-1:0] acc_result;
    
    // Data type specific processing
    logic [7:0]  a_int8, b_int8;
    logic [15:0] a_fp16, b_fp16;
    logic [31:0] a_fp32, b_fp32;
    
    logic [15:0] mult_int8;
    logic [15:0] mult_fp16;
    logic [31:0] mult_fp32;
    
    // Extract data based on type
    always_comb begin
        case (data_type)
            2'b00: begin // INT8
                a_int8 = a_in[7:0];
                b_int8 = weight_reg[7:0];
            end
            2'b01: begin // FP16
                a_fp16 = a_in[15:0];
                b_fp16 = weight_reg[15:0];
            end
            2'b10: begin // FP32
                a_fp32 = a_in[31:0];
                b_fp32 = weight_reg[31:0];
            end
            default: begin
                a_int8 = 8'h0;
                b_int8 = 8'h0;
            end
        endcase
    end
    
    // Multiplication logic
    always_comb begin
        case (data_type)
            2'b00: begin // INT8
                mult_int8 = $signed(a_int8) * $signed(b_int8);
                mult_result = {{16{mult_int8[15]}}, mult_int8};
            end
            2'b01: begin // FP16
                mult_fp16 = fp16_multiply(a_fp16, b_fp16);
                mult_result = {{16{1'b0}}, mult_fp16[15:0]};
            end
            2'b10: begin // FP32
                mult_fp32 = fp32_multiply(a_fp32, b_fp32);
                mult_result = mult_fp32;
            end
            default: mult_result = 32'h0;
        endcase
    end
    
    // Accumulation logic
    always_comb begin
        if (accumulate) begin
            case (data_type)
                2'b00: begin // INT8
                    acc_result = $signed(c_in) + $signed(mult_result);
                end
                2'b01: begin // FP16
                    acc_result = {{16{1'b0}}, fp16_add(c_in[15:0], mult_result[15:0])};
                end
                2'b10: begin // FP32
                    acc_result = fp32_add(c_in, mult_result);
                end
                default: acc_result = mult_result;
            endcase
        end else begin
            acc_result = mult_result;
        end
    end
    
    // Weight register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= '0;
        end else if (load_weight) begin
            weight_reg <= b_in;
        end
    end
    
    // Output registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out <= '0;
            b_out <= '0;
            c_out <= '0;
        end else if (enable) begin
            a_out <= a_in;
            b_out <= b_in;
            c_out <= acc_result;
        end
    end
    
    // Overflow/underflow detection
    always_comb begin
        overflow = 1'b0;
        underflow = 1'b0;
        
        case (data_type)
            2'b00: begin // INT8
                overflow = ($signed(acc_result) > $signed(32'h7FFFFFFF));
                underflow = ($signed(acc_result) < $signed(32'h80000000));
            end
            2'b01: begin // FP16
                // FP16 overflow/underflow detection
                overflow = fp16_is_overflow(acc_result[15:0]);
                underflow = fp16_is_underflow(acc_result[15:0]);
            end
            2'b10: begin // FP32
                // FP32 overflow/underflow detection
                overflow = fp32_is_overflow(acc_result);
                underflow = fp32_is_underflow(acc_result);
            end
            default: begin
                overflow = 1'b0;
                underflow = 1'b0;
            end
        endcase
    end
    
    // Simplified floating point helper functions for synthesis
    function automatic [31:0] fp32_multiply(input [31:0] a, input [31:0] b);
        // Simplified implementation - in production would use dedicated FPU
        logic [63:0] temp_result;
        temp_result = a * b;  // Simplified multiplication
        fp32_multiply = temp_result[31:0];
    endfunction
    
    function automatic [31:0] fp32_add(input [31:0] a, input [31:0] b);
        // Simplified implementation - in production would use dedicated FPU
        fp32_add = a + b;
    endfunction
    
    function automatic [15:0] fp16_multiply(input [15:0] a, input [15:0] b);
        // Simplified implementation - in production would use dedicated FPU
        logic [31:0] temp_result;
        temp_result = a * b;
        fp16_multiply = temp_result[15:0];
    endfunction
    
    function automatic [15:0] fp16_add(input [15:0] a, input [15:0] b);
        // Simplified implementation - in production would use dedicated FPU
        fp16_add = a + b;
    endfunction
    
    function automatic logic fp32_is_overflow(input [31:0] val);
        fp32_is_overflow = (val[30:23] == 8'hFF) && (val[22:0] == 23'h0);
    endfunction
    
    function automatic logic fp32_is_underflow(input [31:0] val);
        fp32_is_underflow = (val[30:23] == 8'h00) && (val[22:0] != 23'h0);
    endfunction
    
    function automatic logic fp16_is_overflow(input [15:0] val);
        fp16_is_overflow = (val[14:10] == 5'h1F) && (val[9:0] == 10'h0);
    endfunction
    
    function automatic logic fp16_is_underflow(input [15:0] val);
        fp16_is_underflow = (val[14:10] == 5'h00) && (val[9:0] != 10'h0);
    endfunction

endmodule
