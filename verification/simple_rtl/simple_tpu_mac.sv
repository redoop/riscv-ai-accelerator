// Simplified TPU MAC unit for Icarus Verilog compatibility
`timescale 1ns/1ps

module simple_tpu_mac #(
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    enable,
    input  logic [2:0]              data_type,
    input  logic [15:0]             a_data,
    input  logic [15:0]             b_data,
    input  logic [31:0]             c_data,
    input  logic                    valid_in,
    
    output logic [31:0]             result,
    output logic                    valid_out,
    output logic                    ready
);

    // Internal registers
    logic [31:0] mac_result;
    logic        result_valid;
    
    // MAC operation: result = a * b + c
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_result <= 32'h0;
            result_valid <= 1'b0;
        end else if (enable && valid_in) begin
            // Perform MAC operation based on data type
            case (data_type)
                3'b000: begin // INT8
                    mac_result <= (a_data[7:0] * b_data[7:0]) + c_data;
                end
                3'b001: begin // INT16
                    mac_result <= (a_data * b_data) + c_data;
                end
                3'b010: begin // INT32
                    mac_result <= (a_data * b_data) + c_data;
                end
                default: begin
                    mac_result <= (a_data * b_data) + c_data;
                end
            endcase
            result_valid <= 1'b1;
        end else begin
            result_valid <= 1'b0;
        end
    end
    
    // Output assignments
    assign result = mac_result;
    assign valid_out = result_valid;
    assign ready = enable;

endmodule