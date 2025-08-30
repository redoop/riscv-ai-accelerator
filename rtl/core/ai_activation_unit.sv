// AI Activation Function Unit
// Implements ReLU, Sigmoid, and Tanh activation functions

`timescale 1ns/1ps

module ai_activation_unit #(
    parameter XLEN = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic [2:0]              activation_type, // 3'b100=ReLU, 3'b101=Sigmoid, 3'b110=Tanh
    input  logic [2:0]              data_type,
    
    // Data interface
    input  logic [XLEN-1:0]         input_data,
    output logic [XLEN-1:0]         result,
    output logic                    valid,
    
    // Error flags
    output logic                    overflow,
    output logic                    underflow
);

    // Activation function types
    localparam [2:0] RELU    = 3'b100;
    localparam [2:0] SIGMOID = 3'b101;
    localparam [2:0] TANH    = 3'b110;
    
    // Internal registers
    logic [DATA_WIDTH-1:0] input_fp;
    logic [DATA_WIDTH-1:0] result_fp;
    logic                  computation_done;
    
    // Pipeline stages for complex functions
    logic [DATA_WIDTH-1:0] stage1_result;
    logic [DATA_WIDTH-1:0] stage2_result;
    logic [DATA_WIDTH-1:0] stage3_result;
    
    // Extract input based on data type
    always_comb begin
        case (data_type)
            3'b101: input_fp = input_data[31:0]; // FP32
            3'b010: input_fp = input_data[31:0]; // INT32
            default: input_fp = input_data[31:0];
        endcase
    end
    
    // ReLU Implementation (Combinational)
    logic [DATA_WIDTH-1:0] relu_result;
    always_comb begin
        case (data_type)
            3'b101: begin // FP32
                // Check sign bit for FP32
                if (input_fp[31] == 1'b1) begin // Negative
                    relu_result = 32'h00000000; // +0.0
                end else begin
                    relu_result = input_fp;
                end
            end
            3'b010: begin // INT32
                if (input_fp[31] == 1'b1) begin // Negative
                    relu_result = 32'h00000000; // 0
                end else begin
                    relu_result = input_fp;
                end
            end
            default: begin
                relu_result = (input_fp[31] == 1'b1) ? 32'h00000000 : input_fp;
            end
        endcase
    end
    
    // Sigmoid Implementation (Pipelined approximation)
    // Using polynomial approximation: sigmoid(x) ≈ 0.5 + 0.25*x for small x
    // For larger x, use saturation
    logic [DATA_WIDTH-1:0] sigmoid_result;
    logic [DATA_WIDTH-1:0] abs_input;
    logic input_sign;
    
    always_comb begin
        input_sign = input_fp[31];
        abs_input = input_sign ? (~input_fp + 1) : input_fp;
    end
    
    // Sigmoid approximation pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_result <= '0;
            stage2_result <= '0;
            stage3_result <= '0;
        end else if (enable && activation_type == SIGMOID) begin
            // Stage 1: Check magnitude
            if (data_type == 3'b101) begin // FP32
                // Simplified: if |x| > 5.0, saturate
                if (abs_input > 32'h40A00000) begin // 5.0 in FP32
                    stage1_result <= input_sign ? 32'h00000000 : 32'h3F800000; // 0.0 or 1.0
                end else begin
                    stage1_result <= input_fp;
                end
            end else begin // INT32
                if (abs_input > 32'd5) begin
                    stage1_result <= input_sign ? 32'h00000000 : 32'h00000001;
                end else begin
                    stage1_result <= input_fp;
                end
            end
            
            // Stage 2: Polynomial approximation
            // sigmoid(x) ≈ 0.5 + 0.25*x (for |x| <= 2)
            if (data_type == 3'b101) begin // FP32
                // Simplified FP32 arithmetic (would use dedicated FPU)
                stage2_result <= 32'h3F000000 + (stage1_result >> 2); // 0.5 + x/4
            end else begin // INT32
                stage2_result <= 32'h00000000 + (stage1_result >>> 2); // Simplified
            end
            
            // Stage 3: Final result
            stage3_result <= stage2_result;
        end
    end
    
    assign sigmoid_result = stage3_result;
    
    // Tanh Implementation (Pipelined approximation)
    // Using approximation: tanh(x) ≈ x for small x, ±1 for large x
    logic [DATA_WIDTH-1:0] tanh_result;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tanh_result <= '0;
        end else if (enable && activation_type == TANH) begin
            if (data_type == 3'b101) begin // FP32
                // Simplified: if |x| > 2.0, saturate to ±1.0
                if (abs_input > 32'h40000000) begin // 2.0 in FP32
                    tanh_result <= input_sign ? 32'hBF800000 : 32'h3F800000; // -1.0 or 1.0
                end else begin
                    // Linear approximation for small values
                    tanh_result <= input_fp;
                end
            end else begin // INT32
                if (abs_input > 32'd2) begin
                    tanh_result <= input_sign ? 32'hFFFFFFFF : 32'h00000001; // -1 or 1
                end else begin
                    tanh_result <= input_fp;
                end
            end
        end
    end
    
    // Result multiplexing and timing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_fp <= '0;
            computation_done <= 1'b0;
            overflow <= 1'b0;
            underflow <= 1'b0;
        end else if (enable) begin
            case (activation_type)
                RELU: begin
                    result_fp <= relu_result;
                    computation_done <= 1'b1;
                    overflow <= 1'b0;
                    underflow <= 1'b0;
                end
                
                SIGMOID: begin
                    result_fp <= sigmoid_result;
                    computation_done <= 1'b1;
                    // Check for overflow/underflow in FP operations
                    overflow <= (data_type == 3'b101) && (sigmoid_result == 32'h7F800000); // +inf
                    underflow <= (data_type == 3'b101) && (sigmoid_result == 32'h00000000); // 0
                end
                
                TANH: begin
                    result_fp <= tanh_result;
                    computation_done <= 1'b1;
                    overflow <= (data_type == 3'b101) && (tanh_result == 32'h7F800000); // +inf
                    underflow <= (data_type == 3'b101) && (tanh_result == 32'hFF800000); // -inf
                end
                
                default: begin
                    result_fp <= input_fp; // Pass through
                    computation_done <= 1'b1;
                    overflow <= 1'b0;
                    underflow <= 1'b0;
                end
            endcase
        end else begin
            computation_done <= 1'b0;
            overflow <= 1'b0;
            underflow <= 1'b0;
        end
    end
    
    // Output assignment
    assign result = {32'b0, result_fp};
    assign valid = computation_done;

endmodule