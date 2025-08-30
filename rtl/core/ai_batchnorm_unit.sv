// AI Batch Normalization Unit
// Implements batch normalization: y = (x - mean) / sqrt(variance + epsilon) * scale + bias

`timescale 1ns/1ps

module ai_batchnorm_unit #(
    parameter XLEN = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic [2:0]              data_type,
    
    // Batch normalization parameters
    input  logic [XLEN-1:0]         input_addr,
    input  logic [XLEN-1:0]         scale_addr,
    input  logic [XLEN-1:0]         bias_addr,
    
    // Memory interface
    output logic [XLEN-1:0]         mem_addr,
    output logic [XLEN-1:0]         mem_wdata,
    output logic [7:0]              mem_wmask,
    output logic                    mem_req,
    output logic                    mem_we,
    input  logic [XLEN-1:0]         mem_rdata,
    input  logic                    mem_ready,
    
    // Result interface
    output logic [XLEN-1:0]         result,
    output logic                    valid
);

    // Batch normalization constants
    localparam [DATA_WIDTH-1:0] EPSILON = 32'h3A83126F; // 1e-3 in FP32
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_INPUT,
        LOAD_SCALE,
        LOAD_BIAS,
        LOAD_MEAN,
        LOAD_VARIANCE,
        COMPUTE_NORM,
        COMPUTE_SCALE,
        COMPUTE_BIAS,
        STORE_OUTPUT,
        DONE
    } batchnorm_state_t;
    
    batchnorm_state_t current_state, next_state;
    
    // Data registers
    logic [DATA_WIDTH-1:0] input_value;
    logic [DATA_WIDTH-1:0] scale_value;
    logic [DATA_WIDTH-1:0] bias_value;
    logic [DATA_WIDTH-1:0] mean_value;
    logic [DATA_WIDTH-1:0] variance_value;
    
    // Computation registers
    logic [DATA_WIDTH-1:0] normalized_value;
    logic [DATA_WIDTH-1:0] scaled_value;
    logic [DATA_WIDTH-1:0] output_value;
    
    // Intermediate computation values
    logic [DATA_WIDTH-1:0] diff_value;        // input - mean
    logic [DATA_WIDTH-1:0] var_plus_eps;      // variance + epsilon
    logic [DATA_WIDTH-1:0] sqrt_var_eps;      // sqrt(variance + epsilon)
    logic [DATA_WIDTH-1:0] inv_sqrt_var_eps;  // 1 / sqrt(variance + epsilon)
    
    // Processing counter (for batch processing)
    logic [15:0] element_count;
    logic [15:0] total_elements;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (enable) begin
                    next_state = LOAD_INPUT;
                end
            end
            
            LOAD_INPUT: begin
                if (mem_ready) begin
                    next_state = LOAD_SCALE;
                end
            end
            
            LOAD_SCALE: begin
                if (mem_ready) begin
                    next_state = LOAD_BIAS;
                end
            end
            
            LOAD_BIAS: begin
                if (mem_ready) begin
                    next_state = LOAD_MEAN;
                end
            end
            
            LOAD_MEAN: begin
                if (mem_ready) begin
                    next_state = LOAD_VARIANCE;
                end
            end
            
            LOAD_VARIANCE: begin
                if (mem_ready) begin
                    next_state = COMPUTE_NORM;
                end
            end
            
            COMPUTE_NORM: begin
                next_state = COMPUTE_SCALE;
            end
            
            COMPUTE_SCALE: begin
                next_state = COMPUTE_BIAS;
            end
            
            COMPUTE_BIAS: begin
                next_state = STORE_OUTPUT;
            end
            
            STORE_OUTPUT: begin
                if (mem_ready) begin
                    if (element_count >= total_elements - 1) begin
                        next_state = DONE;
                    end else begin
                        next_state = LOAD_INPUT;
                    end
                end
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Element counter management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            element_count <= '0;
            total_elements <= 16'd1; // Default to single element
        end else begin
            case (current_state)
                IDLE: begin
                    element_count <= '0;
                    // In a real implementation, total_elements would be passed as parameter
                    total_elements <= 16'd1;
                end
                
                STORE_OUTPUT: begin
                    if (mem_ready) begin
                        element_count <= element_count + 1;
                    end
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Memory interface
    always_comb begin
        case (current_state)
            LOAD_INPUT: begin
                mem_addr = input_addr + (element_count * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            LOAD_SCALE: begin
                // Scale is typically per-channel, so use element_count % channels
                mem_addr = scale_addr + ((element_count % 16) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            LOAD_BIAS: begin
                // Bias is typically per-channel, so use element_count % channels
                mem_addr = bias_addr + ((element_count % 16) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            LOAD_MEAN: begin
                // Mean is typically per-channel
                mem_addr = input_addr + 16'h1000 + ((element_count % 16) * (DATA_WIDTH/8)); // Offset for mean
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            LOAD_VARIANCE: begin
                // Variance is typically per-channel
                mem_addr = input_addr + 16'h2000 + ((element_count % 16) * (DATA_WIDTH/8)); // Offset for variance
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            STORE_OUTPUT: begin
                mem_addr = input_addr + (element_count * (DATA_WIDTH/8)); // Overwrite input
                mem_req = 1'b1;
                mem_we = 1'b1;
                mem_wmask = (DATA_WIDTH == 32) ? 8'b1111 : 8'b11111111;
                mem_wdata = {32'b0, output_value};
            end
            
            default: begin
                mem_addr = '0;
                mem_req = 1'b0;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
        endcase
    end
    
    // Data loading
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_value <= '0;
            scale_value <= '0;
            bias_value <= '0;
            mean_value <= '0;
            variance_value <= '0;
        end else begin
            case (current_state)
                LOAD_INPUT: begin
                    if (mem_ready) begin
                        input_value <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                LOAD_SCALE: begin
                    if (mem_ready) begin
                        scale_value <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                LOAD_BIAS: begin
                    if (mem_ready) begin
                        bias_value <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                LOAD_MEAN: begin
                    if (mem_ready) begin
                        mean_value <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                LOAD_VARIANCE: begin
                    if (mem_ready) begin
                        variance_value <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Batch normalization computation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff_value <= '0;
            var_plus_eps <= '0;
            sqrt_var_eps <= '0;
            inv_sqrt_var_eps <= '0;
            normalized_value <= '0;
            scaled_value <= '0;
            output_value <= '0;
        end else begin
            case (current_state)
                COMPUTE_NORM: begin
                    // Step 1: Compute (input - mean)
                    case (data_type)
                        3'b101: begin // FP32
                            diff_value <= input_value - mean_value;
                            var_plus_eps <= variance_value + EPSILON;
                        end
                        3'b010: begin // INT32
                            diff_value <= input_value - mean_value;
                            var_plus_eps <= variance_value + 32'd1000; // Scaled epsilon for integer
                        end
                        default: begin
                            diff_value <= input_value - mean_value;
                            var_plus_eps <= variance_value + EPSILON;
                        end
                    endcase
                    
                    // Step 2: Compute sqrt(variance + epsilon) - simplified
                    // In real implementation, would use dedicated square root unit
                    case (data_type)
                        3'b101: begin // FP32
                            // Simplified square root approximation
                            sqrt_var_eps <= var_plus_eps; // Placeholder - would use proper sqrt
                            inv_sqrt_var_eps <= 32'h3F800000; // 1.0 - placeholder
                        end
                        3'b010: begin // INT32
                            sqrt_var_eps <= var_plus_eps; // Placeholder
                            inv_sqrt_var_eps <= 32'h00000001; // 1 - placeholder
                        end
                        default: begin
                            sqrt_var_eps <= var_plus_eps;
                            inv_sqrt_var_eps <= 32'h3F800000;
                        end
                    endcase
                    
                    // Step 3: Normalize
                    normalized_value <= diff_value; // Simplified - would divide by sqrt
                end
                
                COMPUTE_SCALE: begin
                    // Step 4: Apply scale
                    case (data_type)
                        3'b101: begin // FP32
                            scaled_value <= normalized_value * scale_value;
                        end
                        3'b010: begin // INT32
                            scaled_value <= normalized_value * scale_value;
                        end
                        default: begin
                            scaled_value <= normalized_value * scale_value;
                        end
                    endcase
                end
                
                COMPUTE_BIAS: begin
                    // Step 5: Apply bias
                    case (data_type)
                        3'b101: begin // FP32
                            output_value <= scaled_value + bias_value;
                        end
                        3'b010: begin // INT32
                            output_value <= scaled_value + bias_value;
                        end
                        default: begin
                            output_value <= scaled_value + bias_value;
                        end
                    endcase
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Output control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= '0;
            valid <= 1'b0;
        end else begin
            case (current_state)
                DONE: begin
                    result <= input_addr; // Return input tensor address (modified in-place)
                    valid <= 1'b1;
                end
                default: begin
                    result <= '0;
                    valid <= 1'b0;
                end
            endcase
        end
    end

endmodule