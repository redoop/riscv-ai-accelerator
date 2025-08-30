// AI Pooling Unit
// Implements Max Pooling and Average Pooling operations

`timescale 1ns/1ps

module ai_pooling_unit #(
    parameter XLEN = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic                    pool_type, // 0=max, 1=average
    input  logic [2:0]              data_type,
    
    // Pooling parameters
    input  logic [XLEN-1:0]         input_addr,
    input  logic [XLEN-1:0]         output_addr,
    input  logic [31:0]             pool_params, // Packed parameters
    
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

    // Pool parameters extraction
    // pool_params format: [in_h:8][in_w:8][pool_h:4][pool_w:4][stride_h:4][stride_w:4]
    logic [7:0] input_height, input_width;
    logic [3:0] pool_height, pool_width;
    logic [3:0] stride_h, stride_w;
    
    assign input_height = pool_params[31:24];
    assign input_width = pool_params[23:16];
    assign pool_height = pool_params[15:12];
    assign pool_width = pool_params[11:8];
    assign stride_h = pool_params[7:4];
    assign stride_w = pool_params[3:0];
    
    // Calculated output dimensions
    logic [7:0] output_height, output_width;
    assign output_height = (input_height - pool_height) / stride_h + 1;
    assign output_width = (input_width - pool_width) / stride_w + 1;
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        POOL_OUTER_Y,
        POOL_OUTER_X,
        POOL_INNER_Y,
        POOL_INNER_X,
        LOAD_INPUT,
        COMPUTE,
        STORE_OUTPUT,
        DONE
    } pool_state_t;
    
    pool_state_t current_state, next_state;
    
    // Loop counters
    logic [7:0] out_y, out_x;     // Output position
    logic [3:0] pool_y, pool_x;   // Pool window position
    logic [7:0] in_y, in_x;       // Input position
    
    // Computation registers
    logic [DATA_WIDTH-1:0] input_element;
    logic [DATA_WIDTH-1:0] max_value;
    logic [DATA_WIDTH-1:0] sum_value;
    logic [7:0]            element_count;
    logic [DATA_WIDTH-1:0] avg_value;
    logic [DATA_WIDTH-1:0] output_element;
    
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
                    next_state = POOL_OUTER_Y;
                end
            end
            
            POOL_OUTER_Y: begin
                if (out_y < output_height) begin
                    next_state = POOL_OUTER_X;
                end else begin
                    next_state = DONE;
                end
            end
            
            POOL_OUTER_X: begin
                if (out_x < output_width) begin
                    next_state = POOL_INNER_Y;
                end else begin
                    next_state = POOL_OUTER_Y;
                end
            end
            
            POOL_INNER_Y: begin
                if (pool_y < pool_height) begin
                    next_state = POOL_INNER_X;
                end else begin
                    next_state = STORE_OUTPUT;
                end
            end
            
            POOL_INNER_X: begin
                if (pool_x < pool_width) begin
                    next_state = LOAD_INPUT;
                end else begin
                    next_state = POOL_INNER_Y;
                end
            end
            
            LOAD_INPUT: begin
                if (mem_ready) begin
                    next_state = COMPUTE;
                end
            end
            
            COMPUTE: begin
                next_state = POOL_INNER_X;
            end
            
            STORE_OUTPUT: begin
                if (mem_ready) begin
                    next_state = POOL_OUTER_X;
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
    
    // Counter management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_y <= '0;
            out_x <= '0;
            pool_y <= '0;
            pool_x <= '0;
        end else begin
            case (current_state)
                IDLE: begin
                    out_y <= '0;
                    out_x <= '0;
                    pool_y <= '0;
                    pool_x <= '0;
                end
                
                POOL_OUTER_Y: begin
                    if (out_y >= output_height) begin
                        out_y <= '0;
                    end else begin
                        out_x <= '0;
                        pool_y <= '0;
                    end
                end
                
                POOL_OUTER_X: begin
                    if (out_x >= output_width) begin
                        out_x <= '0;
                        out_y <= out_y + stride_h;
                    end else begin
                        pool_y <= '0;
                        // Initialize pooling values
                        if (pool_type == 1'b0) begin // Max pooling
                            max_value <= (data_type == 3'b101) ? 32'hFF800000 : 32'h80000000; // -inf or min int
                        end else begin // Average pooling
                            sum_value <= '0;
                            element_count <= '0;
                        end
                    end
                end
                
                POOL_INNER_Y: begin
                    if (pool_y >= pool_height) begin
                        pool_y <= '0;
                    end else begin
                        pool_x <= '0;
                    end
                end
                
                POOL_INNER_X: begin
                    if (pool_x >= pool_width) begin
                        pool_x <= '0;
                        pool_y <= pool_y + 1;
                    end
                end
                
                COMPUTE: begin
                    pool_x <= pool_x + 1;
                    if (pool_type == 1'b1) begin // Average pooling
                        element_count <= element_count + 1;
                    end
                end
                
                STORE_OUTPUT: begin
                    out_x <= out_x + stride_w;
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Input position calculation
    always_comb begin
        in_y = out_y * stride_h + pool_y;
        in_x = out_x * stride_w + pool_x;
    end
    
    // Memory interface
    always_comb begin
        case (current_state)
            LOAD_INPUT: begin
                mem_addr = input_addr + ((in_y * input_width + in_x) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            STORE_OUTPUT: begin
                mem_addr = output_addr + ((out_y * output_width + out_x) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b1;
                mem_wmask = (DATA_WIDTH == 32) ? 8'b1111 : 8'b11111111;
                mem_wdata = {32'b0, output_element};
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
    
    // Pooling computation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_element <= '0;
            max_value <= '0;
            sum_value <= '0;
            avg_value <= '0;
            output_element <= '0;
        end else begin
            case (current_state)
                LOAD_INPUT: begin
                    if (mem_ready) begin
                        input_element <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                COMPUTE: begin
                    if (pool_type == 1'b0) begin // Max pooling
                        case (data_type)
                            3'b101: begin // FP32
                                // Simplified FP32 comparison (would use dedicated comparator)
                                if (input_element > max_value || 
                                    (pool_y == 0 && pool_x == 0)) begin
                                    max_value <= input_element;
                                end
                            end
                            3'b010: begin // INT32
                                if ($signed(input_element) > $signed(max_value) || 
                                    (pool_y == 0 && pool_x == 0)) begin
                                    max_value <= input_element;
                                end
                            end
                            default: begin
                                if (input_element > max_value || 
                                    (pool_y == 0 && pool_x == 0)) begin
                                    max_value <= input_element;
                                end
                            end
                        endcase
                    end else begin // Average pooling
                        case (data_type)
                            3'b101: begin // FP32
                                sum_value <= sum_value + input_element;
                            end
                            3'b010: begin // INT32
                                sum_value <= sum_value + input_element;
                            end
                            default: begin
                                sum_value <= sum_value + input_element;
                            end
                        endcase
                    end
                end
                
                STORE_OUTPUT: begin
                    if (pool_type == 1'b0) begin // Max pooling
                        output_element <= max_value;
                    end else begin // Average pooling
                        // Simplified division (would use dedicated divider)
                        case (data_type)
                            3'b101: begin // FP32
                                // Simplified: divide by pool area
                                avg_value <= sum_value / (pool_height * pool_width);
                                output_element <= avg_value;
                            end
                            3'b010: begin // INT32
                                avg_value <= sum_value / (pool_height * pool_width);
                                output_element <= avg_value;
                            end
                            default: begin
                                avg_value <= sum_value / (pool_height * pool_width);
                                output_element <= avg_value;
                            end
                        endcase
                    end
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
                    result <= output_addr; // Return output tensor address
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