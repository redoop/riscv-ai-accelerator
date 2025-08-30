// AI 2D Convolution Unit
// Implements hardware-accelerated 2D convolution

`timescale 1ns/1ps

module ai_conv2d_unit #(
    parameter XLEN = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic [2:0]              data_type,
    
    // Convolution parameters
    input  logic [XLEN-1:0]         input_addr,
    input  logic [XLEN-1:0]         kernel_addr,
    input  logic [XLEN-1:0]         output_addr,
    input  logic [31:0]             conv_params, // Packed parameters
    
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

    // Convolution parameters extraction
    // conv_params format: [in_h:8][in_w:8][kernel_h:4][kernel_w:4][stride:4][pad:4]
    logic [7:0] input_height, input_width;
    logic [3:0] kernel_height, kernel_width;
    logic [3:0] stride, padding;
    
    assign input_height = conv_params[31:24];
    assign input_width = conv_params[23:16];
    assign kernel_height = conv_params[15:12];
    assign kernel_width = conv_params[11:8];
    assign stride = conv_params[7:4];
    assign padding = conv_params[3:0];
    
    // Calculated output dimensions
    logic [7:0] output_height, output_width;
    assign output_height = (input_height + 2*padding - kernel_height) / stride + 1;
    assign output_width = (input_width + 2*padding - kernel_width) / stride + 1;
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_KERNEL,
        CONV_OUTER_Y,
        CONV_OUTER_X,
        CONV_KERNEL_Y,
        CONV_KERNEL_X,
        LOAD_INPUT,
        COMPUTE,
        STORE_OUTPUT,
        DONE
    } conv_state_t;
    
    conv_state_t current_state, next_state;
    
    // Loop counters
    logic [7:0] out_y, out_x;     // Output position
    logic [3:0] ker_y, ker_x;     // Kernel position
    logic [7:0] in_y, in_x;       // Input position
    
    // Data storage
    logic [DATA_WIDTH-1:0] kernel_data [0:15][0:15]; // Max 16x16 kernel
    logic [DATA_WIDTH-1:0] input_element;
    logic [DATA_WIDTH-1:0] kernel_element;
    logic [DATA_WIDTH-1:0] accumulator;
    logic [DATA_WIDTH-1:0] output_element;
    
    // Control signals
    logic kernel_loaded;
    logic [7:0] kernel_load_cnt;
    
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
                    next_state = LOAD_KERNEL;
                end
            end
            
            LOAD_KERNEL: begin
                if (kernel_loaded) begin
                    next_state = CONV_OUTER_Y;
                end
            end
            
            CONV_OUTER_Y: begin
                if (out_y < output_height) begin
                    next_state = CONV_OUTER_X;
                end else begin
                    next_state = DONE;
                end
            end
            
            CONV_OUTER_X: begin
                if (out_x < output_width) begin
                    next_state = CONV_KERNEL_Y;
                end else begin
                    next_state = CONV_OUTER_Y;
                end
            end
            
            CONV_KERNEL_Y: begin
                if (ker_y < kernel_height) begin
                    next_state = CONV_KERNEL_X;
                end else begin
                    next_state = STORE_OUTPUT;
                end
            end
            
            CONV_KERNEL_X: begin
                if (ker_x < kernel_width) begin
                    next_state = LOAD_INPUT;
                end else begin
                    next_state = CONV_KERNEL_Y;
                end
            end
            
            LOAD_INPUT: begin
                if (mem_ready) begin
                    next_state = COMPUTE;
                end
            end
            
            COMPUTE: begin
                next_state = CONV_KERNEL_X;
            end
            
            STORE_OUTPUT: begin
                if (mem_ready) begin
                    next_state = CONV_OUTER_X;
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
            ker_y <= '0;
            ker_x <= '0;
            kernel_load_cnt <= '0;
        end else begin
            case (current_state)
                IDLE: begin
                    out_y <= '0;
                    out_x <= '0;
                    ker_y <= '0;
                    ker_x <= '0;
                    kernel_load_cnt <= '0;
                end
                
                LOAD_KERNEL: begin
                    if (mem_ready && mem_req) begin
                        kernel_load_cnt <= kernel_load_cnt + 1;
                    end
                end
                
                CONV_OUTER_Y: begin
                    if (out_y >= output_height) begin
                        out_y <= '0;
                    end else begin
                        out_x <= '0;
                        ker_y <= '0;
                        accumulator <= '0;
                    end
                end
                
                CONV_OUTER_X: begin
                    if (out_x >= output_width) begin
                        out_x <= '0;
                        out_y <= out_y + 1;
                    end else begin
                        ker_y <= '0;
                        accumulator <= '0;
                    end
                end
                
                CONV_KERNEL_Y: begin
                    if (ker_y >= kernel_height) begin
                        ker_y <= '0;
                    end else begin
                        ker_x <= '0;
                    end
                end
                
                CONV_KERNEL_X: begin
                    if (ker_x >= kernel_width) begin
                        ker_x <= '0;
                        ker_y <= ker_y + 1;
                    end
                end
                
                COMPUTE: begin
                    ker_x <= ker_x + 1;
                end
                
                STORE_OUTPUT: begin
                    out_x <= out_x + 1;
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Kernel loading
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_loaded <= 1'b0;
            for (int i = 0; i < 16; i++) begin
                for (int j = 0; j < 16; j++) begin
                    kernel_data[i][j] <= '0;
                end
            end
        end else if (current_state == LOAD_KERNEL && mem_ready && mem_req) begin
            kernel_data[kernel_load_cnt / kernel_width][kernel_load_cnt % kernel_width] <= mem_rdata[DATA_WIDTH-1:0];
            if (kernel_load_cnt == (kernel_height * kernel_width - 1)) begin
                kernel_loaded <= 1'b1;
            end
        end else if (current_state == IDLE) begin
            kernel_loaded <= 1'b0;
        end
    end
    
    // Input position calculation with padding
    always_comb begin
        in_y = out_y * stride + ker_y - padding;
        in_x = out_x * stride + ker_x - padding;
    end
    
    // Memory interface
    always_comb begin
        case (current_state)
            LOAD_KERNEL: begin
                mem_addr = kernel_addr + (kernel_load_cnt * (DATA_WIDTH/8));
                mem_req = !kernel_loaded;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
            end
            
            LOAD_INPUT: begin
                // Check bounds for padding
                if (in_y >= input_height || in_x >= input_width || 
                    in_y[7] == 1'b1 || in_x[7] == 1'b1) begin // Negative (padding area)
                    mem_addr = '0;
                    mem_req = 1'b0;
                end else begin
                    mem_addr = input_addr + ((in_y * input_width + in_x) * (DATA_WIDTH/8));
                    mem_req = 1'b1;
                end
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
    
    // Convolution computation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_element <= '0;
            accumulator <= '0;
            output_element <= '0;
        end else begin
            case (current_state)
                CONV_OUTER_X: begin
                    accumulator <= '0; // Reset for new output element
                end
                
                LOAD_INPUT: begin
                    if (in_y >= input_height || in_x >= input_width || 
                        in_y[7] == 1'b1 || in_x[7] == 1'b1) begin
                        input_element <= '0; // Padding value
                    end else if (mem_ready) begin
                        input_element <= mem_rdata[DATA_WIDTH-1:0];
                    end
                end
                
                COMPUTE: begin
                    kernel_element = kernel_data[ker_y][ker_x];
                    
                    case (data_type)
                        3'b101: begin // FP32
                            accumulator <= accumulator + (input_element * kernel_element);
                        end
                        3'b010: begin // INT32
                            accumulator <= accumulator + (input_element * kernel_element);
                        end
                        default: begin
                            accumulator <= accumulator + (input_element * kernel_element);
                        end
                    endcase
                end
                
                STORE_OUTPUT: begin
                    output_element <= accumulator;
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