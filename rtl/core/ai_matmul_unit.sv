// AI Matrix Multiplication Unit
// Implements hardware-accelerated matrix multiplication

`timescale 1ns/1ps

module ai_matmul_unit #(
    parameter XLEN = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic [2:0]              data_type,
    
    // Matrix addresses and dimensions
    input  logic [XLEN-1:0]         matrix_a_addr,
    input  logic [XLEN-1:0]         matrix_b_addr,
    input  logic [XLEN-1:0]         result_addr,
    input  logic [31:0]             dimensions, // [M:N:K] packed
    
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

    // Matrix dimensions
    logic [7:0] M, N, K; // Support up to 256x256 matrices
    assign M = dimensions[23:16];
    assign N = dimensions[15:8];
    assign K = dimensions[7:0];
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_A,
        LOAD_B,
        COMPUTE,
        STORE_RESULT,
        DONE
    } matmul_state_t;
    
    matmul_state_t current_state, next_state;
    
    // Counters
    logic [7:0] i_cnt, j_cnt, k_cnt;
    logic [15:0] addr_offset;
    
    // Accumulator and temporary storage
    logic [DATA_WIDTH-1:0] accumulator;
    logic [DATA_WIDTH-1:0] a_element, b_element;
    logic [DATA_WIDTH-1:0] result_element;
    
    // Memory access control
    logic load_a_req, load_b_req, store_req;
    
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
                    next_state = LOAD_A;
                end
            end
            
            LOAD_A: begin
                if (mem_ready && load_a_req) begin
                    next_state = LOAD_B;
                end
            end
            
            LOAD_B: begin
                if (mem_ready && load_b_req) begin
                    next_state = COMPUTE;
                end
            end
            
            COMPUTE: begin
                if (k_cnt == K-1) begin
                    next_state = STORE_RESULT;
                end else begin
                    next_state = LOAD_A;
                end
            end
            
            STORE_RESULT: begin
                if (mem_ready && store_req) begin
                    if (i_cnt == M-1 && j_cnt == N-1) begin
                        next_state = DONE;
                    end else begin
                        next_state = LOAD_A;
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
    
    // Counter management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            i_cnt <= '0;
            j_cnt <= '0;
            k_cnt <= '0;
        end else begin
            case (current_state)
                IDLE: begin
                    i_cnt <= '0;
                    j_cnt <= '0;
                    k_cnt <= '0;
                end
                
                COMPUTE: begin
                    if (k_cnt == K-1) begin
                        k_cnt <= '0;
                        if (j_cnt == N-1) begin
                            j_cnt <= '0;
                            i_cnt <= i_cnt + 1;
                        end else begin
                            j_cnt <= j_cnt + 1;
                        end
                    end else begin
                        k_cnt <= k_cnt + 1;
                    end
                end
                
                default: begin
                    // Keep current values
                end
            endcase
        end
    end
    
    // Memory address calculation
    always_comb begin
        case (current_state)
            LOAD_A: begin
                mem_addr = matrix_a_addr + ((i_cnt * K + k_cnt) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
                load_a_req = 1'b1;
                load_b_req = 1'b0;
                store_req = 1'b0;
            end
            
            LOAD_B: begin
                mem_addr = matrix_b_addr + ((k_cnt * N + j_cnt) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
                load_a_req = 1'b0;
                load_b_req = 1'b1;
                store_req = 1'b0;
            end
            
            STORE_RESULT: begin
                mem_addr = result_addr + ((i_cnt * N + j_cnt) * (DATA_WIDTH/8));
                mem_req = 1'b1;
                mem_we = 1'b1;
                mem_wmask = (DATA_WIDTH == 32) ? 8'b1111 : 8'b11111111;
                mem_wdata = {32'b0, result_element};
                load_a_req = 1'b0;
                load_b_req = 1'b0;
                store_req = 1'b1;
            end
            
            default: begin
                mem_addr = '0;
                mem_req = 1'b0;
                mem_we = 1'b0;
                mem_wmask = '0;
                mem_wdata = '0;
                load_a_req = 1'b0;
                load_b_req = 1'b0;
                store_req = 1'b0;
            end
        endcase
    end
    
    // Data loading and computation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_element <= '0;
            b_element <= '0;
            accumulator <= '0;
            result_element <= '0;
        end else begin
            case (current_state)
                LOAD_A: begin
                    if (mem_ready) begin
                        case (data_type)
                            3'b101: a_element <= mem_rdata[31:0]; // FP32
                            3'b010: a_element <= mem_rdata[31:0]; // INT32
                            default: a_element <= mem_rdata[31:0];
                        endcase
                    end
                end
                
                LOAD_B: begin
                    if (mem_ready) begin
                        case (data_type)
                            3'b101: b_element <= mem_rdata[31:0]; // FP32
                            3'b010: b_element <= mem_rdata[31:0]; // INT32
                            default: b_element <= mem_rdata[31:0];
                        endcase
                    end
                    
                    // Initialize accumulator for new element
                    if (k_cnt == 0) begin
                        accumulator <= '0;
                    end
                end
                
                COMPUTE: begin
                    // Perform multiply-accumulate
                    case (data_type)
                        3'b101: begin // FP32
                            // Simplified FP32 multiply-add (would use dedicated FPU in real implementation)
                            accumulator <= accumulator + (a_element * b_element);
                        end
                        3'b010: begin // INT32
                            accumulator <= accumulator + (a_element * b_element);
                        end
                        default: begin
                            accumulator <= accumulator + (a_element * b_element);
                        end
                    endcase
                end
                
                STORE_RESULT: begin
                    result_element <= accumulator;
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
                    result <= result_addr; // Return result matrix address
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