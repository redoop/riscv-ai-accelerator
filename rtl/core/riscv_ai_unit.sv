// RISC-V AI Instruction Extension Unit
// Implements custom AI instructions for neural network acceleration

`timescale 1ns/1ps

module riscv_ai_unit #(
    parameter XLEN = 64,
    parameter AI_DATA_WIDTH = 32  // Support for FP32 operations
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    ai_enable,
    input  logic [6:0]              ai_opcode,
    input  logic [2:0]              funct3,
    input  logic [6:0]              funct7,
    input  logic [4:0]              rs1, rs2, rs3, rd,
    
    // Data interface
    input  logic [XLEN-1:0]         rs1_data,
    input  logic [XLEN-1:0]         rs2_data,
    input  logic [XLEN-1:0]         rs3_data,
    output logic [XLEN-1:0]         ai_result,
    
    // Memory interface for AI operations
    output logic [XLEN-1:0]         ai_mem_addr,
    output logic [XLEN-1:0]         ai_mem_wdata,
    output logic [7:0]              ai_mem_wmask,
    output logic                    ai_mem_req,
    output logic                    ai_mem_we,
    input  logic [XLEN-1:0]         ai_mem_rdata,
    input  logic                    ai_mem_ready,
    
    // Status and control
    output logic                    ai_ready,
    output logic                    ai_valid,
    output logic [4:0]              ai_flags  // Error flags
);

    // ========================================
    // AI Instruction Encoding
    // ========================================
    
    // Custom AI opcode (using reserved space)
    localparam [6:0] OP_AI_CUSTOM = 7'b0001011;  // Custom-0 opcode space
    
    // AI instruction funct7 encodings
    localparam [6:0] AI_MATMUL     = 7'b0000001;  // Matrix multiplication
    localparam [6:0] AI_CONV2D     = 7'b0000010;  // 2D Convolution
    localparam [6:0] AI_RELU       = 7'b0000100;  // ReLU activation
    localparam [6:0] AI_SIGMOID    = 7'b0000101;  // Sigmoid activation
    localparam [6:0] AI_TANH       = 7'b0000110;  // Tanh activation
    localparam [6:0] AI_MAXPOOL    = 7'b0001000;  // Max pooling
    localparam [6:0] AI_AVGPOOL    = 7'b0001001;  // Average pooling
    localparam [6:0] AI_BATCHNORM  = 7'b0001010;  // Batch normalization
    localparam [6:0] AI_SYNC       = 7'b0010000;  // Synchronization
    localparam [6:0] AI_FLUSH      = 7'b0010001;  // Pipeline flush
    
    // AI instruction funct3 encodings (data types)
    localparam [2:0] AI_INT8   = 3'b000;
    localparam [2:0] AI_INT16  = 3'b001;
    localparam [2:0] AI_INT32  = 3'b010;
    localparam [2:0] AI_FP16   = 3'b100;
    localparam [2:0] AI_FP32   = 3'b101;
    localparam [2:0] AI_FP64   = 3'b110;

    // ========================================
    // Internal Registers and State
    // ========================================
    
    typedef enum logic [3:0] {
        AI_IDLE,
        AI_DECODE,
        AI_EXECUTE,
        AI_MEMORY_ACCESS,
        AI_COMPUTE,
        AI_WRITEBACK,
        AI_ERROR
    } ai_state_t;
    
    ai_state_t current_state, next_state;
    
    // Instruction decode registers
    logic [6:0]     decoded_funct7;
    logic [2:0]     decoded_funct3;
    logic [4:0]     decoded_rs1, decoded_rs2, decoded_rs3, decoded_rd;
    
    // Operand registers
    logic [XLEN-1:0] operand_a, operand_b, operand_c;
    logic [XLEN-1:0] result_reg;
    
    // Configuration registers
    logic [31:0]    matrix_dims;     // Matrix dimensions for MATMUL
    logic [31:0]    conv_params;     // Convolution parameters
    logic [31:0]    pool_params;     // Pooling parameters
    
    // Status flags
    logic           overflow_flag;
    logic           underflow_flag;
    logic           invalid_op_flag;
    logic           divide_zero_flag;
    logic           memory_fault_flag;

    // ========================================
    // State Machine
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= AI_IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            AI_IDLE: begin
                if (ai_enable && (ai_opcode == OP_AI_CUSTOM)) begin
                    next_state = AI_DECODE;
                end
            end
            
            AI_DECODE: begin
                next_state = AI_EXECUTE;
            end
            
            AI_EXECUTE: begin
                case (decoded_funct7)
                    AI_MATMUL, AI_CONV2D: begin
                        next_state = AI_MEMORY_ACCESS;
                    end
                    AI_RELU, AI_SIGMOID, AI_TANH: begin
                        next_state = AI_COMPUTE;
                    end
                    AI_MAXPOOL, AI_AVGPOOL, AI_BATCHNORM: begin
                        next_state = AI_MEMORY_ACCESS;
                    end
                    AI_SYNC, AI_FLUSH: begin
                        next_state = AI_WRITEBACK;
                    end
                    default: begin
                        next_state = AI_ERROR;
                    end
                endcase
            end
            
            AI_MEMORY_ACCESS: begin
                if (ai_mem_ready) begin
                    next_state = AI_COMPUTE;
                end
            end
            
            AI_COMPUTE: begin
                next_state = AI_WRITEBACK;
            end
            
            AI_WRITEBACK: begin
                next_state = AI_IDLE;
            end
            
            AI_ERROR: begin
                next_state = AI_IDLE;
            end
            
            default: begin
                next_state = AI_IDLE;
            end
        endcase
    end

    // ========================================
    // Instruction Decode Stage
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            decoded_funct7 <= '0;
            decoded_funct3 <= '0;
            decoded_rs1 <= '0;
            decoded_rs2 <= '0;
            decoded_rs3 <= '0;
            decoded_rd <= '0;
            operand_a <= '0;
            operand_b <= '0;
            operand_c <= '0;
        end else if (current_state == AI_DECODE) begin
            decoded_funct7 <= funct7;
            decoded_funct3 <= funct3;
            decoded_rs1 <= rs1;
            decoded_rs2 <= rs2;
            decoded_rs3 <= rs3;
            decoded_rd <= rd;
            operand_a <= rs1_data;
            operand_b <= rs2_data;
            operand_c <= rs3_data;
        end
    end

    // ========================================
    // AI Instruction Execution
    // ========================================
    
    // Matrix Multiplication Unit
    logic [XLEN-1:0] matmul_result;
    logic            matmul_valid;
    
    ai_matmul_unit matmul_unit (
        .clk(clk),
        .rst_n(rst_n),
        .enable(current_state == AI_COMPUTE && decoded_funct7 == AI_MATMUL),
        .data_type(decoded_funct3),
        .matrix_a_addr(operand_a),
        .matrix_b_addr(operand_b),
        .result_addr(operand_c),
        .dimensions(matrix_dims),
        .mem_addr(ai_mem_addr),
        .mem_wdata(ai_mem_wdata),
        .mem_wmask(ai_mem_wmask),
        .mem_req(ai_mem_req),
        .mem_we(ai_mem_we),
        .mem_rdata(ai_mem_rdata),
        .mem_ready(ai_mem_ready),
        .result(matmul_result),
        .valid(matmul_valid)
    );
    
    // Convolution Unit
    logic [XLEN-1:0] conv_result;
    logic            conv_valid;
    
    ai_conv2d_unit conv_unit (
        .clk(clk),
        .rst_n(rst_n),
        .enable(current_state == AI_COMPUTE && decoded_funct7 == AI_CONV2D),
        .data_type(decoded_funct3),
        .input_addr(operand_a),
        .kernel_addr(operand_b),
        .output_addr(operand_c),
        .conv_params(conv_params),
        .mem_addr(ai_mem_addr),
        .mem_wdata(ai_mem_wdata),
        .mem_wmask(ai_mem_wmask),
        .mem_req(ai_mem_req),
        .mem_we(ai_mem_we),
        .mem_rdata(ai_mem_rdata),
        .mem_ready(ai_mem_ready),
        .result(conv_result),
        .valid(conv_valid)
    );
    
    // Activation Function Unit
    logic [XLEN-1:0] activation_result;
    logic            activation_valid;
    
    ai_activation_unit activation_unit (
        .clk(clk),
        .rst_n(rst_n),
        .enable(current_state == AI_COMPUTE && 
                (decoded_funct7 == AI_RELU || decoded_funct7 == AI_SIGMOID || decoded_funct7 == AI_TANH)),
        .activation_type(decoded_funct7[2:0]),
        .data_type(decoded_funct3),
        .input_data(operand_a),
        .result(activation_result),
        .valid(activation_valid),
        .overflow(overflow_flag),
        .underflow(underflow_flag)
    );
    
    // Pooling Unit
    logic [XLEN-1:0] pool_result;
    logic            pool_valid;
    
    ai_pooling_unit pooling_unit (
        .clk(clk),
        .rst_n(rst_n),
        .enable(current_state == AI_COMPUTE && 
                (decoded_funct7 == AI_MAXPOOL || decoded_funct7 == AI_AVGPOOL)),
        .pool_type(decoded_funct7[0]), // 0=max, 1=avg
        .data_type(decoded_funct3),
        .input_addr(operand_a),
        .output_addr(operand_b),
        .pool_params(pool_params),
        .mem_addr(ai_mem_addr),
        .mem_wdata(ai_mem_wdata),
        .mem_wmask(ai_mem_wmask),
        .mem_req(ai_mem_req),
        .mem_we(ai_mem_we),
        .mem_rdata(ai_mem_rdata),
        .mem_ready(ai_mem_ready),
        .result(pool_result),
        .valid(pool_valid)
    );
    
    // Batch Normalization Unit
    logic [XLEN-1:0] batchnorm_result;
    logic            batchnorm_valid;
    
    ai_batchnorm_unit batchnorm_unit (
        .clk(clk),
        .rst_n(rst_n),
        .enable(current_state == AI_COMPUTE && decoded_funct7 == AI_BATCHNORM),
        .data_type(decoded_funct3),
        .input_addr(operand_a),
        .scale_addr(operand_b),
        .bias_addr(operand_c),
        .mem_addr(ai_mem_addr),
        .mem_wdata(ai_mem_wdata),
        .mem_wmask(ai_mem_wmask),
        .mem_req(ai_mem_req),
        .mem_we(ai_mem_we),
        .mem_rdata(ai_mem_rdata),
        .mem_ready(ai_mem_ready),
        .result(batchnorm_result),
        .valid(batchnorm_valid)
    );

    // ========================================
    // Result Multiplexing
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_reg <= '0;
            ai_valid <= 1'b0;
        end else if (current_state == AI_WRITEBACK) begin
            ai_valid <= 1'b1;
            case (decoded_funct7)
                AI_MATMUL:    result_reg <= matmul_result;
                AI_CONV2D:    result_reg <= conv_result;
                AI_RELU, AI_SIGMOID, AI_TANH: result_reg <= activation_result;
                AI_MAXPOOL, AI_AVGPOOL: result_reg <= pool_result;
                AI_BATCHNORM: result_reg <= batchnorm_result;
                AI_SYNC:      result_reg <= 64'h1; // Success indicator
                AI_FLUSH:     result_reg <= 64'h1; // Success indicator
                default:      result_reg <= '0;
            endcase
        end else begin
            ai_valid <= 1'b0;
        end
    end
    
    assign ai_result = result_reg;
    assign ai_ready = (current_state == AI_IDLE) || (current_state == AI_WRITEBACK);

    // ========================================
    // Error Flag Management
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            overflow_flag <= 1'b0;
            underflow_flag <= 1'b0;
            invalid_op_flag <= 1'b0;
            divide_zero_flag <= 1'b0;
            memory_fault_flag <= 1'b0;
        end else begin
            // Clear flags on new instruction
            if (current_state == AI_DECODE) begin
                overflow_flag <= 1'b0;
                underflow_flag <= 1'b0;
                invalid_op_flag <= 1'b0;
                divide_zero_flag <= 1'b0;
                memory_fault_flag <= 1'b0;
            end
            
            // Set error flags based on operation results
            if (current_state == AI_ERROR) begin
                invalid_op_flag <= 1'b1;
            end
            
            if (!ai_mem_ready && ai_mem_req) begin
                memory_fault_flag <= 1'b1;
            end
        end
    end
    
    assign ai_flags = {memory_fault_flag, divide_zero_flag, invalid_op_flag, underflow_flag, overflow_flag};

    // ========================================
    // Parameter Extraction
    // ========================================
    
    // Extract matrix dimensions from rs3_data for MATMUL
    always_comb begin
        if (decoded_funct7 == AI_MATMUL) begin
            matrix_dims = operand_c[31:0]; // M, N, K packed in 32 bits
        end else begin
            matrix_dims = '0;
        end
    end
    
    // Extract convolution parameters from rs3_data for CONV2D
    always_comb begin
        if (decoded_funct7 == AI_CONV2D) begin
            conv_params = operand_c[31:0]; // Height, width, channels, etc.
        end else begin
            conv_params = '0;
        end
    end
    
    // Extract pooling parameters from rs2_data for pooling operations
    always_comb begin
        if (decoded_funct7 == AI_MAXPOOL || decoded_funct7 == AI_AVGPOOL) begin
            pool_params = operand_b[31:0]; // Pool size, stride, etc.
        end else begin
            pool_params = '0;
        end
    end

endmodule