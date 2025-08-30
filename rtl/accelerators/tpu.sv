// TPU (Tensor Processing Unit) - AI Accelerator
// Integrates compute array with memory interface and control logic
// Supports matrix operations and neural network inference

module tpu #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter ARRAY_SIZE = 64
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    enable,
    input  logic                    start,
    output logic                    done,
    output logic                    busy,
    
    // Configuration
    input  logic [7:0]              operation,      // Operation type
    input  logic [1:0]              data_type,      // 00: INT8, 01: FP16, 10: FP32
    input  logic [7:0]              matrix_size_m,  // Matrix dimensions
    input  logic [7:0]              matrix_size_n,
    input  logic [7:0]              matrix_size_k,
    
    // Memory interface
    output logic [ADDR_WIDTH-1:0]   mem_addr,
    output logic                    mem_read,
    output logic                    mem_write,
    output logic [DATA_WIDTH-1:0]   mem_wdata,
    input  logic [DATA_WIDTH-1:0]   mem_rdata,
    input  logic                    mem_ready,
    
    // Status and performance
    output logic [31:0]             cycle_count,
    output logic [31:0]             op_count,
    output logic                    error
);

    // Compute array interface
    logic compute_start;
    logic compute_reset;
    logic compute_accumulate;
    logic compute_input_valid;
    logic [DATA_WIDTH-1:0] compute_input_a;
    logic [DATA_WIDTH-1:0] compute_input_b;
    logic compute_input_ready;
    logic compute_output_valid;
    logic [DATA_WIDTH-1:0] compute_output_data;
    logic compute_output_ready;
    logic compute_busy;
    logic compute_done;
    logic compute_error;
    logic [31:0] compute_cycles;
    logic [31:0] compute_ops;
    
    // Memory management
    logic [ADDR_WIDTH-1:0] base_addr_a, base_addr_b, base_addr_c;
    logic [ADDR_WIDTH-1:0] current_addr;
    logic [15:0] data_counter;
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        SETUP,
        LOAD_MATRIX_A,
        LOAD_MATRIX_B,
        COMPUTE_MATRIX,
        STORE_RESULTS,
        CLEANUP,
        ERROR_STATE
    } tpu_state_t;
    
    tpu_state_t current_state, next_state;
    
    // Instantiate compute array
    tpu_compute_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) compute_array (
        .clk(clk),
        .rst_n(rst_n),
        .start(compute_start),
        .reset_array(compute_reset),
        .data_type(data_type),
        .accumulate_mode(compute_accumulate),
        .matrix_size_m(matrix_size_m),
        .matrix_size_n(matrix_size_n),
        .matrix_size_k(matrix_size_k),
        .input_valid(compute_input_valid),
        .input_data_a(compute_input_a),
        .input_data_b(compute_input_b),
        .input_ready(compute_input_ready),
        .output_valid(compute_output_valid),
        .output_data(compute_output_data),
        .output_ready(compute_output_ready),
        .busy(compute_busy),
        .done(compute_done),
        .error(compute_error),
        .cycle_count(compute_cycles),
        .throughput_ops(compute_ops)
    );
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (enable && start)
                    next_state = SETUP;
            end
            SETUP: begin
                next_state = LOAD_MATRIX_A;
            end
            LOAD_MATRIX_A: begin
                if (data_counter >= (matrix_size_m * matrix_size_k))
                    next_state = LOAD_MATRIX_B;
            end
            LOAD_MATRIX_B: begin
                if (data_counter >= (matrix_size_k * matrix_size_n))
                    next_state = COMPUTE_MATRIX;
            end
            COMPUTE_MATRIX: begin
                if (compute_done)
                    next_state = STORE_RESULTS;
                else if (compute_error)
                    next_state = ERROR_STATE;
            end
            STORE_RESULTS: begin
                if (data_counter >= (matrix_size_m * matrix_size_n))
                    next_state = CLEANUP;
            end
            CLEANUP: begin
                next_state = IDLE;
            end
            ERROR_STATE: begin
                if (!enable)
                    next_state = IDLE;
            end
        endcase
    end
    
    // Control signal generation
    always_comb begin
        compute_start = (current_state == COMPUTE_MATRIX);
        compute_reset = (current_state == SETUP);
        compute_accumulate = (operation[0] == 1'b1); // Bit 0 controls accumulation
        
        compute_input_valid = (current_state == LOAD_MATRIX_A || current_state == LOAD_MATRIX_B) && mem_ready;
        compute_output_ready = (current_state == STORE_RESULTS);
        
        mem_read = (current_state == LOAD_MATRIX_A || current_state == LOAD_MATRIX_B);
        mem_write = (current_state == STORE_RESULTS) && compute_output_valid;
        mem_wdata = compute_output_data;
        
        busy = (current_state != IDLE);
        done = (current_state == CLEANUP);
        error = (current_state == ERROR_STATE) || compute_error;
    end
    
    // Data routing
    always_comb begin
        case (current_state)
            LOAD_MATRIX_A: begin
                compute_input_a = mem_rdata;
                compute_input_b = '0;
            end
            LOAD_MATRIX_B: begin
                compute_input_a = '0;
                compute_input_b = mem_rdata;
            end
            default: begin
                compute_input_a = '0;
                compute_input_b = '0;
            end
        endcase
    end
    
    // Address generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_addr <= '0;
            data_counter <= '0;
        end else begin
            case (current_state)
                SETUP: begin
                    base_addr_a <= 32'h1000_0000; // Configurable base addresses
                    base_addr_b <= 32'h2000_0000;
                    base_addr_c <= 32'h3000_0000;
                    current_addr <= base_addr_a;
                    data_counter <= '0;
                end
                LOAD_MATRIX_A: begin
                    if (mem_ready) begin
                        current_addr <= current_addr + 4;
                        data_counter <= data_counter + 1;
                    end
                end
                LOAD_MATRIX_B: begin
                    if (data_counter == 0)
                        current_addr <= base_addr_b;
                    else if (mem_ready) begin
                        current_addr <= current_addr + 4;
                        data_counter <= data_counter + 1;
                    end
                end
                STORE_RESULTS: begin
                    if (data_counter == 0)
                        current_addr <= base_addr_c;
                    else if (mem_write && mem_ready) begin
                        current_addr <= current_addr + 4;
                        data_counter <= data_counter + 1;
                    end
                end
                default: begin
                    data_counter <= '0;
                end
            endcase
        end
    end
    
    assign mem_addr = current_addr;
    assign cycle_count = compute_cycles;
    assign op_count = compute_ops;

endmodule