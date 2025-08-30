// TPU Compute Array - Top level module with data flow control
// Manages the systolic array and provides pipeline control
// Includes input/output buffering and data type conversion

module tpu_compute_array #(
    parameter ARRAY_SIZE = 64,
    parameter DATA_WIDTH = 32,
    parameter BUFFER_DEPTH = 256
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    start,
    input  logic                    reset_array,
    input  logic [1:0]              data_type,      // 00: INT8, 01: FP16, 10: FP32
    input  logic                    accumulate_mode,
    
    // Configuration
    input  logic [7:0]              matrix_size_m,  // M dimension
    input  logic [7:0]              matrix_size_n,  // N dimension  
    input  logic [7:0]              matrix_size_k,  // K dimension
    
    // Input data streams
    input  logic                    input_valid,
    input  logic [DATA_WIDTH-1:0]   input_data_a,   // Activation data
    input  logic [DATA_WIDTH-1:0]   input_data_b,   // Weight data
    output logic                    input_ready,
    
    // Output data stream
    output logic                    output_valid,
    output logic [DATA_WIDTH-1:0]   output_data,
    input  logic                    output_ready,
    
    // Status and control
    output logic                    busy,
    output logic                    done,
    output logic                    error,
    
    // Performance monitoring
    output logic [31:0]             cycle_count,
    output logic [31:0]             throughput_ops
);

    // Internal state machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_WEIGHTS,
        SETUP_COMPUTE,
        STREAM_DATA,
        COMPUTE_ACTIVE,
        DRAIN_PIPELINE,
        OUTPUT_RESULTS,
        ERROR_STATE
    } compute_state_t;
    
    compute_state_t current_state, next_state;
    
    // Internal control signals
    logic systolic_enable;
    logic systolic_load_weights;
    logic systolic_start_compute;
    logic systolic_accumulate;
    
    // Systolic array interface - flattened for synthesis compatibility
    logic [DATA_WIDTH*ARRAY_SIZE-1:0] systolic_a_inputs;
    logic [DATA_WIDTH*ARRAY_SIZE-1:0] systolic_b_inputs;
    logic [DATA_WIDTH*ARRAY_SIZE-1:0] systolic_c_inputs;
    logic [DATA_WIDTH*ARRAY_SIZE-1:0] systolic_results;
    
    logic systolic_done;
    logic systolic_overflow;
    logic systolic_underflow;   
 
    // Input/Output buffers
    logic [DATA_WIDTH-1:0] input_buffer_a [BUFFER_DEPTH-1:0];
    logic [DATA_WIDTH-1:0] input_buffer_b [BUFFER_DEPTH-1:0];
    logic [DATA_WIDTH-1:0] output_buffer [BUFFER_DEPTH-1:0];
    
    logic [7:0] input_write_ptr, input_read_ptr;
    logic [7:0] output_write_ptr, output_read_ptr;
    logic input_buffer_full, input_buffer_empty;
    logic output_buffer_full, output_buffer_empty;
    
    // Pipeline control counters
    logic [7:0] weight_load_counter;
    logic [7:0] compute_counter;
    logic [7:0] drain_counter;
    logic [7:0] stream_counter;
    logic [15:0] total_cycles;
    
    // Matrix operation tracking
    logic [7:0] current_row, current_col, current_k;
    logic matrix_complete;
    
    // Data flow control
    logic [ARRAY_SIZE-1:0] data_flow_enable;
    logic systolic_ready;
    logic data_streaming_active;
    
    // Performance optimization
    logic [1:0] data_type_reg;
    logic [7:0] matrix_m_reg, matrix_n_reg, matrix_k_reg;
    
    // Instantiate systolic array
    tpu_systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH)
    ) systolic_array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(systolic_enable),
        .data_type(data_type),
        .load_weights(systolic_load_weights),
        .start_compute(systolic_start_compute),
        .accumulate_mode(systolic_accumulate),
        .a_inputs(systolic_a_inputs),
        .b_inputs(systolic_b_inputs),
        .c_inputs(systolic_c_inputs),
        .results(systolic_results),
        .computation_done(systolic_done),
        .overflow_detected(systolic_overflow),
        .underflow_detected(systolic_underflow),
        .cycles_count(cycle_count),
        .ops_count(throughput_ops)
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
                if (start && !input_buffer_empty)
                    next_state = LOAD_WEIGHTS;
            end
            LOAD_WEIGHTS: begin
                if (weight_load_counter >= matrix_size_k - 1)
                    next_state = SETUP_COMPUTE;
            end
            SETUP_COMPUTE: begin
                next_state = STREAM_DATA;
            end
            STREAM_DATA: begin
                if (stream_counter >= ARRAY_SIZE - 1)
                    next_state = COMPUTE_ACTIVE;
            end
            COMPUTE_ACTIVE: begin
                if (matrix_complete)
                    next_state = DRAIN_PIPELINE;
                else if (systolic_overflow || systolic_underflow)
                    next_state = ERROR_STATE;
            end
            DRAIN_PIPELINE: begin
                if (systolic_done)
                    next_state = OUTPUT_RESULTS;
            end
            OUTPUT_RESULTS: begin
                if (output_buffer_empty)
                    next_state = IDLE;
            end
            ERROR_STATE: begin
                if (reset_array)
                    next_state = IDLE;
            end
        endcase
    end    
 
   // Control signal generation
    always_comb begin
        systolic_enable = (current_state != IDLE) && (current_state != ERROR_STATE);
        systolic_load_weights = (current_state == LOAD_WEIGHTS);
        systolic_start_compute = (current_state == COMPUTE_ACTIVE);
        systolic_accumulate = accumulate_mode && (current_state == COMPUTE_ACTIVE);
        
        busy = (current_state != IDLE);
        done = (current_state == OUTPUT_RESULTS) && output_buffer_empty;
        error = (current_state == ERROR_STATE);
        
        input_ready = !input_buffer_full && (current_state == IDLE || current_state == LOAD_WEIGHTS);
        output_valid = !output_buffer_empty && (current_state == OUTPUT_RESULTS);
    end
    
    // Input buffer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_write_ptr <= '0;
            input_read_ptr <= '0;
            input_buffer_full <= 1'b0;
            input_buffer_empty <= 1'b1;
        end else begin
            // Write to input buffer
            if (input_valid && input_ready) begin
                input_buffer_a[input_write_ptr] <= input_data_a;
                input_buffer_b[input_write_ptr] <= input_data_b;
                input_write_ptr <= input_write_ptr + 1;
                input_buffer_empty <= 1'b0;
                
                if (input_write_ptr + 1 == input_read_ptr)
                    input_buffer_full <= 1'b1;
            end
            
            // Read from input buffer
            if (systolic_enable && !input_buffer_empty) begin
                input_read_ptr <= input_read_ptr + 1;
                input_buffer_full <= 1'b0;
                
                if (input_read_ptr + 1 == input_write_ptr)
                    input_buffer_empty <= 1'b1;
            end
        end
    end
    
    // Output buffer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_write_ptr <= '0;
            output_read_ptr <= '0;
            output_buffer_full <= 1'b0;
            output_buffer_empty <= 1'b1;
        end else begin
            // Write to output buffer
            if (systolic_done && !output_buffer_full) begin
                for (int i = 0; i < ARRAY_SIZE; i++) begin
                    if (output_write_ptr < BUFFER_DEPTH) begin
                        output_buffer[output_write_ptr] <= systolic_results[i*DATA_WIDTH +: DATA_WIDTH];
                        output_write_ptr <= output_write_ptr + 1;
                    end
                end
                output_buffer_empty <= 1'b0;
            end
            
            // Read from output buffer
            if (output_valid && output_ready) begin
                output_read_ptr <= output_read_ptr + 1;
                output_buffer_full <= 1'b0;
                
                if (output_read_ptr + 1 == output_write_ptr)
                    output_buffer_empty <= 1'b1;
            end
        end
    end
    
    // Data routing to systolic array
    always_comb begin
        // Default values
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            systolic_a_inputs[i*DATA_WIDTH +: DATA_WIDTH] = '0;
            systolic_b_inputs[i*DATA_WIDTH +: DATA_WIDTH] = '0;
            systolic_c_inputs[i*DATA_WIDTH +: DATA_WIDTH] = '0;
        end
        
        // Route input data based on current operation
        if (current_state == LOAD_WEIGHTS || current_state == COMPUTE_ACTIVE) begin
            for (int i = 0; i < ARRAY_SIZE; i++) begin
                if ((input_read_ptr + i < BUFFER_DEPTH) && (i < matrix_size_m)) begin
                    systolic_a_inputs[i*DATA_WIDTH +: DATA_WIDTH] = input_buffer_a[input_read_ptr + i];
                    systolic_b_inputs[i*DATA_WIDTH +: DATA_WIDTH] = input_buffer_b[input_read_ptr + i];
                end else begin
                    systolic_a_inputs[i*DATA_WIDTH +: DATA_WIDTH] = '0;
                    systolic_b_inputs[i*DATA_WIDTH +: DATA_WIDTH] = '0;
                end
            end
        end
    end
    
    // Output data routing
    assign output_data = output_buffer_empty ? '0 : output_buffer[output_read_ptr];
    
    // Counter management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_load_counter <= '0;
            compute_counter <= '0;
            drain_counter <= '0;
            total_cycles <= '0;
            current_row <= '0;
            current_col <= '0;
            current_k <= '0;
            matrix_complete <= 1'b0;
        end else begin
            case (current_state)
                LOAD_WEIGHTS: begin
                    weight_load_counter <= weight_load_counter + 1;
                end
                COMPUTE_ACTIVE: begin
                    compute_counter <= compute_counter + 1;
                    total_cycles <= total_cycles + 1;
                    
                    // Track matrix computation progress
                    if (current_k < matrix_size_k - 1) begin
                        current_k <= current_k + 1;
                    end else begin
                        current_k <= '0;
                        if (current_col < matrix_size_n - 1) begin
                            current_col <= current_col + 1;
                        end else begin
                            current_col <= '0;
                            if (current_row < matrix_size_m - 1) begin
                                current_row <= current_row + 1;
                            end else begin
                                matrix_complete <= 1'b1;
                            end
                        end
                    end
                end
                DRAIN_PIPELINE: begin
                    drain_counter <= drain_counter + 1;
                end
                IDLE: begin
                    weight_load_counter <= '0;
                    compute_counter <= '0;
                    drain_counter <= '0;
                    current_row <= '0;
                    current_col <= '0;
                    current_k <= '0;
                    matrix_complete <= 1'b0;
                end
            endcase
        end
    end

endmodule