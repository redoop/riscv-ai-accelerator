// TPU Systolic Array - 64x64 MAC units
// Implements matrix multiplication using systolic array architecture
// Supports INT8, FP16, and FP32 data types

`timescale 1ns/1ps

module tpu_systolic_array #(
    parameter ARRAY_SIZE = 64,
    parameter DATA_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    enable,
    
    // Configuration
    input  logic [1:0]              data_type,      // 00: INT8, 01: FP16, 10: FP32
    input  logic                    load_weights,
    input  logic                    start_compute,
    input  logic                    accumulate_mode,
    
    // Input interfaces - flattened for synthesis compatibility
    input  logic [DATA_WIDTH*ARRAY_SIZE-1:0]   a_inputs,  // Left inputs (activations)
    input  logic [DATA_WIDTH*ARRAY_SIZE-1:0]   b_inputs,  // Top inputs (weights)
    input  logic [DATA_WIDTH*ARRAY_SIZE-1:0]   c_inputs,  // Partial sum inputs
    
    // Output interface
    output logic [DATA_WIDTH*ARRAY_SIZE-1:0]   results,   // Bottom outputs
    
    // Status signals
    output logic                    computation_done,
    output logic                    overflow_detected,
    output logic                    underflow_detected,
    
    // Performance counters
    output logic [31:0]             cycles_count,
    output logic [31:0]             ops_count
);

    // Internal signals for systolic array connections
    logic [DATA_WIDTH-1:0] a_wires [ARRAY_SIZE-1:0][ARRAY_SIZE:0];
    logic [DATA_WIDTH-1:0] b_wires [ARRAY_SIZE:0][ARRAY_SIZE-1:0];
    logic [DATA_WIDTH-1:0] c_wires [ARRAY_SIZE:0][ARRAY_SIZE-1:0];
    
    // MAC unit control signals
    logic mac_enable [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic mac_load_weight [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic mac_accumulate [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    
    // Error signals
    logic mac_overflow [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    logic mac_underflow [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0];
    
    // Control state machine
    typedef enum logic [3:0] {
        IDLE,
        LOAD_WEIGHTS,
        PRELOAD_DATA,
        COMPUTE,
        DRAIN,
        DONE,
        ERROR_RECOVERY
    } state_t;
    
    state_t current_state, next_state;
    logic [7:0] cycle_counter;
    logic [7:0] drain_counter;
    logic [7:0] preload_counter;
    
    // Pipeline control
    logic [ARRAY_SIZE-1:0] weight_loaded;
    logic [ARRAY_SIZE-1:0] data_valid_row;
    logic [ARRAY_SIZE-1:0] data_valid_col;
    logic pipeline_full;
    logic pipeline_empty;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            cycle_counter <= '0;
            drain_counter <= '0;
            preload_counter <= '0;
            weight_loaded <= '0;
            data_valid_row <= '0;
            data_valid_col <= '0;
        end else begin
            current_state <= next_state;
            
            case (current_state)
                LOAD_WEIGHTS: begin
                    if (cycle_counter < ARRAY_SIZE - 1) begin
                        cycle_counter <= cycle_counter + 1;
                        weight_loaded[cycle_counter] <= 1'b1;
                    end else begin
                        cycle_counter <= '0;
                        weight_loaded <= {ARRAY_SIZE{1'b1}};
                    end
                end
                PRELOAD_DATA: begin
                    if (preload_counter < ARRAY_SIZE - 1) begin
                        preload_counter <= preload_counter + 1;
                        data_valid_row[preload_counter] <= 1'b1;
                        data_valid_col[preload_counter] <= 1'b1;
                    end else begin
                        preload_counter <= '0;
                    end
                end
                COMPUTE: begin
                    cycle_counter <= cycle_counter + 1;
                    // Shift data valid signals through pipeline
                    data_valid_row <= {data_valid_row[ARRAY_SIZE-2:0], 1'b1};
                    data_valid_col <= {data_valid_col[ARRAY_SIZE-2:0], 1'b1};
                end
                DRAIN: begin
                    if (drain_counter < ARRAY_SIZE - 1) begin
                        drain_counter <= drain_counter + 1;
                        data_valid_row <= data_valid_row >> 1;
                        data_valid_col <= data_valid_col >> 1;
                    end else begin
                        drain_counter <= '0;
                        data_valid_row <= '0;
                        data_valid_col <= '0;
                    end
                end
                IDLE, DONE: begin
                    cycle_counter <= '0;
                    drain_counter <= '0;
                    preload_counter <= '0;
                    weight_loaded <= '0;
                    data_valid_row <= '0;
                    data_valid_col <= '0;
                end
                ERROR_RECOVERY: begin
                    // Reset all pipeline state
                    cycle_counter <= '0;
                    drain_counter <= '0;
                    preload_counter <= '0;
                    weight_loaded <= '0;
                    data_valid_row <= '0;
                    data_valid_col <= '0;
                end
            endcase
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (load_weights)
                    next_state = LOAD_WEIGHTS;
                else if (start_compute && (&weight_loaded))
                    next_state = PRELOAD_DATA;
            end
            LOAD_WEIGHTS: begin
                if (cycle_counter == ARRAY_SIZE - 1)
                    next_state = IDLE;
            end
            PRELOAD_DATA: begin
                if (preload_counter == ARRAY_SIZE - 1)
                    next_state = COMPUTE;
            end
            COMPUTE: begin
                if (!start_compute)
                    next_state = DRAIN;
                else if (overflow_detected || underflow_detected)
                    next_state = ERROR_RECOVERY;
            end
            DRAIN: begin
                if (drain_counter == ARRAY_SIZE - 1)
                    next_state = DONE;
            end
            DONE: begin
                next_state = IDLE;
            end
            ERROR_RECOVERY: begin
                if (!overflow_detected && !underflow_detected)
                    next_state = IDLE;
            end
        endcase
    end
    
    // Control signal generation
    always_comb begin
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            for (int j = 0; j < ARRAY_SIZE; j++) begin
                mac_enable[i][j] = enable && 
                                  (current_state == COMPUTE || current_state == DRAIN) &&
                                  data_valid_row[i] && data_valid_col[j];
                mac_load_weight[i][j] = (current_state == LOAD_WEIGHTS) && 
                                       (cycle_counter == i);
                mac_accumulate[i][j] = accumulate_mode && 
                                      (current_state == COMPUTE) &&
                                      data_valid_row[i] && data_valid_col[j];
            end
        end
        
        // Pipeline control signals
        pipeline_full = &data_valid_row && &data_valid_col;
        pipeline_empty = ~(|data_valid_row) && ~(|data_valid_col);
    end
    
    // Input connections
    always_comb begin
        // Left inputs (activations)
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            a_wires[i][0] = a_inputs[i];
        end
        
        // Top inputs (weights during loading, zeros during compute)
        for (int j = 0; j < ARRAY_SIZE; j++) begin
            if (current_state == LOAD_WEIGHTS)
                b_wires[0][j] = b_inputs[j];
            else
                b_wires[0][j] = '0;
        end
        
        // Partial sum inputs (top row)
        for (int j = 0; j < ARRAY_SIZE; j++) begin
            c_wires[0][j] = c_inputs[j];
        end
    end
    
    // Generate MAC unit array
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : gen_row
            for (j = 0; j < ARRAY_SIZE; j++) begin : gen_col
                tpu_mac_unit #(
                    .DATA_WIDTH(DATA_WIDTH)
                ) mac_unit (
                    .clk(clk),
                    .rst_n(rst_n),
                    .enable(mac_enable[i][j]),
                    .data_type(data_type),
                    .a_in(a_wires[i][j]),
                    .b_in(b_wires[i][j]),
                    .c_in(c_wires[i][j]),
                    .a_out(a_wires[i][j+1]),
                    .b_out(b_wires[i+1][j]),
                    .c_out(c_wires[i+1][j]),
                    .load_weight(mac_load_weight[i][j]),
                    .accumulate(mac_accumulate[i][j]),
                    .overflow(mac_overflow[i][j]),
                    .underflow(mac_underflow[i][j])
                );
            end
        end
    endgenerate
    
    // Output connections
    always_comb begin
        for (int j = 0; j < ARRAY_SIZE; j++) begin
            results[j] = c_wires[ARRAY_SIZE][j];
        end
    end
    
    // Status signals
    always_comb begin
        computation_done = (current_state == DONE);
        
        // Aggregate overflow/underflow signals
        overflow_detected = 1'b0;
        underflow_detected = 1'b0;
        
        for (int i = 0; i < ARRAY_SIZE; i++) begin
            for (int j = 0; j < ARRAY_SIZE; j++) begin
                overflow_detected |= mac_overflow[i][j];
                underflow_detected |= mac_underflow[i][j];
            end
        end
    end
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycles_count <= '0;
            ops_count <= '0;
        end else if (enable) begin
            if (current_state == COMPUTE) begin
                cycles_count <= cycles_count + 1;
                ops_count <= ops_count + (ARRAY_SIZE * ARRAY_SIZE);
            end
        end else begin
            cycles_count <= '0;
            ops_count <= '0;
        end
    end

endmodule