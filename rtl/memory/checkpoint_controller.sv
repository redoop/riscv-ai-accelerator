/**
 * Checkpoint Controller for System State Saving and Recovery
 * Provides checkpoint creation, state saving, and recovery mechanisms
 */

module checkpoint_controller #(
    parameter NUM_CORES = 4,
    parameter NUM_TPUS = 2,
    parameter NUM_VPUS = 2,
    parameter CHECKPOINT_DEPTH = 8,
    parameter STATE_WIDTH = 1024
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    checkpoint_enable,
    input  logic                    checkpoint_trigger,
    input  logic                    recovery_trigger,
    input  logic [2:0]              checkpoint_id,
    input  logic                    auto_checkpoint_en,
    input  logic [15:0]             checkpoint_interval,
    
    // Core state interfaces
    input  logic [STATE_WIDTH-1:0] core_state [NUM_CORES],
    output logic [STATE_WIDTH-1:0] core_state_restore [NUM_CORES],
    input  logic [NUM_CORES-1:0]   core_state_valid,
    output logic [NUM_CORES-1:0]   core_restore_valid,
    
    // TPU state interfaces
    input  logic [STATE_WIDTH-1:0] tpu_state [NUM_TPUS],
    output logic [STATE_WIDTH-1:0] tpu_state_restore [NUM_TPUS],
    input  logic [NUM_TPUS-1:0]    tpu_state_valid,
    output logic [NUM_TPUS-1:0]    tpu_restore_valid,
    
    // VPU state interfaces
    input  logic [STATE_WIDTH-1:0] vpu_state [NUM_VPUS],
    output logic [STATE_WIDTH-1:0] vpu_state_restore [NUM_VPUS],
    input  logic [NUM_VPUS-1:0]    vpu_state_valid,
    output logic [NUM_VPUS-1:0]    vpu_restore_valid,
    
    // Memory state interface
    input  logic [63:0]             memory_checkpoint_addr,
    input  logic [31:0]             memory_checkpoint_size,
    output logic                    memory_save_req,
    output logic                    memory_restore_req,
    output logic [63:0]             memory_save_addr,
    output logic [63:0]             memory_restore_addr,
    output logic [31:0]             memory_transfer_size,
    input  logic                    memory_save_done,
    input  logic                    memory_restore_done,
    
    // Status and control
    output logic                    checkpoint_active,
    output logic                    recovery_active,
    output logic                    checkpoint_complete,
    output logic                    recovery_complete,
    output logic [2:0]              last_checkpoint_id,
    output logic [31:0]             checkpoint_timestamp,
    output logic [7:0]              checkpoint_status,
    
    // Error interface
    input  logic                    checkpoint_error,
    input  logic                    recovery_error,
    output logic                    checkpoint_failed,
    output logic                    recovery_failed
);

    // Checkpoint state machine
    typedef enum logic [3:0] {
        IDLE           = 4'b0000,
        SAVE_CORES     = 4'b0001,
        SAVE_TPUS      = 4'b0010,
        SAVE_VPUS      = 4'b0011,
        SAVE_MEMORY    = 4'b0100,
        CHECKPOINT_DONE = 4'b0101,
        RESTORE_CORES  = 4'b0110,
        RESTORE_TPUS   = 4'b0111,
        RESTORE_VPUS   = 4'b1000,
        RESTORE_MEMORY = 4'b1001,
        RECOVERY_DONE  = 4'b1010,
        ERROR_STATE    = 4'b1111
    } checkpoint_state_t;

    checkpoint_state_t current_state, next_state;
    
    // Internal registers
    logic [2:0] active_checkpoint_id;
    logic [31:0] timestamp_counter;
    logic [15:0] interval_counter;
    logic [3:0] core_save_counter;
    logic [3:0] tpu_save_counter;
    logic [3:0] vpu_save_counter;
    logic [3:0] core_restore_counter;
    logic [3:0] tpu_restore_counter;
    logic [3:0] vpu_restore_counter;
    
    // Checkpoint storage
    logic [STATE_WIDTH-1:0] checkpoint_core_state [CHECKPOINT_DEPTH][NUM_CORES];
    logic [STATE_WIDTH-1:0] checkpoint_tpu_state [CHECKPOINT_DEPTH][NUM_TPUS];
    logic [STATE_WIDTH-1:0] checkpoint_vpu_state [CHECKPOINT_DEPTH][NUM_VPUS];
    logic [63:0] checkpoint_memory_addr [CHECKPOINT_DEPTH];
    logic [31:0] checkpoint_memory_size [CHECKPOINT_DEPTH];
    logic [31:0] checkpoint_timestamps [CHECKPOINT_DEPTH];
    logic [CHECKPOINT_DEPTH-1:0] checkpoint_valid;
    
    // Timestamp counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            timestamp_counter <= 32'b0;
        end else begin
            timestamp_counter <= timestamp_counter + 1;
        end
    end
    
    // Auto checkpoint interval counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            interval_counter <= 16'b0;
        end else if (auto_checkpoint_en) begin
            if (interval_counter >= checkpoint_interval) begin
                interval_counter <= 16'b0;
            end else begin
                interval_counter <= interval_counter + 1;
            end
        end else begin
            interval_counter <= 16'b0;
        end
    end
    
    // Auto checkpoint trigger
    logic auto_checkpoint_trigger;
    assign auto_checkpoint_trigger = auto_checkpoint_en && (interval_counter >= checkpoint_interval);
    
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
                if (recovery_trigger) begin
                    next_state = RESTORE_CORES;
                end else if (checkpoint_trigger || auto_checkpoint_trigger) begin
                    next_state = SAVE_CORES;
                end
            end
            
            SAVE_CORES: begin
                if (core_save_counter >= NUM_CORES) begin
                    next_state = SAVE_TPUS;
                end else if (checkpoint_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            SAVE_TPUS: begin
                if (tpu_save_counter >= NUM_TPUS) begin
                    next_state = SAVE_VPUS;
                end else if (checkpoint_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            SAVE_VPUS: begin
                if (vpu_save_counter >= NUM_VPUS) begin
                    next_state = SAVE_MEMORY;
                end else if (checkpoint_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            SAVE_MEMORY: begin
                if (memory_save_done) begin
                    next_state = CHECKPOINT_DONE;
                end else if (checkpoint_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            CHECKPOINT_DONE: begin
                next_state = IDLE;
            end
            
            RESTORE_CORES: begin
                if (core_restore_counter >= NUM_CORES) begin
                    next_state = RESTORE_TPUS;
                end else if (recovery_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            RESTORE_TPUS: begin
                if (tpu_restore_counter >= NUM_TPUS) begin
                    next_state = RESTORE_VPUS;
                end else if (recovery_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            RESTORE_VPUS: begin
                if (vpu_restore_counter >= NUM_VPUS) begin
                    next_state = RESTORE_MEMORY;
                end else if (recovery_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            RESTORE_MEMORY: begin
                if (memory_restore_done) begin
                    next_state = RECOVERY_DONE;
                end else if (recovery_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            RECOVERY_DONE: begin
                next_state = IDLE;
            end
            
            ERROR_STATE: begin
                next_state = IDLE; // Return to idle after error handling
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Counter management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_save_counter <= 4'b0;
            tpu_save_counter <= 4'b0;
            vpu_save_counter <= 4'b0;
            core_restore_counter <= 4'b0;
            tpu_restore_counter <= 4'b0;
            vpu_restore_counter <= 4'b0;
        end else begin
            case (current_state)
                SAVE_CORES: begin
                    if (core_save_counter < NUM_CORES) begin
                        core_save_counter <= core_save_counter + 1;
                    end
                end
                SAVE_TPUS: begin
                    if (tpu_save_counter < NUM_TPUS) begin
                        tpu_save_counter <= tpu_save_counter + 1;
                    end
                end
                SAVE_VPUS: begin
                    if (vpu_save_counter < NUM_VPUS) begin
                        vpu_save_counter <= vpu_save_counter + 1;
                    end
                end
                RESTORE_CORES: begin
                    if (core_restore_counter < NUM_CORES) begin
                        core_restore_counter <= core_restore_counter + 1;
                    end
                end
                RESTORE_TPUS: begin
                    if (tpu_restore_counter < NUM_TPUS) begin
                        tpu_restore_counter <= tpu_restore_counter + 1;
                    end
                end
                RESTORE_VPUS: begin
                    if (vpu_restore_counter < NUM_VPUS) begin
                        vpu_restore_counter <= vpu_restore_counter + 1;
                    end
                end
                IDLE: begin
                    core_save_counter <= 4'b0;
                    tpu_save_counter <= 4'b0;
                    vpu_save_counter <= 4'b0;
                    core_restore_counter <= 4'b0;
                    tpu_restore_counter <= 4'b0;
                    vpu_restore_counter <= 4'b0;
                end
            endcase
        end
    end
    
    // Checkpoint storage management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            checkpoint_valid <= '0;
            active_checkpoint_id <= 3'b0;
            for (int i = 0; i < CHECKPOINT_DEPTH; i++) begin
                checkpoint_timestamps[i] <= 32'b0;
                checkpoint_memory_addr[i] <= 64'b0;
                checkpoint_memory_size[i] <= 32'b0;
            end
        end else begin
            case (current_state)
                SAVE_CORES: begin
                    if (core_save_counter < NUM_CORES && core_state_valid[core_save_counter]) begin
                        checkpoint_core_state[active_checkpoint_id][core_save_counter] <= core_state[core_save_counter];
                    end
                end
                SAVE_TPUS: begin
                    if (tpu_save_counter < NUM_TPUS && tpu_state_valid[tpu_save_counter]) begin
                        checkpoint_tpu_state[active_checkpoint_id][tpu_save_counter] <= tpu_state[tpu_save_counter];
                    end
                end
                SAVE_VPUS: begin
                    if (vpu_save_counter < NUM_VPUS && vpu_state_valid[vpu_save_counter]) begin
                        checkpoint_vpu_state[active_checkpoint_id][vpu_save_counter] <= vpu_state[vpu_save_counter];
                    end
                end
                SAVE_MEMORY: begin
                    checkpoint_memory_addr[active_checkpoint_id] <= memory_checkpoint_addr;
                    checkpoint_memory_size[active_checkpoint_id] <= memory_checkpoint_size;
                end
                CHECKPOINT_DONE: begin
                    checkpoint_valid[active_checkpoint_id] <= 1'b1;
                    checkpoint_timestamps[active_checkpoint_id] <= timestamp_counter;
                    active_checkpoint_id <= checkpoint_id;
                end
            endcase
        end
    end
    
    // State restoration
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : core_restore_gen
            assign core_state_restore[i] = checkpoint_core_state[checkpoint_id][i];
            assign core_restore_valid[i] = (current_state == RESTORE_CORES) && 
                                          checkpoint_valid[checkpoint_id] &&
                                          (core_restore_counter == i);
        end
        
        for (i = 0; i < NUM_TPUS; i++) begin : tpu_restore_gen
            assign tpu_state_restore[i] = checkpoint_tpu_state[checkpoint_id][i];
            assign tpu_restore_valid[i] = (current_state == RESTORE_TPUS) && 
                                         checkpoint_valid[checkpoint_id] &&
                                         (tpu_restore_counter == i);
        end
        
        for (i = 0; i < NUM_VPUS; i++) begin : vpu_restore_gen
            assign vpu_state_restore[i] = checkpoint_vpu_state[checkpoint_id][i];
            assign vpu_restore_valid[i] = (current_state == RESTORE_VPUS) && 
                                         checkpoint_valid[checkpoint_id] &&
                                         (vpu_restore_counter == i);
        end
    endgenerate
    
    // Memory save/restore control
    assign memory_save_req = (current_state == SAVE_MEMORY);
    assign memory_restore_req = (current_state == RESTORE_MEMORY);
    assign memory_save_addr = checkpoint_memory_addr[active_checkpoint_id];
    assign memory_restore_addr = checkpoint_memory_addr[checkpoint_id];
    assign memory_transfer_size = (current_state == SAVE_MEMORY) ? 
                                 checkpoint_memory_size[active_checkpoint_id] :
                                 checkpoint_memory_size[checkpoint_id];
    
    // Status outputs
    assign checkpoint_active = (current_state >= SAVE_CORES) && (current_state <= SAVE_MEMORY);
    assign recovery_active = (current_state >= RESTORE_CORES) && (current_state <= RESTORE_MEMORY);
    assign checkpoint_complete = (current_state == CHECKPOINT_DONE);
    assign recovery_complete = (current_state == RECOVERY_DONE);
    assign last_checkpoint_id = active_checkpoint_id;
    assign checkpoint_timestamp = checkpoint_timestamps[active_checkpoint_id];
    assign checkpoint_status = {4'b0, current_state};
    assign checkpoint_failed = (current_state == ERROR_STATE) && checkpoint_error;
    assign recovery_failed = (current_state == ERROR_STATE) && recovery_error;

endmodule