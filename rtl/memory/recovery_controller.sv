/**
 * Recovery Controller for Error Recovery and Retry Mechanisms
 * Handles automatic error recovery, retry logic, and system resilience
 */

module recovery_controller #(
    parameter MAX_RETRY_COUNT = 3,
    parameter RECOVERY_TIMEOUT = 1000,
    parameter NUM_RECOVERY_UNITS = 8
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Error inputs
    input  logic                    error_detected,
    input  logic [4:0]              error_type,
    input  logic [2:0]              error_severity,
    input  logic [31:0]             error_addr,
    input  logic                    error_recoverable,
    
    // Recovery control
    input  logic                    recovery_enable,
    input  logic                    manual_recovery_trigger,
    input  logic [2:0]              recovery_strategy,
    input  logic [15:0]             retry_delay,
    
    // Unit control interfaces
    output logic [NUM_RECOVERY_UNITS-1:0] unit_reset,
    output logic [NUM_RECOVERY_UNITS-1:0] unit_isolate,
    output logic [NUM_RECOVERY_UNITS-1:0] unit_enable,
    input  logic [NUM_RECOVERY_UNITS-1:0] unit_ready,
    input  logic [NUM_RECOVERY_UNITS-1:0] unit_error,
    
    // Memory recovery interface
    output logic                    memory_scrub_req,
    output logic [31:0]             memory_scrub_addr,
    output logic [31:0]             memory_scrub_size,
    input  logic                    memory_scrub_done,
    input  logic                    memory_scrub_error,
    
    // Checkpoint interface
    output logic                    checkpoint_restore_req,
    output logic [2:0]              checkpoint_id,
    input  logic                    checkpoint_restore_done,
    input  logic                    checkpoint_restore_error,
    
    // Status and statistics
    output logic                    recovery_active,
    output logic                    recovery_success,
    output logic                    recovery_failed,
    output logic [7:0]              retry_count,
    output logic [31:0]             recovery_time,
    output logic [15:0]             total_recoveries,
    output logic [15:0]             failed_recoveries,
    
    // Configuration
    input  logic [7:0]              max_retries,
    input  logic [31:0]             timeout_cycles,
    input  logic                    auto_recovery_en
);

    // Recovery strategies
    typedef enum logic [2:0] {
        STRATEGY_RETRY      = 3'b000,  // Simple retry
        STRATEGY_RESET      = 3'b001,  // Reset affected unit
        STRATEGY_ISOLATE    = 3'b010,  // Isolate faulty unit
        STRATEGY_CHECKPOINT = 3'b011,  // Restore from checkpoint
        STRATEGY_SCRUB      = 3'b100,  // Memory scrubbing
        STRATEGY_DEGRADE    = 3'b101,  // Graceful degradation
        STRATEGY_FAILOVER   = 3'b110,  // Failover to backup
        STRATEGY_SHUTDOWN   = 3'b111   // Emergency shutdown
    } recovery_strategy_t;

    // Recovery state machine
    typedef enum logic [3:0] {
        IDLE            = 4'b0000,
        ERROR_ANALYSIS  = 4'b0001,
        STRATEGY_SELECT = 4'b0010,
        UNIT_RESET      = 4'b0011,
        UNIT_ISOLATE    = 4'b0100,
        MEMORY_SCRUB    = 4'b0101,
        CHECKPOINT_RESTORE = 4'b0110,
        RETRY_OPERATION = 4'b0111,
        WAIT_RECOVERY   = 4'b1000,
        VERIFY_RECOVERY = 4'b1001,
        RECOVERY_COMPLETE = 4'b1010,
        RECOVERY_TIMEOUT = 4'b1011,
        RECOVERY_FAILURE = 4'b1100
    } recovery_state_t;

    recovery_state_t current_state, next_state;
    
    // Internal registers
    logic [7:0] current_retry_count;
    logic [31:0] recovery_timer;
    logic [15:0] delay_counter;
    logic [15:0] total_recovery_count;
    logic [15:0] failed_recovery_count;
    logic [4:0] current_error_type;
    logic [2:0] current_error_severity;
    logic [31:0] current_error_addr;
    logic [2:0] selected_strategy;
    logic [NUM_RECOVERY_UNITS-1:0] affected_units;
    logic [31:0] recovery_start_time;
    
    // Error analysis and unit mapping
    always_comb begin
        affected_units = '0;
        
        // Map error types to affected units
        case (current_error_type)
            5'd0, 5'd1: affected_units[0] = 1'b1; // L1 cache errors -> Core 0
            5'd2, 5'd3: affected_units[1] = 1'b1; // L2 cache errors -> Core 1
            5'd4, 5'd5: affected_units[2] = 1'b1; // L3 cache errors -> Core 2
            5'd6, 5'd7: affected_units[3] = 1'b1; // Memory errors -> Core 3
            5'd8, 5'd9: affected_units[4] = 1'b1; // Core arithmetic -> Core unit
            5'd10, 5'd11: affected_units[5] = 1'b1; // TPU errors -> TPU unit
            5'd12, 5'd13: affected_units[6] = 1'b1; // VPU errors -> VPU unit
            5'd14, 5'd15: affected_units[7] = 1'b1; // NoC errors -> NoC unit
            default: affected_units = '0;
        endcase
    end
    
    // Recovery strategy selection
    always_comb begin
        selected_strategy = recovery_strategy;
        
        // Auto-select strategy based on error severity if not manually specified
        if (auto_recovery_en && recovery_strategy == 3'b000) begin
            case (current_error_severity)
                3'b000, 3'b001: selected_strategy = STRATEGY_RETRY;      // Info/Warning
                3'b010: selected_strategy = STRATEGY_SCRUB;             // Minor
                3'b011: selected_strategy = STRATEGY_RESET;             // Major
                3'b100: selected_strategy = STRATEGY_CHECKPOINT;        // Critical
                3'b101: selected_strategy = STRATEGY_SHUTDOWN;          // Fatal
                default: selected_strategy = STRATEGY_RETRY;
            endcase
        end
    end
    
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
                if ((error_detected && recovery_enable) || manual_recovery_trigger) begin
                    next_state = ERROR_ANALYSIS;
                end
            end
            
            ERROR_ANALYSIS: begin
                next_state = STRATEGY_SELECT;
            end
            
            STRATEGY_SELECT: begin
                case (selected_strategy)
                    STRATEGY_RESET: next_state = UNIT_RESET;
                    STRATEGY_ISOLATE: next_state = UNIT_ISOLATE;
                    STRATEGY_SCRUB: next_state = MEMORY_SCRUB;
                    STRATEGY_CHECKPOINT: next_state = CHECKPOINT_RESTORE;
                    STRATEGY_RETRY: next_state = RETRY_OPERATION;
                    default: next_state = RETRY_OPERATION;
                endcase
            end
            
            UNIT_RESET: begin
                if (|unit_ready) begin
                    next_state = WAIT_RECOVERY;
                end else if (recovery_timer >= timeout_cycles) begin
                    next_state = RECOVERY_TIMEOUT;
                end
            end
            
            UNIT_ISOLATE: begin
                next_state = WAIT_RECOVERY;
            end
            
            MEMORY_SCRUB: begin
                if (memory_scrub_done) begin
                    if (memory_scrub_error) begin
                        next_state = RECOVERY_FAILURE;
                    end else begin
                        next_state = VERIFY_RECOVERY;
                    end
                end else if (recovery_timer >= timeout_cycles) begin
                    next_state = RECOVERY_TIMEOUT;
                end
            end
            
            CHECKPOINT_RESTORE: begin
                if (checkpoint_restore_done) begin
                    if (checkpoint_restore_error) begin
                        next_state = RECOVERY_FAILURE;
                    end else begin
                        next_state = VERIFY_RECOVERY;
                    end
                end else if (recovery_timer >= timeout_cycles) begin
                    next_state = RECOVERY_TIMEOUT;
                end
            end
            
            RETRY_OPERATION: begin
                if (delay_counter >= retry_delay) begin
                    next_state = WAIT_RECOVERY;
                end
            end
            
            WAIT_RECOVERY: begin
                if (recovery_timer >= timeout_cycles) begin
                    next_state = RECOVERY_TIMEOUT;
                end else begin
                    next_state = VERIFY_RECOVERY;
                end
            end
            
            VERIFY_RECOVERY: begin
                if (|unit_error) begin
                    // Recovery failed, check if we can retry
                    if (current_retry_count < max_retries) begin
                        next_state = STRATEGY_SELECT;
                    end else begin
                        next_state = RECOVERY_FAILURE;
                    end
                end else begin
                    next_state = RECOVERY_COMPLETE;
                end
            end
            
            RECOVERY_COMPLETE: begin
                next_state = IDLE;
            end
            
            RECOVERY_TIMEOUT: begin
                if (current_retry_count < max_retries) begin
                    next_state = STRATEGY_SELECT;
                end else begin
                    next_state = RECOVERY_FAILURE;
                end
            end
            
            RECOVERY_FAILURE: begin
                next_state = IDLE;
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Register updates
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_retry_count <= 8'b0;
            recovery_timer <= 32'b0;
            delay_counter <= 16'b0;
            total_recovery_count <= 16'b0;
            failed_recovery_count <= 16'b0;
            current_error_type <= 5'b0;
            current_error_severity <= 3'b0;
            current_error_addr <= 32'b0;
            recovery_start_time <= 32'b0;
        end else begin
            case (current_state)
                IDLE: begin
                    if ((error_detected && recovery_enable) || manual_recovery_trigger) begin
                        current_error_type <= error_type;
                        current_error_severity <= error_severity;
                        current_error_addr <= error_addr;
                        current_retry_count <= 8'b0;
                        recovery_timer <= 32'b0;
                        recovery_start_time <= recovery_timer;
                    end
                end
                
                ERROR_ANALYSIS: begin
                    recovery_timer <= recovery_timer + 1;
                end
                
                STRATEGY_SELECT: begin
                    recovery_timer <= recovery_timer + 1;
                end
                
                UNIT_RESET, UNIT_ISOLATE, MEMORY_SCRUB, CHECKPOINT_RESTORE, WAIT_RECOVERY: begin
                    recovery_timer <= recovery_timer + 1;
                end
                
                RETRY_OPERATION: begin
                    if (delay_counter < retry_delay) begin
                        delay_counter <= delay_counter + 1;
                    end else begin
                        delay_counter <= 16'b0;
                    end
                    recovery_timer <= recovery_timer + 1;
                end
                
                VERIFY_RECOVERY: begin
                    if (|unit_error && current_retry_count < max_retries) begin
                        current_retry_count <= current_retry_count + 1;
                        recovery_timer <= 32'b0; // Reset timer for retry
                    end
                    recovery_timer <= recovery_timer + 1;
                end
                
                RECOVERY_COMPLETE: begin
                    total_recovery_count <= total_recovery_count + 1;
                end
                
                RECOVERY_TIMEOUT, RECOVERY_FAILURE: begin
                    failed_recovery_count <= failed_recovery_count + 1;
                    if (current_retry_count < max_retries) begin
                        current_retry_count <= current_retry_count + 1;
                        recovery_timer <= 32'b0; // Reset timer for retry
                    end
                end
                
                default: begin
                    recovery_timer <= recovery_timer + 1;
                end
            endcase
        end
    end
    
    // Unit control outputs
    always_comb begin
        unit_reset = '0;
        unit_isolate = '0;
        unit_enable = '1; // Default: all units enabled
        
        case (current_state)
            UNIT_RESET: begin
                unit_reset = affected_units;
                unit_enable = ~affected_units;
            end
            
            UNIT_ISOLATE: begin
                unit_isolate = affected_units;
                unit_enable = ~affected_units;
            end
            
            default: begin
                unit_enable = '1;
            end
        endcase
    end
    
    // Memory scrub control
    assign memory_scrub_req = (current_state == MEMORY_SCRUB);
    assign memory_scrub_addr = current_error_addr;
    assign memory_scrub_size = 32'h1000; // 4KB scrub size
    
    // Checkpoint restore control
    assign checkpoint_restore_req = (current_state == CHECKPOINT_RESTORE);
    assign checkpoint_id = 3'b000; // Use most recent checkpoint
    
    // Status outputs
    assign recovery_active = (current_state != IDLE);
    assign recovery_success = (current_state == RECOVERY_COMPLETE);
    assign recovery_failed = (current_state == RECOVERY_FAILURE);
    assign retry_count = current_retry_count;
    assign recovery_time = recovery_timer - recovery_start_time;
    assign total_recoveries = total_recovery_count;
    assign failed_recoveries = failed_recovery_count;

endmodule