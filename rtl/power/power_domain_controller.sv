/*
 * Power Domain Controller
 * 
 * This module manages power domains and implements power gating
 * for different parts of the chip to save power when idle.
 */

module power_domain_controller #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Power gating control inputs
    input  logic [NUM_CORES-1:0]   core_power_gate_req,
    input  logic                    memory_power_gate_req,
    input  logic                    ai_accel_power_gate_req,
    
    // Activity monitoring
    input  logic [NUM_CORES-1:0]   core_activity,
    input  logic                    memory_activity,
    input  logic [NUM_AI_UNITS-1:0] ai_unit_activity,
    
    // Power domain control outputs
    output logic [NUM_CORES-1:0]   core_power_enable,
    output logic                    l1_cache_power_enable,
    output logic                    l2_cache_power_enable,
    output logic                    memory_ctrl_power_enable,
    output logic [NUM_AI_UNITS-1:0] ai_unit_power_enable,
    output logic                    noc_power_enable,
    
    // Isolation control
    output logic [NUM_CORES-1:0]   core_isolation_enable,
    output logic                    memory_isolation_enable,
    output logic [NUM_AI_UNITS-1:0] ai_unit_isolation_enable,
    
    // Status
    output logic [7:0]              power_domain_status,
    output logic                    power_transition_busy
);

    // Power state definitions
    typedef enum logic [2:0] {
        PWR_ON,
        PWR_ISOLATE,
        PWR_GATE_PREP,
        PWR_GATED,
        PWR_RESTORE_PREP,
        PWR_RESTORE,
        PWR_DEISOLATE
    } power_state_t;

    // Power domain states
    power_state_t core_power_state [NUM_CORES-1:0];
    power_state_t memory_power_state;
    power_state_t ai_unit_power_state [NUM_AI_UNITS-1:0];
    
    // Transition timers
    logic [7:0] core_transition_timer [NUM_CORES-1:0];
    logic [7:0] memory_transition_timer;
    logic [7:0] ai_unit_transition_timer [NUM_AI_UNITS-1:0];
    
    // Activity counters for debouncing
    logic [15:0] core_idle_counter [NUM_CORES-1:0];
    logic [15:0] memory_idle_counter;
    logic [15:0] ai_unit_idle_counter [NUM_AI_UNITS-1:0];
    
    // Power gating thresholds
    localparam IDLE_THRESHOLD = 16'd1000;  // Cycles before considering power gating
    localparam GATE_DELAY = 8'd10;         // Cycles for power gating sequence
    localparam RESTORE_DELAY = 8'd20;      // Cycles for power restore sequence

    // Core power domain management
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : core_power_domains
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    core_power_state[i] <= PWR_ON;
                    core_transition_timer[i] <= '0;
                    core_idle_counter[i] <= '0;
                    core_power_enable[i] <= 1'b1;
                    core_isolation_enable[i] <= 1'b0;
                end else begin
                    // Update idle counter
                    if (core_activity[i]) begin
                        core_idle_counter[i] <= '0;
                    end else if (core_idle_counter[i] < IDLE_THRESHOLD) begin
                        core_idle_counter[i] <= core_idle_counter[i] + 1;
                    end
                    
                    // Power state machine
                    case (core_power_state[i])
                        PWR_ON: begin
                            if (core_power_gate_req[i] && 
                                (core_idle_counter[i] >= IDLE_THRESHOLD)) begin
                                core_power_state[i] <= PWR_ISOLATE;
                                core_transition_timer[i] <= GATE_DELAY;
                            end
                        end
                        
                        PWR_ISOLATE: begin
                            core_isolation_enable[i] <= 1'b1;
                            if (core_transition_timer[i] > 0) begin
                                core_transition_timer[i] <= core_transition_timer[i] - 1;
                            end else begin
                                core_power_state[i] <= PWR_GATE_PREP;
                                core_transition_timer[i] <= 8'd5;
                            end
                        end
                        
                        PWR_GATE_PREP: begin
                            if (core_transition_timer[i] > 0) begin
                                core_transition_timer[i] <= core_transition_timer[i] - 1;
                            end else begin
                                core_power_enable[i] <= 1'b0;
                                core_power_state[i] <= PWR_GATED;
                            end
                        end
                        
                        PWR_GATED: begin
                            if (!core_power_gate_req[i] || core_activity[i]) begin
                                core_power_state[i] <= PWR_RESTORE_PREP;
                                core_transition_timer[i] <= 8'd5;
                            end
                        end
                        
                        PWR_RESTORE_PREP: begin
                            core_power_enable[i] <= 1'b1;
                            if (core_transition_timer[i] > 0) begin
                                core_transition_timer[i] <= core_transition_timer[i] - 1;
                            end else begin
                                core_power_state[i] <= PWR_RESTORE;
                                core_transition_timer[i] <= RESTORE_DELAY;
                            end
                        end
                        
                        PWR_RESTORE: begin
                            if (core_transition_timer[i] > 0) begin
                                core_transition_timer[i] <= core_transition_timer[i] - 1;
                            end else begin
                                core_power_state[i] <= PWR_DEISOLATE;
                                core_transition_timer[i] <= 8'd5;
                            end
                        end
                        
                        PWR_DEISOLATE: begin
                            if (core_transition_timer[i] > 0) begin
                                core_transition_timer[i] <= core_transition_timer[i] - 1;
                            end else begin
                                core_isolation_enable[i] <= 1'b0;
                                core_power_state[i] <= PWR_ON;
                                core_idle_counter[i] <= '0;
                            end
                        end
                        
                        default: core_power_state[i] <= PWR_ON;
                    endcase
                end
            end
        end
    endgenerate

    // Memory power domain management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            memory_power_state <= PWR_ON;
            memory_transition_timer <= '0;
            memory_idle_counter <= '0;
            l1_cache_power_enable <= 1'b1;
            l2_cache_power_enable <= 1'b1;
            memory_ctrl_power_enable <= 1'b1;
            memory_isolation_enable <= 1'b0;
        end else begin
            // Update idle counter
            if (memory_activity) begin
                memory_idle_counter <= '0;
            end else if (memory_idle_counter < IDLE_THRESHOLD) begin
                memory_idle_counter <= memory_idle_counter + 1;
            end
            
            // Memory power state machine (simplified - only L2 cache can be gated)
            case (memory_power_state)
                PWR_ON: begin
                    if (memory_power_gate_req && 
                        (memory_idle_counter >= IDLE_THRESHOLD)) begin
                        memory_power_state <= PWR_ISOLATE;
                        memory_transition_timer <= GATE_DELAY;
                    end
                end
                
                PWR_ISOLATE: begin
                    memory_isolation_enable <= 1'b1;
                    if (memory_transition_timer > 0) begin
                        memory_transition_timer <= memory_transition_timer - 1;
                    end else begin
                        l2_cache_power_enable <= 1'b0;  // Only gate L2 cache
                        memory_power_state <= PWR_GATED;
                    end
                end
                
                PWR_GATED: begin
                    if (!memory_power_gate_req || memory_activity) begin
                        l2_cache_power_enable <= 1'b1;
                        memory_power_state <= PWR_RESTORE;
                        memory_transition_timer <= RESTORE_DELAY;
                    end
                end
                
                PWR_RESTORE: begin
                    if (memory_transition_timer > 0) begin
                        memory_transition_timer <= memory_transition_timer - 1;
                    end else begin
                        memory_isolation_enable <= 1'b0;
                        memory_power_state <= PWR_ON;
                        memory_idle_counter <= '0;
                    end
                end
                
                default: memory_power_state <= PWR_ON;
            endcase
        end
    end

    // AI unit power domain management
    generate
        for (i = 0; i < NUM_AI_UNITS; i++) begin : ai_power_domains
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    ai_unit_power_state[i] <= PWR_ON;
                    ai_unit_transition_timer[i] <= '0;
                    ai_unit_idle_counter[i] <= '0;
                    ai_unit_power_enable[i] <= 1'b1;
                    ai_unit_isolation_enable[i] <= 1'b0;
                end else begin
                    // Update idle counter
                    if (ai_unit_activity[i]) begin
                        ai_unit_idle_counter[i] <= '0;
                    end else if (ai_unit_idle_counter[i] < IDLE_THRESHOLD) begin
                        ai_unit_idle_counter[i] <= ai_unit_idle_counter[i] + 1;
                    end
                    
                    // AI unit power state machine
                    case (ai_unit_power_state[i])
                        PWR_ON: begin
                            if (ai_accel_power_gate_req && 
                                (ai_unit_idle_counter[i] >= IDLE_THRESHOLD)) begin
                                ai_unit_power_state[i] <= PWR_ISOLATE;
                                ai_unit_transition_timer[i] <= GATE_DELAY;
                            end
                        end
                        
                        PWR_ISOLATE: begin
                            ai_unit_isolation_enable[i] <= 1'b1;
                            if (ai_unit_transition_timer[i] > 0) begin
                                ai_unit_transition_timer[i] <= ai_unit_transition_timer[i] - 1;
                            end else begin
                                ai_unit_power_enable[i] <= 1'b0;
                                ai_unit_power_state[i] <= PWR_GATED;
                            end
                        end
                        
                        PWR_GATED: begin
                            if (!ai_accel_power_gate_req || ai_unit_activity[i]) begin
                                ai_unit_power_enable[i] <= 1'b1;
                                ai_unit_power_state[i] <= PWR_RESTORE;
                                ai_unit_transition_timer[i] <= RESTORE_DELAY;
                            end
                        end
                        
                        PWR_RESTORE: begin
                            if (ai_unit_transition_timer[i] > 0) begin
                                ai_unit_transition_timer[i] <= ai_unit_transition_timer[i] - 1;
                            end else begin
                                ai_unit_isolation_enable[i] <= 1'b0;
                                ai_unit_power_state[i] <= PWR_ON;
                                ai_unit_idle_counter[i] <= '0;
                            end
                        end
                        
                        default: ai_unit_power_state[i] <= PWR_ON;
                    endcase
                end
            end
        end
    endgenerate

    // NoC is always powered (critical for system communication)
    assign noc_power_enable = 1'b1;

    // Status reporting
    always_comb begin
        power_domain_status = {
            |core_power_enable,           // Any core powered
            l2_cache_power_enable,        // L2 cache powered
            memory_ctrl_power_enable,     // Memory controller powered
            |ai_unit_power_enable,        // Any AI unit powered
            noc_power_enable,             // NoC powered
            memory_power_state != PWR_ON, // Memory in transition
            |core_isolation_enable,       // Any core isolated
            |ai_unit_isolation_enable     // Any AI unit isolated
        };
        
        // Check if any power transition is in progress
        power_transition_busy = 1'b0;
        for (int j = 0; j < NUM_CORES; j++) begin
            if (core_power_state[j] != PWR_ON && core_power_state[j] != PWR_GATED) begin
                power_transition_busy = 1'b1;
            end
        end
        if (memory_power_state != PWR_ON && memory_power_state != PWR_GATED) begin
            power_transition_busy = 1'b1;
        end
        for (int j = 0; j < NUM_AI_UNITS; j++) begin
            if (ai_unit_power_state[j] != PWR_ON && ai_unit_power_state[j] != PWR_GATED) begin
                power_transition_busy = 1'b1;
            end
        end
    end

endmodule