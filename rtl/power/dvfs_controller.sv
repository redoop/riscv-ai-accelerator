/*
 * DVFS Controller - Dynamic Voltage and Frequency Scaling
 * 
 * This module implements dynamic voltage and frequency scaling for the RISC-V AI chip.
 * It monitors system load and adjusts voltage/frequency accordingly to optimize power consumption.
 * 
 * Features:
 * - Load-aware DVFS policy
 * - Multiple voltage/frequency operating points
 * - Power domain isolation and gating
 * - Smooth transitions between operating points
 */

module dvfs_controller #(
    parameter NUM_CORES = 4,
    parameter NUM_VOLTAGE_LEVELS = 8,
    parameter NUM_FREQUENCY_LEVELS = 8,
    parameter LOAD_MONITOR_WIDTH = 16
) (
    input  logic                           clk,
    input  logic                           rst_n,
    
    // Load monitoring inputs from cores
    input  logic [NUM_CORES-1:0][LOAD_MONITOR_WIDTH-1:0] core_load,
    input  logic [NUM_CORES-1:0]           core_active,
    
    // System load indicators
    input  logic [LOAD_MONITOR_WIDTH-1:0] memory_load,
    input  logic [LOAD_MONITOR_WIDTH-1:0] noc_load,
    input  logic [LOAD_MONITOR_WIDTH-1:0] ai_accel_load,
    
    // Temperature feedback
    input  logic [7:0]                     temperature,
    input  logic                           thermal_alert,
    
    // Power management outputs
    output logic [2:0]                     voltage_level,
    output logic [2:0]                     frequency_level,
    output logic [NUM_CORES-1:0]           core_power_gate,
    output logic                           memory_power_gate,
    output logic                           ai_accel_power_gate,
    
    // Status and control
    input  logic                           dvfs_enable,
    input  logic [2:0]                     min_voltage_level,
    input  logic [2:0]                     max_voltage_level,
    output logic                           dvfs_transition_busy,
    output logic [7:0]                     power_state
);

    // Internal registers
    logic [2:0] current_voltage_level;
    logic [2:0] current_frequency_level;
    logic [2:0] target_voltage_level;
    logic [2:0] target_frequency_level;
    
    // Load calculation
    logic [LOAD_MONITOR_WIDTH+2-1:0] total_load;
    logic [LOAD_MONITOR_WIDTH-1:0] average_load;
    logic [NUM_CORES-1:0] core_idle;
    
    // Transition control
    logic transition_in_progress;
    logic [3:0] transition_counter;
    
    // DVFS policy parameters
    localparam LOAD_THRESHOLD_HIGH = 16'h8000;  // 50% load
    localparam LOAD_THRESHOLD_LOW  = 16'h4000;  // 25% load
    localparam IDLE_THRESHOLD      = 16'h1000;  // 6.25% load
    
    // Calculate total system load
    always_comb begin
        total_load = '0;
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_active[i]) begin
                total_load += {{2{1'b0}}, core_load[i]};
            end
        end
        total_load += {{2{1'b0}}, memory_load} + {{2{1'b0}}, noc_load} + {{2{1'b0}}, ai_accel_load};
        
        // Calculate average load (simplified division)
        average_load = total_load[LOAD_MONITOR_WIDTH+2-1:2];
    end
    
    // Determine core idle states
    always_comb begin
        for (int i = 0; i < NUM_CORES; i++) begin
            core_idle[i] = (core_load[i] < IDLE_THRESHOLD) && core_active[i];
        end
    end
    
    // DVFS Policy Engine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            target_voltage_level <= 3'd4;  // Mid-level
            target_frequency_level <= 3'd4;
        end else if (dvfs_enable && !transition_in_progress) begin
            // Thermal protection - reduce performance if overheating
            if (thermal_alert || temperature > 8'd85) begin
                target_voltage_level <= min_voltage_level;
                target_frequency_level <= 3'd2;  // Low frequency
            end
            // High load - increase performance
            else if (average_load > LOAD_THRESHOLD_HIGH) begin
                target_voltage_level <= max_voltage_level;
                target_frequency_level <= 3'd7;  // High frequency
            end
            // Medium load - balanced performance
            else if (average_load > LOAD_THRESHOLD_LOW) begin
                target_voltage_level <= 3'd4;  // Mid voltage
                target_frequency_level <= 3'd4;  // Mid frequency
            end
            // Low load - power saving mode
            else begin
                target_voltage_level <= min_voltage_level + 1;
                target_frequency_level <= 3'd2;  // Low frequency
            end
        end
    end
    
    // Voltage/Frequency Transition Controller
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_voltage_level <= 3'd4;
            current_frequency_level <= 3'd4;
            transition_in_progress <= 1'b0;
            transition_counter <= 4'd0;
        end else begin
            if (!transition_in_progress) begin
                // Check if transition is needed
                if ((target_voltage_level != current_voltage_level) || 
                    (target_frequency_level != current_frequency_level)) begin
                    transition_in_progress <= 1'b1;
                    transition_counter <= 4'd0;
                end
            end else begin
                // Handle transition sequence
                transition_counter <= transition_counter + 1;
                
                case (transition_counter)
                    4'd2: begin
                        // Step 1: Increase voltage first if scaling up
                        if (target_voltage_level > current_voltage_level) begin
                            current_voltage_level <= target_voltage_level;
                        end
                    end
                    4'd6: begin
                        // Step 2: Change frequency
                        current_frequency_level <= target_frequency_level;
                    end
                    4'd10: begin
                        // Step 3: Decrease voltage if scaling down
                        if (target_voltage_level < current_voltage_level) begin
                            current_voltage_level <= target_voltage_level;
                        end
                    end
                    4'd15: begin
                        // Transition complete
                        transition_in_progress <= 1'b0;
                    end
                    default: begin
                        // Continue counting
                    end
                endcase
            end
        end
    end
    
    // Power Gating Control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_power_gate <= '0;
            memory_power_gate <= 1'b0;
            ai_accel_power_gate <= 1'b0;
        end else if (dvfs_enable) begin
            // Core power gating - gate cores that are idle for extended periods
            for (int i = 0; i < NUM_CORES; i++) begin
                core_power_gate[i] <= core_idle[i] && (current_frequency_level <= 3'd2);
            end
            
            // Memory power gating - partial gating when load is very low
            memory_power_gate <= (average_load < IDLE_THRESHOLD) && 
                                (current_frequency_level <= 3'd1);
            
            // AI accelerator power gating
            ai_accel_power_gate <= (ai_accel_load < IDLE_THRESHOLD) && 
                                  (current_frequency_level <= 3'd2);
        end else begin
            core_power_gate <= '0;
            memory_power_gate <= 1'b0;
            ai_accel_power_gate <= 1'b0;
        end
    end
    
    // Output assignments
    assign voltage_level = current_voltage_level;
    assign frequency_level = current_frequency_level;
    assign dvfs_transition_busy = transition_in_progress;
    
    // Power state encoding
    assign power_state = {
        thermal_alert,
        transition_in_progress,
        |core_power_gate,
        memory_power_gate,
        ai_accel_power_gate,
        current_voltage_level
    };

endmodule