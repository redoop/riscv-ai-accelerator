/*
 * Thermal Management Controller
 * 
 * This module implements comprehensive thermal management for the RISC-V AI chip,
 * including temperature monitoring, thermal protection, and power throttling.
 * 
 * Features:
 * - Multi-zone temperature monitoring
 * - Thermal protection with graduated response
 * - Power consumption monitoring and reporting
 * - Thermal-aware DVFS coordination
 * - Emergency thermal shutdown
 */

module thermal_controller #(
    parameter NUM_TEMP_SENSORS = 8,
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter TEMP_SENSOR_WIDTH = 16
) (
    input  logic                           clk,
    input  logic                           rst_n,
    
    // Temperature sensor inputs
    input  logic [TEMP_SENSOR_WIDTH-1:0] temp_sensors [NUM_TEMP_SENSORS-1:0],
    input  logic [NUM_TEMP_SENSORS-1:0]   temp_sensor_valid,
    
    // Power consumption inputs
    input  logic [15:0]                    core_power [NUM_CORES-1:0],
    input  logic [15:0]                    ai_unit_power [NUM_AI_UNITS-1:0],
    input  logic [15:0]                    memory_power,
    input  logic [15:0]                    noc_power,
    
    // Activity monitoring
    input  logic [NUM_CORES-1:0]          core_activity,
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_activity,
    input  logic                           memory_activity,
    
    // Thermal management outputs
    output logic [7:0]                     max_temperature,
    output logic [7:0]                     avg_temperature,
    output logic [2:0]                     thermal_zone,
    output logic                           thermal_alert,
    output logic                           thermal_critical,
    output logic                           thermal_emergency,
    
    // Thermal throttling controls
    output logic [2:0]                     thermal_throttle_level,
    output logic [NUM_CORES-1:0]          core_thermal_throttle,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_thermal_throttle,
    output logic                           memory_thermal_throttle,
    
    // Power monitoring outputs
    output logic [15:0]                    total_power_consumption,
    output logic [15:0]                    power_budget_remaining,
    output logic                           power_budget_exceeded,
    
    // Configuration interface
    input  logic [31:0]                    thermal_config_addr,
    input  logic [31:0]                    thermal_config_wdata,
    output logic [31:0]                    thermal_config_rdata,
    input  logic                           thermal_config_req,
    input  logic                           thermal_config_we,
    output logic                           thermal_config_ready
);

    // Temperature thresholds (in Celsius)
    localparam TEMP_NORMAL_MAX     = 8'd70;   // Normal operation limit
    localparam TEMP_ALERT_THRESH   = 8'd80;   // Start thermal management
    localparam TEMP_CRITICAL_THRESH = 8'd90;  // Aggressive throttling
    localparam TEMP_EMERGENCY_THRESH = 8'd100; // Emergency shutdown
    
    // Power budget (in mW)
    localparam DEFAULT_POWER_BUDGET = 16'd15000; // 15W default budget

    // Configuration registers
    logic [7:0]  temp_alert_threshold;
    logic [7:0]  temp_critical_threshold;
    logic [7:0]  temp_emergency_threshold;
    logic [15:0] power_budget;
    logic        thermal_mgmt_enable;
    logic [2:0]  thermal_policy;
    
    // Internal temperature processing
    logic [TEMP_SENSOR_WIDTH-1:0] temp_readings [NUM_TEMP_SENSORS-1:0];
    logic [7:0] temp_celsius [NUM_TEMP_SENSORS-1:0];
    logic [10:0] temp_sum;
    logic [2:0] valid_sensor_count;
    
    // Power monitoring
    logic [19:0] total_power_calc;
    logic [15:0] core_power_sum;
    logic [15:0] ai_power_sum;
    
    // Thermal state machine
    typedef enum logic [2:0] {
        THERMAL_NORMAL,
        THERMAL_ALERT,
        THERMAL_CRITICAL,
        THERMAL_EMERGENCY,
        THERMAL_SHUTDOWN
    } thermal_state_t;
    
    thermal_state_t thermal_state;
    logic [15:0] thermal_timer;
    
    // Throttling control
    logic [7:0] throttle_counter;
    logic [2:0] current_throttle_level;

    // Temperature sensor processing
    always_comb begin
        // Initialize all outputs to prevent latches
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            temp_celsius[i] = 8'd25; // Default to room temperature
        end
        max_temperature = 8'd25;
        avg_temperature = 8'd25;
        temp_sum = 11'd0;
        valid_sensor_count = 3'd0;
        thermal_zone = 3'd0;
        
        // Convert temperature readings to Celsius (assuming 16-bit ADC with proper scaling)
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            if (temp_sensor_valid[i]) begin
                // Convert from ADC reading to Celsius
                // Assuming: ADC_value = (Temperature + 40) * 256 / 165
                // So: Temperature = (ADC_value * 165 / 256) - 40
                automatic logic [23:0] temp_calc = temp_sensors[i] * 165;
                automatic logic [15:0] temp_scaled = temp_calc[23:8];
                if (temp_scaled >= 40) begin
                    temp_celsius[i] = temp_scaled[7:0] - 8'd40;
                end else begin
                    temp_celsius[i] = 8'd0;
                end
            end
        end
        
        // Find maximum temperature
        max_temperature = temp_celsius[0];
        for (int i = 1; i < NUM_TEMP_SENSORS; i++) begin
            if (temp_celsius[i] > max_temperature) begin
                max_temperature = temp_celsius[i];
            end
        end
        
        // Calculate average temperature
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            if (temp_sensor_valid[i]) begin
                temp_sum += {{3{1'b0}}, temp_celsius[i]};
                valid_sensor_count += 1;
            end
        end
        
        if (valid_sensor_count > 0) begin
            avg_temperature = temp_sum[10:3] / {{5{1'b0}}, valid_sensor_count};
        end
        
        // Determine thermal zone based on max temperature
        if (max_temperature >= temp_emergency_threshold) begin
            thermal_zone = 3'd4; // Emergency
        end else if (max_temperature >= temp_critical_threshold) begin
            thermal_zone = 3'd3; // Critical
        end else if (max_temperature >= temp_alert_threshold) begin
            thermal_zone = 3'd2; // Alert
        end else if (max_temperature >= TEMP_NORMAL_MAX) begin
            thermal_zone = 3'd1; // Warm
        end
    end

    // Power consumption calculation
    always_comb begin
        // Sum core power consumption
        core_power_sum = '0;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_power_sum += core_power[i];
        end
        
        // Sum AI unit power consumption
        ai_power_sum = '0;
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_power_sum += ai_unit_power[i];
        end
        
        // Calculate total power
        total_power_calc = {{4{1'b0}}, core_power_sum} + 
                          {{4{1'b0}}, ai_power_sum} + 
                          {{4{1'b0}}, memory_power} + 
                          {{4{1'b0}}, noc_power};
        
        total_power_consumption = total_power_calc[15:0];
        
        // Calculate remaining power budget
        if (total_power_consumption < power_budget) begin
            power_budget_remaining = power_budget - total_power_consumption;
            power_budget_exceeded = 1'b0;
        end else begin
            power_budget_remaining = 16'd0;
            power_budget_exceeded = 1'b1;
        end
    end

    // Thermal management state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            thermal_state <= THERMAL_NORMAL;
            thermal_timer <= '0;
            current_throttle_level <= 3'd0;
            throttle_counter <= '0;
        end else if (thermal_mgmt_enable) begin
            thermal_timer <= thermal_timer + 1;
            
            case (thermal_state)
                THERMAL_NORMAL: begin
                    current_throttle_level <= 3'd0;
                    if (max_temperature >= temp_alert_threshold) begin
                        thermal_state <= THERMAL_ALERT;
                        thermal_timer <= '0;
                    end
                end
                
                THERMAL_ALERT: begin
                    // Light throttling
                    current_throttle_level <= 3'd1;
                    
                    if (max_temperature >= temp_critical_threshold) begin
                        thermal_state <= THERMAL_CRITICAL;
                        thermal_timer <= '0;
                    end else if (max_temperature < (temp_alert_threshold - 8'd5)) begin
                        // Hysteresis: return to normal when 5°C below threshold
                        thermal_state <= THERMAL_NORMAL;
                        thermal_timer <= '0;
                    end
                end
                
                THERMAL_CRITICAL: begin
                    // Aggressive throttling
                    current_throttle_level <= 3'd3;
                    
                    if (max_temperature >= temp_emergency_threshold) begin
                        thermal_state <= THERMAL_EMERGENCY;
                        thermal_timer <= '0;
                    end else if (max_temperature < (temp_critical_threshold - 8'd5)) begin
                        thermal_state <= THERMAL_ALERT;
                        thermal_timer <= '0;
                    end
                end
                
                THERMAL_EMERGENCY: begin
                    // Maximum throttling
                    current_throttle_level <= 3'd7;
                    
                    // If temperature doesn't drop within 1000 cycles, shutdown
                    if (thermal_timer > 16'd1000) begin
                        thermal_state <= THERMAL_SHUTDOWN;
                    end else if (max_temperature < (temp_emergency_threshold - 8'd10)) begin
                        thermal_state <= THERMAL_CRITICAL;
                        thermal_timer <= '0;
                    end
                end
                
                THERMAL_SHUTDOWN: begin
                    // Emergency shutdown - requires manual reset
                    current_throttle_level <= 3'd7;
                    // Stay in shutdown until manual intervention
                end
                
                default: thermal_state <= THERMAL_NORMAL;
            endcase
        end else begin
            // Thermal management disabled
            thermal_state <= THERMAL_NORMAL;
            current_throttle_level <= 3'd0;
        end
    end

    // Throttling control logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_thermal_throttle <= '0;
            ai_unit_thermal_throttle <= '0;
            memory_thermal_throttle <= 1'b0;
            throttle_counter <= '0;
        end else begin
            throttle_counter <= throttle_counter + 1;
            
            // Apply throttling based on current level and policy
            case (current_throttle_level)
                3'd0: begin // No throttling
                    core_thermal_throttle <= '0;
                    ai_unit_thermal_throttle <= '0;
                    memory_thermal_throttle <= 1'b0;
                end
                
                3'd1: begin // Light throttling - reduce AI units first
                    core_thermal_throttle <= '0;
                    ai_unit_thermal_throttle <= 2'b01; // Throttle one AI unit
                    memory_thermal_throttle <= 1'b0;
                end
                
                3'd2: begin // Medium throttling
                    core_thermal_throttle <= 4'b0001; // Throttle one core
                    ai_unit_thermal_throttle <= 2'b11; // Throttle both AI units
                    memory_thermal_throttle <= 1'b0;
                end
                
                3'd3: begin // Aggressive throttling
                    core_thermal_throttle <= 4'b0011; // Throttle two cores
                    ai_unit_thermal_throttle <= 2'b11; // Throttle both AI units
                    memory_thermal_throttle <= (throttle_counter[2:0] < 3'd4); // 50% duty cycle
                end
                
                3'd7: begin // Maximum throttling
                    core_thermal_throttle <= 4'b1111; // Throttle all cores
                    ai_unit_thermal_throttle <= 2'b11; // Throttle both AI units
                    memory_thermal_throttle <= 1'b1; // Throttle memory
                end
                
                default: begin
                    core_thermal_throttle <= '0;
                    ai_unit_thermal_throttle <= '0;
                    memory_thermal_throttle <= 1'b0;
                end
            endcase
        end
    end

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temp_alert_threshold <= TEMP_ALERT_THRESH;
            temp_critical_threshold <= TEMP_CRITICAL_THRESH;
            temp_emergency_threshold <= TEMP_EMERGENCY_THRESH;
            power_budget <= DEFAULT_POWER_BUDGET;
            thermal_mgmt_enable <= 1'b1;
            thermal_policy <= 3'd0;
            thermal_config_rdata <= '0;
            thermal_config_ready <= 1'b0;
        end else begin
            thermal_config_ready <= thermal_config_req;
            
            if (thermal_config_req && thermal_config_we) begin
                case (thermal_config_addr[7:0])
                    8'h00: thermal_mgmt_enable <= thermal_config_wdata[0];
                    8'h04: temp_alert_threshold <= thermal_config_wdata[7:0];
                    8'h08: temp_critical_threshold <= thermal_config_wdata[7:0];
                    8'h0C: temp_emergency_threshold <= thermal_config_wdata[7:0];
                    8'h10: power_budget <= thermal_config_wdata[15:0];
                    8'h14: thermal_policy <= thermal_config_wdata[2:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (thermal_config_req && !thermal_config_we) begin
                case (thermal_config_addr[7:0])
                    8'h00: thermal_config_rdata <= {31'b0, thermal_mgmt_enable};
                    8'h04: thermal_config_rdata <= {24'b0, temp_alert_threshold};
                    8'h08: thermal_config_rdata <= {24'b0, temp_critical_threshold};
                    8'h0C: thermal_config_rdata <= {24'b0, temp_emergency_threshold};
                    8'h10: thermal_config_rdata <= {16'b0, power_budget};
                    8'h14: thermal_config_rdata <= {29'b0, thermal_policy};
                    8'h18: thermal_config_rdata <= {24'b0, max_temperature};
                    8'h1C: thermal_config_rdata <= {24'b0, avg_temperature};
                    8'h20: thermal_config_rdata <= {16'b0, total_power_consumption};
                    8'h24: thermal_config_rdata <= {16'b0, power_budget_remaining};
                    8'h28: thermal_config_rdata <= {29'b0, thermal_zone};
                    8'h2C: thermal_config_rdata <= {29'b0, thermal_state};
                    8'h30: thermal_config_rdata <= {29'b0, current_throttle_level};
                    8'h34: thermal_config_rdata <= {28'b0, core_thermal_throttle};
                    8'h38: thermal_config_rdata <= {30'b0, ai_unit_thermal_throttle};
                    8'h3C: thermal_config_rdata <= {30'b0, power_budget_exceeded, memory_thermal_throttle};
                    default: thermal_config_rdata <= 32'h0;
                endcase
            end
        end
    end

    // Output assignments
    assign thermal_throttle_level = current_throttle_level;
    assign thermal_alert = (thermal_state >= THERMAL_ALERT);
    assign thermal_critical = (thermal_state >= THERMAL_CRITICAL);
    assign thermal_emergency = (thermal_state >= THERMAL_EMERGENCY);

endmodule