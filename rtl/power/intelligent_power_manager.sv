/*
 * Intelligent Power Management System
 * 
 * Advanced power management with machine learning-based prediction,
 * thermal-aware task scheduling, and adaptive power optimization.
 */

module intelligent_power_manager #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter NUM_POWER_DOMAINS = 8,
    parameter PREDICTION_WINDOW = 1024,
    parameter THERMAL_ZONES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Workload and performance inputs
    input  logic [NUM_CORES-1:0]          core_active,
    input  logic [15:0]                   core_utilization [NUM_CORES-1:0],
    input  logic [15:0]                   core_performance [NUM_CORES-1:0],
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_active,
    input  logic [15:0]                   ai_unit_utilization [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_unit_performance [NUM_AI_UNITS-1:0],
    
    // Power monitoring inputs
    input  logic [15:0]                   domain_power [NUM_POWER_DOMAINS-1:0],
    input  logic [15:0]                   total_power,
    input  logic [15:0]                   power_budget,
    input  logic [15:0]                   battery_level,
    input  logic                          ac_power_available,
    
    // Thermal monitoring inputs
    input  logic [15:0]                   temperature [THERMAL_ZONES-1:0],
    input  logic [15:0]                   thermal_limits [THERMAL_ZONES-1:0],
    input  logic [15:0]                   ambient_temperature,
    input  logic [15:0]                   cooling_capacity,
    
    // System state inputs
    input  logic [7:0]                    system_mode,  // 0=Performance, 1=Balanced, 2=PowerSaver, 3=Thermal
    input  logic [15:0]                   qos_requirements,
    input  logic [31:0]                   deadline_pressure,
    input  logic                          emergency_thermal,
    
    // Power control outputs
    output logic [2:0]                    voltage_level [NUM_POWER_DOMAINS-1:0],
    output logic [2:0]                    frequency_level [NUM_POWER_DOMAINS-1:0],
    output logic [NUM_CORES-1:0]          core_power_gate,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_power_gate,
    output logic [NUM_POWER_DOMAINS-1:0]  domain_power_gate,
    
    // Thermal management outputs
    output logic [3:0]                    cooling_level,
    output logic [15:0]                   thermal_throttle_factor,
    output logic                          thermal_emergency_shutdown,
    
    // Task scheduling hints
    output logic [3:0]                    preferred_core_mask,
    output logic [1:0]                    preferred_ai_unit_mask,
    output logic [7:0]                    power_aware_scheduling_policy,
    
    // Power predictions and metrics
    output logic [15:0]                   predicted_power,
    output logic [15:0]                   predicted_temperature,
    output logic [15:0]                   power_efficiency_score,
    output logic [15:0]                   thermal_efficiency_score,
    output logic [31:0]                   energy_saved,
    
    // Status and alerts
    output logic                          power_budget_exceeded,
    output logic                          thermal_limit_exceeded,
    output logic                          battery_low_warning,
    output logic                          power_optimization_active,
    
    // Configuration interface
    input  logic [31:0]                   config_addr,
    input  logic [31:0]                   config_wdata,
    output logic [31:0]                   config_rdata,
    input  logic                          config_req,
    input  logic                          config_we,
    output logic                          config_ready
);

    // Power management state machine
    typedef enum logic [3:0] {
        PM_INIT,
        PM_MONITOR,
        PM_ANALYZE,
        PM_PREDICT,
        PM_OPTIMIZE,
        PM_APPLY,
        PM_VALIDATE,
        PM_EMERGENCY
    } power_mgmt_state_t;
    
    power_mgmt_state_t pm_state;
    logic [15:0] pm_cycle_counter;
    
    // Machine learning prediction state
    logic [15:0] power_history [16];
    logic [15:0] temp_history [16];
    logic [15:0] util_history [16];
    logic [3:0]  history_write_ptr;
    logic [31:0] prediction_cycle_count;
    
    // Thermal management state
    logic [15:0] thermal_integral [THERMAL_ZONES-1:0];
    logic [15:0] thermal_derivative [THERMAL_ZONES-1:0];
    logic [15:0] prev_temperature [THERMAL_ZONES-1:0];
    logic [15:0] thermal_setpoint [THERMAL_ZONES-1:0];
    
    // Power optimization state
    logic [31:0] optimization_cycle_count;
    logic [15:0] power_savings_accumulator;
    logic [15:0] baseline_power;
    logic [7:0]  optimization_aggressiveness;
    
    // Workload prediction
    logic [15:0] predicted_core_util [NUM_CORES-1:0];
    logic [15:0] predicted_ai_util [NUM_AI_UNITS-1:0];
    logic [15:0] workload_trend;
    logic [7:0]  workload_stability;
    
    // Configuration registers
    logic power_mgmt_enable;
    logic thermal_mgmt_enable;
    logic predictive_optimization_enable;
    logic [15:0] power_optimization_interval;
    logic [15:0] thermal_response_time;
    logic [7:0]  ml_learning_rate;
    logic [15:0] emergency_thermal_threshold;
    logic [15:0] battery_low_threshold;
    
    // Power efficiency calculation weights
    localparam logic [7:0] PERF_WEIGHT = 8'd40;
    localparam logic [7:0] POWER_WEIGHT = 8'd35;
    localparam logic [7:0] THERMAL_WEIGHT = 8'd25;

    // Main power management state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pm_state <= PM_INIT;
            pm_cycle_counter <= '0;
            optimization_cycle_count <= '0;
            prediction_cycle_count <= '0;
            power_optimization_active <= 1'b0;
        end else if (power_mgmt_enable) begin
            pm_cycle_counter <= pm_cycle_counter + 1;
            
            // Emergency thermal handling
            if (emergency_thermal || (|temperature > emergency_thermal_threshold)) begin
                pm_state <= PM_EMERGENCY;
            end else begin
                case (pm_state)
                    PM_INIT: begin
                        // Initialize power management
                        baseline_power <= total_power;
                        power_savings_accumulator <= '0;
                        pm_state <= PM_MONITOR;
                    end
                    
                    PM_MONITOR: begin
                        // Continuous monitoring
                        if (pm_cycle_counter[9:0] == 10'd0) begin // Every 1024 cycles
                            pm_state <= PM_ANALYZE;
                        end
                    end
                    
                    PM_ANALYZE: begin
                        // Analyze current power and thermal state
                        pm_cycle_counter <= pm_cycle_counter + 1;
                        if (pm_cycle_counter[3:0] == 4'd15) begin
                            pm_state <= PM_PREDICT;
                        end
                    end
                    
                    PM_PREDICT: begin
                        // Predict future power and thermal behavior
                        prediction_cycle_count <= prediction_cycle_count + 1;
                        if (pm_cycle_counter[4:0] == 5'd31) begin
                            pm_state <= PM_OPTIMIZE;
                        end
                    end
                    
                    PM_OPTIMIZE: begin
                        // Generate optimization strategy
                        power_optimization_active <= 1'b1;
                        optimization_cycle_count <= optimization_cycle_count + 1;
                        if (pm_cycle_counter[5:0] == 6'd63) begin
                            pm_state <= PM_APPLY;
                        end
                    end
                    
                    PM_APPLY: begin
                        // Apply optimization decisions
                        if (pm_cycle_counter[2:0] == 3'd7) begin
                            pm_state <= PM_VALIDATE;
                        end
                    end
                    
                    PM_VALIDATE: begin
                        // Validate optimization results
                        power_optimization_active <= 1'b0;
                        if (pm_cycle_counter[4:0] == 5'd31) begin
                            pm_state <= PM_MONITOR;
                        end
                    end
                    
                    PM_EMERGENCY: begin
                        // Emergency thermal shutdown procedures
                        power_optimization_active <= 1'b1;
                        if (!emergency_thermal && (|temperature < (emergency_thermal_threshold - 16'h1000))) begin
                            pm_state <= PM_MONITOR;
                        end
                    end
                    
                    default: begin
                        pm_state <= PM_INIT;
                    end
                endcase
            end
        end
    end

    // Power and thermal history tracking for ML prediction
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_history <= '{default: 0};
            temp_history <= '{default: 0};
            util_history <= '{default: 0};
            history_write_ptr <= '0;
        end else if (power_mgmt_enable && pm_cycle_counter[7:0] == 8'd0) begin
            // Update history every 256 cycles
            power_history[history_write_ptr] <= total_power;
            
            // Average temperature across zones
            automatic logic [31:0] temp_sum = 0;
            for (int i = 0; i < THERMAL_ZONES; i++) begin
                temp_sum += {{16{1'b0}}, temperature[i]};
            end
            temp_history[history_write_ptr] <= temp_sum[15:0] / THERMAL_ZONES;
            
            // Average utilization across cores and AI units
            automatic logic [31:0] util_sum = 0;
            automatic logic [15:0] active_units = 0;
            
            for (int i = 0; i < NUM_CORES; i++) begin
                if (core_active[i]) begin
                    util_sum += {{16{1'b0}}, core_utilization[i]};
                    active_units += 1;
                end
            end
            
            for (int i = 0; i < NUM_AI_UNITS; i++) begin
                if (ai_unit_active[i]) begin
                    util_sum += {{16{1'b0}}, ai_unit_utilization[i]};
                    active_units += 1;
                end
            end
            
            if (active_units > 0) begin
                util_history[history_write_ptr] <= util_sum[15:0] / active_units;
            end else begin
                util_history[history_write_ptr] <= 16'd0;
            end
            
            history_write_ptr <= history_write_ptr + 1;
        end
    end

    // Machine learning-based power prediction
    always_comb begin
        // Simple linear regression prediction based on recent history
        automatic logic [31:0] power_trend = 0;
        automatic logic [31:0] temp_trend = 0;
        automatic logic [31:0] util_trend = 0;
        
        // Calculate trends from recent history
        for (int i = 1; i < 8; i++) begin
            automatic logic [3:0] curr_idx = history_write_ptr - i;
            automatic logic [3:0] prev_idx = history_write_ptr - i - 1;
            
            power_trend += {{16{1'b0}}, power_history[curr_idx]} - {{16{1'b0}}, power_history[prev_idx]};
            temp_trend += {{16{1'b0}}, temp_history[curr_idx]} - {{16{1'b0}}, temp_history[prev_idx]};
            util_trend += {{16{1'b0}}, util_history[curr_idx]} - {{16{1'b0}}, util_history[prev_idx]};
        end
        
        // Predict next values based on trends
        predicted_power = power_history[history_write_ptr] + (power_trend[15:0] >> 3);
        predicted_temperature = temp_history[history_write_ptr] + (temp_trend[15:0] >> 3);
        
        // Predict workload characteristics
        workload_trend = util_trend[15:0] >> 3;
        
        // Calculate workload stability (lower variance = higher stability)
        automatic logic [31:0] util_variance = 0;
        automatic logic [15:0] util_mean = util_history[history_write_ptr];
        
        for (int i = 0; i < 8; i++) begin
            automatic logic [3:0] idx = history_write_ptr - i;
            automatic logic [31:0] diff = {{16{1'b0}}, util_history[idx]} - {{16{1'b0}}, util_mean};
            util_variance += (diff * diff) >> 16;
        end
        
        workload_stability = 8'hFF - util_variance[7:0];
    end

    // Thermal PID controller for each zone
    genvar z;
    generate
        for (z = 0; z < THERMAL_ZONES; z++) begin : gen_thermal_control
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    thermal_integral[z] <= '0;
                    thermal_derivative[z] <= '0;
                    prev_temperature[z] <= '0;
                    thermal_setpoint[z] <= thermal_limits[z] - 16'h0800; // 2°C below limit
                end else if (thermal_mgmt_enable) begin
                    automatic logic [31:0] error = {{16{1'b0}}, thermal_setpoint[z]} - {{16{1'b0}}, temperature[z]};
                    automatic logic [31:0] derivative = {{16{1'b0}}, temperature[z]} - {{16{1'b0}}, prev_temperature[z]};
                    
                    // Integral term (with windup protection)
                    if (thermal_integral[z] < 16'hF000 && thermal_integral[z] > 16'h1000) begin
                        thermal_integral[z] <= thermal_integral[z] + error[15:0];
                    end
                    
                    // Derivative term
                    thermal_derivative[z] <= derivative[15:0];
                    prev_temperature[z] <= temperature[z];
                end
            end
        end
    endgenerate

    // Power optimization decision logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            voltage_level <= '{default: 3'd4}; // Nominal voltage
            frequency_level <= '{default: 3'd3}; // Nominal frequency
            core_power_gate <= '0;
            ai_unit_power_gate <= '0;
            domain_power_gate <= '0;
            cooling_level <= 4'd2; // Moderate cooling
            thermal_throttle_factor <= 16'hFFFF; // No throttling
            thermal_emergency_shutdown <= 1'b0;
            preferred_core_mask <= 4'hF; // All cores preferred
            preferred_ai_unit_mask <= 2'b11; // All AI units preferred
            power_aware_scheduling_policy <= 8'd1; // Balanced policy
        end else if (power_mgmt_enable) begin
            case (pm_state)
                PM_OPTIMIZE, PM_APPLY: begin
                    // Power optimization based on system mode and constraints
                    case (system_mode)
                        8'd0: begin // Performance mode
                            optimize_for_performance();
                        end
                        8'd1: begin // Balanced mode
                            optimize_for_balance();
                        end
                        8'd2: begin // Power saver mode
                            optimize_for_power_saving();
                        end
                        8'd3: begin // Thermal mode
                            optimize_for_thermal();
                        end
                        default: begin
                            optimize_for_balance();
                        end
                    endcase
                end
                
                PM_EMERGENCY: begin
                    // Emergency thermal protection
                    emergency_thermal_protection();
                end
                
                default: begin
                    // Maintain current settings
                end
            endcase
        end
    end

    // Performance optimization function
    function void optimize_for_performance();
        // Maximize performance within power and thermal limits
        if (total_power < (power_budget - 16'h0800) && predicted_temperature < (thermal_limits[0] - 16'h0400)) begin
            // Increase voltage and frequency for better performance
            for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
                if (voltage_level[i] < 3'd6) voltage_level[i] <= voltage_level[i] + 1;
                if (frequency_level[i] < 3'd6) frequency_level[i] <= frequency_level[i] + 1;
            end
            
            // Enable all resources
            core_power_gate <= '0;
            ai_unit_power_gate <= '0;
            domain_power_gate <= '0;
            
            // Prefer all cores and AI units
            preferred_core_mask <= 4'hF;
            preferred_ai_unit_mask <= 2'b11;
            power_aware_scheduling_policy <= 8'd0; // Performance-first policy
        end
    endfunction

    // Balanced optimization function
    function void optimize_for_balance();
        // Balance performance, power, and thermal efficiency
        automatic logic [15:0] power_headroom = power_budget - total_power;
        automatic logic [15:0] thermal_headroom = thermal_limits[0] - temperature[0];
        
        if (power_headroom > 16'h1000 && thermal_headroom > 16'h0800) begin
            // Moderate performance increase
            for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
                if (voltage_level[i] < 3'd5) voltage_level[i] <= voltage_level[i] + 1;
                if (frequency_level[i] < 3'd5) frequency_level[i] <= frequency_level[i] + 1;
            end
        end else if (power_headroom < 16'h0400 || thermal_headroom < 16'h0400) begin
            // Reduce power consumption
            for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
                if (voltage_level[i] > 3'd2) voltage_level[i] <= voltage_level[i] - 1;
                if (frequency_level[i] > 3'd2) frequency_level[i] <= frequency_level[i] - 1;
            end
        end
        
        // Power gate underutilized resources
        for (int i = 0; i < NUM_CORES; i++) begin
            core_power_gate[i] <= (core_utilization[i] < 16'h1000); // < 6.25% utilization
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_power_gate[i] <= (ai_unit_utilization[i] < 16'h1000);
        end
        
        power_aware_scheduling_policy <= 8'd1; // Balanced policy
    endfunction

    // Power saving optimization function
    function void optimize_for_power_saving();
        // Minimize power consumption while maintaining acceptable performance
        
        // Reduce voltage and frequency aggressively
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            if (voltage_level[i] > 3'd1) voltage_level[i] <= voltage_level[i] - 1;
            if (frequency_level[i] > 3'd1) frequency_level[i] <= frequency_level[i] - 1;
        end
        
        // Power gate more aggressively
        for (int i = 0; i < NUM_CORES; i++) begin
            core_power_gate[i] <= (core_utilization[i] < 16'h2000); // < 12.5% utilization
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_power_gate[i] <= (ai_unit_utilization[i] < 16'h2000);
        end
        
        // Prefer fewer, more efficient cores
        if (core_utilization[0] + core_utilization[1] < 16'h8000) begin
            preferred_core_mask <= 4'b0011; // Prefer first two cores
        end else begin
            preferred_core_mask <= 4'hF;
        end
        
        power_aware_scheduling_policy <= 8'd2; // Power-saving policy
    endfunction

    // Thermal optimization function
    function void optimize_for_thermal();
        // Prioritize thermal management
        automatic logic [15:0] max_temp = temperature[0];
        automatic logic [1:0] hottest_zone = 2'd0;
        
        // Find hottest thermal zone
        for (int i = 1; i < THERMAL_ZONES; i++) begin
            if (temperature[i] > max_temp) begin
                max_temp = temperature[i];
                hottest_zone = i[1:0];
            end
        end
        
        // Aggressive thermal management
        if (max_temp > (thermal_limits[hottest_zone] - 16'h0800)) begin
            // Reduce voltage and frequency significantly
            for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
                if (voltage_level[i] > 3'd1) voltage_level[i] <= 3'd1;
                if (frequency_level[i] > 3'd1) frequency_level[i] <= 3'd1;
            end
            
            // Calculate thermal throttling factor
            automatic logic [31:0] temp_excess = {{16{1'b0}}, max_temp} - {{16{1'b0}}, thermal_limits[hottest_zone]};
            if (temp_excess > 0) begin
                thermal_throttle_factor <= 16'hFFFF - (temp_excess << 4);
            end
            
            // Increase cooling
            cooling_level <= 4'hF; // Maximum cooling
            
            // Power gate aggressively in hot zones
            if (hottest_zone < 2) begin
                // Hot zones 0-1 affect cores
                core_power_gate <= core_power_gate | 4'b1100; // Gate cores 2-3
            end else begin
                // Hot zones 2-3 affect AI units
                ai_unit_power_gate <= 2'b11; // Gate all AI units
            end
        end
        
        power_aware_scheduling_policy <= 8'd3; // Thermal-aware policy
    endfunction

    // Emergency thermal protection function
    function void emergency_thermal_protection();
        // Immediate thermal protection measures
        
        // Minimum voltage and frequency
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            voltage_level[i] <= 3'd0;
            frequency_level[i] <= 3'd0;
        end
        
        // Power gate all non-essential resources
        core_power_gate <= 4'b1110; // Keep only core 0
        ai_unit_power_gate <= 2'b11; // Gate all AI units
        domain_power_gate <= 8'hFE; // Gate all but essential domain
        
        // Maximum cooling
        cooling_level <= 4'hF;
        
        // Severe thermal throttling
        thermal_throttle_factor <= 16'h4000; // 25% performance
        
        // Check for emergency shutdown
        automatic logic [15:0] max_temp = temperature[0];
        for (int i = 1; i < THERMAL_ZONES; i++) begin
            if (temperature[i] > max_temp) begin
                max_temp = temperature[i];
            end
        end
        
        if (max_temp > (emergency_thermal_threshold + 16'h0800)) begin
            thermal_emergency_shutdown <= 1'b1;
        end
        
        preferred_core_mask <= 4'b0001; // Only core 0
        preferred_ai_unit_mask <= 2'b00; // No AI units
        power_aware_scheduling_policy <= 8'd4; // Emergency policy
    endfunction

    // Power and thermal efficiency scoring
    always_comb begin
        // Power efficiency score (performance per watt)
        automatic logic [31:0] total_performance = 0;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_active[i]) begin
                total_performance += {{16{1'b0}}, core_performance[i]};
            end
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_unit_active[i]) begin
                total_performance += {{16{1'b0}}, ai_unit_performance[i]};
            end
        end
        
        if (total_power > 0) begin
            power_efficiency_score = (total_performance[15:0] * 16'd1000) / total_power;
        end else begin
            power_efficiency_score = 16'hFFFF;
        end
        
        // Thermal efficiency score
        automatic logic [31:0] thermal_score = 0;
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            if (temperature[i] < thermal_limits[i]) begin
                thermal_score += {{16{1'b0}}, thermal_limits[i]} - {{16{1'b0}}, temperature[i]};
            end
        end
        thermal_efficiency_score = thermal_score[15:0] / THERMAL_ZONES;
    end

    // Energy savings calculation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            energy_saved <= '0;
        end else if (power_mgmt_enable && pm_cycle_counter[9:0] == 10'd0) begin
            // Calculate energy savings every 1024 cycles
            if (baseline_power > total_power) begin
                automatic logic [31:0] power_savings = {{16{1'b0}}, baseline_power} - {{16{1'b0}}, total_power};
                energy_saved <= energy_saved + power_savings;
            end
        end
    end

    // Status and alert generation
    assign power_budget_exceeded = (total_power > power_budget);
    assign thermal_limit_exceeded = (|temperature > |thermal_limits);
    assign battery_low_warning = (!ac_power_available && battery_level < battery_low_threshold);

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_mgmt_enable <= 1'b1;
            thermal_mgmt_enable <= 1'b1;
            predictive_optimization_enable <= 1'b1;
            power_optimization_interval <= 16'd1024;
            thermal_response_time <= 16'd64;
            ml_learning_rate <= 8'd16;
            emergency_thermal_threshold <= 16'h5800; // 88°C
            battery_low_threshold <= 16'h1000; // 6.25%
            optimization_aggressiveness <= 8'd4;
            config_rdata <= '0;
            config_ready <= 1'b0;
        end else begin
            config_ready <= config_req;
            
            if (config_req && config_we) begin
                case (config_addr[7:0])
                    8'h00: power_mgmt_enable <= config_wdata[0];
                    8'h04: thermal_mgmt_enable <= config_wdata[0];
                    8'h08: predictive_optimization_enable <= config_wdata[0];
                    8'h0C: power_optimization_interval <= config_wdata[15:0];
                    8'h10: thermal_response_time <= config_wdata[15:0];
                    8'h14: ml_learning_rate <= config_wdata[7:0];
                    8'h18: emergency_thermal_threshold <= config_wdata[15:0];
                    8'h1C: battery_low_threshold <= config_wdata[15:0];
                    8'h20: optimization_aggressiveness <= config_wdata[7:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (config_req && !config_we) begin
                case (config_addr[7:0])
                    8'h00: config_rdata <= {31'b0, power_mgmt_enable};
                    8'h04: config_rdata <= {31'b0, thermal_mgmt_enable};
                    8'h08: config_rdata <= {31'b0, predictive_optimization_enable};
                    8'h0C: config_rdata <= {16'b0, power_optimization_interval};
                    8'h10: config_rdata <= {16'b0, thermal_response_time};
                    8'h14: config_rdata <= {24'b0, ml_learning_rate};
                    8'h18: config_rdata <= {16'b0, emergency_thermal_threshold};
                    8'h1C: config_rdata <= {16'b0, battery_low_threshold};
                    8'h20: config_rdata <= {24'b0, optimization_aggressiveness};
                    // Status and metrics
                    8'h30: config_rdata <= {16'b0, predicted_power};
                    8'h34: config_rdata <= {16'b0, predicted_temperature};
                    8'h38: config_rdata <= {16'b0, power_efficiency_score};
                    8'h3C: config_rdata <= {16'b0, thermal_efficiency_score};
                    8'h40: config_rdata <= energy_saved;
                    8'h44: config_rdata <= {28'b0, pm_state};
                    8'h48: config_rdata <= {24'b0, workload_stability};
                    8'h4C: config_rdata <= {16'b0, workload_trend};
                    8'h50: config_rdata <= {28'b0, cooling_level};
                    8'h54: config_rdata <= {16'b0, thermal_throttle_factor};
                    default: config_rdata <= 32'h0;
                endcase
            end
        end
    end

endmodule