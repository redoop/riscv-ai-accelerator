/*
 * Real-time Performance Monitor and Adaptive Tuning Controller
 * 
 * This module provides comprehensive performance monitoring and implements
 * adaptive performance tuning algorithms for the RISC-V AI accelerator.
 */

module performance_monitor #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter MONITOR_WINDOW = 1024,
    parameter TUNING_INTERVAL = 4096
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Core performance inputs
    input  logic [NUM_CORES-1:0]          core_active,
    input  logic [15:0]                   core_ipc [NUM_CORES-1:0],
    input  logic [15:0]                   core_cache_miss_rate [NUM_CORES-1:0],
    input  logic [15:0]                   core_branch_miss_rate [NUM_CORES-1:0],
    input  logic [15:0]                   core_load [NUM_CORES-1:0],
    
    // AI accelerator performance inputs
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_active,
    input  logic [15:0]                   ai_unit_utilization [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_unit_throughput [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_unit_efficiency [NUM_AI_UNITS-1:0],
    
    // Memory subsystem performance
    input  logic [15:0]                   memory_bandwidth_util,
    input  logic [15:0]                   l1_hit_rate,
    input  logic [15:0]                   l2_hit_rate,
    input  logic [15:0]                   l3_hit_rate,
    input  logic [15:0]                   memory_latency,
    
    // NoC performance inputs
    input  logic [15:0]                   noc_utilization,
    input  logic [15:0]                   noc_latency,
    input  logic [15:0]                   noc_congestion_level,
    
    // Power and thermal inputs
    input  logic [15:0]                   current_power,
    input  logic [15:0]                   temperature,
    input  logic [2:0]                    voltage_level,
    input  logic [2:0]                    frequency_level,
    
    // Workload characteristics
    input  logic [7:0]                    workload_type,  // 0=CPU, 1=AI, 2=Mixed
    input  logic [15:0]                   workload_intensity,
    input  logic [15:0]                   workload_memory_intensity,
    
    // Performance tuning outputs
    output logic [2:0]                    recommended_voltage,
    output logic [2:0]                    recommended_frequency,
    output logic [NUM_CORES-1:0]          core_power_gate_enable,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_power_gate_enable,
    output logic [3:0]                    cache_prefetch_aggressiveness,
    output logic [3:0]                    memory_scheduler_policy,
    output logic [3:0]                    noc_routing_policy,
    
    // Performance metrics outputs
    output logic [31:0]                   overall_performance_score,
    output logic [15:0]                   energy_efficiency_score,
    output logic [15:0]                   thermal_efficiency_score,
    output logic [15:0]                   resource_utilization_score,
    
    // Alerts and status
    output logic                          performance_degradation_alert,
    output logic                          thermal_throttling_needed,
    output logic                          power_budget_exceeded,
    output logic                          tuning_active,
    
    // Configuration interface
    input  logic [31:0]                   config_addr,
    input  logic [31:0]                   config_wdata,
    output logic [31:0]                   config_rdata,
    input  logic                          config_req,
    input  logic                          config_we,
    output logic                          config_ready
);

    // Performance monitoring state
    logic [31:0] monitor_cycle_count;
    logic [31:0] tuning_cycle_count;
    logic [31:0] performance_history [16];
    logic [3:0]  history_write_ptr;
    
    // Adaptive tuning parameters
    logic [15:0] target_performance_score;
    logic [15:0] performance_threshold_low;
    logic [15:0] performance_threshold_high;
    logic [7:0]  tuning_aggressiveness;
    logic        adaptive_tuning_enable;
    
    // Workload analysis
    logic [15:0] cpu_workload_weight;
    logic [15:0] ai_workload_weight;
    logic [15:0] memory_workload_weight;
    logic [15:0] workload_stability_metric;
    
    // Performance prediction
    logic [15:0] predicted_performance;
    logic [15:0] predicted_power;
    logic [15:0] predicted_temperature;
    
    // Tuning state machine
    typedef enum logic [2:0] {
        TUNING_IDLE,
        TUNING_ANALYZE,
        TUNING_PREDICT,
        TUNING_OPTIMIZE,
        TUNING_APPLY,
        TUNING_VALIDATE
    } tuning_state_t;
    
    tuning_state_t tuning_state;
    logic [7:0] tuning_step_counter;
    
    // Configuration registers
    logic monitor_enable;
    logic [15:0] monitor_window_size;
    logic [15:0] tuning_interval_cycles;
    logic [15:0] power_budget_limit;
    logic [15:0] thermal_limit;
    
    // Performance calculation weights
    localparam logic [7:0] IPC_WEIGHT = 8'd30;
    localparam logic [7:0] CACHE_WEIGHT = 8'd20;
    localparam logic [7:0] AI_UTIL_WEIGHT = 8'd25;
    localparam logic [7:0] MEMORY_WEIGHT = 8'd15;
    localparam logic [7:0] NOC_WEIGHT = 8'd10;

    // Cycle counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            monitor_cycle_count <= '0;
            tuning_cycle_count <= '0;
        end else if (monitor_enable) begin
            monitor_cycle_count <= monitor_cycle_count + 1;
            tuning_cycle_count <= tuning_cycle_count + 1;
            
            if (tuning_cycle_count >= tuning_interval_cycles) begin
                tuning_cycle_count <= '0;
            end
        end
    end

    // Workload analysis
    always_comb begin
        // Analyze current workload characteristics
        automatic logic [31:0] total_core_activity = 0;
        automatic logic [31:0] total_ai_activity = 0;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_active[i]) begin
                total_core_activity += {{16{1'b0}}, core_load[i]};
            end
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_unit_active[i]) begin
                total_ai_activity += {{16{1'b0}}, ai_unit_utilization[i]};
            end
        end
        
        // Calculate workload weights
        cpu_workload_weight = total_core_activity[15:0];
        ai_workload_weight = total_ai_activity[15:0];
        memory_workload_weight = memory_bandwidth_util;
        
        // Calculate workload stability (simplified)
        workload_stability_metric = 16'hFFFF - 
            ((cpu_workload_weight >> 2) + (ai_workload_weight >> 2));
    end

    // Overall performance score calculation
    always_comb begin
        automatic logic [31:0] score_accumulator = 0;
        automatic logic [31:0] weighted_ipc = 0;
        automatic logic [31:0] weighted_cache = 0;
        automatic logic [31:0] weighted_ai = 0;
        automatic logic [31:0] weighted_memory = 0;
        automatic logic [31:0] weighted_noc = 0;
        
        // Calculate weighted IPC score
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_active[i]) begin
                weighted_ipc += {{16{1'b0}}, core_ipc[i]} * {{24{1'b0}}, IPC_WEIGHT};
            end
        end
        weighted_ipc = weighted_ipc / NUM_CORES;
        
        // Calculate weighted cache performance
        weighted_cache = ({{16{1'b0}}, l1_hit_rate} + 
                         {{16{1'b0}}, l2_hit_rate} + 
                         {{16{1'b0}}, l3_hit_rate}) / 3;
        weighted_cache = weighted_cache * {{24{1'b0}}, CACHE_WEIGHT};
        
        // Calculate weighted AI performance
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_unit_active[i]) begin
                weighted_ai += {{16{1'b0}}, ai_unit_efficiency[i]} * {{24{1'b0}}, AI_UTIL_WEIGHT};
            end
        end
        if (NUM_AI_UNITS > 0) begin
            weighted_ai = weighted_ai / NUM_AI_UNITS;
        end
        
        // Calculate weighted memory performance
        weighted_memory = (16'hFFFF - {{16{1'b0}}, memory_latency}) * {{24{1'b0}}, MEMORY_WEIGHT};
        
        // Calculate weighted NoC performance
        weighted_noc = ({{16{1'b0}}, noc_utilization} - {{16{1'b0}}, noc_congestion_level}) * 
                      {{24{1'b0}}, NOC_WEIGHT};
        
        // Combine all weighted scores
        score_accumulator = (weighted_ipc + weighted_cache + weighted_ai + 
                           weighted_memory + weighted_noc) >> 8;
        
        overall_performance_score = score_accumulator;
    end

    // Energy efficiency calculation
    always_comb begin
        if (current_power > 0) begin
            energy_efficiency_score = (overall_performance_score[15:0] * 16'd1000) / current_power;
        end else begin
            energy_efficiency_score = 16'hFFFF;
        end
    end

    // Thermal efficiency calculation
    always_comb begin
        if (temperature > 0) begin
            thermal_efficiency_score = 16'hFFFF - (temperature << 4);
        end else begin
            thermal_efficiency_score = 16'hFFFF;
        end
    end

    // Resource utilization score
    always_comb begin
        automatic logic [31:0] util_sum = 0;
        automatic logic [15:0] active_resources = 0;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_active[i]) begin
                util_sum += {{16{1'b0}}, core_load[i]};
                active_resources += 1;
            end
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_unit_active[i]) begin
                util_sum += {{16{1'b0}}, ai_unit_utilization[i]};
                active_resources += 1;
            end
        end
        
        if (active_resources > 0) begin
            resource_utilization_score = util_sum[15:0] / active_resources;
        end else begin
            resource_utilization_score = 16'd0;
        end
    end

    // Performance history tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            performance_history <= '{default: 0};
            history_write_ptr <= '0;
        end else if (monitor_enable && (monitor_cycle_count[9:0] == 10'd0)) begin
            // Update history every 1024 cycles
            performance_history[history_write_ptr] <= overall_performance_score;
            history_write_ptr <= history_write_ptr + 1;
        end
    end

    // Adaptive tuning state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tuning_state <= TUNING_IDLE;
            tuning_step_counter <= '0;
            recommended_voltage <= 3'd4;  // Default to nominal voltage
            recommended_frequency <= 3'd3; // Default to nominal frequency
            core_power_gate_enable <= '0;
            ai_unit_power_gate_enable <= '0;
            cache_prefetch_aggressiveness <= 4'd2; // Moderate prefetching
            memory_scheduler_policy <= 4'd1; // Default policy
            noc_routing_policy <= 4'd0; // Default routing
            tuning_active <= 1'b0;
        end else if (adaptive_tuning_enable) begin
            case (tuning_state)
                TUNING_IDLE: begin
                    tuning_active <= 1'b0;
                    if (tuning_cycle_count == 0) begin
                        tuning_state <= TUNING_ANALYZE;
                        tuning_step_counter <= '0;
                        tuning_active <= 1'b1;
                    end
                end
                
                TUNING_ANALYZE: begin
                    tuning_step_counter <= tuning_step_counter + 1;
                    
                    // Analyze current performance vs targets
                    if (tuning_step_counter >= 8'd16) begin
                        tuning_state <= TUNING_PREDICT;
                        tuning_step_counter <= '0;
                    end
                end
                
                TUNING_PREDICT: begin
                    tuning_step_counter <= tuning_step_counter + 1;
                    
                    // Predict performance impact of potential changes
                    if (tuning_step_counter >= 8'd8) begin
                        tuning_state <= TUNING_OPTIMIZE;
                        tuning_step_counter <= '0;
                    end
                end
                
                TUNING_OPTIMIZE: begin
                    tuning_step_counter <= tuning_step_counter + 1;
                    
                    // Determine optimal settings
                    if (overall_performance_score[15:0] < performance_threshold_low) begin
                        // Performance is low - increase frequency/voltage
                        if (current_power < power_budget_limit && 
                            temperature < thermal_limit) begin
                            if (recommended_frequency < 3'd7) begin
                                recommended_frequency <= recommended_frequency + 1;
                            end
                            if (recommended_voltage < 3'd7) begin
                                recommended_voltage <= recommended_voltage + 1;
                            end
                        end
                        
                        // Increase cache prefetch aggressiveness
                        if (cache_prefetch_aggressiveness < 4'd7) begin
                            cache_prefetch_aggressiveness <= cache_prefetch_aggressiveness + 1;
                        end
                        
                        // Enable more resources
                        core_power_gate_enable <= '0;
                        ai_unit_power_gate_enable <= '0;
                        
                    end else if (overall_performance_score[15:0] > performance_threshold_high) begin
                        // Performance is high - can reduce power
                        if (recommended_frequency > 3'd1) begin
                            recommended_frequency <= recommended_frequency - 1;
                        end
                        if (recommended_voltage > 3'd1) begin
                            recommended_voltage <= recommended_voltage - 1;
                        end
                        
                        // Power gate unused resources
                        for (int i = 0; i < NUM_CORES; i++) begin
                            if (core_load[i] < 16'h1000) begin // Low utilization
                                core_power_gate_enable[i] <= 1'b1;
                            end
                        end
                        
                        for (int i = 0; i < NUM_AI_UNITS; i++) begin
                            if (ai_unit_utilization[i] < 16'h1000) begin
                                ai_unit_power_gate_enable[i] <= 1'b1;
                            end
                        end
                    end
                    
                    // Adjust memory and NoC policies based on workload
                    case (workload_type)
                        8'd0: begin // CPU-intensive
                            memory_scheduler_policy <= 4'd2; // Favor CPU requests
                            noc_routing_policy <= 4'd1; // Low-latency routing
                        end
                        8'd1: begin // AI-intensive
                            memory_scheduler_policy <= 4'd3; // Favor AI requests
                            noc_routing_policy <= 4'd2; // High-bandwidth routing
                        end
                        8'd2: begin // Mixed workload
                            memory_scheduler_policy <= 4'd1; // Balanced policy
                            noc_routing_policy <= 4'd0; // Adaptive routing
                        end
                        default: begin
                            memory_scheduler_policy <= 4'd1;
                            noc_routing_policy <= 4'd0;
                        end
                    endcase
                    
                    if (tuning_step_counter >= 8'd32) begin
                        tuning_state <= TUNING_APPLY;
                        tuning_step_counter <= '0;
                    end
                end
                
                TUNING_APPLY: begin
                    tuning_step_counter <= tuning_step_counter + 1;
                    
                    // Settings are applied via output signals
                    if (tuning_step_counter >= 8'd4) begin
                        tuning_state <= TUNING_VALIDATE;
                        tuning_step_counter <= '0;
                    end
                end
                
                TUNING_VALIDATE: begin
                    tuning_step_counter <= tuning_step_counter + 1;
                    
                    // Wait for settings to take effect and validate
                    if (tuning_step_counter >= 8'd64) begin
                        tuning_state <= TUNING_IDLE;
                        tuning_step_counter <= '0;
                    end
                end
                
                default: begin
                    tuning_state <= TUNING_IDLE;
                end
            endcase
        end
    end

    // Advanced performance prediction with machine learning-inspired model
    logic [15:0] performance_trend [8];
    logic [3:0] trend_index;
    logic [15:0] trend_slope;
    logic [15:0] workload_adaptation_factor;
    logic [15:0] thermal_impact_factor;
    
    // Performance trend tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            performance_trend <= '{default: 16'h8000};
            trend_index <= '0;
            trend_slope <= 16'h0;
        end else if (monitor_enable && (monitor_cycle_count[11:0] == 12'd0)) begin
            // Update trend every 4096 cycles
            performance_trend[trend_index] <= overall_performance_score[15:0];
            trend_index <= trend_index + 1;
            
            // Calculate trend slope (simplified linear regression)
            if (trend_index >= 4'd2) begin
                automatic logic [31:0] slope_calc = 0;
                automatic logic [15:0] current_val = performance_trend[trend_index];
                automatic logic [15:0] prev_val = performance_trend[(trend_index - 1) & 4'h7];
                
                if (current_val > prev_val) begin
                    slope_calc = {{16{1'b0}}, (current_val - prev_val)};
                end else begin
                    slope_calc = {{16{1'b0}}, (prev_val - current_val)} | 32'h80000000;
                end
                trend_slope <= slope_calc[15:0];
            end
        end
    end
    
    // Workload adaptation factor calculation
    always_comb begin
        case (workload_type)
            8'd0: workload_adaptation_factor = 16'h1200; // CPU-intensive: 1.125x
            8'd1: workload_adaptation_factor = 16'h0E00; // AI-intensive: 0.875x
            8'd2: workload_adaptation_factor = 16'h1000; // Mixed: 1.0x
            8'd5: workload_adaptation_factor = 16'h1400; // Real-time: 1.25x
            8'd6: workload_adaptation_factor = 16'h0C00; // Batch: 0.75x
            default: workload_adaptation_factor = 16'h1000; // Default: 1.0x
        endcase
    end
    
    // Thermal impact factor
    always_comb begin
        if (temperature < 16'h3000) begin // < 48C
            thermal_impact_factor = 16'h1100; // 1.0625x (cool running)
        end else if (temperature < 16'h4000) begin // < 64C
            thermal_impact_factor = 16'h1000; // 1.0x (normal)
        end else if (temperature < 16'h5000) begin // < 80C
            thermal_impact_factor = 16'h0E00; // 0.875x (warm)
        end else begin // >= 80C
            thermal_impact_factor = 16'h0C00; // 0.75x (hot)
        end
    end
    
    // Enhanced performance prediction
    always_comb begin
        // Predict performance based on proposed voltage/frequency changes
        automatic logic [31:0] freq_factor = 32'd100;
        automatic logic [31:0] volt_factor = 32'd100;
        automatic logic [31:0] base_prediction = 0;
        automatic logic [31:0] trend_adjustment = 0;
        automatic logic [31:0] workload_adjustment = 0;
        automatic logic [31:0] thermal_adjustment = 0;
        
        case (recommended_frequency)
            3'd0: freq_factor = 32'd25;   // 25% of nominal
            3'd1: freq_factor = 32'd50;   // 50% of nominal
            3'd2: freq_factor = 32'd75;   // 75% of nominal
            3'd3: freq_factor = 32'd100;  // Nominal
            3'd4: freq_factor = 32'd125;  // 125% of nominal
            3'd5: freq_factor = 32'd150;  // 150% of nominal
            3'd6: freq_factor = 32'd175;  // 175% of nominal
            3'd7: freq_factor = 32'd200;  // 200% of nominal
        endcase
        
        case (recommended_voltage)
            3'd0: volt_factor = 32'd36;   // (0.6V)^2 relative to 1.0V
            3'd1: volt_factor = 32'd49;   // (0.7V)^2
            3'd2: volt_factor = 32'd64;   // (0.8V)^2
            3'd3: volt_factor = 32'd81;   // (0.9V)^2
            3'd4: volt_factor = 32'd100;  // (1.0V)^2
            3'd5: volt_factor = 32'd121;  // (1.1V)^2
            3'd6: volt_factor = 32'd144;  // (1.2V)^2
            3'd7: volt_factor = 32'd169;  // (1.3V)^2
        endcase
        
        // Base performance prediction
        base_prediction = ({{16{1'b0}}, overall_performance_score[15:0]} * freq_factor) / 32'd100;
        
        // Trend-based adjustment
        if (trend_slope[15] == 1'b0) begin // Positive trend
            trend_adjustment = (base_prediction * {{16{1'b0}}, trend_slope}) / 32'h10000;
        end else begin // Negative trend
            trend_adjustment = (base_prediction * {{16{1'b0}}, (~trend_slope + 1)}) / 32'h10000;
            trend_adjustment = ~trend_adjustment + 1; // Make negative
        end
        
        // Workload-specific adjustment
        workload_adjustment = (base_prediction * {{16{1'b0}}, workload_adaptation_factor}) / 32'h1000;
        
        // Thermal adjustment
        thermal_adjustment = (workload_adjustment * {{16{1'b0}}, thermal_impact_factor}) / 32'h1000;
        
        // Final prediction with bounds checking
        automatic logic [31:0] final_prediction = thermal_adjustment + trend_adjustment;
        if (final_prediction > 32'hFFFF) begin
            predicted_performance = 16'hFFFF;
        end else if (final_prediction[31]) begin // Negative
            predicted_performance = 16'h0000;
        end else begin
            predicted_performance = final_prediction[15:0];
        end
        
        // Enhanced power prediction with workload and thermal factors
        automatic logic [31:0] base_power = ({{16{1'b0}}, current_power} * volt_factor * freq_factor) / 32'd10000;
        automatic logic [31:0] workload_power_factor = 0;
        
        case (workload_type)
            8'd0: workload_power_factor = 32'd110; // CPU-intensive: +10%
            8'd1: workload_power_factor = 32'd130; // AI-intensive: +30%
            8'd2: workload_power_factor = 32'd120; // Mixed: +20%
            8'd5: workload_power_factor = 32'd115; // Real-time: +15%
            8'd6: workload_power_factor = 32'd95;  // Batch: -5%
            default: workload_power_factor = 32'd100; // Default
        endcase
        
        predicted_power = ((base_power * workload_power_factor) / 32'd100)[15:0];
        
        // Temperature prediction with thermal resistance model
        automatic logic [31:0] power_delta = {{16{1'b0}}, predicted_power} - {{16{1'b0}}, current_power};
        automatic logic [31:0] temp_rise = (power_delta * 32'd8) / 32'd1; // ~8C per Watt thermal resistance
        predicted_temperature = temperature + temp_rise[15:0];
    end

    // Alert generation
    assign performance_degradation_alert = (overall_performance_score[15:0] < performance_threshold_low);
    assign thermal_throttling_needed = (temperature > thermal_limit) || 
                                      (predicted_temperature > thermal_limit);
    assign power_budget_exceeded = (current_power > power_budget_limit) || 
                                  (predicted_power[15:0] > power_budget_limit);

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            monitor_enable <= 1'b1;
            adaptive_tuning_enable <= 1'b1;
            monitor_window_size <= MONITOR_WINDOW;
            tuning_interval_cycles <= TUNING_INTERVAL;
            target_performance_score <= 16'h8000;
            performance_threshold_low <= 16'h6000;
            performance_threshold_high <= 16'hA000;
            tuning_aggressiveness <= 8'd4;
            power_budget_limit <= 16'h2000;  // 8W default
            thermal_limit <= 16'h5000;       // 80C default
            config_rdata <= '0;
            config_ready <= 1'b0;
        end else begin
            config_ready <= config_req;
            
            if (config_req && config_we) begin
                case (config_addr[7:0])
                    8'h00: monitor_enable <= config_wdata[0];
                    8'h04: adaptive_tuning_enable <= config_wdata[0];
                    8'h08: monitor_window_size <= config_wdata[15:0];
                    8'h0C: tuning_interval_cycles <= config_wdata[15:0];
                    8'h10: target_performance_score <= config_wdata[15:0];
                    8'h14: performance_threshold_low <= config_wdata[15:0];
                    8'h18: performance_threshold_high <= config_wdata[15:0];
                    8'h1C: tuning_aggressiveness <= config_wdata[7:0];
                    8'h20: power_budget_limit <= config_wdata[15:0];
                    8'h24: thermal_limit <= config_wdata[15:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (config_req && !config_we) begin
                case (config_addr[7:0])
                    8'h00: config_rdata <= {31'b0, monitor_enable};
                    8'h04: config_rdata <= {31'b0, adaptive_tuning_enable};
                    8'h08: config_rdata <= {16'b0, monitor_window_size};
                    8'h0C: config_rdata <= {16'b0, tuning_interval_cycles};
                    8'h10: config_rdata <= {16'b0, target_performance_score};
                    8'h14: config_rdata <= {16'b0, performance_threshold_low};
                    8'h18: config_rdata <= {16'b0, performance_threshold_high};
                    8'h1C: config_rdata <= {24'b0, tuning_aggressiveness};
                    8'h20: config_rdata <= {16'b0, power_budget_limit};
                    8'h24: config_rdata <= {16'b0, thermal_limit};
                    // Status and metrics
                    8'h30: config_rdata <= overall_performance_score;
                    8'h34: config_rdata <= {16'b0, energy_efficiency_score};
                    8'h38: config_rdata <= {16'b0, thermal_efficiency_score};
                    8'h3C: config_rdata <= {16'b0, resource_utilization_score};
                    8'h40: config_rdata <= {29'b0, recommended_voltage};
                    8'h44: config_rdata <= {29'b0, recommended_frequency};
                    8'h48: config_rdata <= {28'b0, cache_prefetch_aggressiveness};
                    8'h4C: config_rdata <= {28'b0, memory_scheduler_policy};
                    8'h50: config_rdata <= {28'b0, noc_routing_policy};
                    8'h54: config_rdata <= {16'b0, predicted_performance};
                    8'h58: config_rdata <= {16'b0, predicted_power};
                    8'h5C: config_rdata <= {16'b0, predicted_temperature};
                    default: config_rdata <= 32'h0;
                endcase
            end
        end
    end

endmodule