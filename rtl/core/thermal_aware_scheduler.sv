/*
 * Thermal-Aware Task Scheduler
 * 
 * Advanced task scheduling system that considers thermal characteristics,
 * hotspot avoidance, and dynamic thermal balancing across the chip.
 */

module thermal_aware_scheduler #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter THERMAL_ZONES = 4,
    parameter MAX_TASKS = 16,
    parameter THERMAL_HISTORY_DEPTH = 32
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Task queue inputs
    input  logic [MAX_TASKS-1:0]          task_valid,
    input  logic [7:0]                    task_type [MAX_TASKS-1:0],
    input  logic [15:0]                   task_priority [MAX_TASKS-1:0],
    input  logic [15:0]                   task_deadline [MAX_TASKS-1:0],
    input  logic [15:0]                   task_power_estimate [MAX_TASKS-1:0],
    input  logic [15:0]                   task_thermal_impact [MAX_TASKS-1:0],
    input  logic [7:0]                    task_preferred_resource [MAX_TASKS-1:0],
    
    // Resource availability
    input  logic [NUM_CORES-1:0]          core_available,
    input  logic [15:0]                   core_utilization [NUM_CORES-1:0],
    input  logic [15:0]                   core_performance [NUM_CORES-1:0],
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_available,
    input  logic [15:0]                   ai_unit_utilization [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_unit_performance [NUM_AI_UNITS-1:0],
    
    // Thermal monitoring inputs
    input  logic [15:0]                   temperature [THERMAL_ZONES-1:0],
    input  logic [15:0]                   thermal_limits [THERMAL_ZONES-1:0],
    input  logic [15:0]                   thermal_gradient [THERMAL_ZONES-1:0],
    input  logic [15:0]                   cooling_capacity [THERMAL_ZONES-1:0],
    input  logic [15:0]                   hotspot_locations,
    
    // Power monitoring inputs
    input  logic [15:0]                   core_power [NUM_CORES-1:0],
    input  logic [15:0]                   ai_unit_power [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   total_power,
    input  logic [15:0]                   power_budget,
    
    // Thermal management inputs
    input  logic [3:0]                    cooling_level,
    input  logic [15:0]                   thermal_throttle_factor,
    input  logic                          thermal_emergency,
    
    // Scheduling outputs
    output logic [MAX_TASKS-1:0]          task_scheduled,
    output logic [3:0]                    task_assigned_core [MAX_TASKS-1:0],
    output logic [1:0]                    task_assigned_ai_unit [MAX_TASKS-1:0],
    output logic [15:0]                   task_execution_time [MAX_TASKS-1:0],
    output logic [7:0]                    task_thermal_zone [MAX_TASKS-1:0],
    
    // Resource allocation outputs
    output logic [NUM_CORES-1:0]          core_allocation_mask,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_allocation_mask,
    output logic [3:0]                    thermal_balancing_policy,
    output logic [15:0]                   load_balancing_factor,
    
    // Thermal optimization outputs
    output logic [THERMAL_ZONES-1:0]     zone_load_limit,
    output logic [15:0]                   thermal_scheduling_efficiency,
    output logic [15:0]                   hotspot_avoidance_score,
    output logic [31:0]                   thermal_violations_prevented,
    
    // Status and metrics
    output logic [15:0]                   scheduled_tasks_count,
    output logic [15:0]                   thermal_deadline_misses,
    output logic [15:0]                   average_thermal_balance,
    output logic                          scheduler_active,
    
    // Configuration interface
    input  logic [31:0]                   config_addr,
    input  logic [31:0]                   config_wdata,
    output logic [31:0]                   config_rdata,
    input  logic                          config_req,
    input  logic                          config_we,
    output logic                          config_ready
);

    // Task types
    localparam logic [7:0] TASK_CPU_INTENSIVE    = 8'd0;
    localparam logic [7:0] TASK_AI_INFERENCE     = 8'd1;
    localparam logic [7:0] TASK_AI_TRAINING      = 8'd2;
    localparam logic [7:0] TASK_MEMORY_INTENSIVE = 8'd3;
    localparam logic [7:0] TASK_REAL_TIME        = 8'd4;
    localparam logic [7:0] TASK_BACKGROUND       = 8'd5;

    // Thermal scheduling state machine
    typedef enum logic [2:0] {
        SCHED_IDLE,
        SCHED_ANALYZE_THERMAL,
        SCHED_EVALUATE_TASKS,
        SCHED_OPTIMIZE_PLACEMENT,
        SCHED_ASSIGN_RESOURCES,
        SCHED_VALIDATE_THERMAL
    } thermal_sched_state_t;
    
    thermal_sched_state_t sched_state;
    logic [15:0] sched_cycle_counter;
    
    // Thermal history tracking
    logic [15:0] thermal_history [THERMAL_ZONES-1:0][THERMAL_HISTORY_DEPTH-1:0];
    logic [4:0]  thermal_history_ptr;
    logic [31:0] thermal_prediction_cycle;
    
    // Task scheduling matrices
    logic [15:0] task_thermal_score [MAX_TASKS-1:0];
    logic [15:0] task_resource_score [MAX_TASKS-1:0];
    logic [15:0] task_urgency_score [MAX_TASKS-1:0];
    logic [15:0] task_final_score [MAX_TASKS-1:0];
    
    // Resource thermal mapping
    logic [1:0]  core_thermal_zone [NUM_CORES-1:0];
    logic [1:0]  ai_unit_thermal_zone [NUM_AI_UNITS-1:0];
    logic [15:0] zone_thermal_load [THERMAL_ZONES-1:0];
    logic [15:0] zone_thermal_capacity [THERMAL_ZONES-1:0];
    
    // Thermal prediction and modeling
    logic [15:0] predicted_temperature [THERMAL_ZONES-1:0];
    logic [15:0] thermal_trend [THERMAL_ZONES-1:0];
    logic [15:0] thermal_stability [THERMAL_ZONES-1:0];
    
    // Load balancing state
    logic [31:0] total_scheduled_tasks;
    logic [31:0] thermal_violations_count;
    logic [15:0] thermal_balance_metric;
    
    // Configuration registers
    logic thermal_scheduling_enable;
    logic [15:0] thermal_threshold_margin;
    logic [7:0]  thermal_prediction_weight;
    logic [7:0]  hotspot_avoidance_weight;
    logic [7:0]  load_balancing_weight;
    logic [15:0] scheduling_interval;
    logic [15:0] thermal_violation_threshold;
    
    // Initialize core-to-thermal-zone mapping (platform specific)
    initial begin
        core_thermal_zone[0] = 2'd0;  // Core 0 in zone 0
        core_thermal_zone[1] = 2'd1;  // Core 1 in zone 1
        core_thermal_zone[2] = 2'd2;  // Core 2 in zone 2
        core_thermal_zone[3] = 2'd3;  // Core 3 in zone 3
        
        ai_unit_thermal_zone[0] = 2'd1;  // AI unit 0 in zone 1
        ai_unit_thermal_zone[1] = 2'd2;  // AI unit 1 in zone 2
    end

    // Main thermal scheduling state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sched_state <= SCHED_IDLE;
            sched_cycle_counter <= '0;
            thermal_prediction_cycle <= '0;
            scheduler_active <= 1'b0;
        end else if (thermal_scheduling_enable) begin
            sched_cycle_counter <= sched_cycle_counter + 1;
            
            case (sched_state)
                SCHED_IDLE: begin
                    scheduler_active <= 1'b0;
                    if (sched_cycle_counter >= scheduling_interval || thermal_emergency) begin
                        sched_state <= SCHED_ANALYZE_THERMAL;
                        sched_cycle_counter <= '0;
                        scheduler_active <= 1'b1;
                    end
                end
                
                SCHED_ANALYZE_THERMAL: begin
                    thermal_prediction_cycle <= thermal_prediction_cycle + 1;
                    if (sched_cycle_counter >= 16'd16) begin
                        sched_state <= SCHED_EVALUATE_TASKS;
                        sched_cycle_counter <= '0;
                    end
                end
                
                SCHED_EVALUATE_TASKS: begin
                    if (sched_cycle_counter >= 16'd32) begin
                        sched_state <= SCHED_OPTIMIZE_PLACEMENT;
                        sched_cycle_counter <= '0;
                    end
                end
                
                SCHED_OPTIMIZE_PLACEMENT: begin
                    if (sched_cycle_counter >= 16'd64) begin
                        sched_state <= SCHED_ASSIGN_RESOURCES;
                        sched_cycle_counter <= '0;
                    end
                end
                
                SCHED_ASSIGN_RESOURCES: begin
                    if (sched_cycle_counter >= 16'd16) begin
                        sched_state <= SCHED_VALIDATE_THERMAL;
                        sched_cycle_counter <= '0;
                    end
                end
                
                SCHED_VALIDATE_THERMAL: begin
                    if (sched_cycle_counter >= 16'd8) begin
                        sched_state <= SCHED_IDLE;
                        sched_cycle_counter <= '0;
                    end
                end
                
                default: begin
                    sched_state <= SCHED_IDLE;
                end
            endcase
        end
    end

    // Thermal history tracking and prediction
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            thermal_history <= '{default: '{default: 0}};
            thermal_history_ptr <= '0;
        end else if (thermal_scheduling_enable && sched_cycle_counter[7:0] == 8'd0) begin
            // Update thermal history every 256 cycles
            for (int z = 0; z < THERMAL_ZONES; z++) begin
                thermal_history[z][thermal_history_ptr] <= temperature[z];
            end
            thermal_history_ptr <= thermal_history_ptr + 1;
        end
    end

    // Thermal prediction using linear regression
    genvar z;
    generate
        for (z = 0; z < THERMAL_ZONES; z++) begin : gen_thermal_prediction
            always_comb begin
                // Calculate thermal trend from recent history
                automatic logic [31:0] trend_sum = 0;
                automatic logic [15:0] recent_samples = 8;
                
                for (int i = 1; i < recent_samples; i++) begin
                    automatic logic [4:0] curr_idx = thermal_history_ptr - i;
                    automatic logic [4:0] prev_idx = thermal_history_ptr - i - 1;
                    trend_sum += {{16{1'b0}}, thermal_history[z][curr_idx]} - 
                                {{16{1'b0}}, thermal_history[z][prev_idx]};
                end
                
                thermal_trend[z] = trend_sum[15:0] / (recent_samples - 1);
                
                // Predict future temperature
                predicted_temperature[z] = temperature[z] + thermal_trend[z];
                
                // Calculate thermal stability (inverse of variance)
                automatic logic [31:0] variance_sum = 0;
                automatic logic [15:0] mean_temp = temperature[z];
                
                for (int i = 0; i < 8; i++) begin
                    automatic logic [4:0] idx = thermal_history_ptr - i;
                    automatic logic [31:0] diff = {{16{1'b0}}, thermal_history[z][idx]} - {{16{1'b0}}, mean_temp};
                    variance_sum += (diff * diff) >> 16;
                end
                
                thermal_stability[z] = 16'hFFFF - variance_sum[15:0];
            end
        end
    endgenerate

    // Zone thermal load and capacity calculation
    always_comb begin
        // Initialize zone loads
        for (int z = 0; z < THERMAL_ZONES; z++) begin
            zone_thermal_load[z] = 16'd0;
            zone_thermal_capacity[z] = thermal_limits[z] - temperature[z];
        end
        
        // Accumulate thermal load from cores
        for (int c = 0; c < NUM_CORES; c++) begin
            if (core_available[c]) begin
                automatic logic [1:0] zone = core_thermal_zone[c];
                zone_thermal_load[zone] = zone_thermal_load[zone] + 
                    ((core_power[c] * core_utilization[c]) >> 12);
            end
        end
        
        // Accumulate thermal load from AI units
        for (int a = 0; a < NUM_AI_UNITS; a++) begin
            if (ai_unit_available[a]) begin
                automatic logic [1:0] zone = ai_unit_thermal_zone[a];
                zone_thermal_load[zone] = zone_thermal_load[zone] + 
                    ((ai_unit_power[a] * ai_unit_utilization[a]) >> 12);
            end
        end
    end

    // Task thermal scoring
    genvar t;
    generate
        for (t = 0; t < MAX_TASKS; t++) begin : gen_task_scoring
            always_comb begin
                if (task_valid[t]) begin
                    // Thermal impact score (lower is better for hot zones)
                    automatic logic [15:0] thermal_impact = task_thermal_impact[t];
                    automatic logic [15:0] zone_temp_factor = 16'hFFFF;
                    
                    // Find coolest suitable zone for this task
                    if (task_type[t] == TASK_CPU_INTENSIVE) begin
                        // CPU tasks - consider all core zones
                        automatic logic [15:0] min_temp = temperature[0];
                        for (int z = 1; z < THERMAL_ZONES; z++) begin
                            if (temperature[z] < min_temp) begin
                                min_temp = temperature[z];
                            end
                        end
                        zone_temp_factor = 16'hFFFF - min_temp;
                    end else if (task_type[t] == TASK_AI_INFERENCE || task_type[t] == TASK_AI_TRAINING) begin
                        // AI tasks - consider AI unit zones
                        automatic logic [15:0] ai_zone_temp = (temperature[ai_unit_thermal_zone[0]] + 
                                                             temperature[ai_unit_thermal_zone[1]]) >> 1;
                        zone_temp_factor = 16'hFFFF - ai_zone_temp;
                    end
                    
                    task_thermal_score[t] = (zone_temp_factor * thermal_impact) >> 16;
                    
                    // Resource availability score
                    automatic logic [15:0] resource_score = 16'd0;
                    if (task_type[t] == TASK_CPU_INTENSIVE) begin
                        for (int c = 0; c < NUM_CORES; c++) begin
                            if (core_available[c]) begin
                                resource_score += (16'hFFFF - core_utilization[c]) >> 2;
                            end
                        end
                    end else if (task_type[t] == TASK_AI_INFERENCE || task_type[t] == TASK_AI_TRAINING) begin
                        for (int a = 0; a < NUM_AI_UNITS; a++) begin
                            if (ai_unit_available[a]) begin
                                resource_score += (16'hFFFF - ai_unit_utilization[a]) >> 1;
                            end
                        end
                    end
                    task_resource_score[t] = resource_score;
                    
                    // Urgency score based on deadline
                    automatic logic [31:0] time_to_deadline = {{16{1'b0}}, task_deadline[t]};
                    if (time_to_deadline < 32'd1000) begin
                        task_urgency_score[t] = 16'hFFFF - time_to_deadline[15:0];
                    end else begin
                        task_urgency_score[t] = 16'h1000;
                    end
                    
                    // Final weighted score
                    automatic logic [31:0] weighted_score = 
                        ({{16{1'b0}}, task_thermal_score[t]} * {{24{1'b0}}, thermal_prediction_weight}) +
                        ({{16{1'b0}}, task_resource_score[t]} * {{24{1'b0}}, load_balancing_weight}) +
                        ({{16{1'b0}}, task_urgency_score[t]} * 32'd100);
                    
                    task_final_score[t] = weighted_score[15:0];
                    
                end else begin
                    task_thermal_score[t] = 16'd0;
                    task_resource_score[t] = 16'd0;
                    task_urgency_score[t] = 16'd0;
                    task_final_score[t] = 16'd0;
                end
            end
        end
    endgenerate

    // Task scheduling and resource assignment
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            task_scheduled <= '0;
            task_assigned_core <= '{default: 0};
            task_assigned_ai_unit <= '{default: 0};
            task_execution_time <= '{default: 0};
            task_thermal_zone <= '{default: 0};
            core_allocation_mask <= '0;
            ai_unit_allocation_mask <= '0;
            thermal_balancing_policy <= 4'd1;
            load_balancing_factor <= 16'h8000;
            scheduled_tasks_count <= '0;
            total_scheduled_tasks <= '0;
        end else if (thermal_scheduling_enable && sched_state == SCHED_ASSIGN_RESOURCES) begin
            
            // Reset allocation masks
            core_allocation_mask <= '0;
            ai_unit_allocation_mask <= '0;
            task_scheduled <= '0;
            
            // Schedule tasks in priority order (highest score first)
            automatic logic [MAX_TASKS-1:0] tasks_to_schedule = task_valid;
            automatic logic [15:0] current_scheduled = 0;
            
            // Simple priority-based scheduling (in practice, would use more sophisticated algorithm)
            for (int priority_level = 15; priority_level >= 0; priority_level--) begin
                for (int task_idx = 0; task_idx < MAX_TASKS; task_idx++) begin
                    if (tasks_to_schedule[task_idx] && 
                        task_final_score[task_idx][15:12] == priority_level[3:0]) begin
                        
                        // Try to assign this task to a resource
                        if (task_type[task_idx] == TASK_CPU_INTENSIVE || 
                            task_type[task_idx] == TASK_REAL_TIME) begin
                            
                            // Find best available core
                            automatic logic [3:0] best_core = 4'hF;
                            automatic logic [15:0] best_core_score = 16'd0;
                            
                            for (int c = 0; c < NUM_CORES; c++) begin
                                if (core_available[c] && !core_allocation_mask[c]) begin
                                    automatic logic [1:0] core_zone = core_thermal_zone[c];
                                    automatic logic [31:0] core_score = 
                                        ({{16{1'b0}}, core_performance[c]} * 32'd60) +
                                        ({{16{1'b0}}, zone_thermal_capacity[core_zone]} * 32'd40);
                                    
                                    if (core_score[15:0] > best_core_score) begin
                                        best_core_score = core_score[15:0];
                                        best_core = c[3:0];
                                    end
                                end
                            end
                            
                            // Assign task to best core if found
                            if (best_core != 4'hF) begin
                                task_scheduled[task_idx] <= 1'b1;
                                task_assigned_core[task_idx] <= best_core;
                                task_thermal_zone[task_idx] <= {6'd0, core_thermal_zone[best_core]};
                                core_allocation_mask[best_core] <= 1'b1;
                                
                                // Estimate execution time based on task priority and core performance
                                task_execution_time[task_idx] <= 
                                    (task_priority[task_idx] * 16'd100) / core_performance[best_core];
                                
                                current_scheduled = current_scheduled + 1;
                                tasks_to_schedule[task_idx] = 1'b0;
                            end
                            
                        end else if (task_type[task_idx] == TASK_AI_INFERENCE || 
                                   task_type[task_idx] == TASK_AI_TRAINING) begin
                            
                            // Find best available AI unit
                            automatic logic [1:0] best_ai_unit = 2'b11;
                            automatic logic [15:0] best_ai_score = 16'd0;
                            
                            for (int a = 0; a < NUM_AI_UNITS; a++) begin
                                if (ai_unit_available[a] && !ai_unit_allocation_mask[a]) begin
                                    automatic logic [1:0] ai_zone = ai_unit_thermal_zone[a];
                                    automatic logic [31:0] ai_score = 
                                        ({{16{1'b0}}, ai_unit_performance[a]} * 32'd70) +
                                        ({{16{1'b0}}, zone_thermal_capacity[ai_zone]} * 32'd30);
                                    
                                    if (ai_score[15:0] > best_ai_score) begin
                                        best_ai_score = ai_score[15:0];
                                        best_ai_unit = a[1:0];
                                    end
                                end
                            end
                            
                            // Assign task to best AI unit if found
                            if (best_ai_unit != 2'b11) begin
                                task_scheduled[task_idx] <= 1'b1;
                                task_assigned_ai_unit[task_idx] <= best_ai_unit;
                                task_thermal_zone[task_idx] <= {6'd0, ai_unit_thermal_zone[best_ai_unit]};
                                ai_unit_allocation_mask[best_ai_unit] <= 1'b1;
                                
                                // Estimate execution time for AI tasks
                                task_execution_time[task_idx] <= 
                                    (task_priority[task_idx] * 16'd200) / ai_unit_performance[best_ai_unit];
                                
                                current_scheduled = current_scheduled + 1;
                                tasks_to_schedule[task_idx] = 1'b0;
                            end
                        end
                    end
                end
            end
            
            scheduled_tasks_count <= current_scheduled;
            total_scheduled_tasks <= total_scheduled_tasks + {{16{1'b0}}, current_scheduled};
            
            // Update thermal balancing policy based on current thermal state
            automatic logic [15:0] max_temp = temperature[0];
            automatic logic [15:0] min_temp = temperature[0];
            
            for (int z = 1; z < THERMAL_ZONES; z++) begin
                if (temperature[z] > max_temp) max_temp = temperature[z];
                if (temperature[z] < min_temp) min_temp = temperature[z];
            end
            
            automatic logic [15:0] thermal_spread = max_temp - min_temp;
            
            if (thermal_spread > 16'h1000) begin // > 4°C spread
                thermal_balancing_policy <= 4'd3; // Aggressive balancing
                load_balancing_factor <= 16'hC000; // High load balancing
            end else if (thermal_spread > 16'h0800) begin // > 2°C spread
                thermal_balancing_policy <= 4'd2; // Moderate balancing
                load_balancing_factor <= 16'hA000; // Medium load balancing
            end else begin
                thermal_balancing_policy <= 4'd1; // Normal balancing
                load_balancing_factor <= 16'h8000; // Normal load balancing
            end
        end
    end

    // Zone load limiting
    always_comb begin
        for (int z = 0; z < THERMAL_ZONES; z++) begin
            if (temperature[z] > (thermal_limits[z] - thermal_threshold_margin)) begin
                zone_load_limit[z] = 1'b1; // Limit load in this zone
            end else begin
                zone_load_limit[z] = 1'b0; // No load limit
            end
        end
    end

    // Performance metrics calculation
    always_comb begin
        // Thermal scheduling efficiency
        automatic logic [31:0] efficiency_sum = 0;
        automatic logic [15:0] scheduled_count = 0;
        
        for (int t = 0; t < MAX_TASKS; t++) begin
            if (task_scheduled[t]) begin
                efficiency_sum += {{16{1'b0}}, task_final_score[t]};
                scheduled_count += 1;
            end
        end
        
        if (scheduled_count > 0) begin
            thermal_scheduling_efficiency = efficiency_sum[15:0] / scheduled_count;
        end else begin
            thermal_scheduling_efficiency = 16'd0;
        end
        
        // Hotspot avoidance score
        automatic logic [31:0] hotspot_score = 0;
        for (int z = 0; z < THERMAL_ZONES; z++) begin
            if (temperature[z] < thermal_limits[z]) begin
                hotspot_score += {{16{1'b0}}, thermal_limits[z]} - {{16{1'b0}}, temperature[z]};
            end
        end
        hotspot_avoidance_score = hotspot_score[15:0] / THERMAL_ZONES;
        
        // Thermal balance metric
        automatic logic [15:0] max_zone_temp = temperature[0];
        automatic logic [15:0] min_zone_temp = temperature[0];
        
        for (int z = 1; z < THERMAL_ZONES; z++) begin
            if (temperature[z] > max_zone_temp) max_zone_temp = temperature[z];
            if (temperature[z] < min_zone_temp) min_zone_temp = temperature[z];
        end
        
        average_thermal_balance = 16'hFFFF - (max_zone_temp - min_zone_temp);
    end

    // Thermal violations tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            thermal_violations_count <= '0;
            thermal_deadline_misses <= '0;
        end else if (thermal_scheduling_enable) begin
            // Count thermal violations
            for (int z = 0; z < THERMAL_ZONES; z++) begin
                if (temperature[z] > thermal_limits[z]) begin
                    thermal_violations_count <= thermal_violations_count + 1;
                end
            end
            
            // Count deadline misses due to thermal constraints
            for (int t = 0; t < MAX_TASKS; t++) begin
                if (task_valid[t] && !task_scheduled[t] && task_deadline[t] < 16'd100) begin
                    thermal_deadline_misses <= thermal_deadline_misses + 1;
                end
            end
        end
    end

    // Thermal violations prevented (estimated)
    assign thermal_violations_prevented = thermal_violations_count > 0 ? 
        thermal_violations_count - (thermal_violations_count >> 2) : 32'd0;

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            thermal_scheduling_enable <= 1'b1;
            thermal_threshold_margin <= 16'h0800; // 2°C margin
            thermal_prediction_weight <= 8'd30;
            hotspot_avoidance_weight <= 8'd40;
            load_balancing_weight <= 8'd30;
            scheduling_interval <= 16'd512;
            thermal_violation_threshold <= 16'h0400; // 1°C threshold
            config_rdata <= '0;
            config_ready <= 1'b0;
        end else begin
            config_ready <= config_req;
            
            if (config_req && config_we) begin
                case (config_addr[7:0])
                    8'h00: thermal_scheduling_enable <= config_wdata[0];
                    8'h04: thermal_threshold_margin <= config_wdata[15:0];
                    8'h08: thermal_prediction_weight <= config_wdata[7:0];
                    8'h0C: hotspot_avoidance_weight <= config_wdata[7:0];
                    8'h10: load_balancing_weight <= config_wdata[7:0];
                    8'h14: scheduling_interval <= config_wdata[15:0];
                    8'h18: thermal_violation_threshold <= config_wdata[15:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (config_req && !config_we) begin
                case (config_addr[7:0])
                    8'h00: config_rdata <= {31'b0, thermal_scheduling_enable};
                    8'h04: config_rdata <= {16'b0, thermal_threshold_margin};
                    8'h08: config_rdata <= {24'b0, thermal_prediction_weight};
                    8'h0C: config_rdata <= {24'b0, hotspot_avoidance_weight};
                    8'h10: config_rdata <= {24'b0, load_balancing_weight};
                    8'h14: config_rdata <= {16'b0, scheduling_interval};
                    8'h18: config_rdata <= {16'b0, thermal_violation_threshold};
                    // Status and metrics
                    8'h20: config_rdata <= {16'b0, scheduled_tasks_count};
                    8'h24: config_rdata <= {16'b0, thermal_deadline_misses};
                    8'h28: config_rdata <= {16'b0, average_thermal_balance};
                    8'h2C: config_rdata <= {16'b0, thermal_scheduling_efficiency};
                    8'h30: config_rdata <= {16'b0, hotspot_avoidance_score};
                    8'h34: config_rdata <= thermal_violations_prevented;
                    8'h38: config_rdata <= total_scheduled_tasks;
                    8'h3C: config_rdata <= {28'b0, thermal_balancing_policy};
                    8'h40: config_rdata <= {16'b0, load_balancing_factor};
                    8'h44: config_rdata <= {31'b0, scheduler_active};
                    default: config_rdata <= 32'h0;
                endcase
            end
        end
    end

endmodule