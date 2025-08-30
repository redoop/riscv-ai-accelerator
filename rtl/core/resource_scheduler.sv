/*
 * Workload-Aware Resource Scheduler
 * 
 * This module implements intelligent resource scheduling based on workload
 * characteristics and performance requirements for the RISC-V AI accelerator.
 */

module resource_scheduler #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter NUM_MEMORY_CHANNELS = 4,
    parameter SCHEDULER_WINDOW = 256
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Workload characterization inputs
    input  logic [7:0]                    workload_type [NUM_CORES-1:0],
    input  logic [15:0]                   workload_priority [NUM_CORES-1:0],
    input  logic [15:0]                   workload_deadline [NUM_CORES-1:0],
    input  logic [15:0]                   workload_memory_intensity [NUM_CORES-1:0],
    input  logic [15:0]                   workload_compute_intensity [NUM_CORES-1:0],
    
    // AI workload inputs
    input  logic [7:0]                    ai_workload_type [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_workload_size [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   ai_workload_priority [NUM_AI_UNITS-1:0],
    input  logic                          ai_workload_valid [NUM_AI_UNITS-1:0],
    
    // Resource availability
    input  logic [NUM_CORES-1:0]          core_available,
    input  logic [15:0]                   core_load [NUM_CORES-1:0],
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_available,
    input  logic [15:0]                   ai_unit_load [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   memory_bandwidth_available,
    input  logic [15:0]                   noc_bandwidth_available,
    
    // Performance feedback
    input  logic [15:0]                   core_performance [NUM_CORES-1:0],
    input  logic [15:0]                   ai_unit_performance [NUM_AI_UNITS-1:0],
    input  logic [15:0]                   memory_performance,
    input  logic [15:0]                   noc_performance,
    
    // Power and thermal constraints
    input  logic [15:0]                   power_budget_remaining,
    input  logic [15:0]                   thermal_headroom,
    input  logic [15:0]                   core_power_estimate [NUM_CORES-1:0],
    input  logic [15:0]                   ai_unit_power_estimate [NUM_AI_UNITS-1:0],
    
    // Scheduling outputs
    output logic [NUM_CORES-1:0]          core_allocation_enable,
    output logic [3:0]                    core_allocation_priority [NUM_CORES-1:0],
    output logic [15:0]                   core_time_slice [NUM_CORES-1:0],
    
    output logic [NUM_AI_UNITS-1:0]       ai_unit_allocation_enable,
    output logic [3:0]                    ai_unit_allocation_priority [NUM_AI_UNITS-1:0],
    output logic [15:0]                   ai_unit_time_slice [NUM_AI_UNITS-1:0],
    
    output logic [3:0]                    memory_allocation_policy,
    output logic [15:0]                   memory_bandwidth_allocation [NUM_MEMORY_CHANNELS-1:0],
    output logic [3:0]                    noc_routing_priority,
    output logic [15:0]                   noc_bandwidth_allocation,
    
    // QoS and fairness outputs
    output logic [15:0]                   qos_violation_count,
    output logic [15:0]                   fairness_index,
    output logic [15:0]                   starvation_prevention_active,
    
    // Status and metrics
    output logic [31:0]                   total_scheduled_tasks,
    output logic [15:0]                   average_response_time,
    output logic [15:0]                   resource_utilization_efficiency,
    output logic                          scheduler_active,
    
    // Configuration interface
    input  logic [31:0]                   config_addr,
    input  logic [31:0]                   config_wdata,
    output logic [31:0]                   config_rdata,
    input  logic                          config_req,
    input  logic                          config_we,
    output logic                          config_ready
);

    // Workload type definitions
    localparam logic [7:0] WORKLOAD_CPU_INTENSIVE    = 8'd0;
    localparam logic [7:0] WORKLOAD_MEMORY_INTENSIVE = 8'd1;
    localparam logic [7:0] WORKLOAD_AI_INFERENCE     = 8'd2;
    localparam logic [7:0] WORKLOAD_AI_TRAINING      = 8'd3;
    localparam logic [7:0] WORKLOAD_MIXED            = 8'd4;
    localparam logic [7:0] WORKLOAD_REAL_TIME        = 8'd5;
    localparam logic [7:0] WORKLOAD_BATCH            = 8'd6;
    localparam logic [7:0] WORKLOAD_INTERACTIVE      = 8'd7;

    // Scheduling algorithm types
    typedef enum logic [2:0] {
        SCHED_ROUND_ROBIN,
        SCHED_PRIORITY_BASED,
        SCHED_DEADLINE_AWARE,
        SCHED_WORKLOAD_AWARE,
        SCHED_POWER_AWARE,
        SCHED_ADAPTIVE
    } scheduling_algorithm_t;

    // Internal state
    logic [31:0] scheduler_cycle_count;
    logic [31:0] last_schedule_cycle;
    logic [15:0] schedule_interval;
    
    // Task tracking
    logic [31:0] task_arrival_time [NUM_CORES-1:0];
    logic [31:0] task_completion_time [NUM_CORES-1:0];
    logic [31:0] task_response_time_sum;
    logic [15:0] completed_task_count;
    
    // AI task tracking
    logic [31:0] ai_task_arrival_time [NUM_AI_UNITS-1:0];
    logic [31:0] ai_task_completion_time [NUM_AI_UNITS-1:0];
    
    // Fairness tracking
    logic [31:0] core_service_time [NUM_CORES-1:0];
    logic [31:0] ai_unit_service_time [NUM_AI_UNITS-1:0];
    logic [31:0] total_service_time;
    
    // QoS violation tracking
    logic [15:0] deadline_misses;
    logic [15:0] priority_inversions;
    logic [15:0] starvation_events;
    
    // Configuration registers
    scheduling_algorithm_t current_algorithm;
    logic scheduler_enable;
    logic [15:0] min_time_slice;
    logic [15:0] max_time_slice;
    logic [7:0]  priority_boost_factor;
    logic [15:0] starvation_threshold;
    logic [15:0] deadline_urgency_threshold;
    
    // Workload analysis results
    logic [15:0] cpu_workload_demand;
    logic [15:0] memory_workload_demand;
    logic [15:0] ai_workload_demand;
    logic [15:0] workload_balance_metric;
    
    // Resource allocation decisions
    logic [15:0] core_allocation_score [NUM_CORES-1:0];
    logic [15:0] ai_unit_allocation_score [NUM_AI_UNITS-1:0];
    logic [3:0]  best_core_match [NUM_CORES-1:0];
    logic [3:0]  best_ai_unit_match [NUM_AI_UNITS-1:0];

    // Cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scheduler_cycle_count <= '0;
            last_schedule_cycle <= '0;
        end else if (scheduler_enable) begin
            scheduler_cycle_count <= scheduler_cycle_count + 1;
            
            if ((scheduler_cycle_count - last_schedule_cycle) >= {{16{1'b0}}, schedule_interval}) begin
                last_schedule_cycle <= scheduler_cycle_count;
            end
        end
    end

    // Workload demand analysis
    always_comb begin
        automatic logic [31:0] cpu_demand_sum = 0;
        automatic logic [31:0] memory_demand_sum = 0;
        automatic logic [31:0] ai_demand_sum = 0;
        
        // Analyze CPU workload demand
        for (int i = 0; i < NUM_CORES; i++) begin
            case (workload_type[i])
                WORKLOAD_CPU_INTENSIVE: begin
                    cpu_demand_sum += {{16{1'b0}}, workload_compute_intensity[i]};
                end
                WORKLOAD_MEMORY_INTENSIVE: begin
                    memory_demand_sum += {{16{1'b0}}, workload_memory_intensity[i]};
                end
                WORKLOAD_MIXED: begin
                    cpu_demand_sum += {{17{1'b0}}, workload_compute_intensity[i][15:1]};
                    memory_demand_sum += {{17{1'b0}}, workload_memory_intensity[i][15:1]};
                end
                default: begin
                    cpu_demand_sum += {{17{1'b0}}, workload_compute_intensity[i][15:1]};
                end
            endcase
        end
        
        // Analyze AI workload demand
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_workload_valid[i]) begin
                ai_demand_sum += {{16{1'b0}}, ai_workload_size[i]};
            end
        end
        
        cpu_workload_demand = cpu_demand_sum[15:0];
        memory_workload_demand = memory_demand_sum[15:0];
        ai_workload_demand = ai_demand_sum[15:0];
        
        // Calculate workload balance metric
        automatic logic [31:0] total_demand = cpu_demand_sum + memory_demand_sum + ai_demand_sum;
        if (total_demand > 0) begin
            workload_balance_metric = 16'hFFFF - 
                ((cpu_demand_sum > memory_demand_sum ? cpu_demand_sum - memory_demand_sum : 
                  memory_demand_sum - cpu_demand_sum) * 16'hFFFF / total_demand)[15:0];
        end else begin
            workload_balance_metric = 16'hFFFF;
        end
    end

    // Core allocation scoring
    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c++) begin : gen_core_allocation
            always_comb begin
                automatic logic [31:0] score = 0;
                automatic logic [15:0] workload_match_score = 0;
                automatic logic [15:0] performance_score = 0;
                automatic logic [15:0] power_score = 0;
                automatic logic [15:0] fairness_score = 0;
                
                if (core_available[c]) begin
                    // Workload matching score
                    case (workload_type[c])
                        WORKLOAD_CPU_INTENSIVE: begin
                            workload_match_score = 16'hFFFF - core_load[c];
                        end
                        WORKLOAD_MEMORY_INTENSIVE: begin
                            workload_match_score = memory_bandwidth_available;
                        end
                        WORKLOAD_REAL_TIME: begin
                            workload_match_score = 16'hFFFF - core_load[c] + 
                                                  (16'hFFFF - workload_deadline[c]);
                        end
                        default: begin
                            workload_match_score = 16'h8000 - (core_load[c] >> 1);
                        end
                    endcase
                    
                    // Performance score
                    performance_score = core_performance[c];
                    
                    // Power efficiency score
                    if (core_power_estimate[c] > 0) begin
                        power_score = (core_performance[c] * 16'd1000) / core_power_estimate[c];
                    end else begin
                        power_score = 16'hFFFF;
                    end
                    
                    // Fairness score (favor cores with less service time)
                    if (total_service_time > 0) begin
                        fairness_score = 16'hFFFF - 
                            ((core_service_time[c] * 16'hFFFF) / total_service_time)[15:0];
                    end else begin
                        fairness_score = 16'h8000;
                    end
                    
                    // Combine scores with weights
                    score = ({{16{1'b0}}, workload_match_score} * 32'd40 +
                            {{16{1'b0}}, performance_score} * 32'd30 +
                            {{16{1'b0}}, power_score} * 32'd20 +
                            {{16{1'b0}}, fairness_score} * 32'd10) / 32'd100;
                    
                    core_allocation_score[c] = score[15:0];
                end else begin
                    core_allocation_score[c] = 16'd0;
                end
            end
        end
    endgenerate

    // AI unit allocation scoring
    genvar a;
    generate
        for (a = 0; a < NUM_AI_UNITS; a++) begin : gen_ai_allocation
            always_comb begin
                automatic logic [31:0] score = 0;
                automatic logic [15:0] workload_match_score = 0;
                automatic logic [15:0] performance_score = 0;
                automatic logic [15:0] power_score = 0;
                automatic logic [15:0] fairness_score = 0;
                
                if (ai_unit_available[a] && ai_workload_valid[a]) begin
                    // Workload matching score based on AI workload type
                    case (ai_workload_type[a])
                        WORKLOAD_AI_INFERENCE: begin
                            workload_match_score = 16'hFFFF - ai_unit_load[a];
                        end
                        WORKLOAD_AI_TRAINING: begin
                            workload_match_score = (16'hFFFF - ai_unit_load[a]) + 
                                                  (ai_workload_size[a] >> 2);
                        end
                        default: begin
                            workload_match_score = 16'h8000 - (ai_unit_load[a] >> 1);
                        end
                    endcase
                    
                    // Performance score
                    performance_score = ai_unit_performance[a];
                    
                    // Power efficiency score
                    if (ai_unit_power_estimate[a] > 0) begin
                        power_score = (ai_unit_performance[a] * 16'd1000) / ai_unit_power_estimate[a];
                    end else begin
                        power_score = 16'hFFFF;
                    end
                    
                    // Fairness score
                    if (total_service_time > 0) begin
                        fairness_score = 16'hFFFF - 
                            ((ai_unit_service_time[a] * 16'hFFFF) / total_service_time)[15:0];
                    end else begin
                        fairness_score = 16'h8000;
                    end
                    
                    // Priority boost for high-priority AI tasks
                    if (ai_workload_priority[a] > 16'hC000) begin
                        workload_match_score = workload_match_score + 16'h2000;
                    end
                    
                    // Combine scores
                    score = ({{16{1'b0}}, workload_match_score} * 32'd40 +
                            {{16{1'b0}}, performance_score} * 32'd30 +
                            {{16{1'b0}}, power_score} * 32'd20 +
                            {{16{1'b0}}, fairness_score} * 32'd10) / 32'd100;
                    
                    ai_unit_allocation_score[a] = score[15:0];
                end else begin
                    ai_unit_allocation_score[a] = 16'd0;
                end
            end
        end
    endgenerate

    // Scheduling decision logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_allocation_enable <= '0;
            core_allocation_priority <= '{default: 0};
            core_time_slice <= '{default: 0};
            ai_unit_allocation_enable <= '0;
            ai_unit_allocation_priority <= '{default: 0};
            ai_unit_time_slice <= '{default: 0};
            memory_allocation_policy <= 4'd1;
            memory_bandwidth_allocation <= '{default: 0};
            noc_routing_priority <= 4'd2;
            noc_bandwidth_allocation <= 16'h8000;
            scheduler_active <= 1'b0;
        end else if (scheduler_enable && 
                    (scheduler_cycle_count - last_schedule_cycle) == 0) begin
            
            scheduler_active <= 1'b1;
            
            // Core scheduling based on current algorithm
            case (current_algorithm)
                SCHED_WORKLOAD_AWARE: begin
                    // Allocate cores based on workload matching scores
                    for (int i = 0; i < NUM_CORES; i++) begin
                        if (core_allocation_score[i] > 16'h4000) begin
                            core_allocation_enable[i] <= 1'b1;
                            
                            // Set priority based on workload type and urgency
                            if (workload_type[i] == WORKLOAD_REAL_TIME) begin
                                core_allocation_priority[i] <= 4'd15; // Highest priority
                            end else if (workload_deadline[i] < deadline_urgency_threshold) begin
                                core_allocation_priority[i] <= 4'd12; // High priority
                            end else begin
                                core_allocation_priority[i] <= 4'd8; // Normal priority
                            end
                            
                            // Calculate time slice based on workload characteristics
                            case (workload_type[i])
                                WORKLOAD_CPU_INTENSIVE: begin
                                    core_time_slice[i] <= max_time_slice;
                                end
                                WORKLOAD_MEMORY_INTENSIVE: begin
                                    core_time_slice[i] <= min_time_slice + 
                                        (workload_memory_intensity[i] >> 2);
                                end
                                WORKLOAD_INTERACTIVE: begin
                                    core_time_slice[i] <= min_time_slice;
                                end
                                default: begin
                                    core_time_slice[i] <= (min_time_slice + max_time_slice) >> 1;
                                end
                            endcase
                        end else begin
                            core_allocation_enable[i] <= 1'b0;
                        end
                    end
                end
                
                SCHED_PRIORITY_BASED: begin
                    // Simple priority-based scheduling
                    for (int i = 0; i < NUM_CORES; i++) begin
                        if (core_available[i] && workload_priority[i] > 16'h8000) begin
                            core_allocation_enable[i] <= 1'b1;
                            core_allocation_priority[i] <= workload_priority[i][15:12];
                            core_time_slice[i] <= min_time_slice + 
                                (workload_priority[i] >> 4);
                        end else begin
                            core_allocation_enable[i] <= 1'b0;
                        end
                    end
                end
                
                SCHED_POWER_AWARE: begin
                    // Power-aware scheduling
                    for (int i = 0; i < NUM_CORES; i++) begin
                        if (core_available[i] && 
                            core_power_estimate[i] < (power_budget_remaining >> 2)) begin
                            core_allocation_enable[i] <= 1'b1;
                            core_allocation_priority[i] <= 4'd8;
                            core_time_slice[i] <= min_time_slice;
                        end else begin
                            core_allocation_enable[i] <= 1'b0;
                        end
                    end
                end
                
                default: begin // SCHED_ROUND_ROBIN
                    // Simple round-robin scheduling
                    for (int i = 0; i < NUM_CORES; i++) begin
                        if (core_available[i]) begin
                            core_allocation_enable[i] <= 1'b1;
                            core_allocation_priority[i] <= 4'd8;
                            core_time_slice[i] <= (min_time_slice + max_time_slice) >> 1;
                        end else begin
                            core_allocation_enable[i] <= 1'b0;
                        end
                    end
                end
            endcase
            
            // AI unit scheduling
            for (int i = 0; i < NUM_AI_UNITS; i++) begin
                if (ai_unit_allocation_score[i] > 16'h4000) begin
                    ai_unit_allocation_enable[i] <= 1'b1;
                    ai_unit_allocation_priority[i] <= ai_workload_priority[i][15:12];
                    
                    // Time slice based on workload size
                    ai_unit_time_slice[i] <= min_time_slice + (ai_workload_size[i] >> 4);
                end else begin
                    ai_unit_allocation_enable[i] <= 1'b0;
                end
            end
            
            // Memory allocation policy based on workload characteristics
            if (memory_workload_demand > cpu_workload_demand) begin
                memory_allocation_policy <= 4'd3; // Favor memory-intensive workloads
            end else if (ai_workload_demand > (cpu_workload_demand + memory_workload_demand)) begin
                memory_allocation_policy <= 4'd4; // Favor AI workloads
            end else begin
                memory_allocation_policy <= 4'd1; // Balanced allocation
            end
            
            // Distribute memory bandwidth
            automatic logic [31:0] total_demand = {{16{1'b0}}, cpu_workload_demand} + 
                                                 {{16{1'b0}}, memory_workload_demand} + 
                                                 {{16{1'b0}}, ai_workload_demand};
            if (total_demand > 0) begin
                for (int i = 0; i < NUM_MEMORY_CHANNELS; i++) begin
                    memory_bandwidth_allocation[i] <= 
                        (memory_bandwidth_available * {{16{1'b0}}, memory_workload_demand}) / 
                        total_demand[15:0];
                end
            end
            
            // NoC routing priority based on workload mix
            if (ai_workload_demand > 16'h8000) begin
                noc_routing_priority <= 4'd3; // High bandwidth routing for AI
            end else if (memory_workload_demand > 16'h8000) begin
                noc_routing_priority <= 4'd2; // Balanced routing
            end else begin
                noc_routing_priority <= 4'd1; // Low latency routing for CPU
            end
            
        end else begin
            scheduler_active <= 1'b0;
        end
    end

    // Service time tracking for fairness
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_service_time <= '{default: 0};
            ai_unit_service_time <= '{default: 0};
            total_service_time <= '0;
        end else if (scheduler_enable) begin
            for (int i = 0; i < NUM_CORES; i++) begin
                if (core_allocation_enable[i]) begin
                    core_service_time[i] <= core_service_time[i] + 1;
                    total_service_time <= total_service_time + 1;
                end
            end
            
            for (int i = 0; i < NUM_AI_UNITS; i++) begin
                if (ai_unit_allocation_enable[i]) begin
                    ai_unit_service_time[i] <= ai_unit_service_time[i] + 1;
                    total_service_time <= total_service_time + 1;
                end
            end
        end
    end

    // QoS and fairness metrics calculation
    always_comb begin
        // QoS violation count
        qos_violation_count = deadline_misses + priority_inversions + starvation_events;
        
        // Fairness index calculation (Jain's fairness index)
        automatic logic [63:0] service_sum = 0;
        automatic logic [63:0] service_sum_squares = 0;
        automatic logic [15:0] active_resources = 0;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_service_time[i] > 0) begin
                service_sum += {{32{1'b0}}, core_service_time[i]};
                service_sum_squares += {{32{1'b0}}, core_service_time[i]} * 
                                      {{32{1'b0}}, core_service_time[i]};
                active_resources += 1;
            end
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            if (ai_unit_service_time[i] > 0) begin
                service_sum += {{32{1'b0}}, ai_unit_service_time[i]};
                service_sum_squares += {{32{1'b0}}, ai_unit_service_time[i]} * 
                                      {{32{1'b0}}, ai_unit_service_time[i]};
                active_resources += 1;
            end
        end
        
        if (active_resources > 0 && service_sum_squares > 0) begin
            automatic logic [63:0] numerator = service_sum * service_sum;
            automatic logic [63:0] denominator = {{48{1'b0}}, active_resources} * service_sum_squares;
            fairness_index = (numerator * 64'd65535 / denominator)[15:0];
        end else begin
            fairness_index = 16'hFFFF; // Perfect fairness when no activity
        end
        
        // Starvation prevention active indicator
        starvation_prevention_active = starvation_events;
    end

    // Performance metrics
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_scheduled_tasks <= '0;
            task_response_time_sum <= '0;
            completed_task_count <= '0;
            average_response_time <= '0;
        end else if (scheduler_enable) begin
            // Track task completions and response times
            for (int i = 0; i < NUM_CORES; i++) begin
                if (core_allocation_enable[i] && !core_available[i]) begin
                    // Task completed
                    total_scheduled_tasks <= total_scheduled_tasks + 1;
                    completed_task_count <= completed_task_count + 1;
                    
                    // Calculate response time
                    automatic logic [31:0] response_time = scheduler_cycle_count - task_arrival_time[i];
                    task_response_time_sum <= task_response_time_sum + response_time;
                end
            end
            
            // Calculate average response time
            if (completed_task_count > 0) begin
                average_response_time <= (task_response_time_sum / {{16{1'b0}}, completed_task_count})[15:0];
            end
        end
    end

    // Resource utilization efficiency
    always_comb begin
        automatic logic [31:0] total_capacity = 0;
        automatic logic [31:0] utilized_capacity = 0;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            total_capacity += 32'hFFFF;
            if (core_allocation_enable[i]) begin
                utilized_capacity += {{16{1'b0}}, core_load[i]};
            end
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            total_capacity += 32'hFFFF;
            if (ai_unit_allocation_enable[i]) begin
                utilized_capacity += {{16{1'b0}}, ai_unit_load[i]};
            end
        end
        
        if (total_capacity > 0) begin
            resource_utilization_efficiency = (utilized_capacity * 32'hFFFF / total_capacity)[15:0];
        end else begin
            resource_utilization_efficiency = 16'd0;
        end
    end

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scheduler_enable <= 1'b1;
            current_algorithm <= SCHED_WORKLOAD_AWARE;
            schedule_interval <= SCHEDULER_WINDOW;
            min_time_slice <= 16'd64;
            max_time_slice <= 16'd1024;
            priority_boost_factor <= 8'd2;
            starvation_threshold <= 16'd4096;
            deadline_urgency_threshold <= 16'd1024;
            config_rdata <= '0;
            config_ready <= 1'b0;
        end else begin
            config_ready <= config_req;
            
            if (config_req && config_we) begin
                case (config_addr[7:0])
                    8'h00: scheduler_enable <= config_wdata[0];
                    8'h04: current_algorithm <= scheduling_algorithm_t'(config_wdata[2:0]);
                    8'h08: schedule_interval <= config_wdata[15:0];
                    8'h0C: min_time_slice <= config_wdata[15:0];
                    8'h10: max_time_slice <= config_wdata[15:0];
                    8'h14: priority_boost_factor <= config_wdata[7:0];
                    8'h18: starvation_threshold <= config_wdata[15:0];
                    8'h1C: deadline_urgency_threshold <= config_wdata[15:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (config_req && !config_we) begin
                case (config_addr[7:0])
                    8'h00: config_rdata <= {31'b0, scheduler_enable};
                    8'h04: config_rdata <= {29'b0, current_algorithm};
                    8'h08: config_rdata <= {16'b0, schedule_interval};
                    8'h0C: config_rdata <= {16'b0, min_time_slice};
                    8'h10: config_rdata <= {16'b0, max_time_slice};
                    8'h14: config_rdata <= {24'b0, priority_boost_factor};
                    8'h18: config_rdata <= {16'b0, starvation_threshold};
                    8'h1C: config_rdata <= {16'b0, deadline_urgency_threshold};
                    // Status and metrics
                    8'h20: config_rdata <= total_scheduled_tasks;
                    8'h24: config_rdata <= {16'b0, average_response_time};
                    8'h28: config_rdata <= {16'b0, resource_utilization_efficiency};
                    8'h2C: config_rdata <= {16'b0, qos_violation_count};
                    8'h30: config_rdata <= {16'b0, fairness_index};
                    8'h34: config_rdata <= {16'b0, cpu_workload_demand};
                    8'h38: config_rdata <= {16'b0, memory_workload_demand};
                    8'h3C: config_rdata <= {16'b0, ai_workload_demand};
                    8'h40: config_rdata <= {16'b0, workload_balance_metric};
                    8'h44: config_rdata <= {31'b0, scheduler_active};
                    default: config_rdata <= 32'h0;
                endcase
            end
        end
    end

endmodule