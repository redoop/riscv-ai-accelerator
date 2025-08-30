/*
 * Power and Thermal Optimization System Verification Test
 * 
 * Comprehensive test suite for intelligent power management,
 * thermal-aware scheduling, and power optimization algorithms.
 */

`timescale 1ns / 1ps

module test_power_thermal_optimization;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_AI_UNITS = 2;
    parameter NUM_POWER_DOMAINS = 8;
    parameter THERMAL_ZONES = 4;
    parameter MAX_TASKS = 16;
    parameter PREDICTION_WINDOW = 1024;
    parameter THERMAL_HISTORY_DEPTH = 32;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Intelligent Power Manager signals
    logic [NUM_CORES-1:0] core_active;
    logic [15:0] core_utilization [NUM_CORES-1:0];
    logic [15:0] core_performance [NUM_CORES-1:0];
    logic [NUM_AI_UNITS-1:0] ai_unit_active;
    logic [15:0] ai_unit_utilization [NUM_AI_UNITS-1:0];
    logic [15:0] ai_unit_performance [NUM_AI_UNITS-1:0];
    
    logic [15:0] domain_power [NUM_POWER_DOMAINS-1:0];
    logic [15:0] total_power;
    logic [15:0] power_budget;
    logic [15:0] battery_level;
    logic ac_power_available;
    
    logic [15:0] temperature [THERMAL_ZONES-1:0];
    logic [15:0] thermal_limits [THERMAL_ZONES-1:0];
    logic [15:0] ambient_temperature;
    logic [15:0] cooling_capacity;
    
    logic [7:0] system_mode;
    logic [15:0] qos_requirements;
    logic [31:0] deadline_pressure;
    logic emergency_thermal;
    
    // Power Manager outputs
    logic [2:0] voltage_level [NUM_POWER_DOMAINS-1:0];
    logic [2:0] frequency_level [NUM_POWER_DOMAINS-1:0];
    logic [NUM_CORES-1:0] core_power_gate;
    logic [NUM_AI_UNITS-1:0] ai_unit_power_gate;
    logic [NUM_POWER_DOMAINS-1:0] domain_power_gate;
    logic [3:0] cooling_level;
    logic [15:0] thermal_throttle_factor;
    logic thermal_emergency_shutdown;
    logic [3:0] preferred_core_mask;
    logic [1:0] preferred_ai_unit_mask;
    logic [7:0] power_aware_scheduling_policy;
    logic [15:0] predicted_power;
    logic [15:0] predicted_temperature;
    logic [15:0] power_efficiency_score;
    logic [15:0] thermal_efficiency_score;
    logic [31:0] energy_saved;
    logic power_budget_exceeded;
    logic thermal_limit_exceeded;
    logic battery_low_warning;
    logic power_optimization_active;
    
    // Thermal-Aware Scheduler signals
    logic [MAX_TASKS-1:0] task_valid;
    logic [7:0] task_type [MAX_TASKS-1:0];
    logic [15:0] task_priority [MAX_TASKS-1:0];
    logic [15:0] task_deadline [MAX_TASKS-1:0];
    logic [15:0] task_power_estimate [MAX_TASKS-1:0];
    logic [15:0] task_thermal_impact [MAX_TASKS-1:0];
    logic [7:0] task_preferred_resource [MAX_TASKS-1:0];
    
    logic [NUM_CORES-1:0] sched_core_available;
    logic [NUM_AI_UNITS-1:0] sched_ai_unit_available;
    logic [15:0] memory_bandwidth_available;
    logic [15:0] noc_bandwidth_available;
    logic [15:0] sched_core_performance [NUM_CORES-1:0];
    logic [15:0] sched_ai_unit_performance [NUM_AI_UNITS-1:0];
    logic [15:0] memory_performance;
    logic [15:0] noc_performance;
    logic [15:0] power_budget_remaining;
    logic [15:0] thermal_headroom;
    logic [15:0] sched_core_power_estimate [NUM_CORES-1:0];
    logic [15:0] sched_ai_unit_power_estimate [NUM_AI_UNITS-1:0];
    
    logic [15:0] thermal_gradient [THERMAL_ZONES-1:0];
    logic [15:0] sched_cooling_capacity [THERMAL_ZONES-1:0];
    logic [15:0] hotspot_locations;
    
    // Scheduler outputs
    logic [MAX_TASKS-1:0] task_scheduled;
    logic [3:0] task_assigned_core [MAX_TASKS-1:0];
    logic [1:0] task_assigned_ai_unit [MAX_TASKS-1:0];
    logic [15:0] task_execution_time [MAX_TASKS-1:0];
    logic [7:0] task_thermal_zone [MAX_TASKS-1:0];
    logic [NUM_CORES-1:0] core_allocation_mask;
    logic [NUM_AI_UNITS-1:0] ai_unit_allocation_mask;
    logic [3:0] thermal_balancing_policy;
    logic [15:0] load_balancing_factor;
    logic [THERMAL_ZONES-1:0] zone_load_limit;
    logic [15:0] thermal_scheduling_efficiency;
    logic [15:0] hotspot_avoidance_score;
    logic [31:0] thermal_violations_prevented;
    logic [15:0] scheduled_tasks_count;
    logic [15:0] thermal_deadline_misses;
    logic [15:0] average_thermal_balance;
    logic scheduler_active;
    
    // Configuration interfaces
    logic [31:0] pm_config_addr, pm_config_wdata, pm_config_rdata;
    logic pm_config_req, pm_config_we, pm_config_ready;
    logic [31:0] ts_config_addr, ts_config_wdata, ts_config_rdata;
    logic ts_config_req, ts_config_we, ts_config_ready;
    
    // Test control
    int test_phase;
    int cycle_count;
    logic test_passed;
    logic [31:0] error_count;
    
    // DUT instantiation
    intelligent_power_manager #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .NUM_POWER_DOMAINS(NUM_POWER_DOMAINS),
        .PREDICTION_WINDOW(PREDICTION_WINDOW),
        .THERMAL_ZONES(THERMAL_ZONES)
    ) dut_power_manager (
        .clk(clk),
        .rst_n(rst_n),
        .core_active(core_active),
        .core_utilization(core_utilization),
        .core_performance(core_performance),
        .ai_unit_active(ai_unit_active),
        .ai_unit_utilization(ai_unit_utilization),
        .ai_unit_performance(ai_unit_performance),
        .domain_power(domain_power),
        .total_power(total_power),
        .power_budget(power_budget),
        .battery_level(battery_level),
        .ac_power_available(ac_power_available),
        .temperature(temperature),
        .thermal_limits(thermal_limits),
        .ambient_temperature(ambient_temperature),
        .cooling_capacity(cooling_capacity),
        .system_mode(system_mode),
        .qos_requirements(qos_requirements),
        .deadline_pressure(deadline_pressure),
        .emergency_thermal(emergency_thermal),
        .voltage_level(voltage_level),
        .frequency_level(frequency_level),
        .core_power_gate(core_power_gate),
        .ai_unit_power_gate(ai_unit_power_gate),
        .domain_power_gate(domain_power_gate),
        .cooling_level(cooling_level),
        .thermal_throttle_factor(thermal_throttle_factor),
        .thermal_emergency_shutdown(thermal_emergency_shutdown),
        .preferred_core_mask(preferred_core_mask),
        .preferred_ai_unit_mask(preferred_ai_unit_mask),
        .power_aware_scheduling_policy(power_aware_scheduling_policy),
        .predicted_power(predicted_power),
        .predicted_temperature(predicted_temperature),
        .power_efficiency_score(power_efficiency_score),
        .thermal_efficiency_score(thermal_efficiency_score),
        .energy_saved(energy_saved),
        .power_budget_exceeded(power_budget_exceeded),
        .thermal_limit_exceeded(thermal_limit_exceeded),
        .battery_low_warning(battery_low_warning),
        .power_optimization_active(power_optimization_active),
        .config_addr(pm_config_addr),
        .config_wdata(pm_config_wdata),
        .config_rdata(pm_config_rdata),
        .config_req(pm_config_req),
        .config_we(pm_config_we),
        .config_ready(pm_config_ready)
    );
    
    thermal_aware_scheduler #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .THERMAL_ZONES(THERMAL_ZONES),
        .MAX_TASKS(MAX_TASKS),
        .THERMAL_HISTORY_DEPTH(THERMAL_HISTORY_DEPTH)
    ) dut_thermal_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .task_valid(task_valid),
        .task_type(task_type),
        .task_priority(task_priority),
        .task_deadline(task_deadline),
        .task_power_estimate(task_power_estimate),
        .task_thermal_impact(task_thermal_impact),
        .task_preferred_resource(task_preferred_resource),
        .core_available(sched_core_available),
        .core_utilization(core_utilization),
        .core_performance(sched_core_performance),
        .ai_unit_available(sched_ai_unit_available),
        .ai_unit_utilization(ai_unit_utilization),
        .ai_unit_performance(sched_ai_unit_performance),
        .temperature(temperature),
        .thermal_limits(thermal_limits),
        .thermal_gradient(thermal_gradient),
        .cooling_capacity(sched_cooling_capacity),
        .hotspot_locations(hotspot_locations),
        .core_power(sched_core_power_estimate),
        .ai_unit_power(sched_ai_unit_power_estimate),
        .total_power(total_power),
        .power_budget(power_budget),
        .cooling_level(cooling_level),
        .thermal_throttle_factor(thermal_throttle_factor),
        .thermal_emergency(emergency_thermal),
        .task_scheduled(task_scheduled),
        .task_assigned_core(task_assigned_core),
        .task_assigned_ai_unit(task_assigned_ai_unit),
        .task_execution_time(task_execution_time),
        .task_thermal_zone(task_thermal_zone),
        .core_allocation_mask(core_allocation_mask),
        .ai_unit_allocation_mask(ai_unit_allocation_mask),
        .thermal_balancing_policy(thermal_balancing_policy),
        .load_balancing_factor(load_balancing_factor),
        .zone_load_limit(zone_load_limit),
        .thermal_scheduling_efficiency(thermal_scheduling_efficiency),
        .hotspot_avoidance_score(hotspot_avoidance_score),
        .thermal_violations_prevented(thermal_violations_prevented),
        .scheduled_tasks_count(scheduled_tasks_count),
        .thermal_deadline_misses(thermal_deadline_misses),
        .average_thermal_balance(average_thermal_balance),
        .scheduler_active(scheduler_active),
        .config_addr(ts_config_addr),
        .config_wdata(ts_config_wdata),
        .config_rdata(ts_config_rdata),
        .config_req(ts_config_req),
        .config_we(ts_config_we),
        .config_ready(ts_config_ready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        $display("Starting Power and Thermal Optimization System Test");
        
        // Initialize signals
        rst_n = 0;
        test_phase = 0;
        cycle_count = 0;
        test_passed = 1;
        error_count = 0;
        
        // Initialize all input signals
        core_active = '0;
        ai_unit_active = '0;
        system_mode = 8'd1; // Balanced mode
        qos_requirements = 16'h8000;
        deadline_pressure = 32'h0;
        emergency_thermal = 1'b0;
        ac_power_available = 1'b1;
        
        // Initialize power metrics
        for (int i = 0; i < NUM_CORES; i++) begin
            core_utilization[i] = 16'h4000; // 25% utilization
            core_performance[i] = 16'h8000; // Nominal performance
            sched_core_available[i] = 1'b1;
            sched_core_performance[i] = 16'h8000;
            sched_core_power_estimate[i] = 16'h0800; // 2W
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h2000; // 12.5% utilization
            ai_unit_performance[i] = 16'h8000;
            sched_ai_unit_available[i] = 1'b1;
            sched_ai_unit_performance[i] = 16'h8000;
            sched_ai_unit_power_estimate[i] = 16'h1400; // 5W
        end
        
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            domain_power[i] = 16'h0800; // 2W per domain
        end
        
        total_power = 16'h2000; // 8W total
        power_budget = 16'h3C00; // 15W budget
        battery_level = 16'hC000; // 75% battery
        
        // Initialize thermal metrics
        ambient_temperature = 16'h1900; // 25°C
        cooling_capacity = 16'hD000; // 81.25% cooling capacity
        
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h2800 + (i * 16'h0200); // 40-46°C
            thermal_limits[i] = 16'h5500; // 85°C limit
            thermal_gradient[i] = 16'h0400; // 1°C gradient
            sched_cooling_capacity[i] = 16'hD000;
        end
        
        hotspot_locations = 16'h0002; // Zone 1 is hotspot
        
        // Initialize task queue
        task_valid = '0;
        for (int i = 0; i < MAX_TASKS; i++) begin
            task_type[i] = 8'd0;
            task_priority[i] = 16'h8000;
            task_deadline[i] = 16'hFFFF;
            task_power_estimate[i] = 16'h0400;
            task_thermal_impact[i] = 16'h0200;
            task_preferred_resource[i] = 8'd0;
        end
        
        // Configuration interface initialization
        pm_config_addr = 32'h0;
        pm_config_wdata = 32'h0;
        pm_config_req = 1'b0;
        pm_config_we = 1'b0;
        ts_config_addr = 32'h0;
        ts_config_wdata = 32'h0;
        ts_config_req = 1'b0;
        ts_config_we = 1'b0;
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
        
        $display("Reset complete, starting test phases");
        
        // Test Phase 1: Basic power management
        test_phase = 1;
        $display("Phase 1: Testing basic power management");
        test_basic_power_management();
        
        // Test Phase 2: Thermal management
        test_phase = 2;
        $display("Phase 2: Testing thermal management");
        test_thermal_management();
        
        // Test Phase 3: Power optimization modes
        test_phase = 3;
        $display("Phase 3: Testing power optimization modes");
        test_power_optimization_modes();
        
        // Test Phase 4: Thermal-aware scheduling
        test_phase = 4;
        $display("Phase 4: Testing thermal-aware scheduling");
        test_thermal_aware_scheduling();
        
        // Test Phase 5: Emergency thermal protection
        test_phase = 5;
        $display("Phase 5: Testing emergency thermal protection");
        test_emergency_thermal_protection();
        
        // Test Phase 6: Battery optimization
        test_phase = 6;
        $display("Phase 6: Testing battery optimization");
        test_battery_optimization();
        
        // Test Phase 7: Predictive optimization
        test_phase = 7;
        $display("Phase 7: Testing predictive optimization");
        test_predictive_optimization();
        
        // Test Phase 8: Configuration interfaces
        test_phase = 8;
        $display("Phase 8: Testing configuration interfaces");
        test_configuration_interfaces();
        
        // Final results
        if (test_passed && error_count == 0) begin
            $display("✓ All Power and Thermal Optimization tests PASSED");
        end else begin
            $display("✗ Power and Thermal Optimization tests FAILED with %0d errors", error_count);
        end
        
        $finish;
    end
    
    // Test tasks
    task test_basic_power_management();
        $display("  Testing basic power management functionality");
        
        // Set up normal operating conditions
        core_active = 4'b1111;
        ai_unit_active = 2'b11;
        system_mode = 8'd1; // Balanced mode
        
        repeat(100) @(posedge clk);
        
        // Check power efficiency calculation
        if (power_efficiency_score == 16'h0) begin
            $display("    ✗ Power efficiency score not calculated");
            error_count++;
        end else begin
            $display("    ✓ Power efficiency score calculated: %h", power_efficiency_score);
        end
        
        // Check thermal efficiency calculation
        if (thermal_efficiency_score == 16'h0) begin
            $display("    ✗ Thermal efficiency score not calculated");
            error_count++;
        end else begin
            $display("    ✓ Thermal efficiency score calculated: %h", thermal_efficiency_score);
        end
        
        // Check power budget monitoring
        total_power = 16'h4000; // 16W - over budget
        repeat(50) @(posedge clk);
        
        if (!power_budget_exceeded) begin
            $display("    ✗ Power budget exceeded alert not triggered");
            error_count++;
        end else begin
            $display("    ✓ Power budget exceeded alert triggered");
        end
        
        // Reset to normal power
        total_power = 16'h2000; // 8W
        repeat(50) @(posedge clk);
    endtask
    
    task test_thermal_management();
        $display("  Testing thermal management");
        
        // Create thermal stress scenario
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h5000 + (i * 16'h0100); // 80-83°C
        end
        
        repeat(200) @(posedge clk);
        
        // Check thermal limit detection
        if (!thermal_limit_exceeded) begin
            $display("    ✗ Thermal limit exceeded alert not triggered");
            error_count++;
        end else begin
            $display("    ✓ Thermal limit exceeded alert triggered");
        end
        
        // Check cooling response
        if (cooling_level <= 4'd2) begin
            $display("    ✗ Cooling level not increased for high temperature");
            error_count++;
        end else begin
            $display("    ✓ Cooling level increased: %d", cooling_level);
        end
        
        // Check thermal throttling
        if (thermal_throttle_factor >= 16'hFFFF) begin
            $display("    ✗ Thermal throttling not activated");
            error_count++;
        end else begin
            $display("    ✓ Thermal throttling activated: factor %h", thermal_throttle_factor);
        end
        
        // Reset to normal temperature
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h2800 + (i * 16'h0200); // 40-46°C
        end
        repeat(100) @(posedge clk);
    endtask
    
    task test_power_optimization_modes();
        $display("  Testing power optimization modes");
        
        // Test Performance mode
        system_mode = 8'd0; // Performance mode
        total_power = 16'h1800; // 6W - low power
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h2000; // 32°C - cool
        end
        
        repeat(300) @(posedge clk);
        
        // Should increase voltage/frequency for performance
        logic performance_boost = 1'b0;
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            if (voltage_level[i] > 3'd4 || frequency_level[i] > 3'd3) begin
                performance_boost = 1'b1;
                break;
            end
        end
        
        if (!performance_boost) begin
            $display("    ✗ Performance mode not boosting voltage/frequency");
            error_count++;
        end else begin
            $display("    ✓ Performance mode boosting voltage/frequency");
        end
        
        // Test Power Saver mode
        system_mode = 8'd2; // Power saver mode
        for (int i = 0; i < NUM_CORES; i++) begin
            core_utilization[i] = 16'h1000; // Low utilization
        end
        
        repeat(300) @(posedge clk);
        
        // Should power gate underutilized cores
        if (core_power_gate == '0) begin
            $display("    ✗ Power saver mode not gating underutilized cores");
            error_count++;
        end else begin
            $display("    ✓ Power saver mode gating cores: %b", core_power_gate);
        end
        
        // Reset to balanced mode
        system_mode = 8'd1;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_utilization[i] = 16'h4000;
        end
        repeat(100) @(posedge clk);
    endtask
    
    task test_thermal_aware_scheduling();
        $display("  Testing thermal-aware scheduling");
        
        // Create tasks with different thermal characteristics
        task_valid = 16'h00FF; // 8 tasks
        
        // High thermal impact tasks
        for (int i = 0; i < 4; i++) begin
            task_type[i] = 8'd0; // CPU intensive
            task_priority[i] = 16'h8000;
            task_thermal_impact[i] = 16'h8000; // High thermal impact
            task_power_estimate[i] = 16'h1000; // 4W
        end
        
        // Low thermal impact tasks
        for (int i = 4; i < 8; i++) begin
            task_type[i] = 8'd1; // AI inference
            task_priority[i] = 16'h6000;
            task_thermal_impact[i] = 16'h2000; // Low thermal impact
            task_power_estimate[i] = 16'h0800; // 2W
        end
        
        // Create thermal imbalance
        temperature[0] = 16'h4800; // 72°C - hot zone
        temperature[1] = 16'h2800; // 40°C - cool zone
        temperature[2] = 16'h3000; // 48°C - moderate
        temperature[3] = 16'h2C00; // 44°C - cool
        
        repeat(500) @(posedge clk);
        
        // Check if scheduler is active
        if (!scheduler_active) begin
            $display("    ✗ Thermal-aware scheduler not active");
            error_count++;
        end else begin
            $display("    ✓ Thermal-aware scheduler active");
        end
        
        // Check task scheduling
        if (scheduled_tasks_count == 16'd0) begin
            $display("    ✗ No tasks scheduled");
            error_count++;
        end else begin
            $display("    ✓ Tasks scheduled: %d", scheduled_tasks_count);
        end
        
        // Check thermal balancing
        if (thermal_balancing_policy == 4'd0) begin
            $display("    ✗ Thermal balancing policy not set");
            error_count++;
        end else begin
            $display("    ✓ Thermal balancing policy: %d", thermal_balancing_policy);
        end
        
        // Check hotspot avoidance
        if (hotspot_avoidance_score < 16'h4000) begin
            $display("    ⚠ Low hotspot avoidance score: %h", hotspot_avoidance_score);
        end else begin
            $display("    ✓ Good hotspot avoidance score: %h", hotspot_avoidance_score);
        end
        
        // Reset tasks
        task_valid = '0;
        repeat(100) @(posedge clk);
    endtask
    
    task test_emergency_thermal_protection();
        $display("  Testing emergency thermal protection");
        
        // Create emergency thermal condition
        emergency_thermal = 1'b1;
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h5800; // 88°C - emergency level
        end
        
        repeat(100) @(posedge clk);
        
        // Check emergency shutdown signal
        if (!thermal_emergency_shutdown) begin
            $display("    ✗ Emergency thermal shutdown not triggered");
            error_count++;
        end else begin
            $display("    ✓ Emergency thermal shutdown triggered");
        end
        
        // Check aggressive power reduction
        logic emergency_power_reduction = 1'b0;
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            if (voltage_level[i] <= 3'd1 && frequency_level[i] <= 3'd1) begin
                emergency_power_reduction = 1'b1;
                break;
            end
        end
        
        if (!emergency_power_reduction) begin
            $display("    ✗ Emergency power reduction not applied");
            error_count++;
        end else begin
            $display("    ✓ Emergency power reduction applied");
        end
        
        // Check resource gating
        if (core_power_gate == '0 && ai_unit_power_gate == '0) begin
            $display("    ✗ Emergency resource gating not applied");
            error_count++;
        end else begin
            $display("    ✓ Emergency resource gating applied");
        end
        
        // Check maximum cooling
        if (cooling_level != 4'hF) begin
            $display("    ✗ Maximum cooling not applied");
            error_count++;
        end else begin
            $display("    ✓ Maximum cooling applied");
        end
        
        // Clear emergency condition
        emergency_thermal = 1'b0;
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h3000; // 48°C - safe level
        end
        repeat(200) @(posedge clk);
    endtask
    
    task test_battery_optimization();
        $display("  Testing battery optimization");
        
        // Create low battery scenario
        ac_power_available = 1'b0;
        battery_level = 16'h2000; // 12.5% battery
        
        repeat(200) @(posedge clk);
        
        // Check battery low warning
        if (!battery_low_warning) begin
            $display("    ✗ Battery low warning not triggered");
            error_count++;
        end else begin
            $display("    ✓ Battery low warning triggered");
        end
        
        // Check power optimization activation
        if (!power_optimization_active) begin
            $display("    ✗ Power optimization not activated for low battery");
            error_count++;
        end else begin
            $display("    ✓ Power optimization activated for low battery");
        end
        
        // Check aggressive power saving
        logic battery_power_saving = 1'b0;
        if (core_power_gate != '0 || ai_unit_power_gate != '0) begin
            battery_power_saving = 1'b1;
        end
        
        for (int i = 0; i < NUM_POWER_DOMAINS; i++) begin
            if (voltage_level[i] < 3'd3 || frequency_level[i] < 3'd3) begin
                battery_power_saving = 1'b1;
                break;
            end
        end
        
        if (!battery_power_saving) begin
            $display("    ✗ Battery power saving measures not applied");
            error_count++;
        end else begin
            $display("    ✓ Battery power saving measures applied");
        end
        
        // Restore AC power
        ac_power_available = 1'b1;
        battery_level = 16'hC000; // 75% battery
        repeat(100) @(posedge clk);
    endtask
    
    task test_predictive_optimization();
        $display("  Testing predictive optimization");
        
        // Create power trend scenario
        for (int cycle = 0; cycle < 10; cycle++) begin
            total_power = 16'h2000 + (cycle * 16'h0200); // Increasing power trend
            repeat(100) @(posedge clk);
        end
        
        // Check power prediction
        if (predicted_power <= total_power) begin
            $display("    ✗ Power prediction not showing increasing trend");
            error_count++;
        end else begin
            $display("    ✓ Power prediction shows trend: current=%h, predicted=%h", 
                    total_power, predicted_power);
        end
        
        // Create thermal trend scenario
        for (int cycle = 0; cycle < 8; cycle++) begin
            for (int i = 0; i < THERMAL_ZONES; i++) begin
                temperature[i] = temperature[i] + 16'h0100; // Increasing temperature
            end
            repeat(100) @(posedge clk);
        end
        
        // Check thermal prediction
        if (predicted_temperature <= temperature[0]) begin
            $display("    ✗ Thermal prediction not showing increasing trend");
            error_count++;
        end else begin
            $display("    ✓ Thermal prediction shows trend: current=%h, predicted=%h", 
                    temperature[0], predicted_temperature);
        end
        
        // Check proactive optimization
        if (!power_optimization_active) begin
            $display("    ✗ Proactive optimization not triggered by predictions");
            error_count++;
        end else begin
            $display("    ✓ Proactive optimization triggered by predictions");
        end
        
        // Reset to stable conditions
        total_power = 16'h2000;
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            temperature[i] = 16'h2800;
        end
        repeat(100) @(posedge clk);
    endtask
    
    task test_configuration_interfaces();
        $display("  Testing configuration interfaces");
        
        // Test power manager configuration
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h0; // Disable power management
        pm_config_req = 1'b1;
        pm_config_we = 1'b1;
        @(posedge clk);
        
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 1'b0;
        @(posedge clk);
        
        // Read back configuration
        pm_config_addr = 32'h00;
        pm_config_req = 1'b1;
        pm_config_we = 1'b0;
        @(posedge clk);
        
        while (!pm_config_ready) @(posedge clk);
        
        if (pm_config_rdata[0] != 1'b0) begin
            $display("    ✗ Power manager configuration write/read failed");
            error_count++;
        end else begin
            $display("    ✓ Power manager configuration interface working");
        end
        
        pm_config_req = 1'b0;
        @(posedge clk);
        
        // Test thermal scheduler configuration
        ts_config_addr = 32'h04;
        ts_config_wdata = 32'h1000; // Set thermal threshold margin
        ts_config_req = 1'b1;
        ts_config_we = 1'b1;
        @(posedge clk);
        
        while (!ts_config_ready) @(posedge clk);
        ts_config_req = 1'b0;
        @(posedge clk);
        
        // Read back
        ts_config_addr = 32'h04;
        ts_config_req = 1'b1;
        ts_config_we = 1'b0;
        @(posedge clk);
        
        while (!ts_config_ready) @(posedge clk);
        
        if (ts_config_rdata[15:0] != 16'h1000) begin
            $display("    ✗ Thermal scheduler configuration write/read failed");
            error_count++;
        end else begin
            $display("    ✓ Thermal scheduler configuration interface working");
        end
        
        ts_config_req = 1'b0;
        @(posedge clk);
        
        // Re-enable power management
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h1;
        pm_config_req = 1'b1;
        pm_config_we = 1'b1;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 1'b0;
        @(posedge clk);
    endtask
    
    // Monitor cycle count
    always @(posedge clk) begin
        if (rst_n) begin
            cycle_count <= cycle_count + 1;
        end
    end
    
    // Timeout watchdog
    initial begin
        #100000000; // 100ms timeout
        $display("✗ Test timeout - possible deadlock or infinite loop");
        test_passed = 0;
        $finish;
    end

endmodule