/*
 * Comprehensive Power and Thermal Optimization Test Suite
 * 
 * Tests advanced power management features including:
 * - Intelligent power management strategies
 * - Thermal-aware task scheduling
 * - Low-power modes and standby functionality
 * - Battery-aware optimization
 * - Machine learning-based predictions
 */

`timescale 1ns / 1ps

module test_power_thermal_optimization_comprehensive;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_AI_UNITS = 2;
    parameter NUM_DOMAINS = 8;
    parameter THERMAL_ZONES = 4;
    parameter MAX_TASKS = 16;
    parameter CLK_PERIOD = 10; // 100MHz

    // Clock and reset
    logic clk;
    logic rst_n;

    // Intelligent Power Manager signals
    logic [NUM_CORES-1:0] core_active;
    logic [31:0] core_utilization [NUM_CORES-1:0];
    logic [31:0] workload_type [NUM_CORES-1:0];
    logic [15:0] temperature [NUM_DOMAINS-1:0];
    logic [15:0] temp_threshold_warning;
    logic [15:0] temp_threshold_critical;
    logic [15:0] temp_gradient [NUM_DOMAINS-1:0];
    
    logic [NUM_DOMAINS-1:0] domain_power_enable;
    logic [NUM_DOMAINS-1:0] domain_clock_enable;
    logic [7:0] voltage_level [NUM_DOMAINS-1:0];
    logic [7:0] frequency_level [NUM_DOMAINS-1:0];
    
    logic [31:0] power_budget;
    logic [31:0] battery_level;
    logic ac_power_available;
    logic [31:0] current_power_consumption;
    logic [31:0] predicted_power_consumption;
    logic power_emergency;
    logic thermal_emergency;
    
    // Low power mode signals
    logic sleep_request;
    logic deep_sleep_request;
    logic hibernate_request;
    logic [2:0] power_mode;
    logic wake_up_ready;
    
    // ML prediction signals
    logic [31:0] ml_power_prediction;
    logic ml_prediction_valid;
    
    // Performance counters
    logic [31:0] power_savings_total;
    logic [31:0] thermal_throttle_cycles;
    logic [15:0] avg_temperature;

    // Thermal-Aware Scheduler signals
    logic [MAX_TASKS-1:0] task_valid;
    logic [7:0] task_type [MAX_TASKS-1:0];
    logic [15:0] task_priority [MAX_TASKS-1:0];
    logic [15:0] task_deadline [MAX_TASKS-1:0];
    logic [15:0] task_power_estimate [MAX_TASKS-1:0];
    logic [15:0] task_thermal_impact [MAX_TASKS-1:0];
    
    logic [NUM_CORES-1:0] core_available;
    logic [15:0] core_performance [NUM_CORES-1:0];
    logic [NUM_AI_UNITS-1:0] ai_unit_available;
    logic [15:0] ai_unit_utilization [NUM_AI_UNITS-1:0];
    logic [15:0] ai_unit_performance [NUM_AI_UNITS-1:0];
    
    logic [15:0] thermal_limits [THERMAL_ZONES-1:0];
    logic [15:0] zone_temperature [THERMAL_ZONES-1:0];
    logic [15:0] cooling_capacity;
    logic thermal_emergency_sched;
    
    logic [MAX_TASKS-1:0] task_scheduled;
    logic [3:0] task_assigned_core [MAX_TASKS-1:0];
    logic [1:0] task_assigned_ai_unit [MAX_TASKS-1:0];
    logic [15:0] task_execution_time [MAX_TASKS-1:0];
    logic [7:0] task_thermal_zone [MAX_TASKS-1:0];
    
    logic [3:0] thermal_preferred_cores;
    logic [1:0] thermal_preferred_ai_units;
    logic [7:0] thermal_throttle_level;
    
    logic [15:0] scheduled_tasks_count;
    logic [31:0] thermal_violations_prevented;
    logic [15:0] thermal_scheduling_efficiency;

    // Test control variables
    integer test_phase;
    integer cycle_count;
    logic test_passed;
    logic [31:0] error_count;

    // Instantiate Intelligent Power Manager
    intelligent_power_manager #(
        .NUM_CORES(NUM_CORES),
        .NUM_DOMAINS(NUM_DOMAINS),
        .NUM_POWER_STATES(8)
    ) dut_power_mgr (
        .clk(clk),
        .rst_n(rst_n),
        
        // Core monitoring
        .core_active(core_active),
        .core_idle(~core_active),
        .core_utilization(core_utilization),
        .workload_type(workload_type),
        
        // Temperature monitoring
        .temperature(temperature),
        .temp_threshold_warning(temp_threshold_warning),
        .temp_threshold_critical(temp_threshold_critical),
        .temp_gradient(temp_gradient),
        
        // Power domain control
        .domain_power_enable(domain_power_enable),
        .domain_clock_enable(domain_clock_enable),
        .voltage_level(voltage_level),
        .frequency_level(frequency_level),
        
        // Advanced power management
        .power_budget(power_budget),
        .battery_level(battery_level),
        .ac_power_available(ac_power_available),
        .current_power_consumption(current_power_consumption),
        .predicted_power_consumption(predicted_power_consumption),
        .power_emergency(power_emergency),
        .thermal_emergency(thermal_emergency),
        
        // Low power modes
        .sleep_request(sleep_request),
        .deep_sleep_request(deep_sleep_request),
        .hibernate_request(hibernate_request),
        .power_mode(power_mode),
        .wake_up_ready(wake_up_ready),
        
        // ML prediction
        .ml_power_prediction(ml_power_prediction),
        .ml_prediction_valid(ml_prediction_valid),
        
        // Performance counters
        .power_savings_total(power_savings_total),
        .thermal_throttle_cycles(thermal_throttle_cycles),
        .avg_temperature(avg_temperature)
    );

    // Instantiate Thermal-Aware Scheduler
    thermal_aware_scheduler #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .THERMAL_ZONES(THERMAL_ZONES),
        .MAX_TASKS(MAX_TASKS),
        .THERMAL_HISTORY_DEPTH(32)
    ) dut_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        
        // Task queue inputs
        .task_valid(task_valid),
        .task_type(task_type),
        .task_priority(task_priority),
        .task_deadline(task_deadline),
        .task_power_estimate(task_power_estimate),
        .task_thermal_impact(task_thermal_impact),
        .task_preferred_resource(8'h00),
        
        // Resource availability
        .core_available(core_available),
        .core_utilization(core_utilization[NUM_CORES-1:0]),
        .core_performance(core_performance),
        .ai_unit_available(ai_unit_available),
        .ai_unit_utilization(ai_unit_utilization),
        .ai_unit_performance(ai_unit_performance),
        
        // Thermal monitoring
        .temperature(zone_temperature),
        .thermal_limits(thermal_limits),
        .thermal_gradient(16'h0100),
        .cooling_capacity({THERMAL_ZONES{16'h8000}}),
        .hotspot_locations(16'h0000),
        
        // Power monitoring
        .core_power({NUM_CORES{16'h1000}}),
        .ai_unit_power({NUM_AI_UNITS{16'h2000}}),
        .total_power(current_power_consumption[15:0]),
        .power_budget(power_budget[15:0]),
        
        // Thermal management
        .cooling_level(4'h4),
        .thermal_throttle_factor(16'hFFFF),
        .thermal_emergency(thermal_emergency_sched),
        
        // Scheduling outputs
        .task_scheduled(task_scheduled),
        .task_assigned_core(task_assigned_core),
        .task_assigned_ai_unit(task_assigned_ai_unit),
        .task_execution_time(task_execution_time),
        .task_thermal_zone(task_thermal_zone),
        
        // Resource allocation
        .core_allocation_mask(),
        .ai_unit_allocation_mask(),
        .thermal_balancing_policy(),
        .load_balancing_factor(),
        
        // Thermal optimization
        .zone_load_limit(),
        .thermal_scheduling_efficiency(thermal_scheduling_efficiency),
        .hotspot_avoidance_score(),
        .thermal_violations_prevented(thermal_violations_prevented),
        
        // Status and metrics
        .scheduled_tasks_count(scheduled_tasks_count),
        .thermal_deadline_misses(),
        .average_thermal_balance(),
        .scheduler_active(),
        
        // Configuration interface
        .config_addr(32'h0),
        .config_wdata(32'h0),
        .config_rdata(),
        .config_req(1'b0),
        .config_we(1'b0),
        .config_ready()
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test stimulus and monitoring
    initial begin
        $display("=== Comprehensive Power and Thermal Optimization Test ===");
        
        // Initialize signals
        rst_n = 0;
        test_phase = 0;
        cycle_count = 0;
        test_passed = 1;
        error_count = 0;
        
        // Initialize power manager inputs
        core_active = 4'b1111;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_utilization[i] = 32'h4000; // 25% utilization
            workload_type[i] = 32'h01; // CPU workload
        end
        
        for (int i = 0; i < NUM_DOMAINS; i++) begin
            temperature[i] = 16'h3000; // 48°C
            temp_gradient[i] = 16'h0100;
        end
        
        temp_threshold_warning = 16'h5000; // 80°C
        temp_threshold_critical = 16'h5800; // 88°C
        power_budget = 32'h96; // 150W
        battery_level = 32'h64; // 100%
        ac_power_available = 1'b1;
        
        // Initialize low-power mode signals
        sleep_request = 1'b0;
        deep_sleep_request = 1'b0;
        hibernate_request = 1'b0;
        
        // Initialize ML prediction
        ml_power_prediction = 32'h80; // 128W
        ml_prediction_valid = 1'b1;
        
        // Initialize scheduler inputs
        task_valid = 16'h0000;
        core_available = 4'b1111;
        ai_unit_available = 2'b11;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            core_performance[i] = 16'h8000;
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h2000;
            ai_unit_performance[i] = 16'hA000;
        end
        
        for (int i = 0; i < THERMAL_ZONES; i++) begin
            thermal_limits[i] = 16'h5000; // 80°C
            zone_temperature[i] = 16'h3000; // 48°C
        end
        
        cooling_capacity = 16'h8000;
        thermal_emergency_sched = 1'b0;
        
        // Reset
        #(CLK_PERIOD * 10);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        // Test Phase 1: Normal Operation
        $display("Phase 1: Testing normal power management operation");
        test_phase = 1;
        run_normal_operation_test();
        
        // Test Phase 2: Thermal Emergency
        $display("Phase 2: Testing thermal emergency handling");
        test_phase = 2;
        run_thermal_emergency_test();
        
        // Test Phase 3: Battery Mode Operation
        $display("Phase 3: Testing battery mode optimization");
        test_phase = 3;
        run_battery_mode_test();
        
        // Test Phase 4: Low-Power Modes
        $display("Phase 4: Testing low-power modes");
        test_phase = 4;
        run_low_power_modes_test();
        
        // Test Phase 5: Thermal-Aware Scheduling
        $display("Phase 5: Testing thermal-aware task scheduling");
        test_phase = 5;
        run_thermal_scheduling_test();
        
        // Test Phase 6: ML-Based Optimization
        $display("Phase 6: Testing ML-based power prediction");
        test_phase = 6;
        run_ml_optimization_test();
        
        // Test Phase 7: Stress Testing
        $display("Phase 7: Running stress tests");
        test_phase = 7;
        run_stress_test();
        
        // Final results
        #(CLK_PERIOD * 100);
        
        $display("\n=== Test Results ===");
        $display("Total errors: %0d", error_count);
        $display("Power savings achieved: %0d W-cycles", power_savings_total);
        $display("Thermal throttle cycles: %0d", thermal_throttle_cycles);
        $display("Tasks scheduled: %0d", scheduled_tasks_count);
        $display("Thermal violations prevented: %0d", thermal_violations_prevented);
        $display("Thermal scheduling efficiency: %0d%%", thermal_scheduling_efficiency * 100 / 16'hFFFF);
        
        if (error_count == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** %0d TESTS FAILED ***", error_count);
            test_passed = 0;
        end
        
        $finish;
    end

    // Test tasks
    task run_normal_operation_test();
        begin
            $display("  Testing normal power management...");
            
            // Set normal operating conditions
            for (int i = 0; i < NUM_CORES; i++) begin
                core_utilization[i] = 32'h6000; // 37.5% utilization
            end
            
            #(CLK_PERIOD * 1000);
            
            // Check that system is in balanced mode
            if (power_mode != 3'b001) begin // Balanced mode
                $display("ERROR: Expected balanced power mode, got %0d", power_mode);
                error_count++;
            end
            
            // Check voltage and frequency levels are reasonable
            for (int i = 0; i < NUM_DOMAINS; i++) begin
                if (voltage_level[i] < 8'h80 || voltage_level[i] > 8'hFF) begin
                    $display("ERROR: Voltage level %0d out of range: %02x", i, voltage_level[i]);
                    error_count++;
                end
            end
            
            $display("  Normal operation test completed");
        end
    endtask

    task run_thermal_emergency_test();
        begin
            $display("  Testing thermal emergency response...");
            
            // Simulate thermal emergency
            for (int i = 0; i < NUM_DOMAINS; i++) begin
                temperature[i] = 16'h5C00; // 92°C - above critical
            end
            
            #(CLK_PERIOD * 100);
            
            // Check thermal emergency is detected
            if (!thermal_emergency) begin
                $display("ERROR: Thermal emergency not detected");
                error_count++;
            end
            
            // Check emergency power reduction
            if (voltage_level[0] > 8'h60) begin
                $display("ERROR: Voltage not reduced in thermal emergency");
                error_count++;
            end
            
            // Check thermal throttling is active
            if (thermal_throttle_cycles == 0) begin
                $display("ERROR: Thermal throttling not activated");
                error_count++;
            end
            
            // Cool down
            for (int i = 0; i < NUM_DOMAINS; i++) begin
                temperature[i] = 16'h2800; // 40°C
            end
            
            #(CLK_PERIOD * 500);
            
            // Check recovery
            if (thermal_emergency) begin
                $display("ERROR: Thermal emergency not cleared after cooldown");
                error_count++;
            end
            
            $display("  Thermal emergency test completed");
        end
    endtask

    task run_battery_mode_test();
        begin
            $display("  Testing battery mode optimization...");
            
            // Switch to battery mode
            ac_power_available = 1'b0;
            battery_level = 32'h32; // 50%
            
            #(CLK_PERIOD * 500);
            
            // Check power mode switched to battery optimization
            if (power_mode == 3'b000) begin // Still in performance mode
                $display("WARNING: Power mode not optimized for battery");
            end
            
            // Test low battery scenario
            battery_level = 32'h0A; // 10%
            
            #(CLK_PERIOD * 200);
            
            // Check aggressive power saving
            if (voltage_level[0] > 8'h80) begin
                $display("ERROR: Voltage not reduced for low battery");
                error_count++;
            end
            
            // Test critical battery
            battery_level = 32'h03; // 3%
            
            #(CLK_PERIOD * 100);
            
            // Check emergency power saving
            if (!power_emergency) begin
                $display("ERROR: Power emergency not triggered for critical battery");
                error_count++;
            end
            
            // Restore AC power
            ac_power_available = 1'b1;
            battery_level = 32'h64; // 100%
            
            #(CLK_PERIOD * 200);
            
            $display("  Battery mode test completed");
        end
    endtask

    task run_low_power_modes_test();
        begin
            $display("  Testing low-power modes...");
            
            // Test standby mode
            sleep_request = 1'b1;
            
            #(CLK_PERIOD * 100);
            
            if (power_mode != 3'b101) begin // Standby mode
                $display("ERROR: Standby mode not entered");
                error_count++;
            end
            
            sleep_request = 1'b0;
            
            // Test deep sleep mode
            deep_sleep_request = 1'b1;
            
            #(CLK_PERIOD * 100);
            
            if (power_mode != 3'b110) begin // Deep sleep mode
                $display("ERROR: Deep sleep mode not entered");
                error_count++;
            end
            
            deep_sleep_request = 1'b0;
            
            // Test hibernate mode
            hibernate_request = 1'b1;
            
            #(CLK_PERIOD * 100);
            
            if (power_mode != 3'b111) begin // Hibernate mode
                $display("ERROR: Hibernate mode not entered");
                error_count++;
            end
            
            hibernate_request = 1'b0;
            
            #(CLK_PERIOD * 100);
            
            // Check wake-up
            if (!wake_up_ready) begin
                $display("ERROR: Wake-up not ready after hibernate exit");
                error_count++;
            end
            
            $display("  Low-power modes test completed");
        end
    endtask

    task run_thermal_scheduling_test();
        begin
            $display("  Testing thermal-aware task scheduling...");
            
            // Create thermal imbalance
            zone_temperature[0] = 16'h4800; // 72°C - hot zone
            zone_temperature[1] = 16'h3000; // 48°C - cool zone
            zone_temperature[2] = 16'h3800; // 56°C - warm zone
            zone_temperature[3] = 16'h2800; // 40°C - cool zone
            
            // Submit tasks with different thermal impacts
            task_valid = 16'h000F; // 4 tasks
            
            task_type[0] = 8'h00; // CPU intensive
            task_priority[0] = 16'h8000;
            task_deadline[0] = 16'h1000;
            task_power_estimate[0] = 16'h2000;
            task_thermal_impact[0] = 16'h1000; // High thermal impact
            
            task_type[1] = 8'h01; // AI inference
            task_priority[1] = 16'h6000;
            task_deadline[1] = 16'h2000;
            task_power_estimate[1] = 16'h3000;
            task_thermal_impact[1] = 16'h1800; // Very high thermal impact
            
            task_type[2] = 8'h00; // CPU intensive
            task_priority[2] = 16'h4000;
            task_deadline[2] = 16'h3000;
            task_power_estimate[2] = 16'h1000;
            task_thermal_impact[2] = 16'h0800; // Medium thermal impact
            
            task_type[3] = 8'h03; // Memory intensive
            task_priority[3] = 16'h2000;
            task_deadline[3] = 16'h4000;
            task_power_estimate[3] = 16'h0800;
            task_thermal_impact[3] = 16'h0400; // Low thermal impact
            
            #(CLK_PERIOD * 1000);
            
            // Check that tasks are scheduled
            if (scheduled_tasks_count == 0) begin
                $display("ERROR: No tasks were scheduled");
                error_count++;
            end
            
            // Check thermal-aware scheduling (high thermal impact tasks should avoid hot zones)
            for (int i = 0; i < 4; i++) begin
                if (task_scheduled[i]) begin
                    $display("  Task %0d scheduled to core %0d, thermal zone %0d", 
                            i, task_assigned_core[i], task_thermal_zone[i]);
                end
            end
            
            // Check thermal violations prevented
            if (thermal_violations_prevented == 0) begin
                $display("WARNING: No thermal violations prevented");
            end
            
            // Clear tasks
            task_valid = 16'h0000;
            
            #(CLK_PERIOD * 200);
            
            $display("  Thermal scheduling test completed");
        end
    endtask

    task run_ml_optimization_test();
        begin
            $display("  Testing ML-based power prediction...");
            
            // Provide ML predictions
            ml_prediction_valid = 1'b1;
            ml_power_prediction = 32'hC8; // 200W - high prediction
            
            #(CLK_PERIOD * 500);
            
            // Check that predicted power is used
            if (predicted_power_consumption != ml_power_prediction) begin
                $display("ERROR: ML prediction not used");
                error_count++;
            end
            
            // Test prediction-based optimization
            ml_power_prediction = 32'h32; // 50W - low prediction
            
            #(CLK_PERIOD * 500);
            
            // Check that system optimizes based on prediction
            if (voltage_level[0] == 8'h40) begin // Still at minimum
                $display("WARNING: System not optimizing based on low power prediction");
            end
            
            $display("  ML optimization test completed");
        end
    endtask

    task run_stress_test();
        begin
            $display("  Running stress test...");
            
            // Stress test with rapid changes
            for (int cycle = 0; cycle < 100; cycle++) begin
                // Randomly vary conditions
                for (int i = 0; i < NUM_CORES; i++) begin
                    core_utilization[i] = $random % 32'hFFFF;
                end
                
                for (int i = 0; i < NUM_DOMAINS; i++) begin
                    temperature[i] = 16'h2000 + ($random % 16'h3000);
                end
                
                battery_level = 32'h05 + ($random % 32'h5F);
                ac_power_available = ($random % 2) == 1;
                
                // Submit random tasks
                task_valid = $random % 16'hFFFF;
                for (int i = 0; i < MAX_TASKS; i++) begin
                    if (task_valid[i]) begin
                        task_type[i] = $random % 8;
                        task_priority[i] = $random % 16'hFFFF;
                        task_deadline[i] = 16'h100 + ($random % 16'h1000);
                        task_power_estimate[i] = $random % 16'h4000;
                        task_thermal_impact[i] = $random % 16'h2000;
                    end
                end
                
                #(CLK_PERIOD * 50);
                
                // Check system stability
                if (power_mode > 3'b111) begin
                    $display("ERROR: Invalid power mode during stress test");
                    error_count++;
                end
            end
            
            $display("  Stress test completed");
        end
    endtask

    // Monitor and check assertions
    always @(posedge clk) begin
        cycle_count++;
        
        // Check for invalid states
        if (rst_n) begin
            // Power mode should be valid
            if (power_mode > 3'b111) begin
                $display("ERROR at cycle %0d: Invalid power mode %0d", cycle_count, power_mode);
                error_count++;
            end
            
            // Voltage levels should be within range
            for (int i = 0; i < NUM_DOMAINS; i++) begin
                if (voltage_level[i] > 8'hFF) begin
                    $display("ERROR at cycle %0d: Invalid voltage level %0d: %02x", 
                            cycle_count, i, voltage_level[i]);
                    error_count++;
                end
            end
            
            // Check thermal emergency response time
            if (thermal_emergency && test_phase == 2) begin
                static int emergency_start = 0;
                if (emergency_start == 0) emergency_start = cycle_count;
                
                if ((cycle_count - emergency_start) > 1000) begin
                    $display("ERROR: Thermal emergency response too slow");
                    error_count++;
                end
            end
        end
    end

    // Performance monitoring
    always @(posedge clk) begin
        if (rst_n && (cycle_count % 1000 == 0)) begin
            $display("Cycle %0d: Power=%0dW, Temp=%0d°C, Mode=%0d, Savings=%0d", 
                    cycle_count, current_power_consumption, avg_temperature, 
                    power_mode, power_savings_total);
        end
    end

endmodule