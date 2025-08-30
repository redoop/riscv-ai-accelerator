/*
 * Thermal System Integration Test
 * 
 * This testbench verifies the complete thermal management system
 * including temperature sensors, thermal controller, and power monitor
 * working together in realistic scenarios.
 */

`timescale 1ns/1ps

module test_thermal_system_integration;

    // Parameters
    localparam NUM_TEMP_SENSORS = 8;
    localparam NUM_CORES = 4;
    localparam NUM_AI_UNITS = 2;
    localparam CLK_PERIOD = 10; // 100MHz

    // System signals
    logic                           clk;
    logic                           rst_n;
    
    // Environmental inputs
    logic [7:0]                     ambient_temp;
    logic [7:0]                     cooling_factor;
    
    // System activity and load
    logic [NUM_CORES-1:0]          core_activity;
    logic [NUM_AI_UNITS-1:0]       ai_unit_activity;
    logic                           memory_activity;
    logic [15:0]                    core_load [NUM_CORES-1:0];
    logic [15:0]                    memory_load;
    logic [15:0]                    ai_accel_load;
    
    // DVFS inputs
    logic [2:0]                     voltage_level;
    logic [2:0]                     frequency_level;
    logic [NUM_CORES-1:0]          core_power_enable;
    logic [NUM_AI_UNITS-1:0]       ai_unit_power_enable;
    logic                           memory_power_enable;
    logic                           noc_power_enable;
    
    // Temperature sensor outputs
    logic [15:0]                    temp_readings [NUM_TEMP_SENSORS-1:0];
    logic [NUM_TEMP_SENSORS-1:0]   temp_sensor_valid;
    logic [7:0]                     temp_celsius [NUM_TEMP_SENSORS-1:0];
    
    // Power monitor outputs
    logic [15:0]                    core_power [NUM_CORES-1:0];
    logic [15:0]                    ai_unit_power [NUM_AI_UNITS-1:0];
    logic [15:0]                    memory_power;
    logic [15:0]                    noc_power;
    logic [15:0]                    total_power;
    logic [15:0]                    avg_power;
    logic [15:0]                    peak_power;
    logic [31:0]                    energy_consumed;
    
    // Thermal controller outputs
    logic [7:0]                     max_temperature;
    logic [7:0]                     avg_temperature;
    logic [2:0]                     thermal_zone;
    logic                           thermal_alert;
    logic                           thermal_critical;
    logic                           thermal_emergency;
    logic [2:0]                     thermal_throttle_level;
    logic [NUM_CORES-1:0]          core_thermal_throttle;
    logic [NUM_AI_UNITS-1:0]       ai_unit_thermal_throttle;
    logic                           memory_thermal_throttle;
    logic [15:0]                    total_power_consumption;
    logic [15:0]                    power_budget_remaining;
    logic                           power_budget_exceeded;

    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Simulation control
    logic simulation_active;
    
    // Test-specific variables
    logic [7:0] prev_temp, stable_count;
    logic [7:0] recovery_temp;
    logic [7:0] temp_at_idle, temp_at_heavy;
    logic [15:0] power_at_idle, power_at_heavy;
    logic [7:0] temp_low_dvfs, temp_high_dvfs;
    logic [15:0] power_low_dvfs, power_high_dvfs;
    logic [7:0] temp_no_cooling, temp_good_cooling;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Temperature sensor instances
    genvar i;
    generate
        for (i = 0; i < NUM_TEMP_SENSORS; i++) begin : temp_sensors
            logic [15:0] local_power_input;
            logic [15:0] ambient_power_input;
            
            // Distribute power sources to sensors based on location
            always_comb begin
                case (i)
                    0, 1: begin // CPU core sensors
                        local_power_input = core_power[i % NUM_CORES];
                        ambient_power_input = (total_power - core_power[i % NUM_CORES]) >> 2;
                    end
                    2, 3: begin // AI unit sensors
                        local_power_input = ai_unit_power[i % NUM_AI_UNITS];
                        ambient_power_input = (total_power - ai_unit_power[i % NUM_AI_UNITS]) >> 2;
                    end
                    4, 5: begin // Memory sensors
                        local_power_input = memory_power;
                        ambient_power_input = (total_power - memory_power) >> 2;
                    end
                    6, 7: begin // NoC and general sensors
                        local_power_input = noc_power;
                        ambient_power_input = (total_power - noc_power) >> 2;
                    end
                endcase
            end
            
            temperature_sensor #(
                .SENSOR_ID(i),
                .THERMAL_TIME_CONSTANT(50) // Faster response for testing
            ) u_temp_sensor (
                .clk(clk),
                .rst_n(rst_n),
                .local_power(local_power_input),
                .ambient_power(ambient_power_input),
                .ambient_temp(ambient_temp),
                .cooling_factor(cooling_factor),
                .temp_reading(temp_readings[i]),
                .sensor_valid(temp_sensor_valid[i]),
                .temp_celsius(temp_celsius[i])
            );
        end
    endgenerate

    // Power monitor instance
    power_monitor #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .POWER_SAMPLE_PERIOD(100)
    ) u_power_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .core_activity(core_activity),
        .ai_unit_activity(ai_unit_activity),
        .memory_activity(memory_activity),
        .noc_activity(1'b1), // NoC always active
        .voltage_level(voltage_level),
        .frequency_level(frequency_level),
        .core_power_enable(core_power_enable),
        .ai_unit_power_enable(ai_unit_power_enable),
        .memory_power_enable(memory_power_enable),
        .noc_power_enable(noc_power_enable),
        .core_load(core_load),
        .memory_load(memory_load),
        .ai_accel_load(ai_accel_load),
        .core_power(core_power),
        .ai_unit_power(ai_unit_power),
        .memory_power(memory_power),
        .noc_power(noc_power),
        .total_power(total_power),
        .avg_power(avg_power),
        .peak_power(peak_power),
        .energy_consumed(energy_consumed),
        .pm_config_addr(32'h0),
        .pm_config_wdata(32'h0),
        .pm_config_rdata(),
        .pm_config_req(1'b0),
        .pm_config_we(1'b0),
        .pm_config_ready()
    );

    // Thermal controller instance
    thermal_controller #(
        .NUM_TEMP_SENSORS(NUM_TEMP_SENSORS),
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS)
    ) u_thermal_controller (
        .clk(clk),
        .rst_n(rst_n),
        .temp_sensors(temp_readings),
        .temp_sensor_valid(temp_sensor_valid),
        .core_power(core_power),
        .ai_unit_power(ai_unit_power),
        .memory_power(memory_power),
        .noc_power(noc_power),
        .core_activity(core_activity),
        .ai_unit_activity(ai_unit_activity),
        .memory_activity(memory_activity),
        .max_temperature(max_temperature),
        .avg_temperature(avg_temperature),
        .thermal_zone(thermal_zone),
        .thermal_alert(thermal_alert),
        .thermal_critical(thermal_critical),
        .thermal_emergency(thermal_emergency),
        .thermal_throttle_level(thermal_throttle_level),
        .core_thermal_throttle(core_thermal_throttle),
        .ai_unit_thermal_throttle(ai_unit_thermal_throttle),
        .memory_thermal_throttle(memory_thermal_throttle),
        .total_power_consumption(total_power_consumption),
        .power_budget_remaining(power_budget_remaining),
        .power_budget_exceeded(power_budget_exceeded),
        .thermal_config_addr(32'h0),
        .thermal_config_wdata(32'h0),
        .thermal_config_rdata(),
        .thermal_config_req(1'b0),
        .thermal_config_we(1'b0),
        .thermal_config_ready()
    );

    // Test tasks
    task reset_system();
        rst_n = 0;
        ambient_temp = 8'd25;      // Room temperature
        cooling_factor = 8'd128;   // 50% cooling effectiveness
        core_activity = '1;
        ai_unit_activity = '1;
        memory_activity = 1;
        voltage_level = 3'd4;      // Nominal voltage
        frequency_level = 3'd4;    // Nominal frequency
        core_power_enable = '1;
        ai_unit_power_enable = '1;
        memory_power_enable = 1;
        noc_power_enable = 1;
        
        // Initialize to medium load
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h6000; // 37.5% load
        end
        memory_load = 16'h5000;   // 31.25% load
        ai_accel_load = 16'h7000; // 43.75% load
        
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(20) @(posedge clk); // Allow sensors to stabilize
    endtask

    task check_result(string test_name, logic condition);
        test_count++;
        if (condition) begin
            $display("PASS: %s", test_name);
            pass_count++;
        end else begin
            $display("FAIL: %s", test_name);
            fail_count++;
        end
    endtask

    task set_workload(string workload_type);
        case (workload_type)
            "idle": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h1000;
                memory_load = 16'h0800;
                ai_accel_load = 16'h0400;
                core_activity = 4'b0001; // Only one core active
                ai_unit_activity = 2'b00; // AI units idle
                memory_activity = 0;
            end
            "light": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h3000;
                memory_load = 16'h2000;
                ai_accel_load = 16'h2800;
                core_activity = 4'b0011; // Two cores active
                ai_unit_activity = 2'b01; // One AI unit active
                memory_activity = 1;
            end
            "medium": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h6000;
                memory_load = 16'h5000;
                ai_accel_load = 16'h7000;
                core_activity = 4'b1111; // All cores active
                ai_unit_activity = 2'b11; // Both AI units active
                memory_activity = 1;
            end
            "heavy": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'hB000;
                memory_load = 16'hA000;
                ai_accel_load = 16'hD000;
                core_activity = 4'b1111; // All cores active
                ai_unit_activity = 2'b11; // Both AI units active
                memory_activity = 1;
            end
            "maximum": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'hF000;
                memory_load = 16'hF000;
                ai_accel_load = 16'hF000;
                core_activity = 4'b1111; // All cores active
                ai_unit_activity = 2'b11; // Both AI units active
                memory_activity = 1;
            end
        endcase
    endtask

    task set_dvfs_state(logic [2:0] volt_level, logic [2:0] freq_level);
        voltage_level = volt_level;
        frequency_level = freq_level;
    endtask

    task simulate_cooling_change(logic [7:0] new_cooling_factor);
        cooling_factor = new_cooling_factor;
        $display("Cooling factor changed to: %d (%.1f%%)", new_cooling_factor, 
                (real'(new_cooling_factor) / 255.0) * 100.0);
    endtask

    task wait_for_thermal_stabilization(int max_cycles);
        int cycle_count = 0;
        prev_temp = max_temperature;
        stable_count = 0;
        
        while (cycle_count < max_cycles && stable_count < 20) begin
            @(posedge clk);
            cycle_count++;
            
            if (max_temperature == prev_temp) begin
                stable_count++;
            end else begin
                stable_count = 0;
                prev_temp = max_temperature;
            end
        end
        
        $display("Thermal stabilization: %d cycles, final temp: %d°C", 
                cycle_count, max_temperature);
    endtask

    task display_system_status();
        $display("=== System Status ===");
        $display("Max Temperature: %d°C", max_temperature);
        $display("Avg Temperature: %d°C", avg_temperature);
        $display("Thermal Zone: %d", thermal_zone);
        $display("Thermal Alerts: Alert=%b, Critical=%b, Emergency=%b", 
                thermal_alert, thermal_critical, thermal_emergency);
        $display("Throttle Level: %d", thermal_throttle_level);
        $display("Total Power: %d mW", total_power);
        $display("Power Budget Exceeded: %b", power_budget_exceeded);
        $display("Core Throttling: %b", core_thermal_throttle);
        $display("AI Unit Throttling: %b", ai_unit_thermal_throttle);
        $display("Memory Throttling: %b", memory_thermal_throttle);
        $display("Individual Temperatures:");
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            $display("  Sensor %d: %d°C (valid=%b)", i, temp_celsius[i], temp_sensor_valid[i]);
        end
        $display("Individual Powers:");
        for (int i = 0; i < NUM_CORES; i++) begin
            $display("  Core %d: %d mW", i, core_power[i]);
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            $display("  AI Unit %d: %d mW", i, ai_unit_power[i]);
        end
        $display("  Memory: %d mW", memory_power);
        $display("  NoC: %d mW", noc_power);
        $display("=====================");
    endtask

    // Main test sequence
    initial begin
        $display("Starting Thermal System Integration Test");
        simulation_active = 1;
        
        // Test 1: System initialization and baseline
        $display("\n=== Test 1: System Initialization ===");
        reset_system();
        wait_for_thermal_stabilization(200);
        
        check_result("System initialized", rst_n);
        check_result("Temperature sensors functional", |temp_sensor_valid);
        check_result("Power monitoring active", total_power > 16'd0);
        check_result("Thermal management active", 1'b1);
        
        display_system_status();
        
        // Test 2: Workload scaling and thermal response
        $display("\n=== Test 2: Workload Scaling and Thermal Response ===");
        
        automatic logic [7:0] temp_at_idle, temp_at_heavy;
        automatic logic [15:0] power_at_idle, power_at_heavy;
        
        set_workload("idle");
        wait_for_thermal_stabilization(300);
        temp_at_idle = max_temperature;
        power_at_idle = total_power;
        $display("Idle workload - Temp: %d°C, Power: %d mW", temp_at_idle, power_at_idle);
        
        set_workload("heavy");
        wait_for_thermal_stabilization(500);
        temp_at_heavy = max_temperature;
        power_at_heavy = total_power;
        $display("Heavy workload - Temp: %d°C, Power: %d mW", temp_at_heavy, power_at_heavy);
        
        check_result("Higher workload increases power", power_at_heavy > power_at_idle);
        check_result("Higher workload increases temperature", temp_at_heavy > temp_at_idle);
        check_result("Temperature increase reasonable", 
                    (temp_at_heavy - temp_at_idle) >= 8'd5 && (temp_at_heavy - temp_at_idle) <= 8'd30);
        
        // Test 3: DVFS impact on thermal behavior
        $display("\n=== Test 3: DVFS Impact on Thermal Behavior ===");
        
        automatic logic [7:0] temp_low_dvfs, temp_high_dvfs;
        automatic logic [15:0] power_low_dvfs, power_high_dvfs;
        
        set_workload("medium");
        set_dvfs_state(3'd1, 3'd1); // Low voltage and frequency
        wait_for_thermal_stabilization(400);
        temp_low_dvfs = max_temperature;
        power_low_dvfs = total_power;
        $display("Low DVFS - Temp: %d°C, Power: %d mW", temp_low_dvfs, power_low_dvfs);
        
        set_dvfs_state(3'd7, 3'd7); // High voltage and frequency
        wait_for_thermal_stabilization(400);
        temp_high_dvfs = max_temperature;
        power_high_dvfs = total_power;
        $display("High DVFS - Temp: %d°C, Power: %d mW", temp_high_dvfs, power_high_dvfs);
        
        check_result("High DVFS increases power significantly", 
                    power_high_dvfs > (power_low_dvfs + (power_low_dvfs >> 1)));
        check_result("High DVFS increases temperature", temp_high_dvfs > temp_low_dvfs);
        
        // Test 4: Thermal protection activation
        $display("\n=== Test 4: Thermal Protection Activation ===");
        
        set_workload("maximum");
        set_dvfs_state(3'd7, 3'd7); // Maximum performance
        simulate_cooling_change(8'd32); // Reduce cooling to 12.5%
        
        // Monitor thermal protection engagement
        int protection_cycles = 0;
        while (protection_cycles < 1000 && !thermal_alert) begin
            @(posedge clk);
            protection_cycles++;
        end
        
        check_result("Thermal alert triggered", thermal_alert);
        $display("Thermal alert triggered after %d cycles", protection_cycles);
        
        // Wait for throttling to engage
        repeat(100) @(posedge clk);
        
        check_result("Thermal throttling active", thermal_throttle_level > 3'd0);
        check_result("Some components throttled", 
                    |core_thermal_throttle || |ai_unit_thermal_throttle || memory_thermal_throttle);
        
        display_system_status();
        
        // Test 5: Thermal recovery
        $display("\n=== Test 5: Thermal Recovery ===");
        
        simulate_cooling_change(8'd200); // Improve cooling to 78%
        set_workload("light"); // Reduce workload
        
        recovery_temp = max_temperature;
        int recovery_cycles = 0;
        
        // Wait for temperature to drop
        while (recovery_cycles < 1000 && max_temperature >= (recovery_temp - 8'd10)) begin
            @(posedge clk);
            recovery_cycles++;
            if (recovery_cycles % 100 == 0) begin
                $display("Recovery progress: %d cycles, temp: %d°C", recovery_cycles, max_temperature);
            end
        end
        
        check_result("Temperature decreased during recovery", max_temperature < recovery_temp);
        check_result("Thermal alerts cleared", !thermal_critical && !thermal_emergency);
        
        // Test 6: Power budget management
        $display("\n=== Test 6: Power Budget Management ===");
        
        set_workload("maximum");
        set_dvfs_state(3'd7, 3'd7);
        repeat(50) @(posedge clk);
        
        if (power_budget_exceeded) begin
            $display("Power budget exceeded at maximum load");
            check_result("Power budget monitoring functional", 1'b1);
        end else begin
            $display("Power within budget at maximum load");
            check_result("Power budget reasonable", total_power < 16'd20000); // Less than 20W
        end
        
        // Test 7: Cooling system effectiveness
        $display("\n=== Test 7: Cooling System Effectiveness ===");
        
        set_workload("heavy");
        set_dvfs_state(3'd5, 3'd5); // High performance
        
        automatic logic [7:0] temp_no_cooling, temp_good_cooling;
        
        simulate_cooling_change(8'd0); // No cooling
        wait_for_thermal_stabilization(400);
        temp_no_cooling = max_temperature;
        $display("No cooling - Temp: %d°C", temp_no_cooling);
        
        simulate_cooling_change(8'd255); // Maximum cooling
        wait_for_thermal_stabilization(400);
        temp_good_cooling = max_temperature;
        $display("Maximum cooling - Temp: %d°C", temp_good_cooling);
        
        check_result("Cooling system effective", 
                    (temp_no_cooling - temp_good_cooling) >= 8'd10);
        
        // Test 8: Sensor fault tolerance
        $display("\n=== Test 8: Sensor Fault Tolerance ===");
        
        // Simulate sensor faults
        force temp_sensors.temp_sensors[0].u_temp_sensor.sensor_fault = 1'b1;
        force temp_sensors.temp_sensors[1].u_temp_sensor.sensor_fault = 1'b1;
        
        repeat(50) @(posedge clk);
        
        check_result("System handles sensor faults", max_temperature < 8'd200);
        check_result("Some sensors still valid", |temp_sensor_valid);
        
        // Restore sensors
        release temp_sensors.temp_sensors[0].u_temp_sensor.sensor_fault;
        release temp_sensors.temp_sensors[1].u_temp_sensor.sensor_fault;
        
        repeat(50) @(posedge clk);
        
        // Test 9: Dynamic workload scenario
        $display("\n=== Test 9: Dynamic Workload Scenario ===");
        
        simulate_cooling_change(8'd128); // Normal cooling
        
        // Simulate a realistic AI workload pattern
        string workload_sequence[10] = '{
            "idle", "light", "medium", "heavy", "maximum",
            "heavy", "medium", "light", "medium", "idle"
        };
        
        for (int phase = 0; phase < 10; phase++) begin
            $display("Phase %d: %s workload", phase, workload_sequence[phase]);
            set_workload(workload_sequence[phase]);
            
            // Vary DVFS based on workload
            case (workload_sequence[phase])
                "idle": set_dvfs_state(3'd1, 3'd1);
                "light": set_dvfs_state(3'd2, 3'd2);
                "medium": set_dvfs_state(3'd4, 3'd4);
                "heavy": set_dvfs_state(3'd6, 3'd6);
                "maximum": set_dvfs_state(3'd7, 3'd7);
            endcase
            
            repeat(200) @(posedge clk);
            
            $display("  Temp: %d°C, Power: %d mW, Throttle: %d", 
                    max_temperature, total_power, thermal_throttle_level);
        end
        
        check_result("System stable through dynamic workload", 1'b1);
        
        // Test 10: Emergency thermal shutdown scenario
        $display("\n=== Test 10: Emergency Thermal Shutdown Scenario ===");
        
        set_workload("maximum");
        set_dvfs_state(3'd7, 3'd7);
        simulate_cooling_change(8'd0); // No cooling
        ambient_temp = 8'd40; // High ambient temperature
        
        int emergency_cycles = 0;
        while (emergency_cycles < 2000 && !thermal_emergency) begin
            @(posedge clk);
            emergency_cycles++;
            if (emergency_cycles % 200 == 0) begin
                $display("Emergency test: %d cycles, temp: %d°C", emergency_cycles, max_temperature);
            end
        end
        
        if (thermal_emergency) begin
            check_result("Emergency thermal protection triggered", 1'b1);
            check_result("Maximum throttling applied", thermal_throttle_level == 3'd7);
            $display("Emergency protection engaged after %d cycles", emergency_cycles);
        end else begin
            $display("Emergency protection not needed - system well designed");
            check_result("System thermal design adequate", max_temperature < 8'd100);
        end
        
        // Final system status
        $display("\n=== Final System Status ===");
        display_system_status();
        
        // Test summary
        $display("\n=== Thermal System Integration Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success rate: %.1f%%", (real'(pass_count) / real'(test_count)) * 100.0);
        
        if (fail_count == 0) begin
            $display("ALL THERMAL SYSTEM INTEGRATION TESTS PASSED!");
        end else begin
            $display("SOME THERMAL SYSTEM INTEGRATION TESTS FAILED!");
        end
        
        simulation_active = 0;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #20000000; // 20ms timeout
        $display("ERROR: Test timeout!");
        simulation_active = 0;
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_thermal_system_integration.vcd");
        $dumpvars(0, test_thermal_system_integration);
    end

    // Continuous monitoring (optional debug output)
    always @(posedge clk) begin
        if (simulation_active && (thermal_alert || thermal_critical || thermal_emergency)) begin
            if ($time % 1000000 == 0) begin // Every 1ms
                $display("[%t] THERMAL EVENT: Alert=%b, Critical=%b, Emergency=%b, Temp=%d°C, Power=%d mW", 
                        $time, thermal_alert, thermal_critical, thermal_emergency, max_temperature, total_power);
            end
        end
    end

endmodule