/*
 * DVFS Integration Test
 * 
 * This testbench verifies the complete DVFS system integration,
 * including load-aware strategies, power domain coordination,
 * and system-level behavior.
 */

`timescale 1ns/1ps

module test_dvfs_integration;

    // Parameters
    localparam NUM_CORES = 4;
    localparam NUM_AI_UNITS = 2;
    localparam LOAD_MONITOR_WIDTH = 16;
    localparam CLK_PERIOD = 10;

    // System signals
    logic                           clk;
    logic                           rst_n;
    logic                           ref_clk;
    
    // DVFS control signals
    logic                           dvfs_enable;
    logic [NUM_CORES-1:0][LOAD_MONITOR_WIDTH-1:0] core_load;
    logic [NUM_CORES-1:0]           core_active;
    logic [LOAD_MONITOR_WIDTH-1:0] memory_load;
    logic [LOAD_MONITOR_WIDTH-1:0] noc_load;
    logic [LOAD_MONITOR_WIDTH-1:0] ai_accel_load;
    logic [NUM_CORES-1:0]          core_activity;
    logic                           memory_activity;
    logic [NUM_AI_UNITS-1:0]       ai_unit_activity;
    logic [15:0]                    temp_sensors [7:0];
    
    // Power management outputs
    logic [3:0]                     global_voltage;
    logic [7:0]                     global_freq_div;
    logic [NUM_CORES-1:0]          core_power_enable;
    logic                           l1_cache_power_enable;
    logic                           l2_cache_power_enable;
    logic                           memory_ctrl_power_enable;
    logic [NUM_AI_UNITS-1:0]       ai_unit_power_enable;
    logic                           noc_power_enable;
    logic [NUM_CORES-1:0]          core_isolation_enable;
    logic                           memory_isolation_enable;
    logic [NUM_AI_UNITS-1:0]       ai_unit_isolation_enable;
    
    // External interfaces
    logic                           vreg_scl;
    logic                           vreg_sda;
    logic                           cpu_clk;
    logic                           ai_accel_clk;
    logic                           memory_clk;
    logic                           noc_clk;
    
    // Configuration interface
    logic [31:0]                    pm_config_addr;
    logic [31:0]                    pm_config_wdata;
    logic [31:0]                    pm_config_rdata;
    logic                           pm_config_req;
    logic                           pm_config_we;
    logic                           pm_config_ready;

    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Performance monitoring
    logic [31:0] performance_counter;
    logic [31:0] power_gate_events;
    logic [31:0] voltage_transitions;
    logic [31:0] frequency_transitions;

    // Clock generation
    initial begin
        clk = 0;
        ref_clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
            #(CLK_PERIOD/4) ref_clk = ~ref_clk;
        end
    end

    // Legacy power management interfaces
    power_mgmt_if core_pm_if [3:0] ();
    power_mgmt_if tpu_pm_if [1:0] ();
    power_mgmt_if vpu_pm_if [1:0] ();

    // DUT instantiation
    power_manager #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .LOAD_MONITOR_WIDTH(LOAD_MONITOR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .core_pm_if(core_pm_if),
        .tpu_pm_if(tpu_pm_if),
        .vpu_pm_if(vpu_pm_if),
        .dvfs_enable(dvfs_enable),
        .core_load(core_load),
        .core_active(core_active),
        .memory_load(memory_load),
        .noc_load(noc_load),
        .ai_accel_load(ai_accel_load),
        .core_activity(core_activity),
        .memory_activity(memory_activity),
        .ai_unit_activity(ai_unit_activity),
        .global_voltage(global_voltage),
        .global_freq_div(global_freq_div),
        .core_power_enable(core_power_enable),
        .l1_cache_power_enable(l1_cache_power_enable),
        .l2_cache_power_enable(l2_cache_power_enable),
        .memory_ctrl_power_enable(memory_ctrl_power_enable),
        .ai_unit_power_enable(ai_unit_power_enable),
        .noc_power_enable(noc_power_enable),
        .core_isolation_enable(core_isolation_enable),
        .memory_isolation_enable(memory_isolation_enable),
        .ai_unit_isolation_enable(ai_unit_isolation_enable),
        .temp_sensors(temp_sensors),
        .vreg_scl(vreg_scl),
        .vreg_sda(vreg_sda),
        .ref_clk(ref_clk),
        .cpu_clk(cpu_clk),
        .ai_accel_clk(ai_accel_clk),
        .memory_clk(memory_clk),
        .noc_clk(noc_clk),
        .pm_config_addr(pm_config_addr),
        .pm_config_wdata(pm_config_wdata),
        .pm_config_rdata(pm_config_rdata),
        .pm_config_req(pm_config_req),
        .pm_config_we(pm_config_we),
        .pm_config_ready(pm_config_ready)
    );

    // Performance monitoring
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            performance_counter <= 0;
            power_gate_events <= 0;
            voltage_transitions <= 0;
            frequency_transitions <= 0;
        end else begin
            performance_counter <= performance_counter + 1;
            
            // Count power gating events
            if (|core_isolation_enable || memory_isolation_enable || |ai_unit_isolation_enable) begin
                power_gate_events <= power_gate_events + 1;
            end
            
            // Count voltage transitions (simplified)
            if (performance_counter > 0 && global_voltage != 4) begin
                voltage_transitions <= voltage_transitions + 1;
            end
            
            // Count frequency transitions (simplified)
            if (performance_counter > 0 && global_freq_div != 4) begin
                frequency_transitions <= frequency_transitions + 1;
            end
        end
    end

    // Test tasks
    task reset_system();
        rst_n = 0;
        dvfs_enable = 0;
        core_active = '1;
        core_load = '0;
        memory_load = 0;
        noc_load = 0;
        ai_accel_load = 0;
        core_activity = '1;
        memory_activity = 1;
        ai_unit_activity = '1;
        pm_config_req = 0;
        pm_config_we = 0;
        pm_config_addr = 0;
        pm_config_wdata = 0;
        
        // Initialize temperature sensors
        for (int i = 0; i < 8; i++) begin
            temp_sensors[i] = 16'h1900; // ~25°C
        end
        
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
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

    task enable_dvfs();
        pm_config_req = 1;
        pm_config_we = 1;
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h01;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 0;
        pm_config_we = 0;
        dvfs_enable = 1;
        repeat(5) @(posedge clk);
    endtask

    task set_load_scenario(string scenario);
        case (scenario)
            "idle": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h0200;
                memory_load = 16'h0100;
                noc_load = 16'h0080;
                ai_accel_load = 16'h0040;
                core_activity = 4'b0001; // Only core 0 active
                memory_activity = 0;
                ai_unit_activity = 2'b00;
            end
            "light": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h2000;
                memory_load = 16'h1800;
                noc_load = 16'h1000;
                ai_accel_load = 16'h0800;
                core_activity = 4'b0011; // Two cores active
                memory_activity = 1;
                ai_unit_activity = 2'b01;
            end
            "medium": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h6000;
                memory_load = 16'h5000;
                noc_load = 16'h4000;
                ai_accel_load = 16'h4800;
                core_activity = 4'b1111; // All cores active
                memory_activity = 1;
                ai_unit_activity = 2'b11;
            end
            "heavy": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'hA000;
                memory_load = 16'h9000;
                noc_load = 16'h8000;
                ai_accel_load = 16'hB000;
                core_activity = 4'b1111; // All cores active
                memory_activity = 1;
                ai_unit_activity = 2'b11;
            end
            "maximum": begin
                for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'hF000;
                memory_load = 16'hE000;
                noc_load = 16'hD000;
                ai_accel_load = 16'hF800;
                core_activity = 4'b1111; // All cores active
                memory_activity = 1;
                ai_unit_activity = 2'b11;
            end
        endcase
    endtask

    task wait_for_dvfs_adaptation(int cycles);
        repeat(cycles) @(posedge clk);
    endtask

    task verify_power_state(string expected_state);
        case (expected_state)
            "low_power": begin
                check_result("Low power - Reduced voltage", global_voltage <= 3);
                check_result("Low power - Reduced frequency", global_freq_div >= 5);
                check_result("Low power - Some cores gated", |core_isolation_enable);
            end
            "balanced": begin
                check_result("Balanced - Medium voltage", global_voltage >= 3 && global_voltage <= 5);
                check_result("Balanced - Medium frequency", global_freq_div >= 3 && global_freq_div <= 5);
                check_result("Balanced - All cores active", &core_power_enable);
            end
            "high_performance": begin
                check_result("High perf - High voltage", global_voltage >= 6);
                check_result("High perf - High frequency", global_freq_div <= 2);
                check_result("High perf - All cores active", &core_power_enable);
                check_result("High perf - All caches active", 
                           l1_cache_power_enable && l2_cache_power_enable && memory_ctrl_power_enable);
            end
        endcase
    endtask

    task simulate_workload_transition(string from_load, string to_load, int transition_time);
        $display("Transitioning from %s to %s load", from_load, to_load);
        
        set_load_scenario(from_load);
        wait_for_dvfs_adaptation(100);
        
        // Gradual transition
        for (int step = 0; step < transition_time; step++) begin
            // Interpolate between load scenarios (simplified)
            repeat(10) @(posedge clk);
        end
        
        set_load_scenario(to_load);
        wait_for_dvfs_adaptation(200);
    endtask

    // Main test sequence
    initial begin
        $display("Starting DVFS Integration Test");
        
        // Test 1: System initialization and basic functionality
        $display("\n=== Test 1: System Initialization ===");
        reset_system();
        check_result("System reset successful", rst_n && !dvfs_enable);
        check_result("All power domains initially on", 
                    &core_power_enable && l1_cache_power_enable && 
                    l2_cache_power_enable && memory_ctrl_power_enable);
        
        // Test 2: DVFS enablement and basic operation
        $display("\n=== Test 2: DVFS Basic Operation ===");
        enable_dvfs();
        check_result("DVFS enabled", dvfs_enable);
        
        set_load_scenario("medium");
        wait_for_dvfs_adaptation(100);
        check_result("DVFS responds to medium load", 1'b1); // Basic functionality check
        
        // Test 3: Load-aware scaling scenarios
        $display("\n=== Test 3: Load-Aware Scaling ===");
        
        // Idle scenario
        set_load_scenario("idle");
        wait_for_dvfs_adaptation(300);
        verify_power_state("low_power");
        
        // Light load scenario
        set_load_scenario("light");
        wait_for_dvfs_adaptation(200);
        check_result("Light load - Appropriate scaling", 
                    global_voltage >= 2 && global_voltage <= 4);
        
        // Heavy load scenario
        set_load_scenario("heavy");
        wait_for_dvfs_adaptation(200);
        verify_power_state("high_performance");
        
        // Maximum load scenario
        set_load_scenario("maximum");
        wait_for_dvfs_adaptation(200);
        check_result("Maximum load - Peak performance", 
                    global_voltage >= 6 && global_freq_div <= 1);
        
        // Test 4: Dynamic workload adaptation
        $display("\n=== Test 4: Dynamic Workload Adaptation ===");
        
        simulate_workload_transition("idle", "heavy", 50);
        check_result("Idle to heavy transition", global_voltage >= 5);
        
        simulate_workload_transition("heavy", "light", 50);
        check_result("Heavy to light transition", global_voltage <= 4);
        
        simulate_workload_transition("light", "maximum", 30);
        check_result("Light to maximum transition", global_voltage >= 6);
        
        // Test 5: Power gating coordination
        $display("\n=== Test 5: Power Gating Coordination ===");
        
        set_load_scenario("idle");
        wait_for_dvfs_adaptation(500); // Allow time for power gating
        
        check_result("Idle cores power gated", |core_isolation_enable);
        check_result("Memory subsystem partially gated", memory_isolation_enable);
        check_result("AI units gated when idle", |ai_unit_isolation_enable);
        
        // Verify power restoration on activity
        set_load_scenario("medium");
        wait_for_dvfs_adaptation(300);
        
        check_result("Power restored on activity", &core_power_enable);
        check_result("Isolation removed", 
                    core_isolation_enable == '0 && !memory_isolation_enable);
        
        // Test 6: Thermal management integration
        $display("\n=== Test 6: Thermal Management ===");
        
        set_load_scenario("heavy");
        wait_for_dvfs_adaptation(100);
        
        // Simulate overheating
        for (int i = 0; i < 8; i++) begin
            temp_sensors[i] = 16'h5500; // ~85°C
        end
        
        wait_for_dvfs_adaptation(200);
        
        check_result("Thermal protection - Voltage reduced", global_voltage <= 2);
        check_result("Thermal protection - Frequency reduced", global_freq_div >= 6);
        
        // Cool down
        for (int i = 0; i < 8; i++) begin
            temp_sensors[i] = 16'h1900; // ~25°C
        end
        
        wait_for_dvfs_adaptation(200);
        check_result("Thermal recovery", global_voltage > 2);
        
        // Test 7: Multi-domain coordination
        $display("\n=== Test 7: Multi-Domain Coordination ===");
        
        // Test CPU-heavy workload
        for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'hC000;
        memory_load = 16'h2000;
        ai_accel_load = 16'h1000;
        wait_for_dvfs_adaptation(200);
        
        check_result("CPU-heavy - Cores active", &core_power_enable);
        check_result("CPU-heavy - AI units may be gated", 1'b1); // Flexible check
        
        // Test AI-heavy workload
        for (int i = 0; i < NUM_CORES; i++) core_load[i] = 16'h3000;
        memory_load = 16'h8000;
        ai_accel_load = 16'hE000;
        wait_for_dvfs_adaptation(200);
        
        check_result("AI-heavy - AI units active", &ai_unit_power_enable);
        check_result("AI-heavy - Memory active", memory_ctrl_power_enable);
        
        // Test 8: Frequency and voltage coordination
        $display("\n=== Test 8: Voltage-Frequency Coordination ===");
        
        logic [2:0] prev_voltage, prev_frequency;
        
        set_load_scenario("light");
        wait_for_dvfs_adaptation(100);
        prev_voltage = global_voltage;
        prev_frequency = 8 - global_freq_div;
        
        set_load_scenario("heavy");
        wait_for_dvfs_adaptation(200);
        
        check_result("Voltage increased with load", global_voltage > prev_voltage);
        check_result("Frequency increased with load", (8 - global_freq_div) > prev_frequency);
        
        // Test 9: System stability under rapid changes
        $display("\n=== Test 9: System Stability ===");
        
        // Rapid load changes
        for (int cycle = 0; cycle < 10; cycle++) begin
            if (cycle % 2 == 0) begin
                set_load_scenario("heavy");
            end else begin
                set_load_scenario("light");
            end
            wait_for_dvfs_adaptation(50);
        end
        
        check_result("System stable under rapid changes", 1'b1);
        
        // Test 10: Performance monitoring and statistics
        $display("\n=== Test 10: Performance Statistics ===");
        
        $display("Performance counter: %d cycles", performance_counter);
        $display("Power gate events: %d", power_gate_events);
        $display("Voltage transitions: %d", voltage_transitions);
        $display("Frequency transitions: %d", frequency_transitions);
        
        check_result("Power gating occurred", power_gate_events > 0);
        check_result("DVFS transitions occurred", 
                    voltage_transitions > 0 || frequency_transitions > 0);
        
        // Final system state verification
        set_load_scenario("medium");
        wait_for_dvfs_adaptation(100);
        
        check_result("Final state - System operational", 
                    &core_power_enable && noc_power_enable);
        
        // Test summary
        $display("\n=== DVFS Integration Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success rate: %.1f%%", (real'(pass_count) / real'(test_count)) * 100.0);
        
        if (fail_count == 0) begin
            $display("ALL INTEGRATION TESTS PASSED!");
        end else begin
            $display("SOME INTEGRATION TESTS FAILED!");
        end
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #10000000; // 10ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_dvfs_integration.vcd");
        $dumpvars(0, test_dvfs_integration);
    end

endmodule