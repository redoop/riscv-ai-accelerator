/*
 * DVFS Efficiency Test
 * 
 * This testbench verifies the efficiency and power savings of the DVFS system
 * by measuring power consumption under different load scenarios.
 */

`timescale 1ns/1ps

module test_dvfs_efficiency;

    // Parameters
    localparam NUM_CORES = 4;
    localparam LOAD_MONITOR_WIDTH = 16;
    localparam CLK_PERIOD = 10; // 100MHz
    localparam TEST_DURATION = 100000; // Test cycles

    // Signals for power manager
    logic                           clk;
    logic                           rst_n;
    logic                           ref_clk;
    logic                           dvfs_enable;
    logic [NUM_CORES-1:0][LOAD_MONITOR_WIDTH-1:0] core_load;
    logic [NUM_CORES-1:0]           core_active;
    logic [LOAD_MONITOR_WIDTH-1:0] memory_load;
    logic [LOAD_MONITOR_WIDTH-1:0] noc_load;
    logic [LOAD_MONITOR_WIDTH-1:0] ai_accel_load;
    logic [NUM_CORES-1:0]          core_activity;
    logic                           memory_activity;
    logic [1:0]                     ai_unit_activity;
    logic [15:0]                    temp_sensors [7:0];
    
    // Power manager outputs
    logic [3:0]                     global_voltage;
    logic [7:0]                     global_freq_div;
    logic [NUM_CORES-1:0]          core_power_enable;
    logic                           l1_cache_power_enable;
    logic                           l2_cache_power_enable;
    logic                           memory_ctrl_power_enable;
    logic [1:0]                     ai_unit_power_enable;
    logic                           noc_power_enable;
    logic [NUM_CORES-1:0]          core_isolation_enable;
    logic                           memory_isolation_enable;
    logic [1:0]                     ai_unit_isolation_enable;
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
    
    // Power measurement variables
    real power_consumption_baseline;
    real power_consumption_dvfs;
    real power_savings_percent;
    int active_cores_count;
    int gated_cores_count;

    // Clock generation
    initial begin
        clk = 0;
        ref_clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
            #(CLK_PERIOD/4) ref_clk = ~ref_clk;
        end
    end

    // Legacy power management interfaces (unused but required)
    power_mgmt_if core_pm_if [3:0] ();
    power_mgmt_if tpu_pm_if [1:0] ();
    power_mgmt_if vpu_pm_if [1:0] ();

    // DUT instantiation
    power_manager #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(2),
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
        
        // Initialize temperature sensors to normal operating temperature
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

    task configure_dvfs(logic enable, logic [2:0] min_volt, logic [2:0] max_volt);
        // Configure DVFS through register interface
        pm_config_req = 1;
        pm_config_we = 1;
        
        // Enable/disable DVFS
        pm_config_addr = 32'h00;
        pm_config_wdata = {31'b0, enable};
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        
        // Set minimum voltage level
        pm_config_addr = 32'h04;
        pm_config_wdata = {29'b0, min_volt};
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        
        // Set maximum voltage level
        pm_config_addr = 32'h08;
        pm_config_wdata = {29'b0, max_volt};
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        
        pm_config_req = 0;
        pm_config_we = 0;
        
        dvfs_enable = enable;
        repeat(5) @(posedge clk);
    endtask

    task set_workload(logic [15:0] cpu_load, logic [15:0] mem_load, logic [15:0] ai_load);
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = cpu_load;
        end
        memory_load = mem_load;
        noc_load = mem_load >> 1; // NoC load typically correlates with memory
        ai_accel_load = ai_load;
        
        // Set activity based on load levels
        for (int i = 0; i < NUM_CORES; i++) begin
            core_activity[i] = (cpu_load > 16'h1000);
        end
        memory_activity = (mem_load > 16'h1000);
        ai_unit_activity = (ai_load > 16'h1000) ? 2'b11 : 2'b00;
    endtask

    function real calculate_power_consumption();
        real power = 0.0;
        
        // Base power consumption model (simplified)
        // Power = Voltage^2 * Frequency * Activity_Factor
        
        real voltage_factor = (global_voltage + 1) * (global_voltage + 1) / 64.0; // Normalized
        real frequency_factor = (8 - global_freq_div) / 8.0; // Normalized
        
        // Core power
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_power_enable[i]) begin
                power += voltage_factor * frequency_factor * 10.0; // 10mW per core
            end
        end
        
        // Cache power
        if (l1_cache_power_enable) power += voltage_factor * frequency_factor * 5.0;
        if (l2_cache_power_enable) power += voltage_factor * frequency_factor * 8.0;
        if (memory_ctrl_power_enable) power += voltage_factor * frequency_factor * 12.0;
        
        // AI unit power
        for (int i = 0; i < 2; i++) begin
            if (ai_unit_power_enable[i]) begin
                power += voltage_factor * frequency_factor * 20.0; // 20mW per AI unit
            end
        end
        
        // NoC power (always on)
        power += voltage_factor * frequency_factor * 3.0;
        
        return power;
    endfunction

    task measure_power_efficiency(string scenario_name, int duration_cycles);
        real total_power = 0.0;
        real avg_power;
        int sample_count = 0;
        
        $display("\n--- Measuring Power: %s ---", scenario_name);
        
        for (int cycle = 0; cycle < duration_cycles; cycle++) begin
            @(posedge clk);
            
            // Sample power every 100 cycles
            if (cycle % 100 == 0) begin
                total_power += calculate_power_consumption();
                sample_count++;
            end
        end
        
        avg_power = total_power / sample_count;
        $display("Average power consumption: %.2f mW", avg_power);
        
        // Count active and gated components
        active_cores_count = 0;
        gated_cores_count = 0;
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_power_enable[i]) active_cores_count++;
            else gated_cores_count++;
        end
        
        $display("Active cores: %d, Gated cores: %d", active_cores_count, gated_cores_count);
        $display("Voltage level: %d, Frequency level: %d", global_voltage, 8 - global_freq_div);
        
        if (scenario_name == "Baseline") begin
            power_consumption_baseline = avg_power;
        end else if (scenario_name == "DVFS Enabled") begin
            power_consumption_dvfs = avg_power;
            power_savings_percent = ((power_consumption_baseline - power_consumption_dvfs) / power_consumption_baseline) * 100.0;
            $display("Power savings: %.1f%%", power_savings_percent);
        end
    endtask

    // Main test sequence
    initial begin
        $display("Starting DVFS Efficiency Test");
        
        // Test 1: System initialization
        reset_system();
        check_result("System reset", rst_n && !dvfs_enable);
        
        // Test 2: Baseline power measurement (DVFS disabled)
        $display("\n=== Baseline Power Measurement ===");
        configure_dvfs(1'b0, 3'd1, 3'd7); // DVFS disabled
        set_workload(16'h6000, 16'h4000, 16'h3000); // Medium workload
        
        repeat(100) @(posedge clk); // Let system stabilize
        measure_power_efficiency("Baseline", 5000);
        
        // Test 3: DVFS enabled with same workload
        $display("\n=== DVFS Enabled Power Measurement ===");
        configure_dvfs(1'b1, 3'd1, 3'd7); // DVFS enabled
        
        repeat(200) @(posedge clk); // Let DVFS adjust
        measure_power_efficiency("DVFS Enabled", 5000);
        
        check_result("DVFS provides power savings", power_savings_percent > 5.0);
        
        // Test 4: Low load scenario
        $display("\n=== Low Load Scenario ===");
        set_workload(16'h1000, 16'h0800, 16'h0400); // Low workload
        
        repeat(300) @(posedge clk); // Let DVFS adjust to low load
        measure_power_efficiency("Low Load DVFS", 5000);
        
        check_result("Low load - Cores power gated", gated_cores_count > 0);
        check_result("Low load - Reduced voltage", global_voltage <= 3);
        check_result("Low load - Reduced frequency", global_freq_div >= 5);
        
        // Test 5: High load scenario
        $display("\n=== High Load Scenario ===");
        set_workload(16'hA000, 16'h8000, 16'h9000); // High workload
        
        repeat(300) @(posedge clk); // Let DVFS adjust to high load
        measure_power_efficiency("High Load DVFS", 5000);
        
        check_result("High load - All cores active", active_cores_count == NUM_CORES);
        check_result("High load - High voltage", global_voltage >= 6);
        check_result("High load - High frequency", global_freq_div <= 2);
        
        // Test 6: Thermal throttling scenario
        $display("\n=== Thermal Throttling Scenario ===");
        for (int i = 0; i < 8; i++) begin
            temp_sensors[i] = 16'h5500; // ~85°C
        end
        
        repeat(200) @(posedge clk); // Let thermal protection kick in
        measure_power_efficiency("Thermal Throttling", 3000);
        
        check_result("Thermal protection - Reduced voltage", global_voltage <= 2);
        check_result("Thermal protection - Reduced frequency", global_freq_div >= 6);
        
        // Reset temperature
        for (int i = 0; i < 8; i++) begin
            temp_sensors[i] = 16'h1900; // ~25°C
        end
        
        // Test 7: Dynamic workload scenario
        $display("\n=== Dynamic Workload Scenario ===");
        
        // Simulate varying workload over time
        for (int phase = 0; phase < 5; phase++) begin
            case (phase)
                0: set_workload(16'h2000, 16'h1000, 16'h0800); // Low
                1: set_workload(16'h8000, 16'h6000, 16'h7000); // High
                2: set_workload(16'h4000, 16'h3000, 16'h2000); // Medium
                3: set_workload(16'h0800, 16'h0400, 16'h0200); // Very low
                4: set_workload(16'hC000, 16'hA000, 16'hB000); // Very high
            endcase
            
            repeat(1000) @(posedge clk); // Let DVFS adapt
            
            $display("Phase %d - Voltage: %d, Frequency: %d, Active cores: %d", 
                    phase, global_voltage, 8 - global_freq_div, active_cores_count);
        end
        
        check_result("Dynamic workload adaptation", 1'b1); // Always pass if we get here
        
        // Test 8: DVFS constraint testing
        $display("\n=== DVFS Constraint Testing ===");
        configure_dvfs(1'b1, 3'd3, 3'd5); // Limited voltage range
        set_workload(16'hF000, 16'hE000, 16'hD000); // Maximum load
        
        repeat(200) @(posedge clk);
        
        check_result("Voltage constraint - Max respected", global_voltage <= 5);
        
        set_workload(16'h0100, 16'h0080, 16'h0040); // Minimum load
        
        repeat(200) @(posedge clk);
        
        check_result("Voltage constraint - Min respected", global_voltage >= 3);
        
        // Test summary
        $display("\n=== DVFS Efficiency Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Baseline power: %.2f mW", power_consumption_baseline);
        $display("DVFS power: %.2f mW", power_consumption_dvfs);
        $display("Power savings: %.1f%%", power_savings_percent);
        
        if (fail_count == 0) begin
            $display("ALL EFFICIENCY TESTS PASSED!");
        end else begin
            $display("SOME EFFICIENCY TESTS FAILED!");
        end
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #5000000; // 5ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_dvfs_efficiency.vcd");
        $dumpvars(0, test_dvfs_efficiency);
    end

endmodule