/*
 * DVFS Controller Test
 * 
 * This testbench verifies the functionality of the DVFS controller,
 * including load-aware frequency scaling, power gating, and thermal protection.
 */

`timescale 1ns/1ps

module test_dvfs_controller;

    // Parameters
    localparam NUM_CORES = 4;
    localparam LOAD_MONITOR_WIDTH = 16;
    localparam CLK_PERIOD = 10; // 100MHz

    // Signals
    logic                           clk;
    logic                           rst_n;
    logic [NUM_CORES-1:0][LOAD_MONITOR_WIDTH-1:0] core_load;
    logic [NUM_CORES-1:0]           core_active;
    logic [LOAD_MONITOR_WIDTH-1:0] memory_load;
    logic [LOAD_MONITOR_WIDTH-1:0] noc_load;
    logic [LOAD_MONITOR_WIDTH-1:0] ai_accel_load;
    logic [7:0]                     temperature;
    logic                           thermal_alert;
    logic [2:0]                     voltage_level;
    logic [2:0]                     frequency_level;
    logic [NUM_CORES-1:0]           core_power_gate;
    logic                           memory_power_gate;
    logic                           ai_accel_power_gate;
    logic                           dvfs_enable;
    logic [2:0]                     min_voltage_level;
    logic [2:0]                     max_voltage_level;
    logic                           dvfs_transition_busy;
    logic [7:0]                     power_state;

    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT instantiation
    dvfs_controller #(
        .NUM_CORES(NUM_CORES),
        .LOAD_MONITOR_WIDTH(LOAD_MONITOR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .core_load(core_load),
        .core_active(core_active),
        .memory_load(memory_load),
        .noc_load(noc_load),
        .ai_accel_load(ai_accel_load),
        .temperature(temperature),
        .thermal_alert(thermal_alert),
        .voltage_level(voltage_level),
        .frequency_level(frequency_level),
        .core_power_gate(core_power_gate),
        .memory_power_gate(memory_power_gate),
        .ai_accel_power_gate(ai_accel_power_gate),
        .dvfs_enable(dvfs_enable),
        .min_voltage_level(min_voltage_level),
        .max_voltage_level(max_voltage_level),
        .dvfs_transition_busy(dvfs_transition_busy),
        .power_state(power_state)
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
        temperature = 8'd25;  // Room temperature
        thermal_alert = 0;
        min_voltage_level = 3'd1;
        max_voltage_level = 3'd7;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
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

    task wait_for_transition();
        while (dvfs_transition_busy) begin
            @(posedge clk);
        end
        repeat(5) @(posedge clk);  // Additional settling time
    endtask

    task set_system_load(logic [15:0] load_level);
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = load_level;
        end
        memory_load = load_level;
        noc_load = load_level;
        ai_accel_load = load_level;
    endtask

    // Main test sequence
    initial begin
        $display("Starting DVFS Controller Test");
        
        // Test 1: Reset and initialization
        reset_system();
        check_result("Reset - DVFS disabled", !dvfs_transition_busy);
        check_result("Reset - Default voltage level", voltage_level == 3'd4);
        check_result("Reset - Default frequency level", frequency_level == 3'd4);
        
        // Enable DVFS
        dvfs_enable = 1;
        repeat(10) @(posedge clk);
        
        // Test 2: Low load - should trigger power saving mode
        $display("\n--- Test 2: Low Load Power Saving ---");
        set_system_load(16'h2000);  // 12.5% load
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("Low load - Reduced voltage", voltage_level <= 3'd3);
        check_result("Low load - Reduced frequency", frequency_level <= 3'd3);
        
        // Test 3: High load - should trigger performance mode
        $display("\n--- Test 3: High Load Performance Mode ---");
        set_system_load(16'hA000);  // 62.5% load
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("High load - Increased voltage", voltage_level >= 3'd6);
        check_result("High load - Increased frequency", frequency_level >= 3'd6);
        
        // Test 4: Medium load - balanced mode
        $display("\n--- Test 4: Medium Load Balanced Mode ---");
        set_system_load(16'h6000);  // 37.5% load
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("Medium load - Balanced voltage", voltage_level == 3'd4);
        check_result("Medium load - Balanced frequency", frequency_level == 3'd4);
        
        // Test 5: Thermal protection
        $display("\n--- Test 5: Thermal Protection ---");
        temperature = 8'd90;  // High temperature
        thermal_alert = 1;
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("Thermal protection - Reduced voltage", voltage_level <= min_voltage_level);
        check_result("Thermal protection - Reduced frequency", frequency_level <= 3'd2);
        
        // Reset thermal condition
        temperature = 8'd25;
        thermal_alert = 0;
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        // Test 6: Power gating with idle cores
        $display("\n--- Test 6: Power Gating ---");
        set_system_load(16'h0800);  // Very low load (3.125%)
        core_active[2] = 0;  // Deactivate core 2
        core_active[3] = 0;  // Deactivate core 3
        repeat(30) @(posedge clk);
        wait_for_transition();
        
        check_result("Power gating - Inactive cores gated", 
                    core_power_gate[2] && core_power_gate[3]);
        check_result("Power gating - Active cores not gated", 
                    !core_power_gate[0] && !core_power_gate[1]);
        
        // Test 7: AI accelerator power gating
        ai_accel_load = 16'h0400;  // Very low AI load
        repeat(30) @(posedge clk);
        wait_for_transition();
        
        check_result("AI accelerator power gating", ai_accel_power_gate);
        
        // Test 8: Memory power gating
        memory_load = 16'h0200;  // Very low memory load
        repeat(30) @(posedge clk);
        wait_for_transition();
        
        check_result("Memory power gating", memory_power_gate);
        
        // Test 9: DVFS disable
        $display("\n--- Test 9: DVFS Disable ---");
        dvfs_enable = 0;
        repeat(10) @(posedge clk);
        
        check_result("DVFS disabled - No power gating", 
                    core_power_gate == '0 && !memory_power_gate && !ai_accel_power_gate);
        
        // Test 10: Voltage level constraints
        $display("\n--- Test 10: Voltage Level Constraints ---");
        dvfs_enable = 1;
        min_voltage_level = 3'd3;
        max_voltage_level = 3'd5;
        set_system_load(16'hF000);  // Very high load
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("Voltage constraint - Max level respected", voltage_level <= max_voltage_level);
        
        set_system_load(16'h0100);  // Very low load
        repeat(20) @(posedge clk);
        wait_for_transition();
        
        check_result("Voltage constraint - Min level respected", voltage_level >= min_voltage_level);
        
        // Test summary
        $display("\n=== Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000000;  // 1ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_dvfs_controller.vcd");
        $dumpvars(0, test_dvfs_controller);
    end

endmodule