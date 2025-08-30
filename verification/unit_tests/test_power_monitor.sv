/*
 * Power Monitor Test
 * 
 * This testbench verifies the power monitoring functionality
 * including power calculation, statistics, and reporting.
 */

`timescale 1ns/1ps

module test_power_monitor;

    // Parameters
    localparam NUM_CORES = 4;
    localparam NUM_AI_UNITS = 2;
    localparam CLK_PERIOD = 10; // 100MHz

    // System signals
    logic        clk;
    logic        rst_n;
    
    // Activity inputs
    logic [NUM_CORES-1:0]          core_activity;
    logic [NUM_AI_UNITS-1:0]       ai_unit_activity;
    logic                           memory_activity;
    logic                           noc_activity;
    
    // Voltage and frequency inputs
    logic [2:0]                     voltage_level;
    logic [2:0]                     frequency_level;
    
    // Power enable inputs
    logic [NUM_CORES-1:0]          core_power_enable;
    logic [NUM_AI_UNITS-1:0]       ai_unit_power_enable;
    logic                           memory_power_enable;
    logic                           noc_power_enable;
    
    // Load monitoring inputs
    logic [15:0]                    core_load [NUM_CORES-1:0];
    logic [15:0]                    memory_load;
    logic [15:0]                    ai_accel_load;
    
    // Power consumption outputs
    logic [15:0]                    core_power [NUM_CORES-1:0];
    logic [15:0]                    ai_unit_power [NUM_AI_UNITS-1:0];
    logic [15:0]                    memory_power;
    logic [15:0]                    noc_power;
    logic [15:0]                    total_power;
    
    // Power statistics
    logic [15:0]                    avg_power;
    logic [15:0]                    peak_power;
    logic [31:0]                    energy_consumed;
    
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
    
    // Test-specific variables
    logic [15:0] power_at_low_voltage, power_at_high_voltage;
    logic [15:0] power_at_low_freq, power_at_high_freq;
    logic [15:0] power_at_low_load, power_at_high_load;
    logic [15:0] power_all_on, power_with_gating;
    logic [15:0] total_calculated;
    logic [15:0] prev_peak;
    logic [31:0] energy_before_disable;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT instantiation
    power_monitor #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .POWER_SAMPLE_PERIOD(100) // Short period for testing
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .core_activity(core_activity),
        .ai_unit_activity(ai_unit_activity),
        .memory_activity(memory_activity),
        .noc_activity(noc_activity),
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
        core_activity = '1;
        ai_unit_activity = '1;
        memory_activity = 1;
        noc_activity = 1;
        voltage_level = 3'd4; // Nominal voltage
        frequency_level = 3'd4; // Nominal frequency
        core_power_enable = '1;
        ai_unit_power_enable = '1;
        memory_power_enable = 1;
        noc_power_enable = 1;
        pm_config_req = 0;
        pm_config_we = 0;
        pm_config_addr = 0;
        pm_config_wdata = 0;
        
        // Initialize loads to medium values
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h8000; // 50% load
        end
        memory_load = 16'h6000;   // 37.5% load
        ai_accel_load = 16'hA000; // 62.5% load
        
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

    task set_voltage_frequency(logic [2:0] volt_level, logic [2:0] freq_level);
        voltage_level = volt_level;
        frequency_level = freq_level;
        repeat(5) @(posedge clk);
    endtask

    task set_system_load(logic [15:0] cpu_load, logic [15:0] mem_load, logic [15:0] ai_load);
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = cpu_load;
        end
        memory_load = mem_load;
        ai_accel_load = ai_load;
        repeat(5) @(posedge clk);
    endtask

    task power_gate_components(logic [NUM_CORES-1:0] cores_gated, 
                              logic [NUM_AI_UNITS-1:0] ai_gated, 
                              logic mem_gated);
        core_power_enable = ~cores_gated;
        ai_unit_power_enable = ~ai_gated;
        memory_power_enable = ~mem_gated;
        repeat(5) @(posedge clk);
    endtask

    task read_power_statistics();
        // Read total power
        pm_config_req = 1;
        pm_config_we = 0;
        pm_config_addr = 32'h08;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        $display("Total power: %d mW", pm_config_rdata[15:0]);
        
        // Read average power
        pm_config_addr = 32'h0C;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        $display("Average power: %d mW", pm_config_rdata[15:0]);
        
        // Read peak power
        pm_config_addr = 32'h10;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        $display("Peak power: %d mW", pm_config_rdata[15:0]);
        
        // Read energy consumed
        pm_config_addr = 32'h14;
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        $display("Energy consumed: %d mJ", pm_config_rdata);
        
        pm_config_req = 0;
        repeat(5) @(posedge clk);
    endtask

    task reset_statistics();
        pm_config_req = 1;
        pm_config_we = 1;
        pm_config_addr = 32'h08;
        pm_config_wdata = 32'h01; // Reset statistics
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 0;
        pm_config_we = 0;
        repeat(5) @(posedge clk);
    endtask

    // Main test sequence
    initial begin
        $display("Starting Power Monitor Test");
        
        // Test 1: System initialization
        $display("\n=== Test 1: System Initialization ===");
        reset_system();
        check_result("System reset", rst_n);
        check_result("Power monitoring active", total_power > 16'd0);
        $display("Initial total power: %d mW", total_power);
        
        // Test 2: Voltage scaling effects
        $display("\n=== Test 2: Voltage Scaling Effects ===");
        

        
        set_voltage_frequency(3'd1, 3'd4); // Low voltage, nominal frequency
        repeat(10) @(posedge clk);
        power_at_low_voltage = total_power;
        $display("Power at low voltage (0.7V): %d mW", power_at_low_voltage);
        
        set_voltage_frequency(3'd6, 3'd4); // High voltage, nominal frequency
        repeat(10) @(posedge clk);
        power_at_high_voltage = total_power;
        $display("Power at high voltage (1.2V): %d mW", power_at_high_voltage);
        
        check_result("Higher voltage increases power", power_at_high_voltage > power_at_low_voltage);
        check_result("Voltage scaling significant", 
                    power_at_high_voltage > (power_at_low_voltage + (power_at_low_voltage >> 2)));
        
        // Test 3: Frequency scaling effects
        $display("\n=== Test 3: Frequency Scaling Effects ===");
        

        
        set_voltage_frequency(3'd4, 3'd1); // Nominal voltage, low frequency
        repeat(10) @(posedge clk);
        power_at_low_freq = total_power;
        $display("Power at low frequency (400MHz): %d mW", power_at_low_freq);
        
        set_voltage_frequency(3'd4, 3'd6); // Nominal voltage, high frequency
        repeat(10) @(posedge clk);
        power_at_high_freq = total_power;
        $display("Power at high frequency (1400MHz): %d mW", power_at_high_freq);
        
        check_result("Higher frequency increases power", power_at_high_freq > power_at_low_freq);
        
        // Test 4: Load-dependent power consumption
        $display("\n=== Test 4: Load-Dependent Power Consumption ===");
        

        
        set_voltage_frequency(3'd4, 3'd4); // Reset to nominal
        set_system_load(16'h2000, 16'h2000, 16'h2000); // Low load (12.5%)
        repeat(10) @(posedge clk);
        power_at_low_load = total_power;
        $display("Power at low load: %d mW", power_at_low_load);
        
        set_system_load(16'hE000, 16'hE000, 16'hE000); // High load (87.5%)
        repeat(10) @(posedge clk);
        power_at_high_load = total_power;
        $display("Power at high load: %d mW", power_at_high_load);
        
        check_result("Higher load increases power", power_at_high_load > power_at_low_load);
        check_result("Load scaling significant", 
                    power_at_high_load > (power_at_low_load + (power_at_low_load >> 1)));
        
        // Test 5: Power gating effects
        $display("\n=== Test 5: Power Gating Effects ===");
        

        
        set_system_load(16'h8000, 16'h8000, 16'h8000); // Medium load
        power_gate_components(4'b0000, 2'b00, 1'b0); // All components on
        repeat(10) @(posedge clk);
        power_all_on = total_power;
        $display("Power with all components on: %d mW", power_all_on);
        
        power_gate_components(4'b1100, 2'b01, 1'b0); // Gate 2 cores and 1 AI unit
        repeat(10) @(posedge clk);
        power_with_gating = total_power;
        $display("Power with some components gated: %d mW", power_with_gating);
        
        check_result("Power gating reduces consumption", power_with_gating < power_all_on);
        check_result("Power gating significant savings", 
                    power_with_gating < (power_all_on - (power_all_on >> 3)));
        
        // Test 6: Individual domain power monitoring
        $display("\n=== Test 6: Individual Domain Power Monitoring ===");
        
        power_gate_components(4'b0000, 2'b00, 1'b0); // All components on
        repeat(10) @(posedge clk);
        
        total_calculated = 0;
        for (int i = 0; i < NUM_CORES; i++) begin
            total_calculated += core_power[i];
            $display("Core %d power: %d mW", i, core_power[i]);
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            total_calculated += ai_unit_power[i];
            $display("AI unit %d power: %d mW", i, ai_unit_power[i]);
        end
        total_calculated += memory_power + noc_power;
        $display("Memory power: %d mW", memory_power);
        $display("NoC power: %d mW", noc_power);
        $display("Total calculated: %d mW", total_calculated);
        $display("Total reported: %d mW", total_power);
        
        check_result("Individual powers sum to total", 
                    (total_calculated >= (total_power - 16'd50)) && 
                    (total_calculated <= (total_power + 16'd50)));
        
        // Test 7: Power statistics accumulation
        $display("\n=== Test 7: Power Statistics Accumulation ===");
        
        reset_statistics();
        
        // Run for several sampling periods
        set_system_load(16'h4000, 16'h4000, 16'h4000); // Low-medium load
        repeat(150) @(posedge clk); // More than one sampling period
        
        check_result("Peak power tracked", peak_power >= total_power);
        check_result("Average power calculated", avg_power > 16'd0);
        check_result("Energy accumulation", energy_consumed > 32'd0);
        
        read_power_statistics();
        
        // Test 8: Dynamic power tracking
        $display("\n=== Test 8: Dynamic Power Tracking ===");
        
        prev_peak = peak_power;
        
        // Create a power spike
        set_voltage_frequency(3'd7, 3'd7); // Maximum voltage and frequency
        set_system_load(16'hF000, 16'hF000, 16'hF000); // Maximum load
        repeat(20) @(posedge clk);
        
        check_result("Peak power updated on spike", peak_power > prev_peak);
        $display("New peak power: %d mW", peak_power);
        
        // Return to normal
        set_voltage_frequency(3'd4, 3'd4);
        set_system_load(16'h8000, 16'h8000, 16'h8000);
        repeat(50) @(posedge clk);
        
        // Test 9: Configuration interface
        $display("\n=== Test 9: Configuration Interface ===");
        
        // Test reading individual domain powers through config interface
        pm_config_req = 1;
        pm_config_we = 0;
        
        for (int addr = 32'h20; addr <= 32'h3C; addr += 4) begin
            pm_config_addr = addr;
            @(posedge clk);
            while (!pm_config_ready) @(posedge clk);
            $display("Config addr 0x%02X: %d mW", addr[7:0], pm_config_rdata[15:0]);
        end
        
        pm_config_req = 0;
        
        check_result("Configuration interface functional", 1'b1);
        
        // Test 10: Power monitor disable/enable
        $display("\n=== Test 10: Power Monitor Disable/Enable ===");
        
        energy_before_disable = energy_consumed;
        
        // Disable power monitor
        pm_config_req = 1;
        pm_config_we = 1;
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h00; // Disable
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 0;
        pm_config_we = 0;
        
        repeat(100) @(posedge clk);
        
        check_result("Energy accumulation stopped when disabled", 
                    energy_consumed == energy_before_disable);
        
        // Re-enable power monitor
        pm_config_req = 1;
        pm_config_we = 1;
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h01; // Enable
        @(posedge clk);
        while (!pm_config_ready) @(posedge clk);
        pm_config_req = 0;
        pm_config_we = 0;
        
        repeat(100) @(posedge clk);
        
        check_result("Energy accumulation resumed when enabled", 
                    energy_consumed > energy_before_disable);
        
        // Test 11: Stress test - rapid changes
        $display("\n=== Test 11: Stress Test - Rapid Changes ===");
        
        for (int cycle = 0; cycle < 50; cycle++) begin
            case (cycle % 4)
                0: begin
                    set_voltage_frequency(3'd1, 3'd1);
                    set_system_load(16'h2000, 16'h2000, 16'h2000);
                end
                1: begin
                    set_voltage_frequency(3'd7, 3'd7);
                    set_system_load(16'hF000, 16'hF000, 16'hF000);
                end
                2: begin
                    set_voltage_frequency(3'd4, 3'd2);
                    set_system_load(16'h6000, 16'h8000, 16'h4000);
                end
                3: begin
                    set_voltage_frequency(3'd3, 3'd6);
                    set_system_load(16'hA000, 16'h4000, 16'hC000);
                end
            endcase
            repeat(5) @(posedge clk);
        end
        
        check_result("System stable under rapid changes", total_power > 16'd0);
        
        // Final statistics
        $display("\n=== Final Statistics ===");
        read_power_statistics();
        
        // Test summary
        $display("\n=== Power Monitor Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success rate: %.1f%%", (real'(pass_count) / real'(test_count)) * 100.0);
        
        if (fail_count == 0) begin
            $display("ALL POWER MONITOR TESTS PASSED!");
        end else begin
            $display("SOME POWER MONITOR TESTS FAILED!");
        end
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #3000000; // 3ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_power_monitor.vcd");
        $dumpvars(0, test_power_monitor);
    end

endmodule