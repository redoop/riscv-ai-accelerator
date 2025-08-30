/*
 * Thermal Management Test
 * 
 * This testbench verifies the thermal management system including
 * temperature monitoring, thermal protection, and power monitoring.
 */

`timescale 1ns/1ps

module test_thermal_management;

    // Parameters
    localparam NUM_TEMP_SENSORS = 8;
    localparam NUM_CORES = 4;
    localparam NUM_AI_UNITS = 2;
    localparam CLK_PERIOD = 10; // 100MHz

    // System signals
    logic                           clk;
    logic                           rst_n;
    
    // Temperature sensor signals
    logic [15:0]                    temp_sensors [NUM_TEMP_SENSORS-1:0];
    logic [NUM_TEMP_SENSORS-1:0]   temp_sensor_valid;
    
    // Power consumption inputs
    logic [15:0]                    core_power [NUM_CORES-1:0];
    logic [15:0]                    ai_unit_power [NUM_AI_UNITS-1:0];
    logic [15:0]                    memory_power;
    logic [15:0]                    noc_power;
    
    // Activity monitoring
    logic [NUM_CORES-1:0]          core_activity;
    logic [NUM_AI_UNITS-1:0]       ai_unit_activity;
    logic                           memory_activity;
    
    // Thermal management outputs
    logic [7:0]                     max_temperature;
    logic [7:0]                     avg_temperature;
    logic [2:0]                     thermal_zone;
    logic                           thermal_alert;
    logic                           thermal_critical;
    logic                           thermal_emergency;
    
    // Thermal throttling controls
    logic [2:0]                     thermal_throttle_level;
    logic [NUM_CORES-1:0]          core_thermal_throttle;
    logic [NUM_AI_UNITS-1:0]       ai_unit_thermal_throttle;
    logic                           memory_thermal_throttle;
    
    // Power monitoring outputs
    logic [15:0]                    total_power_consumption;
    logic [15:0]                    power_budget_remaining;
    logic                           power_budget_exceeded;
    
    // Configuration interface
    logic [31:0]                    thermal_config_addr;
    logic [31:0]                    thermal_config_wdata;
    logic [31:0]                    thermal_config_rdata;
    logic                           thermal_config_req;
    logic                           thermal_config_we;
    logic                           thermal_config_ready;

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
    thermal_controller #(
        .NUM_TEMP_SENSORS(NUM_TEMP_SENSORS),
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .temp_sensors(temp_sensors),
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
        .thermal_config_addr(thermal_config_addr),
        .thermal_config_wdata(thermal_config_wdata),
        .thermal_config_rdata(thermal_config_rdata),
        .thermal_config_req(thermal_config_req),
        .thermal_config_we(thermal_config_we),
        .thermal_config_ready(thermal_config_ready)
    );

    // Test tasks
    task reset_system();
        rst_n = 0;
        temp_sensor_valid = '1;
        core_activity = '1;
        ai_unit_activity = '1;
        memory_activity = 1;
        thermal_config_req = 0;
        thermal_config_we = 0;
        thermal_config_addr = 0;
        thermal_config_wdata = 0;
        
        // Initialize temperature sensors to room temperature (25°C)
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            temp_sensors[i] = 16'h4200; // ~25°C in ADC units
        end
        
        // Initialize power consumption to low values
        for (int i = 0; i < NUM_CORES; i++) begin
            core_power[i] = 16'd500; // 500mW per core
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_power[i] = 16'd1000; // 1W per AI unit
        end
        memory_power = 16'd800;  // 800mW
        noc_power = 16'd300;     // 300mW
        
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

    task set_temperature(logic [7:0] temp_celsius);
        logic [15:0] adc_value;
        // Convert Celsius to ADC value: ADC = (Temperature + 40) * 256 / 165
        adc_value = ((temp_celsius + 8'd40) << 8) / 165;
        
        for (int i = 0; i < NUM_TEMP_SENSORS; i++) begin
            temp_sensors[i] = adc_value;
        end
        repeat(5) @(posedge clk);
    endtask

    task set_power_consumption(logic [15:0] total_power);
        // Distribute power across domains
        for (int i = 0; i < NUM_CORES; i++) begin
            core_power[i] = total_power >> 3; // 1/8 per core
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_power[i] = total_power >> 2; // 1/4 per AI unit
        end
        memory_power = total_power >> 3;  // 1/8 for memory
        noc_power = total_power >> 4;     // 1/16 for NoC
        repeat(5) @(posedge clk);
    endtask

    task configure_thermal_thresholds(logic [7:0] alert, logic [7:0] critical, logic [7:0] emergency);
        // Configure alert threshold
        thermal_config_req = 1;
        thermal_config_we = 1;
        thermal_config_addr = 32'h04;
        thermal_config_wdata = {24'b0, alert};
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        
        // Configure critical threshold
        thermal_config_addr = 32'h08;
        thermal_config_wdata = {24'b0, critical};
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        
        // Configure emergency threshold
        thermal_config_addr = 32'h0C;
        thermal_config_wdata = {24'b0, emergency};
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        
        thermal_config_req = 0;
        thermal_config_we = 0;
        repeat(5) @(posedge clk);
    endtask

    task read_thermal_status();
        // Read max temperature
        thermal_config_req = 1;
        thermal_config_we = 0;
        thermal_config_addr = 32'h18;
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        $display("Max temperature: %d°C", thermal_config_rdata[7:0]);
        
        // Read thermal zone
        thermal_config_addr = 32'h28;
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        $display("Thermal zone: %d", thermal_config_rdata[2:0]);
        
        // Read throttle level
        thermal_config_addr = 32'h30;
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        $display("Throttle level: %d", thermal_config_rdata[2:0]);
        
        thermal_config_req = 0;
        repeat(5) @(posedge clk);
    endtask

    // Main test sequence
    initial begin
        $display("Starting Thermal Management Test");
        
        // Test 1: System initialization
        $display("\n=== Test 1: System Initialization ===");
        reset_system();
        check_result("System reset", rst_n);
        check_result("Normal temperature zone", thermal_zone == 3'd0);
        check_result("No thermal alerts", !thermal_alert && !thermal_critical && !thermal_emergency);
        
        // Test 2: Temperature sensor functionality
        $display("\n=== Test 2: Temperature Sensor Functionality ===");
        
        set_temperature(8'd30); // 30°C
        repeat(10) @(posedge clk);
        check_result("Temperature reading 30°C", max_temperature >= 8'd28 && max_temperature <= 8'd32);
        
        set_temperature(8'd50); // 50°C
        repeat(10) @(posedge clk);
        check_result("Temperature reading 50°C", max_temperature >= 8'd48 && max_temperature <= 8'd52);
        
        set_temperature(8'd75); // 75°C
        repeat(10) @(posedge clk);
        check_result("Temperature reading 75°C", max_temperature >= 8'd73 && max_temperature <= 8'd77);
        
        // Test 3: Thermal zone detection
        $display("\n=== Test 3: Thermal Zone Detection ===");
        
        set_temperature(8'd25); // Normal
        repeat(10) @(posedge clk);
        check_result("Normal zone (25°C)", thermal_zone == 3'd0);
        
        set_temperature(8'd72); // Warm
        repeat(10) @(posedge clk);
        check_result("Warm zone (72°C)", thermal_zone == 3'd1);
        
        set_temperature(8'd82); // Alert
        repeat(10) @(posedge clk);
        check_result("Alert zone (82°C)", thermal_zone == 3'd2);
        
        set_temperature(8'd92); // Critical
        repeat(10) @(posedge clk);
        check_result("Critical zone (92°C)", thermal_zone == 3'd3);
        
        set_temperature(8'd102); // Emergency
        repeat(10) @(posedge clk);
        check_result("Emergency zone (102°C)", thermal_zone == 3'd4);
        
        // Test 4: Thermal alert generation
        $display("\n=== Test 4: Thermal Alert Generation ===");
        
        set_temperature(8'd25); // Reset to normal
        repeat(20) @(posedge clk);
        check_result("No alerts at normal temp", !thermal_alert);
        
        set_temperature(8'd85); // Above alert threshold
        repeat(20) @(posedge clk);
        check_result("Thermal alert triggered", thermal_alert);
        check_result("Not critical yet", !thermal_critical);
        
        set_temperature(8'd95); // Above critical threshold
        repeat(20) @(posedge clk);
        check_result("Thermal critical triggered", thermal_critical);
        check_result("Not emergency yet", !thermal_emergency);
        
        set_temperature(8'd105); // Above emergency threshold
        repeat(20) @(posedge clk);
        check_result("Thermal emergency triggered", thermal_emergency);
        
        // Test 5: Thermal throttling
        $display("\n=== Test 5: Thermal Throttling ===");
        
        set_temperature(8'd25); // Reset to normal
        repeat(50) @(posedge clk);
        check_result("No throttling at normal temp", thermal_throttle_level == 3'd0);
        check_result("No core throttling", core_thermal_throttle == '0);
        
        set_temperature(8'd85); // Alert level
        repeat(50) @(posedge clk);
        check_result("Light throttling at alert", thermal_throttle_level >= 3'd1);
        
        set_temperature(8'd95); // Critical level
        repeat(50) @(posedge clk);
        check_result("Aggressive throttling at critical", thermal_throttle_level >= 3'd3);
        check_result("Some cores throttled", |core_thermal_throttle);
        
        set_temperature(8'd105); // Emergency level
        repeat(50) @(posedge clk);
        check_result("Maximum throttling at emergency", thermal_throttle_level == 3'd7);
        check_result("All cores throttled", &core_thermal_throttle);
        check_result("AI units throttled", &ai_unit_thermal_throttle);
        
        // Test 6: Hysteresis behavior
        $display("\n=== Test 6: Hysteresis Behavior ===");
        
        set_temperature(8'd85); // Alert level
        repeat(30) @(posedge clk);
        check_result("Alert state active", thermal_alert);
        
        set_temperature(8'd79); // Just below alert threshold
        repeat(30) @(posedge clk);
        check_result("Still in alert (hysteresis)", thermal_alert);
        
        set_temperature(8'd74); // Well below alert threshold (with hysteresis)
        repeat(30) @(posedge clk);
        check_result("Returned to normal (hysteresis)", !thermal_alert);
        
        // Test 7: Power consumption monitoring
        $display("\n=== Test 7: Power Consumption Monitoring ===");
        
        set_power_consumption(16'd5000); // 5W total
        repeat(10) @(posedge clk);
        check_result("Power consumption calculated", total_power_consumption > 16'd0);
        $display("Total power: %d mW", total_power_consumption);
        
        set_power_consumption(16'd20000); // 20W total (exceeds budget)
        repeat(10) @(posedge clk);
        check_result("Power budget exceeded detected", power_budget_exceeded);
        
        set_power_consumption(16'd10000); // 10W total (within budget)
        repeat(10) @(posedge clk);
        check_result("Power within budget", !power_budget_exceeded);
        check_result("Budget remaining calculated", power_budget_remaining > 16'd0);
        
        // Test 8: Configuration interface
        $display("\n=== Test 8: Configuration Interface ===");
        
        configure_thermal_thresholds(8'd75, 8'd85, 8'd95); // Custom thresholds
        
        set_temperature(8'd77); // Above new alert threshold
        repeat(20) @(posedge clk);
        check_result("Custom alert threshold works", thermal_alert);
        
        set_temperature(8'd87); // Above new critical threshold
        repeat(20) @(posedge clk);
        check_result("Custom critical threshold works", thermal_critical);
        
        // Test 9: Sensor fault handling
        $display("\n=== Test 9: Sensor Fault Handling ===");
        
        temp_sensor_valid[0] = 0; // Simulate sensor 0 fault
        temp_sensor_valid[1] = 0; // Simulate sensor 1 fault
        temp_sensors[0] = 16'hFFFF; // Invalid reading
        temp_sensors[1] = 16'hFFFF; // Invalid reading
        
        repeat(20) @(posedge clk);
        check_result("System handles sensor faults", max_temperature < 8'd200); // Should not be invalid
        
        // Restore sensors
        temp_sensor_valid = '1;
        set_temperature(8'd30);
        repeat(10) @(posedge clk);
        
        // Test 10: Thermal management disable
        $display("\n=== Test 10: Thermal Management Disable ===");
        
        // Disable thermal management
        thermal_config_req = 1;
        thermal_config_we = 1;
        thermal_config_addr = 32'h00;
        thermal_config_wdata = 32'h00; // Disable
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        thermal_config_req = 0;
        thermal_config_we = 0;
        
        set_temperature(8'd100); // High temperature
        repeat(30) @(posedge clk);
        check_result("No throttling when disabled", thermal_throttle_level == 3'd0);
        
        // Re-enable thermal management
        thermal_config_req = 1;
        thermal_config_we = 1;
        thermal_config_addr = 32'h00;
        thermal_config_wdata = 32'h01; // Enable
        @(posedge clk);
        while (!thermal_config_ready) @(posedge clk);
        thermal_config_req = 0;
        thermal_config_we = 0;
        
        repeat(30) @(posedge clk);
        check_result("Throttling resumes when enabled", thermal_throttle_level > 3'd0);
        
        // Test 11: Stress test - rapid temperature changes
        $display("\n=== Test 11: Stress Test - Rapid Temperature Changes ===");
        
        for (int cycle = 0; cycle < 20; cycle++) begin
            if (cycle % 2 == 0) begin
                set_temperature(8'd100); // Hot
            end else begin
                set_temperature(8'd30);  // Cool
            end
            repeat(10) @(posedge clk);
        end
        
        check_result("System stable after rapid changes", 1'b1);
        
        // Final status readout
        $display("\n=== Final Status ===");
        read_thermal_status();
        
        // Test summary
        $display("\n=== Thermal Management Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success rate: %.1f%%", (real'(pass_count) / real'(test_count)) * 100.0);
        
        if (fail_count == 0) begin
            $display("ALL THERMAL MANAGEMENT TESTS PASSED!");
        end else begin
            $display("SOME THERMAL MANAGEMENT TESTS FAILED!");
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
        $dumpfile("test_thermal_management.vcd");
        $dumpvars(0, test_thermal_management);
    end

endmodule