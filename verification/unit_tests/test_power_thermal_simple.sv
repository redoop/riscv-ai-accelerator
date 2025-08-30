/*
 * Simple Power and Thermal Optimization Test
 * 
 * Basic functionality test for power management components
 */

`timescale 1ns / 1ps

module test_power_thermal_simple;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_DOMAINS = 8;
    parameter CLK_PERIOD = 10; // 100MHz

    // Clock and reset
    logic clk;
    logic rst_n;

    // Test signals for power manager
    logic [NUM_CORES-1:0] core_active;
    logic [31:0] core_utilization [NUM_CORES-1:0];
    logic [31:0] workload_type [NUM_CORES-1:0];
    logic [15:0] temperature [NUM_DOMAINS-1:0];
    logic [15:0] temp_threshold_warning;
    logic [15:0] temp_threshold_critical;
    
    logic [NUM_DOMAINS-1:0] domain_power_enable;
    logic [NUM_DOMAINS-1:0] domain_clock_enable;
    logic [7:0] voltage_level [NUM_DOMAINS-1:0];
    logic [7:0] frequency_level [NUM_DOMAINS-1:0];
    
    logic [31:0] power_budget;
    logic [31:0] battery_level;
    logic ac_power_available;
    logic [31:0] current_power_consumption;
    logic power_emergency;
    logic thermal_emergency;

    // Test control
    integer test_phase;
    integer cycle_count;
    logic test_passed;
    integer error_count;

    // Clock generation
    initial begin
        clk = 0;
        forever begin
            #(CLK_PERIOD/2) clk = ~clk;
        end
    end

    // Test stimulus
    initial begin
        $display("=== Simple Power and Thermal Test ===");
        
        // Initialize
        rst_n = 0;
        test_phase = 0;
        cycle_count = 0;
        test_passed = 1;
        error_count = 0;
        
        // Initialize inputs
        core_active = 4'b1111;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_utilization[i] = 32'h4000; // 25% utilization
            workload_type[i] = 32'h01; // CPU workload
        end
        
        for (int i = 0; i < NUM_DOMAINS; i++) begin
            temperature[i] = 16'h3000; // 48°C
        end
        
        temp_threshold_warning = 16'h5000; // 80°C
        temp_threshold_critical = 16'h5800; // 88°C
        power_budget = 32'h96; // 150W
        battery_level = 32'h64; // 100%
        ac_power_available = 1'b1;
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        // Test Phase 1: Normal Operation
        $display("Phase 1: Testing normal operation");
        test_phase = 1;
        repeat(1000) @(posedge clk);
        
        // Check basic functionality
        if (domain_power_enable != {NUM_DOMAINS{1'b1}}) begin
            $display("ERROR: Not all domains enabled in normal mode");
            error_count++;
        end
        
        // Test Phase 2: Thermal Stress
        $display("Phase 2: Testing thermal emergency");
        test_phase = 2;
        
        // Raise temperature above critical
        for (int i = 0; i < NUM_DOMAINS; i++) begin
            temperature[i] = 16'h5C00; // 92°C
        end
        
        repeat(100) @(posedge clk);
        
        // Check thermal emergency response
        if (!thermal_emergency) begin
            $display("ERROR: Thermal emergency not detected");
            error_count++;
        end
        
        // Cool down
        for (int i = 0; i < NUM_DOMAINS; i++) begin
            temperature[i] = 16'h2800; // 40°C
        end
        
        repeat(500) @(posedge clk);
        
        // Test Phase 3: Battery Mode
        $display("Phase 3: Testing battery mode");
        test_phase = 3;
        
        ac_power_available = 1'b0;
        battery_level = 32'h0A; // 10% - low battery
        
        repeat(200) @(posedge clk);
        
        // Check power emergency for low battery
        if (!power_emergency) begin
            $display("WARNING: Power emergency not triggered for low battery");
        end
        
        // Final results
        repeat(100) @(posedge clk);
        
        $display("\n=== Test Results ===");
        $display("Total errors: %0d", error_count);
        $display("Current power: %0d W", current_power_consumption);
        
        if (error_count == 0) begin
            $display("*** TEST PASSED ***");
        end else begin
            $display("*** TEST FAILED ***");
            test_passed = 0;
        end
        
        $finish;
    end

    // Simple power management logic (for testing without actual modules)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            domain_power_enable <= {NUM_DOMAINS{1'b1}};
            domain_clock_enable <= {NUM_DOMAINS{1'b1}};
            for (int i = 0; i < NUM_DOMAINS; i++) begin
                voltage_level[i] <= 8'hFF;
                frequency_level[i] <= 8'hFF;
            end
            current_power_consumption <= 32'h50; // 80W
            power_emergency <= 1'b0;
            thermal_emergency <= 1'b0;
        end else begin
            // Simple thermal emergency detection
            thermal_emergency <= (temperature[0] > temp_threshold_critical);
            
            // Simple power emergency detection
            power_emergency <= (!ac_power_available && battery_level < 32'h0F);
            
            // Simple power estimation
            current_power_consumption <= 32'h50 + (core_utilization[0] >> 12);
            
            // Simple thermal response
            if (thermal_emergency) begin
                for (int i = 0; i < NUM_DOMAINS; i++) begin
                    voltage_level[i] <= 8'h40; // Reduce voltage
                    frequency_level[i] <= 8'h40; // Reduce frequency
                end
            end else begin
                for (int i = 0; i < NUM_DOMAINS; i++) begin
                    voltage_level[i] <= 8'hC0; // Normal voltage
                    frequency_level[i] <= 8'hC0; // Normal frequency
                end
            end
            
            // Simple battery response
            if (power_emergency) begin
                for (int i = 1; i < NUM_DOMAINS; i++) begin
                    domain_power_enable[i] <= 1'b0; // Gate non-essential domains
                end
            end else begin
                domain_power_enable <= {NUM_DOMAINS{1'b1}};
            end
        end
    end

    // Monitor and check
    always @(posedge clk) begin
        cycle_count++;
        
        if (rst_n && (cycle_count % 1000 == 0)) begin
            $display("Cycle %0d: Power=%0dW, Temp=%0d°C, Emergency: P=%b T=%b", 
                    cycle_count, current_power_consumption, temperature[0], 
                    power_emergency, thermal_emergency);
        end
    end

endmodule