// System Integration Testbench
// Top-level testbench for system integration testing

`timescale 1ns/1ps

`include "uvm_macros.svh"
import uvm_pkg::*;
import system_integration_pkg::*;

module tb_system_integration;

    // Clock and reset generation
    logic clk = 0;
    logic rst_n = 0;
    
    // System clock - 1GHz
    always #0.5 clk = ~clk;
    
    // Reset sequence
    initial begin
        rst_n = 0;
        repeat(20) @(posedge clk);
        rst_n = 1;
        `uvm_info("TB", "System reset released", UVM_LOW)
    end
    
    // Interface instantiation
    riscv_ai_interface ai_if(clk, rst_n);
    
    // System-level DUT (comprehensive system model)
    riscv_ai_system_dut system_dut (
        .clk(clk),
        .rst_n(rst_n),
        .ai_if(ai_if.dut_mp)
    );
    
    // UVM test execution
    initial begin
        // Set interface in config DB
        uvm_config_db#(virtual riscv_ai_interface)::set(null, "*", "vif", ai_if);
        
        // Configure test timeout for long integration tests
        uvm_top.set_timeout(10s, 0);  // 10 second timeout
        
        // Start UVM test
        run_test();
    end
    
    // Waveform dumping for debug
    initial begin
        if ($test$plusargs("DUMP_WAVES")) begin
            $dumpfile("tb_system_integration.vcd");
            $dumpvars(0, tb_system_integration);
        end
    end
    
    // System-level monitoring
    initial begin
        forever begin
            @(posedge clk);
            // Monitor system health
            if (system_dut.temperature > 90) begin
                `uvm_fatal("TB", $sformatf("System overheating: %0d°C", system_dut.temperature))
            end
            if (system_dut.power_consumption > 150) begin
                `uvm_fatal("TB", $sformatf("Power consumption too high: %0dW", system_dut.power_consumption))
            end
        end
    end

endmodule

// Comprehensive system DUT model for integration testing
module riscv_ai_system_dut (
    input logic clk,
    input logic rst_n,
    riscv_ai_interface.dut_mp ai_if
);

    // System state
    logic [7:0] temperature = 45;
    logic [15:0] power_consumption = 50;
    logic [31:0] system_cycles = 0;
    
    // Multi-core state
    logic [3:0] core_active = 4'b0000;
    logic [31:0] core_utilization[4] = '{0, 0, 0, 0};
    
    // Cache state
    logic [31:0] cache_hits[4] = '{0, 0, 0, 0};
    logic [31:0] cache_misses[4] = '{0, 0, 0, 0};
    
    // NoC state
    logic [31:0] noc_packets_sent = 0;
    logic [31:0] noc_packets_received = 0;
    
    // System cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            system_cycles <= 0;
        end else begin
            system_cycles <= system_cycles + 1;
        end
    end
    
    // Multi-core simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_active <= 4'b0000;
            for (int i = 0; i < 4; i++) core_utilization[i] <= 0;
        end else begin
            // Simulate core activity
            for (int i = 0; i < 4; i++) begin
                if ($urandom_range(0, 99) < 30) begin  // 30% chance of activity
                    core_active[i] <= 1;
                    core_utilization[i] <= core_utilization[i] + 1;
                end else begin
                    core_active[i] <= 0;
                end
            end
        end
    end
    
    // Cache simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                cache_hits[i] <= 0;
                cache_misses[i] <= 0;
            end
        end else begin
            // Simulate cache accesses
            for (int i = 0; i < 4; i++) begin
                if (core_active[i]) begin
                    if ($urandom_range(0, 99) < 85) begin  // 85% hit rate
                        cache_hits[i] <= cache_hits[i] + 1;
                    end else begin
                        cache_misses[i] <= cache_misses[i] + 1;
                    end
                end
            end
        end
    end
    
    // NoC simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            noc_packets_sent <= 0;
            noc_packets_received <= 0;
        end else begin
            // Simulate NoC traffic
            if (|core_active && $urandom_range(0, 99) < 20) begin
                noc_packets_sent <= noc_packets_sent + 1;
                noc_packets_received <= noc_packets_received + 1;
            end
        end
    end
    
    // Power and thermal simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_consumption <= 50;
            temperature <= 45;
        end else begin
            // Power based on activity
            int active_cores = $countones(core_active);
            power_consumption <= 50 + (active_cores * 15) + $urandom_range(0, 10);
            
            // Temperature follows power with thermal inertia
            int target_temp = 45 + ((power_consumption - 50) >> 2);
            if (temperature < target_temp) begin
                temperature <= temperature + 1;
            end else if (temperature > target_temp) begin
                temperature <= temperature - 1;
            end
        end
    end
    
    // Interface connections
    assign ai_if.ready = 1'b1;  // Always ready for integration testing
    assign ai_if.response_valid = ai_if.valid;  // Immediate response
    assign ai_if.response_data = ai_if.data + 64'h1;  // Simple response
    assign ai_if.error = 1'b0;  // No errors in basic model
    assign ai_if.performance_counter = system_cycles;
    assign ai_if.power_consumption = power_consumption;
    assign ai_if.temperature = temperature;
    assign ai_if.debug_state = {28'b0, core_active};
    assign ai_if.debug_pc = {32'b0, system_cycles};
    assign ai_if.debug_halt = 1'b0;

endmodule