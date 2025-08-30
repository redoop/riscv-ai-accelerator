/*
 * Performance Optimization System Verification Test
 * 
 * Comprehensive test suite for performance monitoring, adaptive tuning,
 * and workload-aware resource scheduling components.
 */

`timescale 1ns / 1ps

module test_performance_optimization;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_AI_UNITS = 2;
    parameter NUM_MEMORY_CHANNELS = 4;
    parameter MONITOR_WINDOW = 1024;
    parameter TUNING_INTERVAL = 4096;
    parameter SCHEDULER_WINDOW = 256;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Performance monitor signals
    logic [NUM_CORES-1:0] core_active;
    logic [15:0] core_ipc [NUM_CORES-1:0];
    logic [15:0] core_cache_miss_rate [NUM_CORES-1:0];
    logic [15:0] core_branch_miss_rate [NUM_CORES-1:0];
    logic [15:0] core_load [NUM_CORES-1:0];
    
    logic [NUM_AI_UNITS-1:0] ai_unit_active;
    logic [15:0] ai_unit_utilization [NUM_AI_UNITS-1:0];
    logic [15:0] ai_unit_throughput [NUM_AI_UNITS-1:0];
    logic [15:0] ai_unit_efficiency [NUM_AI_UNITS-1:0];
    
    logic [15:0] memory_bandwidth_util;
    logic [15:0] l1_hit_rate, l2_hit_rate, l3_hit_rate;
    logic [15:0] memory_latency;
    logic [15:0] noc_utilization, noc_latency, noc_congestion_level;
    logic [15:0] current_power, temperature;
    logic [2:0] voltage_level, frequency_level;
    logic [7:0] workload_type;
    logic [15:0] workload_intensity, workload_memory_intensity;
    
    // Performance monitor outputs
    logic [2:0] recommended_voltage, recommended_frequency;
    logic [NUM_CORES-1:0] core_power_gate_enable;
    logic [NUM_AI_UNITS-1:0] ai_unit_power_gate_enable;
    logic [3:0] cache_prefetch_aggressiveness;
    logic [3:0] memory_scheduler_policy, noc_routing_policy;
    logic [31:0] overall_performance_score;
    logic [15:0] energy_efficiency_score, thermal_efficiency_score;
    logic [15:0] resource_utilization_score;
    logic performance_degradation_alert, thermal_throttling_needed;
    logic power_budget_exceeded, tuning_active;
    
    // Resource scheduler signals
    logic [7:0] sched_workload_type [NUM_CORES-1:0];
    logic [15:0] sched_workload_priority [NUM_CORES-1:0];
    logic [15:0] sched_workload_deadline [NUM_CORES-1:0];
    logic [15:0] sched_workload_memory_intensity [NUM_CORES-1:0];
    logic [15:0] sched_workload_compute_intensity [NUM_CORES-1:0];
    
    logic [7:0] ai_workload_type [NUM_AI_UNITS-1:0];
    logic [15:0] ai_workload_size [NUM_AI_UNITS-1:0];
    logic [15:0] ai_workload_priority [NUM_AI_UNITS-1:0];
    logic ai_workload_valid [NUM_AI_UNITS-1:0];
    
    logic [NUM_CORES-1:0] core_available;
    logic [NUM_AI_UNITS-1:0] ai_unit_available;
    logic [15:0] memory_bandwidth_available, noc_bandwidth_available;
    logic [15:0] core_performance [NUM_CORES-1:0];
    logic [15:0] ai_unit_performance [NUM_AI_UNITS-1:0];
    logic [15:0] memory_performance, noc_performance;
    logic [15:0] power_budget_remaining, thermal_headroom;
    logic [15:0] core_power_estimate [NUM_CORES-1:0];
    logic [15:0] ai_unit_power_estimate [NUM_AI_UNITS-1:0];
    
    // Resource scheduler outputs
    logic [NUM_CORES-1:0] core_allocation_enable;
    logic [3:0] core_allocation_priority [NUM_CORES-1:0];
    logic [15:0] core_time_slice [NUM_CORES-1:0];
    logic [NUM_AI_UNITS-1:0] ai_unit_allocation_enable;
    logic [3:0] ai_unit_allocation_priority [NUM_AI_UNITS-1:0];
    logic [15:0] ai_unit_time_slice [NUM_AI_UNITS-1:0];
    logic [3:0] sched_memory_allocation_policy;
    logic [15:0] memory_bandwidth_allocation [NUM_MEMORY_CHANNELS-1:0];
    logic [3:0] sched_noc_routing_priority;
    logic [15:0] noc_bandwidth_allocation;
    logic [15:0] qos_violation_count, fairness_index;
    logic [15:0] starvation_prevention_active;
    logic [31:0] total_scheduled_tasks;
    logic [15:0] average_response_time, resource_utilization_efficiency;
    logic scheduler_active;
    
    // Configuration interfaces
    logic [31:0] pm_config_addr, pm_config_wdata, pm_config_rdata;
    logic pm_config_req, pm_config_we, pm_config_ready;
    logic [31:0] rs_config_addr, rs_config_wdata, rs_config_rdata;
    logic rs_config_req, rs_config_we, rs_config_ready;
    
    // Test control
    int test_phase;
    int cycle_count;
    logic test_passed;
    logic [31:0] error_count;
    
    // DUT instantiation
    performance_monitor #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .MONITOR_WINDOW(MONITOR_WINDOW),
        .TUNING_INTERVAL(TUNING_INTERVAL)
    ) dut_perf_monitor (
        .clk(clk),
        .rst_n(rst_n),
        .core_active(core_active),
        .core_ipc(core_ipc),
        .core_cache_miss_rate(core_cache_miss_rate),
        .core_branch_miss_rate(core_branch_miss_rate),
        .core_load(core_load),
        .ai_unit_active(ai_unit_active),
        .ai_unit_utilization(ai_unit_utilization),
        .ai_unit_throughput(ai_unit_throughput),
        .ai_unit_efficiency(ai_unit_efficiency),
        .memory_bandwidth_util(memory_bandwidth_util),
        .l1_hit_rate(l1_hit_rate),
        .l2_hit_rate(l2_hit_rate),
        .l3_hit_rate(l3_hit_rate),
        .memory_latency(memory_latency),
        .noc_utilization(noc_utilization),
        .noc_latency(noc_latency),
        .noc_congestion_level(noc_congestion_level),
        .current_power(current_power),
        .temperature(temperature),
        .voltage_level(voltage_level),
        .frequency_level(frequency_level),
        .workload_type(workload_type),
        .workload_intensity(workload_intensity),
        .workload_memory_intensity(workload_memory_intensity),
        .recommended_voltage(recommended_voltage),
        .recommended_frequency(recommended_frequency),
        .core_power_gate_enable(core_power_gate_enable),
        .ai_unit_power_gate_enable(ai_unit_power_gate_enable),
        .cache_prefetch_aggressiveness(cache_prefetch_aggressiveness),
        .memory_scheduler_policy(memory_scheduler_policy),
        .noc_routing_policy(noc_routing_policy),
        .overall_performance_score(overall_performance_score),
        .energy_efficiency_score(energy_efficiency_score),
        .thermal_efficiency_score(thermal_efficiency_score),
        .resource_utilization_score(resource_utilization_score),
        .performance_degradation_alert(performance_degradation_alert),
        .thermal_throttling_needed(thermal_throttling_needed),
        .power_budget_exceeded(power_budget_exceeded),
        .tuning_active(tuning_active),
        .config_addr(pm_config_addr),
        .config_wdata(pm_config_wdata),
        .config_rdata(pm_config_rdata),
        .config_req(pm_config_req),
        .config_we(pm_config_we),
        .config_ready(pm_config_ready)
    );
    
    resource_scheduler #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS),
        .NUM_MEMORY_CHANNELS(NUM_MEMORY_CHANNELS),
        .SCHEDULER_WINDOW(SCHEDULER_WINDOW)
    ) dut_resource_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .workload_type(sched_workload_type),
        .workload_priority(sched_workload_priority),
        .workload_deadline(sched_workload_deadline),
        .workload_memory_intensity(sched_workload_memory_intensity),
        .workload_compute_intensity(sched_workload_compute_intensity),
        .ai_workload_type(ai_workload_type),
        .ai_workload_size(ai_workload_size),
        .ai_workload_priority(ai_workload_priority),
        .ai_workload_valid(ai_workload_valid),
        .core_available(core_available),
        .core_load(core_load),
        .ai_unit_available(ai_unit_available),
        .ai_unit_load(ai_unit_utilization),
        .memory_bandwidth_available(memory_bandwidth_available),
        .noc_bandwidth_available(noc_bandwidth_available),
        .core_performance(core_performance),
        .ai_unit_performance(ai_unit_performance),
        .memory_performance(memory_performance),
        .noc_performance(noc_performance),
        .power_budget_remaining(power_budget_remaining),
        .thermal_headroom(thermal_headroom),
        .core_power_estimate(core_power_estimate),
        .ai_unit_power_estimate(ai_unit_power_estimate),
        .core_allocation_enable(core_allocation_enable),
        .core_allocation_priority(core_allocation_priority),
        .core_time_slice(core_time_slice),
        .ai_unit_allocation_enable(ai_unit_allocation_enable),
        .ai_unit_allocation_priority(ai_unit_allocation_priority),
        .ai_unit_time_slice(ai_unit_time_slice),
        .memory_allocation_policy(sched_memory_allocation_policy),
        .memory_bandwidth_allocation(memory_bandwidth_allocation),
        .noc_routing_priority(sched_noc_routing_priority),
        .noc_bandwidth_allocation(noc_bandwidth_allocation),
        .qos_violation_count(qos_violation_count),
        .fairness_index(fairness_index),
        .starvation_prevention_active(starvation_prevention_active),
        .total_scheduled_tasks(total_scheduled_tasks),
        .average_response_time(average_response_time),
        .resource_utilization_efficiency(resource_utilization_efficiency),
        .scheduler_active(scheduler_active),
        .config_addr(rs_config_addr),
        .config_wdata(rs_config_wdata),
        .config_rdata(rs_config_rdata),
        .config_req(rs_config_req),
        .config_we(rs_config_we),
        .config_ready(rs_config_ready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        $display("Starting Performance Optimization System Test");
        
        // Initialize signals
        rst_n = 0;
        test_phase = 0;
        cycle_count = 0;
        test_passed = 1;
        error_count = 0;
        
        // Initialize all input signals
        core_active = '0;
        ai_unit_active = '0;
        voltage_level = 3'd4; // Nominal voltage
        frequency_level = 3'd3; // Nominal frequency
        workload_type = 8'd0;
        workload_intensity = 16'h8000;
        workload_memory_intensity = 16'h4000;
        
        // Initialize performance metrics
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h8000; // 1.0 IPC
            core_cache_miss_rate[i] = 16'h2000; // 12.5% miss rate
            core_branch_miss_rate[i] = 16'h1000; // 6.25% miss rate
            core_load[i] = 16'h4000; // 25% load
            core_available[i] = 1'b1;
            core_performance[i] = 16'h8000;
            core_power_estimate[i] = 16'h0800; // 2W
            sched_workload_type[i] = 8'd0;
            sched_workload_priority[i] = 16'h8000;
            sched_workload_deadline[i] = 16'hFFFF;
            sched_workload_memory_intensity[i] = 16'h4000;
            sched_workload_compute_intensity[i] = 16'h6000;
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h2000; // 12.5% utilization
            ai_unit_throughput[i] = 16'h4000;
            ai_unit_efficiency[i] = 16'h6000;
            ai_unit_available[i] = 1'b1;
            ai_unit_performance[i] = 16'h8000;
            ai_unit_power_estimate[i] = 16'h1400; // 5W
            ai_workload_type[i] = 8'd2; // AI inference
            ai_workload_size[i] = 16'h8000;
            ai_workload_priority[i] = 16'h8000;
            ai_workload_valid[i] = 1'b0;
        end
        
        memory_bandwidth_util = 16'h3000; // 18.75% utilization
        l1_hit_rate = 16'hD000; // 81.25% hit rate
        l2_hit_rate = 16'hC000; // 75% hit rate
        l3_hit_rate = 16'hB000; // 68.75% hit rate
        memory_latency = 16'h0400; // Low latency
        noc_utilization = 16'h2000;
        noc_latency = 16'h0200;
        noc_congestion_level = 16'h1000;
        current_power = 16'h1000; // 4W
        temperature = 16'h3000; // 48C
        memory_bandwidth_available = 16'hC000;
        noc_bandwidth_available = 16'hA000;
        memory_performance = 16'h8000;
        noc_performance = 16'h8000;
        power_budget_remaining = 16'h2800; // 10W remaining
        thermal_headroom = 16'h5000; // 20C headroom
        
        // Configuration interface initialization
        pm_config_addr = 32'h0;
        pm_config_wdata = 32'h0;
        pm_config_req = 1'b0;
        pm_config_we = 1'b0;
        rs_config_addr = 32'h0;
        rs_config_wdata = 32'h0;
        rs_config_req = 1'b0;
        rs_config_we = 1'b0;
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
        
        $display("Reset complete, starting test phases");
        
        // Test Phase 1: Basic functionality
        test_phase = 1;
        $display("Phase 1: Testing basic performance monitoring");
        test_basic_performance_monitoring();
        
        // Test Phase 2: Adaptive tuning
        test_phase = 2;
        $display("Phase 2: Testing adaptive performance tuning");
        test_adaptive_tuning();
        
        // Test Phase 3: Workload-aware scheduling
        test_phase = 3;
        $display("Phase 3: Testing workload-aware resource scheduling");
        test_workload_aware_scheduling();
        
        // Test Phase 4: Power optimization
        test_phase = 4;
        $display("Phase 4: Testing power optimization");
        test_power_optimization();
        
        // Test Phase 5: Thermal management
        test_phase = 5;
        $display("Phase 5: Testing thermal management");
        test_thermal_management();
        
        // Test Phase 6: Configuration interface
        test_phase = 6;
        $display("Phase 6: Testing configuration interfaces");
        test_configuration_interface();
        
        // Test Phase 7: Stress testing
        test_phase = 7;
        $display("Phase 7: Stress testing with varying workloads");
        test_stress_scenarios();
        
        // Final results
        if (test_passed && error_count == 0) begin
            $display("✓ All Performance Optimization tests PASSED");
        end else begin
            $display("✗ Performance Optimization tests FAILED with %0d errors", error_count);
        end
        
        $finish;
    end
    
    // Test tasks
    task test_basic_performance_monitoring();
        $display("  Testing basic performance score calculation");
        
        // Set up a balanced workload
        core_active = 4'b1111;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h8000 + (i * 16'h1000); // Varying IPC
            core_load[i] = 16'h6000 + (i * 16'h0800); // Varying load
        end
        
        ai_unit_active = 2'b11;
        ai_unit_utilization[0] = 16'h6000; // 37.5%
        ai_unit_utilization[1] = 16'h4000; // 25%
        
        repeat(100) @(posedge clk);
        
        // Check performance score calculation
        if (overall_performance_score < 32'h40000000) begin
            $display("    ✗ Performance score too low: %h", overall_performance_score);
            error_count++;
        end else begin
            $display("    ✓ Performance score calculated: %h", overall_performance_score);
        end
        
        // Check efficiency scores
        if (energy_efficiency_score == 16'h0) begin
            $display("    ✗ Energy efficiency score not calculated");
            error_count++;
        end else begin
            $display("    ✓ Energy efficiency score: %h", energy_efficiency_score);
        end
        
        if (resource_utilization_score == 16'h0) begin
            $display("    ✗ Resource utilization score not calculated");
            error_count++;
        end else begin
            $display("    ✓ Resource utilization score: %h", resource_utilization_score);
        end
    endtask
    
    task test_adaptive_tuning();
        $display("  Testing adaptive performance tuning");
        
        // Create a low-performance scenario
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h4000; // Low IPC
            core_load[i] = 16'h2000; // Low load
        end
        
        l1_hit_rate = 16'h8000; // 50% hit rate (poor)
        current_power = 16'h0800; // Low power (2W)
        
        // Wait for tuning to activate
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Check if tuning recommendations are generated
        if (!tuning_active) begin
            $display("    ✗ Adaptive tuning not activated");
            error_count++;
        end else begin
            $display("    ✓ Adaptive tuning activated");
        end
        
        // Check frequency/voltage recommendations
        if (recommended_frequency <= frequency_level) begin
            $display("    ✗ Frequency not increased for low performance");
            error_count++;
        end else begin
            $display("    ✓ Frequency increase recommended: %d -> %d", 
                    frequency_level, recommended_frequency);
        end
        
        // Check cache prefetch adjustment
        if (cache_prefetch_aggressiveness <= 4'd2) begin
            $display("    ✗ Cache prefetch not increased for poor hit rate");
            error_count++;
        end else begin
            $display("    ✓ Cache prefetch increased: %d", cache_prefetch_aggressiveness);
        end
    endtask
    
    task test_workload_aware_scheduling();
        $display("  Testing workload-aware resource scheduling");
        
        // Set up different workload types
        sched_workload_type[0] = 8'd0; // CPU intensive
        sched_workload_type[1] = 8'd1; // Memory intensive
        sched_workload_type[2] = 8'd5; // Real-time
        sched_workload_type[3] = 8'd6; // Batch
        
        sched_workload_priority[0] = 16'h8000; // Normal
        sched_workload_priority[1] = 16'h6000; // Lower
        sched_workload_priority[2] = 16'hC000; // High (real-time)
        sched_workload_priority[3] = 16'h4000; // Low (batch)
        
        sched_workload_deadline[2] = 16'h0400; // Urgent deadline
        
        // Enable AI workloads
        ai_workload_valid[0] = 1'b1;
        ai_workload_valid[1] = 1'b1;
        ai_workload_priority[0] = 16'hA000; // High priority
        ai_workload_priority[1] = 16'h6000; // Normal priority
        
        repeat(SCHEDULER_WINDOW + 50) @(posedge clk);
        
        // Check scheduling decisions
        if (!scheduler_active) begin
            $display("    ✗ Resource scheduler not active");
            error_count++;
        end else begin
            $display("    ✓ Resource scheduler active");
        end
        
        // Check real-time task gets highest priority
        if (core_allocation_priority[2] < 4'd12) begin
            $display("    ✗ Real-time task not prioritized: priority=%d", 
                    core_allocation_priority[2]);
            error_count++;
        end else begin
            $display("    ✓ Real-time task prioritized: priority=%d", 
                    core_allocation_priority[2]);
        end
        
        // Check AI unit allocation
        if (!ai_unit_allocation_enable[0] && ai_workload_valid[0]) begin
            $display("    ✗ High-priority AI workload not allocated");
            error_count++;
        end else begin
            $display("    ✓ AI workload allocated with priority %d", 
                    ai_unit_allocation_priority[0]);
        end
        
        // Check fairness index
        repeat(1000) @(posedge clk); // Let fairness metrics accumulate
        if (fairness_index < 16'h8000) begin
            $display("    ⚠ Fairness index low: %h", fairness_index);
        end else begin
            $display("    ✓ Good fairness index: %h", fairness_index);
        end
    endtask
    
    task test_power_optimization();
        $display("  Testing power optimization");
        
        // Create high power consumption scenario
        current_power = 16'h3000; // 12W
        power_budget_remaining = 16'h0800; // Only 2W remaining
        
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h2000; // Low utilization
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h1000; // Very low utilization
        end
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Check power budget alert
        if (!power_budget_exceeded) begin
            $display("    ✗ Power budget exceeded alert not triggered");
            error_count++;
        end else begin
            $display("    ✓ Power budget exceeded alert triggered");
        end
        
        // Check power gating recommendations
        if (core_power_gate_enable == '0) begin
            $display("    ✗ Core power gating not recommended for low utilization");
            error_count++;
        end else begin
            $display("    ✓ Core power gating recommended: %b", core_power_gate_enable);
        end
        
        if (ai_unit_power_gate_enable == '0) begin
            $display("    ✗ AI unit power gating not recommended for low utilization");
            error_count++;
        end else begin
            $display("    ✓ AI unit power gating recommended: %b", ai_unit_power_gate_enable);
        end
        
        // Check frequency/voltage reduction
        if (recommended_frequency >= frequency_level) begin
            $display("    ✗ Frequency not reduced for power constraint");
            error_count++;
        end else begin
            $display("    ✓ Frequency reduction recommended: %d -> %d", 
                    frequency_level, recommended_frequency);
        end
    endtask
    
    task test_thermal_management();
        $display("  Testing thermal management");
        
        // Create high temperature scenario
        temperature = 16'h5400; // 84C (close to limit)
        current_power = 16'h2800; // 10W
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Check thermal throttling alert
        if (!thermal_throttling_needed) begin
            $display("    ✗ Thermal throttling alert not triggered");
            error_count++;
        end else begin
            $display("    ✓ Thermal throttling alert triggered");
        end
        
        // Check thermal efficiency score
        if (thermal_efficiency_score > 16'h8000) begin
            $display("    ✗ Thermal efficiency score should be low at high temperature");
            error_count++;
        end else begin
            $display("    ✓ Thermal efficiency score reflects high temperature: %h", 
                    thermal_efficiency_score);
        end
        
        // Increase temperature further
        temperature = 16'h5800; // 88C (above limit)
        repeat(100) @(posedge clk);
        
        // Should recommend aggressive power reduction
        if (recommended_frequency >= frequency_level && recommended_voltage >= voltage_level) begin
            $display("    ✗ No thermal protection measures recommended");
            error_count++;
        end else begin
            $display("    ✓ Thermal protection: freq=%d, volt=%d", 
                    recommended_frequency, recommended_voltage);
        end
    endtask
    
    task test_configuration_interface();
        $display("  Testing configuration interfaces");
        
        // Test performance monitor configuration
        pm_config_addr = 32'h00;
        pm_config_wdata = 32'h0;
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
            $display("    ✗ Performance monitor configuration write/read failed");
            error_count++;
        end else begin
            $display("    ✓ Performance monitor configuration interface working");
        end
        
        pm_config_req = 1'b0;
        @(posedge clk);
        
        // Test resource scheduler configuration
        rs_config_addr = 32'h04;
        rs_config_wdata = 32'h3; // Set to workload-aware scheduling
        rs_config_req = 1'b1;
        rs_config_we = 1'b1;
        @(posedge clk);
        
        while (!rs_config_ready) @(posedge clk);
        rs_config_req = 1'b0;
        @(posedge clk);
        
        // Read back
        rs_config_addr = 32'h04;
        rs_config_req = 1'b1;
        rs_config_we = 1'b0;
        @(posedge clk);
        
        while (!rs_config_ready) @(posedge clk);
        
        if (rs_config_rdata[2:0] != 3'b011) begin
            $display("    ✗ Resource scheduler configuration write/read failed");
            error_count++;
        end else begin
            $display("    ✓ Resource scheduler configuration interface working");
        end
        
        rs_config_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_stress_scenarios();
        $display("  Testing stress scenarios with varying workloads");
        
        // Scenario 1: Rapidly changing workload
        for (int cycle = 0; cycle < 10; cycle++) begin
            // Alternate between CPU-intensive and AI-intensive
            if (cycle % 2 == 0) begin
                workload_type = 8'd0; // CPU intensive
                for (int i = 0; i < NUM_CORES; i++) begin
                    core_load[i] = 16'hC000; // High load
                    core_ipc[i] = 16'hA000; // High IPC
                end
                for (int i = 0; i < NUM_AI_UNITS; i++) begin
                    ai_unit_utilization[i] = 16'h2000; // Low utilization
                end
            end else begin
                workload_type = 8'd1; // AI intensive
                for (int i = 0; i < NUM_CORES; i++) begin
                    core_load[i] = 16'h4000; // Medium load
                    core_ipc[i] = 16'h6000; // Medium IPC
                end
                for (int i = 0; i < NUM_AI_UNITS; i++) begin
                    ai_unit_utilization[i] = 16'hC000; // High utilization
                end
            end
            
            repeat(500) @(posedge clk);
        end
        
        // Check system stability
        if (overall_performance_score == 32'h0) begin
            $display("    ✗ System became unstable during stress test");
            error_count++;
        end else begin
            $display("    ✓ System remained stable during workload changes");
        end
        
        // Scenario 2: Resource contention
        core_active = 4'b1111;
        ai_unit_active = 2'b11;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'hF000; // Very high load
            sched_workload_priority[i] = 16'hF000; // All high priority
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'hF000; // Very high utilization
            ai_workload_valid[i] = 1'b1;
            ai_workload_priority[i] = 16'hF000; // All high priority
        end
        
        memory_bandwidth_util = 16'hE000; // High memory usage
        noc_utilization = 16'hD000; // High NoC usage
        
        repeat(1000) @(posedge clk);
        
        // Check QoS violations
        if (qos_violation_count > 16'h1000) begin
            $display("    ⚠ High QoS violations during contention: %h", qos_violation_count);
        end else begin
            $display("    ✓ QoS maintained during resource contention");
        end
        
        // Check fairness
        if (fairness_index < 16'h4000) begin
            $display("    ⚠ Low fairness during contention: %h", fairness_index);
        end else begin
            $display("    ✓ Good fairness maintained: %h", fairness_index);
        end
    endtask
    
    // Monitor cycle count
    always @(posedge clk) begin
        if (rst_n) begin
            cycle_count <= cycle_count + 1;
        end
    end
    
    // Timeout watchdog
    initial begin
        #50000000; // 50ms timeout
        $display("✗ Test timeout - possible deadlock or infinite loop");
        test_passed = 0;
        $finish;
    end

endmodule