/*
 * Advanced Performance Optimization Verification Test
 * 
 * Comprehensive test suite for enhanced performance monitoring, adaptive tuning,
 * machine learning-inspired optimization, and workload-aware resource scheduling.
 */

`timescale 1ns / 1ps

module test_advanced_performance_optimization;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_AI_UNITS = 2;
    parameter NUM_MEMORY_CHANNELS = 4;
    parameter MONITOR_WINDOW = 1024;
    parameter TUNING_INTERVAL = 4096;
    parameter SCHEDULER_WINDOW = 256;
    parameter TEST_DURATION = 100000; // cycles
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Enhanced performance monitor signals
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
    
    // Configuration interfaces
    logic [31:0] pm_config_addr, pm_config_wdata, pm_config_rdata;
    logic pm_config_req, pm_config_we, pm_config_ready;
    
    // Test control and monitoring
    int test_phase;
    int cycle_count;
    logic test_passed;
    logic [31:0] error_count;
    
    // Workload simulation state
    logic [31:0] workload_cycle_counter;
    logic [7:0] current_workload_phase;
    logic [15:0] workload_transition_timer;
    
    // Performance tracking
    logic [31:0] performance_history [16];
    logic [3:0] history_index;
    logic [31:0] performance_trend_sum;
    logic [15:0] optimization_trigger_count;
    logic [15:0] successful_optimizations;
    
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
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Main test sequence
    initial begin
        $display("Starting Advanced Performance Optimization Test Suite");
        
        // Initialize signals
        rst_n = 0;
        test_phase = 0;
        cycle_count = 0;
        test_passed = 1;
        error_count = 0;
        workload_cycle_counter = 0;
        current_workload_phase = 0;
        workload_transition_timer = 0;
        history_index = 0;
        performance_trend_sum = 0;
        optimization_trigger_count = 0;
        successful_optimizations = 0;
        
        initialize_test_signals();
        
        // Reset sequence
        repeat(20) @(posedge clk);
        rst_n = 1;
        repeat(20) @(posedge clk);
        
        $display("Reset complete, starting advanced test phases");
        
        // Test Phase 1: Enhanced performance trend analysis
        test_phase = 1;
        $display("Phase 1: Testing enhanced performance trend analysis");
        test_performance_trend_analysis();
        
        // Test Phase 2: Workload-aware adaptive tuning
        test_phase = 2;
        $display("Phase 2: Testing workload-aware adaptive tuning");
        test_workload_aware_tuning();
        
        // Test Phase 3: Multi-objective optimization
        test_phase = 3;
        $display("Phase 3: Testing multi-objective optimization");
        test_multi_objective_optimization();
        
        // Test Phase 4: Thermal-aware optimization
        test_phase = 4;
        $display("Phase 4: Testing thermal-aware optimization");
        test_thermal_aware_optimization();
        
        // Test Phase 5: Dynamic workload adaptation
        test_phase = 5;
        $display("Phase 5: Testing dynamic workload adaptation");
        test_dynamic_workload_adaptation();
        
        // Test Phase 6: Machine learning-inspired optimization
        test_phase = 6;
        $display("Phase 6: Testing ML-inspired optimization patterns");
        test_ml_inspired_optimization();
        
        // Test Phase 7: Long-term stability and convergence
        test_phase = 7;
        $display("Phase 7: Testing long-term stability and convergence");
        test_long_term_stability();
        
        // Test Phase 8: Extreme scenario handling
        test_phase = 8;
        $display("Phase 8: Testing extreme scenario handling");
        test_extreme_scenarios();
        
        // Final results
        if (test_passed && error_count == 0) begin
            $display("✓ All Advanced Performance Optimization tests PASSED");
            $display("  Optimization triggers: %d", optimization_trigger_count);
            $display("  Successful optimizations: %d", successful_optimizations);
            if (optimization_trigger_count > 0) begin
                $display("  Success rate: %.1f%%", 
                        (real'(successful_optimizations) / real'(optimization_trigger_count)) * 100.0);
            end
        end else begin
            $display("✗ Advanced Performance Optimization tests FAILED with %0d errors", error_count);
        end
        
        $finish;
    end
    
    // Initialize test signals
    task initialize_test_signals();
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
        end
        
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h2000; // 12.5% utilization
            ai_unit_throughput[i] = 16'h4000;
            ai_unit_efficiency[i] = 16'h6000;
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
        
        // Configuration interface initialization
        pm_config_addr = 32'h0;
        pm_config_wdata = 32'h0;
        pm_config_req = 1'b0;
        pm_config_we = 1'b0;
        
        performance_history = '{default: 32'h0};
    endtask
    
    // Test enhanced performance trend analysis
    task test_performance_trend_analysis();
        $display("  Testing performance trend tracking and prediction");
        
        // Create a performance trend over time
        for (int cycle = 0; cycle < 20; cycle++) begin
            // Simulate declining performance
            for (int i = 0; i < NUM_CORES; i++) begin
                core_ipc[i] = 16'h8000 - (cycle * 16'h0200); // Gradually decrease IPC
                core_load[i] = 16'h4000 + (cycle * 16'h0100); // Gradually increase load
            end
            
            l1_hit_rate = 16'hD000 - (cycle * 16'h0100); // Decrease hit rate
            current_power = 16'h1000 + (cycle * 16'h0080); // Increase power
            
            repeat(TUNING_INTERVAL / 20) @(posedge clk);
            
            // Track performance history
            performance_history[history_index] = overall_performance_score;
            history_index = (history_index + 1) % 16;
        end
        
        // Wait for trend analysis to complete
        repeat(TUNING_INTERVAL) @(posedge clk);
        
        // Check if adaptive tuning detected the trend
        if (!tuning_active) begin
            $display("    ✗ Trend-based tuning not activated for declining performance");
            error_count++;
        end else begin
            $display("    ✓ Trend-based adaptive tuning activated");
        end
        
        // Check if recommendations are appropriate for declining performance
        if (recommended_frequency <= frequency_level) begin
            $display("    ✗ Frequency not increased for declining performance trend");
            error_count++;
        end else begin
            $display("    ✓ Frequency increase recommended for performance trend: %d -> %d", 
                    frequency_level, recommended_frequency);
        end
        
        if (cache_prefetch_aggressiveness <= 4'd2) begin
            $display("    ✗ Cache prefetch not increased for declining cache performance");
            error_count++;
        end else begin
            $display("    ✓ Cache prefetch increased for performance trend: %d", 
                    cache_prefetch_aggressiveness);
        end
    endtask
    
    // Test workload-aware adaptive tuning
    task test_workload_aware_tuning();
        $display("  Testing workload-specific optimization strategies");
        
        // Test CPU-intensive workload optimization
        workload_type = 8'd0; // CPU intensive
        core_active = 4'b1111;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h6000; // Moderate IPC
            core_load[i] = 16'hC000; // High load
        end
        ai_unit_active = 2'b00; // No AI activity
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        if (memory_scheduler_policy != 4'd2) begin
            $display("    ✗ Memory scheduler not optimized for CPU workload: policy=%d", 
                    memory_scheduler_policy);
            error_count++;
        end else begin
            $display("    ✓ Memory scheduler optimized for CPU workload");
        end
        
        // Test AI-intensive workload optimization
        workload_type = 8'd1; // AI intensive
        ai_unit_active = 2'b11;
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'hA000; // High utilization
            ai_unit_efficiency[i] = 16'h8000; // Good efficiency
        end
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h4000; // Lower CPU load
        end
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        if (memory_scheduler_policy != 4'd3) begin
            $display("    ✗ Memory scheduler not optimized for AI workload: policy=%d", 
                    memory_scheduler_policy);
            error_count++;
        end else begin
            $display("    ✓ Memory scheduler optimized for AI workload");
        end
        
        if (noc_routing_policy != 4'd2) begin
            $display("    ✗ NoC routing not optimized for AI workload: policy=%d", 
                    noc_routing_policy);
            error_count++;
        end else begin
            $display("    ✓ NoC routing optimized for AI workload");
        end
        
        // Test mixed workload optimization
        workload_type = 8'd2; // Mixed
        core_active = 4'b1111;
        ai_unit_active = 2'b11;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h8000; // Moderate load
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h6000; // Moderate utilization
        end
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        if (memory_scheduler_policy != 4'd1) begin
            $display("    ✗ Memory scheduler not balanced for mixed workload: policy=%d", 
                    memory_scheduler_policy);
            error_count++;
        end else begin
            $display("    ✓ Memory scheduler balanced for mixed workload");
        end
    endtask
    
    // Test multi-objective optimization
    task test_multi_objective_optimization();
        $display("  Testing multi-objective optimization (performance vs power vs thermal)");
        
        // Create a scenario requiring trade-offs
        current_power = 16'h2800; // High power (10W)
        temperature = 16'h4800; // High temperature (72C)
        
        // Low performance scenario
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h4000; // Low IPC
            core_load[i] = 16'h3000; // Low load
        end
        
        overall_performance_score = 32'h40000000; // Low performance score
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Should balance performance improvement with power/thermal constraints
        logic performance_optimization = (recommended_frequency > frequency_level) || 
                                       (cache_prefetch_aggressiveness > 4'd2);
        logic power_optimization = (core_power_gate_enable != '0) || 
                                 (recommended_voltage < voltage_level);
        logic thermal_optimization = (recommended_frequency < frequency_level) || 
                                   (core_power_gate_enable != '0);
        
        if (!performance_optimization && !power_optimization && !thermal_optimization) begin
            $display("    ✗ No multi-objective optimization detected");
            error_count++;
        end else begin
            $display("    ✓ Multi-objective optimization active:");
            $display("      Performance optimization: %s", performance_optimization ? "Yes" : "No");
            $display("      Power optimization: %s", power_optimization ? "Yes" : "No");
            $display("      Thermal optimization: %s", thermal_optimization ? "Yes" : "No");
        end
        
        // Check for appropriate trade-offs
        if (performance_optimization && power_optimization) begin
            $display("    ✓ Balanced performance and power optimization");
        end else if (power_optimization && thermal_optimization) begin
            $display("    ✓ Balanced power and thermal optimization");
        end else begin
            $display("    ⚠ Optimization may not be well-balanced");
        end
    endtask
    
    // Test thermal-aware optimization
    task test_thermal_aware_optimization();
        $display("  Testing thermal-aware optimization and protection");
        
        // Create thermal stress scenario
        temperature = 16'h5400; // 84C (near thermal limit)
        current_power = 16'h3000; // 12W
        
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'hE000; // Very high load
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'hC000; // High utilization
        end
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Check thermal protection measures
        if (!thermal_throttling_needed) begin
            $display("    ✗ Thermal throttling not triggered at high temperature");
            error_count++;
        end else begin
            $display("    ✓ Thermal throttling alert triggered");
        end
        
        logic thermal_protection_active = (recommended_frequency < frequency_level) || 
                                        (recommended_voltage < voltage_level) ||
                                        (core_power_gate_enable != '0) ||
                                        (ai_unit_power_gate_enable != '0);
        
        if (!thermal_protection_active) begin
            $display("    ✗ No thermal protection measures activated");
            error_count++;
        end else begin
            $display("    ✓ Thermal protection measures activated");
        end
        
        // Test extreme thermal scenario
        temperature = 16'h5800; // 88C (above safe limit)
        repeat(100) @(posedge clk);
        
        // Should trigger aggressive thermal protection
        if (recommended_frequency >= frequency_level && recommended_voltage >= voltage_level) begin
            $display("    ✗ Insufficient thermal protection at extreme temperature");
            error_count++;
        end else begin
            $display("    ✓ Aggressive thermal protection at extreme temperature");
        end
        
        // Check thermal efficiency score
        if (thermal_efficiency_score > 16'h8000) begin
            $display("    ✗ Thermal efficiency score should be low at high temperature");
            error_count++;
        end else begin
            $display("    ✓ Thermal efficiency score reflects high temperature: %h", 
                    thermal_efficiency_score);
        end
    endtask
    
    // Test dynamic workload adaptation
    task test_dynamic_workload_adaptation();
        $display("  Testing dynamic workload adaptation and transitions");
        
        // Simulate rapidly changing workloads
        for (int phase = 0; phase < 8; phase++) begin
            case (phase % 4)
                0: begin // CPU-intensive phase
                    workload_type = 8'd0;
                    core_active = 4'b1111;
                    ai_unit_active = 2'b00;
                    for (int i = 0; i < NUM_CORES; i++) begin
                        core_ipc[i] = 16'hA000;
                        core_load[i] = 16'hD000;
                    end
                    for (int i = 0; i < NUM_AI_UNITS; i++) begin
                        ai_unit_utilization[i] = 16'h1000;
                    end
                end
                1: begin // AI-intensive phase
                    workload_type = 8'd1;
                    core_active = 4'b1100;
                    ai_unit_active = 2'b11;
                    for (int i = 0; i < NUM_CORES; i++) begin
                        core_ipc[i] = 16'h6000;
                        core_load[i] = 16'h5000;
                    end
                    for (int i = 0; i < NUM_AI_UNITS; i++) begin
                        ai_unit_utilization[i] = 16'hB000;
                    end
                end
                2: begin // Memory-intensive phase
                    workload_type = 8'd4; // Mixed with memory focus
                    memory_bandwidth_util = 16'hC000;
                    l1_hit_rate = 16'h6000; // Poor cache performance
                    l2_hit_rate = 16'h5000;
                    memory_latency = 16'h0800; // High latency
                end
                3: begin // Low activity phase
                    workload_type = 8'd6; // Batch
                    core_active = 4'b0011;
                    ai_unit_active = 2'b01;
                    for (int i = 0; i < NUM_CORES; i++) begin
                        core_load[i] = 16'h3000;
                    end
                    for (int i = 0; i < NUM_AI_UNITS; i++) begin
                        ai_unit_utilization[i] = 16'h2000;
                    end
                    current_power = 16'h0800; // Low power
                end
            endcase
            
            repeat(SCHEDULER_WINDOW * 2) @(posedge clk);
            
            // Track optimization responses
            if (tuning_active) begin
                optimization_trigger_count++;
                
                // Check if optimization is appropriate for current workload
                case (phase % 4)
                    0: begin // CPU-intensive
                        if (memory_scheduler_policy == 4'd2) successful_optimizations++;
                    end
                    1: begin // AI-intensive
                        if (memory_scheduler_policy == 4'd3 || noc_routing_policy == 4'd2) 
                            successful_optimizations++;
                    end
                    2: begin // Memory-intensive
                        if (cache_prefetch_aggressiveness > 4'd2) successful_optimizations++;
                    end
                    3: begin // Low activity
                        if (core_power_gate_enable != '0 || ai_unit_power_gate_enable != '0) 
                            successful_optimizations++;
                    end
                endcase
            end
        end
        
        // Check adaptation effectiveness
        if (optimization_trigger_count == 0) begin
            $display("    ✗ No optimizations triggered during workload transitions");
            error_count++;
        end else begin
            real success_rate = (real'(successful_optimizations) / real'(optimization_trigger_count)) * 100.0;
            $display("    ✓ Workload adaptation: %d/%d optimizations successful (%.1f%%)", 
                    successful_optimizations, optimization_trigger_count, success_rate);
            
            if (success_rate < 50.0) begin
                $display("    ⚠ Low adaptation success rate");
            end
        end
    endtask
    
    // Test ML-inspired optimization patterns
    task test_ml_inspired_optimization();
        $display("  Testing machine learning-inspired optimization patterns");
        
        // Test exploration vs exploitation behavior
        logic [15:0] prev_recommendations [8];
        logic [3:0] recommendation_changes = 0;
        
        // Record initial recommendations
        prev_recommendations[0] = {13'b0, recommended_frequency};
        prev_recommendations[1] = {13'b0, recommended_voltage};
        prev_recommendations[2] = {12'b0, cache_prefetch_aggressiveness};
        prev_recommendations[3] = {12'b0, memory_scheduler_policy};
        
        // Create stable workload for learning
        workload_type = 8'd0; // CPU intensive
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h7000;
            core_load[i] = 16'h8000;
        end
        
        // Run multiple optimization cycles
        for (int cycle = 0; cycle < 5; cycle++) begin
            repeat(TUNING_INTERVAL + 100) @(posedge clk);
            
            // Check for recommendation changes (learning behavior)
            if (prev_recommendations[0] != {13'b0, recommended_frequency}) begin
                recommendation_changes++;
                prev_recommendations[0] = {13'b0, recommended_frequency};
            end
            if (prev_recommendations[1] != {13'b0, recommended_voltage}) begin
                recommendation_changes++;
                prev_recommendations[1] = {13'b0, recommended_voltage};
            end
            if (prev_recommendations[2] != {12'b0, cache_prefetch_aggressiveness}) begin
                recommendation_changes++;
                prev_recommendations[2] = {12'b0, cache_prefetch_aggressiveness};
            end
            if (prev_recommendations[3] != {12'b0, memory_scheduler_policy}) begin
                recommendation_changes++;
                prev_recommendations[3] = {12'b0, memory_scheduler_policy};
            end
        end
        
        if (recommendation_changes == 0) begin
            $display("    ⚠ No learning behavior observed (no recommendation changes)");
        end else if (recommendation_changes > 15) begin
            $display("    ⚠ Excessive exploration (too many recommendation changes: %d)", 
                    recommendation_changes);
        end else begin
            $display("    ✓ Balanced exploration/exploitation behavior (%d changes)", 
                    recommendation_changes);
        end
        
        // Test convergence behavior
        logic [31:0] performance_samples [4];
        for (int i = 0; i < 4; i++) begin
            repeat(TUNING_INTERVAL) @(posedge clk);
            performance_samples[i] = overall_performance_score;
        end
        
        // Check for performance convergence
        logic converging = 1;
        for (int i = 1; i < 4; i++) begin
            if (performance_samples[i] < performance_samples[i-1] - 32'h02000000) begin
                converging = 0; // Significant performance drop
            end
        end
        
        if (converging) begin
            $display("    ✓ Performance converging or stable");
        end else begin
            $display("    ⚠ Performance not converging");
        end
    endtask
    
    // Test long-term stability and convergence
    task test_long_term_stability();
        $display("  Testing long-term stability and convergence");
        
        // Set up stable workload
        workload_type = 8'd2; // Mixed workload
        core_active = 4'b1111;
        ai_unit_active = 2'b11;
        
        for (int i = 0; i < NUM_CORES; i++) begin
            core_ipc[i] = 16'h8000;
            core_load[i] = 16'h6000;
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h5000;
        end
        
        current_power = 16'h1800; // 6W
        temperature = 16'h3800; // 56C
        
        // Run for extended period
        logic [31:0] stability_samples [10];
        logic [15:0] optimization_count = 0;
        
        for (int sample = 0; sample < 10; sample++) begin
            repeat(TUNING_INTERVAL * 2) @(posedge clk);
            stability_samples[sample] = overall_performance_score;
            
            if (tuning_active) optimization_count++;
        end
        
        // Analyze stability
        logic [31:0] min_perf = stability_samples[0];
        logic [31:0] max_perf = stability_samples[0];
        logic [31:0] avg_perf = 0;
        
        for (int i = 0; i < 10; i++) begin
            if (stability_samples[i] < min_perf) min_perf = stability_samples[i];
            if (stability_samples[i] > max_perf) max_perf = stability_samples[i];
            avg_perf += stability_samples[i] / 10;
        end
        
        logic [31:0] perf_variance = max_perf - min_perf;
        real variance_percent = (real'(perf_variance) / real'(avg_perf)) * 100.0;
        
        if (variance_percent > 20.0) begin
            $display("    ⚠ High performance variance: %.1f%%", variance_percent);
        end else begin
            $display("    ✓ Stable performance variance: %.1f%%", variance_percent);
        end
        
        // Check optimization frequency
        if (optimization_count > 8) begin
            $display("    ⚠ High optimization frequency: %d/10 intervals", optimization_count);
        end else if (optimization_count == 0) begin
            $display("    ⚠ No optimizations in stable workload");
        end else begin
            $display("    ✓ Reasonable optimization frequency: %d/10 intervals", optimization_count);
        end
        
        // Check for oscillation
        logic oscillating = 0;
        for (int i = 2; i < 10; i++) begin
            if ((stability_samples[i] > stability_samples[i-1]) && 
                (stability_samples[i-1] < stability_samples[i-2])) begin
                oscillating = 1;
            end
        end
        
        if (oscillating) begin
            $display("    ⚠ Performance oscillation detected");
        end else begin
            $display("    ✓ No performance oscillation");
        end
    endtask
    
    // Test extreme scenario handling
    task test_extreme_scenarios();
        $display("  Testing extreme scenario handling and robustness");
        
        // Scenario 1: Thermal emergency
        $display("    Testing thermal emergency scenario");
        temperature = 16'h6000; // 96C (critical)
        current_power = 16'h4000; // 16W (very high)
        
        repeat(100) @(posedge clk);
        
        if (!thermal_throttling_needed) begin
            $display("      ✗ Thermal emergency not detected");
            error_count++;
        end else begin
            $display("      ✓ Thermal emergency detected");
        end
        
        // Should trigger aggressive protection
        if (recommended_frequency >= frequency_level || core_power_gate_enable == '0) begin
            $display("      ✗ Insufficient thermal emergency response");
            error_count++;
        end else begin
            $display("      ✓ Aggressive thermal protection activated");
        end
        
        // Scenario 2: Power budget violation
        $display("    Testing power budget violation scenario");
        current_power = 16'h5000; // 20W (exceeds typical budget)
        temperature = 16'h4000; // Normal temperature
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        if (!power_budget_exceeded) begin
            $display("      ✗ Power budget violation not detected");
            error_count++;
        end else begin
            $display("      ✓ Power budget violation detected");
        end
        
        // Scenario 3: Complete system idle
        $display("    Testing complete system idle scenario");
        core_active = 4'b0000;
        ai_unit_active = 2'b00;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'h0000;
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'h0000;
        end
        current_power = 16'h0400; // 1W (very low)
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // Should enable aggressive power gating
        if (core_power_gate_enable == '0 && ai_unit_power_gate_enable == '0) begin
            $display("      ⚠ No power gating in idle scenario");
        end else begin
            $display("      ✓ Power gating enabled in idle scenario");
        end
        
        // Scenario 4: Maximum utilization
        $display("    Testing maximum utilization scenario");
        core_active = 4'b1111;
        ai_unit_active = 2'b11;
        for (int i = 0; i < NUM_CORES; i++) begin
            core_load[i] = 16'hFFFF;
            core_ipc[i] = 16'hC000;
        end
        for (int i = 0; i < NUM_AI_UNITS; i++) begin
            ai_unit_utilization[i] = 16'hFFFF;
        end
        memory_bandwidth_util = 16'hF000;
        noc_utilization = 16'hE000;
        
        repeat(TUNING_INTERVAL + 100) @(posedge clk);
        
        // System should remain stable under maximum load
        if (overall_performance_score == 32'h0) begin
            $display("      ✗ System unstable under maximum load");
            error_count++;
        end else begin
            $display("      ✓ System stable under maximum load");
        end
    endtask
    
    // Monitor cycle count and performance tracking
    always @(posedge clk) begin
        if (rst_n) begin
            cycle_count <= cycle_count + 1;
            workload_cycle_counter <= workload_cycle_counter + 1;
            
            // Track performance history for analysis
            if (cycle_count % 1000 == 0) begin
                performance_trend_sum <= performance_trend_sum + overall_performance_score;
            end
        end
    end
    
    // Timeout watchdog
    initial begin
        #100000000; // 100ms timeout
        $display("✗ Test timeout - possible deadlock or infinite loop");
        test_passed = 0;
        $finish;
    end

endmodule