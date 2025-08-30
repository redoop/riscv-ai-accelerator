// Integration Analyzer
// Analyzes system integration test results and generates insights

`ifndef INTEGRATION_ANALYZER_SV
`define INTEGRATION_ANALYZER_SV

class integration_analyzer extends uvm_component;
    
    // Analysis configuration
    bit enable_performance_analysis = 1;
    bit enable_scalability_analysis = 1;
    bit enable_reliability_analysis = 1;
    bit generate_detailed_report = 1;
    
    // Analysis results
    real overall_score = 0.0;
    real performance_score = 0.0;
    real scalability_score = 0.0;
    real reliability_score = 0.0;
    real efficiency_score = 0.0;
    
    `uvm_component_utils_begin(integration_analyzer)
        `uvm_field_int(enable_performance_analysis, UVM_ALL_ON)
        `uvm_field_int(enable_scalability_analysis, UVM_ALL_ON)
        `uvm_field_int(enable_reliability_analysis, UVM_ALL_ON)
        `uvm_field_real(overall_score, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "integration_analyzer", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Analyze integration test results
    virtual function void analyze_integration_results(system_state_t state_history[], 
                                                     integration_performance_t performance,
                                                     integration_test_config_t config);
        `uvm_info(get_type_name(), "Analyzing integration test results", UVM_MEDIUM)
        
        if (enable_performance_analysis) begin
            performance_score = analyze_performance(state_history, performance);
        end
        
        if (enable_scalability_analysis) begin
            scalability_score = analyze_scalability(state_history, config);
        end
        
        if (enable_reliability_analysis) begin
            reliability_score = analyze_reliability(state_history, performance);
        end
        
        efficiency_score = analyze_efficiency(state_history, performance);
        
        // Calculate overall score
        overall_score = (performance_score * 0.3 + scalability_score * 0.25 + 
                        reliability_score * 0.25 + efficiency_score * 0.2);
        
        if (generate_detailed_report) begin
            generate_analysis_report(state_history, performance, config);
        end
    endfunction
    
    // Analyze performance metrics
    virtual function real analyze_performance(system_state_t state_history[], 
                                            integration_performance_t performance);
        real score = 0.0;
        
        // Throughput score (0-100)
        if (performance.overall_throughput > 0) begin
            score += $min(performance.overall_throughput / 1000.0 * 25.0, 25.0);
        end
        
        // Latency score (0-100, lower latency = higher score)
        if (performance.latency_p95 > 0) begin
            real latency_score = $max(0.0, 25.0 - performance.latency_p95 / 10.0);
            score += latency_score;
        end
        
        // Utilization score
        if (state_history.size() > 0) begin
            real avg_utilization = calculate_average_utilization(state_history);
            score += avg_utilization * 25.0;
        end
        
        // Cache efficiency score
        real cache_efficiency = calculate_cache_efficiency(state_history);
        score += cache_efficiency * 25.0;
        
        return $min(score, 100.0);
    endfunction
    
    // Analyze scalability
    virtual function real analyze_scalability(system_state_t state_history[], 
                                            integration_test_config_t config);
        real score = 50.0;  // Base score
        
        // Multi-core scaling efficiency
        if (config.num_cores_active > 1) begin
            real scaling_efficiency = calculate_scaling_efficiency(state_history, config);
            score += scaling_efficiency * 30.0;
        end
        
        // Load balancing effectiveness
        real load_balance = calculate_load_balance(state_history, config);
        score += load_balance * 20.0;
        
        return $min(score, 100.0);
    endfunction
    
    // Analyze reliability
    virtual function real analyze_reliability(system_state_t state_history[], 
                                            integration_performance_t performance);
        real score = 100.0;  // Start with perfect score
        
        // Error rate impact
        if (performance.total_operations > 0) begin
            real error_rate = 0.0;
            if (state_history.size() > 0) begin
                error_rate = real'(state_history[state_history.size()-1].error_count) / 
                           real'(performance.total_operations);
            end
            score -= error_rate * 1000.0;  // Heavy penalty for errors
        end
        
        // Thermal stability
        real thermal_stability = calculate_thermal_stability(state_history);
        score = score * thermal_stability;
        
        return $max(score, 0.0);
    endfunction
    
    // Analyze efficiency
    virtual function real analyze_efficiency(system_state_t state_history[], 
                                           integration_performance_t performance);
        real score = 0.0;
        
        // Power efficiency
        if (performance.power_efficiency > 0) begin
            score += $min(performance.power_efficiency * 10.0, 40.0);
        end
        
        // Thermal efficiency
        if (performance.thermal_efficiency > 0) begin
            score += $min(performance.thermal_efficiency * 5.0, 30.0);
        end
        
        // Memory efficiency
        real memory_efficiency = calculate_memory_efficiency(state_history);
        score += memory_efficiency * 30.0;
        
        return $min(score, 100.0);
    endfunction
    
    // Calculate average system utilization
    virtual function real calculate_average_utilization(system_state_t state_history[]);
        real total_utilization = 0.0;
        int valid_samples = 0;
        
        foreach (state_history[i]) begin
            real sample_utilization = 0.0;
            for (int core = 0; core < 4; core++) begin
                sample_utilization += state_history[i].cpu_utilization[core];
            end
            for (int tpu = 0; tpu < 2; tpu++) begin
                sample_utilization += state_history[i].tpu_utilization[tpu];
            end
            for (int vpu = 0; vpu < 2; vpu++) begin
                sample_utilization += state_history[i].vpu_utilization[vpu];
            end
            
            total_utilization += sample_utilization / 8.0;  // Average across all units
            valid_samples++;
        end
        
        return (valid_samples > 0) ? total_utilization / real'(valid_samples) : 0.0;
    endfunction
    
    // Calculate cache efficiency
    virtual function real calculate_cache_efficiency(system_state_t state_history[]);
        real total_hit_rate = 0.0;
        int valid_samples = 0;
        
        foreach (state_history[i]) begin
            real sample_hit_rate = 0.0;
            for (int cache = 0; cache < 4; cache++) begin
                sample_hit_rate += state_history[i].cache_hit_rates[cache];
            end
            total_hit_rate += sample_hit_rate / 4.0;
            valid_samples++;
        end
        
        return (valid_samples > 0) ? total_hit_rate / real'(valid_samples) : 0.0;
    endfunction
    
    // Calculate scaling efficiency
    virtual function real calculate_scaling_efficiency(system_state_t state_history[], 
                                                     integration_test_config_t config);
        // Simplified scaling efficiency calculation
        real active_cores = real'(config.num_cores_active);
        real ideal_scaling = 1.0 / active_cores;
        
        // Calculate actual per-core efficiency
        real avg_utilization = calculate_average_utilization(state_history);
        real actual_efficiency = avg_utilization / active_cores;
        
        return (ideal_scaling > 0) ? actual_efficiency / ideal_scaling : 0.0;
    endfunction
    
    // Calculate load balance
    virtual function real calculate_load_balance(system_state_t state_history[], 
                                               integration_test_config_t config);
        if (state_history.size() == 0) return 0.0;
        
        // Use last state for load balance calculation
        system_state_t last_state = state_history[state_history.size()-1];
        
        real max_util = 0.0, min_util = 1.0;
        for (int core = 0; core < config.num_cores_active; core++) begin
            if (last_state.cpu_utilization[core] > max_util) max_util = last_state.cpu_utilization[core];
            if (last_state.cpu_utilization[core] < min_util) min_util = last_state.cpu_utilization[core];
        end
        
        real imbalance = max_util - min_util;
        return $max(0.0, 1.0 - imbalance);
    endfunction
    
    // Calculate thermal stability
    virtual function real calculate_thermal_stability(system_state_t state_history[]);
        if (state_history.size() < 2) return 1.0;
        
        real max_temp_variation = 0.0;
        for (int i = 1; i < state_history.size(); i++) begin
            real temp_change = $abs(state_history[i].temperature_celsius - 
                                  state_history[i-1].temperature_celsius);
            if (temp_change > max_temp_variation) max_temp_variation = temp_change;
        end
        
        // Penalize high temperature variations
        return $max(0.0, 1.0 - max_temp_variation / 20.0);
    endfunction
    
    // Calculate memory efficiency
    virtual function real calculate_memory_efficiency(system_state_t state_history[]);
        real total_bandwidth_util = 0.0;
        int valid_samples = 0;
        
        foreach (state_history[i]) begin
            total_bandwidth_util += state_history[i].memory_bandwidth_utilization;
            valid_samples++;
        end
        
        return (valid_samples > 0) ? total_bandwidth_util / real'(valid_samples) : 0.0;
    endfunction
    
    // Generate detailed analysis report
    virtual function void generate_analysis_report(system_state_t state_history[], 
                                                  integration_performance_t performance,
                                                  integration_test_config_t config);
        string filename = "integration_analysis_report.txt";
        int file_handle = $fopen(filename, "w");
        
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot create analysis report: %s", filename));
            return;
        end
        
        // Report header
        $fwrite(file_handle, "System Integration Analysis Report\n");
        $fwrite(file_handle, "===================================\n\n");
        $fwrite(file_handle, "Test Configuration:\n");
        $fwrite(file_handle, "  Test Type: %s\n", config.test_type.name());
        $fwrite(file_handle, "  Complexity: %s\n", config.complexity.name());
        $fwrite(file_handle, "  Active Cores: %0d\n", config.num_cores_active);
        $fwrite(file_handle, "  Active TPUs: %0d\n", config.num_tpus_active);
        $fwrite(file_handle, "  Active VPUs: %0d\n", config.num_vpus_active);
        $fwrite(file_handle, "\n");
        
        // Analysis scores
        $fwrite(file_handle, "Analysis Scores:\n");
        $fwrite(file_handle, "  Overall Score: %.2f/100\n", overall_score);
        $fwrite(file_handle, "  Performance Score: %.2f/100\n", performance_score);
        $fwrite(file_handle, "  Scalability Score: %.2f/100\n", scalability_score);
        $fwrite(file_handle, "  Reliability Score: %.2f/100\n", reliability_score);
        $fwrite(file_handle, "  Efficiency Score: %.2f/100\n", efficiency_score);
        $fwrite(file_handle, "\n");
        
        // Performance metrics
        $fwrite(file_handle, "Performance Metrics:\n");
        $fwrite(file_handle, "  Throughput: %.2f ops/sec\n", performance.overall_throughput);
        $fwrite(file_handle, "  Latency P95: %.2f ms\n", performance.latency_p95);
        $fwrite(file_handle, "  Power Efficiency: %.2f ops/W\n", performance.power_efficiency);
        $fwrite(file_handle, "  Thermal Efficiency: %.2f ops/°C\n", performance.thermal_efficiency);
        $fwrite(file_handle, "\n");
        
        // Recommendations
        $fwrite(file_handle, "Recommendations:\n");
        generate_recommendations(file_handle, state_history, performance, config);
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("Analysis report generated: %s", filename), UVM_MEDIUM);
    endfunction
    
    // Generate recommendations based on analysis
    virtual function void generate_recommendations(int file_handle, 
                                                  system_state_t state_history[], 
                                                  integration_performance_t performance,
                                                  integration_test_config_t config);
        if (performance_score < 70.0) begin
            $fwrite(file_handle, "- Consider optimizing critical path performance\n");
        end
        
        if (scalability_score < 70.0) begin
            $fwrite(file_handle, "- Improve load balancing across cores\n");
        end
        
        if (reliability_score < 90.0) begin
            $fwrite(file_handle, "- Address error handling and system stability\n");
        end
        
        if (efficiency_score < 60.0) begin
            $fwrite(file_handle, "- Optimize power and thermal management\n");
        end
        
        real cache_eff = calculate_cache_efficiency(state_history);
        if (cache_eff < 0.8) begin
            $fwrite(file_handle, "- Improve cache hit rates through better data locality\n");
        end
        
        $fwrite(file_handle, "- Monitor system performance under varying workloads\n");
        $fwrite(file_handle, "- Consider implementing adaptive resource management\n");
    endfunction
    
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== INTEGRATION ANALYSIS SUMMARY ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Overall Score: %.2f/100", overall_score), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Performance: %.2f, Scalability: %.2f, Reliability: %.2f, Efficiency: %.2f", 
                 performance_score, scalability_score, reliability_score, efficiency_score), UVM_LOW)
        
        if (overall_score >= 80.0) begin
            `uvm_info(get_type_name(), "EXCELLENT system integration", UVM_LOW)
        end else if (overall_score >= 60.0) begin
            `uvm_info(get_type_name(), "GOOD system integration", UVM_LOW)
        end else begin
            `uvm_warning(get_type_name(), "System integration needs improvement")
        end
    endfunction
    
endclass : integration_analyzer

`endif // INTEGRATION_ANALYZER_SV