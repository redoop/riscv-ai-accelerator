// Benchmark Analyzer
// Advanced analysis and comparison of benchmark results

`ifndef BENCHMARK_ANALYZER_SV
`define BENCHMARK_ANALYZER_SV

class benchmark_analyzer extends uvm_component;
    
    // Analysis configuration
    bit enable_performance_analysis = 1;
    bit enable_efficiency_analysis = 1;
    bit enable_comparison_analysis = 1;
    bit enable_trend_analysis = 1;
    bit generate_charts = 1;
    string baseline_results_file = "";
    
    // Analysis results
    typedef struct {
        string metric_name;
        real current_value;
        real baseline_value;
        real improvement_percent;
        string status;  // "improved", "degraded", "stable"
    } comparison_result_t;
    
    comparison_result_t comparison_results[];
    
    // Performance categories
    typedef struct {
        string category_name;
        real score;
        real weight;
        string[] included_benchmarks;
    } performance_category_t;
    
    performance_category_t performance_categories[];
    
    // Efficiency metrics
    typedef struct {
        string benchmark_name;
        real performance_per_watt;
        real performance_per_area;  // If area data available
        real performance_per_dollar;  // If cost data available
        real efficiency_score;
    } efficiency_metrics_t;
    
    efficiency_metrics_t efficiency_results[];
    
    `uvm_component_utils_begin(benchmark_analyzer)
        `uvm_field_int(enable_performance_analysis, UVM_ALL_ON)
        `uvm_field_int(enable_efficiency_analysis, UVM_ALL_ON)
        `uvm_field_int(enable_comparison_analysis, UVM_ALL_ON)
        `uvm_field_int(enable_trend_analysis, UVM_ALL_ON)
        `uvm_field_string(baseline_results_file, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "benchmark_analyzer", uvm_component parent = null);
        super.new(name, parent);
        initialize_performance_categories();
    endfunction
    
    // Initialize performance categories
    virtual function void initialize_performance_categories();
        performance_categories = new[5];
        
        // Image Processing Performance
        performance_categories[0].category_name = "Image Processing";
        performance_categories[0].weight = 0.25;
        performance_categories[0].included_benchmarks = new[4];
        performance_categories[0].included_benchmarks[0] = "ResNet50-FP32-ImageNet";
        performance_categories[0].included_benchmarks[1] = "ResNet50-INT8-ImageNet";
        performance_categories[0].included_benchmarks[2] = "MobileNet-V2-FP16-ImageNet";
        performance_categories[0].included_benchmarks[3] = "EfficientNet-B0-FP32-ImageNet";
        
        // Object Detection Performance
        performance_categories[1].category_name = "Object Detection";
        performance_categories[1].weight = 0.20;
        performance_categories[1].included_benchmarks = new[3];
        performance_categories[1].included_benchmarks[0] = "YOLOv5-FP32-COCO";
        performance_categories[1].included_benchmarks[1] = "SSD-MobileNet-FP16-COCO";
        performance_categories[1].included_benchmarks[2] = "Faster-RCNN-FP32-COCO";
        
        // Natural Language Processing
        performance_categories[2].category_name = "Natural Language Processing";
        performance_categories[2].weight = 0.25;
        performance_categories[2].included_benchmarks = new[3];
        performance_categories[2].included_benchmarks[0] = "BERT-Base-TextClassification";
        performance_categories[2].included_benchmarks[1] = "BERT-Large-QuestionAnswering";
        performance_categories[2].included_benchmarks[2] = "GPT2-Small-LanguageModeling";
        
        // Recommendation Systems
        performance_categories[3].category_name = "Recommendation Systems";
        performance_categories[3].weight = 0.15;
        performance_categories[3].included_benchmarks = new[3];
        performance_categories[3].included_benchmarks[0] = "Wide-Deep-MovieLens";
        performance_categories[3].included_benchmarks[1] = "DeepFM-MovieLens";
        performance_categories[3].included_benchmarks[2] = "NeuralCF-MovieLens";
        
        // MLPerf Standard Benchmarks
        performance_categories[4].category_name = "MLPerf Standard";
        performance_categories[4].weight = 0.15;
        performance_categories[4].included_benchmarks = new[3];
        performance_categories[4].included_benchmarks[0] = "MLPerf-ResNet50-Inference";
        performance_categories[4].included_benchmarks[1] = "MLPerf-BERT-Inference";
        performance_categories[4].included_benchmarks[2] = "MLPerf-SSD-MobileNet-Inference";
    endfunction
    
    // Analyze benchmark results
    virtual function void analyze_results(benchmark_results_t results[], string benchmark_names[]);
        `uvm_info(get_type_name(), $sformatf("Analyzing %0d benchmark results", results.size()), UVM_MEDIUM)
        
        if (enable_performance_analysis) begin
            analyze_performance(results, benchmark_names);
        end
        
        if (enable_efficiency_analysis) begin
            analyze_efficiency(results, benchmark_names);
        end
        
        if (enable_comparison_analysis && baseline_results_file != "") begin
            analyze_comparison(results, benchmark_names);
        end
        
        if (enable_trend_analysis) begin
            analyze_trends(results, benchmark_names);
        end
        
        generate_analysis_report(results, benchmark_names);
    endfunction
    
    // Analyze overall performance
    virtual function void analyze_performance(benchmark_results_t results[], string benchmark_names[]);
        `uvm_info(get_type_name(), "Performing performance analysis", UVM_MEDIUM)
        
        // Calculate category scores
        for (int cat = 0; cat < performance_categories.size(); cat++) begin
            real category_score = 0.0;
            int benchmarks_found = 0;
            
            for (int bench = 0; bench < performance_categories[cat].included_benchmarks.size(); bench++) begin
                string target_benchmark = performance_categories[cat].included_benchmarks[bench];
                
                // Find matching benchmark
                for (int i = 0; i < benchmark_names.size(); i++) begin
                    if (benchmark_names[i] == target_benchmark) begin
                        // Calculate composite score (accuracy + performance)
                        real accuracy_score = results[i].accuracy_top1;
                        real performance_score = calculate_performance_score(results[i]);
                        real composite_score = (accuracy_score + performance_score) / 2.0;
                        
                        category_score += composite_score;
                        benchmarks_found++;
                        break;
                    end
                end
            end
            
            if (benchmarks_found > 0) begin
                performance_categories[cat].score = category_score / real'(benchmarks_found);
                `uvm_info(get_type_name(), $sformatf("%s Performance Score: %.2f", 
                         performance_categories[cat].category_name, performance_categories[cat].score), UVM_MEDIUM)
            end
        end
        
        // Calculate overall performance score
        real overall_score = 0.0;
        real total_weight = 0.0;
        
        for (int cat = 0; cat < performance_categories.size(); cat++) begin
            overall_score += performance_categories[cat].score * performance_categories[cat].weight;
            total_weight += performance_categories[cat].weight;
        end
        
        if (total_weight > 0) begin
            overall_score = overall_score / total_weight;
            `uvm_info(get_type_name(), $sformatf("Overall Performance Score: %.2f", overall_score), UVM_LOW)
        end
    endfunction
    
    // Calculate performance score for individual benchmark
    virtual function real calculate_performance_score(benchmark_results_t result);
        real throughput_score = 0.0;
        real latency_score = 0.0;
        real efficiency_score = 0.0;
        
        // Throughput score (higher is better)
        if (result.throughput_fps > 0) begin
            throughput_score = $log10(result.throughput_fps + 1) * 20.0;  // Logarithmic scaling
            if (throughput_score > 100.0) throughput_score = 100.0;
        end
        
        // Latency score (lower is better)
        if (result.latency_ms > 0) begin
            latency_score = 100.0 - $log10(result.latency_ms + 1) * 30.0;  // Inverse logarithmic
            if (latency_score < 0.0) latency_score = 0.0;
        end
        
        // Efficiency score (TOPS per Watt)
        if (result.power_consumption_watts > 0 && result.tops_achieved > 0) begin
            real tops_per_watt = result.tops_achieved / result.power_consumption_watts;
            efficiency_score = $log10(tops_per_watt + 1) * 25.0;
            if (efficiency_score > 100.0) efficiency_score = 100.0;
        end
        
        // Weighted average
        return (throughput_score * 0.4 + latency_score * 0.3 + efficiency_score * 0.3);
    endfunction
    
    // Analyze efficiency metrics
    virtual function void analyze_efficiency(benchmark_results_t results[], string benchmark_names[]);
        `uvm_info(get_type_name(), "Performing efficiency analysis", UVM_MEDIUM)
        
        efficiency_results = new[results.size()];
        
        for (int i = 0; i < results.size(); i++) begin
            efficiency_results[i].benchmark_name = benchmark_names[i];
            
            // Performance per Watt
            if (results[i].power_consumption_watts > 0) begin
                efficiency_results[i].performance_per_watt = results[i].tops_achieved / results[i].power_consumption_watts;
            end else begin
                efficiency_results[i].performance_per_watt = 0.0;
            end
            
            // Performance per Area (placeholder - would need actual area data)
            efficiency_results[i].performance_per_area = results[i].tops_achieved / 100.0;  // Assume 100mm² area
            
            // Performance per Dollar (placeholder - would need actual cost data)
            efficiency_results[i].performance_per_dollar = results[i].tops_achieved / 1000.0;  // Assume $1000 cost
            
            // Overall efficiency score
            efficiency_results[i].efficiency_score = calculate_efficiency_score(results[i]);
            
            `uvm_info(get_type_name(), $sformatf("%s Efficiency: %.2f TOPS/W, Score: %.2f", 
                     benchmark_names[i], efficiency_results[i].performance_per_watt, 
                     efficiency_results[i].efficiency_score), UVM_HIGH)
        end
        
        // Find most and least efficient benchmarks
        find_efficiency_extremes();
    endfunction
    
    // Calculate efficiency score
    virtual function real calculate_efficiency_score(benchmark_results_t result);
        real power_efficiency = 0.0;
        real memory_efficiency = 0.0;
        real compute_efficiency = 0.0;
        
        // Power efficiency
        if (result.power_consumption_watts > 0) begin
            power_efficiency = result.tops_achieved / result.power_consumption_watts * 10.0;
            if (power_efficiency > 100.0) power_efficiency = 100.0;
        end
        
        // Memory efficiency
        if (result.memory_bandwidth_gbps > 0) begin
            memory_efficiency = result.tops_achieved / result.memory_bandwidth_gbps * 5.0;
            if (memory_efficiency > 100.0) memory_efficiency = 100.0;
        end
        
        // Compute efficiency (utilization)
        compute_efficiency = result.cache_hit_rate;  // Use cache hit rate as proxy
        
        return (power_efficiency * 0.5 + memory_efficiency * 0.3 + compute_efficiency * 0.2);
    endfunction
    
    // Find efficiency extremes
    virtual function void find_efficiency_extremes();
        if (efficiency_results.size() == 0) return;
        
        int most_efficient_idx = 0;
        int least_efficient_idx = 0;
        
        for (int i = 1; i < efficiency_results.size(); i++) begin
            if (efficiency_results[i].efficiency_score > efficiency_results[most_efficient_idx].efficiency_score) begin
                most_efficient_idx = i;
            end
            if (efficiency_results[i].efficiency_score < efficiency_results[least_efficient_idx].efficiency_score) begin
                least_efficient_idx = i;
            end
        end
        
        `uvm_info(get_type_name(), $sformatf("Most Efficient: %s (Score: %.2f)", 
                 efficiency_results[most_efficient_idx].benchmark_name, 
                 efficiency_results[most_efficient_idx].efficiency_score), UVM_LOW)
        
        `uvm_info(get_type_name(), $sformatf("Least Efficient: %s (Score: %.2f)", 
                 efficiency_results[least_efficient_idx].benchmark_name, 
                 efficiency_results[least_efficient_idx].efficiency_score), UVM_LOW)
    endfunction
    
    // Analyze comparison with baseline
    virtual function void analyze_comparison(benchmark_results_t results[], string benchmark_names[]);
        `uvm_info(get_type_name(), $sformatf("Performing comparison analysis with baseline: %s", baseline_results_file), UVM_MEDIUM)
        
        // Load baseline results (simplified - in real implementation would parse file)
        benchmark_results_t baseline_results[];
        string baseline_names[];
        
        if (!load_baseline_results(baseline_results, baseline_names)) begin
            `uvm_warning(get_type_name(), "Failed to load baseline results, skipping comparison")
            return;
        end
        
        // Compare results
        comparison_results = new[results.size()];
        
        for (int i = 0; i < results.size(); i++) begin
            // Find matching baseline
            int baseline_idx = find_matching_baseline(benchmark_names[i], baseline_names);
            
            if (baseline_idx >= 0) begin
                compare_benchmark_results(results[i], baseline_results[baseline_idx], 
                                        benchmark_names[i], comparison_results[i]);
            end else begin
                `uvm_warning(get_type_name(), $sformatf("No baseline found for: %s", benchmark_names[i]))
            end
        end
        
        // Summarize comparison results
        summarize_comparison_results();
    endfunction
    
    // Load baseline results (simplified implementation)
    virtual function bit load_baseline_results(ref benchmark_results_t baseline_results[], ref string baseline_names[]);
        // Simplified implementation - generate synthetic baseline data
        // In real implementation, would parse actual baseline file
        
        baseline_results = new[5];
        baseline_names = new[5];
        
        // Generate synthetic baseline data (slightly worse than current)
        baseline_names[0] = "ResNet50-FP32-ImageNet";
        baseline_results[0].accuracy_top1 = 75.0;
        baseline_results[0].latency_ms = 12.0;
        baseline_results[0].throughput_fps = 83.0;
        baseline_results[0].tops_achieved = 45.0;
        
        baseline_names[1] = "BERT-Base-TextClassification";
        baseline_results[1].accuracy_top1 = 83.0;
        baseline_results[1].latency_ms = 25.0;
        baseline_results[1].throughput_fps = 40.0;
        baseline_results[1].tops_achieved = 35.0;
        
        // Add more baseline entries...
        
        return 1;
    endfunction
    
    // Find matching baseline benchmark
    virtual function int find_matching_baseline(string benchmark_name, string baseline_names[]);
        for (int i = 0; i < baseline_names.size(); i++) begin
            if (baseline_names[i] == benchmark_name) begin
                return i;
            end
        end
        return -1;
    endfunction
    
    // Compare individual benchmark results
    virtual function void compare_benchmark_results(benchmark_results_t current, benchmark_results_t baseline, 
                                                   string benchmark_name, ref comparison_result_t comparison);
        comparison.metric_name = benchmark_name;
        comparison.current_value = current.accuracy_top1;
        comparison.baseline_value = baseline.accuracy_top1;
        
        if (baseline.accuracy_top1 > 0) begin
            comparison.improvement_percent = (current.accuracy_top1 - baseline.accuracy_top1) / baseline.accuracy_top1 * 100.0;
        end else begin
            comparison.improvement_percent = 0.0;
        end
        
        // Determine status
        if (comparison.improvement_percent > 2.0) begin
            comparison.status = "improved";
        end else if (comparison.improvement_percent < -2.0) begin
            comparison.status = "degraded";
        end else begin
            comparison.status = "stable";
        end
        
        `uvm_info(get_type_name(), $sformatf("%s: %.2f%% vs %.2f%% (%.1f%% %s)", 
                 benchmark_name, comparison.current_value, comparison.baseline_value, 
                 comparison.improvement_percent, comparison.status), UVM_HIGH)
    endfunction
    
    // Summarize comparison results
    virtual function void summarize_comparison_results();
        int improved_count = 0;
        int degraded_count = 0;
        int stable_count = 0;
        real total_improvement = 0.0;
        
        for (int i = 0; i < comparison_results.size(); i++) begin
            case (comparison_results[i].status)
                "improved": begin
                    improved_count++;
                    total_improvement += comparison_results[i].improvement_percent;
                end
                "degraded": begin
                    degraded_count++;
                    total_improvement += comparison_results[i].improvement_percent;
                end
                "stable": stable_count++;
            endcase
        end
        
        `uvm_info(get_type_name(), "=== COMPARISON SUMMARY ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Improved: %0d benchmarks", improved_count), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Degraded: %0d benchmarks", degraded_count), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Stable: %0d benchmarks", stable_count), UVM_LOW)
        
        if (comparison_results.size() > 0) begin
            real average_improvement = total_improvement / real'(comparison_results.size());
            `uvm_info(get_type_name(), $sformatf("Average Improvement: %.2f%%", average_improvement), UVM_LOW)
        end
    endfunction
    
    // Analyze trends (simplified implementation)
    virtual function void analyze_trends(benchmark_results_t results[], string benchmark_names[]);
        `uvm_info(get_type_name(), "Performing trend analysis", UVM_MEDIUM)
        
        // Analyze accuracy vs performance tradeoffs
        analyze_accuracy_performance_tradeoff(results, benchmark_names);
        
        // Analyze precision impact
        analyze_precision_impact(results, benchmark_names);
        
        // Analyze model size impact
        analyze_model_size_impact(results, benchmark_names);
    endfunction
    
    // Analyze accuracy vs performance tradeoff
    virtual function void analyze_accuracy_performance_tradeoff(benchmark_results_t results[], string benchmark_names[]);
        real accuracy_performance_correlation = 0.0;
        int valid_samples = 0;
        
        for (int i = 0; i < results.size(); i++) begin
            if (results[i].accuracy_top1 > 0 && results[i].throughput_fps > 0) begin
                // Simple correlation analysis (in real implementation, would use proper statistical methods)
                real normalized_accuracy = results[i].accuracy_top1 / 100.0;
                real normalized_throughput = results[i].throughput_fps / 1000.0;  // Normalize to 0-1 range
                
                accuracy_performance_correlation += normalized_accuracy * normalized_throughput;
                valid_samples++;
            end
        end
        
        if (valid_samples > 0) begin
            accuracy_performance_correlation = accuracy_performance_correlation / real'(valid_samples);
            `uvm_info(get_type_name(), $sformatf("Accuracy-Performance Correlation: %.3f", accuracy_performance_correlation), UVM_MEDIUM)
        end
    endfunction
    
    // Analyze precision impact
    virtual function void analyze_precision_impact(benchmark_results_t results[], string benchmark_names[]);
        real fp32_avg_accuracy = 0.0, fp16_avg_accuracy = 0.0, int8_avg_accuracy = 0.0;
        real fp32_avg_throughput = 0.0, fp16_avg_throughput = 0.0, int8_avg_throughput = 0.0;
        int fp32_count = 0, fp16_count = 0, int8_count = 0;
        
        for (int i = 0; i < benchmark_names.size(); i++) begin
            if (benchmark_names[i].find("FP32") != -1) begin
                fp32_avg_accuracy += results[i].accuracy_top1;
                fp32_avg_throughput += results[i].throughput_fps;
                fp32_count++;
            end else if (benchmark_names[i].find("FP16") != -1) begin
                fp16_avg_accuracy += results[i].accuracy_top1;
                fp16_avg_throughput += results[i].throughput_fps;
                fp16_count++;
            end else if (benchmark_names[i].find("INT8") != -1) begin
                int8_avg_accuracy += results[i].accuracy_top1;
                int8_avg_throughput += results[i].throughput_fps;
                int8_count++;
            end
        end
        
        `uvm_info(get_type_name(), "=== PRECISION IMPACT ANALYSIS ===", UVM_MEDIUM)
        
        if (fp32_count > 0) begin
            fp32_avg_accuracy /= real'(fp32_count);
            fp32_avg_throughput /= real'(fp32_count);
            `uvm_info(get_type_name(), $sformatf("FP32: Avg Accuracy=%.2f%%, Avg Throughput=%.2f FPS", 
                     fp32_avg_accuracy, fp32_avg_throughput), UVM_MEDIUM)
        end
        
        if (fp16_count > 0) begin
            fp16_avg_accuracy /= real'(fp16_count);
            fp16_avg_throughput /= real'(fp16_count);
            `uvm_info(get_type_name(), $sformatf("FP16: Avg Accuracy=%.2f%%, Avg Throughput=%.2f FPS", 
                     fp16_avg_accuracy, fp16_avg_throughput), UVM_MEDIUM)
        end
        
        if (int8_count > 0) begin
            int8_avg_accuracy /= real'(int8_count);
            int8_avg_throughput /= real'(int8_count);
            `uvm_info(get_type_name(), $sformatf("INT8: Avg Accuracy=%.2f%%, Avg Throughput=%.2f FPS", 
                     int8_avg_accuracy, int8_avg_throughput), UVM_MEDIUM)
        end
    endfunction
    
    // Analyze model size impact
    virtual function void analyze_model_size_impact(benchmark_results_t results[], string benchmark_names[]);
        // Categorize models by size (based on name patterns)
        real small_model_avg_perf = 0.0, large_model_avg_perf = 0.0;
        int small_model_count = 0, large_model_count = 0;
        
        for (int i = 0; i < benchmark_names.size(); i++) begin
            if (benchmark_names[i].find("MobileNet") != -1 || benchmark_names[i].find("Small") != -1) begin
                small_model_avg_perf += results[i].throughput_fps;
                small_model_count++;
            end else if (benchmark_names[i].find("Large") != -1 || benchmark_names[i].find("ResNet") != -1) begin
                large_model_avg_perf += results[i].throughput_fps;
                large_model_count++;
            end
        end
        
        `uvm_info(get_type_name(), "=== MODEL SIZE IMPACT ANALYSIS ===", UVM_MEDIUM)
        
        if (small_model_count > 0) begin
            small_model_avg_perf /= real'(small_model_count);
            `uvm_info(get_type_name(), $sformatf("Small Models: Avg Throughput=%.2f FPS", small_model_avg_perf), UVM_MEDIUM)
        end
        
        if (large_model_count > 0) begin
            large_model_avg_perf /= real'(large_model_count);
            `uvm_info(get_type_name(), $sformatf("Large Models: Avg Throughput=%.2f FPS", large_model_avg_perf), UVM_MEDIUM)
        end
        
        if (small_model_count > 0 && large_model_count > 0) begin
            real size_impact_ratio = small_model_avg_perf / large_model_avg_perf;
            `uvm_info(get_type_name(), $sformatf("Small/Large Model Performance Ratio: %.2fx", size_impact_ratio), UVM_MEDIUM)
        end
    endfunction
    
    // Generate comprehensive analysis report
    virtual function void generate_analysis_report(benchmark_results_t results[], string benchmark_names[]);
        string filename = "benchmark_analysis_report.txt";
        int file_handle;
        
        file_handle = $fopen(filename, "w");
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot create analysis report: %s", filename))
            return;
        end
        
        // Report header
        $fwrite(file_handle, "RISC-V AI Accelerator Benchmark Analysis Report\n");
        $fwrite(file_handle, "================================================\n\n");
        $fwrite(file_handle, "Generated at: %0t\n", $time);
        $fwrite(file_handle, "Total Benchmarks Analyzed: %0d\n\n", results.size());
        
        // Performance category scores
        $fwrite(file_handle, "Performance Category Scores:\n");
        $fwrite(file_handle, "----------------------------\n");
        for (int i = 0; i < performance_categories.size(); i++) begin
            $fwrite(file_handle, "%-30s: %6.2f (Weight: %.2f)\n", 
                   performance_categories[i].category_name, 
                   performance_categories[i].score, 
                   performance_categories[i].weight);
        end
        $fwrite(file_handle, "\n");
        
        // Efficiency analysis
        if (efficiency_results.size() > 0) begin
            $fwrite(file_handle, "Efficiency Analysis:\n");
            $fwrite(file_handle, "-------------------\n");
            for (int i = 0; i < efficiency_results.size(); i++) begin
                $fwrite(file_handle, "%-40s: %6.2f TOPS/W, Score: %6.2f\n", 
                       efficiency_results[i].benchmark_name,
                       efficiency_results[i].performance_per_watt,
                       efficiency_results[i].efficiency_score);
            end
            $fwrite(file_handle, "\n");
        end
        
        // Comparison results
        if (comparison_results.size() > 0) begin
            $fwrite(file_handle, "Baseline Comparison:\n");
            $fwrite(file_handle, "-------------------\n");
            for (int i = 0; i < comparison_results.size(); i++) begin
                $fwrite(file_handle, "%-40s: %6.2f%% -> %6.2f%% (%+6.1f%%) [%s]\n",
                       comparison_results[i].metric_name,
                       comparison_results[i].baseline_value,
                       comparison_results[i].current_value,
                       comparison_results[i].improvement_percent,
                       comparison_results[i].status);
            end
            $fwrite(file_handle, "\n");
        end
        
        // Recommendations
        $fwrite(file_handle, "Recommendations:\n");
        $fwrite(file_handle, "---------------\n");
        generate_recommendations(file_handle, results, benchmark_names);
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("Analysis report generated: %s", filename), UVM_MEDIUM)
    endfunction
    
    // Generate recommendations based on analysis
    virtual function void generate_recommendations(int file_handle, benchmark_results_t results[], string benchmark_names[]);
        // Analyze results and generate actionable recommendations
        
        // Check for low accuracy benchmarks
        for (int i = 0; i < results.size(); i++) begin
            if (results[i].accuracy_top1 < 70.0) begin
                $fwrite(file_handle, "- Consider model optimization for %s (accuracy: %.2f%%)\n", 
                       benchmark_names[i], results[i].accuracy_top1);
            end
        end
        
        // Check for high latency benchmarks
        for (int i = 0; i < results.size(); i++) begin
            if (results[i].latency_ms > 100.0) begin
                $fwrite(file_handle, "- Optimize inference pipeline for %s (latency: %.2f ms)\n", 
                       benchmark_names[i], results[i].latency_ms);
            end
        end
        
        // Check for low efficiency benchmarks
        for (int i = 0; i < efficiency_results.size(); i++) begin
            if (efficiency_results[i].performance_per_watt < 10.0) begin
                $fwrite(file_handle, "- Improve power efficiency for %s (%.2f TOPS/W)\n", 
                       efficiency_results[i].benchmark_name, efficiency_results[i].performance_per_watt);
            end
        end
        
        // General recommendations
        $fwrite(file_handle, "- Consider INT8 quantization for improved throughput\n");
        $fwrite(file_handle, "- Evaluate batch size optimization for better utilization\n");
        $fwrite(file_handle, "- Monitor memory bandwidth utilization\n");
        $fwrite(file_handle, "- Consider model pruning for edge deployment scenarios\n");
    endfunction
    
endclass : benchmark_analyzer

`endif // BENCHMARK_ANALYZER_SV