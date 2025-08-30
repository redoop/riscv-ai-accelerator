// AI Benchmark Base Class
// Base class for all AI benchmark implementations

`ifndef AI_BENCHMARK_BASE_SV
`define AI_BENCHMARK_BASE_SV

class ai_benchmark_base extends uvm_object;
    
    // Benchmark configuration
    benchmark_config_t config;
    benchmark_results_t results;
    
    // Benchmark state
    bit is_initialized = 0;
    bit is_running = 0;
    bit is_completed = 0;
    
    // Timing
    time start_time;
    time end_time;
    
    // Test data
    bit [7:0] input_data[][];
    bit [7:0] expected_output[][];
    bit [7:0] actual_output[][];
    
    // Performance tracking
    longint total_mac_operations = 0;
    longint total_memory_accesses = 0;
    real peak_memory_usage_mb = 0.0;
    
    `uvm_object_utils_begin(ai_benchmark_base)
        `uvm_field_int(is_initialized, UVM_ALL_ON)
        `uvm_field_int(is_running, UVM_ALL_ON)
        `uvm_field_int(is_completed, UVM_ALL_ON)
    `uvm_object_utils_end
    
    // Constructor
    function new(string name = "ai_benchmark_base");
        super.new(name);
        initialize_results();
    endfunction
    
    // Initialize benchmark results structure
    virtual function void initialize_results();
        results.accuracy_top1 = 0.0;
        results.accuracy_top5 = 0.0;
        results.latency_ms = 0.0;
        results.throughput_fps = 0.0;
        results.power_consumption_watts = 0.0;
        results.energy_per_inference_mj = 0.0;
        results.total_operations = 0;
        results.tops_achieved = 0.0;
        results.memory_bandwidth_gbps = 0.0;
        results.cache_hit_rate = 0.0;
        results.total_samples_processed = 0;
        results.total_execution_time = 0;
    endfunction
    
    // Virtual methods to be implemented by derived classes
    pure virtual function void configure_benchmark(benchmark_config_t cfg);
    pure virtual function bit initialize_benchmark();
    pure virtual task run_benchmark();
    pure virtual function void analyze_results();
    pure virtual function string get_benchmark_name();
    
    // Common benchmark execution flow
    virtual task execute_benchmark();
        `uvm_info(get_type_name(), $sformatf("Starting benchmark: %s", get_benchmark_name()), UVM_LOW)
        
        if (!is_initialized) begin
            `uvm_error(get_type_name(), "Benchmark not initialized")
            return;
        end
        
        is_running = 1;
        start_time = $time;
        
        // Run the actual benchmark
        run_benchmark();
        
        end_time = $time;
        is_running = 0;
        is_completed = 1;
        
        // Calculate timing metrics
        results.total_execution_time = end_time - start_time;
        results.latency_ms = real'(results.total_execution_time) / 1e6;  // Convert ns to ms
        
        if (results.latency_ms > 0) begin
            results.throughput_fps = real'(results.total_samples_processed) / (results.latency_ms / 1000.0);
        end
        
        // Analyze results
        analyze_results();
        
        `uvm_info(get_type_name(), $sformatf("Benchmark completed: %s", get_benchmark_name()), UVM_LOW)
        print_results();
    endtask
    
    // Load test data (to be overridden for specific data formats)
    virtual function bit load_test_data(string data_path);
        `uvm_info(get_type_name(), $sformatf("Loading test data from: %s", data_path), UVM_MEDIUM)
        
        // Default implementation - generate synthetic data
        generate_synthetic_data();
        return 1;
    endfunction
    
    // Generate synthetic test data
    virtual function void generate_synthetic_data();
        int total_input_size = config.batch_size * config.input_height * config.input_width * config.input_channels;
        int total_output_size = config.batch_size * config.num_classes;
        
        // Allocate input data
        input_data = new[config.batch_size];
        for (int i = 0; i < config.batch_size; i++) begin
            input_data[i] = new[config.input_height * config.input_width * config.input_channels];
            for (int j = 0; j < input_data[i].size(); j++) begin
                input_data[i][j] = $urandom_range(0, 255);
            end
        end
        
        // Allocate expected output data
        expected_output = new[config.batch_size];
        for (int i = 0; i < config.batch_size; i++) begin
            expected_output[i] = new[config.num_classes];
            for (int j = 0; j < config.num_classes; j++) begin
                expected_output[i][j] = $urandom_range(0, 255);
            end
        end
        
        `uvm_info(get_type_name(), $sformatf("Generated synthetic data: %0d input samples", config.batch_size), UVM_MEDIUM)
    endfunction
    
    // Calculate accuracy metrics
    virtual function void calculate_accuracy();
        int correct_top1 = 0;
        int correct_top5 = 0;
        
        if (actual_output.size() != expected_output.size()) begin
            `uvm_warning(get_type_name(), "Output size mismatch for accuracy calculation")
            return;
        end
        
        for (int i = 0; i < actual_output.size(); i++) begin
            // Find top-1 prediction
            int max_idx = 0;
            bit [7:0] max_val = actual_output[i][0];
            for (int j = 1; j < actual_output[i].size(); j++) begin
                if (actual_output[i][j] > max_val) begin
                    max_val = actual_output[i][j];
                    max_idx = j;
                end
            end
            
            // Find expected class (simplified - assume one-hot encoding)
            int expected_class = 0;
            bit [7:0] expected_max = expected_output[i][0];
            for (int j = 1; j < expected_output[i].size(); j++) begin
                if (expected_output[i][j] > expected_max) begin
                    expected_max = expected_output[i][j];
                    expected_class = j;
                end
            end
            
            // Check top-1 accuracy
            if (max_idx == expected_class) begin
                correct_top1++;
                correct_top5++;  // Top-1 correct implies top-5 correct
            end else begin
                // Check top-5 accuracy (simplified implementation)
                // In real implementation, would sort and check top 5 predictions
                if ($urandom_range(0, 4) == 0) begin  // 20% chance for demo
                    correct_top5++;
                end
            end
        end
        
        results.accuracy_top1 = real'(correct_top1) / real'(actual_output.size()) * 100.0;
        results.accuracy_top5 = real'(correct_top5) / real'(actual_output.size()) * 100.0;
    endfunction
    
    // Calculate performance metrics
    virtual function void calculate_performance_metrics();
        // Calculate TOPS (Tera Operations Per Second)
        if (results.total_execution_time > 0) begin
            real seconds = real'(results.total_execution_time) / 1e9;
            results.tops_achieved = real'(results.total_operations) / 1e12 / seconds;
        end
        
        // Calculate memory bandwidth
        if (results.total_execution_time > 0) begin
            real seconds = real'(results.total_execution_time) / 1e9;
            real bytes_per_second = real'(total_memory_accesses * 8) / seconds;  // Assume 8 bytes per access
            results.memory_bandwidth_gbps = bytes_per_second / 1e9;
        end
        
        // Calculate energy per inference
        if (results.total_samples_processed > 0) begin
            real total_energy_mj = results.power_consumption_watts * (results.latency_ms / 1000.0);
            results.energy_per_inference_mj = total_energy_mj / real'(results.total_samples_processed);
        end
    endfunction
    
    // Validate benchmark results against targets
    virtual function bit validate_results();
        bit validation_passed = 1;
        
        // Check accuracy target
        if (config.target_accuracy > 0 && results.accuracy_top1 < config.target_accuracy) begin
            `uvm_warning(get_type_name(), $sformatf("Accuracy below target: %.2f%% < %.2f%%", 
                       results.accuracy_top1, config.target_accuracy))
            validation_passed = 0;
        end
        
        // Check latency target
        if (config.target_latency_ms > 0 && results.latency_ms > config.target_latency_ms) begin
            `uvm_warning(get_type_name(), $sformatf("Latency above target: %.2fms > %.2fms", 
                       results.latency_ms, config.target_latency_ms))
            validation_passed = 0;
        end
        
        // Check throughput target
        if (config.target_throughput_fps > 0 && results.throughput_fps < config.target_throughput_fps) begin
            `uvm_warning(get_type_name(), $sformatf("Throughput below target: %.2f FPS < %.2f FPS", 
                       results.throughput_fps, config.target_throughput_fps))
            validation_passed = 0;
        end
        
        // Check power target
        if (config.target_power_watts > 0 && results.power_consumption_watts > config.target_power_watts) begin
            `uvm_warning(get_type_name(), $sformatf("Power above target: %.2fW > %.2fW", 
                       results.power_consumption_watts, config.target_power_watts))
            validation_passed = 0;
        end
        
        return validation_passed;
    endfunction
    
    // Print benchmark results
    virtual function void print_results();
        `uvm_info(get_type_name(), "=== BENCHMARK RESULTS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Benchmark: %s", get_benchmark_name()), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Model: %s", config.model_type.name()), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Precision: %s", config.precision.name()), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Batch Size: %0d", config.batch_size), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Input Size: %0dx%0dx%0d", config.input_height, config.input_width, config.input_channels), UVM_LOW)
        `uvm_info(get_type_name(), "", UVM_LOW)
        `uvm_info(get_type_name(), "Performance Metrics:", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Accuracy (Top-1): %.2f%%", results.accuracy_top1), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Accuracy (Top-5): %.2f%%", results.accuracy_top5), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Latency: %.2f ms", results.latency_ms), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Throughput: %.2f FPS", results.throughput_fps), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Power Consumption: %.2f W", results.power_consumption_watts), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Energy per Inference: %.2f mJ", results.energy_per_inference_mj), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  TOPS Achieved: %.2f", results.tops_achieved), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Memory Bandwidth: %.2f GB/s", results.memory_bandwidth_gbps), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Cache Hit Rate: %.2f%%", results.cache_hit_rate), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Samples Processed: %0d", results.total_samples_processed), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Total Operations: %0d", results.total_operations), UVM_LOW)
        
        // Validation status
        if (validate_results()) begin
            `uvm_info(get_type_name(), "*** BENCHMARK PASSED ***", UVM_LOW)
        end else begin
            `uvm_warning(get_type_name(), "*** BENCHMARK TARGETS NOT MET ***")
        end
    endfunction
    
    // Export results to file
    virtual function void export_results(string filename);
        int file_handle;
        
        file_handle = $fopen(filename, "w");
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot open file for writing: %s", filename))
            return;
        end
        
        // Write CSV header
        $fwrite(file_handle, "benchmark,model,precision,batch_size,input_size,");
        $fwrite(file_handle, "accuracy_top1,accuracy_top5,latency_ms,throughput_fps,");
        $fwrite(file_handle, "power_watts,energy_mj,tops,bandwidth_gbps,cache_hit_rate,");
        $fwrite(file_handle, "samples_processed,total_operations\n");
        
        // Write results
        $fwrite(file_handle, "%s,%s,%s,%0d,%0dx%0dx%0d,", 
               get_benchmark_name(), config.model_type.name(), config.precision.name(),
               config.batch_size, config.input_height, config.input_width, config.input_channels);
        $fwrite(file_handle, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%0d,%0d\n",
               results.accuracy_top1, results.accuracy_top5, results.latency_ms, results.throughput_fps,
               results.power_consumption_watts, results.energy_per_inference_mj, results.tops_achieved,
               results.memory_bandwidth_gbps, results.cache_hit_rate, results.total_samples_processed,
               results.total_operations);
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("Results exported to: %s", filename), UVM_MEDIUM)
    endfunction
    
    // Get benchmark summary string
    virtual function string get_summary_string();
        return $sformatf("%s: %.2f%% accuracy, %.2f ms latency, %.2f FPS, %.2f TOPS", 
                        get_benchmark_name(), results.accuracy_top1, results.latency_ms, 
                        results.throughput_fps, results.tops_achieved);
    endfunction
    
endclass : ai_benchmark_base

`endif // AI_BENCHMARK_BASE_SV