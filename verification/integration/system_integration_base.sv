// System Integration Base Test
// Base class for all system-level integration tests

`ifndef SYSTEM_INTEGRATION_BASE_SV
`define SYSTEM_INTEGRATION_BASE_SV

class system_integration_base extends uvm_test;
    
    // Test configuration
    integration_test_config_t config;
    integration_performance_t performance;
    
    // System monitoring
    system_monitor monitor;
    integration_analyzer analyzer;
    
    // Test state
    bit test_running = 0;
    bit test_passed = 0;
    time test_start_time;
    time test_end_time;
    
    // System state tracking
    system_state_t current_state;
    system_state_t state_history[$];
    
    // Error tracking
    int error_count = 0;
    string error_messages[$];
    
    `uvm_component_utils_begin(system_integration_base)
        `uvm_field_int(test_running, UVM_ALL_ON)
        `uvm_field_int(test_passed, UVM_ALL_ON)
        `uvm_field_int(error_count, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "system_integration_base", uvm_component parent = null);
        super.new(name, parent);
        initialize_config();
        initialize_performance();
    endfunction
    
    // Initialize default configuration
    virtual function void initialize_config();
        config.test_type = MULTI_CORE_COORDINATION;
        config.complexity = BASIC_INTEGRATION;
        config.active_components = new[8];
        config.active_components[0] = CPU_CORE_0;
        config.active_components[1] = CPU_CORE_1;
        config.active_components[2] = L1_ICACHE;
        config.active_components[3] = L1_DCACHE;
        config.active_components[4] = L2_CACHE;
        config.active_components[5] = MEMORY_CONTROLLER;
        config.active_components[6] = NOC_ROUTER;
        config.active_components[7] = POWER_MANAGER;
        
        config.num_cores_active = 2;
        config.num_tpus_active = 0;
        config.num_vpus_active = 0;
        config.enable_cache_coherency = 1;
        config.enable_power_management = 1;
        config.enable_thermal_management = 1;
        config.enable_error_injection = 0;
        config.test_duration_ms = 100.0;
        config.max_concurrent_transactions = 64;
        config.workload_pattern = "mixed";
    endfunction
    
    // Initialize performance metrics
    virtual function void initialize_performance();
        performance.overall_throughput = 0.0;
        performance.latency_p50 = 0.0;
        performance.latency_p95 = 0.0;
        performance.latency_p99 = 0.0;
        performance.power_efficiency = 0.0;
        performance.thermal_efficiency = 0.0;
        performance.scalability_factor = 0.0;
        performance.reliability_score = 0.0;
        performance.total_operations = 0;
        performance.total_execution_time = 0;
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Create system monitor
        monitor = system_monitor::type_id::create("monitor", this);
        
        // Create integration analyzer
        analyzer = integration_analyzer::type_id::create("analyzer", this);
        
        // Configure components
        configure_test();
    endfunction
    
    // Configure test - to be overridden by derived classes
    virtual function void configure_test();
        `uvm_info(get_type_name(), "Configuring base integration test", UVM_MEDIUM)
    endfunction
    
    // Run phase
    virtual task run_phase(uvm_phase phase);
        phase.raise_objection(this, "Running system integration test");
        
        test_start_time = $time;
        test_running = 1;
        
        `uvm_info(get_type_name(), $sformatf("Starting %s integration test", config.test_type.name()), UVM_LOW)
        
        // Initialize system
        initialize_system();
        
        // Run the actual test
        run_integration_test();
        
        // Finalize test
        finalize_test();
        
        test_end_time = $time;
        test_running = 0;
        
        // Analyze results
        analyze_test_results();
        
        `uvm_info(get_type_name(), $sformatf("Integration test completed: %s", test_passed ? "PASSED" : "FAILED"), UVM_LOW)
        
        phase.drop_objection(this, "Integration test completed");
    endtask
    
    // Initialize system components
    virtual task initialize_system();
        `uvm_info(get_type_name(), "Initializing system components", UVM_MEDIUM)
        
        // Reset system state
        reset_system_state();
        
        // Initialize active components
        foreach (config.active_components[i]) begin
            initialize_component(config.active_components[i]);
        end
        
        // Wait for system stabilization
        repeat(100) @(posedge monitor.clk);
        
        `uvm_info(get_type_name(), "System initialization complete", UVM_MEDIUM)
    endtask
    
    // Reset system state
    virtual function void reset_system_state();
        current_state.cpu_utilization = '{0.0, 0.0, 0.0, 0.0};
        current_state.tpu_utilization = '{0.0, 0.0};
        current_state.vpu_utilization = '{0.0, 0.0};
        current_state.cache_hit_rates = '{0.0, 0.0, 0.0, 0.0};
        current_state.memory_bandwidth_utilization = 0.0;
        current_state.noc_bandwidth_utilization = 0.0;
        current_state.power_consumption_watts = 10.0;  // Baseline power
        current_state.temperature_celsius = 45.0;      // Baseline temperature
        current_state.active_transactions = 0;
        current_state.completed_transactions = 0;
        current_state.error_count = 0;
        current_state.current_time = $time;
    endfunction
    
    // Initialize individual component
    virtual task initialize_component(system_component_e component);
        case (component)
            CPU_CORE_0, CPU_CORE_1, CPU_CORE_2, CPU_CORE_3: begin
                `uvm_info(get_type_name(), $sformatf("Initializing %s", component.name()), UVM_HIGH)
                // CPU core initialization
                repeat(10) @(posedge monitor.clk);
            end
            
            TPU_UNIT_0, TPU_UNIT_1: begin
                `uvm_info(get_type_name(), $sformatf("Initializing %s", component.name()), UVM_HIGH)
                // TPU initialization
                repeat(20) @(posedge monitor.clk);
            end
            
            VPU_UNIT_0, VPU_UNIT_1: begin
                `uvm_info(get_type_name(), $sformatf("Initializing %s", component.name()), UVM_HIGH)
                // VPU initialization
                repeat(15) @(posedge monitor.clk);
            end
            
            L1_ICACHE, L1_DCACHE, L2_CACHE, L3_CACHE: begin
                `uvm_info(get_type_name(), $sformatf("Initializing %s", component.name()), UVM_HIGH)
                // Cache initialization
                repeat(5) @(posedge monitor.clk);
            end
            
            MEMORY_CONTROLLER: begin
                `uvm_info(get_type_name(), "Initializing memory controller", UVM_HIGH)
                // Memory controller initialization
                repeat(50) @(posedge monitor.clk);
            end
            
            NOC_ROUTER: begin
                `uvm_info(get_type_name(), "Initializing NoC router", UVM_HIGH)
                // NoC router initialization
                repeat(25) @(posedge monitor.clk);
            end
            
            POWER_MANAGER: begin
                `uvm_info(get_type_name(), "Initializing power manager", UVM_HIGH)
                // Power manager initialization
                repeat(30) @(posedge monitor.clk);
            end
            
            THERMAL_CONTROLLER: begin
                `uvm_info(get_type_name(), "Initializing thermal controller", UVM_HIGH)
                // Thermal controller initialization
                repeat(20) @(posedge monitor.clk);
            end
        endcase
    endtask
    
    // Main integration test - to be overridden by derived classes
    virtual task run_integration_test();
        `uvm_info(get_type_name(), "Running base integration test", UVM_MEDIUM)
        
        // Basic system operation test
        fork
            monitor_system_state();
            generate_basic_workload();
        join
    endtask
    
    // Monitor system state continuously
    virtual task monitor_system_state();
        while (test_running) begin
            // Update system state
            update_system_state();
            
            // Store state history
            state_history.push_back(current_state);
            
            // Check for errors or anomalies
            check_system_health();
            
            // Wait before next monitoring cycle
            repeat(100) @(posedge monitor.clk);
        end
    endtask
    
    // Update current system state
    virtual function void update_system_state();
        current_state.current_time = $time;
        
        // Update utilization metrics (simplified simulation)
        for (int i = 0; i < config.num_cores_active; i++) begin
            current_state.cpu_utilization[i] = $urandom_range(20, 95) / 100.0;
        end
        
        for (int i = 0; i < config.num_tpus_active; i++) begin
            current_state.tpu_utilization[i] = $urandom_range(10, 90) / 100.0;
        end
        
        for (int i = 0; i < config.num_vpus_active; i++) begin
            current_state.vpu_utilization[i] = $urandom_range(15, 85) / 100.0;
        end
        
        // Update cache hit rates
        current_state.cache_hit_rates[0] = $urandom_range(85, 98) / 100.0;  // L1I
        current_state.cache_hit_rates[1] = $urandom_range(80, 95) / 100.0;  // L1D
        current_state.cache_hit_rates[2] = $urandom_range(70, 90) / 100.0;  // L2
        current_state.cache_hit_rates[3] = $urandom_range(60, 85) / 100.0;  // L3
        
        // Update bandwidth utilization
        current_state.memory_bandwidth_utilization = $urandom_range(30, 80) / 100.0;
        current_state.noc_bandwidth_utilization = $urandom_range(20, 70) / 100.0;
        
        // Update power and thermal
        real base_power = 10.0 + (config.num_cores_active * 5.0) + 
                         (config.num_tpus_active * 15.0) + (config.num_vpus_active * 10.0);
        real activity_factor = (current_state.cpu_utilization[0] + current_state.cpu_utilization[1] + 
                               current_state.cpu_utilization[2] + current_state.cpu_utilization[3]) / 4.0;
        current_state.power_consumption_watts = base_power * (0.5 + 0.5 * activity_factor);
        
        // Temperature follows power with thermal inertia
        real target_temp = 45.0 + (current_state.power_consumption_watts - 10.0) * 0.8;
        current_state.temperature_celsius = current_state.temperature_celsius * 0.9 + target_temp * 0.1;
    endfunction
    
    // Check system health and detect anomalies
    virtual function void check_system_health();
        // Check temperature limits
        if (current_state.temperature_celsius > 85.0) begin
            error_count++;
            error_messages.push_back($sformatf("Temperature too high: %.1f°C", current_state.temperature_celsius));
            `uvm_warning(get_type_name(), $sformatf("High temperature detected: %.1f°C", current_state.temperature_celsius))
        end
        
        // Check power consumption
        if (current_state.power_consumption_watts > 100.0) begin
            error_count++;
            error_messages.push_back($sformatf("Power consumption too high: %.1fW", current_state.power_consumption_watts));
            `uvm_warning(get_type_name(), $sformatf("High power consumption: %.1fW", current_state.power_consumption_watts))
        end
        
        // Check cache hit rates
        for (int i = 0; i < 4; i++) begin
            if (current_state.cache_hit_rates[i] < 0.5) begin
                error_count++;
                error_messages.push_back($sformatf("Low cache hit rate: Cache %0d = %.1f%%", i, current_state.cache_hit_rates[i] * 100.0));
                `uvm_warning(get_type_name(), $sformatf("Low cache hit rate detected: Cache %0d = %.1f%%", i, current_state.cache_hit_rates[i] * 100.0))
            end
        end
        
        // Update error count in current state
        current_state.error_count = error_count;
    endfunction
    
    // Generate basic workload for testing
    virtual task generate_basic_workload();
        int num_transactions = $urandom_range(100, 500);
        
        `uvm_info(get_type_name(), $sformatf("Generating %0d transactions", num_transactions), UVM_MEDIUM)
        
        for (int i = 0; i < num_transactions && test_running; i++) begin
            // Generate random transaction
            generate_random_transaction();
            
            // Update transaction counters
            current_state.active_transactions++;
            
            // Random delay between transactions
            repeat($urandom_range(10, 100)) @(posedge monitor.clk);
            
            // Complete transaction
            current_state.active_transactions--;
            current_state.completed_transactions++;
            performance.total_operations++;
        end
    endtask
    
    // Generate random transaction
    virtual task generate_random_transaction();
        // Simulate different types of transactions
        int transaction_type = $urandom_range(0, 3);
        
        case (transaction_type)
            0: simulate_cpu_transaction();
            1: simulate_memory_transaction();
            2: simulate_cache_transaction();
            3: simulate_noc_transaction();
        endcase
    endtask
    
    // Simulate CPU transaction
    virtual task simulate_cpu_transaction();
        repeat($urandom_range(5, 20)) @(posedge monitor.clk);
    endtask
    
    // Simulate memory transaction
    virtual task simulate_memory_transaction();
        repeat($urandom_range(10, 50)) @(posedge monitor.clk);
    endtask
    
    // Simulate cache transaction
    virtual task simulate_cache_transaction();
        repeat($urandom_range(1, 10)) @(posedge monitor.clk);
    endtask
    
    // Simulate NoC transaction
    virtual task simulate_noc_transaction();
        repeat($urandom_range(3, 15)) @(posedge monitor.clk);
    endtask
    
    // Finalize test
    virtual task finalize_test();
        `uvm_info(get_type_name(), "Finalizing integration test", UVM_MEDIUM)
        
        // Wait for all transactions to complete
        while (current_state.active_transactions > 0) begin
            repeat(10) @(posedge monitor.clk);
        end
        
        // Final system state update
        update_system_state();
        state_history.push_back(current_state);
        
        `uvm_info(get_type_name(), "Test finalization complete", UVM_MEDIUM)
    endtask
    
    // Analyze test results
    virtual function void analyze_test_results();
        `uvm_info(get_type_name(), "Analyzing integration test results", UVM_MEDIUM)
        
        // Calculate performance metrics
        calculate_performance_metrics();
        
        // Determine test pass/fail
        determine_test_result();
        
        // Generate detailed analysis
        if (analyzer != null) begin
            analyzer.analyze_integration_results(state_history, performance, config);
        end
    endfunction
    
    // Calculate performance metrics
    virtual function void calculate_performance_metrics();
        performance.total_execution_time = test_end_time - test_start_time;
        
        if (performance.total_execution_time > 0) begin
            real seconds = real'(performance.total_execution_time) / 1e9;
            performance.overall_throughput = real'(performance.total_operations) / seconds;
        end
        
        // Calculate average metrics from state history
        if (state_history.size() > 0) begin
            real total_power = 0.0;
            real total_temp = 0.0;
            
            foreach (state_history[i]) begin
                total_power += state_history[i].power_consumption_watts;
                total_temp += state_history[i].temperature_celsius;
            end
            
            real avg_power = total_power / real'(state_history.size());
            real avg_temp = total_temp / real'(state_history.size());
            
            // Power efficiency (operations per watt)
            if (avg_power > 0) begin
                performance.power_efficiency = performance.overall_throughput / avg_power;
            end
            
            // Thermal efficiency (operations per degree above baseline)
            real temp_delta = avg_temp - 45.0;
            if (temp_delta > 0) begin
                performance.thermal_efficiency = performance.overall_throughput / temp_delta;
            end
        end
        
        // Reliability score based on error rate
        if (performance.total_operations > 0) begin
            real error_rate = real'(error_count) / real'(performance.total_operations);
            performance.reliability_score = (1.0 - error_rate) * 100.0;
        end
    endfunction
    
    // Determine overall test result
    virtual function void determine_test_result();
        test_passed = 1;
        
        // Check error count
        if (error_count > (performance.total_operations / 100)) begin  // More than 1% error rate
            test_passed = 0;
            `uvm_error(get_type_name(), $sformatf("High error rate: %0d errors in %0d operations", error_count, performance.total_operations))
        end
        
        // Check reliability score
        if (performance.reliability_score < 95.0) begin
            test_passed = 0;
            `uvm_error(get_type_name(), $sformatf("Low reliability score: %.2f%%", performance.reliability_score))
        end
        
        // Check power efficiency
        if (performance.power_efficiency < 1.0) begin
            `uvm_warning(get_type_name(), $sformatf("Low power efficiency: %.2f ops/W", performance.power_efficiency))
        end
        
        // Additional test-specific checks can be added by derived classes
        perform_additional_checks();
    endfunction
    
    // Additional checks - to be overridden by derived classes
    virtual function void perform_additional_checks();
        // Base implementation does nothing
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== INTEGRATION TEST REPORT ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Test Type: %s", config.test_type.name()), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Complexity: %s", config.complexity.name()), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Duration: %.2f ms", real'(performance.total_execution_time) / 1e6), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Operations: %0d", performance.total_operations), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Throughput: %.2f ops/sec", performance.overall_throughput), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Power Efficiency: %.2f ops/W", performance.power_efficiency), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Thermal Efficiency: %.2f ops/°C", performance.thermal_efficiency), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Reliability Score: %.2f%%", performance.reliability_score), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Error Count: %0d", error_count), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Test Result: %s", test_passed ? "PASSED" : "FAILED"), UVM_LOW)
        
        // Print error messages if any
        if (error_messages.size() > 0) begin
            `uvm_info(get_type_name(), "=== ERROR SUMMARY ===", UVM_LOW)
            foreach (error_messages[i]) begin
                `uvm_info(get_type_name(), error_messages[i], UVM_LOW)
            end
        end
    endfunction
    
endclass : system_integration_base

`endif // SYSTEM_INTEGRATION_BASE_SV