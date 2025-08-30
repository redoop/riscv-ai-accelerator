// Cache Coherency Test
// Tests cache coherency protocols and data consistency across multiple cores

`ifndef CACHE_COHERENCY_TEST_SV
`define CACHE_COHERENCY_TEST_SV

class cache_coherency_test extends system_integration_base;
    
    // Cache coherency specific configuration
    string coherency_protocol = "MESI";  // MESI, MOESI, MSI
    int num_shared_variables = 16;
    int num_coherency_transactions = 1000;
    bit enable_false_sharing_test = 1;
    bit enable_true_sharing_test = 1;
    
    // Cache line tracking
    typedef struct {
        bit [63:0] address;
        bit [2:0] state[4];  // Cache state per core (M=3, E=2, S=1, I=0)
        bit [63:0] data[4];  // Data per core
        int last_writer;
        time last_write_time;
        int read_count[4];
        int write_count[4];
    } cache_line_info_t;
    
    cache_line_info_t shared_cache_lines[];
    
    // Coherency statistics
    int coherency_messages = 0;
    int cache_invalidations = 0;
    int cache_writebacks = 0;
    int false_sharing_events = 0;
    int true_sharing_events = 0;
    
    // Test patterns
    typedef enum {
        PRODUCER_CONSUMER,
        READER_WRITER,
        BARRIER_SYNC,
        FALSE_SHARING,
        TRUE_SHARING,
        RANDOM_ACCESS
    } coherency_pattern_e;
    
    `uvm_component_utils_begin(cache_coherency_test)
        `uvm_field_string(coherency_protocol, UVM_ALL_ON)
        `uvm_field_int(num_shared_variables, UVM_ALL_ON)
        `uvm_field_int(num_coherency_transactions, UVM_ALL_ON)
        `uvm_field_int(enable_false_sharing_test, UVM_ALL_ON)
        `uvm_field_int(enable_true_sharing_test, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "cache_coherency_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    virtual function void configure_test();
        super.configure_test();
        
        config.test_type = CACHE_COHERENCY;
        config.complexity = ADVANCED_INTEGRATION;
        config.num_cores_active = 4;
        config.num_tpus_active = 0;
        config.num_vpus_active = 0;
        config.enable_cache_coherency = 1;
        config.test_duration_ms = 300.0;
        config.max_concurrent_transactions = 256;
        config.workload_pattern = "cache_intensive";
        
        // Initialize shared cache lines
        initialize_shared_cache_lines();
        
        `uvm_info(get_type_name(), $sformatf("Configured cache coherency test with %s protocol", coherency_protocol), UVM_MEDIUM)
    endfunction
    
    // Initialize shared cache lines for testing
    virtual function void initialize_shared_cache_lines();
        shared_cache_lines = new[num_shared_variables];
        
        for (int i = 0; i < num_shared_variables; i++) begin
            shared_cache_lines[i].address = 64'h1000_0000 + (i * 64);  // 64-byte aligned
            shared_cache_lines[i].state = '{0, 0, 0, 0};  // All Invalid initially
            shared_cache_lines[i].data = '{64'h0, 64'h0, 64'h0, 64'h0};
            shared_cache_lines[i].last_writer = -1;
            shared_cache_lines[i].last_write_time = 0;
            shared_cache_lines[i].read_count = '{0, 0, 0, 0};
            shared_cache_lines[i].write_count = '{0, 0, 0, 0};
        end
        
        `uvm_info(get_type_name(), $sformatf("Initialized %0d shared cache lines", num_shared_variables), UVM_MEDIUM)
    endfunction
    
    virtual task run_integration_test();
        `uvm_info(get_type_name(), "Running cache coherency test", UVM_MEDIUM)
        
        fork
            monitor_system_state();
            run_coherency_patterns();
            monitor_coherency_protocol();
            if (enable_false_sharing_test) test_false_sharing();
            if (enable_true_sharing_test) test_true_sharing();
        join
    endtask
    
    // Run various coherency test patterns
    virtual task run_coherency_patterns();
        `uvm_info(get_type_name(), "Starting coherency pattern tests", UVM_MEDIUM)
        
        fork
            // Run different patterns in parallel
            run_pattern_test(PRODUCER_CONSUMER, num_coherency_transactions / 6);
            run_pattern_test(READER_WRITER, num_coherency_transactions / 6);
            run_pattern_test(BARRIER_SYNC, num_coherency_transactions / 6);
            run_pattern_test(RANDOM_ACCESS, num_coherency_transactions / 2);
        join
        
        `uvm_info(get_type_name(), "Coherency pattern tests completed", UVM_MEDIUM)
    endtask
    
    // Run specific coherency pattern test
    virtual task run_pattern_test(coherency_pattern_e pattern, int num_transactions);
        `uvm_info(get_type_name(), $sformatf("Running %s pattern with %0d transactions", 
                 pattern.name(), num_transactions), UVM_HIGH)
        
        case (pattern)
            PRODUCER_CONSUMER: run_producer_consumer_pattern(num_transactions);
            READER_WRITER: run_reader_writer_pattern(num_transactions);
            BARRIER_SYNC: run_barrier_sync_pattern(num_transactions);
            FALSE_SHARING: run_false_sharing_pattern(num_transactions);
            TRUE_SHARING: run_true_sharing_pattern(num_transactions);
            RANDOM_ACCESS: run_random_access_pattern(num_transactions);
        endcase
    endtask
    
    // Producer-Consumer pattern test
    virtual task run_producer_consumer_pattern(int num_transactions);
        int producer_core = 0;
        int consumer_cores[] = '{1, 2, 3};
        int shared_buffer_idx = 0;
        
        fork
            // Producer
            begin
                for (int i = 0; i < num_transactions; i++) begin
                    // Write data to shared buffer
                    write_shared_data(producer_core, shared_buffer_idx, 64'h1000 + i);
                    repeat($urandom_range(10, 50)) @(posedge monitor.clk);
                end
            end
            
            // Consumers
            for (int c = 0; c < consumer_cores.size(); c++) begin
                automatic int consumer_core = consumer_cores[c];
                fork
                    begin
                        for (int i = 0; i < num_transactions / consumer_cores.size(); i++) begin
                            // Read data from shared buffer
                            bit [63:0] data = read_shared_data(consumer_core, shared_buffer_idx);
                            repeat($urandom_range(5, 25)) @(posedge monitor.clk);
                        end
                    end
                join_none
            end
        join
    endtask
    
    // Reader-Writer pattern test
    virtual task run_reader_writer_pattern(int num_transactions);
        int writer_core = 1;
        int reader_cores[] = '{0, 2, 3};
        int shared_var_idx = 1;
        
        fork
            // Writer
            begin
                for (int i = 0; i < num_transactions / 4; i++) begin
                    write_shared_data(writer_core, shared_var_idx, 64'h2000 + i);
                    repeat($urandom_range(50, 200)) @(posedge monitor.clk);
                end
            end
            
            // Readers
            for (int r = 0; r < reader_cores.size(); r++) begin
                automatic int reader_core = reader_cores[r];
                fork
                    begin
                        for (int i = 0; i < num_transactions / 4; i++) begin
                            bit [63:0] data = read_shared_data(reader_core, shared_var_idx);
                            repeat($urandom_range(10, 30)) @(posedge monitor.clk);
                        end
                    end
                join_none
            end
        join
    endtask
    
    // Barrier synchronization pattern test
    virtual task run_barrier_sync_pattern(int num_transactions);
        int barrier_var_idx = 2;
        int num_barriers = num_transactions / 20;
        
        for (int barrier = 0; barrier < num_barriers; barrier++) begin
            `uvm_info(get_type_name(), $sformatf("Barrier synchronization %0d", barrier), UVM_HIGH)
            
            // All cores reach barrier
            fork
                for (int core = 0; core < config.num_cores_active; core++) begin
                    automatic int core_id = core;
                    fork
                        begin
                            // Do some work
                            repeat($urandom_range(100, 500)) @(posedge monitor.clk);
                            
                            // Signal arrival at barrier
                            write_shared_data(core_id, barrier_var_idx, 64'h1);
                            
                            // Wait for all cores to reach barrier
                            while (!all_cores_at_barrier(barrier_var_idx)) begin
                                repeat(10) @(posedge monitor.clk);
                            end
                        end
                    join_none
                end
            join
            
            // Reset barrier
            write_shared_data(0, barrier_var_idx, 64'h0);
        end
    endtask
    
    // Check if all cores have reached the barrier
    virtual function bit all_cores_at_barrier(int barrier_idx);
        for (int core = 0; core < config.num_cores_active; core++) begin
            if (shared_cache_lines[barrier_idx].data[core] == 64'h0) return 0;
        end
        return 1;
    endfunction
    
    // Random access pattern test
    virtual task run_random_access_pattern(int num_transactions);
        fork
            for (int core = 0; core < config.num_cores_active; core++) begin
                automatic int core_id = core;
                fork
                    begin
                        for (int i = 0; i < num_transactions / config.num_cores_active; i++) begin
                            int var_idx = $urandom_range(0, num_shared_variables - 1);
                            
                            if ($urandom_range(0, 1)) begin
                                // Write operation
                                bit [63:0] data = $urandom_range(0, 64'hFFFFFFFF);
                                write_shared_data(core_id, var_idx, data);
                            end else begin
                                // Read operation
                                bit [63:0] data = read_shared_data(core_id, var_idx);
                            end
                            
                            repeat($urandom_range(5, 50)) @(posedge monitor.clk);
                        end
                    end
                join_none
            end
        join
    endtask
    
    // Write shared data with coherency protocol simulation
    virtual task write_shared_data(int core_id, int var_idx, bit [63:0] data);
        `uvm_info(get_type_name(), $sformatf("Core %0d writing 0x%016h to var %0d", 
                 core_id, data, var_idx), UVM_DEBUG)
        
        // Simulate coherency protocol for write
        handle_write_coherency(core_id, var_idx, data);
        
        // Update cache line
        shared_cache_lines[var_idx].data[core_id] = data;
        shared_cache_lines[var_idx].last_writer = core_id;
        shared_cache_lines[var_idx].last_write_time = $time;
        shared_cache_lines[var_idx].write_count[core_id]++;
        
        // Simulate write latency
        repeat($urandom_range(5, 15)) @(posedge monitor.clk);
    endtask
    
    // Read shared data with coherency protocol simulation
    virtual function bit [63:0] read_shared_data(int core_id, int var_idx);
        `uvm_info(get_type_name(), $sformatf("Core %0d reading from var %0d", core_id, var_idx), UVM_DEBUG)
        
        // Simulate coherency protocol for read
        handle_read_coherency(core_id, var_idx);
        
        // Update read count
        shared_cache_lines[var_idx].read_count[core_id]++;
        
        // Return data (from last writer or local cache)
        if (shared_cache_lines[var_idx].last_writer >= 0) begin
            return shared_cache_lines[var_idx].data[shared_cache_lines[var_idx].last_writer];
        end else begin
            return shared_cache_lines[var_idx].data[core_id];
        end
    endfunction
    
    // Handle write coherency protocol
    virtual function void handle_write_coherency(int core_id, int var_idx, bit [63:0] data);
        case (coherency_protocol)
            "MESI": handle_mesi_write(core_id, var_idx, data);
            "MOESI": handle_moesi_write(core_id, var_idx, data);
            "MSI": handle_msi_write(core_id, var_idx, data);
            default: handle_mesi_write(core_id, var_idx, data);
        endcase
    endfunction
    
    // Handle read coherency protocol
    virtual function void handle_read_coherency(int core_id, int var_idx);
        case (coherency_protocol)
            "MESI": handle_mesi_read(core_id, var_idx);
            "MOESI": handle_moesi_read(core_id, var_idx);
            "MSI": handle_msi_read(core_id, var_idx);
            default: handle_mesi_read(core_id, var_idx);
        endcase
    endfunction
    
    // MESI protocol write handling
    virtual function void handle_mesi_write(int core_id, int var_idx, bit [63:0] data);
        // Invalidate other cores' copies
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (i != core_id && shared_cache_lines[var_idx].state[i] != 0) begin
                shared_cache_lines[var_idx].state[i] = 0;  // Invalid
                cache_invalidations++;
                coherency_messages++;
            end
        end
        
        // Set writing core to Modified state
        shared_cache_lines[var_idx].state[core_id] = 3;  // Modified
    endfunction
    
    // MESI protocol read handling
    virtual function void handle_mesi_read(int core_id, int var_idx);
        bit has_modified = 0;
        int modified_core = -1;
        
        // Check if any core has modified copy
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (shared_cache_lines[var_idx].state[i] == 3) begin  // Modified
                has_modified = 1;
                modified_core = i;
                break;
            end
        end
        
        if (has_modified) begin
            // Modified core writes back and goes to Shared
            shared_cache_lines[var_idx].state[modified_core] = 1;  // Shared
            cache_writebacks++;
            coherency_messages++;
        end
        
        // Reading core gets Shared copy
        shared_cache_lines[var_idx].state[core_id] = 1;  // Shared
        
        // If no other cores have copy, reading core gets Exclusive
        bit other_copies = 0;
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (i != core_id && shared_cache_lines[var_idx].state[i] != 0) begin
                other_copies = 1;
                break;
            end
        end
        
        if (!other_copies) begin
            shared_cache_lines[var_idx].state[core_id] = 2;  // Exclusive
        end
    endfunction
    
    // MOESI protocol write handling (simplified)
    virtual function void handle_moesi_write(int core_id, int var_idx, bit [63:0] data);
        // Similar to MESI but with Owner state
        handle_mesi_write(core_id, var_idx, data);
    endfunction
    
    // MOESI protocol read handling (simplified)
    virtual function void handle_moesi_read(int core_id, int var_idx);
        handle_mesi_read(core_id, var_idx);
    endfunction
    
    // MSI protocol write handling
    virtual function void handle_msi_write(int core_id, int var_idx, bit [63:0] data);
        // Invalidate all other copies
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (i != core_id && shared_cache_lines[var_idx].state[i] != 0) begin
                shared_cache_lines[var_idx].state[i] = 0;  // Invalid
                cache_invalidations++;
                coherency_messages++;
            end
        end
        
        // Set writing core to Modified
        shared_cache_lines[var_idx].state[core_id] = 3;  // Modified
    endfunction
    
    // MSI protocol read handling
    virtual function void handle_msi_read(int core_id, int var_idx);
        // All readers get Shared state
        shared_cache_lines[var_idx].state[core_id] = 1;  // Shared
    endfunction
    
    // Test false sharing scenarios
    virtual task test_false_sharing();
        `uvm_info(get_type_name(), "Testing false sharing scenarios", UVM_MEDIUM)
        
        // Create false sharing scenario - different cores accessing different parts of same cache line
        int false_sharing_var = num_shared_variables - 1;
        
        fork
            for (int core = 0; core < config.num_cores_active; core++) begin
                automatic int core_id = core;
                fork
                    begin
                        for (int i = 0; i < 100; i++) begin
                            // Each core writes to different offset in same cache line
                            bit [63:0] data = (64'h1000 << core_id) + i;
                            write_shared_data(core_id, false_sharing_var, data);
                            repeat($urandom_range(10, 30)) @(posedge monitor.clk);
                        end
                    end
                join_none
            end
        join
        
        false_sharing_events += 100 * config.num_cores_active;
        `uvm_info(get_type_name(), "False sharing test completed", UVM_MEDIUM)
    endtask
    
    // Test true sharing scenarios
    virtual task test_true_sharing();
        `uvm_info(get_type_name(), "Testing true sharing scenarios", UVM_MEDIUM)
        
        // Create true sharing scenario - multiple cores accessing same data
        int true_sharing_var = num_shared_variables - 2;
        
        fork
            // One writer
            begin
                for (int i = 0; i < 50; i++) begin
                    write_shared_data(0, true_sharing_var, 64'h5000 + i);
                    repeat($urandom_range(50, 100)) @(posedge monitor.clk);
                end
            end
            
            // Multiple readers
            for (int core = 1; core < config.num_cores_active; core++) begin
                automatic int core_id = core;
                fork
                    begin
                        for (int i = 0; i < 100; i++) begin
                            bit [63:0] data = read_shared_data(core_id, true_sharing_var);
                            repeat($urandom_range(10, 40)) @(posedge monitor.clk);
                        end
                    end
                join_none
            end
        join
        
        true_sharing_events += 50 + (100 * (config.num_cores_active - 1));
        `uvm_info(get_type_name(), "True sharing test completed", UVM_MEDIUM)
    endtask
    
    // Monitor coherency protocol effectiveness
    virtual task monitor_coherency_protocol();
        while (test_running) begin
            repeat(1000) @(posedge monitor.clk);
            
            // Check for coherency violations
            check_coherency_violations();
            
            // Update coherency statistics
            update_coherency_statistics();
        end
    endtask
    
    // Check for coherency violations
    virtual function void check_coherency_violations();
        for (int var_idx = 0; var_idx < num_shared_variables; var_idx++) begin
            // Check for multiple Modified states (violation)
            int modified_count = 0;
            for (int core = 0; core < config.num_cores_active; core++) begin
                if (shared_cache_lines[var_idx].state[core] == 3) modified_count++;
            end
            
            if (modified_count > 1) begin
                error_count++;
                error_messages.push_back($sformatf("Coherency violation: Multiple Modified states for var %0d", var_idx));
                `uvm_error(get_type_name(), $sformatf("Coherency violation detected for variable %0d", var_idx))
            end
            
            // Check data consistency
            check_data_consistency(var_idx);
        end
    endfunction
    
    // Check data consistency across cores
    virtual function void check_data_consistency(int var_idx);
        bit [63:0] reference_data;
        bit reference_set = 0;
        
        // Find reference data from valid cache lines
        for (int core = 0; core < config.num_cores_active; core++) begin
            if (shared_cache_lines[var_idx].state[core] != 0) begin  // Not Invalid
                if (!reference_set) begin
                    reference_data = shared_cache_lines[var_idx].data[core];
                    reference_set = 1;
                end else begin
                    // Check consistency with reference
                    if (shared_cache_lines[var_idx].data[core] != reference_data) begin
                        error_count++;
                        error_messages.push_back($sformatf("Data inconsistency for var %0d: core %0d has different data", var_idx, core));
                        `uvm_error(get_type_name(), $sformatf("Data inconsistency detected for variable %0d", var_idx))
                    end
                end
            end
        end
    endfunction
    
    // Update coherency statistics
    virtual function void update_coherency_statistics();
        // Calculate cache hit rates based on coherency state
        for (int core = 0; core < config.num_cores_active; core++) begin
            int valid_lines = 0;
            for (int var_idx = 0; var_idx < num_shared_variables; var_idx++) begin
                if (shared_cache_lines[var_idx].state[core] != 0) valid_lines++;
            end
            current_state.cache_hit_rates[0] = real'(valid_lines) / real'(num_shared_variables);
        end
    endfunction
    
    // Additional checks specific to cache coherency test
    virtual function void perform_additional_checks();
        super.perform_additional_checks();
        
        // Check coherency message efficiency
        check_coherency_efficiency();
        
        // Check false sharing impact
        check_false_sharing_impact();
        
        // Validate protocol correctness
        validate_protocol_correctness();
    endfunction
    
    // Check coherency protocol efficiency
    virtual function void check_coherency_efficiency();
        if (performance.total_operations > 0) begin
            real messages_per_operation = real'(coherency_messages) / real'(performance.total_operations);
            
            if (messages_per_operation > 2.0) begin
                `uvm_warning(get_type_name(), $sformatf("High coherency overhead: %.2f messages per operation", messages_per_operation))
            end
            
            `uvm_info(get_type_name(), $sformatf("Coherency efficiency: %.2f messages per operation", messages_per_operation), UVM_MEDIUM)
        end
    endfunction
    
    // Check false sharing impact
    virtual function void check_false_sharing_impact();
        if (false_sharing_events > 0) begin
            real false_sharing_ratio = real'(false_sharing_events) / real'(performance.total_operations);
            
            if (false_sharing_ratio > 0.1) begin
                `uvm_warning(get_type_name(), $sformatf("High false sharing: %.2f%% of operations", false_sharing_ratio * 100.0))
            end
            
            `uvm_info(get_type_name(), $sformatf("False sharing ratio: %.2f%%", false_sharing_ratio * 100.0), UVM_MEDIUM)
        end
    endfunction
    
    // Validate protocol correctness
    virtual function void validate_protocol_correctness();
        // Final coherency check
        for (int var_idx = 0; var_idx < num_shared_variables; var_idx++) begin
            check_coherency_violations();
        end
        
        // Check if all coherency messages were handled
        if (coherency_messages != (cache_invalidations + cache_writebacks)) begin
            `uvm_warning(get_type_name(), "Coherency message count mismatch")
        end
    endfunction
    
    // Enhanced reporting for cache coherency test
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== CACHE COHERENCY DETAILS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Coherency Protocol: %s", coherency_protocol), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Shared Variables: %0d", num_shared_variables), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Coherency Transactions: %0d", num_coherency_transactions), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Coherency Messages: %0d", coherency_messages), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Cache Invalidations: %0d", cache_invalidations), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Cache Writebacks: %0d", cache_writebacks), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("False Sharing Events: %0d", false_sharing_events), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("True Sharing Events: %0d", true_sharing_events), UVM_LOW)
        
        // Per-variable statistics
        `uvm_info(get_type_name(), "Per-Variable Access Statistics:", UVM_LOW)
        for (int i = 0; i < num_shared_variables && i < 5; i++) begin  // Show first 5 variables
            int total_reads = 0, total_writes = 0;
            for (int core = 0; core < config.num_cores_active; core++) begin
                total_reads += shared_cache_lines[i].read_count[core];
                total_writes += shared_cache_lines[i].write_count[core];
            end
            `uvm_info(get_type_name(), $sformatf("  Var %0d: %0d reads, %0d writes", i, total_reads, total_writes), UVM_LOW)
        end
    endfunction
    
endclass : cache_coherency_test

`endif // CACHE_COHERENCY_TEST_SV