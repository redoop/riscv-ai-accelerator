// RISC-V AI Base Test
// Base test class for all AI accelerator tests

`ifndef RISCV_AI_BASE_TEST_SV
`define RISCV_AI_BASE_TEST_SV

class riscv_ai_base_test extends uvm_test;
    
    // Test environment
    riscv_ai_env env;
    riscv_ai_env_config env_cfg;
    
    // Virtual interface
    virtual riscv_ai_interface vif;
    
    // Test configuration
    int num_transactions = 1000;
    int test_timeout_ns = 10000000;  // 10ms timeout
    
    `uvm_component_utils_begin(riscv_ai_base_test)
        `uvm_field_int(num_transactions, UVM_ALL_ON)
        `uvm_field_int(test_timeout_ns, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Create environment configuration
        env_cfg = riscv_ai_env_config::type_id::create("env_cfg");
        configure_env(env_cfg);
        
        // Set configuration in config DB
        uvm_config_db#(riscv_ai_env_config)::set(this, "*", "cfg", env_cfg);
        
        // Get virtual interface
        if (!uvm_config_db#(virtual riscv_ai_interface)::get(this, "", "vif", vif)) begin
            `uvm_fatal(get_type_name(), "Virtual interface not found in config DB")
        end
        
        // Set interface in config DB for other components
        uvm_config_db#(virtual riscv_ai_interface)::set(this, "*", "vif", vif);
        
        // Create environment
        env = riscv_ai_env::type_id::create("env", this);
        
        // Set test timeout
        uvm_top.set_timeout(test_timeout_ns, 0);
    endfunction
    
    // Configure environment - to be overridden by derived tests
    virtual function void configure_env(riscv_ai_env_config cfg);
        cfg.num_transactions = num_transactions;
        cfg.enable_scoreboard = 1;
        cfg.enable_coverage = 1;
        cfg.enable_protocol_checking = 1;
        cfg.enable_performance_monitoring = 1;
        cfg.enable_error_injection = 0;
    endfunction
    
    // End of elaboration phase
    virtual function void end_of_elaboration_phase(uvm_phase phase);
        super.end_of_elaboration_phase(phase);
        
        // Print test configuration
        `uvm_info(get_type_name(), $sformatf("Test Configuration:"), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Number of transactions: %0d", num_transactions), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Test timeout: %0d ns", test_timeout_ns), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Scoreboard enabled: %0b", env_cfg.enable_scoreboard), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Coverage enabled: %0b", env_cfg.enable_coverage), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Protocol checking enabled: %0b", env_cfg.enable_protocol_checking), UVM_LOW)
        
        // Print topology
        uvm_top.print_topology();
    endfunction
    
    // Run phase
    virtual task run_phase(uvm_phase phase);
        phase.raise_objection(this, "Starting test");
        
        `uvm_info(get_type_name(), "Test started", UVM_LOW)
        
        // Wait for reset deassertion
        wait(vif.rst_n);
        repeat(10) @(posedge vif.clk);
        
        // Run the main test
        run_test();
        
        // Wait for all transactions to complete
        wait_for_completion();
        
        `uvm_info(get_type_name(), "Test completed", UVM_LOW)
        
        phase.drop_objection(this, "Test completed");
    endtask
    
    // Main test task - to be overridden by derived tests
    virtual task run_test();
        `uvm_info(get_type_name(), "Base test run_test() called - should be overridden", UVM_MEDIUM)
        repeat(100) @(posedge vif.clk);
    endtask
    
    // Wait for test completion
    virtual task wait_for_completion();
        // Wait for all outstanding transactions to complete
        repeat(1000) @(posedge vif.clk);
        
        // Additional completion criteria can be added here
    endtask
    
    // Check phase
    virtual function void check_phase(uvm_phase phase);
        super.check_phase(phase);
        
        // Check test results
        if (env.scoreboard.total_mismatches > 0) begin
            `uvm_error(get_type_name(), $sformatf("Test failed with %0d mismatches", env.scoreboard.total_mismatches))
        end
        
        // Check coverage goals
        if (env_cfg.enable_coverage) begin
            real coverage_pct = env.coverage.total_coverage;
            if (coverage_pct < env_cfg.coverage_goal) begin
                `uvm_warning(get_type_name(), $sformatf("Coverage goal not met: %.2f%% < %.2f%%", 
                           coverage_pct, env_cfg.coverage_goal))
            end
        end
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== TEST SUMMARY ===", UVM_LOW)
        
        // Report scoreboard results
        if (env_cfg.enable_scoreboard) begin
            `uvm_info(get_type_name(), $sformatf("Scoreboard: %0d matches, %0d mismatches, %0d errors", 
                     env.scoreboard.total_matches, env.scoreboard.total_mismatches, env.scoreboard.total_errors), UVM_LOW)
        end
        
        // Report coverage results
        if (env_cfg.enable_coverage) begin
            `uvm_info(get_type_name(), $sformatf("Coverage: %s", env.coverage.get_coverage_summary()), UVM_LOW)
        end
        
        // Determine test result
        if (env.scoreboard.total_mismatches == 0 && 
            (!env_cfg.enable_coverage || env.coverage.total_coverage >= env_cfg.coverage_goal)) begin
            `uvm_info(get_type_name(), "*** TEST PASSED ***", UVM_LOW)
        end else begin
            `uvm_error(get_type_name(), "*** TEST FAILED ***")
        end
    endfunction
    
endclass : riscv_ai_base_test

// Smoke test - basic functionality verification
class riscv_ai_smoke_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_smoke_test)
    
    function new(string name = "riscv_ai_smoke_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 100;  // Smaller test
    endfunction
    
    virtual function void configure_env(riscv_ai_env_config cfg);
        super.configure_env(cfg);
        cfg.coverage_goal = 50.0;  // Lower coverage goal for smoke test
    endfunction
    
    virtual task run_test();
        riscv_ai_random_sequence seq;
        
        `uvm_info(get_type_name(), "Running smoke test", UVM_LOW)
        
        seq = riscv_ai_random_sequence::type_id::create("smoke_seq");
        seq.num_transactions = 50;
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_smoke_test

// Random test - comprehensive randomized testing
class riscv_ai_random_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_random_test)
    
    function new(string name = "riscv_ai_random_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 5000;
    endfunction
    
    virtual task run_test();
        riscv_ai_random_sequence seq;
        
        `uvm_info(get_type_name(), "Running random test", UVM_LOW)
        
        seq = riscv_ai_random_sequence::type_id::create("random_seq");
        seq.num_transactions = num_transactions;
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_random_test

// Matrix multiplication focused test
class riscv_ai_matmul_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_matmul_test)
    
    function new(string name = "riscv_ai_matmul_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 1000;
    endfunction
    
    virtual task run_test();
        riscv_ai_matmul_sequence seq;
        
        `uvm_info(get_type_name(), "Running matrix multiplication test", UVM_LOW)
        
        seq = riscv_ai_matmul_sequence::type_id::create("matmul_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_matmul_test

// Convolution focused test
class riscv_ai_conv2d_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_conv2d_test)
    
    function new(string name = "riscv_ai_conv2d_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 1000;
    endfunction
    
    virtual task run_test();
        riscv_ai_conv2d_sequence seq;
        
        `uvm_info(get_type_name(), "Running convolution test", UVM_LOW)
        
        seq = riscv_ai_conv2d_sequence::type_id::create("conv2d_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_conv2d_test

// Activation function test
class riscv_ai_activation_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_activation_test)
    
    function new(string name = "riscv_ai_activation_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 500;
    endfunction
    
    virtual task run_test();
        riscv_ai_activation_sequence seq;
        
        `uvm_info(get_type_name(), "Running activation function test", UVM_LOW)
        
        seq = riscv_ai_activation_sequence::type_id::create("activation_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_activation_test

// Memory access test
class riscv_ai_memory_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_memory_test)
    
    function new(string name = "riscv_ai_memory_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 2000;
    endfunction
    
    virtual task run_test();
        riscv_ai_memory_sequence seq;
        
        `uvm_info(get_type_name(), "Running memory access test", UVM_LOW)
        
        seq = riscv_ai_memory_sequence::type_id::create("memory_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_memory_test

// Stress test - high load testing
class riscv_ai_stress_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_stress_test)
    
    function new(string name = "riscv_ai_stress_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 10000;
        test_timeout_ns = 100000000;  // 100ms timeout for stress test
    endfunction
    
    virtual function void configure_env(riscv_ai_env_config cfg);
        super.configure_env(cfg);
        cfg.max_outstanding = 32;  // Allow more outstanding transactions
    endfunction
    
    virtual task run_test();
        riscv_ai_stress_sequence seq;
        
        `uvm_info(get_type_name(), "Running stress test", UVM_LOW)
        
        seq = riscv_ai_stress_sequence::type_id::create("stress_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_stress_test

// Power test - power consumption focused testing
class riscv_ai_power_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_power_test)
    
    function new(string name = "riscv_ai_power_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 2000;
    endfunction
    
    virtual function void configure_env(riscv_ai_env_config cfg);
        super.configure_env(cfg);
        cfg.enable_power_analysis = 1;
        cfg.power_threshold_mw = 3000;  // 3W power threshold
    endfunction
    
    virtual task run_test();
        riscv_ai_power_sequence seq;
        
        `uvm_info(get_type_name(), "Running power consumption test", UVM_LOW)
        
        seq = riscv_ai_power_sequence::type_id::create("power_seq");
        
        seq.start(env.agent.sequencer);
    endtask
    
endclass : riscv_ai_power_test

// Error injection test
class riscv_ai_error_test extends riscv_ai_base_test;
    
    `uvm_component_utils(riscv_ai_error_test)
    
    function new(string name = "riscv_ai_error_test", uvm_component parent = null);
        super.new(name, parent);
        num_transactions = 1000;
    endfunction
    
    virtual function void configure_env(riscv_ai_env_config cfg);
        super.configure_env(cfg);
        cfg.enable_error_injection = 1;
        cfg.coverage_goal = 80.0;  // Lower coverage goal due to errors
    endfunction
    
    virtual task run_test();
        riscv_ai_random_sequence seq;
        
        `uvm_info(get_type_name(), "Running error injection test", UVM_LOW)
        
        seq = riscv_ai_random_sequence::type_id::create("error_seq");
        seq.num_transactions = num_transactions;
        
        seq.start(env.agent.sequencer);
    endtask
    
    virtual function void check_phase(uvm_phase phase);
        // Override check phase to allow errors in this test
        if (env.scoreboard.total_errors == 0) begin
            `uvm_warning(get_type_name(), "No errors detected in error injection test")
        end
        
        // Don't fail test due to mismatches in error injection test
        if (env_cfg.enable_coverage) begin
            real coverage_pct = env.coverage.total_coverage;
            if (coverage_pct < env_cfg.coverage_goal) begin
                `uvm_warning(get_type_name(), $sformatf("Coverage goal not met: %.2f%% < %.2f%%", 
                           coverage_pct, env_cfg.coverage_goal))
            end
        end
    endfunction
    
endclass : riscv_ai_error_test

`endif // RISCV_AI_BASE_TEST_SV