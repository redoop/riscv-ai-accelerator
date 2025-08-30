// UVM Configuration File
// Global configuration settings for RISC-V AI verification environment

`ifndef UVM_CONFIG_SV
`define UVM_CONFIG_SV

// Global UVM configuration class
class uvm_global_config extends uvm_object;
    
    // Simulation control
    bit enable_uvm_info = 1;
    bit enable_uvm_warning = 1;
    bit enable_uvm_error = 1;
    bit enable_uvm_fatal = 1;
    
    // Timeout settings
    time default_timeout = 10ms;
    time max_test_timeout = 100ms;
    
    // Coverage settings
    bit enable_functional_coverage = 1;
    bit enable_assertion_coverage = 1;
    real coverage_threshold = 95.0;
    
    // Performance settings
    bit enable_performance_monitoring = 1;
    real performance_threshold_tops = 100.0;
    real bandwidth_threshold_gbps = 500.0;
    
    // Debug settings
    bit enable_transaction_recording = 1;
    bit enable_waveform_dumping = 1;
    string waveform_format = "vcd";  // vcd, fsdb, wlf
    
    `uvm_object_utils_begin(uvm_global_config)
        `uvm_field_int(enable_uvm_info, UVM_ALL_ON)
        `uvm_field_int(enable_uvm_warning, UVM_ALL_ON)
        `uvm_field_int(enable_uvm_error, UVM_ALL_ON)
        `uvm_field_int(enable_uvm_fatal, UVM_ALL_ON)
        `uvm_field_int(default_timeout, UVM_ALL_ON)
        `uvm_field_int(max_test_timeout, UVM_ALL_ON)
        `uvm_field_int(enable_functional_coverage, UVM_ALL_ON)
        `uvm_field_int(enable_assertion_coverage, UVM_ALL_ON)
        `uvm_field_real(coverage_threshold, UVM_ALL_ON)
        `uvm_field_int(enable_performance_monitoring, UVM_ALL_ON)
        `uvm_field_real(performance_threshold_tops, UVM_ALL_ON)
        `uvm_field_real(bandwidth_threshold_gbps, UVM_ALL_ON)
        `uvm_field_int(enable_transaction_recording, UVM_ALL_ON)
        `uvm_field_int(enable_waveform_dumping, UVM_ALL_ON)
        `uvm_field_string(waveform_format, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "uvm_global_config");
        super.new(name);
    endfunction
    
    // Apply configuration from command line
    function void apply_cmdline_config();
        uvm_cmdline_processor clp = uvm_cmdline_processor::get_inst();
        string value;
        
        // Check for coverage disable
        if (clp.get_arg_value("+UVM_NO_COVERAGE", value)) begin
            enable_functional_coverage = 0;
            enable_assertion_coverage = 0;
        end
        
        // Check for performance monitoring disable
        if (clp.get_arg_value("+UVM_NO_PERF", value)) begin
            enable_performance_monitoring = 0;
        end
        
        // Check for waveform format
        if (clp.get_arg_value("+UVM_WAVE_FORMAT=", value)) begin
            waveform_format = value;
        end
        
        // Check for timeout override
        if (clp.get_arg_value("+UVM_TIMEOUT=", value)) begin
            default_timeout = value.atoi() * 1ms;
        end
    endfunction
    
endclass : uvm_global_config

// Test-specific configuration factory
class uvm_test_config_factory;
    
    // Create configuration for specific test
    static function uvm_global_config create_config_for_test(string test_name);
        uvm_global_config cfg = uvm_global_config::type_id::create("test_config");
        
        case (test_name)
            "riscv_ai_smoke_test": begin
                cfg.default_timeout = 5ms;
                cfg.coverage_threshold = 50.0;  // Lower threshold for smoke test
            end
            
            "riscv_ai_stress_test": begin
                cfg.default_timeout = 60ms;  // Longer timeout for stress test
                cfg.enable_performance_monitoring = 1;
                cfg.performance_threshold_tops = 200.0;  // Higher performance expectation
            end
            
            "riscv_ai_power_test": begin
                cfg.enable_performance_monitoring = 1;
                cfg.bandwidth_threshold_gbps = 300.0;  // Lower bandwidth for power efficiency
            end
            
            "riscv_ai_error_test": begin
                cfg.coverage_threshold = 80.0;  // Lower coverage due to error injection
                cfg.enable_uvm_error = 0;  // Don't treat injected errors as UVM errors
            end
            
            default: begin
                // Use default configuration
            end
        endcase
        
        // Apply command line overrides
        cfg.apply_cmdline_config();
        
        return cfg;
    endfunction
    
endclass : uvm_test_config_factory

`endif // UVM_CONFIG_SV