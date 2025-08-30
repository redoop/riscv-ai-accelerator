// RISC-V AI Environment
// Top-level UVM environment for AI accelerator verification

`ifndef RISCV_AI_ENV_SV
`define RISCV_AI_ENV_SV

class riscv_ai_env extends uvm_env;
    
    // Environment components
    riscv_ai_agent agent;
    riscv_ai_scoreboard scoreboard;
    riscv_ai_coverage coverage;
    
    // Configuration object
    riscv_ai_env_config cfg;
    
    `uvm_component_utils(riscv_ai_env)
    
    // Constructor
    function new(string name = "riscv_ai_env", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Get configuration
        if (!uvm_config_db#(riscv_ai_env_config)::get(this, "", "cfg", cfg)) begin
            `uvm_info(get_type_name(), "No environment config found, using defaults", UVM_MEDIUM)
            cfg = riscv_ai_env_config::type_id::create("cfg");
        end
        
        // Create components
        agent = riscv_ai_agent::type_id::create("agent", this);
        scoreboard = riscv_ai_scoreboard::type_id::create("scoreboard", this);
        coverage = riscv_ai_coverage::type_id::create("coverage", this);
        
        // Configure components
        uvm_config_db#(riscv_ai_env_config)::set(this, "agent", "cfg", cfg);
        uvm_config_db#(riscv_ai_env_config)::set(this, "scoreboard", "cfg", cfg);
        uvm_config_db#(riscv_ai_env_config)::set(this, "coverage", "cfg", cfg);
    endfunction
    
    // Connect phase
    virtual function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        
        // Connect agent to scoreboard
        agent.ap_request.connect(scoreboard.ap_request);
        agent.ap_response.connect(scoreboard.ap_response);
        
        // Connect agent to coverage
        agent.ap_request.connect(coverage.analysis_export);
    endfunction
    
endclass : riscv_ai_env

// Environment configuration class
class riscv_ai_env_config extends uvm_object;
    
    // Test configuration
    bit enable_scoreboard = 1;
    bit enable_coverage = 1;
    bit enable_protocol_checking = 1;
    bit enable_performance_monitoring = 1;
    bit enable_error_injection = 0;
    
    // Performance thresholds
    real latency_threshold_ns = 1000.0;
    real throughput_threshold_mbps = 1000.0;
    int power_threshold_mw = 5000;
    
    // Coverage goals
    real coverage_goal = 95.0;
    
    // Test parameters
    int num_transactions = 1000;
    int max_outstanding = 16;
    
    `uvm_object_utils_begin(riscv_ai_env_config)
        `uvm_field_int(enable_scoreboard, UVM_ALL_ON)
        `uvm_field_int(enable_coverage, UVM_ALL_ON)
        `uvm_field_int(enable_protocol_checking, UVM_ALL_ON)
        `uvm_field_int(enable_performance_monitoring, UVM_ALL_ON)
        `uvm_field_int(enable_error_injection, UVM_ALL_ON)
        `uvm_field_real(latency_threshold_ns, UVM_ALL_ON)
        `uvm_field_real(throughput_threshold_mbps, UVM_ALL_ON)
        `uvm_field_int(power_threshold_mw, UVM_ALL_ON)
        `uvm_field_real(coverage_goal, UVM_ALL_ON)
        `uvm_field_int(num_transactions, UVM_ALL_ON)
        `uvm_field_int(max_outstanding, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "riscv_ai_env_config");
        super.new(name);
    endfunction
    
endclass : riscv_ai_env_config

// Multi-core environment for comprehensive testing
class riscv_ai_multi_env extends uvm_env;
    
    // Multiple environments for different subsystems
    riscv_ai_env cpu_envs[4];      // CPU core environments
    riscv_ai_env tpu_envs[2];      // TPU environments
    riscv_ai_env vpu_envs[2];      // VPU environments
    riscv_ai_env memory_env;       // Memory subsystem environment
    
    // Central scoreboard for cross-subsystem checking
    riscv_ai_multi_scoreboard multi_scoreboard;
    
    // System-level coverage
    riscv_ai_system_coverage system_coverage;
    
    // Configuration
    riscv_ai_multi_env_config cfg;
    
    `uvm_component_utils(riscv_ai_multi_env)
    
    // Constructor
    function new(string name = "riscv_ai_multi_env", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Get configuration
        if (!uvm_config_db#(riscv_ai_multi_env_config)::get(this, "", "cfg", cfg)) begin
            cfg = riscv_ai_multi_env_config::type_id::create("cfg");
        end
        
        // Create CPU environments
        for (int i = 0; i < cfg.num_cpu_cores; i++) begin
            cpu_envs[i] = riscv_ai_env::type_id::create($sformatf("cpu_env_%0d", i), this);
        end
        
        // Create TPU environments
        for (int i = 0; i < cfg.num_tpus; i++) begin
            tpu_envs[i] = riscv_ai_env::type_id::create($sformatf("tpu_env_%0d", i), this);
        end
        
        // Create VPU environments
        for (int i = 0; i < cfg.num_vpus; i++) begin
            vpu_envs[i] = riscv_ai_env::type_id::create($sformatf("vpu_env_%0d", i), this);
        end
        
        // Create memory environment
        memory_env = riscv_ai_env::type_id::create("memory_env", this);
        
        // Create system-level components
        multi_scoreboard = riscv_ai_multi_scoreboard::type_id::create("multi_scoreboard", this);
        system_coverage = riscv_ai_system_coverage::type_id::create("system_coverage", this);
    endfunction
    
    // Connect phase
    virtual function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        
        // Connect all environments to multi-scoreboard
        for (int i = 0; i < cfg.num_cpu_cores; i++) begin
            cpu_envs[i].agent.ap_request.connect(multi_scoreboard.cpu_request_exports[i]);
            cpu_envs[i].agent.ap_response.connect(multi_scoreboard.cpu_response_exports[i]);
        end
        
        for (int i = 0; i < cfg.num_tpus; i++) begin
            tpu_envs[i].agent.ap_request.connect(multi_scoreboard.tpu_request_exports[i]);
            tpu_envs[i].agent.ap_response.connect(multi_scoreboard.tpu_response_exports[i]);
        end
        
        for (int i = 0; i < cfg.num_vpus; i++) begin
            vpu_envs[i].agent.ap_request.connect(multi_scoreboard.vpu_request_exports[i]);
            vpu_envs[i].agent.ap_response.connect(multi_scoreboard.vpu_response_exports[i]);
        end
        
        memory_env.agent.ap_request.connect(multi_scoreboard.memory_request_export);
        memory_env.agent.ap_response.connect(multi_scoreboard.memory_response_export);
        
        // Connect to system coverage
        multi_scoreboard.system_transaction_port.connect(system_coverage.analysis_export);
    endfunction
    
endclass : riscv_ai_multi_env

// Multi-environment configuration
class riscv_ai_multi_env_config extends uvm_object;
    
    // System configuration
    int num_cpu_cores = 4;
    int num_tpus = 2;
    int num_vpus = 2;
    
    // Test configuration
    bit enable_cross_subsystem_checking = 1;
    bit enable_system_coverage = 1;
    bit enable_coherency_checking = 1;
    bit enable_power_analysis = 1;
    
    // Individual environment configurations
    riscv_ai_env_config cpu_configs[4];
    riscv_ai_env_config tpu_configs[2];
    riscv_ai_env_config vpu_configs[2];
    riscv_ai_env_config memory_config;
    
    `uvm_object_utils_begin(riscv_ai_multi_env_config)
        `uvm_field_int(num_cpu_cores, UVM_ALL_ON)
        `uvm_field_int(num_tpus, UVM_ALL_ON)
        `uvm_field_int(num_vpus, UVM_ALL_ON)
        `uvm_field_int(enable_cross_subsystem_checking, UVM_ALL_ON)
        `uvm_field_int(enable_system_coverage, UVM_ALL_ON)
        `uvm_field_int(enable_coherency_checking, UVM_ALL_ON)
        `uvm_field_int(enable_power_analysis, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "riscv_ai_multi_env_config");
        super.new(name);
        
        // Create individual configurations
        for (int i = 0; i < num_cpu_cores; i++) begin
            cpu_configs[i] = riscv_ai_env_config::type_id::create($sformatf("cpu_config_%0d", i));
        end
        
        for (int i = 0; i < num_tpus; i++) begin
            tpu_configs[i] = riscv_ai_env_config::type_id::create($sformatf("tpu_config_%0d", i));
        end
        
        for (int i = 0; i < num_vpus; i++) begin
            vpu_configs[i] = riscv_ai_env_config::type_id::create($sformatf("vpu_config_%0d", i));
        end
        
        memory_config = riscv_ai_env_config::type_id::create("memory_config");
    endfunction
    
endclass : riscv_ai_multi_env_config

// Placeholder classes for multi-environment components
class riscv_ai_multi_scoreboard extends uvm_scoreboard;
    
    // Analysis imports for different subsystems
    uvm_analysis_imp_cpu_request #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) cpu_request_exports[4];
    uvm_analysis_imp_cpu_response #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) cpu_response_exports[4];
    uvm_analysis_imp_tpu_request #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) tpu_request_exports[2];
    uvm_analysis_imp_tpu_response #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) tpu_response_exports[2];
    uvm_analysis_imp_vpu_request #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) vpu_request_exports[2];
    uvm_analysis_imp_vpu_response #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) vpu_response_exports[2];
    uvm_analysis_imp_memory_request #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) memory_request_export;
    uvm_analysis_imp_memory_response #(riscv_ai_sequence_item, riscv_ai_multi_scoreboard) memory_response_export;
    
    // System transaction port
    uvm_analysis_port #(riscv_ai_sequence_item) system_transaction_port;
    
    `uvm_component_utils(riscv_ai_multi_scoreboard)
    
    function new(string name = "riscv_ai_multi_scoreboard", uvm_component parent = null);
        super.new(name, parent);
        
        // Create analysis imports
        for (int i = 0; i < 4; i++) begin
            cpu_request_exports[i] = new($sformatf("cpu_request_export_%0d", i), this);
            cpu_response_exports[i] = new($sformatf("cpu_response_export_%0d", i), this);
        end
        
        for (int i = 0; i < 2; i++) begin
            tpu_request_exports[i] = new($sformatf("tpu_request_export_%0d", i), this);
            tpu_response_exports[i] = new($sformatf("tpu_response_export_%0d", i), this);
            vpu_request_exports[i] = new($sformatf("vpu_request_export_%0d", i), this);
            vpu_response_exports[i] = new($sformatf("vpu_response_export_%0d", i), this);
        end
        
        memory_request_export = new("memory_request_export", this);
        memory_response_export = new("memory_response_export", this);
        
        system_transaction_port = new("system_transaction_port", this);
    endfunction
    
    // Write methods for different subsystems
    virtual function void write_cpu_request(riscv_ai_sequence_item req);
        // Cross-subsystem checking logic
    endfunction
    
    virtual function void write_cpu_response(riscv_ai_sequence_item resp);
        // Cross-subsystem checking logic
    endfunction
    
    // Additional write methods would be implemented similarly...
    
endclass : riscv_ai_multi_scoreboard

class riscv_ai_system_coverage extends uvm_subscriber #(riscv_ai_sequence_item);
    
    `uvm_component_utils(riscv_ai_system_coverage)
    
    function new(string name = "riscv_ai_system_coverage", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    virtual function void write(riscv_ai_sequence_item t);
        // System-level coverage collection
    endfunction
    
endclass : riscv_ai_system_coverage

`endif // RISCV_AI_ENV_SV