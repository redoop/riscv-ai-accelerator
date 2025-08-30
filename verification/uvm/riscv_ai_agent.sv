// RISC-V AI Agent
// UVM agent containing driver, monitor, and sequencer

`ifndef RISCV_AI_AGENT_SV
`define RISCV_AI_AGENT_SV

class riscv_ai_agent extends uvm_agent;
    
    // Agent components
    riscv_ai_driver driver;
    riscv_ai_monitor monitor;
    uvm_sequencer #(riscv_ai_sequence_item) sequencer;
    
    // Agent configuration
    bit is_active = UVM_ACTIVE;
    
    // Analysis ports
    uvm_analysis_port #(riscv_ai_sequence_item) ap_request;
    uvm_analysis_port #(riscv_ai_sequence_item) ap_response;
    
    `uvm_component_utils_begin(riscv_ai_agent)
        `uvm_field_enum(uvm_active_passive_enum, is_active, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_agent", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Create monitor (always present)
        monitor = riscv_ai_monitor::type_id::create("monitor", this);
        
        // Create driver and sequencer only if agent is active
        if (is_active == UVM_ACTIVE) begin
            driver = riscv_ai_driver::type_id::create("driver", this);
            sequencer = uvm_sequencer#(riscv_ai_sequence_item)::type_id::create("sequencer", this);
        end
        
        // Create analysis ports
        ap_request = new("ap_request", this);
        ap_response = new("ap_response", this);
    endfunction
    
    // Connect phase
    virtual function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        
        // Connect monitor analysis ports to agent analysis ports
        monitor.ap_request.connect(ap_request);
        monitor.ap_response.connect(ap_response);
        
        // Connect driver to sequencer if active
        if (is_active == UVM_ACTIVE) begin
            driver.seq_item_port.connect(sequencer.seq_item_export);
        end
    endfunction
    
endclass : riscv_ai_agent

// Multi-agent configuration for multiple interfaces
class riscv_ai_multi_agent extends uvm_agent;
    
    // Multiple agents for different interfaces
    riscv_ai_agent cpu_agents[4];      // One agent per CPU core
    riscv_ai_agent tpu_agents[2];      // One agent per TPU
    riscv_ai_agent vpu_agents[2];      // One agent per VPU
    riscv_ai_agent memory_agent;       // Memory interface agent
    
    // Configuration
    int num_cpu_cores = 4;
    int num_tpus = 2;
    int num_vpus = 2;
    
    `uvm_component_utils_begin(riscv_ai_multi_agent)
        `uvm_field_int(num_cpu_cores, UVM_ALL_ON)
        `uvm_field_int(num_tpus, UVM_ALL_ON)
        `uvm_field_int(num_vpus, UVM_ALL_ON)
    `uvm_component_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_multi_agent", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Create CPU agents
        for (int i = 0; i < num_cpu_cores; i++) begin
            cpu_agents[i] = riscv_ai_agent::type_id::create($sformatf("cpu_agent_%0d", i), this);
        end
        
        // Create TPU agents
        for (int i = 0; i < num_tpus; i++) begin
            tpu_agents[i] = riscv_ai_agent::type_id::create($sformatf("tpu_agent_%0d", i), this);
        end
        
        // Create VPU agents
        for (int i = 0; i < num_vpus; i++) begin
            vpu_agents[i] = riscv_ai_agent::type_id::create($sformatf("vpu_agent_%0d", i), this);
        end
        
        // Create memory agent
        memory_agent = riscv_ai_agent::type_id::create("memory_agent", this);
    endfunction
    
    // Connect phase
    virtual function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        
        // Additional connections can be made here if needed
        // For example, connecting agents to a central scoreboard
    endfunction
    
endclass : riscv_ai_multi_agent

`endif // RISCV_AI_AGENT_SV