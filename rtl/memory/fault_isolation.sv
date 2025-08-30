/**
 * Fault Isolation and Graceful Degradation Controller
 * Provides system-level fault isolation and performance degradation strategies
 */

module fault_isolation #(
    parameter NUM_CORES = 4,
    parameter NUM_TPUS = 2,
    parameter NUM_VPUS = 2,
    parameter NUM_MEMORY_BANKS = 4,
    parameter NUM_NOC_ROUTERS = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Fault detection inputs
    input  logic [NUM_CORES-1:0]    core_fault,
    input  logic [NUM_TPUS-1:0]     tpu_fault,
    input  logic [NUM_VPUS-1:0]     vpu_fault,
    input  logic [NUM_MEMORY_BANKS-1:0] memory_bank_fault,
    input  logic [NUM_NOC_ROUTERS-1:0] noc_router_fault,
    input  logic                    power_fault,
    input  logic                    thermal_fault,
    input  logic                    clock_fault,
    
    // Fault severity inputs
    input  logic [2:0]              fault_severity [NUM_CORES + NUM_TPUS + NUM_VPUS + NUM_MEMORY_BANKS + NUM_NOC_ROUTERS + 3],
    
    // Isolation control outputs
    output logic [NUM_CORES-1:0]    core_isolated,
    output logic [NUM_TPUS-1:0]     tpu_isolated,
    output logic [NUM_VPUS-1:0]     vpu_isolated,
    output logic [NUM_MEMORY_BANKS-1:0] memory_bank_isolated,
    output logic [NUM_NOC_ROUTERS-1:0] noc_router_isolated,
    
    // Performance degradation outputs
    output logic [NUM_CORES-1:0]    core_freq_reduce,
    output logic [NUM_TPUS-1:0]     tpu_freq_reduce,
    output logic [NUM_VPUS-1:0]     vpu_freq_reduce,
    output logic [1:0]              memory_bandwidth_limit,
    output logic [1:0]              noc_bandwidth_limit,
    
    // Power management outputs
    output logic [NUM_CORES-1:0]    core_power_gate,
    output logic [NUM_TPUS-1:0]     tpu_power_gate,
    output logic [NUM_VPUS-1:0]     vpu_power_gate,
    output logic [NUM_MEMORY_BANKS-1:0] memory_power_gate,
    
    // System status
    output logic [7:0]              system_health,
    output logic [3:0]              performance_level,
    output logic [15:0]             isolated_units,
    output logic                    emergency_mode,
    output logic                    safe_mode,
    
    // Configuration
    input  logic                    isolation_enable,
    input  logic                    degradation_enable,
    input  logic [2:0]              isolation_threshold,
    input  logic [2:0]              degradation_threshold,
    input  logic                    auto_recovery_enable,
    input  logic [15:0]             recovery_timeout
);

    // Degradation levels
    typedef enum logic [3:0] {
        PERF_FULL       = 4'b0000,  // 100% performance
        PERF_HIGH       = 4'b0001,  // 90% performance
        PERF_MEDIUM     = 4'b0010,  // 75% performance
        PERF_LOW        = 4'b0011,  // 50% performance
        PERF_MINIMAL    = 4'b0100,  // 25% performance
        PERF_EMERGENCY  = 4'b0101,  // 10% performance
        PERF_SAFE       = 4'b0110   // 5% performance
    } performance_level_t;

    // System modes
    typedef enum logic [2:0] {
        MODE_NORMAL     = 3'b000,
        MODE_DEGRADED   = 3'b001,
        MODE_ISOLATED   = 3'b010,
        MODE_RECOVERY   = 3'b011,
        MODE_EMERGENCY  = 3'b100,
        MODE_SAFE       = 3'b101,
        MODE_SHUTDOWN   = 3'b110
    } system_mode_t;

    system_mode_t current_mode, next_mode;
    
    // Internal registers
    logic [NUM_CORES-1:0] core_isolated_reg;
    logic [NUM_TPUS-1:0] tpu_isolated_reg;
    logic [NUM_VPUS-1:0] vpu_isolated_reg;
    logic [NUM_MEMORY_BANKS-1:0] memory_isolated_reg;
    logic [NUM_NOC_ROUTERS-1:0] noc_isolated_reg;
    
    logic [NUM_CORES-1:0] core_degraded;
    logic [NUM_TPUS-1:0] tpu_degraded;
    logic [NUM_VPUS-1:0] vpu_degraded;
    
    logic [15:0] recovery_timer;
    logic [7:0] health_score;
    performance_level_t current_perf_level;
    
    // Fault analysis
    logic [7:0] total_faults;
    logic [7:0] critical_faults;
    logic [7:0] active_cores;
    logic [7:0] active_tpus;
    logic [7:0] active_vpus;
    logic [7:0] active_memory_banks;
    logic [7:0] active_noc_routers;
    
    // Count total and critical faults
    always_comb begin
        total_faults = 0;
        critical_faults = 0;
        
        // Count core faults
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_fault[i]) begin
                total_faults = total_faults + 1;
                if (fault_severity[i] >= 3'b100) begin // Critical or fatal
                    critical_faults = critical_faults + 1;
                end
            end
        end
        
        // Count TPU faults
        for (int i = 0; i < NUM_TPUS; i++) begin
            if (tpu_fault[i]) begin
                total_faults = total_faults + 1;
                if (fault_severity[NUM_CORES + i] >= 3'b100) begin
                    critical_faults = critical_faults + 1;
                end
            end
        end
        
        // Count VPU faults
        for (int i = 0; i < NUM_VPUS; i++) begin
            if (vpu_fault[i]) begin
                total_faults = total_faults + 1;
                if (fault_severity[NUM_CORES + NUM_TPUS + i] >= 3'b100) begin
                    critical_faults = critical_faults + 1;
                end
            end
        end
        
        // Count memory bank faults
        for (int i = 0; i < NUM_MEMORY_BANKS; i++) begin
            if (memory_bank_fault[i]) begin
                total_faults = total_faults + 1;
                if (fault_severity[NUM_CORES + NUM_TPUS + NUM_VPUS + i] >= 3'b100) begin
                    critical_faults = critical_faults + 1;
                end
            end
        end
        
        // Count NoC router faults
        for (int i = 0; i < NUM_NOC_ROUTERS; i++) begin
            if (noc_router_fault[i]) begin
                total_faults = total_faults + 1;
                if (fault_severity[NUM_CORES + NUM_TPUS + NUM_VPUS + NUM_MEMORY_BANKS + i] >= 3'b100) begin
                    critical_faults = critical_faults + 1;
                end
            end
        end
        
        // System-level faults
        if (power_fault) begin
            total_faults = total_faults + 1;
            critical_faults = critical_faults + 1; // Power faults are always critical
        end
        
        if (thermal_fault) begin
            total_faults = total_faults + 1;
            critical_faults = critical_faults + 1; // Thermal faults are always critical
        end
        
        if (clock_fault) begin
            total_faults = total_faults + 1;
            critical_faults = critical_faults + 1; // Clock faults are always critical
        end
    end
    
    // Count active (non-isolated) units
    always_comb begin
        active_cores = NUM_CORES - $countones(core_isolated_reg);
        active_tpus = NUM_TPUS - $countones(tpu_isolated_reg);
        active_vpus = NUM_VPUS - $countones(vpu_isolated_reg);
        active_memory_banks = NUM_MEMORY_BANKS - $countones(memory_isolated_reg);
        active_noc_routers = NUM_NOC_ROUTERS - $countones(noc_isolated_reg);
    end
    
    // System health calculation
    always_comb begin
        health_score = 100;
        
        // Reduce health based on isolated units
        health_score = health_score - ($countones(core_isolated_reg) * 20 / NUM_CORES);
        health_score = health_score - ($countones(tpu_isolated_reg) * 15 / NUM_TPUS);
        health_score = health_score - ($countones(vpu_isolated_reg) * 15 / NUM_VPUS);
        health_score = health_score - ($countones(memory_isolated_reg) * 25 / NUM_MEMORY_BANKS);
        health_score = health_score - ($countones(noc_isolated_reg) * 10 / NUM_NOC_ROUTERS);
        
        // Additional penalties for system-level faults
        if (power_fault) health_score = health_score - 20;
        if (thermal_fault) health_score = health_score - 15;
        if (clock_fault) health_score = health_score - 10;
        
        // Ensure health score doesn't go below 0
        if (health_score > 100) health_score = 0; // Handle underflow
    end
    
    // Performance level determination
    always_comb begin
        if (health_score >= 90) begin
            current_perf_level = PERF_FULL;
        end else if (health_score >= 75) begin
            current_perf_level = PERF_HIGH;
        end else if (health_score >= 50) begin
            current_perf_level = PERF_MEDIUM;
        end else if (health_score >= 25) begin
            current_perf_level = PERF_LOW;
        end else if (health_score >= 10) begin
            current_perf_level = PERF_MINIMAL;
        end else if (health_score >= 5) begin
            current_perf_level = PERF_EMERGENCY;
        end else begin
            current_perf_level = PERF_SAFE;
        end
    end
    
    // System mode state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_mode <= MODE_NORMAL;
        end else begin
            current_mode <= next_mode;
        end
    end
    
    // Next mode logic
    always_comb begin
        next_mode = current_mode;
        
        case (current_mode)
            MODE_NORMAL: begin
                if (critical_faults > 0) begin
                    next_mode = MODE_EMERGENCY;
                end else if (total_faults > 0 && isolation_enable) begin
                    next_mode = MODE_ISOLATED;
                end else if (health_score < 75 && degradation_enable) begin
                    next_mode = MODE_DEGRADED;
                end
            end
            
            MODE_DEGRADED: begin
                if (critical_faults > 0) begin
                    next_mode = MODE_EMERGENCY;
                end else if (total_faults > 0 && isolation_enable) begin
                    next_mode = MODE_ISOLATED;
                end else if (health_score >= 90) begin
                    next_mode = MODE_NORMAL;
                end
            end
            
            MODE_ISOLATED: begin
                if (critical_faults > 0) begin
                    next_mode = MODE_EMERGENCY;
                end else if (auto_recovery_enable && recovery_timer >= recovery_timeout) begin
                    next_mode = MODE_RECOVERY;
                end else if (health_score >= 90 && total_faults == 0) begin
                    next_mode = MODE_NORMAL;
                end
            end
            
            MODE_RECOVERY: begin
                if (critical_faults > 0) begin
                    next_mode = MODE_EMERGENCY;
                end else if (health_score >= 75) begin
                    next_mode = MODE_NORMAL;
                end else begin
                    next_mode = MODE_ISOLATED;
                end
            end
            
            MODE_EMERGENCY: begin
                if (health_score < 10) begin
                    next_mode = MODE_SAFE;
                end else if (critical_faults == 0 && health_score >= 25) begin
                    next_mode = MODE_ISOLATED;
                end
            end
            
            MODE_SAFE: begin
                if (health_score >= 25 && critical_faults == 0) begin
                    next_mode = MODE_EMERGENCY;
                end else if (health_score < 5) begin
                    next_mode = MODE_SHUTDOWN;
                end
            end
            
            MODE_SHUTDOWN: begin
                // Stay in shutdown mode until manual reset
            end
            
            default: begin
                next_mode = MODE_NORMAL;
            end
        endcase
    end
    
    // Recovery timer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            recovery_timer <= 16'b0;
        end else if (current_mode == MODE_ISOLATED) begin
            recovery_timer <= recovery_timer + 1;
        end else begin
            recovery_timer <= 16'b0;
        end
    end
    
    // Isolation logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            core_isolated_reg <= '0;
            tpu_isolated_reg <= '0;
            vpu_isolated_reg <= '0;
            memory_isolated_reg <= '0;
            noc_isolated_reg <= '0;
        end else if (isolation_enable) begin
            // Isolate cores with faults above threshold
            for (int i = 0; i < NUM_CORES; i++) begin
                if (core_fault[i] && fault_severity[i] >= isolation_threshold) begin
                    core_isolated_reg[i] <= 1'b1;
                end else if (current_mode == MODE_RECOVERY) begin
                    core_isolated_reg[i] <= 1'b0; // Try to recover
                end
            end
            
            // Isolate TPUs with faults above threshold
            for (int i = 0; i < NUM_TPUS; i++) begin
                if (tpu_fault[i] && fault_severity[NUM_CORES + i] >= isolation_threshold) begin
                    tpu_isolated_reg[i] <= 1'b1;
                end else if (current_mode == MODE_RECOVERY) begin
                    tpu_isolated_reg[i] <= 1'b0;
                end
            end
            
            // Isolate VPUs with faults above threshold
            for (int i = 0; i < NUM_VPUS; i++) begin
                if (vpu_fault[i] && fault_severity[NUM_CORES + NUM_TPUS + i] >= isolation_threshold) begin
                    vpu_isolated_reg[i] <= 1'b1;
                end else if (current_mode == MODE_RECOVERY) begin
                    vpu_isolated_reg[i] <= 1'b0;
                end
            end
            
            // Isolate memory banks with faults above threshold
            for (int i = 0; i < NUM_MEMORY_BANKS; i++) begin
                if (memory_bank_fault[i] && fault_severity[NUM_CORES + NUM_TPUS + NUM_VPUS + i] >= isolation_threshold) begin
                    memory_isolated_reg[i] <= 1'b1;
                end else if (current_mode == MODE_RECOVERY) begin
                    memory_isolated_reg[i] <= 1'b0;
                end
            end
            
            // Isolate NoC routers with faults above threshold
            for (int i = 0; i < NUM_NOC_ROUTERS; i++) begin
                if (noc_router_fault[i] && fault_severity[NUM_CORES + NUM_TPUS + NUM_VPUS + NUM_MEMORY_BANKS + i] >= isolation_threshold) begin
                    noc_isolated_reg[i] <= 1'b1;
                end else if (current_mode == MODE_RECOVERY) begin
                    noc_isolated_reg[i] <= 1'b0;
                end
            end
        end
    end
    
    // Performance degradation logic
    always_comb begin
        core_degraded = '0;
        tpu_degraded = '0;
        vpu_degraded = '0;
        
        if (degradation_enable) begin
            // Degrade units with faults above degradation threshold but below isolation threshold
            for (int i = 0; i < NUM_CORES; i++) begin
                if (core_fault[i] && fault_severity[i] >= degradation_threshold && fault_severity[i] < isolation_threshold) begin
                    core_degraded[i] = 1'b1;
                end
            end
            
            for (int i = 0; i < NUM_TPUS; i++) begin
                if (tpu_fault[i] && fault_severity[NUM_CORES + i] >= degradation_threshold && fault_severity[NUM_CORES + i] < isolation_threshold) begin
                    tpu_degraded[i] = 1'b1;
                end
            end
            
            for (int i = 0; i < NUM_VPUS; i++) begin
                if (vpu_fault[i] && fault_severity[NUM_CORES + NUM_TPUS + i] >= degradation_threshold && fault_severity[NUM_CORES + NUM_TPUS + i] < isolation_threshold) begin
                    vpu_degraded[i] = 1'b1;
                end
            end
        end
    end
    
    // Output assignments
    assign core_isolated = core_isolated_reg;
    assign tpu_isolated = tpu_isolated_reg;
    assign vpu_isolated = vpu_isolated_reg;
    assign memory_bank_isolated = memory_isolated_reg;
    assign noc_router_isolated = noc_isolated_reg;
    
    assign core_freq_reduce = core_degraded | core_isolated_reg;
    assign tpu_freq_reduce = tpu_degraded | tpu_isolated_reg;
    assign vpu_freq_reduce = vpu_degraded | vpu_isolated_reg;
    
    // Bandwidth limiting based on performance level
    assign memory_bandwidth_limit = (current_perf_level >= PERF_MEDIUM) ? 2'b00 : // No limit
                                   (current_perf_level >= PERF_LOW) ? 2'b01 :     // 75% limit
                                   (current_perf_level >= PERF_MINIMAL) ? 2'b10 : // 50% limit
                                   2'b11; // 25% limit
    
    assign noc_bandwidth_limit = memory_bandwidth_limit;
    
    // Power gating for isolated units
    assign core_power_gate = core_isolated_reg;
    assign tpu_power_gate = tpu_isolated_reg;
    assign vpu_power_gate = vpu_isolated_reg;
    assign memory_power_gate = memory_isolated_reg;
    
    // Status outputs
    assign system_health = health_score;
    assign performance_level = current_perf_level;
    assign isolated_units = {4'b0, $countones(noc_isolated_reg), $countones(memory_isolated_reg), 
                           $countones(vpu_isolated_reg), $countones(tpu_isolated_reg), $countones(core_isolated_reg)};
    assign emergency_mode = (current_mode == MODE_EMERGENCY);
    assign safe_mode = (current_mode == MODE_SAFE);

endmodule