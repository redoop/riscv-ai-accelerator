// Memory Subsystem Integration Module
// Integrates L1, L2, L3 caches and HBM memory controller
// Provides unified memory interface for RISC-V cores and accelerators

module memory_subsystem #(
    parameter NUM_CORES = 4,
    parameter NUM_ACCELERATORS = 4,
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 64,
    parameter CACHE_LINE_WIDTH = 512
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Core interfaces (L1 I-cache and D-cache per core)
    input  logic [NUM_CORES-1:0][ADDR_WIDTH-1:0]   core_icache_addr,
    output logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_icache_rdata,
    input  logic [NUM_CORES-1:0]                   core_icache_req,
    output logic [NUM_CORES-1:0]                   core_icache_ready,
    output logic [NUM_CORES-1:0]                   core_icache_hit,
    
    input  logic [NUM_CORES-1:0][ADDR_WIDTH-1:0]   core_dcache_addr,
    input  logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_dcache_wdata,
    output logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_dcache_rdata,
    input  logic [NUM_CORES-1:0]                   core_dcache_req,
    input  logic [NUM_CORES-1:0]                   core_dcache_we,
    input  logic [NUM_CORES-1:0][DATA_WIDTH/8-1:0] core_dcache_be,
    output logic [NUM_CORES-1:0]                   core_dcache_ready,
    output logic [NUM_CORES-1:0]                   core_dcache_hit,
    
    // Accelerator interfaces (direct to L2)
    axi4_if.slave       accel_if [NUM_ACCELERATORS-1:0],
    
    // HBM physical interfaces
    output logic [3:0]                    hbm_clk_p,
    output logic [3:0]                    hbm_clk_n,
    output logic [3:0]                    hbm_rst_n,
    output logic [3:0][5:0]              hbm_cmd,
    output logic [3:0][33:0]             hbm_addr,
    output logic [3:0]                    hbm_cmd_valid,
    input  logic [3:0]                    hbm_cmd_ready,
    output logic [3:0][1023:0]           hbm_wdata,
    output logic [3:0][127:0]            hbm_wstrb,
    output logic [3:0]                    hbm_wvalid,
    input  logic [3:0]                    hbm_wready,
    input  logic [3:0][1023:0]           hbm_rdata,
    input  logic [3:0]                    hbm_rvalid,
    output logic [3:0]                    hbm_rready,
    
    // Cache coherency interface
    input  logic                    snoop_req,
    input  logic [ADDR_WIDTH-1:0]   snoop_addr,
    input  logic [2:0]              snoop_type,
    output logic                    snoop_hit,
    output logic                    snoop_dirty,
    output logic [2:0]              snoop_resp,
    
    // Performance and error monitoring
    output logic [31:0]             l1_hit_count,
    output logic [31:0]             l1_miss_count,
    output logic [31:0]             l2_hit_count,
    output logic [31:0]             l2_miss_count,
    output logic [31:0]             l3_hit_count,
    output logic [31:0]             l3_miss_count,
    output logic [31:0]             memory_bandwidth,
    output logic                    ecc_error,
    output logic [ADDR_WIDTH-1:0]   error_addr,
    
    // Advanced memory performance monitoring
    output logic [31:0]             hbm_bank_conflicts,
    output logic [31:0]             hbm_row_hit_rate,
    output logic [15:0]             hbm_avg_latency,
    output logic [15:0]             hbm_peak_bandwidth,
    output logic [7:0]              hbm_queue_utilization,
    output logic [15:0]             hbm_power_consumption
);

    // Internal AXI interfaces
    axi4_if #(.ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(CACHE_LINE_WIDTH)) 
        l1_l2_if [NUM_CORES*2-1:0]();  // 2 per core (I-cache + D-cache)
    
    axi4_if #(.ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(CACHE_LINE_WIDTH)) 
        l2_l3_if [NUM_CORES+NUM_ACCELERATORS-1:0]();
    
    axi4_if #(.ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(CACHE_LINE_WIDTH)) 
        l3_mem_if();
    
    // Snoop signals for each cache level
    logic l1_snoop_hit [NUM_CORES*2-1:0];
    logic l1_snoop_dirty [NUM_CORES-1:0];  // Only D-cache can be dirty
    logic [2:0] l1_snoop_resp [NUM_CORES*2-1:0];
    
    logic l2_snoop_hit [NUM_CORES-1:0];
    logic l2_snoop_dirty [NUM_CORES-1:0];
    logic [2:0] l2_snoop_resp [NUM_CORES-1:0];
    
    logic l3_snoop_hit, l3_snoop_dirty;
    logic [2:0] l3_snoop_resp;
    
    // ECC error signals
    logic l2_ecc_single [NUM_CORES-1:0], l2_ecc_double [NUM_CORES-1:0];
    logic l3_ecc_single, l3_ecc_double;
    logic mem_ecc_error;
    logic [ADDR_WIDTH-1:0] l2_ecc_addr [NUM_CORES-1:0];
    logic [ADDR_WIDTH-1:0] l3_ecc_addr, mem_ecc_addr;
    
    // Performance counters
    logic [31:0] l1_hits, l1_misses;
    logic [31:0] l2_hits, l2_misses;
    logic [31:0] mem_read_reqs, mem_write_reqs, mem_total_bw;
    
    // L1 Cache instantiation (per core)
    generate
        for (genvar core = 0; core < NUM_CORES; core++) begin : l1_caches
            
            // L1 Instruction Cache
            l1_icache #(
                .CACHE_SIZE(32*1024),
                .WAYS(4),
                .LINE_SIZE(64),
                .ADDR_WIDTH(ADDR_WIDTH),
                .DATA_WIDTH(DATA_WIDTH)
            ) l1_icache_inst (
                .clk(clk),
                .rst_n(rst_n),
                .cpu_addr(core_icache_addr[core]),
                .cpu_rdata(core_icache_rdata[core]),
                .cpu_req(core_icache_req[core]),
                .cpu_ready(core_icache_ready[core]),
                .cpu_hit(core_icache_hit[core]),
                .l2_if(l1_l2_if[core*2].master),
                .snoop_req(snoop_req),
                .snoop_addr(snoop_addr),
                .snoop_type(snoop_type),
                .snoop_hit(l1_snoop_hit[core*2]),
                .snoop_resp(l1_snoop_resp[core*2])
            );
            
            // L1 Data Cache
            l1_dcache #(
                .CACHE_SIZE(32*1024),
                .WAYS(8),
                .LINE_SIZE(64),
                .ADDR_WIDTH(ADDR_WIDTH),
                .DATA_WIDTH(DATA_WIDTH)
            ) l1_dcache_inst (
                .clk(clk),
                .rst_n(rst_n),
                .cpu_addr(core_dcache_addr[core]),
                .cpu_wdata(core_dcache_wdata[core]),
                .cpu_rdata(core_dcache_rdata[core]),
                .cpu_req(core_dcache_req[core]),
                .cpu_we(core_dcache_we[core]),
                .cpu_be(core_dcache_be[core]),
                .cpu_ready(core_dcache_ready[core]),
                .cpu_hit(core_dcache_hit[core]),
                .l2_if(l1_l2_if[core*2+1].master),
                .snoop_req(snoop_req),
                .snoop_addr(snoop_addr),
                .snoop_type(snoop_type),
                .snoop_hit(l1_snoop_hit[core*2+1]),
                .snoop_dirty(l1_snoop_dirty[core]),
                .snoop_resp(l1_snoop_resp[core*2+1])
            );
            
            // L2 Cache (shared per core cluster)
            l2_cache #(
                .CACHE_SIZE(2*1024*1024),
                .WAYS(16),
                .LINE_SIZE(64),
                .ADDR_WIDTH(ADDR_WIDTH),
                .DATA_WIDTH(CACHE_LINE_WIDTH),
                .NUM_PORTS(2)  // I-cache + D-cache
            ) l2_cache_inst (
                .clk(clk),
                .rst_n(rst_n),
                .l1_if(l1_l2_if[core*2 +: 2]),
                .l3_if(l2_l3_if[core].master),
                .snoop_req(snoop_req),
                .snoop_addr(snoop_addr),
                .snoop_type(snoop_type),
                .snoop_hit(l2_snoop_hit[core]),
                .snoop_dirty(l2_snoop_dirty[core]),
                .snoop_resp(l2_snoop_resp[core]),
                .ecc_single_error(l2_ecc_single[core]),
                .ecc_double_error(l2_ecc_double[core]),
                .ecc_error_addr(l2_ecc_addr[core])
            );
        end
    endgenerate
    
    // Connect accelerator interfaces to L2-L3 interface array
    generate
        for (genvar acc = 0; acc < NUM_ACCELERATORS; acc++) begin : accel_connections
            assign l2_l3_if[NUM_CORES + acc] = accel_if[acc];
        end
    endgenerate
    
    // L3 Cache (shared among all cores and accelerators)
    l3_cache #(
        .CACHE_SIZE(8*1024*1024),
        .WAYS(16),
        .LINE_SIZE(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(CACHE_LINE_WIDTH),
        .NUM_PORTS(NUM_CORES + NUM_ACCELERATORS)
    ) l3_cache_inst (
        .clk(clk),
        .rst_n(rst_n),
        .l2_if(l2_l3_if),
        .mem_if(l3_mem_if.master),
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(l3_snoop_hit),
        .snoop_dirty(l3_snoop_dirty),
        .snoop_resp(l3_snoop_resp),
        .ecc_single_error(l3_ecc_single),
        .ecc_double_error(l3_ecc_double),
        .ecc_error_addr(l3_ecc_addr),
        .hit_count(l3_hit_count),
        .miss_count(l3_miss_count),
        .eviction_count()  // Not used in this interface
    );
    
    // HBM Memory Controller with enhanced monitoring
    hbm_controller #(
        .NUM_CHANNELS(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(CACHE_LINE_WIDTH),
        .HBM_DATA_WIDTH(1024)
    ) hbm_ctrl_inst (
        .clk(clk),
        .rst_n(rst_n),
        .axi_if(l3_mem_if.slave),
        .hbm_clk_p(hbm_clk_p),
        .hbm_clk_n(hbm_clk_n),
        .hbm_rst_n(hbm_rst_n),
        .hbm_cmd(hbm_cmd),
        .hbm_addr(hbm_addr),
        .hbm_cmd_valid(hbm_cmd_valid),
        .hbm_cmd_ready(hbm_cmd_ready),
        .hbm_wdata(hbm_wdata),
        .hbm_wstrb(hbm_wstrb),
        .hbm_wvalid(hbm_wvalid),
        .hbm_wready(hbm_wready),
        .hbm_rdata(hbm_rdata),
        .hbm_rvalid(hbm_rvalid),
        .hbm_rready(hbm_rready),
        .read_requests(mem_read_reqs),
        .write_requests(mem_write_reqs),
        .total_bandwidth(mem_total_bw),
        .avg_latency(hbm_avg_latency),
        .bank_conflict_count(hbm_bank_conflicts),
        .row_hit_rate(hbm_row_hit_rate),
        .instantaneous_bandwidth(),  // Not used in this interface
        .peak_bandwidth(hbm_peak_bandwidth),
        .min_access_latency(),       // Not used in this interface
        .max_access_latency(),       // Not used in this interface
        .queue_utilization(hbm_queue_utilization),
        .power_consumption(hbm_power_consumption),
        .ecc_error(mem_ecc_error),
        .error_addr(mem_ecc_addr),
        .error_type()    // Not used in this interface
    );
    
    // Aggregate snoop responses
    always_comb begin
        snoop_hit = l3_snoop_hit;
        snoop_dirty = l3_snoop_dirty;
        snoop_resp = l3_snoop_resp;
        
        // Check L2 caches
        for (int i = 0; i < NUM_CORES; i++) begin
            if (l2_snoop_hit[i]) begin
                snoop_hit = 1'b1;
                if (l2_snoop_dirty[i]) snoop_dirty = 1'b1;
                if (l2_snoop_resp[i] != 3'b000) snoop_resp = l2_snoop_resp[i];
            end
        end
        
        // Check L1 caches
        for (int i = 0; i < NUM_CORES*2; i++) begin
            if (l1_snoop_hit[i]) begin
                snoop_hit = 1'b1;
                if (l1_snoop_resp[i] != 3'b000) snoop_resp = l1_snoop_resp[i];
            end
        end
        
        // Check L1 D-cache dirty bits
        for (int i = 0; i < NUM_CORES; i++) begin
            if (l1_snoop_dirty[i]) snoop_dirty = 1'b1;
        end
    end
    
    // Aggregate performance counters
    always_comb begin
        l1_hits = 0;
        l1_misses = 0;
        l2_hits = 0;
        l2_misses = 0;
        
        // Sum up L1 hits/misses (would need counters from L1 caches)
        for (int i = 0; i < NUM_CORES; i++) begin
            // L1 counters would be added here if implemented
        end
        
        // L2 counters would be summed here if implemented
        // L3 counters are already provided by l3_cache_inst
    end
    
    assign l1_hit_count = l1_hits;
    assign l1_miss_count = l1_misses;
    assign l2_hit_count = l2_hits;
    assign l2_miss_count = l2_misses;
    assign memory_bandwidth = mem_total_bw;
    
    // Aggregate ECC errors
    always_comb begin
        ecc_error = mem_ecc_error || l3_ecc_single || l3_ecc_double;
        error_addr = mem_ecc_addr;
        
        // Check L2 ECC errors
        for (int i = 0; i < NUM_CORES; i++) begin
            if (l2_ecc_single[i] || l2_ecc_double[i]) begin
                ecc_error = 1'b1;
                error_addr = l2_ecc_addr[i];
            end
        end
        
        // L3 ECC errors
        if (l3_ecc_single || l3_ecc_double) begin
            ecc_error = 1'b1;
            error_addr = l3_ecc_addr;
        end
    end

endmodule