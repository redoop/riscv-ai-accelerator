// L1 Cache Controller Module
// Implements separate L1 instruction and data caches with coherency support

module l1_cache_controller #(
    parameter CACHE_SIZE = 32 * 1024,  // 32KB cache size
    parameter WAYS = 8,                // Associativity
    parameter LINE_SIZE = 64,          // Cache line size in bytes
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 64
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // CPU interface
    input  logic [ADDR_WIDTH-1:0]   cpu_addr,
    input  logic [DATA_WIDTH-1:0]   cpu_wdata,
    output logic [DATA_WIDTH-1:0]   cpu_rdata,
    input  logic                    cpu_req,
    input  logic                    cpu_we,
    input  logic [DATA_WIDTH/8-1:0] cpu_be,
    input  logic                    cpu_is_instr,  // 1 for instruction fetch, 0 for data
    output logic                    cpu_ready,
    output logic                    cpu_hit,
    
    // L2 cache interface
    axi4_if.master                  l2_if,
    
    // Cache coherency interface (for snooping)
    input  logic                    snoop_req,
    input  logic [ADDR_WIDTH-1:0]   snoop_addr,
    input  logic [2:0]              snoop_type,  // MESI protocol states
    output logic                    snoop_hit,
    output logic                    snoop_dirty,
    output logic [2:0]              snoop_resp
);

    // Cache parameters
    localparam SETS = CACHE_SIZE / (WAYS * LINE_SIZE);
    localparam OFFSET_BITS = $clog2(LINE_SIZE);
    localparam INDEX_BITS = $clog2(SETS);
    localparam TAG_BITS = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;
    
    // Address breakdown
    logic [TAG_BITS-1:0]    tag;
    logic [INDEX_BITS-1:0]  index;
    logic [OFFSET_BITS-1:0] offset;
    
    assign {tag, index, offset} = cpu_addr;
    
    // Cache arrays for I-cache and D-cache
    // Tag arrays
    logic [TAG_BITS-1:0]    icache_tag_array [SETS-1:0][WAYS-1:0];
    logic [TAG_BITS-1:0]    dcache_tag_array [SETS-1:0][WAYS-1:0];
    
    // Data arrays (512 bits per line to match AXI interface)
    logic [511:0]           icache_data_array [SETS-1:0][WAYS-1:0];
    logic [511:0]           dcache_data_array [SETS-1:0][WAYS-1:0];
    
    // Valid and dirty bits
    logic                   icache_valid [SETS-1:0][WAYS-1:0];
    logic                   dcache_valid [SETS-1:0][WAYS-1:0];
    logic                   dcache_dirty [SETS-1:0][WAYS-1:0];
    
    // MESI coherency state (Modified, Exclusive, Shared, Invalid)
    logic [1:0]             icache_mesi [SETS-1:0][WAYS-1:0];
    logic [1:0]             dcache_mesi [SETS-1:0][WAYS-1:0];
    
    // LRU replacement policy
    logic [$clog2(WAYS)-1:0] icache_lru [SETS-1:0];
    logic [$clog2(WAYS)-1:0] dcache_lru [SETS-1:0];
    
    // Cache hit detection
    logic [WAYS-1:0]        icache_hit_way;
    logic [WAYS-1:0]        dcache_hit_way;
    logic                   icache_hit;
    logic                   dcache_hit;
    logic                   cache_hit;
    
    // Hit detection logic
    genvar i;
    generate
        for (i = 0; i < WAYS; i++) begin : hit_detection
            assign icache_hit_way[i] = icache_valid[index][i] && 
                                      (icache_tag_array[index][i] == tag) &&
                                      (icache_mesi[index][i] != 2'b00); // Not Invalid
            
            assign dcache_hit_way[i] = dcache_valid[index][i] && 
                                      (dcache_tag_array[index][i] == tag) &&
                                      (dcache_mesi[index][i] != 2'b00); // Not Invalid
        end
    endgenerate
    
    assign icache_hit = |icache_hit_way;
    assign dcache_hit = |dcache_hit_way;
    assign cache_hit = cpu_is_instr ? icache_hit : dcache_hit;
    assign cpu_hit = cache_hit;
    
    // Cache state machine
    typedef enum logic [2:0] {
        IDLE,
        LOOKUP,
        MISS_REQ,
        MISS_WAIT,
        WRITEBACK,
        SNOOP_CHECK
    } cache_state_t;
    
    cache_state_t state, next_state;
    
    // Miss handling
    logic                   miss_pending;
    logic [ADDR_WIDTH-1:0]  miss_addr;
    logic                   miss_is_instr;
    logic [$clog2(WAYS)-1:0] replace_way;
    
    // AXI transaction tracking
    logic                   axi_read_pending;
    logic                   axi_write_pending;
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (cpu_req) begin
                    next_state = LOOKUP;
                end else if (snoop_req) begin
                    next_state = SNOOP_CHECK;
                end
            end
            
            LOOKUP: begin
                if (cache_hit) begin
                    next_state = IDLE;
                end else begin
                    next_state = MISS_REQ;
                end
            end
            
            MISS_REQ: begin
                if (l2_if.arready) begin
                    next_state = MISS_WAIT;
                end
            end
            
            MISS_WAIT: begin
                if (l2_if.rvalid && l2_if.rlast) begin
                    next_state = IDLE;
                end
            end
            
            WRITEBACK: begin
                if (l2_if.bvalid) begin
                    next_state = MISS_REQ;
                end
            end
            
            SNOOP_CHECK: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Cache read data selection
    logic [511:0] selected_cache_line;
    logic [DATA_WIDTH-1:0] cache_read_data;
    
    always_comb begin
        selected_cache_line = '0;
        if (cpu_is_instr) begin
            for (int j = 0; j < WAYS; j++) begin
                if (icache_hit_way[j]) begin
                    selected_cache_line = icache_data_array[index][j];
                end
            end
        end else begin
            for (int j = 0; j < WAYS; j++) begin
                if (dcache_hit_way[j]) begin
                    selected_cache_line = dcache_data_array[index][j];
                end
            end
        end
    end
    
    // Extract the requested data from cache line based on offset
    assign cache_read_data = selected_cache_line[offset*8 +: DATA_WIDTH];
    
    // CPU interface logic
    assign cpu_ready = (state == IDLE) || (state == LOOKUP && cache_hit);
    assign cpu_rdata = cache_hit ? cache_read_data : '0;
    
    // LRU update logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < SETS; s++) begin
                icache_lru[s] <= '0;
                dcache_lru[s] <= '0;
            end
        end else if (cpu_req && cache_hit) begin
            if (cpu_is_instr) begin
                for (int w = 0; w < WAYS; w++) begin
                    if (icache_hit_way[w]) begin
                        icache_lru[index] <= w;
                    end
                end
            end else begin
                for (int w = 0; w < WAYS; w++) begin
                    if (dcache_hit_way[w]) begin
                        dcache_lru[index] <= w;
                    end
                end
            end
        end
    end
    
    // Cache write logic (for D-cache only)
    always_ff @(posedge clk) begin
        if (cpu_req && cpu_we && !cpu_is_instr && dcache_hit) begin
            for (int w = 0; w < WAYS; w++) begin
                if (dcache_hit_way[w]) begin
                    // Write data to cache line with byte enables
                    for (int b = 0; b < DATA_WIDTH/8; b++) begin
                        if (cpu_be[b]) begin
                            dcache_data_array[index][w][offset*8 + b*8 +: 8] <= cpu_wdata[b*8 +: 8];
                        end
                    end
                    dcache_dirty[index][w] <= 1'b1;
                    dcache_mesi[index][w] <= 2'b11; // Modified state
                end
            end
        end
    end
    
    // AXI interface for L2 communication
    assign l2_if.arid = '0;
    assign l2_if.araddr = {miss_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
    assign l2_if.arlen = (LINE_SIZE / (l2_if.DATA_WIDTH/8)) - 1; // Burst length
    assign l2_if.arsize = 3'b110; // $clog2(512/8) = 6
    assign l2_if.arburst = 2'b01; // INCR burst
    assign l2_if.arlock = 1'b0;
    assign l2_if.arcache = 4'b0010; // Normal non-cacheable
    assign l2_if.arprot = 3'b000;
    assign l2_if.arqos = 4'b0000;
    assign l2_if.arvalid = (state == MISS_REQ) && !axi_read_pending;
    assign l2_if.rready = 1'b1;
    
    // Track AXI transactions
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_read_pending <= 1'b0;
            miss_pending <= 1'b0;
        end else begin
            if (l2_if.arvalid && l2_if.arready) begin
                axi_read_pending <= 1'b1;
            end else if (l2_if.rvalid && l2_if.rlast) begin
                axi_read_pending <= 1'b0;
            end
            
            if (state == MISS_REQ) begin
                miss_pending <= 1'b1;
                miss_addr <= cpu_addr;
                miss_is_instr <= cpu_is_instr;
            end else if (state == IDLE) begin
                miss_pending <= 1'b0;
            end
        end
    end
    
    // Cache fill logic
    logic [2:0] fill_beat_count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_beat_count <= '0;
        end else if (l2_if.rvalid) begin
            if (l2_if.rlast) begin
                fill_beat_count <= '0;
            end else begin
                fill_beat_count <= fill_beat_count + 1;
            end
        end
    end
    
    // Determine replacement way (LRU)
    assign replace_way = miss_is_instr ? icache_lru[miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS]] :
                                        dcache_lru[miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS]];
    
    // Fill cache on miss
    always_ff @(posedge clk) begin
        if (l2_if.rvalid && miss_pending) begin
            logic [INDEX_BITS-1:0] miss_index;
            miss_index = miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS];
            
            if (miss_is_instr) begin
                icache_data_array[miss_index][replace_way] <= l2_if.rdata;
                if (l2_if.rlast) begin
                    icache_tag_array[miss_index][replace_way] <= miss_addr[ADDR_WIDTH-1:INDEX_BITS+OFFSET_BITS];
                    icache_valid[miss_index][replace_way] <= 1'b1;
                    icache_mesi[miss_index][replace_way] <= 2'b10; // Exclusive state
                end
            end else begin
                dcache_data_array[miss_index][replace_way] <= l2_if.rdata;
                if (l2_if.rlast) begin
                    dcache_tag_array[miss_index][replace_way] <= miss_addr[ADDR_WIDTH-1:INDEX_BITS+OFFSET_BITS];
                    dcache_valid[miss_index][replace_way] <= 1'b1;
                    dcache_dirty[miss_index][replace_way] <= 1'b0;
                    dcache_mesi[miss_index][replace_way] <= 2'b10; // Exclusive state
                end
            end
        end
    end
    
    // Snoop logic for cache coherency
    logic [TAG_BITS-1:0]    snoop_tag;
    logic [INDEX_BITS-1:0]  snoop_index;
    logic [WAYS-1:0]        snoop_hit_way_i, snoop_hit_way_d;
    
    assign {snoop_tag, snoop_index} = snoop_addr[ADDR_WIDTH-1:OFFSET_BITS];
    
    generate
        for (i = 0; i < WAYS; i++) begin : snoop_hit_detection
            assign snoop_hit_way_i[i] = icache_valid[snoop_index][i] && 
                                       (icache_tag_array[snoop_index][i] == snoop_tag);
            assign snoop_hit_way_d[i] = dcache_valid[snoop_index][i] && 
                                       (dcache_tag_array[snoop_index][i] == snoop_tag);
        end
    endgenerate
    
    assign snoop_hit = |snoop_hit_way_i || |snoop_hit_way_d;
    
    // Check if snooped line is dirty
    logic snoop_line_dirty;
    always_comb begin
        snoop_line_dirty = 1'b0;
        for (int w = 0; w < WAYS; w++) begin
            if (snoop_hit_way_d[w] && dcache_dirty[snoop_index][w]) begin
                snoop_line_dirty = 1'b1;
            end
        end
    end
    assign snoop_dirty = snoop_line_dirty;
    
    // Snoop response based on MESI protocol
    always_comb begin
        snoop_resp = 3'b000; // No response
        if (snoop_hit) begin
            case (snoop_type)
                3'b001: snoop_resp = snoop_dirty ? 3'b100 : 3'b010; // Read request
                3'b010: snoop_resp = 3'b011; // Invalidate
                default: snoop_resp = 3'b000;
            endcase
        end
    end
    
    // Performance counters
    logic [31:0] perf_icache_hits, perf_icache_misses;
    logic [31:0] perf_dcache_hits, perf_dcache_misses;
    logic [31:0] perf_dcache_reads, perf_dcache_writes;
    logic [31:0] perf_snoop_hits, perf_writebacks;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_icache_hits <= '0;
            perf_icache_misses <= '0;
            perf_dcache_hits <= '0;
            perf_dcache_misses <= '0;
            perf_dcache_reads <= '0;
            perf_dcache_writes <= '0;
            perf_snoop_hits <= '0;
            perf_writebacks <= '0;
        end else begin
            if (cpu_req) begin
                if (cpu_is_instr) begin
                    if (icache_hit) begin
                        perf_icache_hits <= perf_icache_hits + 1;
                    end else begin
                        perf_icache_misses <= perf_icache_misses + 1;
                    end
                end else begin
                    if (cpu_we) begin
                        perf_dcache_writes <= perf_dcache_writes + 1;
                    end else begin
                        perf_dcache_reads <= perf_dcache_reads + 1;
                    end
                    
                    if (dcache_hit) begin
                        perf_dcache_hits <= perf_dcache_hits + 1;
                    end else begin
                        perf_dcache_misses <= perf_dcache_misses + 1;
                    end
                end
            end
            
            if (snoop_req && snoop_hit) begin
                perf_snoop_hits <= perf_snoop_hits + 1;
            end
        end
    end
    
    // Writeback handling for dirty cache lines
    logic need_writeback;
    logic [ADDR_WIDTH-1:0] writeback_addr;
    logic [511:0] writeback_data;
    
    always_comb begin
        need_writeback = 1'b0;
        writeback_addr = '0;
        writeback_data = '0;
        
        if (!miss_is_instr && miss_pending) begin
            logic [INDEX_BITS-1:0] miss_index;
            miss_index = miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS];
            
            if (dcache_valid[miss_index][replace_way] && dcache_dirty[miss_index][replace_way]) begin
                need_writeback = 1'b1;
                writeback_addr = {dcache_tag_array[miss_index][replace_way], miss_index, {OFFSET_BITS{1'b0}}};
                writeback_data = dcache_data_array[miss_index][replace_way];
            end
        end
    end
    
    // AXI write interface for writebacks
    assign l2_if.awid = '0;
    assign l2_if.awaddr = writeback_addr;
    assign l2_if.awlen = (LINE_SIZE / (l2_if.DATA_WIDTH/8)) - 1;
    assign l2_if.awsize = 3'b110; // $clog2(512/8) = 6
    assign l2_if.awburst = 2'b01; // INCR
    assign l2_if.awlock = 1'b0;
    assign l2_if.awcache = 4'b0010;
    assign l2_if.awprot = 3'b000;
    assign l2_if.awqos = 4'b0000;
    assign l2_if.awvalid = (state == WRITEBACK);
    
    assign l2_if.wdata = writeback_data;
    assign l2_if.wstrb = '1; // Write entire cache line
    assign l2_if.wlast = 1'b1;
    assign l2_if.wvalid = (state == WRITEBACK) && l2_if.awready;
    assign l2_if.bready = 1'b1;
    
    // Snoop invalidation handling
    always_ff @(posedge clk) begin
        if (snoop_req && snoop_hit && (snoop_type == 3'b010)) begin // Invalidate
            for (int w = 0; w < WAYS; w++) begin
                if (snoop_hit_way_i[w]) begin
                    icache_mesi[snoop_index][w] <= 2'b00; // Invalid
                end
                if (snoop_hit_way_d[w]) begin
                    dcache_mesi[snoop_index][w] <= 2'b00; // Invalid
                end
            end
        end
    end
    
    // Initialize cache arrays
    initial begin
        for (int s = 0; s < SETS; s++) begin
            for (int w = 0; w < WAYS; w++) begin
                icache_valid[s][w] = 1'b0;
                dcache_valid[s][w] = 1'b0;
                dcache_dirty[s][w] = 1'b0;
                icache_mesi[s][w] = 2'b00; // Invalid
                dcache_mesi[s][w] = 2'b00; // Invalid
            end
        end
    end

endmodule