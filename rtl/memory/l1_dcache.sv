// L1 Data Cache Module
// 32KB, 8-way set associative, 64-byte cache lines
// Supports write-back policy with dirty bit tracking

module l1_dcache #(
    parameter CACHE_SIZE = 32 * 1024,  // 32KB
    parameter WAYS = 8,                // 8-way set associative
    parameter LINE_SIZE = 64,          // 64-byte cache lines
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
    output logic                    cpu_ready,
    output logic                    cpu_hit,
    
    // L2 cache interface
    axi4_if.master                  l2_if,
    
    // Cache coherency interface
    input  logic                    snoop_req,
    input  logic [ADDR_WIDTH-1:0]   snoop_addr,
    input  logic [2:0]              snoop_type,
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
    
    // Cache arrays
    logic [TAG_BITS-1:0]    tag_array [SETS-1:0][WAYS-1:0];
    logic [511:0]           data_array [SETS-1:0][WAYS-1:0];  // 512-bit cache lines
    logic                   valid [SETS-1:0][WAYS-1:0];
    logic                   dirty [SETS-1:0][WAYS-1:0];
    logic [1:0]             mesi_state [SETS-1:0][WAYS-1:0];  // MESI coherency
    
    // LRU replacement (pseudo-LRU for 8-way)
    logic [6:0]             lru_bits [SETS-1:0];  // 7 bits for 8-way pseudo-LRU tree
    
    // Hit detection
    logic [WAYS-1:0] hit_way;
    logic cache_hit;
    
    genvar i;
    generate
        for (i = 0; i < WAYS; i++) begin : hit_detection
            assign hit_way[i] = valid[index][i] && 
                               (tag_array[index][i] == tag) &&
                               (mesi_state[index][i] != 2'b00); // Not Invalid
        end
    endgenerate
    
    assign cache_hit = |hit_way;
    assign cpu_hit = cache_hit;
    
    // Cache state machine
    typedef enum logic [2:0] {
        IDLE,
        LOOKUP,
        MISS_REQ,
        MISS_WAIT,
        WRITEBACK_REQ,
        WRITEBACK_WAIT,
        SNOOP_CHECK
    } cache_state_t;
    
    cache_state_t state, next_state;
    
    // Miss and writeback handling
    logic                   miss_pending;
    logic [ADDR_WIDTH-1:0]  miss_addr;
    logic [$clog2(WAYS)-1:0] replace_way;
    logic                   need_writeback;
    logic [ADDR_WIDTH-1:0]  writeback_addr;
    logic [511:0]           writeback_data;
    
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
                end else if (need_writeback) begin
                    next_state = WRITEBACK_REQ;
                end else begin
                    next_state = MISS_REQ;
                end
            end
            
            WRITEBACK_REQ: begin
                if (l2_if.awready) begin
                    next_state = WRITEBACK_WAIT;
                end
            end
            
            WRITEBACK_WAIT: begin
                if (l2_if.bvalid) begin
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
            
            SNOOP_CHECK: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Data output selection
    logic [511:0] selected_line;
    always_comb begin
        selected_line = '0;
        for (int j = 0; j < WAYS; j++) begin
            if (hit_way[j]) begin
                selected_line = data_array[index][j];
            end
        end
    end
    
    // Extract requested data from cache line
    assign cpu_rdata = selected_line[offset*8 +: DATA_WIDTH];
    assign cpu_ready = (state == IDLE) || (state == LOOKUP && cache_hit);
    
    // Pseudo-LRU replacement policy for 8-way cache
    function automatic logic [$clog2(WAYS)-1:0] get_lru_way(logic [6:0] lru_state);
        logic [2:0] way;
        // Pseudo-LRU tree traversal
        way[2] = ~lru_state[0];
        way[1] = way[2] ? ~lru_state[2] : ~lru_state[1];
        way[0] = (way[2:1] == 2'b00) ? ~lru_state[6] :
                 (way[2:1] == 2'b01) ? ~lru_state[5] :
                 (way[2:1] == 2'b10) ? ~lru_state[4] : ~lru_state[3];
        return way;
    endfunction
    
    function automatic logic [6:0] update_lru(logic [6:0] lru_state, logic [2:0] accessed_way);
        logic [6:0] new_lru;
        new_lru = lru_state;
        // Update LRU tree based on accessed way
        new_lru[0] = accessed_way[2];
        if (accessed_way[2]) begin
            new_lru[2] = accessed_way[1];
            if (accessed_way[1]) begin
                new_lru[4] = accessed_way[0];
            end else begin
                new_lru[5] = accessed_way[0];
            end
        end else begin
            new_lru[1] = accessed_way[1];
            if (accessed_way[1]) begin
                new_lru[3] = accessed_way[0];
            end else begin
                new_lru[6] = accessed_way[0];
            end
        end
        return new_lru;
    endfunction
    
    // LRU update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < SETS; s++) begin
                lru_bits[s] <= '0;
            end
        end else if (cpu_req && cache_hit) begin
            for (int w = 0; w < WAYS; w++) begin
                if (hit_way[w]) begin
                    lru_bits[index] <= update_lru(lru_bits[index], w[2:0]);
                end
            end
        end
    end
    
    // Determine replacement way and check for writeback need
    assign replace_way = get_lru_way(lru_bits[index]);
    assign need_writeback = valid[index][replace_way] && dirty[index][replace_way];
    
    // Writeback address and data
    always_comb begin
        writeback_addr = {tag_array[index][replace_way], index, {OFFSET_BITS{1'b0}}};
        writeback_data = data_array[index][replace_way];
    end
    
    // Cache write logic
    always_ff @(posedge clk) begin
        if (cpu_req && cpu_we && cache_hit) begin
            for (int w = 0; w < WAYS; w++) begin
                if (hit_way[w]) begin
                    // Write data with byte enables
                    for (int b = 0; b < DATA_WIDTH/8; b++) begin
                        if (cpu_be[b]) begin
                            data_array[index][w][offset*8 + b*8 +: 8] <= cpu_wdata[b*8 +: 8];
                        end
                    end
                    dirty[index][w] <= 1'b1;
                    mesi_state[index][w] <= 2'b11; // Modified
                end
            end
        end
    end
    
    // AXI read interface
    assign l2_if.arid = '0;
    assign l2_if.araddr = {miss_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
    assign l2_if.arlen = (LINE_SIZE / (l2_if.DATA_WIDTH/8)) - 1;
    assign l2_if.arsize = $clog2(l2_if.DATA_WIDTH/8);
    assign l2_if.arburst = 2'b01; // INCR
    assign l2_if.arlock = 1'b0;
    assign l2_if.arcache = 4'b0010;
    assign l2_if.arprot = 3'b000;
    assign l2_if.arqos = 4'b0000;
    assign l2_if.arvalid = (state == MISS_REQ);
    assign l2_if.rready = 1'b1;
    
    // AXI write interface (for writebacks)
    assign l2_if.awid = '0;
    assign l2_if.awaddr = writeback_addr;
    assign l2_if.awlen = (LINE_SIZE / (l2_if.DATA_WIDTH/8)) - 1;
    assign l2_if.awsize = $clog2(l2_if.DATA_WIDTH/8);
    assign l2_if.awburst = 2'b01; // INCR
    assign l2_if.awlock = 1'b0;
    assign l2_if.awcache = 4'b0010;
    assign l2_if.awprot = 3'b000;
    assign l2_if.awqos = 4'b0000;
    assign l2_if.awvalid = (state == WRITEBACK_REQ);
    
    assign l2_if.wdata = writeback_data;
    assign l2_if.wstrb = '1; // Write entire cache line
    assign l2_if.wlast = 1'b1; // Single beat for now
    assign l2_if.wvalid = (state == WRITEBACK_REQ) && l2_if.awready;
    assign l2_if.bready = 1'b1;
    
    // Miss address tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            miss_pending <= 1'b0;
        end else begin
            if (state == MISS_REQ) begin
                miss_pending <= 1'b1;
                miss_addr <= cpu_addr;
            end else if (state == IDLE) begin
                miss_pending <= 1'b0;
            end
        end
    end
    
    // Cache fill with burst support and error handling
    logic [2:0] fill_beat_count;
    logic [511:0] fill_buffer;
    logic fill_error;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_beat_count <= '0;
            fill_buffer <= '0;
            fill_error <= 1'b0;
        end else if (l2_if.rvalid && miss_pending) begin
            logic [INDEX_BITS-1:0] miss_index;
            miss_index = miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS];
            
            // Check for read errors
            if (l2_if.rresp != 2'b00) begin
                fill_error <= 1'b1;
            end
            
            // Accumulate data for multi-beat transfers
            if (l2_if.DATA_WIDTH == 512) begin
                // Single beat transfer
                if (!fill_error) begin
                    data_array[miss_index][replace_way] <= l2_if.rdata;
                end
            end else begin
                // Multi-beat transfer - accumulate data
                if (!fill_error) begin
                    fill_buffer[fill_beat_count * l2_if.DATA_WIDTH +: l2_if.DATA_WIDTH] <= l2_if.rdata;
                end
                fill_beat_count <= fill_beat_count + 1;
                
                if (l2_if.rlast) begin
                    if (!fill_error) begin
                        data_array[miss_index][replace_way] <= fill_buffer;
                    end
                    fill_beat_count <= '0;
                end
            end
            
            if (l2_if.rlast) begin
                if (!fill_error) begin
                    tag_array[miss_index][replace_way] <= miss_addr[ADDR_WIDTH-1:INDEX_BITS+OFFSET_BITS];
                    valid[miss_index][replace_way] <= 1'b1;
                    dirty[miss_index][replace_way] <= 1'b0;
                    mesi_state[miss_index][replace_way] <= 2'b10; // Exclusive
                end
                fill_error <= 1'b0;
            end
        end
    end
    
    // Snoop logic
    logic [TAG_BITS-1:0]    snoop_tag;
    logic [INDEX_BITS-1:0]  snoop_index;
    logic [WAYS-1:0]        snoop_hit_way;
    
    assign {snoop_tag, snoop_index} = snoop_addr[ADDR_WIDTH-1:OFFSET_BITS];
    
    generate
        for (i = 0; i < WAYS; i++) begin : snoop_hit_detection
            assign snoop_hit_way[i] = valid[snoop_index][i] && 
                                     (tag_array[snoop_index][i] == snoop_tag);
        end
    endgenerate
    
    assign snoop_hit = |snoop_hit_way;
    
    // Check if snooped line is dirty
    logic snoop_line_dirty;
    always_comb begin
        snoop_line_dirty = 1'b0;
        for (int w = 0; w < WAYS; w++) begin
            if (snoop_hit_way[w] && dirty[snoop_index][w]) begin
                snoop_line_dirty = 1'b1;
            end
        end
    end
    assign snoop_dirty = snoop_line_dirty;
    
    // Snoop response
    always_comb begin
        snoop_resp = 3'b000;
        if (snoop_hit) begin
            case (snoop_type)
                3'b001: snoop_resp = snoop_dirty ? 3'b100 : 3'b010; // Read request
                3'b010: snoop_resp = 3'b011; // Invalidate
                default: snoop_resp = 3'b000;
            endcase
        end
    end
    
    // Snoop invalidation
    always_ff @(posedge clk) begin
        if (snoop_req && snoop_hit && (snoop_type == 3'b010)) begin
            for (int w = 0; w < WAYS; w++) begin
                if (snoop_hit_way[w]) begin
                    mesi_state[snoop_index][w] <= 2'b00; // Invalid
                end
            end
        end
    end
    
    // Performance counters
    logic [31:0] perf_hits;
    logic [31:0] perf_misses;
    logic [31:0] perf_reads;
    logic [31:0] perf_writes;
    logic [31:0] perf_writebacks;
    logic [31:0] perf_snoop_hits;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_hits <= '0;
            perf_misses <= '0;
            perf_reads <= '0;
            perf_writes <= '0;
            perf_writebacks <= '0;
            perf_snoop_hits <= '0;
        end else begin
            if (cpu_req) begin
                if (cpu_we) begin
                    perf_writes <= perf_writes + 1;
                end else begin
                    perf_reads <= perf_reads + 1;
                end
                
                if (cache_hit) begin
                    perf_hits <= perf_hits + 1;
                end else begin
                    perf_misses <= perf_misses + 1;
                end
            end
            
            if (state == WRITEBACK_REQ && l2_if.awready) begin
                perf_writebacks <= perf_writebacks + 1;
            end
            
            if (snoop_req && snoop_hit) begin
                perf_snoop_hits <= perf_snoop_hits + 1;
            end
        end
    end
    
    // Initialize arrays
    initial begin
        for (int s = 0; s < SETS; s++) begin
            for (int w = 0; w < WAYS; w++) begin
                valid[s][w] = 1'b0;
                dirty[s][w] = 1'b0;
                mesi_state[s][w] = 2'b00; // Invalid
            end
        end
    end

endmodule