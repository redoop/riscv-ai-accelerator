// L1 Instruction Cache Module
// 32KB, 4-way set associative, 64-byte cache lines

module l1_icache #(
    parameter CACHE_SIZE = 32 * 1024,  // 32KB
    parameter WAYS = 4,                // 4-way set associative
    parameter LINE_SIZE = 64,          // 64-byte cache lines
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 64
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // CPU interface
    input  logic [ADDR_WIDTH-1:0]   cpu_addr,
    output logic [DATA_WIDTH-1:0]   cpu_rdata,
    input  logic                    cpu_req,
    output logic                    cpu_ready,
    output logic                    cpu_hit,
    
    // L2 cache interface
    axi4_if.master                  l2_if,
    
    // Cache coherency interface
    input  logic                    snoop_req,
    input  logic [ADDR_WIDTH-1:0]   snoop_addr,
    input  logic [2:0]              snoop_type,
    output logic                    snoop_hit,
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
    logic [1:0]             mesi_state [SETS-1:0][WAYS-1:0];  // MESI coherency
    
    // LRU replacement
    logic [$clog2(WAYS)-1:0] lru [SETS-1:0];
    
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
        SNOOP_CHECK
    } cache_state_t;
    
    cache_state_t state, next_state;
    
    // Miss handling
    logic                   miss_pending;
    logic [ADDR_WIDTH-1:0]  miss_addr;
    logic [$clog2(WAYS)-1:0] replace_way;
    
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
    
    // LRU update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < SETS; s++) begin
                lru[s] <= '0;
            end
        end else if (cpu_req && cache_hit) begin
            for (int w = 0; w < WAYS; w++) begin
                if (hit_way[w]) begin
                    lru[index] <= w;
                end
            end
        end
    end
    
    // AXI interface
    assign l2_if.arid = '0;
    assign l2_if.araddr = {miss_addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
    assign l2_if.arlen = (LINE_SIZE / (l2_if.DATA_WIDTH/8)) - 1;
    assign l2_if.arsize = $clog2(l2_if.DATA_WIDTH/8);
    assign l2_if.arburst = 2'b01; // INCR
    assign l2_if.arlock = 1'b0;
    assign l2_if.arcache = 4'b0010; // Normal non-cacheable bufferable
    assign l2_if.arprot = 3'b000;   // Data, secure, unprivileged
    assign l2_if.arqos = 4'b0000;   // No QoS
    assign l2_if.arvalid = (state == MISS_REQ);
    assign l2_if.rready = 1'b1;
    
    // Write interface (not used for I-cache)
    assign l2_if.awvalid = 1'b0;
    assign l2_if.wvalid = 1'b0;
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
    
    // Replacement way selection (LRU)
    assign replace_way = lru[miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS]];
    
    // Cache fill with burst support
    logic [2:0] fill_beat_count;
    logic [511:0] fill_buffer;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_beat_count <= '0;
            fill_buffer <= '0;
        end else if (l2_if.rvalid && miss_pending) begin
            logic [INDEX_BITS-1:0] miss_index;
            miss_index = miss_addr[INDEX_BITS+OFFSET_BITS-1:OFFSET_BITS];
            
            // Accumulate data for multi-beat transfers
            if (l2_if.DATA_WIDTH == 512) begin
                // Single beat transfer
                data_array[miss_index][replace_way] <= l2_if.rdata;
            end else begin
                // Multi-beat transfer - accumulate data
                fill_buffer[fill_beat_count * l2_if.DATA_WIDTH +: l2_if.DATA_WIDTH] <= l2_if.rdata;
                fill_beat_count <= fill_beat_count + 1;
                
                if (l2_if.rlast) begin
                    data_array[miss_index][replace_way] <= fill_buffer;
                    fill_beat_count <= '0;
                end
            end
            
            if (l2_if.rlast) begin
                tag_array[miss_index][replace_way] <= miss_addr[ADDR_WIDTH-1:INDEX_BITS+OFFSET_BITS];
                valid[miss_index][replace_way] <= 1'b1;
                mesi_state[miss_index][replace_way] <= 2'b10; // Exclusive
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
    
    // Snoop response
    always_comb begin
        snoop_resp = 3'b000;
        if (snoop_hit) begin
            case (snoop_type)
                3'b001: snoop_resp = 3'b010; // Read hit
                3'b010: snoop_resp = 3'b011; // Invalidate
                default: snoop_resp = 3'b000;
            endcase
        end
    end
    
    // Performance counters
    logic [31:0] perf_hits;
    logic [31:0] perf_misses;
    logic [31:0] perf_accesses;
    logic [31:0] perf_snoop_hits;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_hits <= '0;
            perf_misses <= '0;
            perf_accesses <= '0;
            perf_snoop_hits <= '0;
        end else begin
            if (cpu_req) begin
                perf_accesses <= perf_accesses + 1;
                if (cache_hit) begin
                    perf_hits <= perf_hits + 1;
                end else begin
                    perf_misses <= perf_misses + 1;
                end
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
                mesi_state[s][w] = 2'b00; // Invalid
            end
        end
    end

endmodule