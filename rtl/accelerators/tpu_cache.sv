// TPU Cache System - Weight and Activation Caches
// Provides high-speed access to frequently used data
// Supports cache coherency and prefetching

`timescale 1ns/1ps

module tpu_cache #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter WEIGHT_CACHE_SIZE = 512*1024,  // 512KB weight cache
    parameter ACTIVATION_CACHE_SIZE = 256*1024,  // 256KB activation cache
    parameter CACHE_LINE_SIZE = 64,  // 64 bytes per cache line
    parameter ASSOCIATIVITY = 4      // 4-way set associative
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // CPU/Controller interface
    input  logic [ADDR_WIDTH-1:0]  cpu_addr,
    input  logic                    cpu_read,
    input  logic                    cpu_write,
    input  logic [DATA_WIDTH-1:0]  cpu_wdata,
    output logic [DATA_WIDTH-1:0]  cpu_rdata,
    output logic                    cpu_ready,
    output logic                    cpu_hit,
    
    // Memory interface
    output logic [ADDR_WIDTH-1:0]  mem_addr,
    output logic                    mem_read,
    output logic                    mem_write,
    output logic [DATA_WIDTH-1:0]  mem_wdata,
    input  logic [DATA_WIDTH-1:0]  mem_rdata,
    input  logic                    mem_ready,
    
    // Cache control
    input  logic                    cache_flush,
    input  logic                    cache_invalidate,
    input  logic [1:0]              cache_type,  // 00: weight, 01: activation, 10: unified
    output logic                    cache_ready,
    
    // Performance monitoring
    output logic [31:0]             hit_count,
    output logic [31:0]             miss_count,
    output logic [31:0]             eviction_count
);

    // Cache parameters
    localparam WEIGHT_CACHE_LINES = WEIGHT_CACHE_SIZE / CACHE_LINE_SIZE;
    localparam ACTIVATION_CACHE_LINES = ACTIVATION_CACHE_SIZE / CACHE_LINE_SIZE;
    localparam WEIGHT_SETS = WEIGHT_CACHE_LINES / ASSOCIATIVITY;
    localparam ACTIVATION_SETS = ACTIVATION_CACHE_LINES / ASSOCIATIVITY;
    
    localparam OFFSET_BITS = $clog2(CACHE_LINE_SIZE);
    localparam WEIGHT_INDEX_BITS = $clog2(WEIGHT_SETS);
    localparam ACTIVATION_INDEX_BITS = $clog2(ACTIVATION_SETS);
    localparam TAG_BITS = ADDR_WIDTH - OFFSET_BITS - WEIGHT_INDEX_BITS;
    
    // Weight cache storage - separate arrays for synthesis compatibility
    logic weight_cache_valid [WEIGHT_SETS-1:0][ASSOCIATIVITY-1:0];
    logic weight_cache_dirty [WEIGHT_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [TAG_BITS-1:0] weight_cache_tag [WEIGHT_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [CACHE_LINE_SIZE*8-1:0] weight_cache_data [WEIGHT_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [2:0] weight_cache_lru [WEIGHT_SETS-1:0][ASSOCIATIVITY-1:0];
    
    // Activation cache storage - separate arrays for synthesis compatibility
    logic activation_cache_valid [ACTIVATION_SETS-1:0][ASSOCIATIVITY-1:0];
    logic activation_cache_dirty [ACTIVATION_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [TAG_BITS-1:0] activation_cache_tag [ACTIVATION_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [CACHE_LINE_SIZE*8-1:0] activation_cache_data [ACTIVATION_SETS-1:0][ASSOCIATIVITY-1:0];
    logic [2:0] activation_cache_lru [ACTIVATION_SETS-1:0][ASSOCIATIVITY-1:0];
    
    // Address decomposition
    logic [OFFSET_BITS-1:0]     offset;
    logic [WEIGHT_INDEX_BITS-1:0] weight_index;
    logic [ACTIVATION_INDEX_BITS-1:0] activation_index;
    logic [TAG_BITS-1:0]        tag;
    
    // Cache access signals
    logic weight_cache_access;
    logic activation_cache_access;
    logic weight_hit, activation_hit;
    logic [1:0] weight_hit_way, activation_hit_way;
    logic [1:0] weight_victim_way, activation_victim_way;
    
    // LRU selection variables
    logic [2:0] weight_max_lru, activation_max_lru;
    logic weight_found_invalid, activation_found_invalid;
    
    // State machine
    typedef enum logic [2:0] {
        IDLE,
        CACHE_LOOKUP,
        MEMORY_ACCESS,
        CACHE_FILL,
        WRITEBACK,
        FLUSH_CACHE
    } cache_state_t;
    
    cache_state_t current_state, next_state;
    
    // Miss handling
    logic [ADDR_WIDTH-1:0] miss_addr;
    logic [DATA_WIDTH-1:0] fill_data;
    logic miss_pending;
    
    // Performance counters
    logic [31:0] hit_counter;
    logic [31:0] miss_counter;
    logic [31:0] eviction_counter;
    
    // Address decomposition
    always_comb begin
        offset = cpu_addr[OFFSET_BITS-1:0];
        weight_index = cpu_addr[OFFSET_BITS +: WEIGHT_INDEX_BITS];
        activation_index = cpu_addr[OFFSET_BITS +: ACTIVATION_INDEX_BITS];
        tag = cpu_addr[ADDR_WIDTH-1:OFFSET_BITS+WEIGHT_INDEX_BITS];
    end
    
    // Cache type determination
    always_comb begin
        case (cache_type)
            2'b00: begin  // Weight cache
                weight_cache_access = cpu_read || cpu_write;
                activation_cache_access = 1'b0;
            end
            2'b01: begin  // Activation cache
                weight_cache_access = 1'b0;
                activation_cache_access = cpu_read || cpu_write;
            end
            2'b10: begin  // Unified cache (use weight cache)
                weight_cache_access = cpu_read || cpu_write;
                activation_cache_access = 1'b0;
            end
            default: begin
                weight_cache_access = 1'b0;
                activation_cache_access = 1'b0;
            end
        endcase
    end
    
    // Weight cache hit detection
    always_comb begin
        weight_hit = 1'b0;
        weight_hit_way = 2'b00;
        
        if (weight_cache_access) begin
            for (int i = 0; i < ASSOCIATIVITY; i++) begin
                if (weight_cache_valid[weight_index][i] && 
                    weight_cache_tag[weight_index][i] == tag && !weight_hit) begin
                    weight_hit = 1'b1;
                    weight_hit_way = i[1:0];
                end
            end
        end
    end
    
    // Activation cache hit detection
    always_comb begin
        activation_hit = 1'b0;
        activation_hit_way = 2'b00;
        
        if (activation_cache_access) begin
            for (int i = 0; i < ASSOCIATIVITY; i++) begin
                if (activation_cache_valid[activation_index][i] && 
                    activation_cache_tag[activation_index][i] == tag && !activation_hit) begin
                    activation_hit = 1'b1;
                    activation_hit_way = i[1:0];
                end
            end
        end
    end
    
    // LRU victim selection for weight cache
    always_comb begin
        weight_victim_way = 2'b00;
        weight_max_lru = 3'b000;
        weight_found_invalid = 1'b0;
        
        for (int i = 0; i < ASSOCIATIVITY; i++) begin
            if (!weight_cache_valid[weight_index][i] && !weight_found_invalid) begin
                weight_victim_way = i[1:0];
                weight_found_invalid = 1'b1;
            end else if (weight_cache_lru[weight_index][i] > weight_max_lru && !weight_found_invalid) begin
                weight_max_lru = weight_cache_lru[weight_index][i];
                weight_victim_way = i[1:0];
            end
        end
    end
    
    // LRU victim selection for activation cache
    always_comb begin
        activation_victim_way = 2'b00;
        activation_max_lru = 3'b000;
        activation_found_invalid = 1'b0;
        
        for (int i = 0; i < ASSOCIATIVITY; i++) begin
            if (!activation_cache_valid[activation_index][i] && !activation_found_invalid) begin
                activation_victim_way = i[1:0];
                activation_found_invalid = 1'b1;
            end else if (activation_cache_lru[activation_index][i] > activation_max_lru && !activation_found_invalid) begin
                activation_max_lru = activation_cache_lru[activation_index][i];
                activation_victim_way = i[1:0];
            end
        end
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (cache_flush || cache_invalidate)
                    next_state = FLUSH_CACHE;
                else if (cpu_read || cpu_write)
                    next_state = CACHE_LOOKUP;
            end
            
            CACHE_LOOKUP: begin
                if (weight_hit || activation_hit)
                    next_state = IDLE;
                else
                    next_state = MEMORY_ACCESS;
            end
            
            MEMORY_ACCESS: begin
                if (mem_ready)
                    next_state = CACHE_FILL;
            end
            
            CACHE_FILL: begin
                next_state = IDLE;
            end
            
            WRITEBACK: begin
                if (mem_ready)
                    next_state = CACHE_FILL;
            end
            
            FLUSH_CACHE: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // Cache data access
    always_comb begin
        cpu_rdata = '0;
        cpu_hit = 1'b0;
        
        if (weight_cache_access && weight_hit) begin
            cpu_rdata = weight_cache_data[weight_index][weight_hit_way][offset*8 +: DATA_WIDTH];
            cpu_hit = 1'b1;
        end else if (activation_cache_access && activation_hit) begin
            cpu_rdata = activation_cache_data[activation_index][activation_hit_way][offset*8 +: DATA_WIDTH];
            cpu_hit = 1'b1;
        end else if (current_state == CACHE_FILL) begin
            cpu_rdata = mem_rdata;
        end
    end
    
    // Cache ready signal
    always_comb begin
        case (current_state)
            IDLE: cpu_ready = 1'b1;
            CACHE_LOOKUP: cpu_ready = weight_hit || activation_hit;
            CACHE_FILL: cpu_ready = 1'b1;
            default: cpu_ready = 1'b0;
        endcase
    end
    
    // Weight cache update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < WEIGHT_SETS; i++) begin
                for (int j = 0; j < ASSOCIATIVITY; j++) begin
                    weight_cache_valid[i][j] <= 1'b0;
                    weight_cache_dirty[i][j] <= 1'b0;
                    weight_cache_tag[i][j] <= '0;
                    weight_cache_data[i][j] <= '0;
                    weight_cache_lru[i][j] <= '0;
                end
            end
        end else begin
            case (current_state)
                CACHE_LOOKUP: begin
                    if (weight_cache_access) begin
                        if (weight_hit) begin
                            // Update LRU on hit
                            weight_cache_lru[weight_index][weight_hit_way] <= '0;
                            for (int i = 0; i < ASSOCIATIVITY; i++) begin
                                if (i != weight_hit_way && weight_cache_valid[weight_index][i]) begin
                                    weight_cache_lru[weight_index][i] <= weight_cache_lru[weight_index][i] + 1;
                                end
                            end
                            
                            // Handle write
                            if (cpu_write) begin
                                weight_cache_data[weight_index][weight_hit_way][offset*8 +: DATA_WIDTH] <= cpu_wdata;
                                weight_cache_dirty[weight_index][weight_hit_way] <= 1'b1;
                            end
                        end
                    end
                end
                
                CACHE_FILL: begin
                    if (weight_cache_access && !weight_hit) begin
                        // Fill cache line
                        weight_cache_valid[weight_index][weight_victim_way] <= 1'b1;
                        weight_cache_tag[weight_index][weight_victim_way] <= tag;
                        weight_cache_data[weight_index][weight_victim_way][offset*8 +: DATA_WIDTH] <= mem_rdata;
                        weight_cache_dirty[weight_index][weight_victim_way] <= cpu_write;
                        weight_cache_lru[weight_index][weight_victim_way] <= '0;
                        
                        // Update LRU counters
                        for (int i = 0; i < ASSOCIATIVITY; i++) begin
                            if (i != weight_victim_way && weight_cache_valid[weight_index][i]) begin
                                weight_cache_lru[weight_index][i] <= weight_cache_lru[weight_index][i] + 1;
                            end
                        end
                    end
                end
                
                FLUSH_CACHE: begin
                    if (cache_flush || cache_invalidate) begin
                        for (int i = 0; i < WEIGHT_SETS; i++) begin
                            for (int j = 0; j < ASSOCIATIVITY; j++) begin
                                if (cache_invalidate) begin
                                    weight_cache_valid[i][j] <= 1'b0;
                                end
                                weight_cache_dirty[i][j] <= 1'b0;
                            end
                        end
                    end
                end
            endcase
        end
    end
    
    // Activation cache update (similar to weight cache)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < ACTIVATION_SETS; i++) begin
                for (int j = 0; j < ASSOCIATIVITY; j++) begin
                    activation_cache_valid[i][j] <= 1'b0;
                    activation_cache_dirty[i][j] <= 1'b0;
                    activation_cache_tag[i][j] <= '0;
                    activation_cache_data[i][j] <= '0;
                    activation_cache_lru[i][j] <= '0;
                end
            end
        end else begin
            case (current_state)
                CACHE_LOOKUP: begin
                    if (activation_cache_access) begin
                        if (activation_hit) begin
                            // Update LRU on hit
                            activation_cache_lru[activation_index][activation_hit_way] <= '0;
                            for (int i = 0; i < ASSOCIATIVITY; i++) begin
                                if (i != activation_hit_way && activation_cache_valid[activation_index][i]) begin
                                    activation_cache_lru[activation_index][i] <= activation_cache_lru[activation_index][i] + 1;
                                end
                            end
                            
                            // Handle write
                            if (cpu_write) begin
                                activation_cache_data[activation_index][activation_hit_way][offset*8 +: DATA_WIDTH] <= cpu_wdata;
                                activation_cache_dirty[activation_index][activation_hit_way] <= 1'b1;
                            end
                        end
                    end
                end
                
                CACHE_FILL: begin
                    if (activation_cache_access && !activation_hit) begin
                        // Fill cache line
                        activation_cache_valid[activation_index][activation_victim_way] <= 1'b1;
                        activation_cache_tag[activation_index][activation_victim_way] <= tag;
                        activation_cache_data[activation_index][activation_victim_way][offset*8 +: DATA_WIDTH] <= mem_rdata;
                        activation_cache_dirty[activation_index][activation_victim_way] <= cpu_write;
                        activation_cache_lru[activation_index][activation_victim_way] <= '0;
                        
                        // Update LRU counters
                        for (int i = 0; i < ASSOCIATIVITY; i++) begin
                            if (i != activation_victim_way && activation_cache_valid[activation_index][i]) begin
                                activation_cache_lru[activation_index][i] <= activation_cache_lru[activation_index][i] + 1;
                            end
                        end
                    end
                end
                
                FLUSH_CACHE: begin
                    if (cache_flush || cache_invalidate) begin
                        for (int i = 0; i < ACTIVATION_SETS; i++) begin
                            for (int j = 0; j < ASSOCIATIVITY; j++) begin
                                if (cache_invalidate) begin
                                    activation_cache_valid[i][j] <= 1'b0;
                                end
                                activation_cache_dirty[i][j] <= 1'b0;
                            end
                        end
                    end
                end
            endcase
        end
    end
    
    // Memory interface
    always_comb begin
        mem_addr = cpu_addr;
        mem_read = (current_state == MEMORY_ACCESS) && !mem_ready;
        mem_write = 1'b0;  // Write-through for simplicity
        mem_wdata = cpu_wdata;
    end
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hit_counter <= '0;
            miss_counter <= '0;
            eviction_counter <= '0;
        end else begin
            case (current_state)
                CACHE_LOOKUP: begin
                    if (weight_hit || activation_hit) begin
                        hit_counter <= hit_counter + 1;
                    end else begin
                        miss_counter <= miss_counter + 1;
                    end
                end
                
                CACHE_FILL: begin
                    // Count evictions when replacing valid lines
                    if (weight_cache_access && weight_cache_valid[weight_index][weight_victim_way]) begin
                        eviction_counter <= eviction_counter + 1;
                    end else if (activation_cache_access && activation_cache_valid[activation_index][activation_victim_way]) begin
                        eviction_counter <= eviction_counter + 1;
                    end
                end
            endcase
        end
    end
    
    // Output assignments
    assign hit_count = hit_counter;
    assign miss_count = miss_counter;
    assign eviction_count = eviction_counter;
    assign cache_ready = (current_state == IDLE);

endmodule