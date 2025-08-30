// HBM2E Memory Controller
// Supports 4 channels with advanced scheduling and bandwidth optimization
// Provides 1.6TB/s theoretical peak bandwidth

module hbm_controller #(
    parameter NUM_CHANNELS = 4,           // 4 HBM channels
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,           // 512-bit AXI interface
    parameter HBM_DATA_WIDTH = 1024,      // 1024-bit HBM interface per channel
    parameter BURST_LENGTH = 4,           // HBM burst length
    parameter NUM_BANKS = 16,             // Banks per channel
    parameter NUM_BANK_GROUPS = 4         // Bank groups per channel
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // AXI interface from L3 cache
    axi4_if.slave       axi_if,
    
    // HBM physical interfaces (4 channels)
    output logic [NUM_CHANNELS-1:0]                    hbm_clk_p,
    output logic [NUM_CHANNELS-1:0]                    hbm_clk_n,
    output logic [NUM_CHANNELS-1:0]                    hbm_rst_n,
    
    // HBM command interfaces
    output logic [NUM_CHANNELS-1:0][5:0]              hbm_cmd,
    output logic [NUM_CHANNELS-1:0][33:0]             hbm_addr,
    output logic [NUM_CHANNELS-1:0]                    hbm_cmd_valid,
    input  logic [NUM_CHANNELS-1:0]                    hbm_cmd_ready,
    
    // HBM write data interfaces
    output logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH-1:0] hbm_wdata,
    output logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH/8-1:0] hbm_wstrb,
    output logic [NUM_CHANNELS-1:0]                    hbm_wvalid,
    input  logic [NUM_CHANNELS-1:0]                    hbm_wready,
    
    // HBM read data interfaces
    input  logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH-1:0] hbm_rdata,
    input  logic [NUM_CHANNELS-1:0]                    hbm_rvalid,
    output logic [NUM_CHANNELS-1:0]                    hbm_rready,
    
    // Performance monitoring
    output logic [31:0]             read_requests,
    output logic [31:0]             write_requests,
    output logic [31:0]             total_bandwidth,
    output logic [15:0]             avg_latency,
    
    // Advanced performance metrics
    output logic [31:0]             bank_conflict_count,
    output logic [31:0]             row_hit_rate,
    output logic [15:0]             instantaneous_bandwidth,
    output logic [31:0]             peak_bandwidth,
    output logic [15:0]             min_access_latency,
    output logic [15:0]             max_access_latency,
    output logic [7:0]              queue_utilization,
    output logic [15:0]             power_consumption,
    
    // Error reporting
    output logic                    ecc_error,
    output logic [ADDR_WIDTH-1:0]   error_addr,
    output logic [2:0]              error_type
);

    // Memory controller parameters
    localparam CHANNEL_ADDR_BITS = $clog2(NUM_CHANNELS);
    localparam BANK_ADDR_BITS = $clog2(NUM_BANKS);
    localparam BG_ADDR_BITS = $clog2(NUM_BANK_GROUPS);
    localparam ROW_ADDR_BITS = 15;  // HBM row address width
    localparam COL_ADDR_BITS = 7;   // HBM column address width
    
    // Address mapping
    logic [CHANNEL_ADDR_BITS-1:0]  channel_addr;
    logic [BG_ADDR_BITS-1:0]       bg_addr;
    logic [BANK_ADDR_BITS-1:0]     bank_addr;
    logic [ROW_ADDR_BITS-1:0]      row_addr;
    logic [COL_ADDR_BITS-1:0]      col_addr;
    
    // Request queue for each channel
    typedef struct packed {
        logic [ADDR_WIDTH-1:0]      addr;
        logic [DATA_WIDTH-1:0]      wdata;
        logic [DATA_WIDTH/8-1:0]    wstrb;
        logic [7:0]                 id;
        logic [7:0]                 len;
        logic [2:0]                 size;
        logic                       is_write;
        logic                       valid;
        logic [15:0]                timestamp;
    } mem_request_t;
    
    localparam QUEUE_DEPTH = 16;
    mem_request_t request_queue [NUM_CHANNELS-1:0][QUEUE_DEPTH-1:0];
    logic [$clog2(QUEUE_DEPTH):0] queue_head [NUM_CHANNELS-1:0];
    logic [$clog2(QUEUE_DEPTH):0] queue_tail [NUM_CHANNELS-1:0];
    logic [NUM_CHANNELS-1:0] queue_full;
    logic [NUM_CHANNELS-1:0] queue_empty;
    
    // Bank state tracking for each channel
    typedef struct packed {
        logic                       active;
        logic [ROW_ADDR_BITS-1:0]   open_row;
        logic [15:0]                last_access_time;
        logic                       precharge_pending;
    } bank_state_t;
    
    bank_state_t bank_state [NUM_CHANNELS-1:0][NUM_BANKS-1:0];
    
    // Advanced scheduler state machine for each channel
    typedef enum logic [3:0] {
        IDLE,
        DECODE_REQ,
        ACTIVATE,
        READ_WRITE,
        PRECHARGE,
        REFRESH,
        REORDER_QUEUE,
        BANK_CONFLICT_RESOLVE,
        POWER_DOWN
    } scheduler_state_t;
    
    scheduler_state_t sched_state [NUM_CHANNELS-1:0];
    
    // Current request being processed per channel
    mem_request_t current_req [NUM_CHANNELS-1:0];
    logic [NUM_CHANNELS-1:0] req_active;
    
    // Refresh management
    logic [15:0] refresh_counter;
    logic [NUM_CHANNELS-1:0] refresh_pending;
    logic [NUM_CHANNELS-1:0] refresh_active;
    
    // Enhanced performance counters and monitoring
    logic [31:0] read_count;
    logic [31:0] write_count;
    logic [31:0] cycle_count;
    logic [47:0] latency_sum;
    logic [31:0] request_count;
    
    // Advanced performance metrics
    logic [31:0] bank_conflicts;
    logic [31:0] row_buffer_hits;
    logic [31:0] row_buffer_misses;
    logic [31:0] precharge_cycles;
    logic [31:0] activate_cycles;
    logic [31:0] refresh_cycles;
    logic [31:0] idle_cycles;
    logic [31:0] queue_full_cycles;
    
    // Bandwidth utilization tracking
    logic [31:0] data_cycles;
    logic [31:0] total_cycles;
    logic [15:0] instantaneous_bw;
    logic [31:0] peak_bw_achieved;
    
    // Latency distribution tracking
    logic [31:0] latency_buckets [0:15];  // 16 latency buckets
    logic [15:0] min_latency;
    logic [15:0] max_latency;
    
    // Power and thermal monitoring
    logic [15:0] power_estimate;
    logic [7:0]  temperature_sensor;
    
    // Advanced address mapping function (optimized for HBM)
    function automatic void decode_address(
        input logic [ADDR_WIDTH-1:0] addr,
        output logic [CHANNEL_ADDR_BITS-1:0] channel,
        output logic [BG_ADDR_BITS-1:0] bg,
        output logic [BANK_ADDR_BITS-1:0] bank,
        output logic [ROW_ADDR_BITS-1:0] row,
        output logic [COL_ADDR_BITS-1:0] col
    );
        // Optimized interleaving for AI workloads
        // Distribute sequential accesses across channels and banks
        channel = addr[8:6];  // Bits [8:6] for 4 channels
        bg = addr[5:4];       // Bank group
        bank = addr[11:9];    // Bank within group
        row = addr[26:12];    // Row address
        col = addr[19:13];    // Column address (cache line aligned)
    endfunction
    
    // Request priority calculation for advanced scheduling
    function automatic logic [7:0] calculate_priority(
        input mem_request_t req,
        input logic [15:0] current_time,
        input bank_state_t bank_st
    );
        logic [7:0] priority;
        logic [7:0] age_factor;
        logic [7:0] locality_factor;
        logic [7:0] type_factor;
        
        // Age-based priority (older requests get higher priority)
        age_factor = (current_time - req.timestamp) >> 4;
        
        // Row buffer locality (hits get higher priority)
        if (bank_st.active && bank_st.open_row == req.addr[26:12]) begin
            locality_factor = 8'h80;  // High priority for row buffer hits
        end else begin
            locality_factor = 8'h20;  // Lower priority for misses
        end
        
        // Request type priority (reads slightly higher than writes)
        type_factor = req.is_write ? 8'h40 : 8'h60;
        
        // Combine factors
        priority = (age_factor >> 2) + (locality_factor >> 2) + (type_factor >> 2);
        
        return priority;
    endfunction
    
    // Advanced queue reordering for better performance
    function automatic integer find_best_request(
        input mem_request_t queue [QUEUE_DEPTH-1:0],
        input logic [$clog2(QUEUE_DEPTH):0] head,
        input logic [$clog2(QUEUE_DEPTH):0] tail,
        input bank_state_t banks [NUM_BANKS-1:0],
        input logic [15:0] current_time
    );
        integer best_idx;
        logic [7:0] best_priority;
        logic [7:0] current_priority;
        logic [BANK_ADDR_BITS-1:0] bank_addr;
        
        best_idx = head;
        best_priority = 8'h00;
        
        for (integer i = head; i != tail; i = (i + 1) % QUEUE_DEPTH) begin
            if (queue[i].valid) begin
                bank_addr = queue[i].addr[11:9];
                current_priority = calculate_priority(queue[i], current_time, banks[bank_addr]);
                
                if (current_priority > best_priority) begin
                    best_priority = current_priority;
                    best_idx = i;
                end
            end
        end
        
        return best_idx;
    endfunction
    
    // Queue management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                queue_head[ch] <= '0;
                queue_tail[ch] <= '0;
            end
        end else begin
            // Enqueue new requests
            if (axi_if.arvalid && axi_if.arready) begin
                decode_address(axi_if.araddr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
                
                if (!queue_full[channel_addr]) begin
                    request_queue[channel_addr][queue_tail[channel_addr]].addr <= axi_if.araddr;
                    request_queue[channel_addr][queue_tail[channel_addr]].id <= axi_if.arid;
                    request_queue[channel_addr][queue_tail[channel_addr]].len <= axi_if.arlen;
                    request_queue[channel_addr][queue_tail[channel_addr]].size <= axi_if.arsize;
                    request_queue[channel_addr][queue_tail[channel_addr]].is_write <= 1'b0;
                    request_queue[channel_addr][queue_tail[channel_addr]].valid <= 1'b1;
                    request_queue[channel_addr][queue_tail[channel_addr]].timestamp <= cycle_count[15:0];
                    
                    queue_tail[channel_addr] <= queue_tail[channel_addr] + 1;
                end
            end
            
            if (axi_if.awvalid && axi_if.awready) begin
                decode_address(axi_if.awaddr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
                
                if (!queue_full[channel_addr]) begin
                    request_queue[channel_addr][queue_tail[channel_addr]].addr <= axi_if.awaddr;
                    request_queue[channel_addr][queue_tail[channel_addr]].id <= axi_if.awid;
                    request_queue[channel_addr][queue_tail[channel_addr]].len <= axi_if.awlen;
                    request_queue[channel_addr][queue_tail[channel_addr]].size <= axi_if.awsize;
                    request_queue[channel_addr][queue_tail[channel_addr]].is_write <= 1'b1;
                    request_queue[channel_addr][queue_tail[channel_addr]].valid <= 1'b1;
                    request_queue[channel_addr][queue_tail[channel_addr]].timestamp <= cycle_count[15:0];
                    
                    queue_tail[channel_addr] <= queue_tail[channel_addr] + 1;
                end
            end
            
            // Capture write data
            if (axi_if.wvalid && axi_if.wready) begin
                // Find the corresponding write request and update data
                for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    for (int i = 0; i < QUEUE_DEPTH; i++) begin
                        if (request_queue[ch][i].valid && request_queue[ch][i].is_write && 
                            request_queue[ch][i].id == axi_if.awid) begin
                            request_queue[ch][i].wdata <= axi_if.wdata;
                            request_queue[ch][i].wstrb <= axi_if.wstrb;
                        end
                    end
                end
            end
            
            // Dequeue completed requests
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if (req_active[ch] && 
                    ((current_req[ch].is_write && hbm_wvalid[ch] && hbm_wready[ch]) ||
                     (!current_req[ch].is_write && hbm_rvalid[ch] && hbm_rready[ch]))) begin
                    
                    queue_head[ch] <= queue_head[ch] + 1;
                    req_active[ch] <= 1'b0;
                end
            end
        end
    end
    
    // Queue status
    generate
        for (genvar ch = 0; ch < NUM_CHANNELS; ch++) begin : queue_status
            assign queue_full[ch] = (queue_tail[ch] - queue_head[ch]) >= QUEUE_DEPTH;
            assign queue_empty[ch] = (queue_tail[ch] == queue_head[ch]);
        end
    endgenerate
    
    // Memory scheduler for each channel
    generate
        for (genvar ch = 0; ch < NUM_CHANNELS; ch++) begin : channel_scheduler
            
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    sched_state[ch] <= IDLE;
                    req_active[ch] <= 1'b0;
                    refresh_active[ch] <= 1'b0;
                    
                    // Initialize bank states
                    for (int b = 0; b < NUM_BANKS; b++) begin
                        bank_state[ch][b].active <= 1'b0;
                        bank_state[ch][b].precharge_pending <= 1'b0;
                    end
                    
                end else begin
                    case (sched_state[ch])
                        IDLE: begin
                            if (refresh_pending[ch]) begin
                                sched_state[ch] <= REFRESH;
                                refresh_active[ch] <= 1'b1;
                            end else if (!queue_empty[ch] && !req_active[ch]) begin
                                // Use advanced scheduling to find best request
                                sched_state[ch] <= REORDER_QUEUE;
                            end
                        end
                        
                        REORDER_QUEUE: begin
                            integer best_req_idx;
                            best_req_idx = find_best_request(
                                request_queue[ch], 
                                queue_head[ch], 
                                queue_tail[ch],
                                bank_state[ch],
                                cycle_count[15:0]
                            );
                            
                            current_req[ch] <= request_queue[ch][best_req_idx];
                            req_active[ch] <= 1'b1;
                            sched_state[ch] <= DECODE_REQ;
                        end
                        
                        DECODE_REQ: begin
                            decode_address(current_req[ch].addr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
                            
                            // Check for bank conflicts and optimize
                            if (bank_state[ch][bank_addr].active && 
                                bank_state[ch][bank_addr].open_row == row_addr) begin
                                // Row buffer hit - proceed directly
                                row_buffer_hits <= row_buffer_hits + 1;
                                sched_state[ch] <= READ_WRITE;
                            end else if (bank_state[ch][bank_addr].active) begin
                                // Row buffer miss - need to precharge first
                                row_buffer_misses <= row_buffer_misses + 1;
                                bank_conflicts <= bank_conflicts + 1;
                                sched_state[ch] <= BANK_CONFLICT_RESOLVE;
                            end else begin
                                // Bank not active - need to activate
                                sched_state[ch] <= ACTIVATE;
                            end
                        end
                        
                        BANK_CONFLICT_RESOLVE: begin
                            // Intelligent precharge decision
                            // Check if other pending requests can use the open row
                            logic should_precharge;
                            should_precharge = 1'b1;
                            
                            // Look ahead in queue for row buffer hits
                            for (int i = 0; i < QUEUE_DEPTH; i++) begin
                                if (request_queue[ch][i].valid && 
                                    request_queue[ch][i].addr[11:9] == bank_addr &&
                                    request_queue[ch][i].addr[26:12] == bank_state[ch][bank_addr].open_row) begin
                                    should_precharge = 1'b0;  // Keep row open
                                end
                            end
                            
                            if (should_precharge) begin
                                sched_state[ch] <= PRECHARGE;
                            end else begin
                                // Delay this request, try another
                                sched_state[ch] <= IDLE;
                                req_active[ch] <= 1'b0;
                            end
                        end
                        
                        ACTIVATE: begin
                            if (hbm_cmd_ready[ch]) begin
                                bank_state[ch][bank_addr].active <= 1'b1;
                                bank_state[ch][bank_addr].open_row <= row_addr;
                                bank_state[ch][bank_addr].last_access_time <= cycle_count[15:0];
                                sched_state[ch] <= READ_WRITE;
                            end
                        end
                        
                        READ_WRITE: begin
                            if (hbm_cmd_ready[ch]) begin
                                bank_state[ch][bank_addr].last_access_time <= cycle_count[15:0];
                                sched_state[ch] <= IDLE;
                            end
                        end
                        
                        PRECHARGE: begin
                            if (hbm_cmd_ready[ch]) begin
                                bank_state[ch][bank_addr].active <= 1'b0;
                                bank_state[ch][bank_addr].precharge_pending <= 1'b0;
                                sched_state[ch] <= ACTIVATE;
                            end
                        end
                        
                        REFRESH: begin
                            if (hbm_cmd_ready[ch]) begin
                                refresh_active[ch] <= 1'b0;
                                sched_state[ch] <= IDLE;
                            end
                        end
                    endcase
                end
            end
            
            // HBM command generation
            always_comb begin
                hbm_cmd[ch] = 6'b000000;  // NOP
                hbm_addr[ch] = '0;
                hbm_cmd_valid[ch] = 1'b0;
                
                if (req_active[ch]) begin
                    decode_address(current_req[ch].addr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
                    
                    case (sched_state[ch])
                        ACTIVATE: begin
                            hbm_cmd[ch] = 6'b001111;  // ACTIVATE
                            hbm_addr[ch] = {bg_addr, bank_addr, row_addr};
                            hbm_cmd_valid[ch] = 1'b1;
                        end
                        
                        READ_WRITE: begin
                            if (current_req[ch].is_write) begin
                                hbm_cmd[ch] = 6'b001100;  // WRITE
                            end else begin
                                hbm_cmd[ch] = 6'b001101;  // READ
                            end
                            hbm_addr[ch] = {bg_addr, bank_addr, 1'b0, col_addr, 6'b000000};
                            hbm_cmd_valid[ch] = 1'b1;
                        end
                        
                        PRECHARGE: begin
                            hbm_cmd[ch] = 6'b001010;  // PRECHARGE
                            hbm_addr[ch] = {bg_addr, bank_addr, 15'b0};
                            hbm_cmd_valid[ch] = 1'b1;
                        end
                        
                        REFRESH: begin
                            hbm_cmd[ch] = 6'b001000;  // REFRESH
                            hbm_cmd_valid[ch] = 1'b1;
                        end
                    endcase
                end
            end
            
            // HBM data interfaces
            assign hbm_wdata[ch] = {2{current_req[ch].wdata}};  // Replicate 512->1024 bits
            assign hbm_wstrb[ch] = {2{current_req[ch].wstrb}};
            assign hbm_wvalid[ch] = req_active[ch] && current_req[ch].is_write && 
                                   (sched_state[ch] == READ_WRITE) && hbm_cmd_ready[ch];
            assign hbm_rready[ch] = 1'b1;
            
        end
    endgenerate
    
    // HBM clock generation
    assign hbm_clk_p = {NUM_CHANNELS{clk}};
    assign hbm_clk_n = {NUM_CHANNELS{~clk}};
    assign hbm_rst_n = {NUM_CHANNELS{rst_n}};
    
    // AXI interface responses
    logic axi_ar_ready, axi_aw_ready, axi_w_ready;
    logic axi_r_valid, axi_b_valid;
    logic [DATA_WIDTH-1:0] axi_r_data;
    logic [7:0] axi_r_id, axi_b_id;
    
    // AXI ready signals based on queue availability
    always_comb begin
        axi_ar_ready = 1'b0;
        axi_aw_ready = 1'b0;
        
        if (axi_if.arvalid) begin
            decode_address(axi_if.araddr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
            axi_ar_ready = !queue_full[channel_addr];
        end
        
        if (axi_if.awvalid) begin
            decode_address(axi_if.awaddr, channel_addr, bg_addr, bank_addr, row_addr, col_addr);
            axi_aw_ready = !queue_full[channel_addr];
        end
    end
    
    assign axi_if.arready = axi_ar_ready;
    assign axi_if.awready = axi_aw_ready;
    assign axi_if.wready = axi_aw_ready;  // Simplified - assume write data comes with address
    
    // AXI read response (simplified - from any channel)
    always_comb begin
        axi_r_valid = 1'b0;
        axi_r_data = '0;
        axi_r_id = '0;
        
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            if (hbm_rvalid[ch] && req_active[ch] && !current_req[ch].is_write) begin
                axi_r_valid = 1'b1;
                axi_r_data = hbm_rdata[ch][DATA_WIDTH-1:0];  // Take lower 512 bits
                axi_r_id = current_req[ch].id;
            end
        end
    end
    
    assign axi_if.rvalid = axi_r_valid;
    assign axi_if.rdata = axi_r_data;
    assign axi_if.rid = axi_r_id;
    assign axi_if.rresp = 2'b00;  // OKAY
    assign axi_if.rlast = 1'b1;   // Single beat responses
    
    // AXI write response (simplified)
    always_comb begin
        axi_b_valid = 1'b0;
        axi_b_id = '0;
        
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            if (hbm_wvalid[ch] && hbm_wready[ch] && req_active[ch] && current_req[ch].is_write) begin
                axi_b_valid = 1'b1;
                axi_b_id = current_req[ch].id;
            end
        end
    end
    
    assign axi_if.bvalid = axi_b_valid;
    assign axi_if.bid = axi_b_id;
    assign axi_if.bresp = 2'b00;  // OKAY
    
    // Refresh management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            refresh_counter <= '0;
            refresh_pending <= '0;
        end else begin
            refresh_counter <= refresh_counter + 1;
            
            // Generate refresh every 7.8us (assuming 1GHz clock)
            if (refresh_counter >= 16'd7800) begin
                refresh_counter <= '0;
                refresh_pending <= '1;
            end
            
            // Clear refresh pending when refresh completes
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if (refresh_active[ch] && hbm_cmd_ready[ch] && (sched_state[ch] == REFRESH)) begin
                    refresh_pending[ch] <= 1'b0;
                end
            end
        end
    end
    
    // Enhanced performance monitoring and bandwidth tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_count <= '0;
            write_count <= '0;
            cycle_count <= '0;
            latency_sum <= '0;
            request_count <= '0;
            
            // Initialize advanced counters
            bank_conflicts <= '0;
            row_buffer_hits <= '0;
            row_buffer_misses <= '0;
            precharge_cycles <= '0;
            activate_cycles <= '0;
            refresh_cycles <= '0;
            idle_cycles <= '0;
            queue_full_cycles <= '0;
            data_cycles <= '0;
            total_cycles <= '0;
            peak_bw_achieved <= '0;
            min_latency <= 16'hFFFF;
            max_latency <= '0;
            
            // Initialize latency buckets
            for (int i = 0; i < 16; i++) begin
                latency_buckets[i] <= '0;
            end
            
        end else begin
            cycle_count <= cycle_count + 1;
            total_cycles <= total_cycles + 1;
            
            // Count requests and track bandwidth
            if (axi_if.arvalid && axi_if.arready) begin
                read_count <= read_count + 1;
                request_count <= request_count + 1;
                data_cycles <= data_cycles + 1;
            end
            
            if (axi_if.awvalid && axi_if.awready) begin
                write_count <= write_count + 1;
                request_count <= request_count + 1;
                data_cycles <= data_cycles + 1;
            end
            
            // Track queue full conditions
            logic any_queue_full;
            any_queue_full = |queue_full;
            if (any_queue_full) begin
                queue_full_cycles <= queue_full_cycles + 1;
            end
            
            // Track scheduler states for performance analysis
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                case (sched_state[ch])
                    IDLE: idle_cycles <= idle_cycles + 1;
                    ACTIVATE: activate_cycles <= activate_cycles + 1;
                    PRECHARGE: precharge_cycles <= precharge_cycles + 1;
                    REFRESH: refresh_cycles <= refresh_cycles + 1;
                endcase
            end
            
            // Enhanced latency tracking with distribution
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if (req_active[ch] && 
                    ((current_req[ch].is_write && hbm_wvalid[ch] && hbm_wready[ch]) ||
                     (!current_req[ch].is_write && hbm_rvalid[ch]))) begin
                    
                    logic [15:0] latency;
                    logic [3:0] bucket_idx;
                    
                    latency = cycle_count[15:0] - current_req[ch].timestamp;
                    latency_sum <= latency_sum + latency;
                    
                    // Update min/max latency
                    if (latency < min_latency) min_latency <= latency;
                    if (latency > max_latency) max_latency <= latency;
                    
                    // Update latency distribution buckets
                    bucket_idx = (latency < 16) ? latency[3:0] : 4'hF;
                    latency_buckets[bucket_idx] <= latency_buckets[bucket_idx] + 1;
                end
            end
            
            // Calculate instantaneous bandwidth (every 1024 cycles)
            if (cycle_count[9:0] == 10'h3FF) begin
                logic [31:0] current_bw;
                current_bw = (data_cycles * 64 * 1000) / 1024;  // MB/s
                instantaneous_bw <= current_bw[15:0];
                
                if (current_bw > peak_bw_achieved) begin
                    peak_bw_achieved <= current_bw;
                end
                
                data_cycles <= '0;  // Reset for next measurement window
            end
            
            // Power estimation based on activity
            logic [7:0] activity_factor;
            activity_factor = 0;
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if (req_active[ch]) activity_factor = activity_factor + 32;
            end
            power_estimate <= 16'd1000 + {8'h0, activity_factor};  // Base + dynamic power
        end
    end
    
    // Enhanced performance outputs
    assign read_requests = read_count;
    assign write_requests = write_count;
    assign total_bandwidth = (total_cycles > 0) ? 
                           (data_cycles * 64 * 1000000) / total_cycles : 32'h0; // Bytes/sec
    assign avg_latency = (request_count > 0) ? latency_sum[31:0] / request_count : 16'h0;
    
    // Advanced performance metrics outputs
    assign bank_conflict_count = bank_conflicts;
    assign row_hit_rate = (row_buffer_hits + row_buffer_misses > 0) ? 
                         (row_buffer_hits * 100) / (row_buffer_hits + row_buffer_misses) : 32'h0;
    assign instantaneous_bandwidth = instantaneous_bw;
    assign peak_bandwidth = peak_bw_achieved;
    assign min_access_latency = min_latency;
    assign max_access_latency = max_latency;
    
    // Queue utilization calculation
    logic [31:0] total_queue_entries;
    always_comb begin
        total_queue_entries = 0;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            total_queue_entries = total_queue_entries + (queue_tail[ch] - queue_head[ch]);
        end
    end
    assign queue_utilization = (total_queue_entries * 100) / (NUM_CHANNELS * QUEUE_DEPTH);
    assign power_consumption = power_estimate;
    
    // Error detection (simplified)
    assign ecc_error = 1'b0;      // Would be connected to HBM ECC logic
    assign error_addr = '0;
    assign error_type = 3'b000;

endmodule