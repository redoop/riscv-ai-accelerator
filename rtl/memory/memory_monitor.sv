// Memory Performance Monitor
// Advanced monitoring and analysis of memory subsystem performance
// Provides real-time bandwidth, latency, and efficiency metrics

module memory_monitor #(
    parameter NUM_CHANNELS = 4,
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,
    parameter MONITOR_DEPTH = 1024
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Memory controller interface monitoring
    input  logic                    mem_req_valid,
    input  logic                    mem_req_ready,
    input  logic [ADDR_WIDTH-1:0]   mem_req_addr,
    input  logic                    mem_req_write,
    input  logic [7:0]              mem_req_id,
    input  logic [15:0]             mem_req_timestamp,
    
    input  logic                    mem_resp_valid,
    input  logic                    mem_resp_ready,
    input  logic [7:0]              mem_resp_id,
    input  logic [15:0]             mem_resp_timestamp,
    
    // HBM interface monitoring
    input  logic [NUM_CHANNELS-1:0] hbm_cmd_valid,
    input  logic [NUM_CHANNELS-1:0] hbm_cmd_ready,
    input  logic [NUM_CHANNELS-1:0][5:0] hbm_cmd,
    input  logic [NUM_CHANNELS-1:0] hbm_wvalid,
    input  logic [NUM_CHANNELS-1:0] hbm_wready,
    input  logic [NUM_CHANNELS-1:0] hbm_rvalid,
    input  logic [NUM_CHANNELS-1:0] hbm_rready,
    
    // Performance metrics outputs
    output logic [31:0]             current_bandwidth,      // MB/s
    output logic [31:0]             peak_bandwidth,         // MB/s
    output logic [31:0]             average_bandwidth,      // MB/s
    output logic [15:0]             current_latency,        // cycles
    output logic [15:0]             min_latency,           // cycles
    output logic [15:0]             max_latency,           // cycles
    output logic [15:0]             avg_latency,           // cycles
    output logic [7:0]              channel_utilization [NUM_CHANNELS-1:0], // %
    output logic [31:0]             total_requests,
    output logic [31:0]             total_responses,
    output logic [31:0]             outstanding_requests,
    
    // Efficiency metrics
    output logic [31:0]             row_buffer_hits,
    output logic [31:0]             row_buffer_misses,
    output logic [7:0]              hit_rate_percent,
    output logic [31:0]             bank_conflicts,
    output logic [31:0]             channel_conflicts,
    
    // Latency distribution (histogram)
    output logic [31:0]             latency_histogram [0:15],
    
    // Bandwidth utilization over time
    output logic [15:0]             bandwidth_history [0:63],
    output logic [5:0]              history_index,
    
    // Error and anomaly detection
    output logic                    bandwidth_anomaly,
    output logic                    latency_anomaly,
    output logic                    stall_detected,
    output logic [15:0]             stall_duration,
    
    // Configuration and control
    input  logic                    monitor_enable,
    input  logic                    reset_counters,
    input  logic [15:0]             measurement_window,     // cycles
    input  logic [31:0]             bandwidth_threshold,    // MB/s
    input  logic [15:0]             latency_threshold       // cycles
);

    // Internal counters and state
    logic [31:0] cycle_counter;
    logic [31:0] measurement_cycle;
    logic [31:0] window_requests;
    logic [31:0] window_responses;
    logic [47:0] window_latency_sum;
    logic [31:0] window_data_bytes;
    
    // Request tracking for latency measurement
    typedef struct packed {
        logic [7:0]  id;
        logic [15:0] timestamp;
        logic        valid;
        logic        is_write;
    } request_entry_t;
    
    request_entry_t request_tracker [MONITOR_DEPTH-1:0];
    logic [$clog2(MONITOR_DEPTH)-1:0] tracker_head;
    logic [$clog2(MONITOR_DEPTH)-1:0] tracker_tail;
    
    // Channel activity tracking
    logic [31:0] channel_active_cycles [NUM_CHANNELS-1:0];
    logic [31:0] channel_requests [NUM_CHANNELS-1:0];
    logic [31:0] channel_responses [NUM_CHANNELS-1:0];
    
    // Bank state tracking for row buffer analysis
    typedef struct packed {
        logic [14:0] open_row;
        logic        active;
        logic [15:0] last_access;
    } bank_state_t;
    
    bank_state_t bank_states [NUM_CHANNELS-1:0][15:0];  // 16 banks per channel
    
    // Performance history buffers
    logic [15:0] latency_samples [0:255];
    logic [7:0]  latency_sample_idx;
    logic [31:0] bandwidth_samples [0:255];
    logic [7:0]  bandwidth_sample_idx;
    
    // Anomaly detection state
    logic [31:0] stall_start_cycle;
    logic        in_stall;
    logic [15:0] consecutive_low_bw;
    logic [15:0] consecutive_high_lat;
    
    // Address decoding for bank analysis
    function automatic void decode_hbm_address(
        input logic [ADDR_WIDTH-1:0] addr,
        output logic [1:0] channel,
        output logic [3:0] bank,
        output logic [14:0] row
    );
        channel = addr[7:6];   // Channel bits
        bank = addr[11:8];     // Bank bits
        row = addr[26:12];     // Row bits
    endfunction
    
    // Main monitoring logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all counters and state
            cycle_counter <= '0;
            measurement_cycle <= '0;
            window_requests <= '0;
            window_responses <= '0;
            window_latency_sum <= '0;
            window_data_bytes <= '0;
            tracker_head <= '0;
            tracker_tail <= '0;
            latency_sample_idx <= '0;
            bandwidth_sample_idx <= '0;
            stall_start_cycle <= '0;
            in_stall <= 1'b0;
            consecutive_low_bw <= '0;
            consecutive_high_lat <= '0;
            
            // Initialize arrays
            for (int i = 0; i < MONITOR_DEPTH; i++) begin
                request_tracker[i].valid <= 1'b0;
            end
            
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                channel_active_cycles[ch] <= '0;
                channel_requests[ch] <= '0;
                channel_responses[ch] <= '0;
                
                for (int b = 0; b < 16; b++) begin
                    bank_states[ch][b].active <= 1'b0;
                end
            end
            
            for (int i = 0; i < 16; i++) begin
                latency_histogram[i] <= '0;
            end
            
            for (int i = 0; i < 64; i++) begin
                bandwidth_history[i] <= '0;
            end
            
        end else if (monitor_enable) begin
            cycle_counter <= cycle_counter + 1;
            measurement_cycle <= measurement_cycle + 1;
            
            // Track memory requests
            if (mem_req_valid && mem_req_ready) begin
                // Add to request tracker
                request_tracker[tracker_tail].id <= mem_req_id;
                request_tracker[tracker_tail].timestamp <= cycle_counter[15:0];
                request_tracker[tracker_tail].valid <= 1'b1;
                request_tracker[tracker_tail].is_write <= mem_req_write;
                tracker_tail <= tracker_tail + 1;
                
                window_requests <= window_requests + 1;
                total_requests <= total_requests + 1;
                
                // Analyze bank access patterns
                logic [1:0] ch;
                logic [3:0] bank;
                logic [14:0] row;
                decode_hbm_address(mem_req_addr, ch, bank, row);
                
                if (bank_states[ch][bank].active && bank_states[ch][bank].open_row == row) begin
                    row_buffer_hits <= row_buffer_hits + 1;
                end else begin
                    row_buffer_misses <= row_buffer_misses + 1;
                    if (bank_states[ch][bank].active) begin
                        bank_conflicts <= bank_conflicts + 1;
                    end
                end
                
                // Update bank state
                bank_states[ch][bank].active <= 1'b1;
                bank_states[ch][bank].open_row <= row;
                bank_states[ch][bank].last_access <= cycle_counter[15:0];
                
                channel_requests[ch] <= channel_requests[ch] + 1;
            end
            
            // Track memory responses and calculate latency
            if (mem_resp_valid && mem_resp_ready) begin
                // Find matching request
                for (int i = 0; i < MONITOR_DEPTH; i++) begin
                    if (request_tracker[i].valid && request_tracker[i].id == mem_resp_id) begin
                        logic [15:0] latency;
                        logic [3:0] hist_bucket;
                        
                        latency = cycle_counter[15:0] - request_tracker[i].timestamp;
                        window_latency_sum <= window_latency_sum + latency;
                        window_responses <= window_responses + 1;
                        total_responses <= total_responses + 1;
                        
                        // Update latency statistics
                        if (latency < min_latency || min_latency == 0) begin
                            min_latency <= latency;
                        end
                        if (latency > max_latency) begin
                            max_latency <= latency;
                        end
                        
                        // Update latency histogram
                        hist_bucket = (latency < 16) ? latency[3:0] : 4'hF;
                        latency_histogram[hist_bucket] <= latency_histogram[hist_bucket] + 1;
                        
                        // Store in sample buffer
                        latency_samples[latency_sample_idx] <= latency;
                        latency_sample_idx <= latency_sample_idx + 1;
                        
                        // Mark request as completed
                        request_tracker[i].valid <= 1'b0;
                        
                        // Count data bytes transferred
                        window_data_bytes <= window_data_bytes + (DATA_WIDTH / 8);
                        
                        break;
                    end
                end
            end
            
            // Track HBM channel activity
            for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                if (hbm_cmd_valid[ch] || hbm_wvalid[ch] || hbm_rvalid[ch]) begin
                    channel_active_cycles[ch] <= channel_active_cycles[ch] + 1;
                end
                
                if (hbm_rvalid[ch] && hbm_rready[ch]) begin
                    channel_responses[ch] <= channel_responses[ch] + 1;
                end
            end
            
            // Measurement window processing
            if (measurement_cycle >= measurement_window) begin
                logic [31:0] window_bandwidth;
                logic [15:0] window_avg_latency;
                
                // Calculate bandwidth for this window (MB/s)
                window_bandwidth = (window_data_bytes * 1000) / measurement_window;
                
                // Calculate average latency for this window
                window_avg_latency = (window_responses > 0) ? 
                                   window_latency_sum[31:0] / window_responses : 16'h0;
                
                // Update current metrics
                current_bandwidth <= window_bandwidth;
                current_latency <= window_avg_latency;
                
                // Update peak bandwidth
                if (window_bandwidth > peak_bandwidth) begin
                    peak_bandwidth <= window_bandwidth;
                end
                
                // Store in history buffers
                bandwidth_history[history_index] <= window_bandwidth[15:0];
                history_index <= history_index + 1;
                
                bandwidth_samples[bandwidth_sample_idx] <= window_bandwidth;
                bandwidth_sample_idx <= bandwidth_sample_idx + 1;
                
                // Anomaly detection
                if (window_bandwidth < bandwidth_threshold) begin
                    consecutive_low_bw <= consecutive_low_bw + 1;
                    if (consecutive_low_bw >= 16'd10) begin
                        bandwidth_anomaly <= 1'b1;
                    end
                end else begin
                    consecutive_low_bw <= '0;
                    bandwidth_anomaly <= 1'b0;
                end
                
                if (window_avg_latency > latency_threshold) begin
                    consecutive_high_lat <= consecutive_high_lat + 1;
                    if (consecutive_high_lat >= 16'd10) begin
                        latency_anomaly <= 1'b1;
                    end
                end else begin
                    consecutive_high_lat <= '0;
                    latency_anomaly <= 1'b0;
                end
                
                // Stall detection
                if (window_requests == 0 && window_responses == 0) begin
                    if (!in_stall) begin
                        stall_start_cycle <= cycle_counter;
                        in_stall <= 1'b1;
                    end
                    stall_duration <= cycle_counter - stall_start_cycle;
                    stall_detected <= 1'b1;
                end else begin
                    in_stall <= 1'b0;
                    stall_detected <= 1'b0;
                    stall_duration <= '0;
                end
                
                // Reset window counters
                measurement_cycle <= '0;
                window_requests <= '0;
                window_responses <= '0;
                window_latency_sum <= '0;
                window_data_bytes <= '0;
            end
            
            // Reset counters if requested
            if (reset_counters) begin
                total_requests <= '0;
                total_responses <= '0;
                row_buffer_hits <= '0;
                row_buffer_misses <= '0;
                bank_conflicts <= '0;
                channel_conflicts <= '0;
                peak_bandwidth <= '0;
                min_latency <= 16'hFFFF;
                max_latency <= '0;
                
                for (int i = 0; i < 16; i++) begin
                    latency_histogram[i] <= '0;
                end
                
                for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
                    channel_active_cycles[ch] <= '0;
                    channel_requests[ch] <= '0;
                    channel_responses[ch] <= '0;
                end
            end
        end
    end
    
    // Calculate derived metrics
    always_comb begin
        // Outstanding requests
        outstanding_requests = total_requests - total_responses;
        
        // Average latency
        avg_latency = (total_responses > 0) ? latency_sum[31:0] / total_responses : 16'h0;
        
        // Hit rate percentage
        logic [31:0] total_accesses;
        total_accesses = row_buffer_hits + row_buffer_misses;
        hit_rate_percent = (total_accesses > 0) ? 
                          (row_buffer_hits * 100) / total_accesses : 8'h0;
        
        // Channel utilization
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            channel_utilization[ch] = (cycle_counter > 0) ? 
                                    (channel_active_cycles[ch] * 100) / cycle_counter : 8'h0;
        end
        
        // Average bandwidth over all samples
        logic [47:0] total_bw;
        total_bw = 0;
        for (int i = 0; i < 256; i++) begin
            total_bw = total_bw + bandwidth_samples[i];
        end
        average_bandwidth = total_bw[31:0] / 256;
    end
    
    // Latency sum calculation (for average)
    logic [47:0] latency_sum;
    always_comb begin
        latency_sum = 0;
        for (int i = 0; i < 256; i++) begin
            latency_sum = latency_sum + latency_samples[i];
        end
    end

endmodule