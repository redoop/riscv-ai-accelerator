// Advanced HBM Controller Test
// Tests the enhanced HBM controller with advanced scheduling and monitoring

`timescale 1ns/1ps

module test_hbm_controller_advanced;

    // Test parameters
    parameter NUM_CHANNELS = 4;
    parameter ADDR_WIDTH = 64;
    parameter DATA_WIDTH = 512;
    parameter HBM_DATA_WIDTH = 1024;
    parameter CLK_PERIOD = 5.0;  // 200MHz
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // AXI interface
    axi4_if #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) axi_if();
    
    // HBM interface signals
    logic [NUM_CHANNELS-1:0]                    hbm_clk_p;
    logic [NUM_CHANNELS-1:0]                    hbm_clk_n;
    logic [NUM_CHANNELS-1:0]                    hbm_rst_n;
    logic [NUM_CHANNELS-1:0][5:0]              hbm_cmd;
    logic [NUM_CHANNELS-1:0][33:0]             hbm_addr;
    logic [NUM_CHANNELS-1:0]                    hbm_cmd_valid;
    logic [NUM_CHANNELS-1:0]                    hbm_cmd_ready;
    logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH-1:0] hbm_wdata;
    logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH/8-1:0] hbm_wstrb;
    logic [NUM_CHANNELS-1:0]                    hbm_wvalid;
    logic [NUM_CHANNELS-1:0]                    hbm_wready;
    logic [NUM_CHANNELS-1:0][HBM_DATA_WIDTH-1:0] hbm_rdata;
    logic [NUM_CHANNELS-1:0]                    hbm_rvalid;
    logic [NUM_CHANNELS-1:0]                    hbm_rready;
    
    // Performance monitoring signals
    logic [31:0]                    read_requests;
    logic [31:0]                    write_requests;
    logic [31:0]                    total_bandwidth;
    logic [15:0]                    avg_latency;
    logic [31:0]                    bank_conflict_count;
    logic [31:0]                    row_hit_rate;
    logic [15:0]                    instantaneous_bandwidth;
    logic [31:0]                    peak_bandwidth;
    logic [15:0]                    min_access_latency;
    logic [15:0]                    max_access_latency;
    logic [7:0]                     queue_utilization;
    logic [15:0]                    power_consumption;
    
    // Error signals
    logic                           ecc_error;
    logic [ADDR_WIDTH-1:0]          error_addr;
    logic [2:0]                     error_type;
    
    // Test control
    integer test_phase;
    integer requests_sent;
    integer responses_received;
    logic test_complete;
    
    // Performance tracking
    real start_time, end_time;
    real achieved_bw;
    integer max_latency_observed;
    integer min_latency_observed;
    
    // DUT instantiation
    hbm_controller #(
        .NUM_CHANNELS(NUM_CHANNELS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .HBM_DATA_WIDTH(HBM_DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .axi_if(axi_if.slave),
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
        .read_requests(read_requests),
        .write_requests(write_requests),
        .total_bandwidth(total_bandwidth),
        .avg_latency(avg_latency),
        .bank_conflict_count(bank_conflict_count),
        .row_hit_rate(row_hit_rate),
        .instantaneous_bandwidth(instantaneous_bandwidth),
        .peak_bandwidth(peak_bandwidth),
        .min_access_latency(min_access_latency),
        .max_access_latency(max_access_latency),
        .queue_utilization(queue_utilization),
        .power_consumption(power_consumption),
        .ecc_error(ecc_error),
        .error_addr(error_addr),
        .error_type(error_type)
    );
    
    // HBM Models
    generate
        for (genvar ch = 0; ch < NUM_CHANNELS; ch++) begin : hbm_models
            hbm_channel_model #(
                .DATA_WIDTH(HBM_DATA_WIDTH),
                .ADDR_WIDTH(34)
            ) hbm_model (
                .clk(clk),
                .rst_n(rst_n),
                .cmd(hbm_cmd[ch]),
                .addr(hbm_addr[ch]),
                .cmd_valid(hbm_cmd_valid[ch]),
                .cmd_ready(hbm_cmd_ready[ch]),
                .wdata(hbm_wdata[ch]),
                .wstrb(hbm_wstrb[ch]),
                .wvalid(hbm_wvalid[ch]),
                .wready(hbm_wready[ch]),
                .rdata(hbm_rdata[ch]),
                .rvalid(hbm_rvalid[ch]),
                .rready(hbm_rready[ch])
            );
        end
    endgenerate
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Reset generation
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
    end
    
    // Test sequence
    initial begin
        // Initialize
        test_phase = 0;
        requests_sent = 0;
        responses_received = 0;
        test_complete = 0;
        max_latency_observed = 0;
        min_latency_observed = 32'hFFFFFFFF;
        
        // Initialize AXI interface
        axi_if.arid = 0;
        axi_if.araddr = 0;
        axi_if.arlen = 0;
        axi_if.arsize = 3'b110;  // 64 bytes
        axi_if.arburst = 2'b01;  // INCR
        axi_if.arvalid = 0;
        axi_if.rready = 1;
        
        axi_if.awid = 0;
        axi_if.awaddr = 0;
        axi_if.awlen = 0;
        axi_if.awsize = 3'b110;
        axi_if.awburst = 2'b01;
        axi_if.awvalid = 0;
        axi_if.wdata = 0;
        axi_if.wstrb = '1;
        axi_if.wlast = 1;
        axi_if.wvalid = 0;
        axi_if.bready = 1;
        
        // Wait for reset
        wait(rst_n);
        #(CLK_PERIOD * 10);
        
        $display("=== Starting Advanced HBM Controller Test ===");
        
        // Test Phase 1: Basic functionality
        test_phase = 1;
        $display("Phase 1: Basic Read/Write Test");
        test_basic_functionality();
        
        // Test Phase 2: Bandwidth stress test
        test_phase = 2;
        $display("Phase 2: Bandwidth Stress Test");
        test_bandwidth_stress();
        
        // Test Phase 3: Latency optimization test
        test_phase = 3;
        $display("Phase 3: Latency Optimization Test");
        test_latency_optimization();
        
        // Test Phase 4: Bank conflict test
        test_phase = 4;
        $display("Phase 4: Bank Conflict Test");
        test_bank_conflicts();
        
        // Test Phase 5: Channel utilization test
        test_phase = 5;
        $display("Phase 5: Channel Utilization Test");
        test_channel_utilization();
        
        // Test Phase 6: Power and thermal test
        test_phase = 6;
        $display("Phase 6: Power and Thermal Test");
        test_power_thermal();
        
        // Final performance analysis
        analyze_final_performance();
        
        $display("=== Advanced HBM Controller Test Complete ===");
        test_complete = 1;
        #(CLK_PERIOD * 100);
        $finish;
    end
    
    // Test Phase 1: Basic functionality
    task test_basic_functionality();
        integer i;
        logic [ADDR_WIDTH-1:0] addr;
        
        $display("  Testing basic read operations...");
        
        // Send 100 read requests
        for (i = 0; i < 100; i++) begin
            addr = i * 64;  // Cache line aligned
            
            @(posedge clk);
            axi_if.arid <= i[7:0];
            axi_if.araddr <= addr;
            axi_if.arvalid <= 1;
            
            wait(axi_if.arready);
            @(posedge clk);
            axi_if.arvalid <= 0;
            requests_sent++;
        end
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        $display("  Basic test completed: %0d requests, %0d responses", 
                requests_sent, responses_received);
        
        // Check basic performance metrics
        if (avg_latency > 0) begin
            $display("  Average latency: %0d cycles", avg_latency);
        end
        
        if (total_bandwidth > 0) begin
            $display("  Total bandwidth: %0d MB/s", total_bandwidth / 1000000);
        end
    endtask
    
    // Test Phase 2: Bandwidth stress test
    task test_bandwidth_stress();
        integer i;
        logic [ADDR_WIDTH-1:0] addr;
        integer start_requests, start_responses;
        
        start_requests = requests_sent;
        start_responses = responses_received;
        start_time = $realtime;
        
        $display("  Starting bandwidth stress test...");
        
        // Generate high-rate sequential requests
        fork
            begin
                // Request generator
                for (i = 0; i < 1000; i++) begin
                    addr = (i * 64) % (1024 * 1024);  // 1MB address space
                    
                    @(posedge clk);
                    axi_if.arid <= i[7:0];
                    axi_if.araddr <= addr;
                    axi_if.arvalid <= 1;
                    
                    wait(axi_if.arready);
                    @(posedge clk);
                    axi_if.arvalid <= 0;
                    requests_sent++;
                    
                    // Small delay to prevent overwhelming
                    if (i % 10 == 0) begin
                        repeat(2) @(posedge clk);
                    end
                end
            end
            
            begin
                // Monitor bandwidth during test
                repeat(1000) begin
                    @(posedge clk);
                    if (instantaneous_bandwidth > 0) begin
                        $display("    Instantaneous BW: %0d MB/s, Queue: %0d%%", 
                                instantaneous_bandwidth, queue_utilization);
                    end
                    repeat(100) @(posedge clk);  // Sample every 100 cycles
                end
            end
        join
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        end_time = $realtime;
        achieved_bw = ((requests_sent - start_requests) * 64 * 1000000000.0) / 
                     (end_time - start_time);  // MB/s
        
        $display("  Bandwidth stress test completed");
        $display("    Requests: %0d, Time: %0.2f ns", 
                requests_sent - start_requests, end_time - start_time);
        $display("    Achieved bandwidth: %0.2f MB/s", achieved_bw);
        $display("    Peak bandwidth: %0d MB/s", peak_bandwidth / 1000000);
        $display("    Row buffer hit rate: %0d%%", row_hit_rate);
    endtask
    
    // Test Phase 3: Latency optimization test
    task test_latency_optimization();
        integer i;
        logic [ADDR_WIDTH-1:0] addr;
        integer start_requests;
        
        start_requests = requests_sent;
        
        $display("  Testing latency optimization...");
        
        // Generate requests with good locality
        for (i = 0; i < 200; i++) begin
            // Alternate between two cache lines in same row
            addr = (i % 2) ? 64'h1000 : 64'h1040;
            
            @(posedge clk);
            axi_if.arid <= i[7:0];
            axi_if.araddr <= addr;
            axi_if.arvalid <= 1;
            
            wait(axi_if.arready);
            @(posedge clk);
            axi_if.arvalid <= 0;
            requests_sent++;
            
            repeat(5) @(posedge clk);  // Allow processing time
        end
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        $display("  Latency optimization test completed");
        $display("    Min latency: %0d cycles", min_access_latency);
        $display("    Max latency: %0d cycles", max_access_latency);
        $display("    Average latency: %0d cycles", avg_latency);
        $display("    Row buffer hit rate: %0d%%", row_hit_rate);
        
        // Check if latency is within acceptable range
        if (avg_latency > 100) begin
            $warning("Average latency higher than expected: %0d cycles", avg_latency);
        end else begin
            $display("    PASS: Latency within acceptable range");
        end
    endtask
    
    // Test Phase 4: Bank conflict test
    task test_bank_conflicts();
        integer i;
        logic [ADDR_WIDTH-1:0] addr1, addr2;
        integer conflicts_before, conflicts_after;
        
        conflicts_before = bank_conflict_count;
        
        $display("  Testing bank conflict handling...");
        
        // Generate addresses that cause bank conflicts
        addr1 = 64'h0000_1000;  // Bank 0, Row 0
        addr2 = 64'h0100_1000;  // Bank 0, Row 1 (different row, same bank)
        
        for (i = 0; i < 100; i++) begin
            @(posedge clk);
            axi_if.arid <= i[7:0];
            axi_if.araddr <= (i % 2) ? addr1 : addr2;
            axi_if.arvalid <= 1;
            
            wait(axi_if.arready);
            @(posedge clk);
            axi_if.arvalid <= 0;
            requests_sent++;
            
            repeat(3) @(posedge clk);
        end
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        conflicts_after = bank_conflict_count;
        
        $display("  Bank conflict test completed");
        $display("    Bank conflicts generated: %0d", conflicts_after - conflicts_before);
        $display("    Total bank conflicts: %0d", bank_conflict_count);
        $display("    Row buffer hit rate: %0d%%", row_hit_rate);
        
        // Verify that conflicts were detected
        if ((conflicts_after - conflicts_before) > 0) begin
            $display("    PASS: Bank conflicts properly detected and handled");
        end else begin
            $warning("Expected bank conflicts not detected");
        end
    endtask
    
    // Test Phase 5: Channel utilization test
    task test_channel_utilization();
        integer i, ch;
        logic [ADDR_WIDTH-1:0] addr;
        
        $display("  Testing channel utilization...");
        
        // Generate requests targeting specific channels
        for (i = 0; i < 400; i++) begin
            ch = i % NUM_CHANNELS;
            // Encode channel in address bits [7:6]
            addr = {$random & 32'hFFFF_FF00, ch[1:0], 6'h00};
            
            @(posedge clk);
            axi_if.arid <= i[7:0];
            axi_if.araddr <= addr;
            axi_if.arvalid <= 1;
            
            wait(axi_if.arready);
            @(posedge clk);
            axi_if.arvalid <= 0;
            requests_sent++;
            
            repeat(2) @(posedge clk);
        end
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        $display("  Channel utilization test completed");
        $display("    Queue utilization: %0d%%", queue_utilization);
        
        // Check if channels are being utilized
        if (queue_utilization > 10) begin
            $display("    PASS: Good channel utilization achieved");
        end else begin
            $warning("Low channel utilization: %0d%%", queue_utilization);
        end
    endtask
    
    // Test Phase 6: Power and thermal test
    task test_power_thermal();
        integer i;
        logic [ADDR_WIDTH-1:0] addr;
        integer power_samples [0:9];
        integer avg_power;
        
        $display("  Testing power and thermal characteristics...");
        
        // Generate sustained load and monitor power
        for (i = 0; i < 500; i++) begin
            addr = {$random, $random} & ~64'h3F;
            
            @(posedge clk);
            axi_if.arid <= i[7:0];
            axi_if.araddr <= addr;
            axi_if.arvalid <= 1;
            
            wait(axi_if.arready);
            @(posedge clk);
            axi_if.arvalid <= 0;
            requests_sent++;
            
            // Sample power every 50 requests
            if (i % 50 == 0 && i < 500) begin
                power_samples[i/50] = power_consumption;
            end
            
            repeat(1) @(posedge clk);
        end
        
        // Wait for all responses
        while (responses_received < requests_sent) begin
            @(posedge clk);
        end
        
        // Calculate average power
        avg_power = 0;
        for (i = 0; i < 10; i++) begin
            avg_power += power_samples[i];
        end
        avg_power = avg_power / 10;
        
        $display("  Power and thermal test completed");
        $display("    Average power consumption: %0d mW", avg_power);
        $display("    Peak power consumption: %0d mW", power_consumption);
        
        // Check power consumption is reasonable
        if (avg_power < 2000) begin  // Less than 2W
            $display("    PASS: Power consumption within acceptable range");
        end else begin
            $warning("High power consumption: %0d mW", avg_power);
        end
    endtask
    
    // Final performance analysis
    task analyze_final_performance();
        real efficiency;
        
        $display("\n=== Final Performance Analysis ===");
        $display("Total requests sent: %0d", requests_sent);
        $display("Total responses received: %0d", responses_received);
        $display("Final bandwidth: %0d MB/s", total_bandwidth / 1000000);
        $display("Peak bandwidth achieved: %0d MB/s", peak_bandwidth / 1000000);
        $display("Average latency: %0d cycles", avg_latency);
        $display("Latency range: %0d - %0d cycles", min_access_latency, max_access_latency);
        $display("Row buffer hit rate: %0d%%", row_hit_rate);
        $display("Bank conflicts: %0d", bank_conflict_count);
        $display("Queue utilization: %0d%%", queue_utilization);
        $display("Power consumption: %0d mW", power_consumption);
        
        // Calculate efficiency metrics
        efficiency = (responses_received * 100.0) / requests_sent;
        $display("Response efficiency: %0.2f%%", efficiency);
        
        // Performance thresholds check
        $display("\n=== Performance Validation ===");
        
        if (total_bandwidth >= 32'd800_000_000) begin  // 800 MB/s
            $display("PASS: Bandwidth meets requirement (>= 800 MB/s)");
        end else begin
            $error("FAIL: Bandwidth below requirement: %0d MB/s", total_bandwidth / 1000000);
        end
        
        if (avg_latency <= 16'd80) begin  // 80 cycles max
            $display("PASS: Latency meets requirement (<= 80 cycles)");
        end else begin
            $error("FAIL: Latency above requirement: %0d cycles", avg_latency);
        end
        
        if (row_hit_rate >= 32'd60) begin  // 60% minimum
            $display("PASS: Row buffer hit rate acceptable (>= 60%%)");
        end else begin
            $warning("WARNING: Low row buffer hit rate: %0d%%", row_hit_rate);
        end
        
        if (efficiency >= 99.0) begin
            $display("PASS: High response efficiency");
        end else begin
            $warning("WARNING: Response efficiency below optimal: %0.2f%%", efficiency);
        end
        
        $display("=== Performance Validation Complete ===\n");
    endtask
    
    // Response tracking
    always @(posedge clk) begin
        if (axi_if.rvalid && axi_if.rready) begin
            responses_received <= responses_received + 1;
            
            // Track latency extremes
            if (avg_latency > max_latency_observed) begin
                max_latency_observed <= avg_latency;
            end
            if (avg_latency < min_latency_observed && avg_latency > 0) begin
                min_latency_observed <= avg_latency;
            end
        end
    end
    
    // Continuous monitoring
    always @(posedge clk) begin
        if (rst_n && !test_complete) begin
            // Monitor for anomalies
            if (ecc_error) begin
                $error("ECC error detected at address 0x%h, type: %0d", error_addr, error_type);
            end
            
            // Monitor queue utilization
            if (queue_utilization > 90) begin
                $display("WARNING: High queue utilization: %0d%%", queue_utilization);
            end
        end
    end

endmodule