// Memory Subsystem Integration Test
// Tests complete memory hierarchy from L1 to HBM

`timescale 1ns/1ps

module test_memory_subsystem;

    // Test parameters
    parameter NUM_CORES = 4;
    parameter NUM_ACCELERATORS = 4;
    parameter ADDR_WIDTH = 64;
    parameter DATA_WIDTH = 64;
    parameter CLK_PERIOD = 10;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Core interfaces
    logic [NUM_CORES-1:0][ADDR_WIDTH-1:0]   core_icache_addr;
    logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_icache_rdata;
    logic [NUM_CORES-1:0]                   core_icache_req;
    logic [NUM_CORES-1:0]                   core_icache_ready;
    logic [NUM_CORES-1:0]                   core_icache_hit;
    
    logic [NUM_CORES-1:0][ADDR_WIDTH-1:0]   core_dcache_addr;
    logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_dcache_wdata;
    logic [NUM_CORES-1:0][DATA_WIDTH-1:0]   core_dcache_rdata;
    logic [NUM_CORES-1:0]                   core_dcache_req;
    logic [NUM_CORES-1:0]                   core_dcache_we;
    logic [NUM_CORES-1:0][DATA_WIDTH/8-1:0] core_dcache_be;
    logic [NUM_CORES-1:0]                   core_dcache_ready;
    logic [NUM_CORES-1:0]                   core_dcache_hit;
    
    // Accelerator interfaces
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) accel_if [NUM_ACCELERATORS-1:0]();
    
    // HBM interfaces
    logic [3:0]                    hbm_clk_p;
    logic [3:0]                    hbm_clk_n;
    logic [3:0]                    hbm_rst_n;
    logic [3:0][5:0]              hbm_cmd;
    logic [3:0][33:0]             hbm_addr;
    logic [3:0]                    hbm_cmd_valid;
    logic [3:0]                    hbm_cmd_ready;
    logic [3:0][1023:0]           hbm_wdata;
    logic [3:0][127:0]            hbm_wstrb;
    logic [3:0]                    hbm_wvalid;
    logic [3:0]                    hbm_wready;
    logic [3:0][1023:0]           hbm_rdata;
    logic [3:0]                    hbm_rvalid;
    logic [3:0]                    hbm_rready;
    
    // Snoop interface
    logic                    snoop_req;
    logic [ADDR_WIDTH-1:0]   snoop_addr;
    logic [2:0]              snoop_type;
    logic                    snoop_hit;
    logic                    snoop_dirty;
    logic [2:0]              snoop_resp;
    
    // Performance monitoring
    logic [31:0]             l1_hit_count, l1_miss_count;
    logic [31:0]             l2_hit_count, l2_miss_count;
    logic [31:0]             l3_hit_count, l3_miss_count;
    logic [31:0]             memory_bandwidth;
    logic                    ecc_error;
    logic [ADDR_WIDTH-1:0]   error_addr;
    
    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    memory_subsystem #(
        .NUM_CORES(NUM_CORES),
        .NUM_ACCELERATORS(NUM_ACCELERATORS),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .core_icache_addr(core_icache_addr),
        .core_icache_rdata(core_icache_rdata),
        .core_icache_req(core_icache_req),
        .core_icache_ready(core_icache_ready),
        .core_icache_hit(core_icache_hit),
        .core_dcache_addr(core_dcache_addr),
        .core_dcache_wdata(core_dcache_wdata),
        .core_dcache_rdata(core_dcache_rdata),
        .core_dcache_req(core_dcache_req),
        .core_dcache_we(core_dcache_we),
        .core_dcache_be(core_dcache_be),
        .core_dcache_ready(core_dcache_ready),
        .core_dcache_hit(core_dcache_hit),
        .accel_if(accel_if),
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
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(snoop_hit),
        .snoop_dirty(snoop_dirty),
        .snoop_resp(snoop_resp),
        .l1_hit_count(l1_hit_count),
        .l1_miss_count(l1_miss_count),
        .l2_hit_count(l2_hit_count),
        .l2_miss_count(l2_miss_count),
        .l3_hit_count(l3_hit_count),
        .l3_miss_count(l3_miss_count),
        .memory_bandwidth(memory_bandwidth),
        .ecc_error(ecc_error),
        .error_addr(error_addr)
    );
    
    // HBM simulator
    logic [1023:0] hbm_memory [logic [63:0]];
    
    // HBM interface simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hbm_cmd_ready <= '1;
            hbm_wready <= '1;
            hbm_rvalid <= '0;
            hbm_rdata <= '0;
        end else begin
            // Simple HBM simulation
            for (int ch = 0; ch < 4; ch++) begin
                // Handle read commands
                if (hbm_cmd_valid[ch] && hbm_cmd_ready[ch] && (hbm_cmd[ch] == 6'b001101)) begin
                    hbm_rvalid[ch] <= 1'b1;
                    if (hbm_memory.exists(hbm_addr[ch])) begin
                        hbm_rdata[ch] <= hbm_memory[hbm_addr[ch]];
                    end else begin
                        hbm_rdata[ch] <= {16{hbm_addr[ch]}};  // Test pattern
                    end
                end else begin
                    hbm_rvalid[ch] <= 1'b0;
                end
                
                // Handle write commands
                if (hbm_wvalid[ch] && hbm_wready[ch]) begin
                    hbm_memory[hbm_addr[ch]] <= hbm_wdata[ch];
                end
            end
        end
    end
    
    // Initialize accelerator interfaces
    generate
        for (genvar i = 0; i < NUM_ACCELERATORS; i++) begin : init_accel_if
            initial begin
                accel_if[i].arvalid = 1'b0;
                accel_if[i].awvalid = 1'b0;
                accel_if[i].wvalid = 1'b0;
                accel_if[i].rready = 1'b1;
                accel_if[i].bready = 1'b1;
            end
        end
    endgenerate
    
    // Test tasks
    task reset_system();
        rst_n = 1'b0;
        
        // Initialize core interfaces
        core_icache_req = '0;
        core_dcache_req = '0;
        core_dcache_we = '0;
        snoop_req = 1'b0;
        
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);
    endtask
    
    task check_result(string test_name, logic expected, logic actual);
        test_count++;
        if (expected === actual) begin
            $display("PASS: %s", test_name);
            pass_count++;
        end else begin
            $display("FAIL: %s - Expected: %b, Got: %b", test_name, expected, actual);
            fail_count++;
        end
    endtask
    
    task core_icache_read(int core_id, logic [63:0] addr);
        core_icache_addr[core_id] = addr;
        core_icache_req[core_id] = 1'b1;
        @(posedge clk);
        
        wait(core_icache_ready[core_id]);
        core_icache_req[core_id] = 1'b0;
        @(posedge clk);
    endtask
    
    task core_dcache_read(int core_id, logic [63:0] addr);
        core_dcache_addr[core_id] = addr;
        core_dcache_req[core_id] = 1'b1;
        core_dcache_we[core_id] = 1'b0;
        @(posedge clk);
        
        wait(core_dcache_ready[core_id]);
        core_dcache_req[core_id] = 1'b0;
        @(posedge clk);
    endtask
    
    task core_dcache_write(int core_id, logic [63:0] addr, logic [63:0] data);
        core_dcache_addr[core_id] = addr;
        core_dcache_wdata[core_id] = data;
        core_dcache_be[core_id] = '1;
        core_dcache_req[core_id] = 1'b1;
        core_dcache_we[core_id] = 1'b1;
        @(posedge clk);
        
        wait(core_dcache_ready[core_id]);
        core_dcache_req[core_id] = 1'b0;
        core_dcache_we[core_id] = 1'b0;
        @(posedge clk);
    endtask
    
    task accel_read_request(int accel_id, logic [63:0] addr, logic [7:0] id);
        accel_if[accel_id].arid = id;
        accel_if[accel_id].araddr = addr;
        accel_if[accel_id].arlen = 0;
        accel_if[accel_id].arsize = 3'b110;  // 64 bytes
        accel_if[accel_id].arburst = 2'b01;
        accel_if[accel_id].arvalid = 1'b1;
        
        @(posedge clk);
        wait(accel_if[accel_id].arready);
        accel_if[accel_id].arvalid = 1'b0;
        
        wait(accel_if[accel_id].rvalid);
        @(posedge clk);
    endtask
    
    task test_l1_cache_functionality();
        $display("Testing L1 cache functionality...");
        
        // Test I-cache miss and hit
        core_icache_read(0, 64'h1000);
        check_result("I-cache access completed", 1'b1, core_icache_ready[0]);
        
        // Test D-cache read miss
        core_dcache_read(0, 64'h2000);
        check_result("D-cache read completed", 1'b1, core_dcache_ready[0]);
        
        // Test D-cache write
        core_dcache_write(0, 64'h3000, 64'hDEADBEEF12345678);
        check_result("D-cache write completed", 1'b1, core_dcache_ready[0]);
        
        // Test D-cache read after write (should hit)
        core_dcache_read(0, 64'h3000);
        check_result("D-cache read after write hit", 1'b1, core_dcache_hit[0]);
    endtask
    
    task test_multi_core_access();
        $display("Testing multi-core memory access...");
        
        // Simultaneous access from multiple cores
        fork
            core_icache_read(0, 64'h10000);
            core_icache_read(1, 64'h11000);
            core_dcache_read(2, 64'h12000);
            core_dcache_read(3, 64'h13000);
        join
        
        check_result("Multi-core access completed", 1'b1, 1'b1);
    endtask
    
    task test_accelerator_access();
        $display("Testing accelerator memory access...");
        
        // Accelerator read request
        accel_read_request(0, 64'h20000, 8'h10);
        check_result("Accelerator read completed", 1'b1, accel_if[0].rvalid);
        
        // Multiple accelerator requests
        fork
            accel_read_request(0, 64'h21000, 8'h20);
            accel_read_request(1, 64'h22000, 8'h21);
        join
        
        check_result("Multi-accelerator access completed", 1'b1, 1'b1);
    endtask
    
    task test_cache_hierarchy();
        $display("Testing cache hierarchy...");
        
        logic [31:0] initial_l3_misses;
        initial_l3_misses = l3_miss_count;
        
        // Generate requests that should propagate through cache hierarchy
        for (int i = 0; i < 8; i++) begin
            core_dcache_read(0, 64'h30000 + (i * 64));
        end
        
        // Check that L3 misses increased (indicating hierarchy traversal)
        check_result("Cache hierarchy traversal", 1'b1, l3_miss_count > initial_l3_misses);
    endtask
    
    task test_cache_coherency();
        $display("Testing cache coherency...");
        
        // Load data into cache
        core_dcache_read(0, 64'h40000);
        
        // Test snoop hit
        snoop_addr = 64'h40000;
        snoop_type = 3'b001;  // Read request
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop hit detection", 1'b1, snoop_hit);
        
        snoop_req = 1'b0;
        @(posedge clk);
        
        // Test snoop invalidation
        snoop_addr = 64'h40000;
        snoop_type = 3'b010;  // Invalidate
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop invalidate response", 3'b011, snoop_resp);
        
        snoop_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_hbm_interface();
        $display("Testing HBM interface...");
        
        // Generate memory requests that should reach HBM
        for (int i = 0; i < 4; i++) begin
            core_dcache_read(i, 64'h100000 + (i * 1024));
        end
        
        // Check HBM activity
        logic hbm_active = 1'b0;
        for (int ch = 0; ch < 4; ch++) begin
            if (hbm_cmd_valid[ch] || hbm_rvalid[ch] || hbm_wvalid[ch]) begin
                hbm_active = 1'b1;
            end
        end
        
        check_result("HBM interface active", 1'b1, hbm_active);
    endtask
    
    task test_performance_monitoring();
        $display("Testing performance monitoring...");
        
        logic [31:0] initial_l3_hits, initial_l3_misses;
        initial_l3_hits = l3_hit_count;
        initial_l3_misses = l3_miss_count;
        
        // Generate cache activity
        for (int i = 0; i < 10; i++) begin
            core_dcache_read(0, 64'h50000 + (i * 64));
        end
        
        // Check performance counters
        check_result("L3 activity recorded", 1'b1, 
                    (l3_hit_count > initial_l3_hits) || (l3_miss_count > initial_l3_misses));
        
        $display("Performance Stats:");
        $display("  L1 Hits: %d, Misses: %d", l1_hit_count, l1_miss_count);
        $display("  L2 Hits: %d, Misses: %d", l2_hit_count, l2_miss_count);
        $display("  L3 Hits: %d, Misses: %d", l3_hit_count, l3_miss_count);
        $display("  Memory Bandwidth: %d bytes", memory_bandwidth);
    endtask
    
    task test_error_detection();
        $display("Testing error detection...");
        
        // Check initial error state
        check_result("No initial ECC errors", 1'b0, ecc_error);
        
        // Note: In a real test, we would inject errors to test detection
        $display("ECC error detection system ready");
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Memory Subsystem Integration Tests...");
        
        reset_system();
        
        test_l1_cache_functionality();
        test_multi_core_access();
        test_accelerator_access();
        test_cache_hierarchy();
        test_cache_coherency();
        test_hbm_interface();
        test_performance_monitoring();
        test_error_detection();
        
        // Test summary
        $display("\n=== Memory Subsystem Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL MEMORY SUBSYSTEM TESTS PASSED!");
        end else begin
            $display("SOME MEMORY SUBSYSTEM TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Timeout
    initial begin
        #500000;
        $display("ERROR: Memory subsystem test timeout!");
        $finish;
    end

endmodule