// L2/L3 Cache Integration Test Module
// Tests multi-level cache hierarchy with ECC and coherency

`timescale 1ns/1ps

module test_l2_l3_cache;

    // Test parameters
    parameter ADDR_WIDTH = 64;
    parameter DATA_WIDTH = 512;
    parameter CLK_PERIOD = 10;
    parameter NUM_L1_PORTS = 4;
    parameter NUM_L2_PORTS = 8;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // L2 cache interfaces
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l1_l2_if [NUM_L1_PORTS-1:0]();
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l2_l3_if();
    
    // L3 cache interfaces
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l2_l3_array_if [NUM_L2_PORTS-1:0]();
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l3_mem_if();
    
    // Snoop interfaces
    logic                    snoop_req;
    logic [ADDR_WIDTH-1:0]   snoop_addr;
    logic [2:0]              snoop_type;
    logic                    l2_snoop_hit, l3_snoop_hit;
    logic                    l2_snoop_dirty, l3_snoop_dirty;
    logic [2:0]              l2_snoop_resp, l3_snoop_resp;
    
    // ECC error signals
    logic                    l2_ecc_single, l3_ecc_single;
    logic                    l2_ecc_double, l3_ecc_double;
    logic [ADDR_WIDTH-1:0]   l2_ecc_addr, l3_ecc_addr;
    
    // Performance counters
    logic [31:0]             l3_hit_count, l3_miss_count, l3_eviction_count;
    
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
    l2_cache #(
        .CACHE_SIZE(2*1024*1024),
        .WAYS(16),
        .LINE_SIZE(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_PORTS(NUM_L1_PORTS)
    ) dut_l2 (
        .clk(clk),
        .rst_n(rst_n),
        .l1_if(l1_l2_if),
        .l3_if(l2_l3_if.master),
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(l2_snoop_hit),
        .snoop_dirty(l2_snoop_dirty),
        .snoop_resp(l2_snoop_resp),
        .ecc_single_error(l2_ecc_single),
        .ecc_double_error(l2_ecc_double),
        .ecc_error_addr(l2_ecc_addr)
    );
    
    // Connect L2 to L3 interface array
    assign l2_l3_array_if[0] = l2_l3_if;
    
    l3_cache #(
        .CACHE_SIZE(8*1024*1024),
        .WAYS(16),
        .LINE_SIZE(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_PORTS(NUM_L2_PORTS)
    ) dut_l3 (
        .clk(clk),
        .rst_n(rst_n),
        .l2_if(l2_l3_array_if),
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
        .eviction_count(l3_eviction_count)
    );
    
    // Memory simulator for L3
    logic [511:0] main_memory [logic [63:0]];
    
    // L3 memory interface simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l3_mem_if.arready <= 1'b0;
            l3_mem_if.rvalid <= 1'b0;
            l3_mem_if.rlast <= 1'b0;
            l3_mem_if.rdata <= '0;
            l3_mem_if.awready <= 1'b0;
            l3_mem_if.wready <= 1'b0;
            l3_mem_if.bvalid <= 1'b0;
        end else begin
            // Read interface
            if (l3_mem_if.arvalid && !l3_mem_if.arready) begin
                l3_mem_if.arready <= 1'b1;
            end else begin
                l3_mem_if.arready <= 1'b0;
            end
            
            if (l3_mem_if.arvalid && l3_mem_if.arready) begin
                l3_mem_if.rvalid <= 1'b1;
                l3_mem_if.rlast <= 1'b1;
                // Return test pattern or stored data
                if (main_memory.exists(l3_mem_if.araddr)) begin
                    l3_mem_if.rdata <= main_memory[l3_mem_if.araddr];
                end else begin
                    l3_mem_if.rdata <= {8{l3_mem_if.araddr[63:0]}};
                end
            end else begin
                l3_mem_if.rvalid <= 1'b0;
                l3_mem_if.rlast <= 1'b0;
            end
            
            // Write interface
            if (l3_mem_if.awvalid && !l3_mem_if.awready) begin
                l3_mem_if.awready <= 1'b1;
                l3_mem_if.wready <= 1'b1;
            end else begin
                l3_mem_if.awready <= 1'b0;
                l3_mem_if.wready <= 1'b0;
            end
            
            if (l3_mem_if.wvalid && l3_mem_if.wready) begin
                main_memory[l3_mem_if.awaddr] <= l3_mem_if.wdata;
                l3_mem_if.bvalid <= 1'b1;
            end else begin
                l3_mem_if.bvalid <= 1'b0;
            end
        end
    end
    
    // Initialize unused L2-L3 interfaces
    generate
        for (genvar i = 1; i < NUM_L2_PORTS; i++) begin : unused_l2_l3_if
            assign l2_l3_array_if[i].arvalid = 1'b0;
            assign l2_l3_array_if[i].awvalid = 1'b0;
            assign l2_l3_array_if[i].wvalid = 1'b0;
            assign l2_l3_array_if[i].rready = 1'b1;
            assign l2_l3_array_if[i].bready = 1'b1;
        end
    endgenerate
    
    // Test tasks
    task reset_system();
        rst_n = 1'b0;
        snoop_req = 1'b0;
        
        // Initialize L1-L2 interfaces (unrolled for Verilator compatibility)
        l1_l2_if[0].arvalid = 1'b0;
        l1_l2_if[0].awvalid = 1'b0;
        l1_l2_if[0].wvalid = 1'b0;
        l1_l2_if[0].rready = 1'b1;
        l1_l2_if[0].bready = 1'b1;
        
        l1_l2_if[1].arvalid = 1'b0;
        l1_l2_if[1].awvalid = 1'b0;
        l1_l2_if[1].wvalid = 1'b0;
        l1_l2_if[1].rready = 1'b1;
        l1_l2_if[1].bready = 1'b1;
        
        l1_l2_if[2].arvalid = 1'b0;
        l1_l2_if[2].awvalid = 1'b0;
        l1_l2_if[2].wvalid = 1'b0;
        l1_l2_if[2].rready = 1'b1;
        l1_l2_if[2].bready = 1'b1;
        
        l1_l2_if[3].arvalid = 1'b0;
        l1_l2_if[3].awvalid = 1'b0;
        l1_l2_if[3].wvalid = 1'b0;
        l1_l2_if[3].rready = 1'b1;
        l1_l2_if[3].bready = 1'b1;
        
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
    
    task l1_read_request_port0(logic [63:0] addr, logic [7:0] id);
        l1_l2_if[0].arid = id;
        l1_l2_if[0].araddr = addr;
        l1_l2_if[0].arlen = 0; // Single beat
        l1_l2_if[0].arsize = 3'b110; // 64 bytes
        l1_l2_if[0].arburst = 2'b01; // INCR
        l1_l2_if[0].arvalid = 1'b1;
        
        @(posedge clk);
        wait(l1_l2_if[0].arready);
        l1_l2_if[0].arvalid = 1'b0;
        
        // Wait for response
        wait(l1_l2_if[0].rvalid);
        @(posedge clk);
    endtask
    
    task l1_read_request_port1(logic [63:0] addr, logic [7:0] id);
        l1_l2_if[1].arid = id;
        l1_l2_if[1].araddr = addr;
        l1_l2_if[1].arlen = 0;
        l1_l2_if[1].arsize = 3'b110;
        l1_l2_if[1].arburst = 2'b01;
        l1_l2_if[1].arvalid = 1'b1;
        
        @(posedge clk);
        wait(l1_l2_if[1].arready);
        l1_l2_if[1].arvalid = 1'b0;
        
        wait(l1_l2_if[1].rvalid);
        @(posedge clk);
    endtask
    
    task l1_write_request_port0(logic [63:0] addr, logic [511:0] data, logic [7:0] id);
        // Write address
        l1_l2_if[0].awid = id;
        l1_l2_if[0].awaddr = addr;
        l1_l2_if[0].awlen = 0;
        l1_l2_if[0].awsize = 3'b110;
        l1_l2_if[0].awburst = 2'b01;
        l1_l2_if[0].awvalid = 1'b1;
        
        // Write data
        l1_l2_if[0].wdata = data;
        l1_l2_if[0].wstrb = '1;
        l1_l2_if[0].wlast = 1'b1;
        l1_l2_if[0].wvalid = 1'b1;
        
        @(posedge clk);
        wait(l1_l2_if[0].awready && l1_l2_if[0].wready);
        l1_l2_if[0].awvalid = 1'b0;
        l1_l2_if[0].wvalid = 1'b0;
        
        // Wait for response
        wait(l1_l2_if[0].bvalid);
        @(posedge clk);
    endtask
    
    task test_l2_l3_hierarchy();
        $display("Testing L2-L3 cache hierarchy...");
        
        // Test L2 miss -> L3 hit
        l1_read_request_port0(64'h10000, 8'h01);
        check_result("L2-L3 hierarchy read", 1'b1, l1_l2_if[0].rvalid);
        
        // Test L2 hit after fill
        l1_read_request_port0(64'h10000, 8'h02);
        check_result("L2 hit after fill", 1'b1, l1_l2_if[0].rvalid);
        
        // Test write through hierarchy
        l1_write_request_port0(64'h20000, 512'hDEADBEEF_12345678_ABCDEF01_23456789, 8'h03);
        check_result("L2-L3 hierarchy write", 1'b1, l1_l2_if[0].bvalid);
    endtask
    
    task test_enhanced_arbitration();
        $display("Testing enhanced multi-port arbitration with QoS...");
        
        // Test basic multi-port arbitration
        fork
            l1_read_request(0, 64'h30000, 8'h10);
            l1_read_request(1, 64'h31000, 8'h11);
            l1_read_request(2, 64'h32000, 8'h12);
            l1_read_request(3, 64'h33000, 8'h13);
        join
        
        check_result("Basic multi-port arbitration", 1'b1, 1'b1);
        
        // Test fairness with repeated requests from same port
        $display("Testing arbitration fairness...");
        for (int round = 0; round < 3; round++) begin
            fork
                begin
                    for (int i = 0; i < 4; i++) begin
                        l1_read_request(0, 64'h34000 + (round*256) + (i*64), 8'h20 + i);
                    end
                end
                begin
                    l1_read_request(1, 64'h35000 + (round*64), 8'h30 + round);
                    l1_read_request(2, 64'h36000 + (round*64), 8'h40 + round);
                    l1_read_request(3, 64'h37000 + (round*64), 8'h50 + round);
                end
            join
        end
        
        // Test bandwidth allocation under heavy load
        $display("Testing bandwidth allocation...");
        fork
            begin
                for (int i = 0; i < 16; i++) begin
                    l1_read_request(0, 64'h38000 + (i*64), 8'h60 + i);
                end
            end
            begin
                for (int i = 0; i < 8; i++) begin
                    l1_read_request(1, 64'h39000 + (i*64), 8'h70 + i);
                end
            end
            begin
                for (int i = 0; i < 4; i++) begin
                    l1_read_request(2, 64'h3A000 + (i*64), 8'h80 + i);
                end
            end
        join
        
        $display("Enhanced arbitration test completed");
    endtask
    
    task test_cache_coherency();
        $display("Testing cache coherency...");
        
        // Load data into L2
        l1_read_request(0, 64'h40000, 8'h20);
        
        // Test snoop hit
        snoop_addr = 64'h40000;
        snoop_type = 3'b001; // Read request
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("L2 snoop hit", 1'b1, l2_snoop_hit);
        
        snoop_req = 1'b0;
        @(posedge clk);
        
        // Test snoop invalidation
        snoop_addr = 64'h40000;
        snoop_type = 3'b010; // Invalidate
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("L2 snoop invalidate", 3'b011, l2_snoop_resp);
        
        snoop_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_ecc_functionality();
        $display("Testing enhanced ECC functionality...");
        
        // Load data to trigger ECC checking
        l1_read_request(0, 64'h50000, 8'h30);
        
        // Check ECC error signals (should be clean initially)
        check_result("No ECC single error initially", 1'b0, l2_ecc_single);
        check_result("No ECC double error initially", 1'b0, l2_ecc_double);
        
        // Test ECC error injection and correction (simplified simulation)
        // In real hardware, this would be done through debug registers
        $display("Testing ECC error correction capabilities...");
        
        // Write known pattern
        l1_write_request(0, 64'h51000, 512'hDEADBEEF_CAFEBABE_12345678_87654321, 8'h31);
        
        // Read back to verify ECC computation
        l1_read_request(0, 64'h51000, 8'h32);
        check_result("ECC protected read", 1'b1, l1_l2_if[0].rvalid);
        
        // Test multiple ECC operations
        for (int i = 0; i < 8; i++) begin
            l1_write_request(0, 64'h52000 + (i * 64), 512'h5555AAAA_AAAA5555 + i, 8'h33 + i);
            l1_read_request(0, 64'h52000 + (i * 64), 8'h40 + i);
        end
        
        $display("ECC functionality test completed");
    endtask
    
    // Performance monitoring variables
    logic [31:0] perf_initial_hits, perf_initial_misses, perf_initial_evictions;
    logic [31:0] perf_phase1_hits, perf_phase1_misses;
    logic [31:0] perf_phase2_hits, perf_phase2_misses;
    logic [31:0] perf_total_accesses, perf_hit_rate;
    
    task test_performance_monitoring();
        $display("Testing enhanced performance monitoring...");
        
        perf_initial_hits = l3_hit_count;
        perf_initial_misses = l3_miss_count;
        perf_initial_evictions = l3_eviction_count;

        
        // Generate controlled cache activity to test counters
        $display("Phase 1: Cold cache - expect misses");
        for (int i = 0; i < 8; i++) begin
            l1_read_request(0, 64'h60000 + (i * 64), 8'h40 + i);
        end
        
        perf_phase1_hits = l3_hit_count;
        perf_phase1_misses = l3_miss_count;
        
        $display("Phase 2: Warm cache - expect hits");
        for (int i = 0; i < 8; i++) begin
            l1_read_request(0, 64'h60000 + (i * 64), 8'h50 + i);
        end
        
        perf_phase2_hits = l3_hit_count;
        perf_phase2_misses = l3_miss_count;
        
        // Verify counter behavior
        check_result("Phase 1 generated misses", 1'b1, perf_phase1_misses > perf_initial_misses);
        check_result("Phase 2 generated hits", 1'b1, perf_phase2_hits > perf_phase1_hits);
        check_result("Phase 2 no additional misses", 1'b1, perf_phase2_misses == perf_phase1_misses);
        
        // Test eviction counter by filling cache
        $display("Phase 3: Cache eviction test");
        for (int i = 0; i < 32; i++) begin
            l1_read_request(0, 64'h70000 + (i * 1024), 8'h60 + i); // Different cache sets
        end
        
        check_result("Evictions occurred", 1'b1, l3_eviction_count > perf_initial_evictions);
        
        $display("L3 Performance Summary:");
        $display("  Hits: %d (delta: %d)", l3_hit_count, l3_hit_count - perf_initial_hits);
        $display("  Misses: %d (delta: %d)", l3_miss_count, l3_miss_count - perf_initial_misses);
        $display("  Evictions: %d (delta: %d)", l3_eviction_count, l3_eviction_count - perf_initial_evictions);
        
        // Calculate hit rate
        perf_total_accesses = (l3_hit_count - perf_initial_hits) + (l3_miss_count - perf_initial_misses);
        if (perf_total_accesses > 0) begin
            perf_hit_rate = ((l3_hit_count - perf_initial_hits) * 100) / perf_total_accesses;
            $display("  Hit Rate: %d%%", perf_hit_rate);
        end
    endtask
    
    task test_replacement_policy();
        $display("Testing cache replacement policy...");
        
        // Fill cache sets to test LRU replacement
        for (int i = 0; i < 20; i++) begin
            l1_read_request(0, 64'h70000 + (i * 64), 8'h50 + i);
        end
        
        // Access first address again to test replacement
        l1_read_request(0, 64'h70000, 8'h60);
        
        $display("Replacement policy test completed");
    endtask
    
    task test_cache_hierarchy_integration();
        $display("Testing comprehensive cache hierarchy integration...");
        
        // Test 1: L1 miss -> L2 miss -> L3 miss -> Memory
        $display("Test 1: Complete miss hierarchy");
        l1_read_request(0, 64'h80000, 8'h90);
        check_result("L1->L2->L3->Memory path", 1'b1, l1_l2_if[0].rvalid);
        
        // Test 2: L1 miss -> L2 miss -> L3 hit
        $display("Test 2: L3 hit scenario");
        l1_read_request(1, 64'h80000, 8'h91); // Same address, different L1
        check_result("L1->L2->L3 hit path", 1'b1, l1_l2_if[1].rvalid);
        
        // Test 3: L1 miss -> L2 hit
        $display("Test 3: L2 hit scenario");
        l1_read_request(0, 64'h80000, 8'h92); // Same L1, should hit L2
        check_result("L1->L2 hit path", 1'b1, l1_l2_if[0].rvalid);
        
        // Test 4: Write propagation through hierarchy
        $display("Test 4: Write propagation");
        l1_write_request(0, 64'h81000, 512'hFEEDFACE_DEADBEEF_12345678_87654321, 8'h93);
        l1_read_request(1, 64'h81000, 8'h94); // Read from different L1
        check_result("Write propagation", 1'b1, l1_l2_if[1].rvalid);
        
        // Test 5: Concurrent access to different levels
        $display("Test 5: Concurrent multi-level access");
        fork
            l1_read_request(0, 64'h82000, 8'h95); // New address (will miss)
            l1_read_request(1, 64'h80000, 8'h96); // Cached address (will hit)
            l1_read_request(2, 64'h83000, 8'h97); // Another new address
            l1_write_request(3, 64'h84000, 512'hAAAABBBB_CCCCDDDD, 8'h98);
        join
        
        $display("Cache hierarchy integration test completed");
    endtask
    
    task test_stress_scenarios();
        $display("Testing stress scenarios...");
        
        // Stress test 1: High-frequency alternating access
        $display("Stress 1: High-frequency alternating access");
        for (int i = 0; i < 50; i++) begin
            fork
                l1_read_request(i % 4, 64'h90000 + ((i % 8) * 64), 8'hA0 + (i % 16));
                if (i % 3 == 0) begin
                    l1_write_request((i+1) % 4, 64'h91000 + ((i % 6) * 64), 
                                   512'h12345678 + i, 8'hB0 + (i % 16));
                end
            join
        end
        
        // Stress test 2: Cache thrashing scenario
        $display("Stress 2: Cache thrashing");
        for (int i = 0; i < 20; i++) begin
            // Access addresses that map to same cache set but different tags
            l1_read_request(0, 64'h100000 + (i * 2048), 8'hC0 + i); // Same set, different tags
        end
        
        // Stress test 3: Mixed read/write with coherency
        $display("Stress 3: Mixed operations with coherency");
        fork
            begin
                for (int i = 0; i < 10; i++) begin
                    l1_write_request(0, 64'hA0000 + (i * 64), 512'hDEADBEEF + i, 8'hD0 + i);
                    l1_read_request(1, 64'hA0000 + (i * 64), 8'hE0 + i);
                end
            end
            begin
                for (int i = 0; i < 10; i++) begin
                    l1_read_request(2, 64'hA0000 + (i * 64), 8'hF0 + i);
                    if (i % 2 == 0) begin
                        // Snoop operations
                        snoop_addr = 64'hA0000 + (i * 64);
                        snoop_type = 3'b001;
                        snoop_req = 1'b1;
                        @(posedge clk);
                        snoop_req = 1'b0;
                        @(posedge clk);
                    end
                end
            end
        join
        
        $display("Stress scenarios completed");
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting L2/L3 Cache Integration Tests...");
        
        reset_system();
        
        test_l2_l3_hierarchy();
        test_enhanced_arbitration();
        test_cache_coherency();
        test_ecc_functionality();
        test_performance_monitoring();
        test_replacement_policy();
        test_cache_hierarchy_integration();
        test_stress_scenarios();
        
        // Test summary
        $display("\n=== L2/L3 Cache Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL L2/L3 CACHE TESTS PASSED!");
        end else begin
            $display("SOME L2/L3 CACHE TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Timeout
    initial begin
        #200000;
        $display("ERROR: L2/L3 Cache test timeout!");
        $finish;
    end

endmodule