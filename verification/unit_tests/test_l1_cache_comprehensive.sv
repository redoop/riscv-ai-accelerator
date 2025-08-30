// Comprehensive L1 Cache Test Module
// Tests all aspects of L1 cache functionality including coherency, performance, and error handling

`timescale 1ns/1ps

module test_l1_cache_comprehensive;

    // Test parameters
    parameter ADDR_WIDTH = 64;
    parameter DATA_WIDTH = 64;
    parameter CLK_PERIOD = 10;
    parameter CACHE_SIZE = 32 * 1024;
    parameter WAYS = 8;
    parameter LINE_SIZE = 64;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // CPU interface signals
    logic [ADDR_WIDTH-1:0]   cpu_addr;
    logic [DATA_WIDTH-1:0]   cpu_wdata;
    logic [DATA_WIDTH-1:0]   cpu_rdata;
    logic                    cpu_req;
    logic                    cpu_we;
    logic [DATA_WIDTH/8-1:0] cpu_be;
    logic                    cpu_is_instr;
    logic                    cpu_ready;
    logic                    cpu_hit;
    
    // L2 interface
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l2_if();
    
    // Snoop interface
    logic                    snoop_req;
    logic [ADDR_WIDTH-1:0]   snoop_addr;
    logic [2:0]              snoop_type;
    logic                    snoop_hit;
    logic                    snoop_dirty;
    logic [2:0]              snoop_resp;
    
    // Test statistics
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Performance tracking
    int icache_accesses = 0;
    int dcache_accesses = 0;
    int cache_hits = 0;
    int cache_misses = 0;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    l1_cache_controller #(
        .CACHE_SIZE(CACHE_SIZE),
        .WAYS(WAYS),
        .LINE_SIZE(LINE_SIZE),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .cpu_addr(cpu_addr),
        .cpu_wdata(cpu_wdata),
        .cpu_rdata(cpu_rdata),
        .cpu_req(cpu_req),
        .cpu_we(cpu_we),
        .cpu_be(cpu_be),
        .cpu_is_instr(cpu_is_instr),
        .cpu_ready(cpu_ready),
        .cpu_hit(cpu_hit),
        .l2_if(l2_if.master),
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(snoop_hit),
        .snoop_dirty(snoop_dirty),
        .snoop_resp(snoop_resp)
    );
    
    // L2 cache simulator
    logic [511:0] l2_memory [logic [63:0]];
    logic l2_delay_counter;
    
    // L2 interface simulation with realistic delays
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l2_if.arready <= 1'b0;
            l2_if.rvalid <= 1'b0;
            l2_if.rlast <= 1'b0;
            l2_if.rdata <= '0;
            l2_if.rresp <= 2'b00;
            l2_if.awready <= 1'b0;
            l2_if.wready <= 1'b0;
            l2_if.bvalid <= 1'b0;
            l2_if.bresp <= 2'b00;
            l2_delay_counter <= 1'b0;
        end else begin
            // Read interface with delay
            if (l2_if.arvalid && !l2_if.arready && !l2_delay_counter) begin
                l2_delay_counter <= 1'b1;
            end else if (l2_delay_counter) begin
                l2_if.arready <= 1'b1;
                l2_delay_counter <= 1'b0;
            end else begin
                l2_if.arready <= 1'b0;
            end
            
            if (l2_if.arvalid && l2_if.arready) begin
                l2_if.rvalid <= 1'b1;
                l2_if.rlast <= 1'b1;
                l2_if.rresp <= 2'b00; // OKAY
                // Generate test pattern based on address
                l2_if.rdata <= {8{l2_if.araddr[63:0]}};
            end else begin
                l2_if.rvalid <= 1'b0;
                l2_if.rlast <= 1'b0;
            end
            
            // Write interface
            if (l2_if.awvalid && !l2_if.awready) begin
                l2_if.awready <= 1'b1;
                l2_if.wready <= 1'b1;
            end else begin
                l2_if.awready <= 1'b0;
                l2_if.wready <= 1'b0;
            end
            
            if (l2_if.wvalid && l2_if.wready) begin
                l2_if.bvalid <= 1'b1;
                l2_if.bresp <= 2'b00; // OKAY
            end else begin
                l2_if.bvalid <= 1'b0;
            end
        end
    end
    
    // Test utility tasks
    task reset_system();
        rst_n = 1'b0;
        cpu_req = 1'b0;
        cpu_we = 1'b0;
        cpu_is_instr = 1'b0;
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
    
    task cpu_access(input logic [63:0] addr, input logic is_instr, input logic write_en, 
                   input logic [63:0] wdata, input logic [7:0] be);
        cpu_addr = addr;
        cpu_is_instr = is_instr;
        cpu_we = write_en;
        cpu_wdata = wdata;
        cpu_be = be;
        cpu_req = 1'b1;
        
        @(posedge clk);
        wait(cpu_ready);
        
        // Track statistics
        if (is_instr) icache_accesses++;
        else dcache_accesses++;
        
        if (cpu_hit) cache_hits++;
        else cache_misses++;
        
        cpu_req = 1'b0;
        @(posedge clk);
    endtask
    
    // Comprehensive test cases
    task test_basic_functionality();
        $display("=== Testing Basic Cache Functionality ===");
        
        // Test I-cache miss and hit
        cpu_access(64'h1000, 1'b1, 1'b0, 64'h0, 8'h00);
        check_result("I-cache first access (miss expected)", 1'b0, cpu_hit);
        
        cpu_access(64'h1000, 1'b1, 1'b0, 64'h0, 8'h00);
        check_result("I-cache second access (hit expected)", 1'b1, cpu_hit);
        
        // Test D-cache read miss and hit
        cpu_access(64'h2000, 1'b0, 1'b0, 64'h0, 8'h00);
        check_result("D-cache read miss", 1'b0, cpu_hit);
        
        cpu_access(64'h2000, 1'b0, 1'b0, 64'h0, 8'h00);
        check_result("D-cache read hit", 1'b1, cpu_hit);
        
        // Test D-cache write hit
        cpu_access(64'h2000, 1'b0, 1'b1, 64'hDEADBEEF12345678, 8'hFF);
        check_result("D-cache write hit", 1'b1, cpu_hit);
    endtask
    
    task test_cache_coherency();
        $display("=== Testing Cache Coherency ===");
        
        // Load data into cache
        cpu_access(64'h3000, 1'b0, 1'b0, 64'h0, 8'h00);
        
        // Test snoop hit
        snoop_addr = 64'h3000;
        snoop_type = 3'b001; // Read request
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop hit detection", 1'b1, snoop_hit);
        
        snoop_req = 1'b0;
        @(posedge clk);
        
        // Test snoop invalidation
        snoop_addr = 64'h3000;
        snoop_type = 3'b010; // Invalidate
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop invalidate response", 3'b011, snoop_resp);
        
        snoop_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_replacement_policy();
        $display("=== Testing Replacement Policy ===");
        
        // Fill cache with more lines than ways to force replacement
        for (int i = 0; i < WAYS + 2; i++) begin
            cpu_access(64'h4000 + (i * LINE_SIZE), 1'b1, 1'b0, 64'h0, 8'h00);
        end
        
        // Access first line again - may or may not hit depending on replacement
        cpu_access(64'h4000, 1'b1, 1'b0, 64'h0, 8'h00);
        $display("Replacement test: First line hit = %b", cpu_hit);
    endtask
    
    task test_byte_enables();
        $display("=== Testing Byte Enable Functionality ===");
        
        // Load cache line
        cpu_access(64'h5000, 1'b0, 1'b0, 64'h0, 8'h00);
        
        // Test partial writes with different byte enables
        cpu_access(64'h5000, 1'b0, 1'b1, 64'hFFFFFFFFFFFFFFFF, 8'b00001111);
        check_result("Partial write (lower 4 bytes)", 1'b1, cpu_hit);
        
        cpu_access(64'h5000, 1'b0, 1'b1, 64'h0000000000000000, 8'b11110000);
        check_result("Partial write (upper 4 bytes)", 1'b1, cpu_hit);
        
        cpu_access(64'h5000, 1'b0, 1'b1, 64'hAAAAAAAAAAAAAAAA, 8'b10101010);
        check_result("Partial write (alternating bytes)", 1'b1, cpu_hit);
    endtask
    
    task test_address_aliasing();
        $display("=== Testing Address Aliasing ===");
        
        // Test different addresses that map to same cache set
        logic [63:0] base_addr = 64'h6000;
        logic [63:0] alias_addr = base_addr + (CACHE_SIZE / WAYS); // Same set, different tag
        
        cpu_access(base_addr, 1'b0, 1'b0, 64'h0, 8'h00);
        cpu_access(alias_addr, 1'b0, 1'b0, 64'h0, 8'h00);
        
        // Both should be in cache if there are enough ways
        cpu_access(base_addr, 1'b0, 1'b0, 64'h0, 8'h00);
        logic first_hit = cpu_hit;
        
        cpu_access(alias_addr, 1'b0, 1'b0, 64'h0, 8'h00);
        logic second_hit = cpu_hit;
        
        $display("Address aliasing test: base_hit=%b, alias_hit=%b", first_hit, second_hit);
    endtask
    
    task test_mixed_access_patterns();
        $display("=== Testing Mixed Access Patterns ===");
        
        // Interleave instruction and data accesses
        for (int i = 0; i < 8; i++) begin
            cpu_access(64'h7000 + (i * 64), 1'b1, 1'b0, 64'h0, 8'h00); // I-cache
            cpu_access(64'h8000 + (i * 64), 1'b0, 1'b0, 64'h0, 8'h00); // D-cache read
            cpu_access(64'h8000 + (i * 64), 1'b0, 1'b1, 64'h12345678 + i, 8'hFF); // D-cache write
        end
        
        $display("Mixed access pattern completed");
    endtask
    
    task test_performance_characteristics();
        $display("=== Testing Performance Characteristics ===");
        
        int start_time, end_time;
        int hit_latency, miss_latency;
        
        // Measure hit latency
        cpu_access(64'h9000, 1'b0, 1'b0, 64'h0, 8'h00); // Prime cache
        
        start_time = $time;
        cpu_access(64'h9000, 1'b0, 1'b0, 64'h0, 8'h00); // Hit
        end_time = $time;
        hit_latency = end_time - start_time;
        
        // Measure miss latency
        start_time = $time;
        cpu_access(64'hA000, 1'b0, 1'b0, 64'h0, 8'h00); // Miss
        end_time = $time;
        miss_latency = end_time - start_time;
        
        $display("Hit latency: %0d ns, Miss latency: %0d ns", hit_latency, miss_latency);
        check_result("Miss latency > Hit latency", 1'b1, miss_latency > hit_latency);
    endtask
    
    task test_stress_patterns();
        $display("=== Testing Stress Patterns ===");
        
        // Sequential access pattern
        for (int i = 0; i < 32; i++) begin
            cpu_access(64'hB000 + (i * 8), 1'b0, 1'b0, 64'h0, 8'h00);
        end
        
        // Random access pattern
        for (int i = 0; i < 16; i++) begin
            logic [63:0] random_addr = 64'hC000 + ($random % 1024) * 64;
            cpu_access(random_addr, 1'b0, 1'b0, 64'h0, 8'h00);
        end
        
        // Strided access pattern
        for (int i = 0; i < 16; i++) begin
            cpu_access(64'hD000 + (i * 256), 1'b0, 1'b0, 64'h0, 8'h00);
        end
        
        $display("Stress pattern testing completed");
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Comprehensive L1 Cache Tests...");
        $display("Cache Configuration: %0dKB, %0d-way, %0dB lines", 
                 CACHE_SIZE/1024, WAYS, LINE_SIZE);
        
        reset_system();
        
        test_basic_functionality();
        test_cache_coherency();
        test_replacement_policy();
        test_byte_enables();
        test_address_aliasing();
        test_mixed_access_patterns();
        test_performance_characteristics();
        test_stress_patterns();
        
        // Final statistics
        $display("\n=== Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        $display("Success rate: %0.1f%%", (pass_count * 100.0) / test_count);
        
        $display("\n=== Performance Statistics ===");
        $display("I-cache accesses: %d", icache_accesses);
        $display("D-cache accesses: %d", dcache_accesses);
        $display("Total cache hits: %d", cache_hits);
        $display("Total cache misses: %d", cache_misses);
        $display("Hit rate: %0.1f%%", (cache_hits * 100.0) / (cache_hits + cache_misses));
        
        if (fail_count == 0) begin
            $display("\n🎉 ALL TESTS PASSED! 🎉");
        end else begin
            $display("\n❌ SOME TESTS FAILED! ❌");
        end
        
        $finish;
    end
    
    // Timeout protection
    initial begin
        #500000; // 500us timeout
        $display("ERROR: Test timeout reached!");
        $finish;
    end
    
    // Monitor for debugging
    initial begin
        $monitor("Time: %0t, CPU: addr=%h req=%b we=%b hit=%b ready=%b", 
                 $time, cpu_addr, cpu_req, cpu_we, cpu_hit, cpu_ready);
    end

endmodule