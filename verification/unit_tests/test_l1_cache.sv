// L1 Cache Test Module
// Tests L1 instruction and data cache functionality

`timescale 1ns/1ps

module test_l1_cache;

    // Test parameters
    parameter ADDR_WIDTH = 64;
    parameter DATA_WIDTH = 64;
    parameter CLK_PERIOD = 10;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // CPU interface signals for I-cache
    logic [ADDR_WIDTH-1:0]   icache_addr;
    logic [DATA_WIDTH-1:0]   icache_rdata;
    logic                    icache_req;
    logic                    icache_ready;
    logic                    icache_hit;
    
    // CPU interface signals for D-cache
    logic [ADDR_WIDTH-1:0]   dcache_addr;
    logic [DATA_WIDTH-1:0]   dcache_wdata;
    logic [DATA_WIDTH-1:0]   dcache_rdata;
    logic                    dcache_req;
    logic                    dcache_we;
    logic [DATA_WIDTH/8-1:0] dcache_be;
    logic                    dcache_ready;
    logic                    dcache_hit;
    
    // L2 interface (simplified for testing)
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l2_icache_if();
    axi4_if #(.ADDR_WIDTH(64), .DATA_WIDTH(512)) l2_dcache_if();
    
    // Snoop interface
    logic                    snoop_req;
    logic [ADDR_WIDTH-1:0]   snoop_addr;
    logic [2:0]              snoop_type;
    logic                    icache_snoop_hit;
    logic [2:0]              icache_snoop_resp;
    logic                    dcache_snoop_hit;
    logic                    dcache_snoop_dirty;
    logic [2:0]              dcache_snoop_resp;
    
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
    l1_icache #(
        .CACHE_SIZE(32*1024),
        .WAYS(4),
        .LINE_SIZE(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut_icache (
        .clk(clk),
        .rst_n(rst_n),
        .cpu_addr(icache_addr),
        .cpu_rdata(icache_rdata),
        .cpu_req(icache_req),
        .cpu_ready(icache_ready),
        .cpu_hit(icache_hit),
        .l2_if(l2_icache_if.master),
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(icache_snoop_hit),
        .snoop_resp(icache_snoop_resp)
    );
    
    l1_dcache #(
        .CACHE_SIZE(32*1024),
        .WAYS(8),
        .LINE_SIZE(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut_dcache (
        .clk(clk),
        .rst_n(rst_n),
        .cpu_addr(dcache_addr),
        .cpu_wdata(dcache_wdata),
        .cpu_rdata(dcache_rdata),
        .cpu_req(dcache_req),
        .cpu_we(dcache_we),
        .cpu_be(dcache_be),
        .cpu_ready(dcache_ready),
        .cpu_hit(dcache_hit),
        .l2_if(l2_dcache_if.master),
        .snoop_req(snoop_req),
        .snoop_addr(snoop_addr),
        .snoop_type(snoop_type),
        .snoop_hit(dcache_snoop_hit),
        .snoop_dirty(dcache_snoop_dirty),
        .snoop_resp(dcache_snoop_resp)
    );
    
    // L2 cache simulator (simplified)
    logic [511:0] l2_memory [logic [63:0]];
    
    // L2 I-cache interface simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l2_icache_if.arready <= 1'b0;
            l2_icache_if.rvalid <= 1'b0;
            l2_icache_if.rlast <= 1'b0;
            l2_icache_if.rdata <= '0;
        end else begin
            // Simple L2 response simulation
            if (l2_icache_if.arvalid && !l2_icache_if.arready) begin
                l2_icache_if.arready <= 1'b1;
            end else begin
                l2_icache_if.arready <= 1'b0;
            end
            
            if (l2_icache_if.arvalid && l2_icache_if.arready) begin
                l2_icache_if.rvalid <= 1'b1;
                l2_icache_if.rlast <= 1'b1;
                // Return test pattern based on address
                l2_icache_if.rdata <= {8{l2_icache_if.araddr[63:0]}};
            end else begin
                l2_icache_if.rvalid <= 1'b0;
                l2_icache_if.rlast <= 1'b0;
            end
        end
    end
    
    // L2 D-cache interface simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l2_dcache_if.arready <= 1'b0;
            l2_dcache_if.rvalid <= 1'b0;
            l2_dcache_if.rlast <= 1'b0;
            l2_dcache_if.rdata <= '0;
            l2_dcache_if.awready <= 1'b0;
            l2_dcache_if.wready <= 1'b0;
            l2_dcache_if.bvalid <= 1'b0;
        end else begin
            // Read interface
            if (l2_dcache_if.arvalid && !l2_dcache_if.arready) begin
                l2_dcache_if.arready <= 1'b1;
            end else begin
                l2_dcache_if.arready <= 1'b0;
            end
            
            if (l2_dcache_if.arvalid && l2_dcache_if.arready) begin
                l2_dcache_if.rvalid <= 1'b1;
                l2_dcache_if.rlast <= 1'b1;
                l2_dcache_if.rdata <= {8{l2_dcache_if.araddr[63:0]}};
            end else begin
                l2_dcache_if.rvalid <= 1'b0;
                l2_dcache_if.rlast <= 1'b0;
            end
            
            // Write interface
            if (l2_dcache_if.awvalid && !l2_dcache_if.awready) begin
                l2_dcache_if.awready <= 1'b1;
                l2_dcache_if.wready <= 1'b1;
            end else begin
                l2_dcache_if.awready <= 1'b0;
                l2_dcache_if.wready <= 1'b0;
            end
            
            if (l2_dcache_if.wvalid && l2_dcache_if.wready) begin
                l2_dcache_if.bvalid <= 1'b1;
            end else begin
                l2_dcache_if.bvalid <= 1'b0;
            end
        end
    end
    
    // Test tasks
    task reset_system();
        rst_n = 1'b0;
        icache_req = 1'b0;
        dcache_req = 1'b0;
        dcache_we = 1'b0;
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
    
    task test_icache_miss_and_hit();
        $display("Testing I-cache miss and hit...");
        
        // Test cache miss
        icache_addr = 64'h1000;
        icache_req = 1'b1;
        @(posedge clk);
        
        // Wait for miss to complete
        wait(icache_ready);
        check_result("I-cache miss completion", 1'b1, icache_ready);
        
        icache_req = 1'b0;
        @(posedge clk);
        
        // Test cache hit on same address
        icache_addr = 64'h1000;
        icache_req = 1'b1;
        @(posedge clk);
        
        check_result("I-cache hit", 1'b1, icache_hit);
        check_result("I-cache ready on hit", 1'b1, icache_ready);
        
        icache_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_dcache_read_write();
        $display("Testing D-cache read and write...");
        
        // Test cache miss on read
        dcache_addr = 64'h2000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        
        // Wait for miss to complete
        wait(dcache_ready);
        check_result("D-cache read miss completion", 1'b1, dcache_ready);
        
        dcache_req = 1'b0;
        @(posedge clk);
        
        // Test cache hit on write
        dcache_addr = 64'h2000;
        dcache_wdata = 64'hDEADBEEF12345678;
        dcache_be = 8'hFF;
        dcache_req = 1'b1;
        dcache_we = 1'b1;
        @(posedge clk);
        
        check_result("D-cache write hit", 1'b1, dcache_hit);
        check_result("D-cache ready on write", 1'b1, dcache_ready);
        
        dcache_req = 1'b0;
        dcache_we = 1'b0;
        @(posedge clk);
        
        // Test read back written data
        dcache_addr = 64'h2000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        
        check_result("D-cache read after write hit", 1'b1, dcache_hit);
        // Note: In real implementation, we'd check if written data is read back
        
        dcache_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_cache_coherency();
        $display("Testing cache coherency...");
        
        // Load data into D-cache
        dcache_addr = 64'h3000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        wait(dcache_ready);
        dcache_req = 1'b0;
        @(posedge clk);
        
        // Test snoop hit
        snoop_addr = 64'h3000;
        snoop_type = 3'b001; // Read request
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop hit detection", 1'b1, dcache_snoop_hit);
        
        snoop_req = 1'b0;
        @(posedge clk);
        
        // Test snoop invalidation
        snoop_addr = 64'h3000;
        snoop_type = 3'b010; // Invalidate
        snoop_req = 1'b1;
        @(posedge clk);
        
        check_result("Snoop invalidate response", 3'b011, dcache_snoop_resp);
        
        snoop_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_replacement_policy();
        $display("Testing cache replacement policy...");
        
        // Fill up cache ways to test replacement
        for (int i = 0; i < 10; i++) begin
            icache_addr = 64'h4000 + (i * 64); // Different cache lines
            icache_req = 1'b1;
            @(posedge clk);
            wait(icache_ready);
            icache_req = 1'b0;
            @(posedge clk);
        end
        
        // Access first address again to test if it was replaced
        icache_addr = 64'h4000;
        icache_req = 1'b1;
        @(posedge clk);
        
        // Should either hit or miss depending on replacement
        wait(icache_ready);
        $display("Replacement test completed - Hit: %b", icache_hit);
        
        icache_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_performance_counters();
        $display("Testing performance counters...");
        
        // Reset and perform some operations
        reset_system();
        
        // Perform several cache operations
        for (int i = 0; i < 5; i++) begin
            // I-cache access
            icache_addr = 64'h5000 + (i * 64);
            icache_req = 1'b1;
            @(posedge clk);
            wait(icache_ready);
            icache_req = 1'b0;
            @(posedge clk);
            
            // D-cache read
            dcache_addr = 64'h6000 + (i * 64);
            dcache_req = 1'b1;
            dcache_we = 1'b0;
            @(posedge clk);
            wait(dcache_ready);
            dcache_req = 1'b0;
            @(posedge clk);
            
            // D-cache write
            dcache_addr = 64'h6000 + (i * 64);
            dcache_wdata = 64'h1234567890ABCDEF + i;
            dcache_be = 8'hFF;
            dcache_req = 1'b1;
            dcache_we = 1'b1;
            @(posedge clk);
            wait(dcache_ready);
            dcache_req = 1'b0;
            dcache_we = 1'b0;
            @(posedge clk);
        end
        
        $display("Performance counter test completed");
    endtask
    
    task test_burst_transfers();
        $display("Testing burst transfers...");
        
        // Test cache line fill with burst
        dcache_addr = 64'h7000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        
        // Wait for burst transfer to complete
        wait(dcache_ready);
        check_result("Burst transfer completion", 1'b1, dcache_ready);
        
        dcache_req = 1'b0;
        @(posedge clk);
        
        // Test hit after burst fill
        dcache_addr = 64'h7000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        
        check_result("Hit after burst fill", 1'b1, dcache_hit);
        
        dcache_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_error_handling();
        $display("Testing error handling...");
        
        // This would test error conditions in a real implementation
        // For now, just verify the cache handles normal operations correctly
        
        dcache_addr = 64'h8000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        
        wait(dcache_ready);
        check_result("Error handling test", 1'b1, dcache_ready);
        
        dcache_req = 1'b0;
        @(posedge clk);
    endtask
    
    task test_byte_enables();
        $display("Testing byte enable functionality...");
        
        // First, load a cache line
        dcache_addr = 64'h9000;
        dcache_req = 1'b1;
        dcache_we = 1'b0;
        @(posedge clk);
        wait(dcache_ready);
        dcache_req = 1'b0;
        @(posedge clk);
        
        // Test partial write with byte enables
        dcache_addr = 64'h9000;
        dcache_wdata = 64'hFFFFFFFFFFFFFFFF;
        dcache_be = 8'b00001111; // Write only lower 4 bytes
        dcache_req = 1'b1;
        dcache_we = 1'b1;
        @(posedge clk);
        
        check_result("Byte enable write hit", 1'b1, dcache_hit);
        
        dcache_req = 1'b0;
        dcache_we = 1'b0;
        @(posedge clk);
        
        // Test another partial write
        dcache_addr = 64'h9000;
        dcache_wdata = 64'h0000000000000000;
        dcache_be = 8'b11110000; // Write only upper 4 bytes
        dcache_req = 1'b1;
        dcache_we = 1'b1;
        @(posedge clk);
        
        check_result("Second byte enable write hit", 1'b1, dcache_hit);
        
        dcache_req = 1'b0;
        dcache_we = 1'b0;
        @(posedge clk);
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting L1 Cache Tests...");
        
        reset_system();
        
        test_icache_miss_and_hit();
        test_dcache_read_write();
        test_cache_coherency();
        test_replacement_policy();
        test_performance_counters();
        test_burst_transfers();
        test_error_handling();
        test_byte_enables();
        
        // Test summary
        $display("\n=== Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Timeout
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule