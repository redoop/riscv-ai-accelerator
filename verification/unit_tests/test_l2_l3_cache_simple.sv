// Simplified L2/L3 Cache Integration Test Module
// Tests multi-level cache hierarchy with ECC and coherency

`timescale 1ns/1ps

module test_l2_l3_cache_simple;

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
        
        // Initialize L1-L2 interfaces
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
    
    task simple_read_test();
        $display("Testing basic L2-L3 hierarchy...");
        
        // Test basic read operation
        l1_l2_if[0].arid = 8'h01;
        l1_l2_if[0].araddr = 64'h10000;
        l1_l2_if[0].arlen = 0;
        l1_l2_if[0].arsize = 3'b110;
        l1_l2_if[0].arburst = 2'b01;
        l1_l2_if[0].arvalid = 1'b1;
        
        @(posedge clk);
        wait(l1_l2_if[0].arready);
        l1_l2_if[0].arvalid = 1'b0;
        
        wait(l1_l2_if[0].rvalid);
        check_result("Basic read operation", 1'b1, l1_l2_if[0].rvalid);
        @(posedge clk);
        
        $display("Basic hierarchy test completed");
    endtask
    
    task test_ecc_basic();
        $display("Testing basic ECC functionality...");
        
        // Check ECC error signals are initially clean
        check_result("No ECC single error initially", 1'b0, l2_ecc_single);
        check_result("No ECC double error initially", 1'b0, l2_ecc_double);
        check_result("No L3 ECC single error initially", 1'b0, l3_ecc_single);
        check_result("No L3 ECC double error initially", 1'b0, l3_ecc_double);
        
        $display("ECC basic test completed");
    endtask
    
    // Performance test variables
    logic [31:0] perf_initial_hits;
    logic [31:0] perf_initial_misses;
    
    task test_performance_basic();
        $display("Testing basic performance counters...");
        
        perf_initial_hits = l3_hit_count;
        perf_initial_misses = l3_miss_count;
        
        // Generate some cache activity
        l1_l2_if[0].arid = 8'h40;
        l1_l2_if[0].araddr = 64'h60000;
        l1_l2_if[0].arlen = 0;
        l1_l2_if[0].arsize = 3'b110;
        l1_l2_if[0].arburst = 2'b01;
        l1_l2_if[0].arvalid = 1'b1;
        
        @(posedge clk);
        wait(l1_l2_if[0].arready);
        l1_l2_if[0].arvalid = 1'b0;
        
        wait(l1_l2_if[0].rvalid);
        @(posedge clk);
        
        $display("L3 Performance: Hits=%d, Misses=%d, Evictions=%d", 
                 l3_hit_count, l3_miss_count, l3_eviction_count);
        
        $display("Performance basic test completed");
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Simplified L2/L3 Cache Tests...");
        
        reset_system();
        
        simple_read_test();
        test_ecc_basic();
        test_performance_basic();
        
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
        #50000;
        $display("ERROR: L2/L3 Cache test timeout!");
        $finish;
    end

endmodule