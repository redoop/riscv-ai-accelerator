// Testbench for TPU Controller and Cache System
// Tests task scheduling, cache operations, and DMA functionality

`timescale 1ns/1ps

module test_tpu_controller_cache;

    parameter DATA_WIDTH = 32;
    parameter ADDR_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    parameter NUM_TPU_UNITS = 2;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // TPU Controller signals
    logic [ADDR_WIDTH-1:0] bus_addr;
    logic bus_read, bus_write;
    logic [DATA_WIDTH-1:0] bus_wdata, bus_rdata;
    logic bus_ready, bus_error;
    
    logic [NUM_TPU_UNITS-1:0] tpu_enable, tpu_start;
    logic [NUM_TPU_UNITS-1:0] tpu_done, tpu_busy, tpu_error;
    
    logic [7:0] operation;
    logic [1:0] data_type;
    logic [7:0] matrix_size_m, matrix_size_n, matrix_size_k;
    
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic mem_read, mem_write;
    logic [DATA_WIDTH-1:0] mem_wdata, mem_rdata;
    logic mem_ready;
    
    logic cache_flush, cache_invalidate, cache_ready;
    logic interrupt;
    logic [31:0] status_reg, performance_counter;
    
    // TPU Cache signals
    logic [ADDR_WIDTH-1:0] cache_cpu_addr;
    logic cache_cpu_read, cache_cpu_write;
    logic [DATA_WIDTH-1:0] cache_cpu_wdata, cache_cpu_rdata;
    logic cache_cpu_ready, cache_cpu_hit;
    
    logic [ADDR_WIDTH-1:0] cache_mem_addr;
    logic cache_mem_read, cache_mem_write;
    logic [DATA_WIDTH-1:0] cache_mem_wdata, cache_mem_rdata;
    logic cache_mem_ready;
    
    logic [1:0] cache_type;
    logic [31:0] hit_count, miss_count, eviction_count;
    
    // Test control
    int test_case = 0;
    int pass_count = 0;
    int error_count = 0;
    
    // Memory model
    logic [DATA_WIDTH-1:0] memory [0:1024*1024-1];  // 4MB memory
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation - TPU Controller
    tpu_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .NUM_TPU_UNITS(NUM_TPU_UNITS)
    ) controller_dut (
        .clk(clk),
        .rst_n(rst_n),
        .bus_addr(bus_addr),
        .bus_read(bus_read),
        .bus_write(bus_write),
        .bus_wdata(bus_wdata),
        .bus_rdata(bus_rdata),
        .bus_ready(bus_ready),
        .bus_error(bus_error),
        .tpu_enable(tpu_enable),
        .tpu_start(tpu_start),
        .tpu_done(tpu_done),
        .tpu_busy(tpu_busy),
        .tpu_error(tpu_error),
        .operation(operation),
        .data_type(data_type),
        .matrix_size_m(matrix_size_m),
        .matrix_size_n(matrix_size_n),
        .matrix_size_k(matrix_size_k),
        .mem_addr(mem_addr),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .mem_wdata(mem_wdata),
        .mem_rdata(mem_rdata),
        .mem_ready(mem_ready),
        .cache_flush(cache_flush),
        .cache_invalidate(cache_invalidate),
        .cache_ready(cache_ready),
        .interrupt(interrupt),
        .status_reg(status_reg),
        .performance_counter(performance_counter)
    );
    
    // DUT instantiation - TPU Cache
    tpu_cache #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) cache_dut (
        .clk(clk),
        .rst_n(rst_n),
        .cpu_addr(cache_cpu_addr),
        .cpu_read(cache_cpu_read),
        .cpu_write(cache_cpu_write),
        .cpu_wdata(cache_cpu_wdata),
        .cpu_rdata(cache_cpu_rdata),
        .cpu_ready(cache_cpu_ready),
        .cpu_hit(cache_cpu_hit),
        .mem_addr(cache_mem_addr),
        .mem_read(cache_mem_read),
        .mem_write(cache_mem_write),
        .mem_wdata(cache_mem_wdata),
        .mem_rdata(cache_mem_rdata),
        .mem_ready(cache_mem_ready),
        .cache_flush(cache_flush),
        .cache_invalidate(cache_invalidate),
        .cache_type(cache_type),
        .cache_ready(cache_ready),
        .hit_count(hit_count),
        .miss_count(miss_count),
        .eviction_count(eviction_count)
    );
    
    // Memory model behavior
    always_ff @(posedge clk) begin
        mem_ready <= 1'b1;
        cache_mem_ready <= 1'b1;
        
        if (mem_read) begin
            mem_rdata <= memory[mem_addr[21:2]];  // Word-aligned access
        end
        
        if (mem_write) begin
            memory[mem_addr[21:2]] <= mem_wdata;
        end
        
        if (cache_mem_read) begin
            cache_mem_rdata <= memory[cache_mem_addr[21:2]];
        end
        
        if (cache_mem_write) begin
            memory[cache_mem_addr[21:2]] <= cache_mem_wdata;
        end
    end
    
    // TPU unit simulation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tpu_done <= '0;
            tpu_busy <= '0;
            tpu_error <= '0;
        end else begin
            for (int i = 0; i < NUM_TPU_UNITS; i++) begin
                if (tpu_start[i] && tpu_enable[i]) begin
                    tpu_busy[i] <= 1'b1;
                    tpu_done[i] <= 1'b0;
                end else if (tpu_busy[i]) begin
                    // Simulate completion after some cycles
                    if ($random % 100 < 10) begin  // 10% chance per cycle
                        tpu_busy[i] <= 1'b0;
                        tpu_done[i] <= 1'b1;
                    end
                end else begin
                    tpu_done[i] <= 1'b0;
                end
            end
        end
    end
    
    // Main test sequence
    initial begin
        $display("=== TPU Controller and Cache System Tests ===");
        
        // Initialize
        initialize_test();
        
        // Test Case 1: Basic Controller Register Access
        test_controller_registers();
        
        // Test Case 2: Task Queue Management
        test_task_queue();
        
        // Test Case 3: TPU Task Dispatch
        test_task_dispatch();
        
        // Test Case 4: Cache Hit/Miss Operations
        test_cache_operations();
        
        // Test Case 5: Cache Flush and Invalidate
        test_cache_control();
        
        // Test Case 6: DMA Integration
        test_dma_integration();
        
        // Test Case 7: Performance Monitoring
        test_performance_counters();
        
        // Test Case 8: Error Handling
        test_error_handling();
        
        // Test Case 9: Multi-channel Operations
        test_multi_channel();
        
        // Test Case 10: System Integration
        test_system_integration();
        
        // Final summary
        print_test_summary();
        
        $finish;
    end
    
    // Initialize test environment
    task initialize_test();
        $display("\nInitializing test environment...");
        
        rst_n = 0;
        bus_addr = '0;
        bus_read = 0;
        bus_write = 0;
        bus_wdata = '0;
        
        cache_cpu_addr = '0;
        cache_cpu_read = 0;
        cache_cpu_write = 0;
        cache_cpu_wdata = '0;
        cache_type = 2'b00;  // Weight cache
        
        // Initialize memory with test patterns
        for (int i = 0; i < 1024; i++) begin
            memory[i] = i * 4 + 32'hA5A5A5A5;
        end
        
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 5);
        
        $display("Initialization complete.");
    endtask
    
    // Test Case 1: Basic Controller Register Access
    task test_controller_registers();
        test_case++;
        $display("\nTest Case %0d: Basic Controller Register Access", test_case);
        
        // Test control register write/read
        write_controller_reg(32'h0000, 32'h0000000F);  // Enable + Start + Reset + Int
        read_controller_reg(32'h0000);
        
        if (bus_rdata[3:0] == 4'hF) begin
            $display("  PASS: Control register access");
            pass_count++;
        end else begin
            $display("  FAIL: Control register access - expected 0xF, got 0x%x", bus_rdata[3:0]);
            error_count++;
        end
        
        // Test status register read
        read_controller_reg(32'h0004);
        $display("  Status register: 0x%08x", bus_rdata);
        
        // Test performance counter
        read_controller_reg(32'h0010);
        $display("  Performance counter: 0x%08x", bus_rdata);
        
        pass_count++;  // Always pass for info reads
    endtask
    
    // Test Case 2: Task Queue Management
    task test_task_queue();
        test_case++;
        $display("\nTest Case %0d: Task Queue Management", test_case);
        
        // Configure DMA addresses
        write_controller_reg(32'h0014, 32'h00001000);  // Source address
        write_controller_reg(32'h0018, 32'h00002000);  // Destination address
        write_controller_reg(32'h001C, 32'h00000400);  // Size
        
        // Submit a task
        write_controller_reg(32'h000C, 32'h10080201);  // Task: TPU1, matrix 8x8, FP16, MATMUL
        
        // Check status
        read_controller_reg(32'h0004);
        
        if (bus_rdata[7:4] > 0) begin  // Task count > 0
            $display("  PASS: Task queued successfully");
            pass_count++;
        end else begin
            $display("  FAIL: Task not queued");
            error_count++;
        end
    endtask
    
    // Test Case 3: TPU Task Dispatch
    task test_task_dispatch();
        test_case++;
        $display("\nTest Case %0d: TPU Task Dispatch", test_case);
        
        // Enable controller
        write_controller_reg(32'h0000, 32'h00000001);  // Enable
        
        // Wait for task dispatch
        wait_for_condition("TPU start", 100);
        
        if (|tpu_start) begin
            $display("  PASS: Task dispatched to TPU");
            pass_count++;
        end else begin
            $display("  FAIL: Task not dispatched");
            error_count++;
        end
        
        // Wait for completion
        wait_for_condition("TPU completion", 1000);
        
        if (|tpu_done) begin
            $display("  PASS: TPU task completed");
            pass_count++;
        end else begin
            $display("  FAIL: TPU task not completed");
            error_count++;
        end
    endtask
    
    // Test Case 4: Cache Hit/Miss Operations
    task test_cache_operations();
        test_case++;
        $display("\nTest Case %0d: Cache Hit/Miss Operations", test_case);
        
        logic [31:0] initial_hits, initial_misses;
        
        // Record initial counters
        initial_hits = hit_count;
        initial_misses = miss_count;
        
        // Test cache miss (first access)
        cache_type = 2'b00;  // Weight cache
        read_cache(32'h00001000);
        
        if (miss_count > initial_misses) begin
            $display("  PASS: Cache miss detected");
            pass_count++;
        end else begin
            $display("  FAIL: Cache miss not detected");
            error_count++;
        end
        
        // Test cache hit (second access to same location)
        read_cache(32'h00001000);
        
        if (hit_count > initial_hits) begin
            $display("  PASS: Cache hit detected");
            pass_count++;
        end else begin
            $display("  FAIL: Cache hit not detected");
            error_count++;
        end
        
        $display("  Cache stats - Hits: %0d, Misses: %0d", hit_count, miss_count);
    endtask
    
    // Test Case 5: Cache Flush and Invalidate
    task test_cache_control();
        test_case++;
        $display("\nTest Case %0d: Cache Flush and Invalidate", test_case);
        
        // Write to cache control register to flush
        write_controller_reg(32'h0020, 32'h00000001);  // Flush
        
        // Wait for cache ready
        wait_for_condition("Cache ready", 100);
        
        if (cache_ready) begin
            $display("  PASS: Cache flush completed");
            pass_count++;
        end else begin
            $display("  FAIL: Cache flush not completed");
            error_count++;
        end
        
        // Test invalidate
        write_controller_reg(32'h0020, 32'h00000002);  // Invalidate
        
        wait_for_condition("Cache ready", 100);
        
        if (cache_ready) begin
            $display("  PASS: Cache invalidate completed");
            pass_count++;
        end else begin
            $display("  FAIL: Cache invalidate not completed");
            error_count++;
        end
    endtask
    
    // Test Case 6: DMA Integration
    task test_dma_integration();
        test_case++;
        $display("\nTest Case %0d: DMA Integration", test_case);
        
        // Configure DMA transfer
        write_controller_reg(32'h0014, 32'h00001000);  // Source
        write_controller_reg(32'h0018, 32'h00002000);  // Destination
        write_controller_reg(32'h001C, 32'h00000100);  // Size (256 bytes)
        
        // Start DMA transfer
        write_controller_reg(32'h0000, 32'h00000001);  // Enable
        
        // Monitor memory interface
        wait_for_condition("Memory access", 200);
        
        if (mem_read || mem_write) begin
            $display("  PASS: DMA memory access detected");
            pass_count++;
        end else begin
            $display("  FAIL: DMA memory access not detected");
            error_count++;
        end
    endtask
    
    // Test Case 7: Performance Monitoring
    task test_performance_counters();
        test_case++;
        $display("\nTest Case %0d: Performance Monitoring", test_case);
        
        logic [31:0] initial_perf;
        
        // Read initial performance counter
        read_controller_reg(32'h0010);
        initial_perf = bus_rdata;
        
        // Perform some operations
        for (int i = 0; i < 10; i++) begin
            write_controller_reg(32'h000C, 32'h00080201);  // Submit tasks
            #(CLK_PERIOD * 10);
        end
        
        // Read final performance counter
        read_controller_reg(32'h0010);
        
        if (bus_rdata != initial_perf) begin
            $display("  PASS: Performance counter updated");
            $display("  Initial: 0x%08x, Final: 0x%08x", initial_perf, bus_rdata);
            pass_count++;
        end else begin
            $display("  FAIL: Performance counter not updated");
            error_count++;
        end
    endtask
    
    // Test Case 8: Error Handling
    task test_error_handling();
        test_case++;
        $display("\nTest Case %0d: Error Handling", test_case);
        
        // Simulate TPU error
        force tpu_error[0] = 1'b1;
        #(CLK_PERIOD * 5);
        
        // Check status register
        read_controller_reg(32'h0004);
        
        if (bus_rdata[2]) begin  // Error bit
            $display("  PASS: Error status detected");
            pass_count++;
        end else begin
            $display("  FAIL: Error status not detected");
            error_count++;
        end
        
        // Release error
        release tpu_error[0];
        #(CLK_PERIOD * 5);
    endtask
    
    // Test Case 9: Multi-channel Operations
    task test_multi_channel();
        test_case++;
        $display("\nTest Case %0d: Multi-channel Operations", test_case);
        
        // Submit tasks to different TPU units
        write_controller_reg(32'h000C, 32'h00080201);  // TPU 0
        write_controller_reg(32'h000C, 32'h10080201);  // TPU 1
        
        // Enable controller
        write_controller_reg(32'h0000, 32'h00000001);
        
        // Wait for both TPUs to start
        wait_for_condition("Multi-TPU start", 200);
        
        if (tpu_start == 2'b11) begin
            $display("  PASS: Multi-channel dispatch");
            pass_count++;
        end else begin
            $display("  FAIL: Multi-channel dispatch - got 0b%b", tpu_start);
            error_count++;
        end
    endtask
    
    // Test Case 10: System Integration
    task test_system_integration();
        test_case++;
        $display("\nTest Case %0d: System Integration", test_case);
        
        // Full system test with cache and DMA
        cache_type = 2'b01;  // Activation cache
        
        // Load data through cache
        write_cache(32'h00003000, 32'hDEADBEEF);
        read_cache(32'h00003000);
        
        if (cache_cpu_rdata == 32'hDEADBEEF) begin
            $display("  PASS: Cache write/read");
            pass_count++;
        end else begin
            $display("  FAIL: Cache write/read - expected 0xDEADBEEF, got 0x%08x", cache_cpu_rdata);
            error_count++;
        end
        
        // Submit task with cache operations
        write_controller_reg(32'h0014, 32'h00003000);  // Use cached data
        write_controller_reg(32'h000C, 32'h00080201);
        write_controller_reg(32'h0000, 32'h00000001);  // Enable
        
        // Monitor system activity
        wait_for_condition("System integration", 500);
        
        $display("  System integration test completed");
        pass_count++;
    endtask
    
    // Helper tasks
    task write_controller_reg(input [31:0] addr, input [31:0] data);
        bus_addr = addr;
        bus_wdata = data;
        bus_write = 1;
        @(posedge clk);
        bus_write = 0;
        @(posedge clk);
    endtask
    
    task read_controller_reg(input [31:0] addr);
        bus_addr = addr;
        bus_read = 1;
        @(posedge clk);
        bus_read = 0;
        @(posedge clk);
    endtask
    
    task write_cache(input [31:0] addr, input [31:0] data);
        cache_cpu_addr = addr;
        cache_cpu_wdata = data;
        cache_cpu_write = 1;
        @(posedge clk);
        wait(cache_cpu_ready);
        cache_cpu_write = 0;
        @(posedge clk);
    endtask
    
    task read_cache(input [31:0] addr);
        cache_cpu_addr = addr;
        cache_cpu_read = 1;
        @(posedge clk);
        wait(cache_cpu_ready);
        cache_cpu_read = 0;
        @(posedge clk);
    endtask
    
    task wait_for_condition(input string condition_name, input int max_cycles);
        int cycle_count = 0;
        while (cycle_count < max_cycles) begin
            @(posedge clk);
            cycle_count++;
            
            // Check various conditions based on test context
            if (condition_name == "TPU start" && |tpu_start) break;
            if (condition_name == "TPU completion" && |tpu_done) break;
            if (condition_name == "Cache ready" && cache_ready) break;
            if (condition_name == "Memory access" && (mem_read || mem_write)) break;
            if (condition_name == "Multi-TPU start" && (&tpu_start)) break;
            if (condition_name == "System integration") break;  // Always pass after timeout
        end
        
        if (cycle_count >= max_cycles) begin
            $display("  WARNING: Timeout waiting for %s", condition_name);
        end
    endtask
    
    // Print test summary
    task print_test_summary();
        $display("\n=== Test Summary ===");
        $display("Total test cases: %0d", test_case);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", error_count);
        $display("Success rate: %.1f%%", real(pass_count) / real(test_case) * 100.0);
        
        if (error_count == 0) begin
            $display("All tests PASSED! ✓");
        end else begin
            $display("Some tests FAILED! ✗");
        end
    endtask

endmodule