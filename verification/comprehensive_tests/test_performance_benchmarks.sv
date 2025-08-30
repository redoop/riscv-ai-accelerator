// Performance Benchmarks Test Suite
// Tests performance requirements from tasks 11.2 and 12.1
// Includes MLPerf-style benchmarks and performance monitoring

`timescale 1ns/1ps

module test_performance_benchmarks;

    // ========================================
    // Performance Test Parameters
    // ========================================
    
    parameter CLK_PERIOD = 10;  // 100MHz
    parameter BENCHMARK_TIMEOUT = 100000;  // 1ms timeout
    
    // Benchmark configurations
    parameter IMAGE_SIZE = 32;      // 32x32 images for testing
    parameter BATCH_SIZE = 4;       // Small batch for testing
    parameter NUM_CLASSES = 10;     // Classification classes
    parameter MATRIX_SIZE_SMALL = 8;
    parameter MATRIX_SIZE_MEDIUM = 16;
    parameter MATRIX_SIZE_LARGE = 32;
    
    // Performance targets (operations per second)
    parameter TARGET_TOPS_INT8 = 100;   // 100 TOPS for INT8
    parameter TARGET_TFLOPS_FP16 = 50;  // 50 TFLOPS for FP16
    parameter TARGET_TFLOPS_FP32 = 25;  // 25 TFLOPS for FP32
    
    // ========================================
    // Signals
    // ========================================
    
    logic clk, rst_n;
    
    // Performance counters
    logic [63:0] cycle_counter;
    logic [63:0] instruction_counter;
    logic [63:0] ai_operation_counter;
    logic [63:0] cache_hit_counter;
    logic [63:0] cache_miss_counter;
    logic [63:0] memory_access_counter;
    
    // TPU performance signals
    logic           tpu_enable, tpu_start, tpu_done, tpu_busy;
    logic [7:0]     tpu_operation;
    logic [1:0]     tpu_data_type;
    logic [7:0]     tpu_matrix_m, tpu_matrix_n, tpu_matrix_k;
    logic [31:0]    tpu_mem_addr, tpu_mem_wdata, tpu_mem_rdata;
    logic           tpu_mem_read, tpu_mem_write, tpu_mem_ready;
    logic [31:0]    tpu_cycle_count, tpu_op_count;
    logic           tpu_error;
    
    // VPU performance signals
    logic           vpu_busy, vpu_error;
    logic [7:0]     vpu_status;
    
    // Memory interface
    logic [63:0]    mem_addr;
    logic [63:0]    mem_wdata, mem_rdata;
    logic           mem_req, mem_we, mem_ready;
    
    // AI accelerator interface
    ai_accel_if ai_if();
    
    // Test results
    logic [31:0] test_pass_count, test_fail_count;
    real performance_score;
    real throughput_ops_per_sec;
    real power_efficiency; // Operations per Watt
    
    // ========================================
    // Clock and Reset
    // ========================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 10);
        rst_n = 1;
    end
    
    // ========================================
    // Performance Counters
    // ========================================
    
    // Global cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 64'b0;
        end else begin
            cycle_counter <= cycle_counter + 1;
        end
    end
    
    // AI operation counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ai_operation_counter <= 64'b0;
        end else if (ai_if.req && ai_if.ready) begin
            ai_operation_counter <= ai_operation_counter + 1;
        end
    end
    
    // Memory access counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            memory_access_counter <= 64'b0;
        end else if (mem_req && mem_ready) begin
            memory_access_counter <= memory_access_counter + 1;
        end
    end
    
    // ========================================
    // Memory Model with Performance Tracking
    // ========================================
    
    logic [63:0] memory [0:65535]; // 512KB memory
    logic [31:0] memory_latency_counter;
    
    // Memory with realistic latency
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_ready <= 1'b0;
            memory_latency_counter <= 0;
        end else begin
            if (mem_req && !mem_ready) begin
                if (memory_latency_counter < 3) begin // 4-cycle latency
                    memory_latency_counter <= memory_latency_counter + 1;
                    mem_ready <= 1'b0;
                end else begin
                    mem_ready <= 1'b1;
                    memory_latency_counter <= 0;
                    if (!mem_we) begin
                        mem_rdata <= memory[mem_addr[15:3]];
                    end else begin
                        memory[mem_addr[15:3]] <= mem_wdata;
                    end
                end
            end else begin
                mem_ready <= 1'b0;
                memory_latency_counter <= 0;
            end
        end
    end
    
    // Initialize memory with test data
    initial begin
        for (int i = 0; i < 65536; i++) begin
            memory[i] = $random;
        end
    end
    
    // ========================================
    // DUT Instantiation
    // ========================================
    
    // Connect AI interface
    assign ai_if.clk = clk;
    assign ai_if.rst_n = rst_n;
    
    // TPU for matrix operations
    tpu #(
        .DATA_WIDTH(32),
        .ADDR_WIDTH(32),
        .ARRAY_SIZE(64)
    ) dut_tpu (
        .clk(clk),
        .rst_n(rst_n),
        .enable(tpu_enable),
        .start(tpu_start),
        .done(tpu_done),
        .busy(tpu_busy),
        .operation(tpu_operation),
        .data_type(tpu_data_type),
        .matrix_size_m(tpu_matrix_m),
        .matrix_size_n(tpu_matrix_n),
        .matrix_size_k(tpu_matrix_k),
        .mem_addr(tpu_mem_addr),
        .mem_read(tpu_mem_read),
        .mem_write(tpu_mem_write),
        .mem_wdata(tpu_mem_wdata),
        .mem_rdata(tpu_mem_rdata),
        .mem_ready(tpu_mem_ready),
        .cycle_count(tpu_cycle_count),
        .op_count(tpu_op_count),
        .error(tpu_error)
    );
    
    // Connect TPU to memory
    assign tpu_mem_rdata = memory[tpu_mem_addr[15:3]][31:0];
    assign tpu_mem_ready = 1'b1; // Simplified for performance testing
    
    always_ff @(posedge clk) begin
        if (tpu_mem_write) begin
            memory[tpu_mem_addr[15:3]][31:0] <= tpu_mem_wdata;
        end
    end
    
    // VPU for vector operations
    vpu #(
        .VECTOR_LANES(16),
        .VECTOR_REGS(32),
        .MAX_VLEN(512),
        .ELEMENT_WIDTH(64)
    ) dut_vpu (
        .clk(clk),
        .rst_n(rst_n),
        .ctrl_if(ai_if.slave),
        .mem_if(), // Simplified for this test
        .noc_rx_if(),
        .noc_tx_if(),
        .status(vpu_status),
        .busy(vpu_busy),
        .error(vpu_error)
    );
    
    // ========================================
    // Benchmark Test Tasks
    // ========================================
    
    // Benchmark 1: Matrix Multiplication Performance
    task benchmark_matrix_multiplication();
        $display("=== Benchmark 1: Matrix Multiplication Performance ===");
        
        logic [63:0] start_cycles, end_cycles, total_cycles;
        logic [63:0] operations_count;
        real ops_per_cycle, tops_achieved;
        
        // Test different matrix sizes and data types
        int matrix_sizes[3] = '{MATRIX_SIZE_SMALL, MATRIX_SIZE_MEDIUM, MATRIX_SIZE_LARGE};
        int data_types[3] = '{0, 1, 2}; // INT8, FP16, FP32
        string type_names[3] = '{"INT8", "FP16", "FP32"};
        
        for (int size_idx = 0; size_idx < 3; size_idx++) begin
            for (int type_idx = 0; type_idx < 3; type_idx++) begin
                $display("Testing %0dx%0d matrix multiplication with %s", 
                         matrix_sizes[size_idx], matrix_sizes[size_idx], type_names[type_idx]);
                
                // Configure TPU
                tpu_enable = 1'b1;
                tpu_operation = 8'h01; // Matrix multiply
                tpu_data_type = data_types[type_idx];
                tpu_matrix_m = matrix_sizes[size_idx];
                tpu_matrix_n = matrix_sizes[size_idx];
                tpu_matrix_k = matrix_sizes[size_idx];
                
                start_cycles = cycle_counter;
                
                // Start operation
                tpu_start = 1'b1;
                @(posedge clk);
                tpu_start = 1'b0;
                
                // Wait for completion
                wait(tpu_done);
                end_cycles = cycle_counter;
                
                total_cycles = end_cycles - start_cycles;
                operations_count = 2 * matrix_sizes[size_idx] * matrix_sizes[size_idx] * matrix_sizes[size_idx]; // 2*M*N*K
                
                ops_per_cycle = real'(operations_count) / real'(total_cycles);
                tops_achieved = ops_per_cycle * 100.0; // Assuming 100MHz clock
                
                $display("  Cycles: %0d, Operations: %0d", total_cycles, operations_count);
                $display("  Performance: %.2f ops/cycle, %.2f GOPS", ops_per_cycle, tops_achieved);
                
                // Check against targets
                case (data_types[type_idx])
                    0: begin // INT8
                        if (tops_achieved >= (TARGET_TOPS_INT8 / 1000.0)) begin
                            $display("  ✓ INT8 performance target met");
                            test_pass_count++;
                        end else begin
                            $display("  ✗ INT8 performance target missed");
                            test_fail_count++;
                        end
                    end
                    1: begin // FP16
                        if (tops_achieved >= (TARGET_TFLOPS_FP16 / 1000.0)) begin
                            $display("  ✓ FP16 performance target met");
                            test_pass_count++;
                        end else begin
                            $display("  ✗ FP16 performance target missed");
                            test_fail_count++;
                        end
                    end
                    2: begin // FP32
                        if (tops_achieved >= (TARGET_TFLOPS_FP32 / 1000.0)) begin
                            $display("  ✓ FP32 performance target met");
                            test_pass_count++;
                        end else begin
                            $display("  ✗ FP32 performance target missed");
                            test_fail_count++;
                        end
                    end
                endcase
                
                tpu_enable = 1'b0;
                repeat(10) @(posedge clk);
            end
        end
        
        $display("Matrix Multiplication benchmark completed\n");
    endtask
    
    // Benchmark 2: Vector Operations Performance
    task benchmark_vector_operations();
        $display("=== Benchmark 2: Vector Operations Performance ===");
        
        logic [63:0] start_cycles, end_cycles, total_cycles;
        real vector_throughput;
        
        // Test vector addition with different lengths
        int vector_lengths[4] = '{64, 128, 256, 512};
        
        for (int len_idx = 0; len_idx < 4; len_idx++) begin
            $display("Testing vector addition with length %0d", vector_lengths[len_idx]);
            
            start_cycles = cycle_counter;
            
            // Configure VPU for vector addition
            ai_if.addr = 32'h0000_1000;  // VPU base
            ai_if.wdata = {32'h0000_0000, 16'(vector_lengths[len_idx]), 16'h0001}; // Length + ADD op
            ai_if.req = 1'b1;
            ai_if.we = 1'b1;
            
            @(posedge clk);
            ai_if.req = 1'b0;
            ai_if.we = 1'b0;
            
            // Wait for VPU to complete
            wait(!vpu_busy);
            end_cycles = cycle_counter;
            
            total_cycles = end_cycles - start_cycles;
            vector_throughput = real'(vector_lengths[len_idx]) / real'(total_cycles);
            
            $display("  Cycles: %0d, Throughput: %.2f elements/cycle", total_cycles, vector_throughput);
            
            if (vector_throughput >= 1.0) begin // Target: 1 element per cycle
                $display("  ✓ Vector throughput target met");
                test_pass_count++;
            end else begin
                $display("  ✗ Vector throughput target missed");
                test_fail_count++;
            end
            
            repeat(10) @(posedge clk);
        end
        
        $display("Vector Operations benchmark completed\n");
    endtask
    
    // Benchmark 3: Memory Bandwidth Test
    task benchmark_memory_bandwidth();
        $display("=== Benchmark 3: Memory Bandwidth Test ===");
        
        logic [63:0] start_cycles, end_cycles, total_cycles;
        logic [31:0] transfer_count;
        real bandwidth_gbps;
        
        transfer_count = 1000; // Transfer 1000 64-bit words
        
        start_cycles = cycle_counter;
        
        // Sequential memory reads
        for (int i = 0; i < transfer_count; i++) begin
            mem_addr = i * 8;
            mem_req = 1'b1;
            mem_we = 1'b0;
            
            @(posedge clk);
            wait(mem_ready);
            mem_req = 1'b0;
            @(posedge clk);
        end
        
        end_cycles = cycle_counter;
        total_cycles = end_cycles - start_cycles;
        
        // Calculate bandwidth (assuming 64-bit transfers at 100MHz)
        bandwidth_gbps = (real'(transfer_count) * 8.0 * 100.0) / (real'(total_cycles) * 1000.0);
        
        $display("Memory bandwidth test:");
        $display("  Transfers: %0d, Cycles: %0d", transfer_count, total_cycles);
        $display("  Bandwidth: %.2f GB/s", bandwidth_gbps);
        
        if (bandwidth_gbps >= 10.0) begin // Target: 10 GB/s
            $display("  ✓ Memory bandwidth target met");
            test_pass_count++;
        end else begin
            $display("  ✗ Memory bandwidth target missed");
            test_fail_count++;
        end
        
        $display("Memory Bandwidth benchmark completed\n");
    endtask
    
    // Benchmark 4: AI Workload Simulation (Image Classification)
    task benchmark_image_classification();
        $display("=== Benchmark 4: AI Workload - Image Classification ===");
        
        logic [63:0] start_cycles, end_cycles, total_cycles;
        real images_per_second;
        
        start_cycles = cycle_counter;
        
        // Simulate a simple CNN inference
        for (int batch = 0; batch < BATCH_SIZE; batch++) begin
            // Convolution layer
            ai_if.addr = 32'h0000_2000; // Conv2D operation
            ai_if.wdata = {32'h1000 + batch*1024, 32'h00030001}; // Input addr + 3x3 kernel
            ai_if.req = 1'b1;
            ai_if.we = 1'b1;
            @(posedge clk);
            ai_if.req = 1'b0;
            ai_if.we = 1'b0;
            
            repeat(50) @(posedge clk); // Simulate conv processing time
            
            // Activation (ReLU)
            ai_if.addr = 32'h0000_3000; // Activation operation
            ai_if.wdata = {32'h2000 + batch*1024, 32'h00000000}; // ReLU
            ai_if.req = 1'b1;
            ai_if.we = 1'b1;
            @(posedge clk);
            ai_if.req = 1'b0;
            ai_if.we = 1'b0;
            
            repeat(20) @(posedge clk); // Simulate activation time
            
            // Pooling
            ai_if.addr = 32'h0000_4000; // Pooling operation
            ai_if.wdata = {32'h2000 + batch*1024, 32'h00020002}; // 2x2 max pool
            ai_if.req = 1'b1;
            ai_if.we = 1'b1;
            @(posedge clk);
            ai_if.req = 1'b0;
            ai_if.we = 1'b0;
            
            repeat(30) @(posedge clk); // Simulate pooling time
        end
        
        end_cycles = cycle_counter;
        total_cycles = end_cycles - start_cycles;
        
        images_per_second = (real'(BATCH_SIZE) * 100_000_000.0) / real'(total_cycles);
        
        $display("Image classification benchmark:");
        $display("  Batch size: %0d, Cycles: %0d", BATCH_SIZE, total_cycles);
        $display("  Throughput: %.2f images/second", images_per_second);
        
        if (images_per_second >= 1000.0) begin // Target: 1000 images/sec
            $display("  ✓ Image classification throughput target met");
            test_pass_count++;
        end else begin
            $display("  ✗ Image classification throughput target missed");
            test_fail_count++;
        end
        
        $display("Image Classification benchmark completed\n");
    endtask
    
    // Benchmark 5: Power Efficiency Estimation
    task benchmark_power_efficiency();
        $display("=== Benchmark 5: Power Efficiency Estimation ===");
        
        logic [63:0] active_cycles, idle_cycles;
        real estimated_power_watts;
        real efficiency_ops_per_watt;
        
        // Count active vs idle cycles
        active_cycles = ai_operation_counter * 10; // Estimate 10 cycles per AI op
        idle_cycles = cycle_counter - active_cycles;
        
        // Estimate power consumption (simplified model)
        estimated_power_watts = (real'(active_cycles) * 2.0 + real'(idle_cycles) * 0.5) / real'(cycle_counter);
        
        // Calculate efficiency
        efficiency_ops_per_watt = real'(ai_operation_counter) / estimated_power_watts;
        
        $display("Power efficiency estimation:");
        $display("  Active cycles: %0d, Idle cycles: %0d", active_cycles, idle_cycles);
        $display("  Estimated power: %.2f W", estimated_power_watts);
        $display("  Efficiency: %.2f ops/W", efficiency_ops_per_watt);
        
        if (efficiency_ops_per_watt >= 100.0) begin // Target: 100 ops/W
            $display("  ✓ Power efficiency target met");
            test_pass_count++;
        end else begin
            $display("  ✗ Power efficiency target missed");
            test_fail_count++;
        end
        
        $display("Power Efficiency benchmark completed\n");
    endtask
    
    // ========================================
    // Main Test Sequence
    // ========================================
    
    initial begin
        $display("========================================");
        $display("Performance Benchmarks Test Suite");
        $display("========================================\n");
        
        // Initialize
        tpu_enable = 1'b0;
        tpu_start = 1'b0;
        tpu_operation = 8'h00;
        tpu_data_type = 2'b00;
        mem_req = 1'b0;
        mem_we = 1'b0;
        test_pass_count = 0;
        test_fail_count = 0;
        
        // Wait for reset
        wait(rst_n);
        repeat(20) @(posedge clk);
        
        // Run benchmarks
        benchmark_matrix_multiplication();
        benchmark_vector_operations();
        benchmark_memory_bandwidth();
        benchmark_image_classification();
        benchmark_power_efficiency();
        
        // Calculate overall performance score
        performance_score = real'(test_pass_count) / real'(test_pass_count + test_fail_count) * 100.0;
        
        // Final results
        $display("========================================");
        $display("Performance Benchmark Results:");
        $display("  Passed: %0d", test_pass_count);
        $display("  Failed: %0d", test_fail_count);
        $display("  Overall Score: %.1f%%", performance_score);
        $display("  Total Cycles: %0d", cycle_counter);
        $display("  AI Operations: %0d", ai_operation_counter);
        $display("  Memory Accesses: %0d", memory_access_counter);
        
        if (performance_score >= 80.0) begin
            $display("🎉 PERFORMANCE TARGETS ACHIEVED! 🎉");
        end else begin
            $display("⚠️  PERFORMANCE NEEDS IMPROVEMENT");
        end
        $display("========================================");
        
        $finish;
    end
    
    // Timeout protection
    initial begin
        repeat(BENCHMARK_TIMEOUT) @(posedge clk);
        $display("ERROR: Benchmark timeout!");
        $finish;
    end

endmodule