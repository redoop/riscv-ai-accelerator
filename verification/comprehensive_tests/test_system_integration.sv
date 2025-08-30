// System Integration Test Suite
// Tests complete system functionality (Task 11.3)
// Multi-core coordination, hardware-software co-verification, stress testing

`timescale 1ns/1ps

module test_system_integration;

    // ========================================
    // System Parameters
    // ========================================
    
    parameter CLK_PERIOD = 10;  // 100MHz
    parameter NUM_CORES = 4;    // Multi-core system
    parameter STRESS_TEST_DURATION = 50000; // 500us stress test
    parameter CACHE_SIZE = 1024; // Cache entries
    
    // ========================================
    // System Signals
    // ========================================
    
    logic clk, rst_n;
    logic system_enable;
    logic [NUM_CORES-1:0] core_enable;
    logic [NUM_CORES-1:0] core_busy;
    logic [NUM_CORES-1:0] core_error;
    
    // Global system bus
    logic [63:0]    sys_addr;
    logic [63:0]    sys_wdata, sys_rdata;
    logic [7:0]     sys_be;
    logic           sys_req, sys_we, sys_ready;
    logic [3:0]     sys_id;
    
    // Cache coherency signals
    logic [NUM_CORES-1:0] cache_invalidate;
    logic [NUM_CORES-1:0] cache_flush;
    logic [NUM_CORES-1:0] cache_coherent;
    
    // NoC (Network on Chip) signals
    logic [NUM_CORES-1:0] noc_tx_valid;
    logic [NUM_CORES-1:0] noc_tx_ready;
    logic [NUM_CORES-1:0] noc_rx_valid;
    logic [NUM_CORES-1:0] noc_rx_ready;
    logic [31:0] noc_tx_data [NUM_CORES-1:0];
    logic [31:0] noc_rx_data [NUM_CORES-1:0];
    
    // Power management signals
    logic [NUM_CORES-1:0] power_gate_enable;
    logic [NUM_CORES-1:0] clock_gate_enable;
    logic [7:0] voltage_level [NUM_CORES-1:0];
    logic [7:0] frequency_level [NUM_CORES-1:0];
    
    // Thermal management
    logic [15:0] temperature [NUM_CORES-1:0];
    logic [NUM_CORES-1:0] thermal_throttle;
    logic [NUM_CORES-1:0] thermal_shutdown;
    
    // Interrupt system
    logic [NUM_CORES-1:0] ext_irq;
    logic [NUM_CORES-1:0] timer_irq;
    logic [NUM_CORES-1:0] ipi_irq; // Inter-processor interrupt
    
    // AI accelerator coordination
    logic [NUM_CORES-1:0] ai_accel_req;
    logic [NUM_CORES-1:0] ai_accel_grant;
    logic [NUM_CORES-1:0] ai_accel_busy;
    
    // Test control and monitoring
    logic [31:0] test_pass_count, test_fail_count;
    logic [63:0] system_cycle_counter;
    logic [31:0] error_count [NUM_CORES-1:0];
    logic [31:0] workload_completion [NUM_CORES-1:0];
    
    // ========================================
    // Clock and Reset
    // ========================================
    
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 20);
        rst_n = 1;
    end
    
    // System cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            system_cycle_counter <= 64'b0;
        end else begin
            system_cycle_counter <= system_cycle_counter + 1;
        end
    end
    
    // ========================================
    // System Memory Model
    // ========================================
    
    logic [63:0] system_memory [0:65535]; // 512KB shared memory
    logic [31:0] memory_access_count;
    logic [31:0] cache_hit_count, cache_miss_count;
    
    // Memory controller with arbitration
    logic [3:0] memory_arbiter_state;
    logic [3:0] current_master;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_ready <= 1'b0;
            memory_access_count <= 0;
            current_master <= 0;
        end else begin
            if (sys_req) begin
                memory_access_count <= memory_access_count + 1;
                
                // Simple round-robin arbitration
                if (sys_id == current_master) begin
                    sys_ready <= 1'b1;
                    if (!sys_we) begin
                        sys_rdata <= system_memory[sys_addr[15:3]];
                    end else begin
                        system_memory[sys_addr[15:3]] <= sys_wdata;
                    end
                    current_master <= (current_master + 1) % NUM_CORES;
                end else begin
                    sys_ready <= 1'b0;
                end
            end else begin
                sys_ready <= 1'b0;
            end
        end
    end
    
    // Initialize system memory
    initial begin
        for (int i = 0; i < 65536; i++) begin
            system_memory[i] = 64'h0123456789ABCDEF + (i << 8);
        end
    end
    
    // ========================================
    // Multi-Core System Instantiation
    // ========================================
    
    // AI accelerator interfaces for each core
    ai_accel_if ai_if [NUM_CORES-1:0] ();
    
    genvar core_id;
    generate
        for (core_id = 0; core_id < NUM_CORES; core_id++) begin : gen_cores
            
            // Connect AI interfaces
            assign ai_if[core_id].clk = clk;
            assign ai_if[core_id].rst_n = rst_n;
            
            // Core-specific memory interface
            logic [63:0] core_imem_addr, core_dmem_addr;
            logic [31:0] core_imem_rdata;
            logic [63:0] core_dmem_wdata, core_dmem_rdata;
            logic [7:0]  core_dmem_wmask;
            logic        core_imem_req, core_imem_ready;
            logic        core_dmem_req, core_dmem_we, core_dmem_ready;
            
            // Simple instruction memory per core
            logic [31:0] core_instructions [0:255];
            
            // Initialize core-specific instructions
            initial begin
                case (core_id)
                    0: begin // Core 0: Matrix operations
                        core_instructions[0] = 32'h00100093;  // addi x1, x0, 1
                        core_instructions[1] = 32'h00200113;  // addi x2, x0, 2
                        core_instructions[2] = 32'h002081b3;  // add x3, x1, x2
                        core_instructions[3] = 32'h0000006f;  // j 0 (loop)
                    end
                    1: begin // Core 1: Vector operations
                        core_instructions[0] = 32'h00300193;  // addi x3, x0, 3
                        core_instructions[1] = 32'h00400213;  // addi x4, x0, 4
                        core_instructions[2] = 32'h004181b3;  // add x3, x3, x4
                        core_instructions[3] = 32'h0000006f;  // j 0 (loop)
                    end
                    2: begin // Core 2: AI workload
                        core_instructions[0] = 32'h00500293;  // addi x5, x0, 5
                        core_instructions[1] = 32'h00600313;  // addi x6, x0, 6
                        core_instructions[2] = 32'h006282b3;  // add x5, x5, x6
                        core_instructions[3] = 32'h0000006f;  // j 0 (loop)
                    end
                    3: begin // Core 3: Memory intensive
                        core_instructions[0] = 32'h00700393;  // addi x7, x0, 7
                        core_instructions[1] = 32'h00800413;  // addi x8, x0, 8
                        core_instructions[2] = 32'h008383b3;  // add x7, x7, x8
                        core_instructions[3] = 32'h0000006f;  // j 0 (loop)
                    end
                endcase
                
                // Fill remaining with NOPs
                for (int i = 4; i < 256; i++) begin
                    core_instructions[i] = 32'h00000013; // NOP
                end
            end
            
            // Core instruction memory interface
            assign core_imem_ready = 1'b1;
            assign core_imem_rdata = core_instructions[core_imem_addr[9:2]];
            
            // Core data memory interface (connects to system bus)
            assign core_dmem_ready = sys_ready && (sys_id == core_id);
            assign core_dmem_rdata = sys_rdata;
            
            // Connect core to system bus
            always_comb begin
                if (core_dmem_req) begin
                    sys_addr = core_dmem_addr;
                    sys_wdata = core_dmem_wdata;
                    sys_be = core_dmem_wmask;
                    sys_req = 1'b1;
                    sys_we = core_dmem_we;
                    sys_id = core_id;
                end else begin
                    sys_addr = 64'b0;
                    sys_wdata = 64'b0;
                    sys_be = 8'b0;
                    sys_req = 1'b0;
                    sys_we = 1'b0;
                    sys_id = 4'b0;
                end
            end
            
            // RISC-V Core instance
            riscv_core #(
                .XLEN(64),
                .VLEN(512)
            ) core_inst (
                .clk(clk),
                .rst_n(rst_n && core_enable[core_id]),
                .imem_addr(core_imem_addr),
                .imem_req(core_imem_req),
                .imem_rdata(core_imem_rdata),
                .imem_ready(core_imem_ready),
                .dmem_addr(core_dmem_addr),
                .dmem_wdata(core_dmem_wdata),
                .dmem_wmask(core_dmem_wmask),
                .dmem_req(core_dmem_req),
                .dmem_we(core_dmem_we),
                .dmem_rdata(core_dmem_rdata),
                .dmem_ready(core_dmem_ready),
                .ai_if(ai_if[core_id].master),
                .ext_irq(ext_irq[core_id]),
                .timer_irq(timer_irq[core_id]),
                .soft_irq(ipi_irq[core_id])
            );
            
            // Core status monitoring
            assign core_busy[core_id] = core_imem_req || core_dmem_req;
            
            // Error counting per core
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    error_count[core_id] <= 0;
                    workload_completion[core_id] <= 0;
                end else begin
                    if (ai_if[core_id].error) begin
                        error_count[core_id] <= error_count[core_id] + 1;
                    end
                    if (ai_if[core_id].task_done) begin
                        workload_completion[core_id] <= workload_completion[core_id] + 1;
                    end
                end
            end
            
            assign core_error[core_id] = (error_count[core_id] > 0);
        end
    endgenerate
    
    // ========================================
    // Shared AI Accelerators
    // ========================================
    
    // TPU shared among cores
    logic           shared_tpu_enable, shared_tpu_start, shared_tpu_done, shared_tpu_busy;
    logic [7:0]     shared_tpu_operation;
    logic [1:0]     shared_tpu_data_type;
    logic [31:0]    shared_tpu_mem_addr, shared_tpu_mem_wdata, shared_tpu_mem_rdata;
    logic           shared_tpu_mem_read, shared_tpu_mem_write, shared_tpu_mem_ready;
    logic [31:0]    shared_tpu_cycle_count, shared_tpu_op_count;
    logic           shared_tpu_error;
    
    tpu #(
        .DATA_WIDTH(32),
        .ADDR_WIDTH(32),
        .ARRAY_SIZE(64)
    ) shared_tpu (
        .clk(clk),
        .rst_n(rst_n),
        .enable(shared_tpu_enable),
        .start(shared_tpu_start),
        .done(shared_tpu_done),
        .busy(shared_tpu_busy),
        .operation(shared_tpu_operation),
        .data_type(shared_tpu_data_type),
        .matrix_size_m(8'd16),
        .matrix_size_n(8'd16),
        .matrix_size_k(8'd16),
        .mem_addr(shared_tpu_mem_addr),
        .mem_read(shared_tpu_mem_read),
        .mem_write(shared_tpu_mem_write),
        .mem_wdata(shared_tpu_mem_wdata),
        .mem_rdata(shared_tpu_mem_rdata),
        .mem_ready(shared_tpu_mem_ready),
        .cycle_count(shared_tpu_cycle_count),
        .op_count(shared_tpu_op_count),
        .error(shared_tpu_error)
    );
    
    // TPU arbitration logic
    logic [1:0] tpu_arbiter_state;
    logic [1:0] tpu_current_core;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tpu_current_core <= 2'b0;
            ai_accel_grant <= 4'b0;
        end else begin
            // Simple round-robin arbitration for TPU access
            if (!shared_tpu_busy) begin
                for (int i = 0; i < NUM_CORES; i++) begin
                    if (ai_accel_req[i] && (i == tpu_current_core)) begin
                        ai_accel_grant[i] <= 1'b1;
                        tpu_current_core <= (tpu_current_core + 1) % NUM_CORES;
                        break;
                    end else begin
                        ai_accel_grant[i] <= 1'b0;
                    end
                end
            end
        end
    end
    
    // Connect TPU to system memory
    assign shared_tpu_mem_rdata = system_memory[shared_tpu_mem_addr[15:3]][31:0];
    assign shared_tpu_mem_ready = 1'b1;
    
    always_ff @(posedge clk) begin
        if (shared_tpu_mem_write) begin
            system_memory[shared_tpu_mem_addr[15:3]][31:0] <= shared_tpu_mem_wdata;
        end
    end
    
    // ========================================
    // Cache Coherency Controller
    // ========================================
    
    logic [NUM_CORES-1:0] cache_state [0:CACHE_SIZE-1];
    logic [31:0] cache_coherency_events;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cache_coherency_events <= 0;
            cache_coherent <= '1; // All caches start coherent
        end else begin
            // Simulate cache coherency protocol
            for (int core = 0; core < NUM_CORES; core++) begin
                if (ai_if[core].req && ai_if[core].we) begin
                    // Write operation - invalidate other caches
                    cache_invalidate <= ~(1 << core);
                    cache_coherency_events <= cache_coherency_events + 1;
                end
            end
        end
    end
    
    // ========================================
    // Power and Thermal Management
    // ========================================
    
    // Simulate temperature based on activity
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_CORES; i++) begin
                temperature[i] <= 16'd300; // 30.0°C
                voltage_level[i] <= 8'd100; // 1.0V
                frequency_level[i] <= 8'd100; // 100%
                thermal_throttle[i] <= 1'b0;
                thermal_shutdown[i] <= 1'b0;
            end
        end else begin
            for (int i = 0; i < NUM_CORES; i++) begin
                // Temperature increases with activity
                if (core_busy[i]) begin
                    temperature[i] <= temperature[i] + 1;
                end else if (temperature[i] > 300) begin
                    temperature[i] <= temperature[i] - 1;
                end
                
                // Thermal management
                if (temperature[i] > 16'd800) begin // 80°C
                    thermal_throttle[i] <= 1'b1;
                    frequency_level[i] <= 8'd50; // Reduce to 50%
                end else if (temperature[i] > 16'd1000) begin // 100°C
                    thermal_shutdown[i] <= 1'b1;
                    core_enable[i] <= 1'b0;
                end else begin
                    thermal_throttle[i] <= 1'b0;
                    thermal_shutdown[i] <= 1'b0;
                    frequency_level[i] <= 8'd100;
                end
            end
        end
    end
    
    // ========================================
    // Test Tasks
    // ========================================
    
    // Test 1: Multi-core coordination
    task test_multicore_coordination();
        $display("=== Test 1: Multi-core Coordination ===");
        
        // Enable all cores
        core_enable = '1;
        system_enable = 1'b1;
        
        repeat(1000) @(posedge clk);
        
        // Check if all cores are active
        if (&core_busy) begin
            $display("✓ All cores are active and executing");
            test_pass_count++;
        end else begin
            $display("✗ Not all cores are active: %b", core_busy);
            test_fail_count++;
        end
        
        // Check memory arbitration
        if (memory_access_count > 0) begin
            $display("✓ Memory arbitration working, accesses: %0d", memory_access_count);
            test_pass_count++;
        end else begin
            $display("✗ No memory accesses detected");
            test_fail_count++;
        end
        
        $display("Multi-core coordination test completed\n");
    endtask
    
    // Test 2: AI accelerator sharing
    task test_ai_accelerator_sharing();
        $display("=== Test 2: AI Accelerator Sharing ===");
        
        // Request TPU access from multiple cores
        ai_accel_req = 4'b1111; // All cores request
        
        repeat(100) @(posedge clk);
        
        // Check if arbitration is working
        if ($countones(ai_accel_grant) <= 1) begin
            $display("✓ TPU arbitration working correctly");
            test_pass_count++;
        end else begin
            $display("✗ TPU arbitration failed: %b", ai_accel_grant);
            test_fail_count++;
        end
        
        ai_accel_req = 4'b0000;
        
        $display("AI accelerator sharing test completed\n");
    endtask
    
    // Test 3: Cache coherency
    task test_cache_coherency();
        $display("=== Test 3: Cache Coherency ===");
        
        logic [31:0] initial_events = cache_coherency_events;
        
        // Simulate write operations from different cores
        for (int core = 0; core < NUM_CORES; core++) begin
            ai_if[core].addr = 32'h1000 + (core * 64);
            ai_if[core].wdata = 64'hDEADBEEF00000000 + core;
            ai_if[core].req = 1'b1;
            ai_if[core].we = 1'b1;
            
            repeat(10) @(posedge clk);
            
            ai_if[core].req = 1'b0;
            ai_if[core].we = 1'b0;
        end
        
        repeat(50) @(posedge clk);
        
        if (cache_coherency_events > initial_events) begin
            $display("✓ Cache coherency events detected: %0d", 
                     cache_coherency_events - initial_events);
            test_pass_count++;
        end else begin
            $display("✗ No cache coherency events detected");
            test_fail_count++;
        end
        
        $display("Cache coherency test completed\n");
    endtask
    
    // Test 4: Thermal management
    task test_thermal_management();
        $display("=== Test 4: Thermal Management ===");
        
        // Force high activity to increase temperature
        repeat(2000) @(posedge clk);
        
        // Check if thermal throttling activates
        if (|thermal_throttle) begin
            $display("✓ Thermal throttling activated");
            test_pass_count++;
        end else begin
            $display("✗ Thermal throttling not activated");
            test_fail_count++;
        end
        
        // Check temperature monitoring
        logic temp_increasing = 1'b0;
        for (int i = 0; i < NUM_CORES; i++) begin
            if (temperature[i] > 16'd350) begin // Above 35°C
                temp_increasing = 1'b1;
            end
        end
        
        if (temp_increasing) begin
            $display("✓ Temperature monitoring working");
            test_pass_count++;
        end else begin
            $display("✗ Temperature not increasing with activity");
            test_fail_count++;
        end
        
        $display("Thermal management test completed\n");
    endtask
    
    // Test 5: Stress test
    task test_system_stress();
        $display("=== Test 5: System Stress Test ===");
        
        logic [31:0] initial_errors = 0;
        logic [31:0] final_errors = 0;
        
        // Count initial errors
        for (int i = 0; i < NUM_CORES; i++) begin
            initial_errors += error_count[i];
        end
        
        // Run stress test
        core_enable = '1;
        ai_accel_req = 4'b1111;
        
        // Generate interrupts
        ext_irq = 4'b1010;
        timer_irq = 4'b0101;
        
        repeat(STRESS_TEST_DURATION) @(posedge clk);
        
        // Stop stress
        ai_accel_req = 4'b0000;
        ext_irq = 4'b0000;
        timer_irq = 4'b0000;
        
        // Count final errors
        for (int i = 0; i < NUM_CORES; i++) begin
            final_errors += error_count[i];
        end
        
        if ((final_errors - initial_errors) < 10) begin // Allow some errors
            $display("✓ System stable under stress, errors: %0d", 
                     final_errors - initial_errors);
            test_pass_count++;
        end else begin
            $display("✗ System unstable under stress, errors: %0d", 
                     final_errors - initial_errors);
            test_fail_count++;
        end
        
        $display("System stress test completed\n");
    endtask
    
    // Test 6: Hardware-software co-verification
    task test_hw_sw_coverification();
        $display("=== Test 6: Hardware-Software Co-verification ===");
        
        logic [31:0] total_completions = 0;
        
        // Check workload completion across all cores
        for (int i = 0; i < NUM_CORES; i++) begin
            total_completions += workload_completion[i];
        end
        
        if (total_completions > 0) begin
            $display("✓ Hardware-software interaction working, completions: %0d", 
                     total_completions);
            test_pass_count++;
        end else begin
            $display("✗ No hardware-software interactions detected");
            test_fail_count++;
        end
        
        // Check system bus utilization
        if (memory_access_count > 1000) begin
            $display("✓ System bus well utilized: %0d accesses", memory_access_count);
            test_pass_count++;
        end else begin
            $display("✗ Low system bus utilization: %0d accesses", memory_access_count);
            test_fail_count++;
        end
        
        $display("Hardware-software co-verification completed\n");
    endtask
    
    // ========================================
    // Main Test Sequence
    // ========================================
    
    initial begin
        $display("========================================");
        $display("System Integration Test Suite");
        $display("========================================\n");
        
        // Initialize
        system_enable = 1'b0;
        core_enable = 4'b0000;
        ai_accel_req = 4'b0000;
        ext_irq = 4'b0000;
        timer_irq = 4'b0000;
        ipi_irq = 4'b0000;
        test_pass_count = 0;
        test_fail_count = 0;
        
        // Wait for reset
        wait(rst_n);
        repeat(50) @(posedge clk);
        
        // Run integration tests
        test_multicore_coordination();
        test_ai_accelerator_sharing();
        test_cache_coherency();
        test_thermal_management();
        test_system_stress();
        test_hw_sw_coverification();
        
        // Final results
        $display("========================================");
        $display("System Integration Test Results:");
        $display("  Passed: %0d", test_pass_count);
        $display("  Failed: %0d", test_fail_count);
        $display("  System Cycles: %0d", system_cycle_counter);
        $display("  Memory Accesses: %0d", memory_access_count);
        $display("  Cache Coherency Events: %0d", cache_coherency_events);
        
        // Per-core statistics
        for (int i = 0; i < NUM_CORES; i++) begin
            $display("  Core %0d - Errors: %0d, Completions: %0d, Temp: %0d.%0d°C", 
                     i, error_count[i], workload_completion[i], 
                     temperature[i]/10, temperature[i]%10);
        end
        
        if (test_fail_count == 0) begin
            $display("🎉 ALL SYSTEM INTEGRATION TESTS PASSED! 🎉");
        end else begin
            $display("❌ %0d SYSTEM INTEGRATION TESTS FAILED", test_fail_count);
        end
        $display("========================================");
        
        $finish;
    end
    
    // Timeout protection
    initial begin
        repeat(STRESS_TEST_DURATION * 2) @(posedge clk);
        $display("ERROR: System integration test timeout!");
        $finish;
    end
    
    // System monitoring
    always @(posedge clk) begin
        if (system_cycle_counter % 10000 == 0) begin
            $display("System Status @ %0t: Cores=%b, Temp=[%0d,%0d,%0d,%0d], Mem=%0d", 
                     $time, core_busy, 
                     temperature[0]/10, temperature[1]/10, 
                     temperature[2]/10, temperature[3]/10,
                     memory_access_count);
        end
    end

endmodule