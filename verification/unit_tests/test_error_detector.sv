/**
 * Testbench for Error Detection and Reporting System
 * Verifies error collection, prioritization, and reporting
 */

`timescale 1ns/1ps

module test_error_detector;

    // Parameters
    parameter NUM_CORES = 4;
    parameter NUM_TPUS = 2;
    parameter NUM_VPUS = 2;
    parameter CLK_PERIOD = 10;

    // Signals
    logic clk;
    logic rst_n;
    
    // ECC error inputs
    logic [NUM_CORES-1:0]    l1_cache_single_error;
    logic [NUM_CORES-1:0]    l1_cache_double_error;
    logic                    l2_cache_single_error;
    logic                    l2_cache_double_error;
    logic                    l3_cache_single_error;
    logic                    l3_cache_double_error;
    logic                    memory_single_error;
    logic                    memory_double_error;
    
    // Compute unit error inputs
    logic [NUM_CORES-1:0]    core_arithmetic_error;
    logic [NUM_CORES-1:0]    core_pipeline_error;
    logic [NUM_TPUS-1:0]     tpu_compute_error;
    logic [NUM_TPUS-1:0]     tpu_overflow_error;
    logic [NUM_VPUS-1:0]     vpu_compute_error;
    logic [NUM_VPUS-1:0]     vpu_overflow_error;
    
    // System error inputs
    logic                    noc_deadlock_error;
    logic                    noc_timeout_error;
    logic                    power_domain_error;
    logic                    thermal_error;
    logic                    clock_error;
    
    // Error status outputs
    logic                    error_interrupt;
    logic [31:0]             error_status;
    logic [31:0]             error_mask;
    logic [7:0]              error_severity;
    
    // Error logging interface
    logic                    error_log_valid;
    logic [63:0]             error_log_data;
    logic [31:0]             error_timestamp;
    
    // Control interface
    logic                    error_clear;
    logic [31:0]             error_mask_set;
    logic                    error_inject_enable;
    logic [4:0]              error_inject_type;

    // DUT instantiation
    error_detector #(
        .NUM_CORES(NUM_CORES),
        .NUM_TPUS(NUM_TPUS),
        .NUM_VPUS(NUM_VPUS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .l1_cache_single_error(l1_cache_single_error),
        .l1_cache_double_error(l1_cache_double_error),
        .l2_cache_single_error(l2_cache_single_error),
        .l2_cache_double_error(l2_cache_double_error),
        .l3_cache_single_error(l3_cache_single_error),
        .l3_cache_double_error(l3_cache_double_error),
        .memory_single_error(memory_single_error),
        .memory_double_error(memory_double_error),
        .core_arithmetic_error(core_arithmetic_error),
        .core_pipeline_error(core_pipeline_error),
        .tpu_compute_error(tpu_compute_error),
        .tpu_overflow_error(tpu_overflow_error),
        .vpu_compute_error(vpu_compute_error),
        .vpu_overflow_error(vpu_overflow_error),
        .noc_deadlock_error(noc_deadlock_error),
        .noc_timeout_error(noc_timeout_error),
        .power_domain_error(power_domain_error),
        .thermal_error(thermal_error),
        .clock_error(clock_error),
        .error_interrupt(error_interrupt),
        .error_status(error_status),
        .error_mask(error_mask),
        .error_severity(error_severity),
        .error_log_valid(error_log_valid),
        .error_log_data(error_log_data),
        .error_timestamp(error_timestamp),
        .error_clear(error_clear),
        .error_mask_set(error_mask_set),
        .error_inject_enable(error_inject_enable),
        .error_inject_type(error_inject_type)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test variables
    int error_count;
    int test_count;

    // Test tasks
    task reset_system();
        rst_n = 0;
        l1_cache_single_error = 0;
        l1_cache_double_error = 0;
        l2_cache_single_error = 0;
        l2_cache_double_error = 0;
        l3_cache_single_error = 0;
        l3_cache_double_error = 0;
        memory_single_error = 0;
        memory_double_error = 0;
        core_arithmetic_error = 0;
        core_pipeline_error = 0;
        tpu_compute_error = 0;
        tpu_overflow_error = 0;
        vpu_compute_error = 0;
        vpu_overflow_error = 0;
        noc_deadlock_error = 0;
        noc_timeout_error = 0;
        power_domain_error = 0;
        thermal_error = 0;
        clock_error = 0;
        error_clear = 0;
        error_mask_set = 32'hFFFFFFFF;
        error_inject_enable = 0;
        error_inject_type = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
    endtask

    task clear_errors();
        error_clear = 1;
        @(posedge clk);
        error_clear = 0;
        @(posedge clk);
    endtask

    task set_error_mask(input [31:0] mask);
        error_mask_set = mask;
        @(posedge clk);
    endtask

    task inject_single_error(input int error_bit);
        case (error_bit)
            0: l1_cache_single_error[0] = 1;
            1: l1_cache_double_error[0] = 1;
            2: l2_cache_single_error = 1;
            3: l2_cache_double_error = 1;
            4: l3_cache_single_error = 1;
            5: l3_cache_double_error = 1;
            6: memory_single_error = 1;
            7: memory_double_error = 1;
            8: core_arithmetic_error[0] = 1;
            9: core_pipeline_error[0] = 1;
            10: tpu_compute_error[0] = 1;
            11: tpu_overflow_error[0] = 1;
            12: vpu_compute_error[0] = 1;
            13: vpu_overflow_error[0] = 1;
            14: noc_deadlock_error = 1;
            15: noc_timeout_error = 1;
            16: power_domain_error = 1;
            17: thermal_error = 1;
            18: clock_error = 1;
        endcase
        @(posedge clk);
        // Clear the error
        case (error_bit)
            0: l1_cache_single_error[0] = 0;
            1: l1_cache_double_error[0] = 0;
            2: l2_cache_single_error = 0;
            3: l2_cache_double_error = 0;
            4: l3_cache_single_error = 0;
            5: l3_cache_double_error = 0;
            6: memory_single_error = 0;
            7: memory_double_error = 0;
            8: core_arithmetic_error[0] = 0;
            9: core_pipeline_error[0] = 0;
            10: tpu_compute_error[0] = 0;
            11: tpu_overflow_error[0] = 0;
            12: vpu_compute_error[0] = 0;
            13: vpu_overflow_error[0] = 0;
            14: noc_deadlock_error = 0;
            15: noc_timeout_error = 0;
            16: power_domain_error = 0;
            17: thermal_error = 0;
            18: clock_error = 0;
        endcase
    endtask

    // Test scenarios
    initial begin
        $display("Starting Error Detector Test");
        
        // Initialize
        error_count = 0;
        test_count = 0;
        reset_system();

        // Test 1: Basic error detection
        $display("Test 1: Basic error detection");
        set_error_mask(32'h00000000); // Unmask all errors
        inject_single_error(0); // L1 cache single error
        
        if (error_status[0] && error_interrupt) begin
            $display("PASS: Basic error detection");
        end else begin
            $display("FAIL: Basic error detection - Status: %h, Interrupt: %b", error_status, error_interrupt);
            error_count++;
        end
        test_count++;
        clear_errors();

        // Test 2: Error masking
        $display("Test 2: Error masking");
        set_error_mask(32'hFFFFFFFF); // Mask all errors
        inject_single_error(2); // L2 cache single error
        
        if (error_status[2] && !error_interrupt) begin
            $display("PASS: Error masking");
        end else begin
            $display("FAIL: Error masking - Status: %h, Interrupt: %b", error_status, error_interrupt);
            error_count++;
        end
        test_count++;
        clear_errors();

        // Test 3: Severity classification
        $display("Test 3: Severity classification");
        set_error_mask(32'h00000000); // Unmask all errors
        
        // Test minor error (single-bit ECC)
        inject_single_error(0);
        @(posedge clk);
        if (error_severity[2:0] == 3'b010) begin // SEV_MINOR
            $display("PASS: Minor error severity");
        end else begin
            $display("FAIL: Minor error severity - Expected: 010, Got: %b", error_severity[2:0]);
            error_count++;
        end
        clear_errors();
        
        // Test fatal error (double-bit ECC)
        inject_single_error(1);
        @(posedge clk);
        if (error_severity[2:0] == 3'b101) begin // SEV_FATAL
            $display("PASS: Fatal error severity");
        end else begin
            $display("FAIL: Fatal error severity - Expected: 101, Got: %b", error_severity[2:0]);
            error_count++;
        end
        test_count++;
        clear_errors();

        // Test 4: Error logging
        $display("Test 4: Error logging");
        inject_single_error(8); // Core arithmetic error
        @(posedge clk);
        
        if (error_log_valid && error_log_data[22:18] == 5'd8) begin
            $display("PASS: Error logging");
        end else begin
            $display("FAIL: Error logging - Valid: %b, Error Type: %d", error_log_valid, error_log_data[22:18]);
            error_count++;
        end
        test_count++;
        clear_errors();

        // Test 5: Multiple simultaneous errors
        $display("Test 5: Multiple simultaneous errors");
        set_error_mask(32'h00000000); // Unmask all errors
        
        // Inject multiple errors simultaneously
        l1_cache_single_error[0] = 1;
        l2_cache_single_error = 1;
        core_arithmetic_error[0] = 1;
        @(posedge clk);
        
        if (error_status[0] && error_status[2] && error_status[8] && error_interrupt) begin
            $display("PASS: Multiple simultaneous errors");
        end else begin
            $display("FAIL: Multiple simultaneous errors - Status: %h", error_status);
            error_count++;
        end
        
        // Clear error inputs
        l1_cache_single_error[0] = 0;
        l2_cache_single_error = 0;
        core_arithmetic_error[0] = 0;
        test_count++;
        clear_errors();

        // Test 6: Error injection functionality
        $display("Test 6: Error injection functionality");
        error_inject_enable = 1;
        error_inject_type = 5'd10; // TPU compute error
        @(posedge clk);
        error_inject_enable = 0;
        @(posedge clk);
        
        if (error_status[10]) begin
            $display("PASS: Error injection functionality");
        end else begin
            $display("FAIL: Error injection functionality - Status: %h", error_status);
            error_count++;
        end
        test_count++;
        clear_errors();

        // Test 7: Priority handling
        $display("Test 7: Priority handling");
        set_error_mask(32'h00000000); // Unmask all errors
        
        // Inject both minor and fatal errors
        l1_cache_single_error[0] = 1; // Minor
        l1_cache_double_error[0] = 1; // Fatal
        @(posedge clk);
        
        if (error_severity[2:0] == 3'b101) begin // Should report fatal
            $display("PASS: Priority handling");
        end else begin
            $display("FAIL: Priority handling - Expected: 101, Got: %b", error_severity[2:0]);
            error_count++;
        end
        
        l1_cache_single_error[0] = 0;
        l1_cache_double_error[0] = 0;
        test_count++;
        clear_errors();

        // Test summary
        $display("\n=== Error Detector Test Summary ===");
        $display("Total tests: %d", test_count);
        $display("Failed tests: %d", error_count);
        $display("Success rate: %.1f%%", (test_count - error_count) * 100.0 / test_count);
        
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end

        $finish;
    end

    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_error_detector.vcd");
        $dumpvars(0, test_error_detector);
    end

endmodule