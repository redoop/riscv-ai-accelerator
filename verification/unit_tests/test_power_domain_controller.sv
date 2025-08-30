/*
 * Power Domain Controller Test
 * 
 * This testbench verifies the power domain controller functionality,
 * including power gating sequences and isolation control.
 */

`timescale 1ns/1ps

module test_power_domain_controller;

    // Parameters
    localparam NUM_CORES = 4;
    localparam NUM_AI_UNITS = 2;
    localparam CLK_PERIOD = 10;

    // Signals
    logic                    clk;
    logic                    rst_n;
    logic [NUM_CORES-1:0]   core_power_gate_req;
    logic                    memory_power_gate_req;
    logic                    ai_accel_power_gate_req;
    logic [NUM_CORES-1:0]   core_activity;
    logic                    memory_activity;
    logic [NUM_AI_UNITS-1:0] ai_unit_activity;
    logic [NUM_CORES-1:0]   core_power_enable;
    logic                    l1_cache_power_enable;
    logic                    l2_cache_power_enable;
    logic                    memory_ctrl_power_enable;
    logic [NUM_AI_UNITS-1:0] ai_unit_power_enable;
    logic                    noc_power_enable;
    logic [NUM_CORES-1:0]   core_isolation_enable;
    logic                    memory_isolation_enable;
    logic [NUM_AI_UNITS-1:0] ai_unit_isolation_enable;
    logic [7:0]              power_domain_status;
    logic                    power_transition_busy;

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
    power_domain_controller #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .core_power_gate_req(core_power_gate_req),
        .memory_power_gate_req(memory_power_gate_req),
        .ai_accel_power_gate_req(ai_accel_power_gate_req),
        .core_activity(core_activity),
        .memory_activity(memory_activity),
        .ai_unit_activity(ai_unit_activity),
        .core_power_enable(core_power_enable),
        .l1_cache_power_enable(l1_cache_power_enable),
        .l2_cache_power_enable(l2_cache_power_enable),
        .memory_ctrl_power_enable(memory_ctrl_power_enable),
        .ai_unit_power_enable(ai_unit_power_enable),
        .noc_power_enable(noc_power_enable),
        .core_isolation_enable(core_isolation_enable),
        .memory_isolation_enable(memory_isolation_enable),
        .ai_unit_isolation_enable(ai_unit_isolation_enable),
        .power_domain_status(power_domain_status),
        .power_transition_busy(power_transition_busy)
    );

    // Test tasks
    task reset_system();
        rst_n = 0;
        core_power_gate_req = '0;
        memory_power_gate_req = 0;
        ai_accel_power_gate_req = 0;
        core_activity = '1;
        memory_activity = 1;
        ai_unit_activity = '1;
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
    endtask

    task check_result(string test_name, logic condition);
        test_count++;
        if (condition) begin
            $display("PASS: %s", test_name);
            pass_count++;
        end else begin
            $display("FAIL: %s", test_name);
            fail_count++;
        end
    endtask

    task wait_for_idle_timeout(int core_id);
        // Wait for idle counter to reach threshold (1000 cycles)
        repeat(1100) @(posedge clk);
    endtask

    task wait_for_power_transition();
        while (power_transition_busy) begin
            @(posedge clk);
        end
        repeat(5) @(posedge clk);
    endtask

    // Main test sequence
    initial begin
        $display("Starting Power Domain Controller Test");
        
        // Test 1: Reset and initialization
        reset_system();
        check_result("Reset - All cores powered", &core_power_enable);
        check_result("Reset - All caches powered", 
                    l1_cache_power_enable && l2_cache_power_enable && memory_ctrl_power_enable);
        check_result("Reset - All AI units powered", &ai_unit_power_enable);
        check_result("Reset - NoC always powered", noc_power_enable);
        check_result("Reset - No isolation", 
                    core_isolation_enable == '0 && !memory_isolation_enable && ai_unit_isolation_enable == '0);
        
        // Test 2: Core power gating sequence
        $display("\n--- Test 2: Core Power Gating ---");
        core_activity[0] = 0;  // Make core 0 idle
        core_power_gate_req[0] = 1;  // Request power gating for core 0
        
        wait_for_idle_timeout(0);
        
        // Check isolation happens first
        check_result("Core 0 - Isolation enabled during gating", core_isolation_enable[0]);
        
        // Wait for complete power gating sequence
        wait_for_power_transition();
        
        check_result("Core 0 - Power gated after sequence", !core_power_enable[0]);
        check_result("Core 0 - Other cores still powered", 
                    core_power_enable[1] && core_power_enable[2] && core_power_enable[3]);
        
        // Test 3: Core power restore
        $display("\n--- Test 3: Core Power Restore ---");
        core_activity[0] = 1;  // Core 0 becomes active again
        
        wait_for_power_transition();
        
        check_result("Core 0 - Power restored", core_power_enable[0]);
        check_result("Core 0 - Isolation removed", !core_isolation_enable[0]);
        
        // Test 4: Memory power gating (L2 cache only)
        $display("\n--- Test 4: Memory Power Gating ---");
        memory_activity = 0;
        memory_power_gate_req = 1;
        
        wait_for_idle_timeout(-1);  // Wait for memory idle timeout
        
        check_result("Memory - Isolation enabled", memory_isolation_enable);
        
        wait_for_power_transition();
        
        check_result("Memory - L2 cache gated", !l2_cache_power_enable);
        check_result("Memory - L1 cache still powered", l1_cache_power_enable);
        check_result("Memory - Memory controller still powered", memory_ctrl_power_enable);
        
        // Test 5: Memory power restore
        $display("\n--- Test 5: Memory Power Restore ---");
        memory_activity = 1;
        
        wait_for_power_transition();
        
        check_result("Memory - L2 cache restored", l2_cache_power_enable);
        check_result("Memory - Isolation removed", !memory_isolation_enable);
        
        // Test 6: AI unit power gating
        $display("\n--- Test 6: AI Unit Power Gating ---");
        ai_unit_activity[0] = 0;  // Make AI unit 0 idle
        ai_accel_power_gate_req = 1;
        
        wait_for_idle_timeout(-1);
        
        check_result("AI Unit 0 - Isolation enabled", ai_unit_isolation_enable[0]);
        
        wait_for_power_transition();
        
        check_result("AI Unit 0 - Power gated", !ai_unit_power_enable[0]);
        check_result("AI Unit 1 - Still powered", ai_unit_power_enable[1]);
        
        // Test 7: Multiple core power gating
        $display("\n--- Test 7: Multiple Core Power Gating ---");
        core_activity[1] = 0;
        core_activity[2] = 0;
        core_power_gate_req[1] = 1;
        core_power_gate_req[2] = 1;
        
        wait_for_idle_timeout(-1);
        wait_for_power_transition();
        
        check_result("Multiple cores - Cores 1&2 gated", 
                    !core_power_enable[1] && !core_power_enable[2]);
        check_result("Multiple cores - Cores 0&3 still powered", 
                    core_power_enable[0] && core_power_enable[3]);
        
        // Test 8: Power gating cancellation by activity
        $display("\n--- Test 8: Power Gating Cancellation ---");
        core_activity[3] = 0;
        core_power_gate_req[3] = 1;
        
        // Wait partway through idle timeout, then make core active again
        repeat(500) @(posedge clk);
        core_activity[3] = 1;
        
        repeat(700) @(posedge clk);  // Complete the original timeout period
        
        check_result("Gating cancellation - Core 3 still powered", core_power_enable[3]);
        check_result("Gating cancellation - Core 3 not isolated", !core_isolation_enable[3]);
        
        // Test 9: Power domain status reporting
        $display("\n--- Test 9: Power Domain Status ---");
        logic expected_status;
        expected_status = {
            |core_power_enable,           // Any core powered
            l2_cache_power_enable,        // L2 cache powered  
            memory_ctrl_power_enable,     // Memory controller powered
            |ai_unit_power_enable,        // Any AI unit powered
            noc_power_enable,             // NoC powered
            1'b0,                         // Memory not in transition
            |core_isolation_enable,       // Any core isolated
            |ai_unit_isolation_enable     // Any AI unit isolated
        };
        
        check_result("Status reporting - Correct status bits", power_domain_status == expected_status);
        
        // Test 10: Restore all power domains
        $display("\n--- Test 10: Full System Restore ---");
        core_power_gate_req = '0;
        memory_power_gate_req = 0;
        ai_accel_power_gate_req = 0;
        core_activity = '1;
        memory_activity = 1;
        ai_unit_activity = '1;
        
        wait_for_power_transition();
        
        check_result("Full restore - All cores powered", &core_power_enable);
        check_result("Full restore - All memory powered", 
                    l1_cache_power_enable && l2_cache_power_enable && memory_ctrl_power_enable);
        check_result("Full restore - All AI units powered", &ai_unit_power_enable);
        check_result("Full restore - No isolation", 
                    core_isolation_enable == '0 && !memory_isolation_enable && ai_unit_isolation_enable == '0);
        
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

    // Timeout watchdog
    initial begin
        #2000000;  // 2ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("test_power_domain_controller.vcd");
        $dumpvars(0, test_power_domain_controller);
    end

endmodule