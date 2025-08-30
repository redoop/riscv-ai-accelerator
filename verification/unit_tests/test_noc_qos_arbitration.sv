// Test for NoC QoS Arbitration and Fairness
// Comprehensive test for QoS-aware arbitration mechanisms

`timescale 1ns/1ps
`include "noc_packet.sv"

module test_noc_qos_arbitration;

    // Test parameters
    parameter int CLK_PERIOD = 10;
    parameter int TEST_DURATION = 10000;
    parameter int NUM_REQUESTS = 8;
    
    // DUT signals
    logic clk;
    logic rst_n;
    
    // QoS Arbiter signals
    logic [NUM_REQUESTS-1:0] request;
    qos_level_t qos_level[NUM_REQUESTS];
    logic [7:0] aging_threshold;
    logic fairness_enable;
    logic [NUM_REQUESTS-1:0] grant;
    logic [NUM_REQUESTS-1:0] starved_requests;
    logic [31:0] total_grants[4];
    logic [7:0] max_wait_time;
    
    // Test control
    logic [31:0] test_cycle;
    logic [31:0] grants_received[NUM_REQUESTS];
    logic [31:0] qos_grants[4];
    logic test_passed;
    
    // Task variables
    logic [31:0] min_grants, max_grants;
    logic low_priority_served;
    logic [31:0] total_grants_before, total_grants_after;
    logic [31:0] cycles_tested;
    logic [31:0] grants_in_period;
    logic [7:0] utilization;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    qos_arbiter #(
        .WIDTH(NUM_REQUESTS),
        .QOS_LEVELS(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .request(request),
        .qos_level(qos_level),
        .aging_threshold(aging_threshold),
        .fairness_enable(fairness_enable),
        .grant(grant),
        .starved_requests(starved_requests),
        .total_grants(total_grants),
        .max_wait_time(max_wait_time)
    );
    
    // Test stimulus and monitoring
    initial begin
        // Initialize
        rst_n = 0;
        request = 0;
        aging_threshold = 8'd16;
        fairness_enable = 1'b1;
        test_cycle = 0;
        
        for (int i = 0; i < NUM_REQUESTS; i++) begin
            qos_level[i] = QOS_NORMAL;
            grants_received[i] = 0;
        end
        
        for (int q = 0; q < 4; q++) begin
            qos_grants[q] = 0;
        end
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        $display("Starting NoC QoS Arbitration Test");
        
        // Test 1: Basic QoS Priority Test
        $display("Test 1: Basic QoS Priority");
        test_qos_priority();
        
        // Test 2: Fairness Test
        $display("Test 2: Fairness Mechanism");
        test_fairness();
        
        // Test 3: Aging and Starvation Prevention
        $display("Test 3: Aging and Starvation Prevention");
        test_aging_mechanism();
        
        // Test 4: Mixed QoS Load Test
        $display("Test 4: Mixed QoS Load");
        test_mixed_qos_load();
        
        // Test 5: Performance Under Load
        $display("Test 5: Performance Under Load");
        test_performance_under_load();
        
        // Evaluate results
        evaluate_test_results();
        
        $display("NoC QoS Arbitration Test Completed");
        $finish;
    end
    
    // Test 1: Basic QoS Priority
    task test_qos_priority();
        begin
            $display("  Testing QoS priority ordering...");
            
            // Set up requests with different QoS levels
            qos_level[0] = QOS_LOW;
            qos_level[1] = QOS_NORMAL;
            qos_level[2] = QOS_HIGH;
            qos_level[3] = QOS_URGENT;
            qos_level[4] = QOS_LOW;
            qos_level[5] = QOS_NORMAL;
            qos_level[6] = QOS_HIGH;
            qos_level[7] = QOS_URGENT;
            
            fairness_enable = 1'b0;  // Disable fairness for pure priority test
            
            // Apply all requests simultaneously
            request = 8'hFF;
            
            // Monitor grants for several cycles
            for (int cycle = 0; cycle < 100; cycle++) begin
                @(posedge clk);
                
                // Check that highest priority gets served first
                if (|grant) begin
                    logic urgent_granted, high_granted, normal_granted, low_granted;
                    urgent_granted = grant[3] || grant[7];
                    high_granted = grant[2] || grant[6];
                    normal_granted = grant[1] || grant[5];
                    low_granted = grant[0] || grant[4];
                    
                    // URGENT should be granted before others
                    if (urgent_granted) begin
                        if (high_granted || normal_granted || low_granted) begin
                            $error("Priority violation: Lower priority granted with URGENT");
                        end
                    end
                    // HIGH should be granted before NORMAL and LOW
                    else if (high_granted) begin
                        if (normal_granted || low_granted) begin
                            $error("Priority violation: Lower priority granted with HIGH");
                        end
                    end
                    // NORMAL should be granted before LOW
                    else if (normal_granted) begin
                        if (low_granted) begin
                            $error("Priority violation: LOW granted with NORMAL");
                        end
                    end
                    
                    // Update grant counters
                    for (int i = 0; i < NUM_REQUESTS; i++) begin
                        if (grant[i]) grants_received[i]++;
                    end
                end
            end
            
            request = 0;
            repeat(10) @(posedge clk);
            $display("  QoS priority test completed");
        end
    endtask
    
    // Test 2: Fairness Mechanism
    task test_fairness();
        begin
            $display("  Testing fairness mechanism...");
            
            // Set all requests to same QoS level
            for (int i = 0; i < NUM_REQUESTS; i++) begin
                qos_level[i] = QOS_NORMAL;
                grants_received[i] = 0;
            end
            
            fairness_enable = 1'b1;
            request = 8'hFF;  // All requesting
            
            // Run for many cycles to check fairness
            for (int cycle = 0; cycle < 500; cycle++) begin
                @(posedge clk);
                
                if (|grant) begin
                    for (int i = 0; i < NUM_REQUESTS; i++) begin
                        if (grant[i]) grants_received[i]++;
                    end
                end
            end
            
            // Check fairness - all should have similar grant counts
            min_grants = grants_received[0];
            max_grants = grants_received[0];
            
            for (int i = 1; i < NUM_REQUESTS; i++) begin
                if (grants_received[i] < min_grants) min_grants = grants_received[i];
                if (grants_received[i] > max_grants) max_grants = grants_received[i];
            end
            
            // Fairness check: max should not be more than 2x min
            if (max_grants > (min_grants * 2)) begin
                $error("Fairness violation: max_grants=%0d, min_grants=%0d", max_grants, min_grants);
            end else begin
                $display("  Fairness check passed: max_grants=%0d, min_grants=%0d", max_grants, min_grants);
            end
            
            request = 0;
            repeat(10) @(posedge clk);
        end
    endtask
    
    // Test 3: Aging Mechanism
    task test_aging_mechanism();
        begin
            $display("  Testing aging and starvation prevention...");
            
            // Set up scenario where one request might be starved
            qos_level[0] = QOS_LOW;     // Potential victim
            qos_level[1] = QOS_HIGH;    // Higher priority
            qos_level[2] = QOS_HIGH;
            qos_level[3] = QOS_HIGH;
            
            for (int i = 4; i < NUM_REQUESTS; i++) begin
                qos_level[i] = QOS_NORMAL;
            end
            
            fairness_enable = 1'b1;
            aging_threshold = 8'd20;  // Age after 20 cycles
            
            // Start with all requesting
            request = 8'hFF;
            
            low_priority_served = 1'b0;
            
            // Monitor for aging effect
            for (int cycle = 0; cycle < 200; cycle++) begin
                @(posedge clk);
                
                if (grant[0]) begin  // Low priority request served
                    low_priority_served = 1'b1;
                    $display("  Low priority request served at cycle %0d", cycle);
                end
                
                // Check starvation detection
                if (starved_requests[0]) begin
                    $display("  Starvation detected for request 0 at cycle %0d", cycle);
                end
            end
            
            if (!low_priority_served) begin
                $error("Aging mechanism failed: Low priority request never served");
            end else begin
                $display("  Aging mechanism working: Low priority eventually served");
            end
            
            request = 0;
            repeat(10) @(posedge clk);
        end
    endtask
    
    // Test 4: Mixed QoS Load
    task test_mixed_qos_load();
        begin
            $display("  Testing mixed QoS load scenario...");
            
            // Reset counters
            for (int i = 0; i < NUM_REQUESTS; i++) begin
                grants_received[i] = 0;
            end
            for (int q = 0; q < 4; q++) begin
                qos_grants[q] = 0;
            end
            
            // Set up mixed QoS levels
            qos_level[0] = QOS_LOW;
            qos_level[1] = QOS_LOW;
            qos_level[2] = QOS_NORMAL;
            qos_level[3] = QOS_NORMAL;
            qos_level[4] = QOS_NORMAL;
            qos_level[5] = QOS_HIGH;
            qos_level[6] = QOS_HIGH;
            qos_level[7] = QOS_URGENT;
            
            fairness_enable = 1'b1;
            
            // Variable load pattern
            for (int cycle = 0; cycle < 1000; cycle++) begin
                // Vary request pattern
                case (cycle % 8)
                    0: request = 8'b11111111;  // All requesting
                    1: request = 8'b11110000;  // High priority only
                    2: request = 8'b00001111;  // Low priority only
                    3: request = 8'b10101010;  // Alternating
                    4: request = 8'b11000011;  // Mixed
                    5: request = 8'b01010101;  // Alternating
                    6: request = 8'b11111000;  // Mostly high
                    7: request = 8'b00011111;  // Mostly low
                endcase
                
                @(posedge clk);
                
                // Count grants by QoS level
                for (int i = 0; i < NUM_REQUESTS; i++) begin
                    if (grant[i]) begin
                        grants_received[i]++;
                        case (qos_level[i])
                            QOS_LOW:    qos_grants[0]++;
                            QOS_NORMAL: qos_grants[1]++;
                            QOS_HIGH:   qos_grants[2]++;
                            QOS_URGENT: qos_grants[3]++;
                        endcase
                    end
                end
            end
            
            // Analyze QoS distribution
            $display("  QoS Grant Distribution:");
            $display("    LOW: %0d, NORMAL: %0d, HIGH: %0d, URGENT: %0d", 
                     qos_grants[0], qos_grants[1], qos_grants[2], qos_grants[3]);
            
            // Verify QoS ordering (higher should get more service)
            if (qos_grants[3] < qos_grants[2] || qos_grants[2] < qos_grants[1]) begin
                $warning("QoS ordering not strictly maintained under mixed load");
            end
            
            request = 0;
            repeat(10) @(posedge clk);
        end
    endtask
    
    // Test 5: Performance Under Load
    task test_performance_under_load();
        begin
            $display("  Testing performance under high load...");
            
            cycles_tested = 1000;
            
            // Measure baseline
            total_grants_before = 0;
            for (int q = 0; q < 4; q++) begin
                total_grants_before += total_grants[q];
            end
            
            // High load test
            request = 8'hFF;  // All requesting continuously
            fairness_enable = 1'b1;
            
            for (int cycle = 0; cycle < cycles_tested; cycle++) begin
                @(posedge clk);
            end
            
            // Measure final
            total_grants_after = 0;
            for (int q = 0; q < 4; q++) begin
                total_grants_after += total_grants[q];
            end
            
            grants_in_period = total_grants_after - total_grants_before;
            utilization = 8'((grants_in_period * 100) / cycles_tested);
            
            $display("  Utilization: %0d%% (%0d grants in %0d cycles)", 
                     utilization, grants_in_period, cycles_tested);
            
            if (utilization < 80) begin
                $warning("Low utilization under high load: %0d%%", utilization);
            end
            
            request = 0;
            repeat(10) @(posedge clk);
        end
    endtask
    
    // Evaluate overall test results
    task evaluate_test_results();
        begin
            test_passed = 1'b1;
            
            $display("\n=== Test Results Summary ===");
            
            // Check for any starvation
            if (|starved_requests) begin
                $display("FAIL: Starvation detected");
                test_passed = 1'b0;
            end else begin
                $display("PASS: No starvation detected");
            end
            
            // Check maximum wait time
            if (max_wait_time > 50) begin
                $display("WARN: High maximum wait time: %0d cycles", max_wait_time);
            end else begin
                $display("PASS: Reasonable maximum wait time: %0d cycles", max_wait_time);
            end
            
            // Overall result
            if (test_passed) begin
                $display("OVERALL: QoS Arbitration Test PASSED");
            end else begin
                $display("OVERALL: QoS Arbitration Test FAILED");
            end
        end
    endtask
    
    // Monitor and display key metrics
    always @(posedge clk) begin
        test_cycle <= test_cycle + 1;
        
        // Periodic status display
        if (test_cycle % 1000 == 0 && test_cycle > 0) begin
            $display("  Cycle %0d: Max wait time: %0d, Starved: %b", 
                     test_cycle, max_wait_time, starved_requests);
        end
    end

endmodule