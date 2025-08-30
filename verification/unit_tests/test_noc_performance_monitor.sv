// Test for NoC Performance and Fairness Monitor
// Comprehensive test for network performance monitoring and analysis

`timescale 1ns/1ps
`include "noc_packet.sv"

module test_noc_performance_monitor;

    // Test parameters
    parameter int CLK_PERIOD = 10;
    parameter int MESH_SIZE_X = 4;
    parameter int MESH_SIZE_Y = 4;
    parameter int NUM_ROUTERS = MESH_SIZE_X * MESH_SIZE_Y;
    parameter int MONITOR_WINDOW = 256;  // Smaller window for testing
    
    // DUT signals
    logic clk;
    logic rst_n;
    
    // Router performance inputs
    logic [31:0] packets_routed[NUM_ROUTERS];
    logic [31:0] buffer_occupancy[NUM_ROUTERS][5];
    logic [31:0] qos_grants[NUM_ROUTERS][4];
    logic [7:0]  allocation_efficiency[NUM_ROUTERS];
    logic        fairness_violation[NUM_ROUTERS];
    logic        congestion_detected[NUM_ROUTERS];
    
    // Network-wide metrics
    logic [31:0] total_throughput;
    logic [15:0] average_latency;
    logic [7:0]  network_efficiency;
    logic [7:0]  qos_compliance_score;
    logic [7:0]  fairness_index;
    
    // Per-QoS metrics
    logic [31:0] qos_throughput[4];
    logic [15:0] qos_avg_latency[4];
    logic [7:0]  qos_violation_rate[4];
    
    // Hotspot and congestion analysis
    logic [15:0] congestion_map;
    logic [3:0]  worst_congested_router;
    logic [7:0]  congestion_severity;
    
    // Fairness metrics
    logic [7:0]  jain_fairness_index;
    logic [15:0] starved_flows;
    logic [31:0] max_packet_delay;
    
    // Performance alerts
    logic        performance_degradation;
    logic        fairness_alert;
    logic        congestion_alert;
    
    // Test control
    logic [31:0] test_cycle;
    logic test_passed;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    noc_performance_monitor #(
        .MESH_SIZE_X(MESH_SIZE_X),
        .MESH_SIZE_Y(MESH_SIZE_Y),
        .NUM_ROUTERS(NUM_ROUTERS),
        .MONITOR_WINDOW(MONITOR_WINDOW)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .packets_routed(packets_routed),
        .buffer_occupancy(buffer_occupancy),
        .qos_grants(qos_grants),
        .allocation_efficiency(allocation_efficiency),
        .fairness_violation(fairness_violation),
        .congestion_detected(congestion_detected),
        .total_throughput(total_throughput),
        .average_latency(average_latency),
        .network_efficiency(network_efficiency),
        .qos_compliance_score(qos_compliance_score),
        .fairness_index(fairness_index),
        .qos_throughput(qos_throughput),
        .qos_avg_latency(qos_avg_latency),
        .qos_violation_rate(qos_violation_rate),
        .congestion_map(congestion_map),
        .worst_congested_router(worst_congested_router),
        .congestion_severity(congestion_severity),
        .jain_fairness_index(jain_fairness_index),
        .starved_flows(starved_flows),
        .max_packet_delay(max_packet_delay),
        .performance_degradation(performance_degradation),
        .fairness_alert(fairness_alert),
        .congestion_alert(congestion_alert)
    );
    
    // Test stimulus and monitoring
    initial begin
        // Initialize
        rst_n = 0;
        test_cycle = 0;
        
        // Initialize all inputs
        initialize_inputs();
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        $display("Starting NoC Performance Monitor Test");
        
        // Test 1: Baseline Performance Monitoring
        $display("Test 1: Baseline Performance Monitoring");
        test_baseline_monitoring();
        
        // Test 2: Throughput Measurement
        $display("Test 2: Throughput Measurement");
        test_throughput_measurement();
        
        // Test 3: QoS Performance Tracking
        $display("Test 3: QoS Performance Tracking");
        test_qos_performance();
        
        // Test 4: Fairness Analysis
        $display("Test 4: Fairness Analysis");
        test_fairness_analysis();
        
        // Test 5: Congestion Detection and Analysis
        $display("Test 5: Congestion Detection and Analysis");
        test_congestion_analysis();
        
        // Test 6: Performance Alert Generation
        $display("Test 6: Performance Alert Generation");
        test_performance_alerts();
        
        // Test 7: Long-term Monitoring
        $display("Test 7: Long-term Monitoring");
        test_longterm_monitoring();
        
        // Evaluate results
        evaluate_test_results();
        
        $display("NoC Performance Monitor Test Completed");
        $finish;
    end
    
    // Initialize all inputs to default values
    task initialize_inputs();
        begin
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                packets_routed[router] = 0;
                allocation_efficiency[router] = 8'd100;
                fairness_violation[router] = 1'b0;
                congestion_detected[router] = 1'b0;
                
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = 0;
                end
                
                for (int qos = 0; qos < 4; qos++) begin
                    qos_grants[router][qos] = 0;
                end
            end
        end
    endtask
    
    // Test 1: Baseline Performance Monitoring
    task test_baseline_monitoring();
        begin
            $display("  Testing baseline performance monitoring...");
            
            // Set minimal activity
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                allocation_efficiency[router] = 8'd95;
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = 1;
                end
            end
            
            // Run for monitoring window
            repeat(MONITOR_WINDOW + 50) @(posedge clk);
            
            // Check baseline metrics
            if (network_efficiency < 90) begin
                $error("Low network efficiency in baseline: %0d%%", network_efficiency);
            end
            
            if (performance_degradation) begin
                $error("Performance degradation detected in baseline");
            end
            
            if (congestion_alert) begin
                $error("Congestion alert in baseline conditions");
            end
            
            $display("  Baseline metrics - Efficiency: %0d%%, Latency: %0d", 
                     network_efficiency, average_latency);
        end
    endtask
    
    // Test 2: Throughput Measurement
    task test_throughput_measurement();
        begin
            $display("  Testing throughput measurement...");
            
            // Simulate packet routing activity
            logic [31:0] expected_throughput = 0;
            
            for (int cycle = 0; cycle < MONITOR_WINDOW; cycle++) begin
                // Increment packet counters for some routers
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    if (router % 2 == 0) begin  // Even routers active
                        packets_routed[router] = packets_routed[router] + 2;
                        expected_throughput += 2;
                    end
                end
                @(posedge clk);
            end
            
            // Wait for measurement window to complete
            repeat(50) @(posedge clk);
            
            // Check throughput measurement
            if (total_throughput == 0) begin
                $error("No throughput measured despite packet activity");
            end else begin
                $display("  Measured throughput: %0d packets", total_throughput);
            end
            
            // Check individual QoS throughput
            logic [31:0] qos_sum = qos_throughput[0] + qos_throughput[1] + 
                                  qos_throughput[2] + qos_throughput[3];
            
            if (qos_sum == 0) begin
                $warning("No QoS-specific throughput measured");
            end
        end
    endtask
    
    // Test 3: QoS Performance Tracking
    task test_qos_performance();
        begin
            $display("  Testing QoS performance tracking...");
            
            // Reset packet counters
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                packets_routed[router] = 0;
                for (int qos = 0; qos < 4; qos++) begin
                    qos_grants[router][qos] = 0;
                end
            end
            
            // Simulate QoS-specific activity
            for (int cycle = 0; cycle < MONITOR_WINDOW; cycle++) begin
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    // Different QoS levels get different service rates
                    qos_grants[router][0] += 1;  // QOS_LOW
                    qos_grants[router][1] += 3;  // QOS_NORMAL
                    qos_grants[router][2] += 4;  // QOS_HIGH
                    qos_grants[router][3] += 2;  // QOS_URGENT
                end
                @(posedge clk);
            end
            
            // Wait for measurement
            repeat(50) @(posedge clk);
            
            // Check QoS distribution
            $display("  QoS Throughput - LOW: %0d, NORMAL: %0d, HIGH: %0d, URGENT: %0d",
                     qos_throughput[0], qos_throughput[1], qos_throughput[2], qos_throughput[3]);
            
            // Verify QoS ordering (HIGH should have highest throughput in this test)
            if (qos_throughput[2] <= qos_throughput[1] || qos_throughput[2] <= qos_throughput[0]) begin
                $warning("QoS throughput ordering not as expected");
            end
            
            // Check QoS compliance score
            if (qos_compliance_score < 80) begin
                $warning("Low QoS compliance score: %0d%%", qos_compliance_score);
            end
        end
    endtask
    
    // Test 4: Fairness Analysis
    task test_fairness_analysis();
        begin
            $display("  Testing fairness analysis...");
            
            // Reset for fairness test
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                packets_routed[router] = 0;
                fairness_violation[router] = 1'b0;
            end
            
            // Create unfair distribution
            for (int cycle = 0; cycle < MONITOR_WINDOW; cycle++) begin
                // Some routers get much more traffic
                packets_routed[0] += 10;  // High traffic
                packets_routed[1] += 10;  // High traffic
                packets_routed[2] += 1;   // Low traffic
                packets_routed[3] += 1;   // Low traffic
                
                // Rest get moderate traffic
                for (int router = 4; router < NUM_ROUTERS; router++) begin
                    packets_routed[router] += 3;
                end
                
                @(posedge clk);
            end
            
            // Wait for analysis
            repeat(50) @(posedge clk);
            
            // Check fairness metrics
            $display("  Fairness Index: %0d%%, Jain's Index: %0d%%", 
                     fairness_index, jain_fairness_index);
            
            if (fairness_index > 90) begin
                $warning("High fairness index despite unfair distribution");
            end
            
            // Test with fair distribution
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                packets_routed[router] = 0;
            end
            
            for (int cycle = 0; cycle < MONITOR_WINDOW; cycle++) begin
                // Equal traffic for all routers
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    packets_routed[router] += 5;
                end
                @(posedge clk);
            end
            
            repeat(50) @(posedge clk);
            
            $display("  Fair distribution - Fairness Index: %0d%%, Jain's Index: %0d%%", 
                     fairness_index, jain_fairness_index);
            
            if (jain_fairness_index < 95) begin
                $warning("Low Jain's fairness index with equal distribution");
            end
        end
    endtask
    
    // Test 5: Congestion Detection and Analysis
    task test_congestion_analysis();
        begin
            $display("  Testing congestion detection and analysis...");
            
            // Clear congestion state
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                congestion_detected[router] = 1'b0;
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = 2;
                end
            end
            
            repeat(50) @(posedge clk);
            
            // Check no congestion state
            if (congestion_severity > 10) begin
                $error("High congestion severity without congestion");
            end
            
            // Create congestion at specific routers
            congestion_detected[5] = 1'b1;
            congestion_detected[10] = 1'b1;
            
            for (int port = 0; port < 5; port++) begin
                buffer_occupancy[5][port] = 30;   // High occupancy
                buffer_occupancy[10][port] = 25;  // High occupancy
            end
            
            repeat(100) @(posedge clk);
            
            // Check congestion detection
            if (congestion_map[5] != 1'b1 || congestion_map[10] != 1'b1) begin
                $error("Congestion map not reflecting actual congestion");
            end
            
            if (congestion_severity == 0) begin
                $error("No congestion severity detected");
            end
            
            $display("  Congestion detected - Severity: %0d%%, Map: %b, Worst: %0d", 
                     congestion_severity, congestion_map, worst_congested_router);
            
            // Check worst congested router identification
            if (worst_congested_router != 4'd5 && worst_congested_router != 4'd10) begin
                $warning("Worst congested router not correctly identified");
            end
        end
    endtask
    
    // Test 6: Performance Alert Generation
    task test_performance_alerts();
        begin
            $display("  Testing performance alert generation...");
            
            // Reset to good conditions
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                allocation_efficiency[router] = 8'd95;
                fairness_violation[router] = 1'b0;
                congestion_detected[router] = 1'b0;
            end
            
            repeat(50) @(posedge clk);
            
            // Check no alerts
            if (performance_degradation || fairness_alert || congestion_alert) begin
                $error("Alerts active under good conditions");
            end
            
            // Trigger performance degradation
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                allocation_efficiency[router] = 8'd60;  // Below threshold
            end
            
            repeat(50) @(posedge clk);
            
            if (!performance_degradation) begin
                $error("Performance degradation alert not triggered");
            end else begin
                $display("  Performance degradation alert correctly triggered");
            end
            
            // Trigger fairness alert
            allocation_efficiency[0] = 8'd95;  // Reset performance
            fairness_violation[0] = 1'b1;
            fairness_violation[1] = 1'b1;
            
            repeat(50) @(posedge clk);
            
            if (!fairness_alert) begin
                $error("Fairness alert not triggered");
            end else begin
                $display("  Fairness alert correctly triggered");
            end
            
            // Trigger congestion alert
            fairness_violation[0] = 1'b0;  // Reset fairness
            fairness_violation[1] = 1'b0;
            
            for (int router = 0; router < NUM_ROUTERS/2; router++) begin
                congestion_detected[router] = 1'b1;
            end
            
            repeat(100) @(posedge clk);
            
            if (!congestion_alert) begin
                $error("Congestion alert not triggered");
            end else begin
                $display("  Congestion alert correctly triggered");
            end
        end
    endtask
    
    // Test 7: Long-term Monitoring
    task test_longterm_monitoring();
        begin
            $display("  Testing long-term monitoring behavior...");
            
            // Reset to baseline
            initialize_inputs();
            
            // Run for multiple monitoring windows
            for (int window = 0; window < 3; window++) begin
                $display("    Monitoring window %0d", window);
                
                // Vary activity patterns
                for (int cycle = 0; cycle < MONITOR_WINDOW; cycle++) begin
                    for (int router = 0; router < NUM_ROUTERS; router++) begin
                        // Sinusoidal activity pattern
                        logic [31:0] activity = 5 + (3 * $sin(cycle * router));
                        packets_routed[router] += activity;
                        
                        // Vary buffer occupancy
                        for (int port = 0; port < 5; port++) begin
                            buffer_occupancy[router][port] = 2 + (cycle % 8);
                        end
                    end
                    @(posedge clk);
                end
                
                // Check metrics at end of each window
                $display("    Window %0d metrics - Throughput: %0d, Efficiency: %0d%%, Latency: %0d",
                         window, total_throughput, network_efficiency, average_latency);
            end
            
            $display("  Long-term monitoring test completed");
        end
    endtask
    
    // Evaluate overall test results
    task evaluate_test_results();
        begin
            test_passed = 1'b1;
            
            $display("\n=== Performance Monitor Test Results ===");
            
            // Check final metrics
            $display("Final Network Metrics:");
            $display("  Total Throughput: %0d packets", total_throughput);
            $display("  Network Efficiency: %0d%%", network_efficiency);
            $display("  Average Latency: %0d cycles", average_latency);
            $display("  QoS Compliance: %0d%%", qos_compliance_score);
            $display("  Fairness Index: %0d%%", fairness_index);
            $display("  Congestion Severity: %0d%%", congestion_severity);
            
            // Check alert states
            $display("Alert States:");
            $display("  Performance Degradation: %b", performance_degradation);
            $display("  Fairness Alert: %b", fairness_alert);
            $display("  Congestion Alert: %b", congestion_alert);
            
            // Validate reasonable ranges
            if (network_efficiency > 100) begin
                $error("Invalid network efficiency: %0d%%", network_efficiency);
                test_passed = 1'b0;
            end
            
            if (fairness_index > 100) begin
                $error("Invalid fairness index: %0d%%", fairness_index);
                test_passed = 1'b0;
            end
            
            if (congestion_severity > 100) begin
                $error("Invalid congestion severity: %0d%%", congestion_severity);
                test_passed = 1'b0;
            end
            
            // Overall result
            if (test_passed) begin
                $display("OVERALL: Performance Monitor Test PASSED");
            end else begin
                $display("OVERALL: Performance Monitor Test FAILED");
            end
        end
    endtask
    
    // Monitor and display key metrics periodically
    always @(posedge clk) begin
        test_cycle <= test_cycle + 1;
        
        // Display metrics every 1000 cycles during long tests
        if (test_cycle % 1000 == 0 && test_cycle > 0) begin
            $display("  Cycle %0d: Throughput=%0d, Efficiency=%0d%%, Latency=%0d, Alerts=%b%b%b", 
                     test_cycle, total_throughput, network_efficiency, average_latency,
                     performance_degradation, fairness_alert, congestion_alert);
        end
    end

endmodule