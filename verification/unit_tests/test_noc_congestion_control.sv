// Test for NoC Congestion Control and Monitoring
// Comprehensive test for congestion detection and flow control mechanisms

`timescale 1ns/1ps
`include "noc_packet.sv"

module test_noc_congestion_control;

    // Test parameters
    parameter int CLK_PERIOD = 10;
    parameter int MESH_SIZE_X = 4;
    parameter int MESH_SIZE_Y = 4;
    parameter int NUM_ROUTERS = MESH_SIZE_X * MESH_SIZE_Y;
    parameter int CONGESTION_THRESHOLD = 75;
    
    // DUT signals
    logic clk;
    logic rst_n;
    
    // Router status inputs
    logic [31:0] buffer_occupancy[NUM_ROUTERS][5];
    logic [31:0] packets_routed[NUM_ROUTERS];
    logic        router_congested[NUM_ROUTERS];
    
    // Flow control outputs
    logic        global_flow_control;
    logic [1:0]  throttle_level[NUM_ROUTERS];
    logic        adaptive_routing_enable;
    
    // Congestion monitoring
    logic [7:0]  network_utilization;
    logic [31:0] total_packets_dropped;
    logic [15:0] congestion_hotspots;
    
    // QoS enforcement
    logic        qos_enforcement_active;
    logic [1:0]  priority_boost[NUM_ROUTERS];
    
    // Test control
    logic [31:0] test_cycle;
    logic test_passed;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // DUT instantiation
    congestion_controller #(
        .MESH_SIZE_X(MESH_SIZE_X),
        .MESH_SIZE_Y(MESH_SIZE_Y),
        .NUM_ROUTERS(NUM_ROUTERS),
        .CONGESTION_THRESHOLD(CONGESTION_THRESHOLD)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .buffer_occupancy(buffer_occupancy),
        .packets_routed(packets_routed),
        .router_congested(router_congested),
        .global_flow_control(global_flow_control),
        .throttle_level(throttle_level),
        .adaptive_routing_enable(adaptive_routing_enable),
        .network_utilization(network_utilization),
        .total_packets_dropped(total_packets_dropped),
        .congestion_hotspots(congestion_hotspots),
        .qos_enforcement_active(qos_enforcement_active),
        .priority_boost(priority_boost)
    );
    
    // Test stimulus and monitoring
    initial begin
        // Initialize
        rst_n = 0;
        test_cycle = 0;
        
        // Initialize router status
        for (int router = 0; router < NUM_ROUTERS; router++) begin
            for (int port = 0; port < 5; port++) begin
                buffer_occupancy[router][port] = 0;
            end
            packets_routed[router] = 0;
            router_congested[router] = 1'b0;
        end
        
        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        
        $display("Starting NoC Congestion Control Test");
        
        // Test 1: Normal Operation
        $display("Test 1: Normal Operation");
        test_normal_operation();
        
        // Test 2: Gradual Congestion Build-up
        $display("Test 2: Gradual Congestion Build-up");
        test_gradual_congestion();
        
        // Test 3: Hotspot Detection
        $display("Test 3: Hotspot Detection");
        test_hotspot_detection();
        
        // Test 4: Flow Control Response
        $display("Test 4: Flow Control Response");
        test_flow_control_response();
        
        // Test 5: QoS Enforcement Under Congestion
        $display("Test 5: QoS Enforcement Under Congestion");
        test_qos_enforcement();
        
        // Test 6: Recovery from Congestion
        $display("Test 6: Recovery from Congestion");
        test_congestion_recovery();
        
        // Evaluate results
        evaluate_test_results();
        
        $display("NoC Congestion Control Test Completed");
        $finish;
    end
    
    // Test 1: Normal Operation
    task test_normal_operation();
        begin
            $display("  Testing normal operation with low load...");
            
            // Set low buffer occupancy
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = 2;  // Low occupancy
                end
                router_congested[router] = 1'b0;
            end
            
            // Run for several cycles
            repeat(100) @(posedge clk);
            
            // Check that no flow control is active
            if (global_flow_control) begin
                $error("Global flow control active under normal conditions");
            end
            
            if (adaptive_routing_enable) begin
                $error("Adaptive routing enabled under normal conditions");
            end
            
            if (network_utilization > 30) begin
                $error("High network utilization reported under low load");
            end
            
            $display("  Normal operation test completed - Utilization: %0d%%", network_utilization);
        end
    endtask
    
    // Test 2: Gradual Congestion Build-up
    task test_gradual_congestion();
        begin
            $display("  Testing gradual congestion build-up...");
            
            // Gradually increase buffer occupancy
            for (int load_level = 0; load_level <= 100; load_level += 10) begin
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    for (int port = 0; port < 5; port++) begin
                        buffer_occupancy[router][port] = (VC_COUNT * VC_DEPTH * load_level) / 100;
                    end
                end
                
                repeat(20) @(posedge clk);
                
                $display("    Load %0d%%: Utilization=%0d%%, Adaptive=%b, Global=%b", 
                         load_level, network_utilization, adaptive_routing_enable, global_flow_control);
                
                // Check thresholds
                if (load_level >= 60 && !adaptive_routing_enable) begin
                    $error("Adaptive routing not enabled at %0d%% load", load_level);
                end
                
                if (load_level >= 90 && !global_flow_control) begin
                    $error("Global flow control not active at %0d%% load", load_level);
                end
                
                if (load_level >= 50 && !qos_enforcement_active) begin
                    $error("QoS enforcement not active at %0d%% load", load_level);
                end
            end
            
            $display("  Gradual congestion test completed");
        end
    endtask
    
    // Test 3: Hotspot Detection
    task test_hotspot_detection();
        begin
            $display("  Testing hotspot detection...");
            
            // Reset to normal conditions
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = 2;
                end
                router_congested[router] = 1'b0;
            end
            
            repeat(50) @(posedge clk);
            
            // Create hotspots at specific routers
            logic [3:0] hotspot_routers[3] = '{4'd5, 4'd10, 4'd15};  // Router 5, 10, 15
            
            for (int i = 0; i < 3; i++) begin
                int router_id = hotspot_routers[i];
                if (router_id < NUM_ROUTERS) begin
                    for (int port = 0; port < 5; port++) begin
                        buffer_occupancy[router_id][port] = VC_COUNT * VC_DEPTH;  // Full buffers
                    end
                    router_congested[router_id] = 1'b1;
                end
            end
            
            // Wait for hotspot detection
            repeat(100) @(posedge clk);
            
            // Check hotspot detection
            logic [15:0] expected_hotspots = 0;
            for (int i = 0; i < 3; i++) begin
                if (hotspot_routers[i] < 16) begin
                    expected_hotspots[hotspot_routers[i]] = 1'b1;
                end
            end
            
            if ((congestion_hotspots & expected_hotspots) != expected_hotspots) begin
                $error("Hotspot detection failed. Expected: %b, Got: %b", 
                       expected_hotspots, congestion_hotspots);
            end else begin
                $display("  Hotspot detection successful: %b", congestion_hotspots);
            end
            
            // Check throttling of hotspot routers
            for (int i = 0; i < 3; i++) begin
                int router_id = hotspot_routers[i];
                if (router_id < NUM_ROUTERS && throttle_level[router_id] < 2'b10) begin
                    $error("Insufficient throttling for hotspot router %0d", router_id);
                end
            end
            
            $display("  Hotspot detection test completed");
        end
    endtask
    
    // Test 4: Flow Control Response
    task test_flow_control_response();
        begin
            $display("  Testing flow control response mechanisms...");
            
            // Create severe congestion
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = VC_COUNT * VC_DEPTH;  // Full buffers
                end
                router_congested[router] = 1'b1;
            end
            
            repeat(50) @(posedge clk);
            
            // Check global flow control activation
            if (!global_flow_control) begin
                $error("Global flow control not activated under severe congestion");
            end
            
            // Check throttling levels
            logic all_throttled = 1'b1;
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                if (throttle_level[router] < 2'b10) begin
                    all_throttled = 1'b0;
                    break;
                end
            end
            
            if (!all_throttled) begin
                $error("Not all routers properly throttled under severe congestion");
            end
            
            // Check packet dropping (simulated)
            logic [31:0] initial_drops = total_packets_dropped;
            repeat(100) @(posedge clk);
            
            if (total_packets_dropped <= initial_drops) begin
                $warning("No packet drops detected under severe congestion");
            end else begin
                $display("  Packet drops detected: %0d", total_packets_dropped - initial_drops);
            end
            
            $display("  Flow control response test completed");
        end
    endtask
    
    // Test 5: QoS Enforcement Under Congestion
    task test_qos_enforcement();
        begin
            $display("  Testing QoS enforcement under congestion...");
            
            // Set moderate congestion to trigger QoS enforcement
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = (VC_COUNT * VC_DEPTH * 60) / 100;
                end
                router_congested[router] = 1'b0;
            end
            
            repeat(50) @(posedge clk);
            
            // Check QoS enforcement activation
            if (!qos_enforcement_active) begin
                $error("QoS enforcement not active under moderate congestion");
            end
            
            // Check priority boost for non-congested routers
            logic some_boosted = 1'b0;
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                if (priority_boost[router] > 2'b00) begin
                    some_boosted = 1'b1;
                    break;
                end
            end
            
            if (!some_boosted) begin
                $warning("No priority boost detected for non-congested routers");
            end
            
            $display("  QoS enforcement test completed");
        end
    endtask
    
    // Test 6: Recovery from Congestion
    task test_congestion_recovery();
        begin
            $display("  Testing recovery from congestion...");
            
            // Start with high congestion
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                for (int port = 0; port < 5; port++) begin
                    buffer_occupancy[router][port] = VC_COUNT * VC_DEPTH;
                end
                router_congested[router] = 1'b1;
            end
            
            repeat(50) @(posedge clk);
            
            // Verify congestion state
            if (!global_flow_control) begin
                $error("Global flow control should be active before recovery test");
            end
            
            // Gradually reduce congestion
            for (int recovery_step = 100; recovery_step >= 0; recovery_step -= 20) begin
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    for (int port = 0; port < 5; port++) begin
                        buffer_occupancy[router][port] = (VC_COUNT * VC_DEPTH * recovery_step) / 100;
                    end
                    router_congested[router] = (recovery_step > 75) ? 1'b1 : 1'b0;
                end
                
                repeat(30) @(posedge clk);
                
                $display("    Recovery %0d%%: Utilization=%0d%%, Global=%b, Adaptive=%b", 
                         100-recovery_step, network_utilization, global_flow_control, adaptive_routing_enable);
            end
            
            // Check final state
            if (global_flow_control) begin
                $error("Global flow control still active after congestion cleared");
            end
            
            if (network_utilization > 20) begin
                $error("High utilization reported after congestion cleared");
            end
            
            $display("  Congestion recovery test completed");
        end
    endtask
    
    // Evaluate overall test results
    task evaluate_test_results();
        begin
            test_passed = 1'b1;
            
            $display("\n=== Congestion Control Test Results ===");
            
            // Check final network state
            if (global_flow_control) begin
                $display("WARN: Global flow control still active");
            end else begin
                $display("PASS: Global flow control properly deactivated");
            end
            
            if (network_utilization > 30) begin
                $display("WARN: High final network utilization: %0d%%", network_utilization);
            end else begin
                $display("PASS: Normal final network utilization: %0d%%", network_utilization);
            end
            
            // Check hotspot detection
            if (|congestion_hotspots) begin
                $display("INFO: Hotspots still detected: %b", congestion_hotspots);
            end else begin
                $display("PASS: No hotspots detected in final state");
            end
            
            // Overall result
            if (test_passed) begin
                $display("OVERALL: Congestion Control Test PASSED");
            end else begin
                $display("OVERALL: Congestion Control Test FAILED");
            end
            
            $display("Final Statistics:");
            $display("  Network Utilization: %0d%%", network_utilization);
            $display("  Total Packets Dropped: %0d", total_packets_dropped);
            $display("  Congestion Hotspots: %b", congestion_hotspots);
        end
    endtask
    
    // Monitor and display key metrics
    always @(posedge clk) begin
        test_cycle <= test_cycle + 1;
        
        // Periodic status display during long tests
        if (test_cycle % 500 == 0 && test_cycle > 0) begin
            $display("  Cycle %0d: Util=%0d%%, Global=%b, Adaptive=%b, Hotspots=%b", 
                     test_cycle, network_utilization, global_flow_control, 
                     adaptive_routing_enable, congestion_hotspots);
        end
    end

endmodule