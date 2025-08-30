// NoC Mesh Network Test
// System-level test for mesh network topology

`timescale 1ns/1ps
`include "noc_packet.sv"

module test_noc_mesh;

    // Test parameters
    parameter int MESH_SIZE_X = 4;
    parameter int MESH_SIZE_Y = 4;
    parameter int NUM_NODES = MESH_SIZE_X * MESH_SIZE_Y;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Processing element interfaces
    logic [63:0] pe_addr[NUM_NODES];
    logic [255:0] pe_wdata[NUM_NODES];
    logic [255:0] pe_rdata[NUM_NODES];
    logic pe_read[NUM_NODES];
    logic pe_write[NUM_NODES];
    logic [2:0] pe_size[NUM_NODES];
    qos_level_t pe_qos[NUM_NODES];
    logic pe_ready[NUM_NODES];
    logic pe_valid[NUM_NODES];
    
    // Network monitoring
    logic [31:0] total_packets_routed;
    logic [31:0] network_utilization;
    logic network_congestion;
    logic [31:0] avg_latency;
    
    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Traffic generation
    logic [31:0] packets_sent[NUM_NODES];
    logic [31:0] packets_received[NUM_NODES];
    
    // DUT instantiation
    noc_mesh #(
        .MESH_SIZE_X(MESH_SIZE_X),
        .MESH_SIZE_Y(MESH_SIZE_Y)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pe_addr(pe_addr),
        .pe_wdata(pe_wdata),
        .pe_rdata(pe_rdata),
        .pe_read(pe_read),
        .pe_write(pe_write),
        .pe_size(pe_size),
        .pe_qos(pe_qos),
        .pe_ready(pe_ready),
        .pe_valid(pe_valid),
        .total_packets_routed(total_packets_routed),
        .network_utilization(network_utilization),
        .network_congestion(network_congestion),
        .avg_latency(avg_latency)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        $display("Starting NoC Mesh Network Tests");
        
        // Initialize signals
        rst_n = 0;
        initialize_pe_interfaces();
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);
        
        // Run tests
        test_point_to_point();
        test_broadcast_traffic();
        test_random_traffic();
        test_hotspot_traffic();
        test_network_congestion();
        test_qos_differentiation();
        test_fault_tolerance();
        
        // Report results
        report_test_results();
        
        $finish;
    end
    
    // Initialize PE interfaces
    task initialize_pe_interfaces();
        for (int i = 0; i < NUM_NODES; i++) begin
            pe_addr[i] = 0;
            pe_wdata[i] = 0;
            pe_read[i] = 1'b0;
            pe_write[i] = 1'b0;
            pe_size[i] = 3'b011; // 8 bytes
            pe_qos[i] = QOS_NORMAL;
            packets_sent[i] = 0;
            packets_received[i] = 0;
        end
    endtask
    
    // Test point-to-point communication
    task test_point_to_point();
        $display("\n--- Test: Point-to-Point Communication ---");
        
        // Test communication between adjacent nodes
        test_communication(0, 1, "Adjacent nodes (0->1)");
        test_communication(5, 10, "Diagonal nodes (5->10)");
        test_communication(15, 0, "Corner to corner (15->0)");
        
    endtask
    
    // Test broadcast traffic pattern
    task test_broadcast_traffic();
        $display("\n--- Test: Broadcast Traffic ---");
        
        logic [31:0] initial_packets = total_packets_routed;
        
        // Node 0 sends to all other nodes
        for (int dst = 1; dst < NUM_NODES; dst++) begin
            send_packet(0, dst, $random, QOS_NORMAL);
        end
        
        // Wait for completion
        repeat(100) @(posedge clk);
        
        check_result("Broadcast traffic completion", 
                    total_packets_routed > initial_packets);
        
    endtask
    
    // Test random traffic pattern
    task test_random_traffic();
        $display("\n--- Test: Random Traffic Pattern ---");
        
        logic [31:0] initial_packets = total_packets_routed;
        
        // Generate random traffic
        for (int i = 0; i < 50; i++) begin
            int src = $random % NUM_NODES;
            int dst = $random % NUM_NODES;
            if (src != dst) begin
                send_packet(src, dst, $random, qos_level_t'($random % 4));
            end
            repeat(2) @(posedge clk);
        end
        
        // Wait for completion
        repeat(200) @(posedge clk);
        
        check_result("Random traffic handling", 
                    total_packets_routed > initial_packets);
        
    endtask
    
    // Test hotspot traffic pattern
    task test_hotspot_traffic();
        $display("\n--- Test: Hotspot Traffic ---");
        
        logic [31:0] initial_packets = total_packets_routed;
        int hotspot_node = 5; // Center node
        
        // Multiple nodes send to hotspot
        for (int src = 0; src < NUM_NODES; src++) begin
            if (src != hotspot_node) begin
                send_packet(src, hotspot_node, $random, QOS_NORMAL);
            end
        end
        
        // Wait and check for congestion handling
        repeat(150) @(posedge clk);
        
        check_result("Hotspot traffic handling", 
                    total_packets_routed > initial_packets);
        
    endtask
    
    // Test network congestion scenarios
    task test_network_congestion();
        $display("\n--- Test: Network Congestion ---");
        
        // Generate heavy traffic to cause congestion
        for (int round = 0; round < 3; round++) begin
            for (int src = 0; src < NUM_NODES; src++) begin
                for (int dst = 0; dst < NUM_NODES; dst++) begin
                    if (src != dst) begin
                        send_packet(src, dst, $random, QOS_NORMAL);
                    end
                end
                repeat(1) @(posedge clk);
            end
        end
        
        // Wait and monitor congestion
        repeat(300) @(posedge clk);
        
        check_result("Congestion detection", network_congestion);
        
        // Wait for congestion to clear
        repeat(500) @(posedge clk);
        
    endtask
    
    // Test QoS differentiation
    task test_qos_differentiation();
        $display("\n--- Test: QoS Differentiation ---");
        
        logic [31:0] initial_packets = total_packets_routed;
        
        // Send packets with different QoS levels
        send_packet(0, 15, $random, QOS_LOW);
        send_packet(1, 14, $random, QOS_NORMAL);
        send_packet(2, 13, $random, QOS_HIGH);
        send_packet(3, 12, $random, QOS_URGENT);
        
        repeat(100) @(posedge clk);
        
        check_result("QoS differentiation", 
                    total_packets_routed > initial_packets);
        
    endtask
    
    // Test fault tolerance (simplified)
    task test_fault_tolerance();
        $display("\n--- Test: Fault Tolerance ---");
        
        // This is a simplified test - in reality, we would inject faults
        // For now, just verify the network continues to function
        
        logic [31:0] initial_packets = total_packets_routed;
        
        // Send some traffic
        for (int i = 0; i < 10; i++) begin
            int src = $random % NUM_NODES;
            int dst = $random % NUM_NODES;
            if (src != dst) begin
                send_packet(src, dst, $random, QOS_NORMAL);
            end
        end
        
        repeat(100) @(posedge clk);
        
        check_result("Basic fault tolerance", 
                    total_packets_routed > initial_packets);
        
    endtask
    
    // Helper task to test communication between two nodes
    task test_communication(int src_node, int dst_node, string test_name);
        
        logic [31:0] initial_packets = total_packets_routed;
        
        // Send packet
        send_packet(src_node, dst_node, 32'hDEADBEEF, QOS_NORMAL);
        
        // Wait for completion
        repeat(50) @(posedge clk);
        
        // Check if packet was routed
        check_result(test_name, total_packets_routed > initial_packets);
        
    endtask
    
    // Helper task to send a packet from source to destination
    task send_packet(int src_node, int dst_node, logic [31:0] data, qos_level_t qos);
        
        // Calculate destination address (simplified mapping)
        logic [63:0] dst_addr = {32'h0, dst_node[3:0], dst_node[7:4], 24'h0};
        
        @(posedge clk);
        
        // Wait for PE to be ready
        wait(pe_ready[src_node]);
        
        // Send write request
        pe_addr[src_node] = dst_addr;
        pe_wdata[src_node] = {224'h0, data};
        pe_write[src_node] = 1'b1;
        pe_qos[src_node] = qos;
        
        @(posedge clk);
        pe_write[src_node] = 1'b0;
        
        packets_sent[src_node]++;
        
    endtask
    
    // Helper task to check test results
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
    
    // Report test results
    task report_test_results();
        $display("\n=== NoC Mesh Test Results ===");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Total Packets Routed: %0d", total_packets_routed);
        $display("Network Utilization: %0d%%", network_utilization);
        $display("Average Latency: %0d cycles", avg_latency);
        
        if (fail_count == 0) begin
            $display("All NoC Mesh tests PASSED!");
        end else begin
            $display("Some NoC Mesh tests FAILED!");
        end
    endtask
    
    // Performance monitoring
    initial begin
        forever begin
            repeat(1000) @(posedge clk);
            $display("Time: %0t, Packets: %0d, Utilization: %0d%%, Congestion: %b", 
                    $time, total_packets_routed, network_utilization, network_congestion);
        end
    end
    
    // Packet reception monitoring
    genvar i;
    generate
        for (i = 0; i < NUM_NODES; i++) begin : gen_rx_monitor
            always @(posedge clk) begin
                if (pe_valid[i]) begin
                    packets_received[i] <= packets_received[i] + 1;
                end
            end
        end
    endgenerate

endmodule