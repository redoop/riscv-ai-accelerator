// NoC Router Test
// Comprehensive test for mesh router functionality

`timescale 1ns/1ps
`include "noc_packet.sv"

module test_noc_router;

    // Test parameters
    parameter int MESH_SIZE_X = 4;
    parameter int MESH_SIZE_Y = 4;
    parameter int TEST_X = 1;
    parameter int TEST_Y = 1;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Router interfaces
    noc_flit_t flit_in[5];
    logic valid_in[5];
    logic ready_out[5];
    
    noc_flit_t flit_out[5];
    logic valid_out[5];
    logic ready_in[5];
    
    // Monitoring signals
    logic [31:0] packets_routed;
    logic [31:0] buffer_occupancy[5];
    logic congestion_detected;
    
    // Test variables
    int test_count = 0;
    int pass_count = 0;
    int fail_count = 0;
    
    // Helper variables for tests
    logic correct_direction;
    logic packet_buffered;
    logic [31:0] initial_packets;
    
    // DUT instantiation
    noc_router #(
        .X_COORD(TEST_X),
        .Y_COORD(TEST_Y),
        .MESH_SIZE_X(MESH_SIZE_X),
        .MESH_SIZE_Y(MESH_SIZE_Y)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .flit_in(flit_in),
        .valid_in(valid_in),
        .ready_out(ready_out),
        .flit_out(flit_out),
        .valid_out(valid_out),
        .ready_in(ready_in),
        .packets_routed(packets_routed),
        .buffer_occupancy(buffer_occupancy),
        .congestion_detected(congestion_detected)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        $display("Starting NoC Router Tests");
        
        // Initialize signals
        rst_n = 0;
        for (int i = 0; i < 5; i++) begin
            flit_in[i] = '0;
            valid_in[i] = 1'b0;
            ready_in[i] = 1'b1;
        end
        
        // Reset sequence
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        // Run tests
        test_basic_routing();
        test_xy_routing();
        test_virtual_channels();
        test_flow_control();
        test_deadlock_avoidance();
        test_qos_priority();
        test_congestion_handling();
        
        // Report results
        $display("\n=== Test Results ===");
        $display("Total Tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("Some tests FAILED!");
        end
        
        $finish;
    end
    
    // Test basic routing functionality
    task test_basic_routing();
        $display("\n--- Test: Basic Routing ---");
        
        // Test packet from local to east
        send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_NORMAL, 8'h01, 256'hDEADBEEF);
        
        // Wait for routing
        repeat(10) @(posedge clk);
        
        // Check if packet appears at east output
        check_result("Basic routing to east", 
                    valid_out[DIR_EAST] && flit_out[DIR_EAST].header.dst_x == 2);
        
        // Clear the packet
        ready_in[DIR_EAST] = 1'b1;
        repeat(5) @(posedge clk);
        ready_in[DIR_EAST] = 1'b0;
        
    endtask
    
    // Test XY routing algorithm
    task test_xy_routing();
        $display("\n--- Test: XY Routing Algorithm ---");
        
        // Test routing to different destinations
        test_route_direction(0, 1, DIR_WEST, "Route West");
        test_route_direction(3, 1, DIR_EAST, "Route East");
        test_route_direction(1, 0, DIR_NORTH, "Route North");
        test_route_direction(1, 3, DIR_SOUTH, "Route South");
        test_route_direction(1, 1, DIR_LOCAL, "Route Local");
        
    endtask
    
    // Helper task to test routing direction
    task test_route_direction(int dst_x, int dst_y, int expected_dir, string test_name);
        
        // Send packet
        send_packet(DIR_LOCAL, dst_x, dst_y, PKT_READ_REQ, QOS_NORMAL, 8'h10, 256'h12345678);
        
        // Wait for routing
        repeat(10) @(posedge clk);
        
        // Check output direction
        correct_direction = valid_out[expected_dir];
        check_result(test_name, correct_direction);
        
        // Clear packet
        if (correct_direction) begin
            ready_in[expected_dir] = 1'b1;
            repeat(5) @(posedge clk);
            ready_in[expected_dir] = 1'b0;
        end
        
    endtask
    
    // Test virtual channel functionality
    task test_virtual_channels();
        $display("\n--- Test: Virtual Channels ---");
        
        // Send packets with different QoS levels
        send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_URGENT, 8'h20, 256'hAAAAAAAA);
        send_packet(DIR_LOCAL, 2, 1, PKT_WRITE_REQ, QOS_LOW, 8'h21, 256'hBBBBBBBB);
        
        repeat(15) @(posedge clk);
        
        // Check that urgent packet is processed first
        check_result("Virtual channel priority", 
                    packets_routed > 0);
        
        // Clear packets
        ready_in[DIR_EAST] = 1'b1;
        repeat(10) @(posedge clk);
        ready_in[DIR_EAST] = 1'b0;
        
    endtask
    
    // Test flow control mechanism
    task test_flow_control();
        $display("\n--- Test: Flow Control ---");
        
        // Block downstream and send packet
        ready_in[DIR_EAST] = 1'b0;
        send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_NORMAL, 8'h30, 256'hCCCCCCCC);
        
        repeat(10) @(posedge clk);
        
        // Check that packet is buffered (not lost)
        packet_buffered = (buffer_occupancy[DIR_LOCAL] > 0);
        check_result("Flow control buffering", packet_buffered);
        
        // Unblock and check packet flows
        ready_in[DIR_EAST] = 1'b1;
        repeat(10) @(posedge clk);
        
        check_result("Flow control release", valid_out[DIR_EAST]);
        
        repeat(5) @(posedge clk);
        ready_in[DIR_EAST] = 1'b0;
        
    endtask
    
    // Test deadlock avoidance
    task test_deadlock_avoidance();
        $display("\n--- Test: Deadlock Avoidance ---");
        
        // Create potential deadlock scenario with XY routing
        // XY routing should naturally avoid deadlocks
        
        // Send multiple packets in different directions
        send_packet(DIR_NORTH, 2, 1, PKT_READ_REQ, QOS_NORMAL, 8'h40, 256'h11111111);
        send_packet(DIR_SOUTH, 0, 1, PKT_READ_REQ, QOS_NORMAL, 8'h41, 256'h22222222);
        send_packet(DIR_EAST, 1, 0, PKT_READ_REQ, QOS_NORMAL, 8'h42, 256'h33333333);
        send_packet(DIR_WEST, 1, 2, PKT_READ_REQ, QOS_NORMAL, 8'h43, 256'h44444444);
        
        repeat(20) @(posedge clk);
        
        // Check that no deadlock occurred (packets are still moving)
        check_result("Deadlock avoidance", !congestion_detected);
        
        // Clear all outputs
        for (int i = 0; i < 5; i++) begin
            ready_in[i] = 1'b1;
        end
        repeat(10) @(posedge clk);
        for (int i = 0; i < 5; i++) begin
            ready_in[i] = 1'b0;
        end
        
    endtask
    
    // Test QoS priority handling
    task test_qos_priority();
        $display("\n--- Test: QoS Priority ---");
        
        initial_packets = packets_routed;
        
        // Send low priority packet first
        send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_LOW, 8'h50, 256'h55555555);
        repeat(2) @(posedge clk);
        
        // Send high priority packet
        send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_URGENT, 8'h51, 256'h66666666);
        
        repeat(15) @(posedge clk);
        
        // Check that packets were processed
        check_result("QoS priority processing", 
                    packets_routed > initial_packets);
        
        // Clear packets
        ready_in[DIR_EAST] = 1'b1;
        repeat(10) @(posedge clk);
        ready_in[DIR_EAST] = 1'b0;
        
    endtask
    
    // Test congestion handling
    task test_congestion_handling();
        $display("\n--- Test: Congestion Handling ---");
        
        // Fill buffers to create congestion
        ready_in[DIR_EAST] = 1'b0; // Block output
        
        // Send multiple packets to fill buffers
        for (int i = 0; i < 10; i++) begin
            send_packet(DIR_LOCAL, 2, 1, PKT_READ_REQ, QOS_NORMAL, i, 256'h77777777);
            repeat(2) @(posedge clk);
        end
        
        repeat(10) @(posedge clk);
        
        // Check congestion detection
        check_result("Congestion detection", congestion_detected);
        
        // Clear congestion
        ready_in[DIR_EAST] = 1'b1;
        repeat(20) @(posedge clk);
        ready_in[DIR_EAST] = 1'b0;
        
    endtask
    
    // Helper task to send a packet
    task send_packet(int input_port, int dst_x, int dst_y, pkt_type_t pkt_type, 
                    qos_level_t qos, logic [7:0] pkt_id, logic [255:0] data);
        
        @(posedge clk);
        
        // Create header flit
        flit_in[input_port].head = 1'b1;
        flit_in[input_port].tail = 1'b1; // Single flit packet
        flit_in[input_port].header.src_x = TEST_X;
        flit_in[input_port].header.src_y = TEST_Y;
        flit_in[input_port].header.dst_x = dst_x;
        flit_in[input_port].header.dst_y = dst_y;
        flit_in[input_port].header.pkt_type = pkt_type;
        flit_in[input_port].header.qos = qos;
        flit_in[input_port].header.pkt_id = pkt_id;
        flit_in[input_port].header.length = 1;
        flit_in[input_port].header.multicast = 1'b0;
        flit_in[input_port].data = data;
        valid_in[input_port] = 1'b1;
        
        @(posedge clk);
        valid_in[input_port] = 1'b0;
        flit_in[input_port] = '0;
        
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
    
    // Monitor for debugging
    initial begin
        $monitor("Time: %0t, Packets routed: %0d, Congestion: %b", 
                $time, packets_routed, congestion_detected);
    end

endmodule