// Network Packet Processor Test Module
// Tests packet classification and protocol processing

`timescale 1ns/1ps

module test_network_packet_processor;

    // Test parameters
    parameter DATA_WIDTH = 64;
    parameter ADDR_WIDTH = 32;
    parameter MAX_PACKET_SIZE = 1518;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // Input packet interface
    logic [DATA_WIDTH-1:0]   pkt_data_in;
    logic                    pkt_valid_in;
    logic                    pkt_ready_in;
    logic                    pkt_sop;
    logic                    pkt_eop;
    logic [15:0]             pkt_length;
    
    // Output packet interface
    logic [DATA_WIDTH-1:0]   pkt_data_out;
    logic                    pkt_valid_out;
    logic                    pkt_ready_out;
    logic                    pkt_sop_out;
    logic                    pkt_eop_out;
    logic [15:0]             pkt_length_out;
    
    // Packet classification results
    logic [3:0]              pkt_type;
    logic [31:0]             src_ip;
    logic [31:0]             dst_ip;
    logic [15:0]             src_port;
    logic [15:0]             dst_port;
    logic [7:0]              protocol;
    
    // Configuration
    logic [47:0]             local_mac;
    logic [31:0]             local_ip;
    logic                    promiscuous_mode;
    
    // Statistics
    logic [31:0]             processed_packets;
    logic [31:0]             dropped_packets;
    logic [31:0]             error_packets;
    
    // DMA interface
    logic                    dma_req;
    logic                    dma_ack;
    logic [ADDR_WIDTH-1:0]   dma_addr;
    logic [15:0]             dma_length;
    logic                    dma_write;
    
    // Interrupts
    logic                    packet_received_irq;
    logic                    error_irq;
    
    // Device Under Test
    network_packet_processor #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_PACKET_SIZE(MAX_PACKET_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pkt_data_in(pkt_data_in),
        .pkt_valid_in(pkt_valid_in),
        .pkt_ready_in(pkt_ready_in),
        .pkt_sop(pkt_sop),
        .pkt_eop(pkt_eop),
        .pkt_length(pkt_length),
        .pkt_data_out(pkt_data_out),
        .pkt_valid_out(pkt_valid_out),
        .pkt_ready_out(pkt_ready_out),
        .pkt_sop_out(pkt_sop_out),
        .pkt_eop_out(pkt_eop_out),
        .pkt_length_out(pkt_length_out),
        .pkt_type(pkt_type),
        .src_ip(src_ip),
        .dst_ip(dst_ip),
        .src_port(src_port),
        .dst_port(dst_port),
        .protocol(protocol),
        .local_mac(local_mac),
        .local_ip(local_ip),
        .promiscuous_mode(promiscuous_mode),
        .processed_packets(processed_packets),
        .dropped_packets(dropped_packets),
        .error_packets(error_packets),
        .dma_req(dma_req),
        .dma_ack(dma_ack),
        .dma_addr(dma_addr),
        .dma_length(dma_length),
        .dma_write(dma_write),
        .packet_received_irq(packet_received_irq),
        .error_irq(error_irq)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test variables
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;
    
    // Test packet storage
    logic [7:0] test_packet [0:1517];
    integer packet_words;
    
    // Test tasks
    task reset_system();
        begin
            rst_n = 0;
            
            // Initialize input signals
            pkt_data_in = 0;
            pkt_valid_in = 0;
            pkt_sop = 0;
            pkt_eop = 0;
            pkt_length = 0;
            pkt_ready_out = 1;
            
            // Configuration
            local_mac = 48'h123456789ABC;
            local_ip = 32'hC0A80101;  // 192.168.1.1
            promiscuous_mode = 0;
            
            // DMA
            dma_ack = 1;
            
            repeat(10) @(posedge clk);
            rst_n = 1;
            repeat(10) @(posedge clk);
        end
    endtask
    
    task check_result(input string test_name, input logic [31:0] expected, input logic [31:0] actual);
        begin
            test_count++;
            if (expected === actual) begin
                $display("PASS: %s", test_name);
                pass_count++;
            end else begin
                $display("FAIL: %s - Expected: %h, Actual: %h", test_name, expected, actual);
                fail_count++;
            end
        end
    endtask
    
    task create_ethernet_header(input [47:0] dst_mac, input [47:0] src_mac, input [15:0] ethertype);
        begin
            // Destination MAC
            test_packet[0] = dst_mac[47:40];
            test_packet[1] = dst_mac[39:32];
            test_packet[2] = dst_mac[31:24];
            test_packet[3] = dst_mac[23:16];
            test_packet[4] = dst_mac[15:8];
            test_packet[5] = dst_mac[7:0];
            
            // Source MAC
            test_packet[6] = src_mac[47:40];
            test_packet[7] = src_mac[39:32];
            test_packet[8] = src_mac[31:24];
            test_packet[9] = src_mac[23:16];
            test_packet[10] = src_mac[15:8];
            test_packet[11] = src_mac[7:0];
            
            // EtherType
            test_packet[12] = ethertype[15:8];
            test_packet[13] = ethertype[7:0];
        end
    endtask
    
    task create_ip_header(input [31:0] src_ip_addr, input [31:0] dst_ip_addr, input [7:0] proto, input [15:0] payload_len);
        begin
            // IP header starts at byte 14
            test_packet[14] = 8'h45;  // Version (4) + IHL (5)
            test_packet[15] = 8'h00;  // TOS
            test_packet[16] = (20 + payload_len) >> 8;  // Total length high
            test_packet[17] = (20 + payload_len) & 8'hFF;  // Total length low
            test_packet[18] = 8'h12;  // Identification high
            test_packet[19] = 8'h34;  // Identification low
            test_packet[20] = 8'h40;  // Flags + Fragment offset high
            test_packet[21] = 8'h00;  // Fragment offset low
            test_packet[22] = 8'h40;  // TTL
            test_packet[23] = proto;  // Protocol
            test_packet[24] = 8'h00;  // Header checksum high (simplified)
            test_packet[25] = 8'h00;  // Header checksum low
            
            // Source IP
            test_packet[26] = src_ip_addr[31:24];
            test_packet[27] = src_ip_addr[23:16];
            test_packet[28] = src_ip_addr[15:8];
            test_packet[29] = src_ip_addr[7:0];
            
            // Destination IP
            test_packet[30] = dst_ip_addr[31:24];
            test_packet[31] = dst_ip_addr[23:16];
            test_packet[32] = dst_ip_addr[15:8];
            test_packet[33] = dst_ip_addr[7:0];
        end
    endtask
    
    task create_tcp_header(input [15:0] src_port_val, input [15:0] dst_port_val);
        begin
            // TCP header starts at byte 34
            test_packet[34] = src_port_val[15:8];   // Source port high
            test_packet[35] = src_port_val[7:0];    // Source port low
            test_packet[36] = dst_port_val[15:8];   // Destination port high
            test_packet[37] = dst_port_val[7:0];    // Destination port low
            test_packet[38] = 8'h00;  // Sequence number
            test_packet[39] = 8'h00;
            test_packet[40] = 8'h00;
            test_packet[41] = 8'h01;
            test_packet[42] = 8'h00;  // Acknowledgment number
            test_packet[43] = 8'h00;
            test_packet[44] = 8'h00;
            test_packet[45] = 8'h00;
            test_packet[46] = 8'h50;  // Data offset (5) + Reserved
            test_packet[47] = 8'h18;  // Flags (PSH + ACK)
            test_packet[48] = 8'h20;  // Window size
            test_packet[49] = 8'h00;
            test_packet[50] = 8'h00;  // Checksum
            test_packet[51] = 8'h00;
            test_packet[52] = 8'h00;  // Urgent pointer
            test_packet[53] = 8'h00;
        end
    endtask
    
    task create_udp_header(input [15:0] src_port_val, input [15:0] dst_port_val, input [15:0] payload_len);
        begin
            // UDP header starts at byte 34
            test_packet[34] = src_port_val[15:8];   // Source port high
            test_packet[35] = src_port_val[7:0];    // Source port low
            test_packet[36] = dst_port_val[15:8];   // Destination port high
            test_packet[37] = dst_port_val[7:0];    // Destination port low
            test_packet[38] = (8 + payload_len) >> 8;  // Length high
            test_packet[39] = (8 + payload_len) & 8'hFF;  // Length low
            test_packet[40] = 8'h00;  // Checksum high
            test_packet[41] = 8'h00;  // Checksum low
        end
    endtask
    
    task send_packet(input integer size);
        integer i, word_idx;
        logic [DATA_WIDTH-1:0] word_data;
        begin
            packet_words = (size + 7) / 8;  // Round up to word boundary
            
            @(posedge clk);
            pkt_sop = 1;
            pkt_length = size;
            
            for (i = 0; i < packet_words; i++) begin
                // Pack bytes into 64-bit word
                word_data = 0;
                for (word_idx = 0; word_idx < 8; word_idx++) begin
                    if (i * 8 + word_idx < size) begin
                        word_data[word_idx*8 +: 8] = test_packet[i * 8 + word_idx];
                    end
                end
                
                pkt_data_in = word_data;
                pkt_valid_in = 1;
                pkt_eop = (i == packet_words - 1);
                
                wait(pkt_ready_in);
                @(posedge clk);
                pkt_sop = 0;
            end
            
            pkt_valid_in = 0;
            pkt_eop = 0;
        end
    endtask
    
    task test_arp_packet();
        begin
            $display("Testing ARP Packet Processing...");
            
            // Create ARP packet
            create_ethernet_header(local_mac, 48'hAABBCCDDEEFF, 16'h0806);  // ARP
            
            // Add minimal ARP payload
            for (int i = 14; i < 60; i++) begin
                test_packet[i] = 8'h00;
            end
            
            send_packet(60);
            
            // Wait for processing
            repeat(50) @(posedge clk);
            
            check_result("ARP Packet Type", 4'h1, pkt_type);
            check_result("Processed Packets", 32'h1, processed_packets);
        end
    endtask
    
    task test_tcp_packet();
        begin
            $display("Testing TCP Packet Processing...");
            
            // Create TCP packet
            create_ethernet_header(local_mac, 48'hAABBCCDDEEFF, 16'h0800);  // IPv4
            create_ip_header(32'hC0A80102, local_ip, 8'h06, 20);  // TCP
            create_tcp_header(16'h1234, 16'h5678);
            
            // Add minimal payload
            for (int i = 54; i < 74; i++) begin
                test_packet[i] = i[7:0];
            end
            
            send_packet(74);
            
            // Wait for processing
            repeat(100) @(posedge clk);
            
            check_result("TCP Packet Type", 4'h3, pkt_type);
            check_result("Source IP", 32'hC0A80102, src_ip);
            check_result("Destination IP", local_ip, dst_ip);
            check_result("Source Port", 16'h1234, src_port);
            check_result("Destination Port", 16'h5678, dst_port);
            check_result("Protocol", 8'h06, protocol);
        end
    endtask
    
    task test_udp_packet();
        begin
            $display("Testing UDP Packet Processing...");
            
            // Create UDP packet
            create_ethernet_header(local_mac, 48'hAABBCCDDEEFF, 16'h0800);  // IPv4
            create_ip_header(32'hC0A80103, local_ip, 8'h11, 16);  // UDP
            create_udp_header(16'h9ABC, 16'hDEF0, 8);
            
            // Add minimal payload
            for (int i = 42; i < 50; i++) begin
                test_packet[i] = i[7:0];
            end
            
            send_packet(50);
            
            // Wait for processing
            repeat(100) @(posedge clk);
            
            check_result("UDP Packet Type", 4'h4, pkt_type);
            check_result("Source IP", 32'hC0A80103, src_ip);
            check_result("Destination IP", local_ip, dst_ip);
            check_result("Source Port", 16'h9ABC, src_port);
            check_result("Destination Port", 16'hDEF0, dst_port);
            check_result("Protocol", 8'h11, protocol);
        end
    endtask
    
    task test_packet_filtering();
        begin
            $display("Testing Packet Filtering...");
            
            // Create packet for different MAC (should be dropped)
            create_ethernet_header(48'hFFFFFFFFFFFF, 48'hAABBCCDDEEFF, 16'h0800);
            create_ip_header(32'hC0A80104, 32'hC0A80105, 8'h06, 20);  // Different IP
            create_tcp_header(16'h1111, 16'h2222);
            
            send_packet(54);
            
            // Wait for processing
            repeat(100) @(posedge clk);
            
            // Should be dropped due to wrong destination
            check_result("Dropped Packets", 32'h1, dropped_packets);
        end
    endtask
    
    task test_promiscuous_mode();
        begin
            $display("Testing Promiscuous Mode...");
            
            // Enable promiscuous mode
            promiscuous_mode = 1;
            
            // Create packet for different MAC
            create_ethernet_header(48'hDEADBEEFCAFE, 48'hAABBCCDDEEFF, 16'h0800);
            create_ip_header(32'hC0A80106, 32'hC0A80107, 8'h06, 20);
            create_tcp_header(16'h3333, 16'h4444);
            
            send_packet(54);
            
            // Wait for processing
            repeat(100) @(posedge clk);
            
            // Should be processed in promiscuous mode
            check_result("TCP Packet Type (Promiscuous)", 4'h3, pkt_type);
        end
    endtask
    
    task test_dma_interface();
        begin
            $display("Testing DMA Interface...");
            
            // Create a packet that should trigger DMA
            create_ethernet_header(local_mac, 48'hAABBCCDDEEFF, 16'h0800);
            create_ip_header(32'hC0A80108, local_ip, 8'h06, 20);
            create_tcp_header(16'h5555, 16'h6666);
            
            send_packet(54);
            
            // Wait for DMA request
            wait(dma_req);
            check_result("DMA Request", 1'b1, dma_req);
            check_result("DMA Write", 1'b1, dma_write);
            
            // Check interrupt
            wait(packet_received_irq);
            check_result("Packet Received IRQ", 1'b1, packet_received_irq);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Network Packet Processor Tests...");
        
        reset_system();
        
        test_arp_packet();
        test_tcp_packet();
        test_udp_packet();
        test_packet_filtering();
        test_promiscuous_mode();
        test_dma_interface();
        
        // Test summary
        $display("\n=== Network Packet Processor Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All Network Packet Processor tests PASSED!");
        end else begin
            $display("Some Network Packet Processor tests FAILED!");
        end
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #5000000;  // 5ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end
    
    // Waveform dumping
    initial begin
        $dumpfile("test_network_packet_processor.vcd");
        $dumpvars(0, test_network_packet_processor);
    end

endmodule