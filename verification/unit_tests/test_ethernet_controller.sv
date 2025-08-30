// Ethernet Controller Test Module
// Comprehensive test for Gigabit Ethernet MAC and PHY interface

`timescale 1ns/1ps

module test_ethernet_controller;

    // Test parameters
    parameter DATA_WIDTH = 64;
    parameter ADDR_WIDTH = 32;
    parameter FIFO_DEPTH = 1024;
    parameter CLK_PERIOD = 8;  // 125MHz for Gigabit Ethernet
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // RGMII interface
    logic        rgmii_txc;
    logic        rgmii_tx_ctl;
    logic [3:0]  rgmii_txd;
    logic        rgmii_rxc;
    logic        rgmii_rx_ctl;
    logic [3:0]  rgmii_rxd;
    
    // MDIO interface
    logic        mdio_mdc;
    logic        mdio_mdio;
    logic        phy_reset_n;
    
    // AXI configuration interface
    logic [31:0] axi_awaddr;
    logic [2:0]  axi_awprot;
    logic        axi_awvalid;
    logic        axi_awready;
    logic [31:0] axi_wdata;
    logic [3:0]  axi_wstrb;
    logic        axi_wvalid;
    logic        axi_wready;
    logic [1:0]  axi_bresp;
    logic        axi_bvalid;
    logic        axi_bready;
    logic [31:0] axi_araddr;
    logic [2:0]  axi_arprot;
    logic        axi_arvalid;
    logic        axi_arready;
    logic [31:0] axi_rdata;
    logic [1:0]  axi_rresp;
    logic        axi_rvalid;
    logic        axi_rready;
    
    // DMA interface
    logic                dma_tx_req;
    logic                dma_tx_ack;
    logic [ADDR_WIDTH-1:0] dma_tx_addr;
    logic [15:0]         dma_tx_length;
    logic [DATA_WIDTH-1:0] dma_tx_data;
    logic                dma_tx_valid;
    logic                dma_tx_ready;
    
    logic                dma_rx_req;
    logic                dma_rx_ack;
    logic [ADDR_WIDTH-1:0] dma_rx_addr;
    logic [15:0]         dma_rx_length;
    logic [DATA_WIDTH-1:0] dma_rx_data;
    logic                dma_rx_valid;
    logic                dma_rx_ready;
    
    // Interrupt and status
    logic        tx_complete_irq;
    logic        rx_complete_irq;
    logic        error_irq;
    logic        link_up;
    logic [1:0]  link_speed;
    logic        full_duplex;
    logic [31:0] tx_packet_count;
    logic [31:0] rx_packet_count;
    logic [31:0] error_count;
    
    // AXI interface instance
    axi4_if axi_cfg_if (
        .aclk(clk),
        .aresetn(rst_n)
    );
    
    // Connect AXI signals
    assign axi_cfg_if.awaddr = axi_awaddr;
    assign axi_cfg_if.awprot = axi_awprot;
    assign axi_cfg_if.awvalid = axi_awvalid;
    assign axi_awready = axi_cfg_if.awready;
    assign axi_cfg_if.wdata = axi_wdata;
    assign axi_cfg_if.wstrb = axi_wstrb;
    assign axi_cfg_if.wvalid = axi_wvalid;
    assign axi_wready = axi_cfg_if.wready;
    assign axi_bresp = axi_cfg_if.bresp;
    assign axi_bvalid = axi_cfg_if.bvalid;
    assign axi_cfg_if.bready = axi_bready;
    assign axi_cfg_if.araddr = axi_araddr;
    assign axi_cfg_if.arprot = axi_arprot;
    assign axi_cfg_if.arvalid = axi_arvalid;
    assign axi_arready = axi_cfg_if.arready;
    assign axi_rdata = axi_cfg_if.rdata;
    assign axi_rresp = axi_cfg_if.rresp;
    assign axi_rvalid = axi_cfg_if.rvalid;
    assign axi_cfg_if.rready = axi_rready;
    
    // Device Under Test
    ethernet_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .rgmii_txc(rgmii_txc),
        .rgmii_tx_ctl(rgmii_tx_ctl),
        .rgmii_txd(rgmii_txd),
        .rgmii_rxc(rgmii_rxc),
        .rgmii_rx_ctl(rgmii_rx_ctl),
        .rgmii_rxd(rgmii_rxd),
        .mdio_mdc(mdio_mdc),
        .mdio_mdio(mdio_mdio),
        .phy_reset_n(phy_reset_n),
        .axi_cfg_if(axi_cfg_if.slave),
        .dma_tx_req(dma_tx_req),
        .dma_tx_ack(dma_tx_ack),
        .dma_tx_addr(dma_tx_addr),
        .dma_tx_length(dma_tx_length),
        .dma_tx_data(dma_tx_data),
        .dma_tx_valid(dma_tx_valid),
        .dma_tx_ready(dma_tx_ready),
        .dma_rx_req(dma_rx_req),
        .dma_rx_ack(dma_rx_ack),
        .dma_rx_addr(dma_rx_addr),
        .dma_rx_length(dma_rx_length),
        .dma_rx_data(dma_rx_data),
        .dma_rx_valid(dma_rx_valid),
        .dma_rx_ready(dma_rx_ready),
        .tx_complete_irq(tx_complete_irq),
        .rx_complete_irq(rx_complete_irq),
        .error_irq(error_irq),
        .link_up(link_up),
        .link_speed(link_speed),
        .full_duplex(full_duplex),
        .tx_packet_count(tx_packet_count),
        .rx_packet_count(rx_packet_count),
        .error_count(error_count)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // RGMII RX clock generation (simulated PHY)
    initial begin
        rgmii_rxc = 0;
        forever #(CLK_PERIOD/2) rgmii_rxc = ~rgmii_rxc;
    end
    
    // Test variables
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;
    
    // Test packet data
    logic [7:0] test_packet [0:63];
    integer packet_size;
    
    // Test tasks
    task reset_system();
        begin
            rst_n = 0;
            
            // Initialize DMA signals
            dma_tx_ack = 0;
            dma_tx_data = 0;
            dma_tx_valid = 0;
            dma_rx_ack = 0;
            dma_rx_ready = 1;
            
            // Initialize RGMII RX signals
            rgmii_rx_ctl = 0;
            rgmii_rxd = 0;
            
            // Initialize AXI signals
            axi_awaddr = 0;
            axi_awprot = 0;
            axi_awvalid = 0;
            axi_wdata = 0;
            axi_wstrb = 0;
            axi_wvalid = 0;
            axi_bready = 1;
            axi_araddr = 0;
            axi_arprot = 0;
            axi_arvalid = 0;
            axi_rready = 1;
            
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
    
    task axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            axi_awaddr = addr;
            axi_awvalid = 1;
            axi_wdata = data;
            axi_wstrb = 4'hF;
            axi_wvalid = 1;
            
            wait(axi_awready && axi_wready);
            @(posedge clk);
            axi_awvalid = 0;
            axi_wvalid = 0;
            
            wait(axi_bvalid);
            @(posedge clk);
        end
    endtask
    
    task axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            axi_araddr = addr;
            axi_arvalid = 1;
            
            wait(axi_arready);
            @(posedge clk);
            axi_arvalid = 0;
            
            wait(axi_rvalid);
            data = axi_rdata;
            @(posedge clk);
        end
    endtask
    
    task create_test_packet();
        begin
            // Create a simple Ethernet frame
            // Destination MAC: FF:FF:FF:FF:FF:FF (broadcast)
            test_packet[0] = 8'hFF; test_packet[1] = 8'hFF; test_packet[2] = 8'hFF;
            test_packet[3] = 8'hFF; test_packet[4] = 8'hFF; test_packet[5] = 8'hFF;
            
            // Source MAC: 12:34:56:78:9A:BC
            test_packet[6] = 8'h12; test_packet[7] = 8'h34; test_packet[8] = 8'h56;
            test_packet[9] = 8'h78; test_packet[10] = 8'h9A; test_packet[11] = 8'hBC;
            
            // EtherType: 0x0800 (IPv4)
            test_packet[12] = 8'h08; test_packet[13] = 8'h00;
            
            // Simple payload
            for (int i = 14; i < 60; i++) begin
                test_packet[i] = i[7:0];
            end
            
            packet_size = 60;  // Minimum Ethernet frame size
        end
    endtask
    
    task send_rgmii_packet();
        integer i, j;
        begin
            $display("Sending RGMII packet...");
            
            // Send preamble
            rgmii_rx_ctl = 1;
            for (i = 0; i < 7; i++) begin
                @(posedge rgmii_rxc);
                rgmii_rxd = 4'h5;
                @(posedge rgmii_rxc);
                rgmii_rxd = 4'h5;
            end
            
            // Send SFD
            @(posedge rgmii_rxc);
            rgmii_rxd = 4'h5;
            @(posedge rgmii_rxc);
            rgmii_rxd = 4'hD;
            
            // Send packet data
            for (i = 0; i < packet_size; i++) begin
                @(posedge rgmii_rxc);
                rgmii_rxd = test_packet[i][3:0];  // Lower nibble
                @(posedge rgmii_rxc);
                rgmii_rxd = test_packet[i][7:4];  // Upper nibble
            end
            
            // End of frame
            @(posedge rgmii_rxc);
            rgmii_rx_ctl = 0;
            rgmii_rxd = 0;
        end
    endtask
    
    task test_configuration();
        logic [31:0] read_data;
        begin
            $display("Testing Ethernet Configuration...");
            
            // Test MAC address configuration
            axi_write(32'h00, 32'h12345678);  // MAC low
            axi_write(32'h04, 32'h00009ABC);  // MAC high
            
            axi_read(32'h00, read_data);
            check_result("MAC Address Low", 32'h12345678, read_data);
            
            axi_read(32'h04, read_data);
            check_result("MAC Address High", 32'h00009ABC, read_data);
            
            // Test control register
            axi_write(32'h08, 32'h00000007);  // Enable all modes
            axi_read(32'h08, read_data);
            check_result("Control Register", 32'h00000007, read_data);
        end
    endtask
    
    task test_link_status();
        logic [31:0] read_data;
        begin
            $display("Testing Link Status...");
            
            check_result("Link Up", 1'b1, link_up);
            check_result("Link Speed", 2'b10, link_speed);  // 1000 Mbps
            check_result("Full Duplex", 1'b1, full_duplex);
            
            // Read status via AXI
            axi_read(32'h2C, read_data);
            check_result("Status Register", 32'h00000006, read_data);  // Full duplex + 1000M
        end
    endtask
    
    task test_packet_transmission();
        begin
            $display("Testing Packet Transmission...");
            
            create_test_packet();
            
            // Setup DMA for transmission
            dma_tx_ack = 1;
            dma_tx_length = packet_size;
            
            // Wait for DMA request
            wait(dma_tx_req);
            check_result("TX DMA Request", 1'b1, dma_tx_req);
            
            // Provide packet data
            for (int i = 0; i < packet_size; i += 8) begin
                @(posedge clk);
                dma_tx_data = {test_packet[i+7], test_packet[i+6], test_packet[i+5], test_packet[i+4],
                              test_packet[i+3], test_packet[i+2], test_packet[i+1], test_packet[i]};
                dma_tx_valid = 1;
                wait(dma_tx_ready);
                @(posedge clk);
                dma_tx_valid = 0;
            end
            
            // Wait for transmission complete
            wait(tx_complete_irq);
            check_result("TX Complete IRQ", 1'b1, tx_complete_irq);
            
            // Check packet count
            repeat(10) @(posedge clk);
            check_result("TX Packet Count", 32'h00000001, tx_packet_count);
        end
    endtask
    
    task test_packet_reception();
        begin
            $display("Testing Packet Reception...");
            
            create_test_packet();
            
            // Setup DMA for reception
            dma_rx_ack = 1;
            
            // Send packet via RGMII
            fork
                send_rgmii_packet();
            join_none
            
            // Wait for RX DMA request
            wait(dma_rx_req);
            check_result("RX DMA Request", 1'b1, dma_rx_req);
            
            // Receive packet data
            while (dma_rx_valid) begin
                @(posedge clk);
                dma_rx_ready = 1;
                // In real test, would verify received data
            end
            
            // Wait for reception complete
            wait(rx_complete_irq);
            check_result("RX Complete IRQ", 1'b1, rx_complete_irq);
            
            // Check packet count
            repeat(10) @(posedge clk);
            check_result("RX Packet Count", 32'h00000001, rx_packet_count);
        end
    endtask
    
    task test_statistics();
        logic [31:0] read_data;
        begin
            $display("Testing Statistics Counters...");
            
            // Read TX packet count
            axi_read(32'h20, read_data);
            $display("TX Packet Count: %d", read_data);
            
            // Read RX packet count
            axi_read(32'h24, read_data);
            $display("RX Packet Count: %d", read_data);
            
            // Read error count
            axi_read(32'h28, read_data);
            check_result("Error Count", 32'h00000000, read_data);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Ethernet Controller Tests...");
        
        reset_system();
        
        test_configuration();
        test_link_status();
        test_packet_transmission();
        test_packet_reception();
        test_statistics();
        
        // Test summary
        $display("\n=== Ethernet Controller Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All Ethernet Controller tests PASSED!");
        end else begin
            $display("Some Ethernet Controller tests FAILED!");
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
        $dumpfile("test_ethernet_controller.vcd");
        $dumpvars(0, test_ethernet_controller);
    end

endmodule