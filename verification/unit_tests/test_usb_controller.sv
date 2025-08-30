// USB Controller Test Module
// Tests USB 3.1 host controller functionality

`timescale 1ns/1ps

module test_usb_controller;

    // Test parameters
    parameter DATA_WIDTH = 64;
    parameter ADDR_WIDTH = 32;
    parameter MAX_DEVICES = 127;
    parameter MAX_ENDPOINTS = 16;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // USB interface
    logic        usb_ss_tx_p, usb_ss_tx_n;
    logic        usb_ss_rx_p, usb_ss_rx_n;
    logic        usb_hs_dp, usb_hs_dm;
    logic        usb_hs_dp_in, usb_hs_dm_in;
    logic        usb_vbus_en, usb_overcurrent, usb_reset;
    
    // AXI interface signals
    logic [31:0] axi_awaddr, axi_wdata, axi_araddr, axi_rdata;
    logic [2:0]  axi_awprot, axi_arprot;
    logic        axi_awvalid, axi_awready, axi_wvalid, axi_wready;
    logic [3:0]  axi_wstrb;
    logic [1:0]  axi_bresp, axi_rresp;
    logic        axi_bvalid, axi_bready, axi_arvalid, axi_arready;
    logic        axi_rvalid, axi_rready;
    
    // DMA interface
    logic                dma_req, dma_ack;
    logic [ADDR_WIDTH-1:0] dma_addr;
    logic [15:0]         dma_length;
    logic                dma_write;
    logic [DATA_WIDTH-1:0] dma_wdata, dma_rdata;
    logic                dma_valid, dma_ready;
    
    // Status and interrupts
    logic        port_change_irq, transfer_complete_irq, error_irq;
    logic [3:0]  port_count;
    logic [7:0]  device_count;
    logic [31:0] transfer_count, error_count;
    
    // AXI interface instance
    axi4_if axi_if (
        .aclk(clk),
        .aresetn(rst_n)
    );
    
    // Connect AXI signals
    assign axi_if.awaddr = axi_awaddr;
    assign axi_if.awprot = axi_awprot;
    assign axi_if.awvalid = axi_awvalid;
    assign axi_awready = axi_if.awready;
    assign axi_if.wdata = axi_wdata;
    assign axi_if.wstrb = axi_wstrb;
    assign axi_if.wvalid = axi_wvalid;
    assign axi_wready = axi_if.wready;
    assign axi_bresp = axi_if.bresp;
    assign axi_bvalid = axi_if.bvalid;
    assign axi_if.bready = axi_bready;
    assign axi_if.araddr = axi_araddr;
    assign axi_if.arprot = axi_arprot;
    assign axi_if.arvalid = axi_arvalid;
    assign axi_arready = axi_if.arready;
    assign axi_rdata = axi_if.rdata;
    assign axi_rresp = axi_if.rresp;
    assign axi_rvalid = axi_if.rvalid;
    assign axi_if.rready = axi_rready;
    
    // Device Under Test
    usb_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_DEVICES(MAX_DEVICES),
        .MAX_ENDPOINTS(MAX_ENDPOINTS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .usb_ss_tx_p(usb_ss_tx_p),
        .usb_ss_tx_n(usb_ss_tx_n),
        .usb_ss_rx_p(usb_ss_rx_p),
        .usb_ss_rx_n(usb_ss_rx_n),
        .usb_hs_dp(usb_hs_dp),
        .usb_hs_dm(usb_hs_dm),
        .usb_hs_dp_in(usb_hs_dp_in),
        .usb_hs_dm_in(usb_hs_dm_in),
        .usb_vbus_en(usb_vbus_en),
        .usb_overcurrent(usb_overcurrent),
        .usb_reset(usb_reset),
        .axi_if(axi_if.slave),
        .dma_req(dma_req),
        .dma_ack(dma_ack),
        .dma_addr(dma_addr),
        .dma_length(dma_length),
        .dma_write(dma_write),
        .dma_wdata(dma_wdata),
        .dma_rdata(dma_rdata),
        .dma_valid(dma_valid),
        .dma_ready(dma_ready),
        .port_change_irq(port_change_irq),
        .transfer_complete_irq(transfer_complete_irq),
        .error_irq(error_irq),
        .port_count(port_count),
        .device_count(device_count),
        .transfer_count(transfer_count),
        .error_count(error_count)
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
    
    // Test tasks
    task reset_system();
        begin
            rst_n = 0;
            
            // Initialize USB signals
            usb_ss_rx_p = 0;
            usb_ss_rx_n = 0;
            usb_hs_dp_in = 0;
            usb_hs_dm_in = 0;
            usb_overcurrent = 0;
            
            // Initialize DMA signals
            dma_ack = 1;
            dma_rdata = 0;
            dma_valid = 0;
            
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
    
    task test_controller_initialization();
        logic [31:0] read_data;
        begin
            $display("Testing USB Controller Initialization...");
            
            // Test initial state
            check_result("USB Reset Active", 1'b1, usb_reset);
            check_result("VBUS Disabled", 1'b0, usb_vbus_en);
            
            // Enable controller
            axi_write(32'h00, 32'h00000001);  // Set Run/Stop bit
            
            repeat(20) @(posedge clk);
            
            // Check status
            axi_read(32'h04, read_data);
            $display("USB Status: %h", read_data);
            
            check_result("VBUS Enabled", 1'b1, usb_vbus_en);
            check_result("Port Count", 4'h1, port_count);
        end
    endtask
    
    task test_device_enumeration();
        logic [31:0] read_data;
        begin
            $display("Testing Device Enumeration...");
            
            // Simulate device connection
            repeat(50) @(posedge clk);
            
            // Check device count
            axi_read(32'h20, read_data);
            check_result("Device Count", 8'h1, read_data[7:0]);
        end
    endtask
    
    task test_transfer_operations();
        begin
            $display("Testing Transfer Operations...");
            
            // Submit a transfer descriptor
            axi_write(32'h1000, 32'h00010001);  // Device 1, Endpoint 1, Bulk OUT
            
            // Wait for DMA request
            wait(dma_req);
            check_result("DMA Request", 1'b1, dma_req);
            
            // Provide DMA data
            @(posedge clk);
            dma_valid = 1;
            dma_rdata = 64'hDEADBEEFCAFEBABE;
            
            wait(dma_ready);
            @(posedge clk);
            dma_valid = 0;
            
            // Wait for transfer completion
            wait(transfer_complete_irq);
            check_result("Transfer Complete IRQ", 1'b1, transfer_complete_irq);
            
            repeat(10) @(posedge clk);
            check_result("Transfer Count", 32'h1, transfer_count);
        end
    endtask
    
    task test_error_handling();
        begin
            $display("Testing Error Handling...");
            
            // Simulate overcurrent condition
            @(posedge clk);
            usb_overcurrent = 1;
            
            repeat(10) @(posedge clk);
            
            // Check error response
            // In a real implementation, this would trigger error handling
            usb_overcurrent = 0;
        end
    endtask
    
    task test_power_management();
        logic [31:0] read_data;
        begin
            $display("Testing Power Management...");
            
            // Test suspend
            axi_write(32'h00, 32'h00000005);  // Set Run/Stop and Suspend bits
            
            repeat(20) @(posedge clk);
            
            // Test resume
            axi_write(32'h00, 32'h00000001);  // Clear Suspend bit
            
            repeat(20) @(posedge clk);
            
            axi_read(32'h04, read_data);
            $display("USB Status after resume: %h", read_data);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting USB Controller Tests...");
        
        reset_system();
        
        test_controller_initialization();
        test_device_enumeration();
        test_transfer_operations();
        test_error_handling();
        test_power_management();
        
        // Test summary
        $display("\n=== USB Controller Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All USB Controller tests PASSED!");
        end else begin
            $display("Some USB Controller tests FAILED!");
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
        $dumpfile("test_usb_controller.vcd");
        $dumpvars(0, test_usb_controller);
    end

endmodule