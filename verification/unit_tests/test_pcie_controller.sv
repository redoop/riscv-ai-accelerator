// PCIe Controller Test Module
// Comprehensive test for PCIe 4.0 controller functionality

`timescale 1ns/1ps

module test_pcie_controller;

    // Test parameters
    parameter LANES = 16;
    parameter GEN = 4;
    parameter DATA_WIDTH = 512;
    parameter ADDR_WIDTH = 64;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // PCIe physical interface
    logic [LANES-1:0]    pcie_rx_p;
    logic [LANES-1:0]    pcie_rx_n;
    logic [LANES-1:0]    pcie_tx_p;
    logic [LANES-1:0]    pcie_tx_n;
    
    // AXI interface
    logic [31:0]         axi_awaddr;
    logic [2:0]          axi_awprot;
    logic                axi_awvalid;
    logic                axi_awready;
    logic [31:0]         axi_wdata;
    logic [3:0]          axi_wstrb;
    logic                axi_wvalid;
    logic                axi_wready;
    logic [1:0]          axi_bresp;
    logic                axi_bvalid;
    logic                axi_bready;
    logic [31:0]         axi_araddr;
    logic [2:0]          axi_arprot;
    logic                axi_arvalid;
    logic                axi_arready;
    logic [31:0]         axi_rdata;
    logic [1:0]          axi_rresp;
    logic                axi_rvalid;
    logic                axi_rready;
    
    // DMA interface
    logic                dma_req_valid;
    logic                dma_req_ready;
    logic [ADDR_WIDTH-1:0] dma_src_addr;
    logic [ADDR_WIDTH-1:0] dma_dst_addr;
    logic [31:0]         dma_length;
    logic                dma_write;
    logic                dma_done;
    logic                dma_error;
    
    // Configuration and status
    logic [15:0]         device_id;
    logic [15:0]         vendor_id;
    logic                link_up;
    logic [3:0]          link_width;
    logic [2:0]          link_speed;
    
    // Interrupt interface
    logic [31:0]         msi_vector;
    logic                msi_valid;
    logic                msi_ready;
    
    // Error reporting
    logic                correctable_error;
    logic                uncorrectable_error;
    logic [15:0]         error_code;
    
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
    pcie_controller #(
        .LANES(LANES),
        .GEN(GEN),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pcie_rx_p(pcie_rx_p),
        .pcie_rx_n(pcie_rx_n),
        .pcie_tx_p(pcie_tx_p),
        .pcie_tx_n(pcie_tx_n),
        .axi_if(axi_if.slave),
        .dma_req_valid(dma_req_valid),
        .dma_req_ready(dma_req_ready),
        .dma_src_addr(dma_src_addr),
        .dma_dst_addr(dma_dst_addr),
        .dma_length(dma_length),
        .dma_write(dma_write),
        .dma_done(dma_done),
        .dma_error(dma_error),
        .device_id(device_id),
        .vendor_id(vendor_id),
        .link_up(link_up),
        .link_width(link_width),
        .link_speed(link_speed),
        .msi_vector(msi_vector),
        .msi_valid(msi_valid),
        .msi_ready(msi_ready),
        .correctable_error(correctable_error),
        .uncorrectable_error(uncorrectable_error),
        .error_code(error_code)
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
            device_id = 16'h1234;
            vendor_id = 16'h5678;
            dma_req_ready = 1;
            msi_ready = 1;
            dma_done = 0;
            dma_error = 0;
            
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
    
    task check_result(input string test_name, input logic expected, input logic actual);
        begin
            test_count++;
            if (expected === actual) begin
                $display("PASS: %s", test_name);
                pass_count++;
            end else begin
                $display("FAIL: %s - Expected: %b, Actual: %b", test_name, expected, actual);
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
    
    task test_link_training();
        begin
            $display("Testing PCIe Link Training...");
            
            // Wait for link training to complete
            repeat(100) @(posedge clk);
            
            check_result("Link Up", 1'b1, link_up);
            check_result("Link Width", 4'd16, link_width);
            check_result("Link Speed", 3'd4, link_speed);
        end
    endtask
    
    task test_configuration_space();
        logic [31:0] read_data;
        begin
            $display("Testing Configuration Space Access...");
            
            // Test device/vendor ID read
            axi_read(32'h00000000, read_data);
            check_result("Device/Vendor ID", {device_id, vendor_id}, read_data);
            
            // Test status/command register
            axi_read(32'h00000004, read_data);
            check_result("Status/Command", 32'h00100007, read_data);
            
            // Test class code
            axi_read(32'h00000008, read_data);
            check_result("Class Code", 32'h06040001, read_data);
        end
    endtask
    
    task test_dma_operations();
        begin
            $display("Testing DMA Operations...");
            
            // Simulate DMA request
            wait(dma_req_valid);
            check_result("DMA Request Valid", 1'b1, dma_req_valid);
            
            // Acknowledge DMA request
            @(posedge clk);
            dma_done = 1;
            @(posedge clk);
            dma_done = 0;
            
            // Check MSI interrupt generation
            wait(msi_valid);
            check_result("MSI Valid", 1'b1, msi_valid);
            check_result("MSI Vector", 32'h00000001, msi_vector);
        end
    endtask
    
    task test_error_handling();
        begin
            $display("Testing Error Handling...");
            
            // Simulate DMA error
            @(posedge clk);
            dma_error = 1;
            @(posedge clk);
            dma_error = 0;
            
            // Check error reporting
            repeat(10) @(posedge clk);
            // Error handling verification would go here
        end
    endtask
    
    task test_msi_interrupts();
        begin
            $display("Testing MSI Interrupts...");
            
            // Test MSI ready/valid handshake
            wait(msi_valid);
            check_result("MSI Valid", 1'b1, msi_valid);
            
            @(posedge clk);
            msi_ready = 0;
            @(posedge clk);
            msi_ready = 1;
            
            repeat(5) @(posedge clk);
            check_result("MSI Cleared", 1'b0, msi_valid);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting PCIe Controller Tests...");
        
        reset_system();
        
        test_link_training();
        test_configuration_space();
        test_dma_operations();
        test_error_handling();
        test_msi_interrupts();
        
        // Test summary
        $display("\n=== PCIe Controller Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All PCIe Controller tests PASSED!");
        end else begin
            $display("Some PCIe Controller tests FAILED!");
        end
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #1000000;  // 1ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end
    
    // Waveform dumping
    initial begin
        $dumpfile("test_pcie_controller.vcd");
        $dumpvars(0, test_pcie_controller);
    end

endmodule

// Simple AXI4 interface for testing
interface axi4_if (
    input logic aclk,
    input logic aresetn
);
    
    // Write address channel
    logic [31:0] awaddr;
    logic [2:0]  awprot;
    logic        awvalid;
    logic        awready;
    
    // Write data channel
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic        wvalid;
    logic        wready;
    
    // Write response channel
    logic [1:0]  bresp;
    logic        bvalid;
    logic        bready;
    
    // Read address channel
    logic [31:0] araddr;
    logic [2:0]  arprot;
    logic        arvalid;
    logic        arready;
    
    // Read data channel
    logic [31:0] rdata;
    logic [1:0]  rresp;
    logic        rvalid;
    logic        rready;
    
    modport master (
        output awaddr, awprot, awvalid,
        input  awready,
        output wdata, wstrb, wvalid,
        input  wready,
        input  bresp, bvalid,
        output bready,
        output araddr, arprot, arvalid,
        input  arready,
        input  rdata, rresp, rvalid,
        output rready
    );
    
    modport slave (
        input  awaddr, awprot, awvalid,
        output awready,
        input  wdata, wstrb, wvalid,
        output wready,
        output bresp, bvalid,
        input  bready,
        input  araddr, arprot, arvalid,
        output arready,
        output rdata, rresp, rvalid,
        input  rready
    );
    
endinterface