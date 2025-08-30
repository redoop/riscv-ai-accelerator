// Storage Controllers Test Module
// Tests SATA and NVMe controller functionality

`timescale 1ns/1ps

module test_storage_controllers;

    // Test parameters
    parameter DATA_WIDTH = 32;
    parameter ADDR_WIDTH = 64;
    parameter MAX_PORTS = 4;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // SATA interface
    logic [MAX_PORTS-1:0] sata_tx_p, sata_tx_n;
    logic [MAX_PORTS-1:0] sata_rx_p, sata_rx_n;
    logic        sata_refclk_p, sata_refclk_n;
    
    // NVMe PCIe interface
    logic [63:0] pcie_rx_data, pcie_tx_data;
    logic        pcie_rx_valid, pcie_rx_ready;
    logic        pcie_tx_valid, pcie_tx_ready;
    
    // AXI interface signals for SATA
    logic [31:0] sata_axi_awaddr, sata_axi_wdata, sata_axi_araddr, sata_axi_rdata;
    logic [2:0]  sata_axi_awprot, sata_axi_arprot;
    logic        sata_axi_awvalid, sata_axi_awready, sata_axi_wvalid, sata_axi_wready;
    logic [3:0]  sata_axi_wstrb;
    logic [1:0]  sata_axi_bresp, sata_axi_rresp;
    logic        sata_axi_bvalid, sata_axi_bready, sata_axi_arvalid, sata_axi_arready;
    logic        sata_axi_rvalid, sata_axi_rready;
    
    // DMA interface for SATA
    logic                sata_dma_req, sata_dma_ack;
    logic [ADDR_WIDTH-1:0] sata_dma_addr;
    logic [15:0]         sata_dma_length;
    logic                sata_dma_write;
    logic [DATA_WIDTH-1:0] sata_dma_wdata, sata_dma_rdata;
    logic                sata_dma_valid, sata_dma_ready;
    
    // SATA status and interrupts
    logic [MAX_PORTS-1:0] sata_port_irq;
    logic        sata_global_irq;
    logic [MAX_PORTS-1:0] sata_port_present, sata_port_active;
    logic [1:0]  sata_link_speed [MAX_PORTS-1:0];
    logic [31:0] sata_command_count, sata_error_count;
    
    // NVMe memory interface
    logic                nvme_mem_req;
    logic [ADDR_WIDTH-1:0] nvme_mem_addr;
    logic [63:0]         nvme_mem_wdata, nvme_mem_rdata;
    logic                nvme_mem_write;
    logic [7:0]          nvme_mem_size;
    logic                nvme_mem_ready, nvme_mem_valid;
    
    // NVMe DMA interface
    logic                nvme_dma_req, nvme_dma_ack;
    logic [ADDR_WIDTH-1:0] nvme_dma_src_addr, nvme_dma_dst_addr;
    logic [31:0]         nvme_dma_length;
    logic                nvme_dma_write, nvme_dma_done, nvme_dma_error;
    
    // NVMe status and interrupts
    logic [15:0]         nvme_msi_vector;
    logic                nvme_msi_valid, nvme_msi_ready;
    logic                nvme_controller_ready;
    logic [15:0]         nvme_active_queues;
    logic [31:0]         nvme_commands_processed, nvme_error_count;
    logic [63:0]         nvme_data_transferred;
    
    // AXI interface instances
    axi4_if sata_axi_if (
        .aclk(clk),
        .aresetn(rst_n)
    );
    
    // Connect SATA AXI signals
    assign sata_axi_if.awaddr = sata_axi_awaddr;
    assign sata_axi_if.awprot = sata_axi_awprot;
    assign sata_axi_if.awvalid = sata_axi_awvalid;
    assign sata_axi_awready = sata_axi_if.awready;
    assign sata_axi_if.wdata = sata_axi_wdata;
    assign sata_axi_if.wstrb = sata_axi_wstrb;
    assign sata_axi_if.wvalid = sata_axi_wvalid;
    assign sata_axi_wready = sata_axi_if.wready;
    assign sata_axi_bresp = sata_axi_if.bresp;
    assign sata_axi_bvalid = sata_axi_if.bvalid;
    assign sata_axi_if.bready = sata_axi_bready;
    assign sata_axi_if.araddr = sata_axi_araddr;
    assign sata_axi_if.arprot = sata_axi_arprot;
    assign sata_axi_if.arvalid = sata_axi_arvalid;
    assign sata_axi_arready = sata_axi_if.arready;
    assign sata_axi_rdata = sata_axi_if.rdata;
    assign sata_axi_rresp = sata_axi_if.rresp;
    assign sata_axi_rvalid = sata_axi_if.rvalid;
    assign sata_axi_if.rready = sata_axi_rready;
    
    // SATA Controller DUT
    sata_controller #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_PORTS(MAX_PORTS)
    ) sata_dut (
        .clk(clk),
        .rst_n(rst_n),
        .sata_tx_p(sata_tx_p),
        .sata_tx_n(sata_tx_n),
        .sata_rx_p(sata_rx_p),
        .sata_rx_n(sata_rx_n),
        .sata_refclk_p(sata_refclk_p),
        .sata_refclk_n(sata_refclk_n),
        .axi_if(sata_axi_if.slave),
        .dma_req(sata_dma_req),
        .dma_ack(sata_dma_ack),
        .dma_addr(sata_dma_addr),
        .dma_length(sata_dma_length),
        .dma_write(sata_dma_write),
        .dma_wdata(sata_dma_wdata),
        .dma_rdata(sata_dma_rdata),
        .dma_valid(sata_dma_valid),
        .dma_ready(sata_dma_ready),
        .port_irq(sata_port_irq),
        .global_irq(sata_global_irq),
        .port_present(sata_port_present),
        .port_active(sata_port_active),
        .link_speed(sata_link_speed),
        .command_count(sata_command_count),
        .error_count(sata_error_count)
    );
    
    // NVMe Controller DUT
    nvme_controller #(
        .DATA_WIDTH(64),
        .ADDR_WIDTH(ADDR_WIDTH),
        .MAX_QUEUES(64),
        .QUEUE_DEPTH(1024),
        .MAX_NAMESPACES(16)
    ) nvme_dut (
        .clk(clk),
        .rst_n(rst_n),
        .pcie_rx_data(pcie_rx_data),
        .pcie_rx_valid(pcie_rx_valid),
        .pcie_rx_ready(pcie_rx_ready),
        .pcie_tx_data(pcie_tx_data),
        .pcie_tx_valid(pcie_tx_valid),
        .pcie_tx_ready(pcie_tx_ready),
        .mem_req(nvme_mem_req),
        .mem_addr(nvme_mem_addr),
        .mem_wdata(nvme_mem_wdata),
        .mem_rdata(nvme_mem_rdata),
        .mem_write(nvme_mem_write),
        .mem_size(nvme_mem_size),
        .mem_ready(nvme_mem_ready),
        .mem_valid(nvme_mem_valid),
        .dma_req(nvme_dma_req),
        .dma_ack(nvme_dma_ack),
        .dma_src_addr(nvme_dma_src_addr),
        .dma_dst_addr(nvme_dma_dst_addr),
        .dma_length(nvme_dma_length),
        .dma_write(nvme_dma_write),
        .dma_done(nvme_dma_done),
        .dma_error(nvme_dma_error),
        .msi_vector(nvme_msi_vector),
        .msi_valid(nvme_msi_valid),
        .msi_ready(nvme_msi_ready),
        .controller_ready(nvme_controller_ready),
        .active_queues(nvme_active_queues),
        .commands_processed(nvme_commands_processed),
        .error_count(nvme_error_count),
        .data_transferred(nvme_data_transferred)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // SATA reference clock generation
    initial begin
        sata_refclk_p = 0;
        sata_refclk_n = 1;
        forever begin
            #(CLK_PERIOD/4);
            sata_refclk_p = ~sata_refclk_p;
            sata_refclk_n = ~sata_refclk_n;
        end
    end
    
    // Test variables
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;
    
    // Test tasks
    task reset_system();
        begin
            rst_n = 0;
            
            // Initialize SATA signals
            sata_rx_p = '0;
            sata_rx_n = '0;
            sata_dma_ack = 1;
            sata_dma_rdata = 0;
            sata_dma_valid = 0;
            
            // Initialize SATA AXI signals
            sata_axi_awaddr = 0;
            sata_axi_awprot = 0;
            sata_axi_awvalid = 0;
            sata_axi_wdata = 0;
            sata_axi_wstrb = 0;
            sata_axi_wvalid = 0;
            sata_axi_bready = 1;
            sata_axi_araddr = 0;
            sata_axi_arprot = 0;
            sata_axi_arvalid = 0;
            sata_axi_rready = 1;
            
            // Initialize NVMe signals
            pcie_rx_data = 0;
            pcie_rx_valid = 0;
            pcie_tx_ready = 1;
            nvme_mem_rdata = 0;
            nvme_mem_ready = 1;
            nvme_mem_valid = 0;
            nvme_dma_ack = 1;
            nvme_dma_done = 0;
            nvme_dma_error = 0;
            nvme_msi_ready = 1;
            
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
    
    task sata_axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            sata_axi_awaddr = addr;
            sata_axi_awvalid = 1;
            sata_axi_wdata = data;
            sata_axi_wstrb = 4'hF;
            sata_axi_wvalid = 1;
            
            wait(sata_axi_awready && sata_axi_wready);
            @(posedge clk);
            sata_axi_awvalid = 0;
            sata_axi_wvalid = 0;
            
            wait(sata_axi_bvalid);
            @(posedge clk);
        end
    endtask
    
    task sata_axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            sata_axi_araddr = addr;
            sata_axi_arvalid = 1;
            
            wait(sata_axi_arready);
            @(posedge clk);
            sata_axi_arvalid = 0;
            
            wait(sata_axi_rvalid);
            data = sata_axi_rdata;
            @(posedge clk);
        end
    endtask
    
    task test_sata_initialization();
        logic [31:0] read_data;
        begin
            $display("Testing SATA Controller Initialization...");
            
            // Read capabilities register
            sata_axi_read(32'h000, read_data);
            $display("SATA Capabilities: %h", read_data);
            
            // Check ports implemented
            sata_axi_read(32'h00C, read_data);
            check_result("SATA Ports Implemented", 32'h0000000F, read_data);  // 4 ports
            
            // Check version
            sata_axi_read(32'h010, read_data);
            check_result("SATA Version", 32'h00010300, read_data);  // AHCI 1.3
            
            // Check port status
            check_result("Port 0 Present", 1'b1, sata_port_present[0]);
            check_result("Port 0 Active", 1'b1, sata_port_active[0]);
            check_result("Link Speed", 2'b10, sata_link_speed[0]);  // 6 Gbps
        end
    endtask
    
    task test_sata_command_processing();
        logic [31:0] read_data;
        begin
            $display("Testing SATA Command Processing...");
            
            // Configure port 0
            sata_axi_write(32'h100, 32'h10000000);  // Command List Base
            sata_axi_write(32'h108, 32'h20000000);  // FIS Base
            sata_axi_write(32'h114, 32'h00000001);  // Interrupt Enable
            sata_axi_write(32'h118, 32'h00000001);  // Start command processing
            
            // Issue a command
            sata_axi_write(32'h138, 32'h00000001);  // Command Issue (slot 0)
            
            // Wait for DMA request
            wait(sata_dma_req);
            check_result("SATA DMA Request", 1'b1, sata_dma_req);
            
            // Provide DMA data
            @(posedge clk);
            sata_dma_valid = 1;
            sata_dma_rdata = 32'hDEADBEEF;
            
            repeat(10) @(posedge clk);
            sata_dma_valid = 0;
            
            // Wait for command completion
            repeat(50) @(posedge clk);
            
            // Check statistics
            check_result("SATA Command Count", 32'h1, sata_command_count);
        end
    endtask
    
    task test_nvme_initialization();
        begin
            $display("Testing NVMe Controller Initialization...");
            
            // Send controller configuration via PCIe
            @(posedge clk);
            pcie_rx_data = {32'h14, 32'h00000001};  // Enable controller
            pcie_rx_valid = 1;
            
            wait(pcie_rx_ready);
            @(posedge clk);
            pcie_rx_valid = 0;
            
            // Wait for controller to become ready
            repeat(50) @(posedge clk);
            
            check_result("NVMe Controller Ready", 1'b1, nvme_controller_ready);
            check_result("NVMe Active Queues", 16'h1, nvme_active_queues);  // Admin queue
        end
    endtask
    
    task test_nvme_command_processing();
        begin
            $display("Testing NVMe Command Processing...");
            
            // Simulate memory interface for command fetching
            fork
                begin
                    wait(nvme_mem_req);
                    repeat(5) @(posedge clk);
                    nvme_mem_valid = 1;
                    @(posedge clk);
                    nvme_mem_valid = 0;
                end
            join_none
            
            // Wait for DMA request (data transfer)
            wait(nvme_dma_req);
            check_result("NVMe DMA Request", 1'b1, nvme_dma_req);
            
            // Complete DMA transfer
            @(posedge clk);
            nvme_dma_done = 1;
            @(posedge clk);
            nvme_dma_done = 0;
            
            // Wait for MSI interrupt
            wait(nvme_msi_valid);
            check_result("NVMe MSI Valid", 1'b1, nvme_msi_valid);
            
            repeat(20) @(posedge clk);
            
            // Check statistics
            check_result("NVMe Commands Processed", 32'h1, nvme_commands_processed);
        end
    endtask
    
    task test_storage_performance();
        begin
            $display("Testing Storage Performance...");
            
            // Test multiple SATA commands
            for (int i = 0; i < 5; i++) begin
                sata_axi_write(32'h138, 32'h00000001);  // Issue command
                repeat(20) @(posedge clk);
            end
            
            // Test NVMe data transfer
            repeat(100) @(posedge clk);
            
            $display("SATA Commands: %d", sata_command_count);
            $display("NVMe Commands: %d", nvme_commands_processed);
            $display("NVMe Data Transferred: %d bytes", nvme_data_transferred);
        end
    endtask
    
    task test_error_handling();
        begin
            $display("Testing Storage Error Handling...");
            
            // Simulate NVMe DMA error
            @(posedge clk);
            nvme_dma_error = 1;
            @(posedge clk);
            nvme_dma_error = 0;
            
            repeat(20) @(posedge clk);
            
            // Check error count
            if (nvme_error_count > 0) begin
                $display("NVMe Error Count: %d", nvme_error_count);
            end
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting Storage Controllers Tests...");
        
        reset_system();
        
        test_sata_initialization();
        test_nvme_initialization();
        test_sata_command_processing();
        test_nvme_command_processing();
        test_storage_performance();
        test_error_handling();
        
        // Test summary
        $display("\n=== Storage Controllers Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All Storage Controller tests PASSED!");
        end else begin
            $display("Some Storage Controller tests FAILED!");
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
        $dumpfile("test_storage_controllers.vcd");
        $dumpvars(0, test_storage_controllers);
    end

endmodule