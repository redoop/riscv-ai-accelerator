// GPIO and SPI Controllers Test Module
// Tests GPIO and SPI controller functionality

`timescale 1ns/1ps

module test_gpio_spi_controllers;

    // Test parameters
    parameter NUM_GPIO_PINS = 32;
    parameter MAX_SPI_SLAVES = 8;
    parameter SPI_FIFO_DEPTH = 16;
    parameter CLK_PERIOD = 10;  // 100MHz
    
    // Test signals
    logic        clk;
    logic        rst_n;
    
    // GPIO signals
    wire [NUM_GPIO_PINS-1:0] gpio_pins;
    logic [NUM_GPIO_PINS-1:0] gpio_pins_drive;
    logic [NUM_GPIO_PINS-1:0] gpio_pins_oe;
    logic        gpio_irq;
    logic [NUM_GPIO_PINS-1:0] gpio_pin_status;
    
    // SPI signals
    logic        spi_sclk;
    logic        spi_mosi;
    logic        spi_miso;
    logic [MAX_SPI_SLAVES-1:0] spi_cs_n;
    logic        spi_busy;
    logic [7:0]  spi_tx_fifo_level, spi_rx_fifo_level;
    logic        spi_tx_complete_irq, spi_rx_complete_irq, spi_error_irq;
    
    // AXI interface signals for GPIO
    logic [31:0] gpio_axi_awaddr, gpio_axi_wdata, gpio_axi_araddr, gpio_axi_rdata;
    logic [2:0]  gpio_axi_awprot, gpio_axi_arprot;
    logic        gpio_axi_awvalid, gpio_axi_awready, gpio_axi_wvalid, gpio_axi_wready;
    logic [3:0]  gpio_axi_wstrb;
    logic [1:0]  gpio_axi_bresp, gpio_axi_rresp;
    logic        gpio_axi_bvalid, gpio_axi_bready, gpio_axi_arvalid, gpio_axi_arready;
    logic        gpio_axi_rvalid, gpio_axi_rready;
    
    // AXI interface signals for SPI
    logic [31:0] spi_axi_awaddr, spi_axi_wdata, spi_axi_araddr, spi_axi_rdata;
    logic [2:0]  spi_axi_awprot, spi_axi_arprot;
    logic        spi_axi_awvalid, spi_axi_awready, spi_axi_wvalid, spi_axi_wready;
    logic [3:0]  spi_axi_wstrb;
    logic [1:0]  spi_axi_bresp, spi_axi_rresp;
    logic        spi_axi_bvalid, spi_axi_bready, spi_axi_arvalid, spi_axi_arready;
    logic        spi_axi_rvalid, spi_axi_rready;
    
    // AXI interface instances
    axi4_if gpio_axi_if (
        .aclk(clk),
        .aresetn(rst_n)
    );
    
    axi4_if spi_axi_if (
        .aclk(clk),
        .aresetn(rst_n)
    );
    
    // Connect GPIO AXI signals
    assign gpio_axi_if.awaddr = gpio_axi_awaddr;
    assign gpio_axi_if.awprot = gpio_axi_awprot;
    assign gpio_axi_if.awvalid = gpio_axi_awvalid;
    assign gpio_axi_awready = gpio_axi_if.awready;
    assign gpio_axi_if.wdata = gpio_axi_wdata;
    assign gpio_axi_if.wstrb = gpio_axi_wstrb;
    assign gpio_axi_if.wvalid = gpio_axi_wvalid;
    assign gpio_axi_wready = gpio_axi_if.wready;
    assign gpio_axi_bresp = gpio_axi_if.bresp;
    assign gpio_axi_bvalid = gpio_axi_if.bvalid;
    assign gpio_axi_if.bready = gpio_axi_bready;
    assign gpio_axi_if.araddr = gpio_axi_araddr;
    assign gpio_axi_if.arprot = gpio_axi_arprot;
    assign gpio_axi_if.arvalid = gpio_axi_arvalid;
    assign gpio_axi_arready = gpio_axi_if.arready;
    assign gpio_axi_rdata = gpio_axi_if.rdata;
    assign gpio_axi_rresp = gpio_axi_if.rresp;
    assign gpio_axi_rvalid = gpio_axi_if.rvalid;
    assign gpio_axi_if.rready = gpio_axi_rready;
    
    // Connect SPI AXI signals
    assign spi_axi_if.awaddr = spi_axi_awaddr;
    assign spi_axi_if.awprot = spi_axi_awprot;
    assign spi_axi_if.awvalid = spi_axi_awvalid;
    assign spi_axi_awready = spi_axi_if.awready;
    assign spi_axi_if.wdata = spi_axi_wdata;
    assign spi_axi_if.wstrb = spi_axi_wstrb;
    assign spi_axi_if.wvalid = spi_axi_wvalid;
    assign spi_axi_wready = spi_axi_if.wready;
    assign spi_axi_bresp = spi_axi_if.bresp;
    assign spi_axi_bvalid = spi_axi_if.bvalid;
    assign spi_axi_if.bready = spi_axi_bready;
    assign spi_axi_if.araddr = spi_axi_araddr;
    assign spi_axi_if.arprot = spi_axi_arprot;
    assign spi_axi_if.arvalid = spi_axi_arvalid;
    assign spi_axi_arready = spi_axi_if.arready;
    assign spi_axi_rdata = spi_axi_if.rdata;
    assign spi_axi_rresp = spi_axi_if.rresp;
    assign spi_axi_rvalid = spi_axi_if.rvalid;
    assign spi_axi_if.rready = spi_axi_rready;
    
    // GPIO pin simulation
    genvar i;
    generate
        for (i = 0; i < NUM_GPIO_PINS; i++) begin : gpio_sim
            assign gpio_pins[i] = gpio_pins_oe[i] ? gpio_pins_drive[i] : 1'bz;
        end
    endgenerate
    
    // GPIO Controller DUT
    gpio_controller #(
        .NUM_PINS(NUM_GPIO_PINS),
        .DATA_WIDTH(32)
    ) gpio_dut (
        .clk(clk),
        .rst_n(rst_n),
        .gpio_pins(gpio_pins),
        .axi_if(gpio_axi_if.slave),
        .gpio_irq(gpio_irq),
        .pin_status(gpio_pin_status)
    );
    
    // SPI Controller DUT
    spi_controller #(
        .DATA_WIDTH(32),
        .MAX_SLAVES(MAX_SPI_SLAVES),
        .FIFO_DEPTH(SPI_FIFO_DEPTH)
    ) spi_dut (
        .clk(clk),
        .rst_n(rst_n),
        .spi_sclk(spi_sclk),
        .spi_mosi(spi_mosi),
        .spi_miso(spi_miso),
        .spi_cs_n(spi_cs_n),
        .axi_if(spi_axi_if.slave),
        .tx_complete_irq(spi_tx_complete_irq),
        .rx_complete_irq(spi_rx_complete_irq),
        .error_irq(spi_error_irq),
        .spi_busy(spi_busy),
        .tx_fifo_level(spi_tx_fifo_level),
        .rx_fifo_level(spi_rx_fifo_level)
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
            
            // Initialize GPIO simulation signals
            gpio_pins_drive = '0;
            gpio_pins_oe = '0;
            
            // Initialize SPI simulation signals
            spi_miso = 0;
            
            // Initialize GPIO AXI signals
            gpio_axi_awaddr = 0;
            gpio_axi_awprot = 0;
            gpio_axi_awvalid = 0;
            gpio_axi_wdata = 0;
            gpio_axi_wstrb = 0;
            gpio_axi_wvalid = 0;
            gpio_axi_bready = 1;
            gpio_axi_araddr = 0;
            gpio_axi_arprot = 0;
            gpio_axi_arvalid = 0;
            gpio_axi_rready = 1;
            
            // Initialize SPI AXI signals
            spi_axi_awaddr = 0;
            spi_axi_awprot = 0;
            spi_axi_awvalid = 0;
            spi_axi_wdata = 0;
            spi_axi_wstrb = 0;
            spi_axi_wvalid = 0;
            spi_axi_bready = 1;
            spi_axi_araddr = 0;
            spi_axi_arprot = 0;
            spi_axi_arvalid = 0;
            spi_axi_rready = 1;
            
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
    
    task gpio_axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            gpio_axi_awaddr = addr;
            gpio_axi_awvalid = 1;
            gpio_axi_wdata = data;
            gpio_axi_wstrb = 4'hF;
            gpio_axi_wvalid = 1;
            
            wait(gpio_axi_awready && gpio_axi_wready);
            @(posedge clk);
            gpio_axi_awvalid = 0;
            gpio_axi_wvalid = 0;
            
            wait(gpio_axi_bvalid);
            @(posedge clk);
        end
    endtask
    
    task gpio_axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            gpio_axi_araddr = addr;
            gpio_axi_arvalid = 1;
            
            wait(gpio_axi_arready);
            @(posedge clk);
            gpio_axi_arvalid = 0;
            
            wait(gpio_axi_rvalid);
            data = gpio_axi_rdata;
            @(posedge clk);
        end
    endtask
    
    task spi_axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            spi_axi_awaddr = addr;
            spi_axi_awvalid = 1;
            spi_axi_wdata = data;
            spi_axi_wstrb = 4'hF;
            spi_axi_wvalid = 1;
            
            wait(spi_axi_awready && spi_axi_wready);
            @(posedge clk);
            spi_axi_awvalid = 0;
            spi_axi_wvalid = 0;
            
            wait(spi_axi_bvalid);
            @(posedge clk);
        end
    endtask
    
    task spi_axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            spi_axi_araddr = addr;
            spi_axi_arvalid = 1;
            
            wait(spi_axi_arready);
            @(posedge clk);
            spi_axi_arvalid = 0;
            
            wait(spi_axi_rvalid);
            data = spi_axi_rdata;
            @(posedge clk);
        end
    endtask
    
    task test_gpio_basic_io();
        logic [31:0] read_data;
        begin
            $display("Testing GPIO Basic I/O...");
            
            // Configure pins 0-7 as outputs
            gpio_axi_write(32'h08, 32'h000000FF);  // Direction register
            
            // Write output data
            gpio_axi_write(32'h00, 32'h000000AA);  // Data out register
            
            repeat(5) @(posedge clk);
            
            // Read back output data
            gpio_axi_read(32'h04, read_data);  // Data in register
            check_result("GPIO Output Data", 32'h000000AA, read_data & 32'h000000FF);
            
            // Configure pins 8-15 as inputs and simulate input
            gpio_pins_drive[15:8] = 8'h55;
            gpio_pins_oe[15:8] = 8'hFF;
            
            repeat(5) @(posedge clk);
            
            // Read input data
            gpio_axi_read(32'h04, read_data);
            check_result("GPIO Input Data", 32'h00005500, read_data & 32'h0000FF00);
        end
    endtask
    
    task test_gpio_interrupts();
        logic [31:0] read_data;
        begin
            $display("Testing GPIO Interrupts...");
            
            // Configure pin 16 for rising edge interrupt
            gpio_axi_write(32'h14, 32'h00010000);  // IRQ enable
            gpio_axi_write(32'h18, 32'h00010000);  // IRQ type (edge)
            gpio_axi_write(32'h1C, 32'h00010000);  // IRQ polarity (rising)
            
            // Simulate rising edge on pin 16
            gpio_pins_drive[16] = 1'b0;
            gpio_pins_oe[16] = 1'b1;
            repeat(5) @(posedge clk);
            
            gpio_pins_drive[16] = 1'b1;
            repeat(5) @(posedge clk);
            
            // Check interrupt status
            gpio_axi_read(32'h20, read_data);  // IRQ status
            check_result("GPIO IRQ Status", 32'h00010000, read_data & 32'h00010000);
            check_result("GPIO IRQ Signal", 1'b1, gpio_irq);
            
            // Clear interrupt
            gpio_axi_write(32'h20, 32'h00010000);  // Write 1 to clear
            
            repeat(5) @(posedge clk);
            check_result("GPIO IRQ Cleared", 1'b0, gpio_irq);
        end
    endtask
    
    task test_spi_configuration();
        logic [31:0] read_data;
        begin
            $display("Testing SPI Configuration...");
            
            // Configure SPI controller
            spi_axi_write(32'h00, 32'h00000403);  // Enable, Master, Mode 0, 8 bits
            spi_axi_write(32'h08, 32'h00000010);  // Clock divider
            spi_axi_write(32'h0C, 32'h00000001);  // Select slave 0
            
            // Read back configuration
            spi_axi_read(32'h00, read_data);
            check_result("SPI Control Register", 32'h00000403, read_data);
            
            spi_axi_read(32'h04, read_data);  // Status register
            $display("SPI Status: %h", read_data);
        end
    endtask
    
    task test_spi_data_transfer();
        logic [31:0] read_data;
        begin
            $display("Testing SPI Data Transfer...");
            
            // Send data via SPI
            spi_axi_write(32'h10, 32'h12345678);  // TX data
            
            // Simulate SPI slave response
            fork
                begin
                    // Wait for SPI clock and provide MISO data
                    wait(spi_sclk);
                    spi_miso = 1'b1;
                    wait(!spi_sclk);
                    spi_miso = 1'b0;
                    wait(spi_sclk);
                    spi_miso = 1'b1;
                    wait(!spi_sclk);
                    spi_miso = 1'b1;
                    // Continue pattern...
                end
            join_none
            
            // Wait for transfer completion
            wait(spi_tx_complete_irq);
            check_result("SPI TX Complete IRQ", 1'b1, spi_tx_complete_irq);
            
            // Check chip select
            check_result("SPI CS Active", 1'b0, spi_cs_n[0]);  // Active low
            
            repeat(20) @(posedge clk);
            
            // Read received data
            spi_axi_read(32'h14, read_data);  // RX data
            $display("SPI RX Data: %h", read_data);
        end
    endtask
    
    task test_spi_fifo_operations();
        logic [31:0] read_data;
        begin
            $display("Testing SPI FIFO Operations...");
            
            // Fill TX FIFO
            for (int i = 0; i < 4; i++) begin
                spi_axi_write(32'h10, 32'h11111111 + i);
            end
            
            // Check FIFO levels
            spi_axi_read(32'h18, read_data);
            $display("FIFO Levels - TX: %d, RX: %d", read_data[7:0], read_data[15:8]);
            
            // Check TX FIFO not empty
            spi_axi_read(32'h04, read_data);  // Status
            check_result("TX FIFO Not Empty", 1'b0, read_data[0]);  // Empty bit should be 0
        end
    endtask
    
    task test_spi_modes();
        begin
            $display("Testing SPI Modes...");
            
            // Test SPI Mode 1 (CPOL=0, CPHA=1)
            spi_axi_write(32'h00, 32'h00000407);  // Mode 1
            spi_axi_write(32'h10, 32'hABCDEF00);  // Send data
            
            repeat(100) @(posedge clk);
            
            // Test SPI Mode 2 (CPOL=1, CPHA=0)
            spi_axi_write(32'h00, 32'h0000040B);  // Mode 2
            spi_axi_write(32'h10, 32'h55AA55AA);  // Send data
            
            repeat(100) @(posedge clk);
            
            // Test SPI Mode 3 (CPOL=1, CPHA=1)
            spi_axi_write(32'h00, 32'h0000040F);  // Mode 3
            spi_axi_write(32'h10, 32'hFFFFFFFF);  // Send data
            
            repeat(100) @(posedge clk);
        end
    endtask
    
    task test_loopback_mode();
        logic [31:0] read_data;
        begin
            $display("Testing SPI Loopback Mode...");
            
            // Enable loopback mode
            spi_axi_write(32'h00, 32'h00040403);  // Loopback + Enable + Master + Mode 0
            
            // Send test pattern
            spi_axi_write(32'h10, 32'h5A5A5A5A);
            
            // Wait for completion
            wait(spi_rx_complete_irq);
            
            // Read back data (should match in loopback)
            spi_axi_read(32'h14, read_data);
            check_result("SPI Loopback Data", 32'h5A5A5A5A, read_data);
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting GPIO and SPI Controllers Tests...");
        
        reset_system();
        
        // GPIO Tests
        test_gpio_basic_io();
        test_gpio_interrupts();
        
        // SPI Tests
        test_spi_configuration();
        test_spi_data_transfer();
        test_spi_fifo_operations();
        test_spi_modes();
        test_loopback_mode();
        
        // Test summary
        $display("\n=== GPIO and SPI Controllers Test Summary ===");
        $display("Total Tests: %d", test_count);
        $display("Passed: %d", pass_count);
        $display("Failed: %d", fail_count);
        
        if (fail_count == 0) begin
            $display("All GPIO and SPI Controller tests PASSED!");
        end else begin
            $display("Some GPIO and SPI Controller tests FAILED!");
        end
        
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #3000000;  // 3ms timeout
        $display("ERROR: Test timeout!");
        $finish;
    end
    
    // Waveform dumping
    initial begin
        $dumpfile("test_gpio_spi_controllers.vcd");
        $dumpvars(0, test_gpio_spi_controllers);
    end

endmodule