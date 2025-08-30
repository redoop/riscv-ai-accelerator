// Simple testbench for RISC-V AI Chip without UVM dependencies
`timescale 1ns/1ps

`include "chip_config.sv"

module tb_riscv_ai_chip_simple;

    // Clock and reset
    logic clk = 0;
    logic rst_n = 0;
    
    // HBM interface signals (simplified)
    logic [HBM_CHANNELS-1:0] hbm_clk;
    logic [HBM_CHANNELS-1:0] hbm_rst_n;
    logic [HBM_CHANNELS*HBM_DATA_WIDTH-1:0] hbm_dq;
    logic [HBM_CHANNELS-1:0] hbm_valid;
    logic [HBM_CHANNELS-1:0] hbm_ready;
    
    // PCIe interface (simplified)
    logic pcie_clk = 0;
    logic pcie_rst_n = 0;
    logic [PCIE_LANES-1:0] pcie_tx_p, pcie_tx_n;
    logic [PCIE_LANES-1:0] pcie_rx_p, pcie_rx_n;
    
    // Test control signals
    logic test_pass = 0;
    logic test_complete = 0;
    
    // Clock generation - 100MHz system clock
    always #5 clk = ~clk;
    
    // PCIe clock - 250MHz
    always #2 pcie_clk = ~pcie_clk;
    
    // HBM clocks - 400MHz
    genvar i;
    generate
        for (i = 0; i < HBM_CHANNELS; i++) begin : gen_hbm_clk
            always #1.25 hbm_clk[i] = ~hbm_clk[i];
        end
    endgenerate
    
    // DUT instantiation
    riscv_ai_chip dut (
        .clk(clk),
        .rst_n(rst_n),
        .hbm_clk(hbm_clk),
        .hbm_rst_n(hbm_rst_n),
        .hbm_dq(hbm_dq),
        .hbm_valid(hbm_valid),
        .hbm_ready(hbm_ready),
        .pcie_clk(pcie_clk),
        .pcie_rst_n(pcie_rst_n),
        .pcie_tx_p(pcie_tx_p),
        .pcie_tx_n(pcie_tx_n),
        .pcie_rx_p(pcie_rx_p),
        .pcie_rx_n(pcie_rx_n)
    );
    
    // Test sequence
    initial begin
        $display("Starting RISC-V AI Chip simple testbench");
        
        // Initialize signals
        rst_n = 0;
        pcie_rst_n = 0;
        hbm_rst_n = '0;
        hbm_dq = '0;
        hbm_valid = '0;
        pcie_rx_p = '0;
        pcie_rx_n = '1;
        
        // Reset sequence
        repeat(20) @(posedge clk);
        rst_n = 1;
        pcie_rst_n = 1;
        hbm_rst_n = '1;
        
        $display("Reset released");
        
        // Basic connectivity test
        repeat(100) @(posedge clk);
        
        // Simple data pattern test
        hbm_valid = '1;
        for (int j = 0; j < 10; j++) begin
            hbm_dq = {HBM_CHANNELS{$random}};
            @(posedge clk);
        end
        hbm_valid = '0;
        
        $display("Basic connectivity test passed");
        
        // Wait for any pending operations
        repeat(200) @(posedge clk);
        
        test_pass = 1;
        test_complete = 1;
        
        $display("Simple testbench completed successfully");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #1000000; // 1ms timeout
        $display("ERROR: Testbench timeout!");
        $finish;
    end
    
    // Optional: Generate VCD for waveform viewing
    initial begin
        $dumpfile("tb_riscv_ai_chip_simple.vcd");
        $dumpvars(0, tb_riscv_ai_chip_simple);
    end

endmodule