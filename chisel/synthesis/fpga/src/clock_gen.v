// 时钟生成模块
// 使用 Xilinx MMCM 生成多个时钟域

module clock_gen (
    input  wire clk_in,         // 输入时钟 (250 MHz from PCIe)
    input  wire rst_in,         // 输入复位
    output wire clk_100m,       // 100 MHz 系统时钟
    output wire clk_50m,        // 50 MHz UART 时钟
    output wire clk_125m,       // 125 MHz 调试时钟
    output wire locked          // PLL 锁定信号
);

    wire clk_fb;
    wire clk_100m_unbuf;
    wire clk_50m_unbuf;
    wire clk_125m_unbuf;
    
    // Xilinx MMCM 实例
    // 输入: 250 MHz
    // 输出: 100 MHz, 50 MHz, 125 MHz
    MMCME2_ADV #(
        .BANDWIDTH          ("OPTIMIZED"),
        .CLKFBOUT_MULT_F    (4.0),          // VCO = 250 * 4 = 1000 MHz
        .CLKIN1_PERIOD      (4.0),          // 250 MHz = 4ns
        .CLKOUT0_DIVIDE_F   (10.0),         // 1000 / 10 = 100 MHz
        .CLKOUT1_DIVIDE     (20),           // 1000 / 20 = 50 MHz
        .CLKOUT2_DIVIDE     (8),            // 1000 / 8 = 125 MHz
        .DIVCLK_DIVIDE      (1),
        .REF_JITTER1        (0.010)
    ) mmcm_inst (
        .CLKIN1     (clk_in),
        .CLKFBIN    (clk_fb),
        .RST        (rst_in),
        .PWRDWN     (1'b0),
        .CLKOUT0    (clk_100m_unbuf),
        .CLKOUT1    (clk_50m_unbuf),
        .CLKOUT2    (clk_125m_unbuf),
        .CLKFBOUT   (clk_fb),
        .LOCKED     (locked)
    );
    
    // 全局时钟缓冲
    BUFG bufg_100m (.I(clk_100m_unbuf), .O(clk_100m));
    BUFG bufg_50m  (.I(clk_50m_unbuf),  .O(clk_50m));
    BUFG bufg_125m (.I(clk_125m_unbuf), .O(clk_125m));

endmodule
