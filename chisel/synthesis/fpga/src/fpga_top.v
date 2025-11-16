// FPGA 顶层封装 - RISC-V AI 加速器
// 适配 AWS F1 Shell-CL 接口

module fpga_top (
    // 时钟和复位
    input  wire         clk_main_a0,        // 主时钟 (来自 AWS Shell)
    input  wire         rst_main_n,         // 复位信号 (低电平有效)
    
    // PCIe 接口 (简化版，实际使用 AWS Shell 提供的 AXI 接口)
    input  wire [31:0]  pcie_bar_addr,      // PCIe BAR 地址
    input  wire [31:0]  pcie_bar_wdata,     // PCIe 写数据
    output wire [31:0]  pcie_bar_rdata,     // PCIe 读数据
    input  wire         pcie_bar_wen,       // PCIe 写使能
    input  wire         pcie_bar_ren,       // PCIe 读使能
    
    // 调试接口
    output wire [7:0]   debug_status        // 调试状态输出
);

    // 内部信号
    wire        sys_clk;
    wire        sys_rst;
    wire        uart_tx;
    wire        uart_rx;
    wire [31:0] gpio_out;
    wire [31:0] gpio_in;
    wire [31:0] gpio_oe;
    
    // 时钟和复位处理
    assign sys_clk = clk_main_a0;
    assign sys_rst = ~rst_main_n;  // 转换为高电平有效
    
    // UART 信号映射到 PCIe BAR
    reg uart_rx_reg;
    always @(posedge sys_clk) begin
        if (pcie_bar_wen && pcie_bar_addr == 32'h1000)
            uart_rx_reg <= pcie_bar_wdata[0];
    end
    assign uart_rx = uart_rx_reg;
    
    // GPIO 映射到 PCIe BAR
    reg [31:0] gpio_in_reg;
    always @(posedge sys_clk) begin
        if (pcie_bar_wen && pcie_bar_addr == 32'h1004)
            gpio_in_reg <= pcie_bar_wdata;
    end
    assign gpio_in = gpio_in_reg;
    
    // PCIe 读数据多路复用
    reg [31:0] pcie_rdata_reg;
    always @(posedge sys_clk) begin
        if (pcie_bar_ren) begin
            case (pcie_bar_addr)
                32'h1000: pcie_rdata_reg <= {31'b0, uart_tx};
                32'h1004: pcie_rdata_reg <= gpio_out;
                32'h1008: pcie_rdata_reg <= gpio_oe;
                default:  pcie_rdata_reg <= 32'hDEADBEEF;
            endcase
        end
    end
    assign pcie_bar_rdata = pcie_rdata_reg;
    
    // 实例化 SoC 核心
    SimpleEdgeAiSoC soc_inst (
        .clock          (sys_clk),
        .reset          (sys_rst),
        .io_uart_tx     (uart_tx),
        .io_uart_rx     (uart_rx),
        .io_gpio_out    (gpio_out),
        .io_gpio_in     (gpio_in),
        .io_gpio_oe     (gpio_oe)
    );
    
    // 调试状态输出
    assign debug_status = {
        sys_rst,        // [7] 复位状态
        uart_tx,        // [6] UART TX
        uart_rx,        // [5] UART RX
        gpio_out[4:0]   // [4:0] GPIO 低 5 位
    };
    
    // ILA 调试探针（可选）
    `ifdef USE_ILA
    ila_0 ila_inst (
        .clk    (sys_clk),
        .probe0 (sys_rst),
        .probe1 (uart_tx),
        .probe2 (uart_rx),
        .probe3 (gpio_out),
        .probe4 (gpio_in)
    );
    `endif

endmodule
