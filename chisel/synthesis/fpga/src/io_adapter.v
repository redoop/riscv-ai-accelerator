// IO 适配器模块
// 处理 FPGA IO 和 SoC 接口之间的转换

module io_adapter (
    input  wire         clk,
    input  wire         rst,
    
    // SoC 侧接口
    input  wire         soc_uart_tx,
    output wire         soc_uart_rx,
    input  wire [31:0]  soc_gpio_out,
    output wire [31:0]  soc_gpio_in,
    input  wire [31:0]  soc_gpio_oe,
    
    // FPGA 侧接口 (PCIe BAR 寄存器)
    input  wire [31:0]  bar_addr,
    input  wire [31:0]  bar_wdata,
    output reg  [31:0]  bar_rdata,
    input  wire         bar_wen,
    input  wire         bar_ren
);

    // 寄存器地址定义
    localparam ADDR_UART_TX   = 32'h1000;
    localparam ADDR_UART_RX   = 32'h1004;
    localparam ADDR_GPIO_OUT  = 32'h1008;
    localparam ADDR_GPIO_IN   = 32'h100C;
    localparam ADDR_GPIO_OE   = 32'h1010;
    localparam ADDR_STATUS    = 32'h1014;
    
    // 内部寄存器
    reg         uart_rx_reg;
    reg [31:0]  gpio_in_reg;
    reg [31:0]  status_reg;
    
    // UART RX 写入
    always @(posedge clk) begin
        if (rst) begin
            uart_rx_reg <= 1'b1;  // UART 空闲为高
        end else if (bar_wen && bar_addr == ADDR_UART_RX) begin
            uart_rx_reg <= bar_wdata[0];
        end
    end
    assign soc_uart_rx = uart_rx_reg;
    
    // GPIO IN 写入
    always @(posedge clk) begin
        if (rst) begin
            gpio_in_reg <= 32'h0;
        end else if (bar_wen && bar_addr == ADDR_GPIO_IN) begin
            gpio_in_reg <= bar_wdata;
        end
    end
    assign soc_gpio_in = gpio_in_reg;
    
    // 状态寄存器
    always @(posedge clk) begin
        if (rst) begin
            status_reg <= 32'h0;
        end else begin
            status_reg <= {
                24'h0,
                soc_uart_tx,    // [7]
                uart_rx_reg,    // [6]
                6'h0
            };
        end
    end
    
    // BAR 读取
    always @(posedge clk) begin
        if (rst) begin
            bar_rdata <= 32'h0;
        end else if (bar_ren) begin
            case (bar_addr)
                ADDR_UART_TX:  bar_rdata <= {31'h0, soc_uart_tx};
                ADDR_UART_RX:  bar_rdata <= {31'h0, uart_rx_reg};
                ADDR_GPIO_OUT: bar_rdata <= soc_gpio_out;
                ADDR_GPIO_IN:  bar_rdata <= gpio_in_reg;
                ADDR_GPIO_OE:  bar_rdata <= soc_gpio_oe;
                ADDR_STATUS:   bar_rdata <= status_reg;
                default:       bar_rdata <= 32'hDEADBEEF;
            endcase
        end
    end

endmodule
