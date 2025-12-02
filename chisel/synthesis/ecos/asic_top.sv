//`define ip_0 3'd0 // project_1854
`define ip_1 3'd1 // SimpleEdgeAiSoC (RISC-V AI Accelerator)
//`define ip_2 3'd2 // project_1839
//`define ip_3 3'd3   // ysyxSoCASIC
//`define ip_4 3'd4 // project_1988
//`define ip_5 3'd5 // project_1993

module asic_top (
    input   ip_sel_pad0,
    input   ip_sel_pad1,
    input   ip_sel_pad2,
    input   sys_clk_i_pad,
    output  sys_clk_o_pad,
    input   rst_n_pad,
    inout   io_pad0,
    inout   io_pad1,
    inout   io_pad2,
    inout   io_pad3,
    inout   io_pad4,
    inout   io_pad5,
    inout   io_pad6,
    inout   io_pad7,
    inout   io_pad8,
    inout   io_pad9,
    inout   io_pad10,
    inout   io_pad11,
    inout   io_pad12,
    inout   io_pad13,
    inout   io_pad14,
    inout   io_pad15,
    inout   io_pad16,
    inout   io_pad17,
    inout   io_pad18,
    inout   io_pad19,
    inout   io_pad20,
    inout   io_pad21,
    inout   io_pad22,
    inout   io_pad23,
    inout   io_pad24,
    inout   io_pad25,
    inout   io_pad26,
    inout   io_pad27,
    inout   io_pad28,
    inout   io_pad29,
    inout   io_pad30,
    inout   io_pad31,
    inout   io_pad32,
    inout   io_pad33,
    inout   io_pad34,
    inout   io_pad35,
    inout   io_pad36,
    inout   io_pad37,
    inout   io_pad38,
    inout   io_pad39,
    inout   io_pad40,
    inout   io_pad41,
    inout   io_pad42,
    inout   io_pad43,
    inout   io_pad44,
    inout   io_pad45,
    inout   io_pad46,
    inout   io_pad47,
    inout   io_pad48,
    inout   io_pad49,
    inout   io_pad50,
    inout   io_pad51,
    inout   io_pad52,
    inout   io_pad53,
    inout   io_pad54,
    inout   io_pad55,
    inout   io_pad56,
    inout   io_pad57,
    inout   io_pad58,
    inout   io_pad59,
    inout   io_pad60,
    inout   io_pad61,
    inout   io_pad62,
    inout   io_pad63,
    inout   io_pad64,
    inout   io_pad65,
    inout   io_pad66,
    inout   io_pad67,
    inout   io_pad68,
    inout   io_pad69,
    inout   io_pad70,
    inout   io_pad71,
    inout   io_pad72,
    inout   io_pad73,
    inout   io_pad74,
    inout   io_pad75,
    inout   io_pad76,
    inout   io_pad77,
    inout   io_pad78,
    inout   io_pad79,
    inout   io_pad80,
    inout   io_pad81
);

    logic [2:0]     ip_sel;
    logic           sys_clk;
    logic           rst_n;
    logic [81:0]    io_pad_i;
    logic [81:0]    io_pad_o;
    logic [81:0]    io_pad_oe;
    logic [7:0]     io_pad_pu;
    logic [7:0]     io_pad_pd;

    // clock and reset signals for different IPs
    logic           clk_100m;
    logic           clk_50m;
    logic           clk_25m;
    logic           rst_100m_n;
    logic           rst_50m_n;
    logic           rst_25m_n;

    // ip0 signals
    logic           ip0_clk_100m;
    logic           ip0_jtag_jtrstn;
    logic           ip0_jtag_jtck;
    logic           ip0_jtag_jtms;
    logic           ip0_jtag_jtdi;
    logic           ip0_uart_rx;
    logic           ip0_spi_miso;
    logic           ip0_int_n_i;
    logic           ip0_sram_wait;
    logic [3:0]     ip0_sram_ben_i;
    logic [21:0]    ip0_sram_addr_i;
    logic [31:0]    ip0_sram_data_i;
    logic [7:0]     ip0_gpios_i;

    logic           ip0_jtag_jtdo;
    logic           ip0_uart_tx;
    logic           ip0_spi_mcsn;
    logic           ip0_spi_mclk;
    logic           ip0_spi_mosi;
    logic           ip0_sram_csn;
    logic           ip0_sram_wen;
    logic           ip0_sram_oen;
    logic [3:0]     ip0_sram_ben_o;
    logic [3:0]     ip0_sram_ben_oe;
    logic [21:0]    ip0_sram_addr_o;
    logic [21:0]    ip0_sram_addr_oe;
    logic [31:0]    ip0_sram_data_o;
    logic [31:0]    ip0_sram_data_oe;
    logic [7:0]     ip0_gpios_o;
    logic [7:0]     ip0_gpios_oe;
    logic [7:0]     ip0_gpios_pu;
    logic [7:0]     ip0_gpios_pd;

    // ip1 signals
    logic           ip1_clk_100m;
    logic           ip1_io_uart_tx;
    logic           ip1_io_uart_rx;
    logic           ip1_io_lcd_spi_clk;
    logic           ip1_io_lcd_spi_mosi;
    logic           ip1_io_lcd_spi_cs;
    logic           ip1_io_lcd_spi_dc;
    logic           ip1_io_lcd_spi_rst;
    logic           ip1_io_lcd_backlight;
    logic [31:0]    ip1_io_gpio_out;
    logic [31:0]    ip1_io_gpio_in;
    logic           ip1_io_trap;
    logic           ip1_io_compact_irq;
    logic           ip1_io_bitnet_irq;
    logic           ip1_io_uart_tx_irq;
    logic           ip1_io_uart_rx_irq;

    // ip2 signals
    logic           ip2_clk_25m;
    logic           ip2_TJUT_instload;
    logic           ip2_TJUT_breakpoint;
    logic           ip2_TJUT_invalid;
    logic           ip2_TJUT_uart_rxd;
    logic           ip2_TJUT_uart_txd;
    logic [7:0]     ip2_TJUT_gpio1_dir;
    logic [7:0]     ip2_TJUT_gpio2_dir;
    logic [7:0]     ip2_TJUT_gpio1_out;
    logic [7:0]     ip2_TJUT_gpio2_out;
    logic [7:0]     ip2_TJUT_gpio1_in;
    logic [7:0]     ip2_TJUT_gpio2_in;

    // ip3 signals (ysyxSoCASIC)
    logic           s_ysyx_clk_100m;
    logic           s_ysyx_clock_half;
    logic           s_ysyx_oldIPClock;
    logic           s_ysyx_oldIPReset;
    logic           s_ysyx_uart0_rx;   
    logic           s_ysyx_uart0_tx;   
    logic           s_ysyx_spi_sck; 
    logic [7:0]     s_ysyx_spi_ss; 
    logic           s_ysyx_spi_mosi;
    logic           s_ysyx_spi_miso;
    logic [1:0]     s_ysyx_core_sel;
    logic           s_ysyx_core_irq;

    // ip4 signals
    logic           ip4_clk_100m;
    logic           ip4_uart_rx;
    logic           ip4_uart_tx;
    logic [3:0]     ip4_gpio_in;
    logic [3:0]     ip4_gpio_out;
    logic           ip4_jtag_tck_pin;
    logic           ip4_jtag_tms_pin;
    logic           ip4_jtag_tdi_pin;
    logic           ip4_jtag_tdo_pin;

    // ip5 signals
    logic           ip5_clk_100m;
    logic           ip5_cfg_wr_en;
    logic           ip5_cfg_rd_en;
    logic [15:0]    ip5_cfg_addr;
    logic [31:0]    ip5_cfg_wdata;
    logic [31:0]    ip5_cfg_rdata;
    logic           ip5_halted;
    logic [31:0]    ip5_npu_result_o;

    always_comb begin 
        io_pad_o  = '0;
        io_pad_oe = '0;
        io_pad_pu = '0;
        io_pad_pd = '0;

        // Default assignments for ip0 signals
        ip0_jtag_jtrstn  = 1'b0;
        ip0_jtag_jtck    = 1'b0;
        ip0_jtag_jtms    = 1'b0;
        ip0_jtag_jtdi    = 1'b0;
        ip0_uart_rx      = 1'b0;
        ip0_spi_miso     = 1'b0;
        ip0_int_n_i      = 1'b0;
        ip0_sram_wait    = 1'b0;
        ip0_sram_ben_i   = '0;
        ip0_sram_addr_i  = '0;
        ip0_sram_data_i  = '0;
        ip0_gpios_i      = '0;

        // Default assignments for ip1 signals
        ip1_io_uart_rx       = 1'b0;
        ip1_io_gpio_in       = '0;

        // Default assignments for ip2 signals
        ip2_TJUT_instload    = 1'b0;
        ip2_TJUT_uart_rxd    = 1'b0;
        ip2_TJUT_gpio1_in    = '0;
        ip2_TJUT_gpio2_in    = '0;

        // Default assignments for ip3 signals
        s_ysyx_uart0_rx    = 1'b0;
        s_ysyx_spi_miso    = 1'b0;
        s_ysyx_core_sel    = 2'b00;

        // Default assignments for ip4 signals
        ip4_uart_rx        = 1'b0;
        ip4_gpio_in        = '0;
        ip4_jtag_tck_pin   = 1'b0;
        ip4_jtag_tms_pin   = 1'b0;
        ip4_jtag_tdi_pin   = 1'b0;

        // Default assignments for ip5 signals
        ip5_cfg_wr_en      = 1'b0;
        ip5_cfg_rd_en      = 1'b0;
        ip5_cfg_addr       = '0;
        ip5_cfg_wdata      = '0;

        unique case (ip_sel)
        `ifdef ip_0
            `ip_0: begin
                io_pad_oe[3:0]   = ip0_sram_ben_oe;
                io_pad_oe[25:4]  = ip0_sram_addr_oe;
                io_pad_oe[57:26] = ip0_sram_data_oe;
                io_pad_oe[65:58] = ip0_gpios_oe;
                io_pad_oe[73:66] = 8'b0;
                io_pad_oe[81:74] = 8'b1;

                ip0_sram_ben_i    = io_pad_i[3:0];
                ip0_sram_addr_i   = io_pad_i[25:4];
                ip0_sram_data_i   = io_pad_i[57:26];
                ip0_gpios_i       = io_pad_i[65:58];

                io_pad_o[3:0]    = ip0_sram_ben_o;
                io_pad_o[25:4]   = ip0_sram_addr_o;
                io_pad_o[57:26]  = ip0_sram_data_o;
                io_pad_o[65:58]  = ip0_gpios_o;
                io_pad_pu[7:0]   = ip0_gpios_pu;
                io_pad_pd[7:0]   = ip0_gpios_pd;
                
                ip0_jtag_jtrstn  = io_pad_i[66];
                ip0_jtag_jtck    = io_pad_i[67];
                ip0_jtag_jtms    = io_pad_i[68];
                ip0_jtag_jtdi    = io_pad_i[69];
                ip0_uart_rx      = io_pad_i[70];
                ip0_spi_miso     = io_pad_i[71];
                ip0_int_n_i      = io_pad_i[72];
                ip0_sram_wait    = io_pad_i[73];

                io_pad_o[74] = ip0_jtag_jtdo;
                io_pad_o[75] = ip0_uart_tx;
                io_pad_o[76] = ip0_spi_mcsn;
                io_pad_o[77] = ip0_spi_mclk;
                io_pad_o[78] = ip0_spi_mosi;
                io_pad_o[79] = ip0_sram_csn;
                io_pad_o[80] = ip0_sram_wen;
                io_pad_o[81] = ip0_sram_oen;
            end
        `endif
        `ifdef ip_1
            `ip_1: begin
                io_pad_oe[31:0] = 32'b0;
                io_pad_oe[63:32] = 32'b1;
                io_pad_oe[64] = 1'b1;
                io_pad_oe[65] = 1'b0;
                io_pad_oe[76:66] = 11'b1;

                ip1_io_gpio_in = io_pad_i[31:0];
                io_pad_o[63:32] = ip1_io_gpio_out;

                io_pad_o[64] = ip1_io_uart_tx;
                ip1_io_uart_rx = io_pad_i[65];
                io_pad_o[66] = ip1_io_uart_tx_irq;
                io_pad_o[67] = ip1_io_uart_rx_irq;
                io_pad_o[68] = ip1_io_lcd_spi_clk;
                io_pad_o[69] = ip1_io_lcd_spi_mosi;
                io_pad_o[70] = ip1_io_lcd_spi_cs;
                io_pad_o[71] = ip1_io_lcd_spi_dc;
                io_pad_o[72] = ip1_io_lcd_spi_rst;
                io_pad_o[73] = ip1_io_lcd_backlight;
                io_pad_o[74] = ip1_io_trap;
                io_pad_o[75] = ip1_io_compact_irq;
                io_pad_o[76] = ip1_io_bitnet_irq;
            end
        `endif
        `ifdef ip_2
            `ip_2: begin
                io_pad_oe[7:0]  = ip2_TJUT_gpio1_dir;
                io_pad_oe[15:8] = ip2_TJUT_gpio2_dir;
                io_pad_oe[16] = 1'b0;
                io_pad_oe[18:17] = 2'b1;
                io_pad_oe[19] = 1'b0;
                io_pad_oe[20] = 1'b1;

                ip2_TJUT_gpio1_in = io_pad_i[7:0];
                ip2_TJUT_gpio2_in = io_pad_i[15:8];
                ip2_TJUT_instload = io_pad_i[16];
                ip2_TJUT_uart_rxd = io_pad_i[19];

                io_pad_o[7:0]   = ip2_TJUT_gpio1_out;
                io_pad_o[15:8]  = ip2_TJUT_gpio2_out;
                io_pad_o[17]    = ip2_TJUT_breakpoint;
                io_pad_o[18]    = ip2_TJUT_invalid;
                io_pad_o[20]    = ip2_TJUT_uart_txd;
            end
        `endif
        `ifdef ip_3
            `ip_3: begin
                io_pad_oe[0]     = 1'b0;
                io_pad_oe[1]     = 1'b1;
                io_pad_oe[2]     = 1'b1;
                io_pad_oe[10:3]  = 8'b1;
                io_pad_oe[11]    = 1'b1;
                io_pad_oe[12]    = 1'b0;
                io_pad_oe[14:13] = 2'b0;
                io_pad_oe[15]    = 1'b0;

                s_ysyx_uart0_rx  = io_pad_i[0];
                s_ysyx_spi_miso  = io_pad_i[12];
                s_ysyx_core_sel  = io_pad_i[14:13];
                s_ysyx_core_irq  = io_pad_i[15];

                io_pad_o[1]      = s_ysyx_uart0_tx;
                io_pad_o[2]      = s_ysyx_spi_sck;
                io_pad_o[10:3]   = s_ysyx_spi_ss;
                io_pad_o[11]     = s_ysyx_spi_mosi;
            end
        `endif
        `ifdef ip_4
            `ip_4: begin
                io_pad_oe[0]    = 1'b0;
                io_pad_oe[1]    = 1'b1;
                io_pad_oe[5:2]  = 4'b0;
                io_pad_oe[9:6]  = 4'b1;
                io_pad_oe[12:10]= 3'b0;
                io_pad_oe[13]   = 1'b1;

                ip4_uart_rx    = io_pad_i[0];
                ip4_gpio_in    = io_pad_i[5:2];
                ip4_jtag_tck_pin = io_pad_i[10];
                ip4_jtag_tms_pin = io_pad_i[11];
                ip4_jtag_tdi_pin = io_pad_i[12];

                io_pad_o[1]    = ip4_uart_tx;
                io_pad_o[9:6]  = ip4_gpio_out;
                io_pad_o[13]   = ip4_jtag_tdo_pin;
            end 
            end 
        `endif
        `ifdef ip_5
            `ip_5: begin
                io_pad_oe[15:0]  = 16'b0;
                io_pad_oe[16]    = 1'b0;
                io_pad_oe[48:17] = ip5_cfg_wr_en ? 32'b0 : 32'b1;
                io_pad_oe[49]    = 1'b1;
                io_pad_oe[81:50] = 32'b1;

                ip5_cfg_addr    = io_pad_i[15:0];
                ip5_cfg_wr_en   = io_pad_i[16];
                ip5_cfg_rd_en   = ~ip5_cfg_wr_en;
                ip5_cfg_wdata   = io_pad_i[48:17];

                io_pad_o[48:17] = ip5_cfg_rdata;
                io_pad_o[49]    = ip5_halted;
                io_pad_o[81:50] = ip5_npu_result_o;
            end
        `endif
            default: begin
            end
        endcase
    end

`ifdef ip_0 
digi_top u_digi_top (
    .TEST_PIN      (),
    .sys_clk_i     (ip0_clk_100m),
    .sys_por_n_i   (rst_100m_n),
    .jtag_jtrstn   (ip0_jtag_jtrstn),
    .jtag_jtck     (ip0_jtag_jtck),
    .jtag_jtms     (ip0_jtag_jtms),
    .jtag_jtdi     (ip0_jtag_jtdi),
    .uart_rx       (ip0_uart_rx),
    .spi_miso      (ip0_spi_miso),
    .int_n_i       (ip0_int_n_i),
    .sram_wait     (ip0_sram_wait),

    .jtag_jtdo     (ip0_jtag_jtdo),
    .uart_tx       (ip0_uart_tx),
    .spi_mcsn      (ip0_spi_mcsn),
    .spi_mclk      (ip0_spi_mclk),
    .spi_mosi      (ip0_spi_mosi),
    .sram_csn      (ip0_sram_csn),
    .sram_wen      (ip0_sram_wen),
    .sram_oen      (ip0_sram_oen),

    .sram_ben_i    (ip0_sram_ben_i),
    .sram_ben_o    (ip0_sram_ben_o),
    .sram_ben_oe   (ip0_sram_ben_oe),
    .sram_addr_i   (ip0_sram_addr_i),
    .sram_addr_o   (ip0_sram_addr_o),
    .sram_addr_oe  (ip0_sram_addr_oe),
    .sram_data_i   (ip0_sram_data_i),
    .sram_data_o   (ip0_sram_data_o),
    .sram_data_oe  (ip0_sram_data_oe),
    .gpios_i       (ip0_gpios_i),
    .gpios_o       (ip0_gpios_o),
    .gpios_oe      (ip0_gpios_oe),
    .gpios_pu      (ip0_gpios_pu),
    .gpios_pd      (ip0_gpios_pd)
);
`endif

`ifdef ip_1
ip1_SimpleEdgeAiSoC u_SimpleEdgeAiSoC (
    .clock              (ip1_clk_100m),   
    .reset              (~rst_100m_n),
    .io_uart_tx         (ip1_io_uart_tx),
    .io_uart_rx         (ip1_io_uart_rx),
    .io_lcd_spi_clk     (ip1_io_lcd_spi_clk),    
    .io_lcd_spi_mosi    (ip1_io_lcd_spi_mosi),       
    .io_lcd_spi_cs      (ip1_io_lcd_spi_cs),   
    .io_lcd_spi_dc      (ip1_io_lcd_spi_dc),   
    .io_lcd_spi_rst     (ip1_io_lcd_spi_rst),   
    .io_lcd_backlight   (ip1_io_lcd_backlight),       
    .io_gpio_out        (ip1_io_gpio_out),    
    .io_gpio_in         (ip1_io_gpio_in),
    .io_trap            (ip1_io_trap),
    .io_compact_irq     (ip1_io_compact_irq),    
    .io_bitnet_irq      (ip1_io_bitnet_irq),    
    .io_uart_tx_irq     (ip1_io_uart_tx_irq),    
    .io_uart_rx_irq     (ip1_io_uart_rx_irq)
);
`endif

`ifdef ip_2
TJUT_TOP u_TJUT_TOP (
    .clk            (ip2_clk_25m)  ,
    .rstn           (rst_50m_n) ,
    .instload       (ip2_TJUT_instload),
    .breakpoint     (ip2_TJUT_breakpoint),
    .invalid        (ip2_TJUT_invalid),
    .uart_rxd       (ip2_TJUT_uart_rxd),
    .uart_txd       (ip2_TJUT_uart_txd),
    .gpio1_dir      (ip2_TJUT_gpio1_dir),
    .gpio2_dir      (ip2_TJUT_gpio2_dir),
    .gpio1_out      (ip2_TJUT_gpio1_out),
    .gpio2_out      (ip2_TJUT_gpio2_out),
    .gpio1_in       (ip2_TJUT_gpio1_in),
    .gpio2_in       (ip2_TJUT_gpio2_in)
);
`endif

`ifdef ip_3
ysyxSoCASIC u_ysyxSoCASIC (     
    .clock            (s_ysyx_clk_100m   ),   
    .reset            (~rst_100m_n       ),   
    .clock_half       (clk_100m          ),  
    .oldIPClock       (clk_100m          ),  
    .oldIPReset       (~rst_100m_n       ),  
    .uart0_rx         (s_ysyx_uart0_rx  ),    
    .uart0_tx         (s_ysyx_uart0_tx  ),    
    .spi_sck          (s_ysyx_spi_sck   ),  
    .spi_ss           (s_ysyx_spi_ss    ),  
    .spi_mosi         (s_ysyx_spi_mosi  ),
    .spi_miso         (s_ysyx_spi_miso  ),
    .core_sel         (s_ysyx_core_sel  ),
    .core_irq         (s_ysyx_core_irq  )
);
`endif

`ifdef ip_4
SynapticChip u_SynapticChip(
    .clk                (ip4_clk_100m),
    .rst_n              (rst_100m_n),

    .uart_rx            (ip4_uart_rx),
    .uart_tx            (ip4_uart_tx),

    .gpio_in            (ip4_gpio_in),
    .gpio_out           (ip4_gpio_out),

    .jtag_tck_pin       (ip4_jtag_tck_pin),
    .jtag_tms_pin       (ip4_jtag_tms_pin),
    .jtag_tdi_pin       (ip4_jtag_tdi_pin),
    .jtag_tdo_pin       (ip4_jtag_tdo_pin)
);
`endif

`ifdef ip_5
riscv_npu_top u_riscv_npu_top (
    .clk                    (ip5_clk_100m   ),
    .rst_n                  (rst_100m_n     ),
    .cfg_wr_en              (ip5_cfg_wr_en  ),
    .cfg_rd_en              (ip5_cfg_rd_en  ),
    .cfg_addr               (ip5_cfg_addr   ),
    .cfg_wdata              (ip5_cfg_wdata  ),
    .cfg_rdata              (ip5_cfg_rdata  ),
    .halted                 (ip5_halted     ),        
    .npu_result_o           (ip5_npu_result_o  )
);
`endif

tc_clk_buf u_ip0_clk_buf (.clk_i(clk_100m),       .clk_o(ip0_clk_100m));
tc_clk_buf u_ip1_clk_buf (.clk_i(clk_100m),       .clk_o(ip1_clk_100m));
tc_clk_buf u_ip2_clk_buf (.clk_i(clk_25m),        .clk_o(ip2_clk_25m));
tc_clk_buf u_ip3_clk_buf (.clk_i(clk_100m),       .clk_o(s_ysyx_clk_100m));
tc_clk_buf u_ip4_clk_buf (.clk_i(clk_100m),       .clk_o(ip4_clk_100m));
tc_clk_buf u_ip5_clk_buf (.clk_i(clk_100m),       .clk_o(ip5_clk_100m));

rcu u_rcu (
    .sys_clk_i          (sys_clk),
    .arst_n_i           (rst_n),  

    .clk_100m_o         (clk_100m),  
    .clk_50m_o          (clk_50m),   
    .clk_25m_o          (clk_25m),   
    .rst_100m_n_o       (rst_100m_n),
    .rst_50m_n_o        (rst_50m_n), 
    .rst_25m_n_o        (rst_25m_n)  
);

P65_1233_PWE u_sys_clk_pad (.E(1'b1), .XIN(sys_clk_i_pad), .XOUT(sys_clk_o_pad), .XC(sys_clk));

P65_1233_PBMUX u_rst_n_pad (.C(rst_n), .A(), .PAD(rst_n_pad), .IE(1'b1), .CS(1'b0), .I(1'b0), .OE(1'b0), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_ip_sel_pad0 (.C(ip_sel[0]), .A(), .PAD(ip_sel_pad0), .IE(1'b1), .CS(1'b0), .I(1'b0), .OE(1'b0), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_ip_sel_pad1 (.C(ip_sel[1]), .A(), .PAD(ip_sel_pad1), .IE(1'b1), .CS(1'b0), .I(1'b0), .OE(1'b0), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_ip_sel_pad2 (.C(ip_sel[2]), .A(), .PAD(ip_sel_pad2), .IE(1'b1), .CS(1'b0), .I(1'b0), .OE(1'b0), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));

P65_1233_PBMUX u_io_pad0 (.C(io_pad_i[0]),   .A(), .PAD(io_pad0),  .IE(~io_pad_oe[0]),  .CS(1'b0), .I(io_pad_o[0]),  .OE(io_pad_oe[0]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad1 (.C(io_pad_i[1]),   .A(), .PAD(io_pad1),  .IE(~io_pad_oe[1]),  .CS(1'b0), .I(io_pad_o[1]),  .OE(io_pad_oe[1]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad2 (.C(io_pad_i[2]),   .A(), .PAD(io_pad2),  .IE(~io_pad_oe[2]),  .CS(1'b0), .I(io_pad_o[2]),  .OE(io_pad_oe[2]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad3 (.C(io_pad_i[3]),   .A(), .PAD(io_pad3),  .IE(~io_pad_oe[3]),  .CS(1'b0), .I(io_pad_o[3]),  .OE(io_pad_oe[3]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad4 (.C(io_pad_i[4]),   .A(), .PAD(io_pad4),  .IE(~io_pad_oe[4]),  .CS(1'b0), .I(io_pad_o[4]),  .OE(io_pad_oe[4]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad5 (.C(io_pad_i[5]),   .A(), .PAD(io_pad5),  .IE(~io_pad_oe[5]),  .CS(1'b0), .I(io_pad_o[5]),  .OE(io_pad_oe[5]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad6 (.C(io_pad_i[6]),   .A(), .PAD(io_pad6),  .IE(~io_pad_oe[6]),  .CS(1'b0), .I(io_pad_o[6]),  .OE(io_pad_oe[6]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1)); 
P65_1233_PBMUX u_io_pad7 (.C(io_pad_i[7]),   .A(), .PAD(io_pad7),  .IE(~io_pad_oe[7]),  .CS(1'b0), .I(io_pad_o[7]),  .OE(io_pad_oe[7]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad8 (.C(io_pad_i[8]),   .A(), .PAD(io_pad8),  .IE(~io_pad_oe[8]),  .CS(1'b0), .I(io_pad_o[8]),  .OE(io_pad_oe[8]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad9 (.C(io_pad_i[9]),   .A(), .PAD(io_pad9),  .IE(~io_pad_oe[9]),  .CS(1'b0), .I(io_pad_o[9]),  .OE(io_pad_oe[9]),  .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad10 (.C(io_pad_i[10]), .A(), .PAD(io_pad10), .IE(~io_pad_oe[10]), .CS(1'b0), .I(io_pad_o[10]), .OE(io_pad_oe[10]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad11 (.C(io_pad_i[11]), .A(), .PAD(io_pad11), .IE(~io_pad_oe[11]), .CS(1'b0), .I(io_pad_o[11]), .OE(io_pad_oe[11]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad12 (.C(io_pad_i[12]), .A(), .PAD(io_pad12), .IE(~io_pad_oe[12]), .CS(1'b0), .I(io_pad_o[12]), .OE(io_pad_oe[12]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad13 (.C(io_pad_i[13]), .A(), .PAD(io_pad13), .IE(~io_pad_oe[13]), .CS(1'b0), .I(io_pad_o[13]), .OE(io_pad_oe[13]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad14 (.C(io_pad_i[14]), .A(), .PAD(io_pad14), .IE(~io_pad_oe[14]), .CS(1'b0), .I(io_pad_o[14]), .OE(io_pad_oe[14]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad15 (.C(io_pad_i[15]), .A(), .PAD(io_pad15), .IE(~io_pad_oe[15]), .CS(1'b0), .I(io_pad_o[15]), .OE(io_pad_oe[15]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad16 (.C(io_pad_i[16]), .A(), .PAD(io_pad16), .IE(~io_pad_oe[16]), .CS(1'b0), .I(io_pad_o[16]), .OE(io_pad_oe[16]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad17 (.C(io_pad_i[17]), .A(), .PAD(io_pad17), .IE(~io_pad_oe[17]), .CS(1'b0), .I(io_pad_o[17]), .OE(io_pad_oe[17]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad18 (.C(io_pad_i[18]), .A(), .PAD(io_pad18), .IE(~io_pad_oe[18]), .CS(1'b0), .I(io_pad_o[18]), .OE(io_pad_oe[18]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad19 (.C(io_pad_i[19]), .A(), .PAD(io_pad19), .IE(~io_pad_oe[19]), .CS(1'b0), .I(io_pad_o[19]), .OE(io_pad_oe[19]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad20 (.C(io_pad_i[20]), .A(), .PAD(io_pad20), .IE(~io_pad_oe[20]), .CS(1'b0), .I(io_pad_o[20]), .OE(io_pad_oe[20]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad21 (.C(io_pad_i[21]), .A(), .PAD(io_pad21), .IE(~io_pad_oe[21]), .CS(1'b0), .I(io_pad_o[21]), .OE(io_pad_oe[21]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad22 (.C(io_pad_i[22]), .A(), .PAD(io_pad22), .IE(~io_pad_oe[22]), .CS(1'b0), .I(io_pad_o[22]), .OE(io_pad_oe[22]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad23 (.C(io_pad_i[23]), .A(), .PAD(io_pad23), .IE(~io_pad_oe[23]), .CS(1'b0), .I(io_pad_o[23]), .OE(io_pad_oe[23]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad24 (.C(io_pad_i[24]), .A(), .PAD(io_pad24), .IE(~io_pad_oe[24]), .CS(1'b0), .I(io_pad_o[24]), .OE(io_pad_oe[24]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad25 (.C(io_pad_i[25]), .A(), .PAD(io_pad25), .IE(~io_pad_oe[25]), .CS(1'b0), .I(io_pad_o[25]), .OE(io_pad_oe[25]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad26 (.C(io_pad_i[26]), .A(), .PAD(io_pad26), .IE(~io_pad_oe[26]), .CS(1'b0), .I(io_pad_o[26]), .OE(io_pad_oe[26]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad27 (.C(io_pad_i[27]), .A(), .PAD(io_pad27), .IE(~io_pad_oe[27]), .CS(1'b0), .I(io_pad_o[27]), .OE(io_pad_oe[27]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad28 (.C(io_pad_i[28]), .A(), .PAD(io_pad28), .IE(~io_pad_oe[28]), .CS(1'b0), .I(io_pad_o[28]), .OE(io_pad_oe[28]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad29 (.C(io_pad_i[29]), .A(), .PAD(io_pad29), .IE(~io_pad_oe[29]), .CS(1'b0), .I(io_pad_o[29]), .OE(io_pad_oe[29]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad30 (.C(io_pad_i[30]), .A(), .PAD(io_pad30), .IE(~io_pad_oe[30]), .CS(1'b0), .I(io_pad_o[30]), .OE(io_pad_oe[30]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad31 (.C(io_pad_i[31]), .A(), .PAD(io_pad31), .IE(~io_pad_oe[31]), .CS(1'b0), .I(io_pad_o[31]), .OE(io_pad_oe[31]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad32 (.C(io_pad_i[32]), .A(), .PAD(io_pad32), .IE(~io_pad_oe[32]), .CS(1'b0), .I(io_pad_o[32]), .OE(io_pad_oe[32]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad33 (.C(io_pad_i[33]), .A(), .PAD(io_pad33), .IE(~io_pad_oe[33]), .CS(1'b0), .I(io_pad_o[33]), .OE(io_pad_oe[33]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad34 (.C(io_pad_i[34]), .A(), .PAD(io_pad34), .IE(~io_pad_oe[34]), .CS(1'b0), .I(io_pad_o[34]), .OE(io_pad_oe[34]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad35 (.C(io_pad_i[35]), .A(), .PAD(io_pad35), .IE(~io_pad_oe[35]), .CS(1'b0), .I(io_pad_o[35]), .OE(io_pad_oe[35]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad36 (.C(io_pad_i[36]), .A(), .PAD(io_pad36), .IE(~io_pad_oe[36]), .CS(1'b0), .I(io_pad_o[36]), .OE(io_pad_oe[36]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad37 (.C(io_pad_i[37]), .A(), .PAD(io_pad37), .IE(~io_pad_oe[37]), .CS(1'b0), .I(io_pad_o[37]), .OE(io_pad_oe[37]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad38 (.C(io_pad_i[38]), .A(), .PAD(io_pad38), .IE(~io_pad_oe[38]), .CS(1'b0), .I(io_pad_o[38]), .OE(io_pad_oe[38]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad39 (.C(io_pad_i[39]), .A(), .PAD(io_pad39), .IE(~io_pad_oe[39]), .CS(1'b0), .I(io_pad_o[39]), .OE(io_pad_oe[39]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad40 (.C(io_pad_i[40]), .A(), .PAD(io_pad40), .IE(~io_pad_oe[40]), .CS(1'b0), .I(io_pad_o[40]), .OE(io_pad_oe[40]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad41 (.C(io_pad_i[41]), .A(), .PAD(io_pad41), .IE(~io_pad_oe[41]), .CS(1'b0), .I(io_pad_o[41]), .OE(io_pad_oe[41]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad42 (.C(io_pad_i[42]), .A(), .PAD(io_pad42), .IE(~io_pad_oe[42]), .CS(1'b0), .I(io_pad_o[42]), .OE(io_pad_oe[42]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad43 (.C(io_pad_i[43]), .A(), .PAD(io_pad43), .IE(~io_pad_oe[43]), .CS(1'b0), .I(io_pad_o[43]), .OE(io_pad_oe[43]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad44 (.C(io_pad_i[44]), .A(), .PAD(io_pad44), .IE(~io_pad_oe[44]), .CS(1'b0), .I(io_pad_o[44]), .OE(io_pad_oe[44]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad45 (.C(io_pad_i[45]), .A(), .PAD(io_pad45), .IE(~io_pad_oe[45]), .CS(1'b0), .I(io_pad_o[45]), .OE(io_pad_oe[45]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad46 (.C(io_pad_i[46]), .A(), .PAD(io_pad46), .IE(~io_pad_oe[46]), .CS(1'b0), .I(io_pad_o[46]), .OE(io_pad_oe[46]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad47 (.C(io_pad_i[47]), .A(), .PAD(io_pad47), .IE(~io_pad_oe[47]), .CS(1'b0), .I(io_pad_o[47]), .OE(io_pad_oe[47]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad48 (.C(io_pad_i[48]), .A(), .PAD(io_pad48), .IE(~io_pad_oe[48]), .CS(1'b0), .I(io_pad_o[48]), .OE(io_pad_oe[48]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad49 (.C(io_pad_i[49]), .A(), .PAD(io_pad49), .IE(~io_pad_oe[49]), .CS(1'b0), .I(io_pad_o[49]), .OE(io_pad_oe[49]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad50 (.C(io_pad_i[50]), .A(), .PAD(io_pad50), .IE(~io_pad_oe[50]), .CS(1'b0), .I(io_pad_o[50]), .OE(io_pad_oe[50]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad51 (.C(io_pad_i[51]), .A(), .PAD(io_pad51), .IE(~io_pad_oe[51]), .CS(1'b0), .I(io_pad_o[51]), .OE(io_pad_oe[51]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad52 (.C(io_pad_i[52]), .A(), .PAD(io_pad52), .IE(~io_pad_oe[52]), .CS(1'b0), .I(io_pad_o[52]), .OE(io_pad_oe[52]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad53 (.C(io_pad_i[53]), .A(), .PAD(io_pad53), .IE(~io_pad_oe[53]), .CS(1'b0), .I(io_pad_o[53]), .OE(io_pad_oe[53]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad54 (.C(io_pad_i[54]), .A(), .PAD(io_pad54), .IE(~io_pad_oe[54]), .CS(1'b0), .I(io_pad_o[54]), .OE(io_pad_oe[54]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad55 (.C(io_pad_i[55]), .A(), .PAD(io_pad55), .IE(~io_pad_oe[55]), .CS(1'b0), .I(io_pad_o[55]), .OE(io_pad_oe[55]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad56 (.C(io_pad_i[56]), .A(), .PAD(io_pad56), .IE(~io_pad_oe[56]), .CS(1'b0), .I(io_pad_o[56]), .OE(io_pad_oe[56]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad57 (.C(io_pad_i[57]), .A(), .PAD(io_pad57), .IE(~io_pad_oe[57]), .CS(1'b0), .I(io_pad_o[57]), .OE(io_pad_oe[57]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad58 (.C(io_pad_i[58]), .A(), .PAD(io_pad58), .IE(~io_pad_oe[58]), .CS(1'b0), .I(io_pad_o[58]), .OE(io_pad_oe[58]), .OD(1'b0), .PU(io_pad_pu[0]), .PD(io_pad_pd[0]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad59 (.C(io_pad_i[59]), .A(), .PAD(io_pad59), .IE(~io_pad_oe[59]), .CS(1'b0), .I(io_pad_o[59]), .OE(io_pad_oe[59]), .OD(1'b0), .PU(io_pad_pu[1]), .PD(io_pad_pd[1]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad60 (.C(io_pad_i[60]), .A(), .PAD(io_pad60), .IE(~io_pad_oe[60]), .CS(1'b0), .I(io_pad_o[60]), .OE(io_pad_oe[60]), .OD(1'b0), .PU(io_pad_pu[2]), .PD(io_pad_pd[2]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad61 (.C(io_pad_i[61]), .A(), .PAD(io_pad61), .IE(~io_pad_oe[61]), .CS(1'b0), .I(io_pad_o[61]), .OE(io_pad_oe[61]), .OD(1'b0), .PU(io_pad_pu[3]), .PD(io_pad_pd[3]), .DS0(1'b0), .DS1(1'b1));  
P65_1233_PBMUX u_io_pad62 (.C(io_pad_i[62]), .A(), .PAD(io_pad62), .IE(~io_pad_oe[62]), .CS(1'b0), .I(io_pad_o[62]), .OE(io_pad_oe[62]), .OD(1'b0), .PU(io_pad_pu[4]), .PD(io_pad_pd[4]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad63 (.C(io_pad_i[63]), .A(), .PAD(io_pad63), .IE(~io_pad_oe[63]), .CS(1'b0), .I(io_pad_o[63]), .OE(io_pad_oe[63]), .OD(1'b0), .PU(io_pad_pu[5]), .PD(io_pad_pd[5]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad64 (.C(io_pad_i[64]), .A(), .PAD(io_pad64), .IE(~io_pad_oe[64]), .CS(1'b0), .I(io_pad_o[64]), .OE(io_pad_oe[64]), .OD(1'b0), .PU(io_pad_pu[6]), .PD(io_pad_pd[6]), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad65 (.C(io_pad_i[65]), .A(), .PAD(io_pad65), .IE(~io_pad_oe[65]), .CS(1'b0), .I(io_pad_o[65]), .OE(io_pad_oe[65]), .OD(1'b0), .PU(io_pad_pu[7]), .PD(io_pad_pd[7]), .DS0(1'b0), .DS1(1'b1));

P65_1233_PBMUX u_io_pad66 (.C(io_pad_i[66]), .A(), .PAD(io_pad66), .IE(~io_pad_oe[66]), .CS(1'b0), .I(io_pad_o[66]), .OE(io_pad_oe[66]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad67 (.C(io_pad_i[67]), .A(), .PAD(io_pad67), .IE(~io_pad_oe[67]), .CS(1'b0), .I(io_pad_o[67]), .OE(io_pad_oe[67]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad68 (.C(io_pad_i[68]), .A(), .PAD(io_pad68), .IE(~io_pad_oe[68]), .CS(1'b0), .I(io_pad_o[68]), .OE(io_pad_oe[68]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad69 (.C(io_pad_i[69]), .A(), .PAD(io_pad69), .IE(~io_pad_oe[69]), .CS(1'b0), .I(io_pad_o[69]), .OE(io_pad_oe[69]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad70 (.C(io_pad_i[70]), .A(), .PAD(io_pad70), .IE(~io_pad_oe[70]), .CS(1'b0), .I(io_pad_o[70]), .OE(io_pad_oe[70]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad71 (.C(io_pad_i[71]), .A(), .PAD(io_pad71), .IE(~io_pad_oe[71]), .CS(1'b0), .I(io_pad_o[71]), .OE(io_pad_oe[71]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad72 (.C(io_pad_i[72]), .A(), .PAD(io_pad72), .IE(~io_pad_oe[72]), .CS(1'b0), .I(io_pad_o[72]), .OE(io_pad_oe[72]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad73 (.C(io_pad_i[73]), .A(), .PAD(io_pad73), .IE(~io_pad_oe[73]), .CS(1'b0), .I(io_pad_o[73]), .OE(io_pad_oe[73]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad74 (.C(io_pad_i[74]), .A(), .PAD(io_pad74), .IE(~io_pad_oe[74]), .CS(1'b0), .I(io_pad_o[74]), .OE(io_pad_oe[74]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad75 (.C(io_pad_i[75]), .A(), .PAD(io_pad75), .IE(~io_pad_oe[75]), .CS(1'b0), .I(io_pad_o[75]), .OE(io_pad_oe[75]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad76 (.C(io_pad_i[76]), .A(), .PAD(io_pad76), .IE(~io_pad_oe[76]), .CS(1'b0), .I(io_pad_o[76]), .OE(io_pad_oe[76]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad77 (.C(io_pad_i[77]), .A(), .PAD(io_pad77), .IE(~io_pad_oe[77]), .CS(1'b0), .I(io_pad_o[77]), .OE(io_pad_oe[77]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad78 (.C(io_pad_i[78]), .A(), .PAD(io_pad78), .IE(~io_pad_oe[78]), .CS(1'b0), .I(io_pad_o[78]), .OE(io_pad_oe[78]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad79 (.C(io_pad_i[79]), .A(), .PAD(io_pad79), .IE(~io_pad_oe[79]), .CS(1'b0), .I(io_pad_o[79]), .OE(io_pad_oe[79]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad80 (.C(io_pad_i[80]), .A(), .PAD(io_pad80), .IE(~io_pad_oe[80]), .CS(1'b0), .I(io_pad_o[80]), .OE(io_pad_oe[80]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));
P65_1233_PBMUX u_io_pad81 (.C(io_pad_i[81]), .A(), .PAD(io_pad81), .IE(~io_pad_oe[81]), .CS(1'b0), .I(io_pad_o[81]), .OE(io_pad_oe[81]), .OD(1'b0), .PU(1'b0), .PD(1'b0), .DS0(1'b0), .DS1(1'b1));

endmodule