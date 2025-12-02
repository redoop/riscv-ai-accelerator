`timescale 1ns / 1ps

module soc_tb ();
  localparam real OSC_CLK_25M_PEROID = 40;

  wire ext_rst_n_i_pad;
  wire osc_clk_25m_i_pad;
  wire old_ip_spi_flash_clk_pad;
  wire [1:0] old_ip_spi_flash_cs_pad;
  wire old_ip_spi_flash_mosi_pad;
  wire old_ip_spi_flash_miso_pad;
  wire old_ip_uart_rx_pad;
  wire old_ip_uart_tx_pad;

  logic r_osc_clk_25m, r_ext_rst_n;
  logic [2:0] r_ip_sel;

  assign osc_clk_25m_i_pad            = r_osc_clk_25m;
  assign ext_rst_n_i_pad              = r_ext_rst_n;

  asic_top u_asic_top (
      .ip_sel_pad0       (r_ip_sel[0]),
      .ip_sel_pad1       (r_ip_sel[1]),
      .ip_sel_pad2       (r_ip_sel[2]),
      .sys_clk_i_pad     (osc_clk_25m_i_pad),
      .sys_clk_o_pad     (),
      .rst_n_pad         (ext_rst_n_i_pad),
      .io_pad0           (old_ip_uart_rx_pad),
      .io_pad1           (old_ip_uart_tx_pad),
      .io_pad2           (old_ip_spi_flash_clk_pad),
      .io_pad3           (old_ip_spi_flash_cs_pad[0]),
      .io_pad4           (old_ip_spi_flash_cs_pad[1]),
      .io_pad5           (),
      .io_pad6           (),
      .io_pad7           (),
      .io_pad8           (),
      .io_pad9           (),
      .io_pad10          (),
      .io_pad11          (old_ip_spi_flash_mosi_pad),
      .io_pad12          (old_ip_spi_flash_miso_pad),
      .io_pad13          (1'b0),
      .io_pad14          (1'b0),
      .io_pad15          (1'b0),
      .io_pad16          (),
      .io_pad17          (),
      .io_pad18          (),
      .io_pad19          (),
      .io_pad20          (),
      .io_pad21          (),
      .io_pad22          (),
      .io_pad23          (),
      .io_pad24          (),
      .io_pad25          (),
      .io_pad26          (),
      .io_pad27          (),
      .io_pad28          (),
      .io_pad29          (),
      .io_pad30          (),
      .io_pad31          (),
      .io_pad32          (),
      .io_pad33          (),
      .io_pad34          (),
      .io_pad35          (),
      .io_pad36          (),
      .io_pad37          (),
      .io_pad38          (),
      .io_pad39          (),
      .io_pad40          (),
      .io_pad41          (),
      .io_pad42          (),
      .io_pad43          (),
      .io_pad44          (),
      .io_pad45          (),
      .io_pad46          (),
      .io_pad47          (),
      .io_pad48          (),
      .io_pad49          (),
      .io_pad50          (),
      .io_pad51          (),
      .io_pad52          (),
      .io_pad53          (),
      .io_pad54          (),
      .io_pad55          (),
      .io_pad56          (),
      .io_pad57          (),
      .io_pad58          (),
      .io_pad59          (),
      .io_pad60          (),
      .io_pad61          (),
      .io_pad62          (),
      .io_pad63          (),
      .io_pad64          (),
      .io_pad65          (),
      .io_pad66          (),
      .io_pad67          (),
      .io_pad68          (),
      .io_pad69          (),
      .io_pad70          (),
      .io_pad71          (),
      .io_pad72          (),
      .io_pad73          (),
      .io_pad74          (),
      .io_pad75          (),
      .io_pad76          (),
      .io_pad77          (),
      .io_pad78          (),
      .io_pad79          (),
      .io_pad80          (),
      .io_pad81          ()
  );

  N25Qxxx u_old_ip_spi_N25Qxxx (
      .C_       (old_ip_spi_flash_clk_pad),
      .S        (old_ip_spi_flash_cs_pad[0]),
      .DQ0      (old_ip_spi_flash_mosi_pad),
      .DQ1      (old_ip_spi_flash_miso_pad),
      .HOLD_DQ3 (),
      .Vpp_W_DQ2(),
      .Vcc      ('d3000)
  );

  tty #(115200, 0) u_old_ip_uart_tty (
      .STX(old_ip_uart_rx_pad),
      .SRX(old_ip_uart_tx_pad)
  );

  task sim_reset(int delay);
    r_ext_rst_n = 1'b0;
    repeat (delay) @(posedge osc_clk_25m_i_pad);
    #1 r_ext_rst_n = 1'b1;
  endtask

  initial begin
    r_osc_clk_25m = 1'b0;
    forever begin
      #(OSC_CLK_25M_PEROID / 2) r_osc_clk_25m <= ~r_osc_clk_25m;
    end
  end
    
  initial begin
      sim_reset(400);
      #105852000
      while (1) begin
        
      end
  end

    initial begin
        if      ($test$plusargs("ip_sel00")) r_ip_sel = 3'd0;
        else if ($test$plusargs("ip_sel01")) r_ip_sel = 3'd1;
        else if ($test$plusargs("ip_sel02")) r_ip_sel = 3'd2;
        else if ($test$plusargs("ip_sel03")) r_ip_sel = 3'd3;
        else if ($test$plusargs("ip_sel04")) r_ip_sel = 3'd4;
        else if ($test$plusargs("ip_sel05")) r_ip_sel = 3'd5;

        if ($test$plusargs("dump_all")) begin
          $fsdbDumpfile("soc_tb.fsdb");
          $fsdbDumpvars(0, soc_tb, "+all");
        end

        if      ($test$plusargs("asm-flash"))       #1000000    $finish;
        else if ($test$plusargs("hello-flash"))     #1000000    $finish;
        else if ($test$plusargs("hello-mem"))     #40000000   $finish;
        else if ($test$plusargs("hello-sdram"))   #40000000   $finish;
        else if ($test$plusargs("memtest-flash"))   #600000000  $finish;
        else if ($test$plusargs("memtest-mem"))     #120000000  $finish;
        else if ($test$plusargs("memtest-sdram")) #60000000   $finish;
        else if ($test$plusargs("rtthread-flash"))  #1600000000 $finish;
        else if ($test$plusargs("rtthread-mem"))    #200000000  $finish;
        else if ($test$plusargs("rtthread-sdram"))  #200000000  $finish;

    end

endmodule