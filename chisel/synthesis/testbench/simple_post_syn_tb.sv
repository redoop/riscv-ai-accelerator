// 简化的逻辑综合后网表仿真测试平台
`timescale 1ns/1ps

module post_syn_tb;

  // 时钟和复位
  logic clk;
  logic reset;
  
  // UART 接口
  logic uart_rx;
  logic uart_tx;
  
  // GPIO 接口 - 使用单独的信号
  logic gpio_out_0, gpio_out_1, gpio_out_2, gpio_out_3;
  logic gpio_out_4, gpio_out_5, gpio_out_6, gpio_out_7;
  logic gpio_out_8, gpio_out_9, gpio_out_10, gpio_out_11;
  logic gpio_out_12, gpio_out_13, gpio_out_14, gpio_out_15;
  logic gpio_out_16, gpio_out_17, gpio_out_18, gpio_out_19;
  logic gpio_out_20, gpio_out_21, gpio_out_22, gpio_out_23;
  logic gpio_out_24, gpio_out_25, gpio_out_26, gpio_out_27;
  logic gpio_out_28, gpio_out_29, gpio_out_30, gpio_out_31;
  
  logic gpio_in_0, gpio_in_1, gpio_in_2, gpio_in_3;
  logic gpio_in_4, gpio_in_5, gpio_in_6, gpio_in_7;
  logic gpio_in_8, gpio_in_9, gpio_in_10, gpio_in_11;
  logic gpio_in_12, gpio_in_13, gpio_in_14, gpio_in_15;
  logic gpio_in_16, gpio_in_17, gpio_in_18, gpio_in_19;
  logic gpio_in_20, gpio_in_21, gpio_in_22, gpio_in_23;
  logic gpio_in_24, gpio_in_25, gpio_in_26, gpio_in_27;
  logic gpio_in_28, gpio_in_29, gpio_in_30, gpio_in_31;
  
  // 中断信号
  logic compact_irq;
  logic bitnet_irq;
  logic trap;
  
  // 时钟生成
  initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100MHz 时钟
  end
  
  // 实例化 DUT (综合后的网表)
  SimpleEdgeAiSoC dut (
    .clock(clk),
    .reset(reset),
    .io_uart_rx(uart_rx),
    .io_uart_tx(uart_tx),
    .\io_gpio_out[0] (gpio_out_0),
    .\io_gpio_out[1] (gpio_out_1),
    .\io_gpio_out[2] (gpio_out_2),
    .\io_gpio_out[3] (gpio_out_3),
    .\io_gpio_out[4] (gpio_out_4),
    .\io_gpio_out[5] (gpio_out_5),
    .\io_gpio_out[6] (gpio_out_6),
    .\io_gpio_out[7] (gpio_out_7),
    .\io_gpio_out[8] (gpio_out_8),
    .\io_gpio_out[9] (gpio_out_9),
    .\io_gpio_out[10] (gpio_out_10),
    .\io_gpio_out[11] (gpio_out_11),
    .\io_gpio_out[12] (gpio_out_12),
    .\io_gpio_out[13] (gpio_out_13),
    .\io_gpio_out[14] (gpio_out_14),
    .\io_gpio_out[15] (gpio_out_15),
    .\io_gpio_out[16] (gpio_out_16),
    .\io_gpio_out[17] (gpio_out_17),
    .\io_gpio_out[18] (gpio_out_18),
    .\io_gpio_out[19] (gpio_out_19),
    .\io_gpio_out[20] (gpio_out_20),
    .\io_gpio_out[21] (gpio_out_21),
    .\io_gpio_out[22] (gpio_out_22),
    .\io_gpio_out[23] (gpio_out_23),
    .\io_gpio_out[24] (gpio_out_24),
    .\io_gpio_out[25] (gpio_out_25),
    .\io_gpio_out[26] (gpio_out_26),
    .\io_gpio_out[27] (gpio_out_27),
    .\io_gpio_out[28] (gpio_out_28),
    .\io_gpio_out[29] (gpio_out_29),
    .\io_gpio_out[30] (gpio_out_30),
    .\io_gpio_out[31] (gpio_out_31),
    .\io_gpio_in[0] (gpio_in_0),
    .\io_gpio_in[1] (gpio_in_1),
    .\io_gpio_in[2] (gpio_in_2),
    .\io_gpio_in[3] (gpio_in_3),
    .\io_gpio_in[4] (gpio_in_4),
    .\io_gpio_in[5] (gpio_in_5),
    .\io_gpio_in[6] (gpio_in_6),
    .\io_gpio_in[7] (gpio_in_7),
    .\io_gpio_in[8] (gpio_in_8),
    .\io_gpio_in[9] (gpio_in_9),
    .\io_gpio_in[10] (gpio_in_10),
    .\io_gpio_in[11] (gpio_in_11),
    .\io_gpio_in[12] (gpio_in_12),
    .\io_gpio_in[13] (gpio_in_13),
    .\io_gpio_in[14] (gpio_in_14),
    .\io_gpio_in[15] (gpio_in_15),
    .\io_gpio_in[16] (gpio_in_16),
    .\io_gpio_in[17] (gpio_in_17),
    .\io_gpio_in[18] (gpio_in_18),
    .\io_gpio_in[19] (gpio_in_19),
    .\io_gpio_in[20] (gpio_in_20),
    .\io_gpio_in[21] (gpio_in_21),
    .\io_gpio_in[22] (gpio_in_22),
    .\io_gpio_in[23] (gpio_in_23),
    .\io_gpio_in[24] (gpio_in_24),
    .\io_gpio_in[25] (gpio_in_25),
    .\io_gpio_in[26] (gpio_in_26),
    .\io_gpio_in[27] (gpio_in_27),
    .\io_gpio_in[28] (gpio_in_28),
    .\io_gpio_in[29] (gpio_in_29),
    .\io_gpio_in[30] (gpio_in_30),
    .\io_gpio_in[31] (gpio_in_31),
    .io_trap(trap),
    .io_compact_irq(compact_irq),
    .io_bitnet_irq(bitnet_irq)
  );
  
  // 测试序列
  initial begin
    // 初始化信号
    reset = 1;
    uart_rx = 1;
    {gpio_in_31, gpio_in_30, gpio_in_29, gpio_in_28,
     gpio_in_27, gpio_in_26, gpio_in_25, gpio_in_24,
     gpio_in_23, gpio_in_22, gpio_in_21, gpio_in_20,
     gpio_in_19, gpio_in_18, gpio_in_17, gpio_in_16,
     gpio_in_15, gpio_in_14, gpio_in_13, gpio_in_12,
     gpio_in_11, gpio_in_10, gpio_in_9, gpio_in_8,
     gpio_in_7, gpio_in_6, gpio_in_5, gpio_in_4,
     gpio_in_3, gpio_in_2, gpio_in_1, gpio_in_0} = 32'h0;
    
    // 打印测试信息
    $display("========================================");
    $display("逻辑综合后网表仿真测试");
    $display("========================================");
    $display("设计: SimpleEdgeAiSoC");
    $display("时钟频率: 100 MHz");
    $display("========================================");
    
    // 复位序列
    $display("[%0t] 开始复位...", $time);
    repeat(10) @(posedge clk);
    reset = 0;
    $display("[%0t] 复位完成", $time);
    
    // 测试 1: 系统启动
    $display("\n测试 1: 系统启动");
    repeat(100) @(posedge clk);
    
    if (!trap) begin
      $display("✓ 系统启动正常，无 trap");
    end else begin
      $display("✗ 检测到 trap 信号");
    end
    
    // 测试 2: GPIO 功能
    $display("\n测试 2: GPIO 功能");
    {gpio_in_31, gpio_in_30, gpio_in_29, gpio_in_28,
     gpio_in_27, gpio_in_26, gpio_in_25, gpio_in_24,
     gpio_in_23, gpio_in_22, gpio_in_21, gpio_in_20,
     gpio_in_19, gpio_in_18, gpio_in_17, gpio_in_16,
     gpio_in_15, gpio_in_14, gpio_in_13, gpio_in_12,
     gpio_in_11, gpio_in_10, gpio_in_9, gpio_in_8,
     gpio_in_7, gpio_in_6, gpio_in_5, gpio_in_4,
     gpio_in_3, gpio_in_2, gpio_in_1, gpio_in_0} = 32'hAAAA_AAAA;
    repeat(10) @(posedge clk);
    $display("GPIO 输入: 0xAAAAAAAA");
    
    // 测试 3: 长时间运行
    $display("\n测试 3: 长时间运行稳定性");
    repeat(1000) @(posedge clk);
    
    if (!trap) begin
      $display("✓ 系统运行稳定");
    end else begin
      $display("✗ 系统出现异常");
    end
    
    // 测试完成
    $display("\n========================================");
    $display("测试完成");
    $display("========================================");
    $display("总仿真时间: %0t ns", $time);
    $display("总时钟周期: %0d", $time/10);
    
    $finish;
  end
  
  // 监控关键信号
  always @(posedge clk) begin
    if (trap) begin
      $display("[%0t] 警告: 检测到 TRAP 信号!", $time);
    end
  end
  
  // 波形记录
  initial begin
    $dumpfile("synthesis/waves/post_syn.vcd");
    $dumpvars(0, post_syn_tb);
  end
  
  // 超时保护
  initial begin
    #100000;  // 100us 超时
    $display("仿真超时");
    $finish;
  end

endmodule
