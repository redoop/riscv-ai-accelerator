// 测试工具模块
// 提供常用的测试辅助功能

`timescale 1ns/1ps

package test_utils;

  // 颜色定义 (用于终端输出)
  parameter string COLOR_RED    = "\033[0;31m";
  parameter string COLOR_GREEN  = "\033[0;32m";
  parameter string COLOR_YELLOW = "\033[1;33m";
  parameter string COLOR_BLUE   = "\033[0;34m";
  parameter string COLOR_RESET  = "\033[0m";
  
  // 打印带颜色的消息
  function void print_pass(string msg);
    $display("%s✓ %s%s", COLOR_GREEN, msg, COLOR_RESET);
  endfunction
  
  function void print_fail(string msg);
    $display("%s✗ %s%s", COLOR_RED, msg, COLOR_RESET);
  endfunction
  
  function void print_warn(string msg);
    $display("%s⚠ %s%s", COLOR_YELLOW, msg, COLOR_RESET);
  endfunction
  
  function void print_info(string msg);
    $display("%s• %s%s", COLOR_BLUE, msg, COLOR_RESET);
  endfunction
  
  // 比较两个值
  function bit compare_values(
    input logic [31:0] actual,
    input logic [31:0] expected,
    input string name
  );
    if (actual === expected) begin
      print_pass($sformatf("%s: 0x%08h (正确)", name, actual));
      return 1;
    end else begin
      print_fail($sformatf("%s: 0x%08h (期望 0x%08h)", 
                          name, actual, expected));
      return 0;
    end
  endfunction
  
  // 等待信号变化
  task wait_for_signal(
    ref logic signal,
    input logic expected_value,
    input integer timeout_cycles,
    ref logic clk
  );
    integer count = 0;
    while (signal !== expected_value && count < timeout_cycles) begin
      @(posedge clk);
      count++;
    end
    
    if (count >= timeout_cycles) begin
      print_fail($sformatf("等待信号超时 (%0d 周期)", timeout_cycles));
    end else begin
      print_pass($sformatf("信号在 %0d 周期后变化", count));
    end
  endtask
  
  // 生成随机数据
  function logic [31:0] random_data();
    return $random;
  endfunction
  
  // 计算 CRC32
  function logic [31:0] crc32(logic [31:0] data);
    logic [31:0] crc = 32'hFFFFFFFF;
    for (int i = 0; i < 32; i++) begin
      if ((crc[31] ^ data[i]) == 1'b1)
        crc = (crc << 1) ^ 32'h04C11DB7;
      else
        crc = crc << 1;
    end
    return ~crc;
  endfunction
  
  // 打印分隔线
  function void print_separator(string title = "");
    if (title == "") begin
      $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    end else begin
      $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      $display("%s", title);
      $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    end
  endfunction
  
  // 打印测试头部
  function void print_test_header(string test_name, integer test_num);
    $display("");
    print_separator($sformatf("测试 %0d: %s", test_num, test_name));
  endfunction
  
  // 打印测试结果
  function void print_test_result(bit passed);
    if (passed)
      print_pass("测试通过");
    else
      print_fail("测试失败");
    $display("");
  endfunction

endpackage

// UART 发送器模型
module uart_tx_model #(
  parameter BAUD_RATE = 115200,
  parameter CLK_FREQ = 100000000
) (
  input  logic       clk,
  input  logic       reset,
  input  logic [7:0] data,
  input  logic       valid,
  output logic       ready,
  output logic       tx
);

  localparam CYCLES_PER_BIT = CLK_FREQ / BAUD_RATE;
  
  typedef enum logic [1:0] {
    IDLE,
    START,
    DATA,
    STOP
  } state_t;
  
  state_t state;
  integer bit_count;
  integer cycle_count;
  logic [7:0] data_reg;
  
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      state <= IDLE;
      tx <= 1'b1;
      ready <= 1'b1;
      bit_count <= 0;
      cycle_count <= 0;
    end else begin
      case (state)
        IDLE: begin
          tx <= 1'b1;
          ready <= 1'b1;
          if (valid) begin
            data_reg <= data;
            ready <= 1'b0;
            state <= START;
            cycle_count <= 0;
          end
        end
        
        START: begin
          tx <= 1'b0;  // 起始位
          if (cycle_count >= CYCLES_PER_BIT - 1) begin
            state <= DATA;
            bit_count <= 0;
            cycle_count <= 0;
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
        
        DATA: begin
          tx <= data_reg[bit_count];
          if (cycle_count >= CYCLES_PER_BIT - 1) begin
            if (bit_count >= 7) begin
              state <= STOP;
              cycle_count <= 0;
            end else begin
              bit_count <= bit_count + 1;
              cycle_count <= 0;
            end
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
        
        STOP: begin
          tx <= 1'b1;  // 停止位
          if (cycle_count >= CYCLES_PER_BIT - 1) begin
            state <= IDLE;
            ready <= 1'b1;
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
      endcase
    end
  end

endmodule

// UART 接收器模型
module uart_rx_model #(
  parameter BAUD_RATE = 115200,
  parameter CLK_FREQ = 100000000
) (
  input  logic       clk,
  input  logic       reset,
  input  logic       rx,
  output logic [7:0] data,
  output logic       valid
);

  localparam CYCLES_PER_BIT = CLK_FREQ / BAUD_RATE;
  
  typedef enum logic [1:0] {
    IDLE,
    START,
    DATA,
    STOP
  } state_t;
  
  state_t state;
  integer bit_count;
  integer cycle_count;
  logic [7:0] data_reg;
  
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      state <= IDLE;
      valid <= 1'b0;
      bit_count <= 0;
      cycle_count <= 0;
    end else begin
      valid <= 1'b0;
      
      case (state)
        IDLE: begin
          if (rx == 1'b0) begin  // 检测起始位
            state <= START;
            cycle_count <= 0;
          end
        end
        
        START: begin
          if (cycle_count >= CYCLES_PER_BIT/2) begin
            if (rx == 1'b0) begin  // 确认起始位
              state <= DATA;
              bit_count <= 0;
              cycle_count <= 0;
            end else begin
              state <= IDLE;
            end
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
        
        DATA: begin
          if (cycle_count >= CYCLES_PER_BIT - 1) begin
            data_reg[bit_count] <= rx;
            if (bit_count >= 7) begin
              state <= STOP;
              cycle_count <= 0;
            end else begin
              bit_count <= bit_count + 1;
              cycle_count <= 0;
            end
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
        
        STOP: begin
          if (cycle_count >= CYCLES_PER_BIT - 1) begin
            if (rx == 1'b1) begin  // 确认停止位
              data <= data_reg;
              valid <= 1'b1;
            end
            state <= IDLE;
          end else begin
            cycle_count <= cycle_count + 1;
          end
        end
      endcase
    end
  end

endmodule
