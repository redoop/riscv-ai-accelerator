// Copyright (C) 2022 ETH Zurich, University of Bologna Copyright and related
// rights are licensed under the Solderpad Hardware License, Version 0.51 (the
// "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law or
// agreed to in writing, software, hardware and materials distributed under this
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
// OF ANY KIND, either express or implied. See the License for the specific
// language governing permissions and limitations under the License.
// SPDX-License-Identifier: SHL-0.51
//
// -- Adaptable modifications are redistributed under compatible License --
//
// Copyright (c) 2023-2025 Yuchi Miao <miaoyuchi@ict.ac.cn>
// common is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//             http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// div_val: 2 ^ DIV_VALUE_WIDTH
module clk_int_even_div_static #(
    parameter int DIV_VALUE_WIDTH = 2
) (
    input  logic clk_i,
    input  logic rst_n_i,
    output logic clk_o
);
  // if (DIV_VALUE <= 0 || DIV_VALUE % 2) begin
  //   $error("DIV_VALUE must be strictly larger than 0 and be even value");
  // end

  logic [DIV_VALUE_WIDTH-1:0] s_cnt_d, s_cnt_q;
  logic s_clk_d, s_clk_q;

  assign s_cnt_d = s_cnt_q + 1'b1;
  dffr #(DIV_VALUE_WIDTH) u_cnt_dffr (
      clk_i,
      rst_n_i,
      s_cnt_d,
      s_cnt_q
  );

  assign s_clk_d = s_cnt_q[DIV_VALUE_WIDTH-1];
  dffr #(1) u_clk_dffr (
      clk_i,
      rst_n_i,
      s_clk_d,
      s_clk_q
  );

  //   clk_buf(s_clk_q, clk_o);
  assign clk_o = s_clk_q;
endmodule

module clk_int_odd_div_static #(
    parameter int DIV_VALUE = 3
) (
    input  logic clk_i,
    input  logic rst_n_i,
    output logic clk_o
);
  // if (DIV_VALUE < 2 || DIV_VALUE % 2 == 0) begin
  //   $error("DIV_VALUE must be strictly larger than 0 and be odd value");
  // end

  localparam int DIV_VALUE_WIDTH = $clog2(DIV_VALUE) + 1;
  logic [DIV_VALUE_WIDTH-1:0] s_cnt_d, s_cnt_q;
  logic s_clk1_d, s_clk1_q, s_clk2_d, s_clk2_q;

  always_comb begin
    s_cnt_d = s_cnt_q + 1'b1;
    if (s_cnt_q == DIV_VALUE - 1) begin
      s_cnt_d = '0;
    end
  end

  dffr #(DIV_VALUE_WIDTH) u_cnt_dffr (
      clk_i,
      rst_n_i,
      s_cnt_d,
      s_cnt_q
  );

  always_comb begin
    s_clk1_d = s_clk1_q;
    if (s_cnt_q == (DIV_VALUE - 1) / 2 - 1) begin
      s_clk1_d = ~s_clk1_q;
    end
  end

  dffr #(1) u_clk1_dffr (
      clk_i,
      rst_n_i,
      s_clk1_d,
      s_clk1_q
  );

  always_comb begin
    s_clk2_d = s_clk2_q;
    if (s_cnt_q == DIV_VALUE - 1) begin
      s_clk2_d = ~s_clk2_q;
    end
  end

  ndffr #(1) u_clk_ndffr (
      clk_i,
      rst_n_i,
      s_clk2_d,
      s_clk2_q
  );

  clk_xor2 u_clk_xor2 (
      s_clk1_q,
      s_clk2_q,
      clk_o
  );
endmodule

// NOTE: need to make sure the div_i is driven by reg
// div_val: (div_i + 1)
module clk_int_div_simple #(
    parameter int DIV_VALUE_WIDTH  = 32,
    parameter int DONE_DELAY_WIDTH = 3
) (
    input  logic                       clk_i,
    input  logic                       rst_n_i,
    input  logic [DIV_VALUE_WIDTH-1:0] div_i,
    input  logic                       clk_init_i,
    input  logic                       div_valid_i,
    output logic                       div_ready_o,
    output logic                       div_done_o,
    output logic [DIV_VALUE_WIDTH-1:0] clk_cnt_o,
    output logic                       clk_fir_trg_o,
    output logic                       clk_sec_trg_o,
    output logic                       clk_o
);

  logic [DIV_VALUE_WIDTH-1:0] s_cnt_d, s_cnt_q;
  logic [DONE_DELAY_WIDTH-1:0] s_div_done_d, s_div_done_q;
  logic s_clk_d, s_clk_q;
  logic div_hdshk;

  assign div_ready_o   = 1'b1;
  assign div_hdshk     = div_valid_i & div_ready_o;
  assign clk_cnt_o     = s_cnt_q;
  assign clk_fir_trg_o = div_i == '0 ? '0 : s_cnt_q == (div_i - 1) / 2;
  assign clk_sec_trg_o = s_cnt_q == div_i;

  always_comb begin
    s_cnt_d = s_cnt_q + 1'b1;
    if (div_hdshk || div_i == '0) s_cnt_d = '0;
    else if (clk_sec_trg_o) s_cnt_d = '0;
  end
  dffr #(DIV_VALUE_WIDTH) u_cnt_dffr (
      clk_i,
      rst_n_i,
      s_cnt_d,
      s_cnt_q
  );

  // if div_i == 0, clk_o = clk_i
  // if div_i == 1, clk_o = clk_i / 2 chg on s_cnt_q == 0
  // if div_i == 2, clk_o = clk_i / 3 chg on s_cnt_q == 0
  // if div_i == 3, clk_o = clk_i / 4 chg on s_cnt_q == 1
  assign clk_o = div_i == '0 ? clk_i : s_clk_q;
  always_comb begin
    if (div_hdshk) s_clk_d = clk_init_i;
    else if (clk_fir_trg_o || clk_sec_trg_o) s_clk_d = ~s_clk_q;
    else s_clk_d = s_clk_q;
  end
  dffr #(1) u_clk_dffr (
      clk_i,
      rst_n_i,
      s_clk_d,
      s_clk_q
  );

  assign div_done_o = s_div_done_q == {DONE_DELAY_WIDTH{1'b1}};
  always_comb begin
    s_div_done_d = s_div_done_q;
    if (div_hdshk) begin
      s_div_done_d = '0;
    end else if (clk_sec_trg_o && s_div_done_q < {DONE_DELAY_WIDTH{1'b1}}) begin
      s_div_done_d = s_div_done_q + 1'b1;
    end
  end
  dffr #(DONE_DELAY_WIDTH) u_ready_dffr (
      clk_i,
      rst_n_i,
      s_div_done_d,
      s_div_done_q
  );

endmodule

module clk_int_even_div #(
    parameter int DIV_VALUE             = 2,
    parameter bit ENABLE_CLOCK_IN_RESET = 1'b0
) (
    input  logic clk_i,
    input  logic rst_n_i,
    input  logic en_i,
    input  logic div_i,
    input  logic div_valid_i,
    output logic div_done_o,
    output logic clk_o
);

endmodule
