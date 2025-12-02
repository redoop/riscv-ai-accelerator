// Copyright (c) 2023-2025 Yuchi Miao <miaoyuchi@ict.ac.cn>
// common is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//             http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

module xchecker #(
    parameter DATA_WIDTH = 1
) (
    input logic                  clk_i,
    input logic [DATA_WIDTH-1:0] dat_i
);

  XCHECKER :
  assert property (@(posedge clk_i) ((^(dat_i)) !== 1'bx))
  else
    $fatal("\n [error]: detected x value here which can lead to the x state propagation problem\n");

endmodule
