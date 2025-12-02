// Copyright (c) 2023-2025 Yuchi Miao <miaoyuchi@ict.ac.cn>
// common is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//             http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

module rst_sync #(
    parameter int STAGE = 3
) (
    input  logic clk_i,
    input  logic rst_n_i,
    output logic rst_n_o
);

  logic [STAGE-1:0] s_rst_sync;
  for (genvar i = 0; i < STAGE; i++) begin
    if (i == 0) begin : RST_SYNC_0_BLOCK
      dffr #(1) u_sync_dffr (
          clk_i,
          rst_n_i,
          1'b1,
          s_rst_sync[0]
      );
    end else begin : RST_SYNC_N0_BLOCK
      dffr #(1) u_sync_dffr (
          clk_i,
          rst_n_i,
          s_rst_sync[i-1],
          s_rst_sync[i]
      );
    end
  end

  assign rst_n_o = s_rst_sync[STAGE-1];

endmodule
