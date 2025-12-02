module rcu (
    input  logic sys_clk_i,
    input  logic arst_n_i,  

    output logic clk_100m_o,  
    output logic clk_50m_o,   
    output logic clk_25m_o,   
    output logic rst_100m_n_o,
    output logic rst_50m_n_o, 
    output logic rst_25m_n_o  
);

  logic clk_50m_internal;
  logic clk_25m_internal;
  logic rst_100m_internal; 

  rst_sync #( .STAGE(3) ) u_rst_sync_100m (
      .clk_i(sys_clk_i),
      .rst_n_i(arst_n_i),
      .rst_n_o(rst_100m_internal)
  );

  clk_int_even_div_static #(
      .DIV_VALUE_WIDTH(1)
  ) u_div2 (
      .clk_i(sys_clk_i),
      .rst_n_i(arst_n_i),
      .clk_o(clk_50m_internal)
  );

  clk_int_even_div_static #(
      .DIV_VALUE_WIDTH(2)
  ) u_div4 (
      .clk_i(sys_clk_i),
      .rst_n_i(arst_n_i),
      .clk_o(clk_25m_internal)
  );

  tc_clk_buf u_buf_100m (.clk_i(sys_clk_i),       .clk_o(clk_100m_o));
  tc_clk_buf u_buf_50m  (.clk_i(clk_50m_internal), .clk_o(clk_50m_o));
  tc_clk_buf u_buf_25m  (.clk_i(clk_25m_internal), .clk_o(clk_25m_o));

  assign rst_100m_n_o = rst_100m_internal;

  rst_sync #( .STAGE(3) ) u_rst_sync_50m (
      .clk_i(clk_50m_o), 
      .rst_n_i(arst_n_i),
      .rst_n_o(rst_50m_n_o)
  );

  rst_sync #( .STAGE(3) ) u_rst_sync_25m (
      .clk_i(clk_25m_o), 
      .rst_n_i(arst_n_i),
      .rst_n_o(rst_25m_n_o)
  );
  
endmodule