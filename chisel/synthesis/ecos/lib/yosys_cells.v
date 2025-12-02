// Yosys internal cells behavioral models for Icarus Verilog

module \$_DLATCH_P_ (E, D, Q);
  input E, D;
  output Q;
  reg q_reg;
  
  always @(E or D) begin
    if (E)
      q_reg = D;
  end
  
  assign Q = q_reg;
endmodule
