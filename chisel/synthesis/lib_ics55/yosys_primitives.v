// Yosys Internal Primitives for Icarus Verilog Simulation

// D-Latch with positive enable
module \$_DLATCH_P_ (E, D, Q);
  input E, D;
  output reg Q;
  
  always @* begin
    if (E)
      Q = D;
  end
endmodule
