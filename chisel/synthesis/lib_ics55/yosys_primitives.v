// Yosys primitive cells for Icarus Verilog simulation
// These are behavioral models of Yosys internal cells

// Positive level-sensitive D-latch
// When E is high, Q follows D
// When E is low, Q holds its value
module \$_DLATCH_P_ (
    input D,
    input E,
    output reg Q
);
    always @* begin
        if (E)
            Q = D;
    end
endmodule
