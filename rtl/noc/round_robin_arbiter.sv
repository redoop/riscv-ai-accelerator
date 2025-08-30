// Round-robin arbiter module
// Provides fair arbitration among multiple requesters

module round_robin_arbiter #(
    parameter int WIDTH = 4
) (
    input  logic clk,
    input  logic rst_n,
    input  logic [WIDTH-1:0] request,
    output logic [WIDTH-1:0] grant
);

    logic [WIDTH-1:0] priority_reg;
    logic [WIDTH-1:0] masked_request;
    logic [WIDTH-1:0] unmasked_grant;
    logic [WIDTH-1:0] masked_grant;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            priority_reg <= 1;
        end else if (|grant) begin
            priority_reg <= {grant[WIDTH-2:0], grant[WIDTH-1]};
        end
    end
    
    assign masked_request = request & ~(priority_reg - 1);
    
    priority_encoder #(.WIDTH(WIDTH)) pe_masked (
        .request(masked_request),
        .grant(masked_grant)
    );
    
    priority_encoder #(.WIDTH(WIDTH)) pe_unmasked (
        .request(request),
        .grant(unmasked_grant)
    );
    
    assign grant = |masked_request ? masked_grant : unmasked_grant;

endmodule