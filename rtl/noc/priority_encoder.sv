// Priority encoder module
// Encodes the highest priority request

module priority_encoder #(
    parameter int WIDTH = 4
) (
    input  logic [WIDTH-1:0] request,
    output logic [WIDTH-1:0] grant
);

    always_comb begin
        grant = 0;
        for (int i = 0; i < WIDTH; i++) begin
            if (request[i]) begin
                grant[i] = 1'b1;
                break;
            end
        end
    end

endmodule