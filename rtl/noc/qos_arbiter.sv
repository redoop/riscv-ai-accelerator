// QoS-aware Arbiter for NoC
// Implements priority-based arbitration with fairness guarantees

`include "noc_packet.sv"

module qos_arbiter #(
    parameter int WIDTH = 4,
    parameter int QOS_LEVELS = 4
) (
    input  logic clk,
    input  logic rst_n,
    
    // Request inputs with QoS levels
    input  logic [WIDTH-1:0] request,
    input  qos_level_t qos_level[WIDTH],
    
    // Fairness and aging controls
    input  logic [7:0] aging_threshold,
    input  logic fairness_enable,
    
    // Grant output
    output logic [WIDTH-1:0] grant,
    
    // Status outputs
    output logic [WIDTH-1:0] starved_requests,
    output logic [31:0] total_grants[QOS_LEVELS],
    output logic [7:0] max_wait_time
);

    // Age counters for fairness
    logic [7:0] age_counter[WIDTH];
    logic [WIDTH-1:0] aged_requests;
    
    // QoS priority matrices
    logic [WIDTH-1:0] qos_requests[QOS_LEVELS];
    logic [WIDTH-1:0] qos_grants[QOS_LEVELS];
    
    // Round-robin state for each QoS level
    logic [WIDTH-1:0] rr_priority[QOS_LEVELS];
    
    // Performance counters
    logic [31:0] grant_counters[QOS_LEVELS];
    logic [7:0] current_max_wait;
    
    // Separate requests by QoS level
    genvar i, q;
    generate
        for (q = 0; q < QOS_LEVELS; q++) begin : gen_qos_separation
            always_comb begin
                qos_requests[q] = 0;
                for (int j = 0; j < WIDTH; j++) begin
                    if (request[j] && (qos_level[j] == qos_level_t'(q))) begin
                        qos_requests[q][j] = 1'b1;
                    end
                end
            end
        end
    endgenerate
    
    // Age tracking for fairness
    generate
        for (i = 0; i < WIDTH; i++) begin : gen_aging
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    age_counter[i] <= 0;
                end else begin
                    if (grant[i]) begin
                        age_counter[i] <= 0;  // Reset age on grant
                    end else if (request[i]) begin
                        if (age_counter[i] < 8'hFF) begin
                            age_counter[i] <= age_counter[i] + 1;
                        end
                    end else begin
                        age_counter[i] <= 0;  // Reset age when not requesting
                    end
                end
            end
            
            assign aged_requests[i] = (age_counter[i] >= aging_threshold);
        end
    endgenerate
    
    // Round-robin arbiters for each QoS level
    generate
        for (q = 0; q < QOS_LEVELS; q++) begin : gen_qos_arbiters
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rr_priority[q] <= 1;
                end else if (|qos_grants[q]) begin
                    // Rotate priority after grant
                    rr_priority[q] <= {qos_grants[q][WIDTH-2:0], qos_grants[q][WIDTH-1]};
                end
            end
            
            // Priority-based round-robin within QoS level
            logic [WIDTH-1:0] masked_req, unmasked_grant, masked_grant;
            
            assign masked_req = qos_requests[q] & ~(rr_priority[q] - 1);
            
            priority_encoder #(.WIDTH(WIDTH)) pe_masked (
                .request(masked_req),
                .grant(masked_grant)
            );
            
            priority_encoder #(.WIDTH(WIDTH)) pe_unmasked (
                .request(qos_requests[q]),
                .grant(unmasked_grant)
            );
            
            assign qos_grants[q] = |masked_req ? masked_grant : unmasked_grant;
        end
    endgenerate
    
    // Final grant selection with QoS priority and fairness
    always_comb begin
        grant = 0;
        
        if (fairness_enable && |aged_requests) begin
            // Prioritize aged requests for fairness
            for (int i = 0; i < WIDTH; i++) begin
                if (aged_requests[i] && request[i]) begin
                    grant[i] = 1'b1;
                    break;
                end
            end
        end else begin
            // Normal QoS-based arbitration (highest priority first)
            for (int q = QOS_LEVELS-1; q >= 0; q--) begin
                if (|qos_grants[q]) begin
                    grant = qos_grants[q];
                    break;
                end
            end
        end
    end
    
    // Performance monitoring
    generate
        for (q = 0; q < QOS_LEVELS; q++) begin : gen_perf_counters
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    grant_counters[q] <= 0;
                end else begin
                    if (|qos_grants[q]) begin
                        grant_counters[q] <= grant_counters[q] + 1;
                    end
                end
            end
            
            assign total_grants[q] = grant_counters[q];
        end
    endgenerate
    
    // Starvation detection
    assign starved_requests = aged_requests & request;
    
    // Maximum wait time calculation
    always_comb begin
        current_max_wait = 0;
        for (int i = 0; i < WIDTH; i++) begin
            if (request[i] && age_counter[i] > current_max_wait) begin
                current_max_wait = age_counter[i];
            end
        end
    end
    
    assign max_wait_time = current_max_wait;

endmodule