// NoC Performance and Fairness Monitor
// Comprehensive monitoring of network performance, QoS, and fairness metrics

`include "noc_packet.sv"

module noc_performance_monitor #(
    parameter int MESH_SIZE_X = 4,
    parameter int MESH_SIZE_Y = 4,
    parameter int NUM_ROUTERS = MESH_SIZE_X * MESH_SIZE_Y,
    parameter int MONITOR_WINDOW = 1024  // Monitoring window in cycles
) (
    input  logic clk,
    input  logic rst_n,
    
    // Router performance inputs
    input  logic [31:0] packets_routed[NUM_ROUTERS],
    input  logic [31:0] buffer_occupancy[NUM_ROUTERS][5],
    input  logic [31:0] qos_grants[NUM_ROUTERS][4],
    input  logic [7:0]  allocation_efficiency[NUM_ROUTERS],
    input  logic        fairness_violation[NUM_ROUTERS],
    input  logic        congestion_detected[NUM_ROUTERS],
    
    // Network-wide metrics
    output logic [31:0] total_throughput,
    output logic [15:0] average_latency,
    output logic [7:0]  network_efficiency,
    output logic [7:0]  qos_compliance_score,
    output logic [7:0]  fairness_index,
    
    // Per-QoS metrics
    output logic [31:0] qos_throughput[4],
    output logic [15:0] qos_avg_latency[4],
    output logic [7:0]  qos_violation_rate[4],
    
    // Hotspot and congestion analysis
    output logic [15:0] congestion_map,
    output logic [3:0]  worst_congested_router,
    output logic [7:0]  congestion_severity,
    
    // Fairness metrics
    output logic [7:0]  jain_fairness_index,
    output logic [15:0] starved_flows,
    output logic [31:0] max_packet_delay,
    
    // Performance alerts
    output logic        performance_degradation,
    output logic        fairness_alert,
    output logic        congestion_alert
);

    // Internal monitoring state
    logic [31:0] cycle_counter;
    logic [31:0] window_start_cycle;
    logic [31:0] packets_in_window[NUM_ROUTERS];
    logic [31:0] prev_packets_routed[NUM_ROUTERS];
    
    // Latency tracking (simplified model)
    logic [15:0] packet_latencies[256];  // Circular buffer for latency samples
    logic [7:0]  latency_write_ptr;
    logic [31:0] total_latency_sum;
    logic [15:0] latency_sample_count;
    
    // QoS performance tracking
    logic [31:0] qos_packets_window[4];
    logic [31:0] qos_violations[4];
    logic [31:0] prev_qos_grants[NUM_ROUTERS][4];
    
    // Fairness calculation variables
    logic [63:0] throughput_sum;
    logic [63:0] throughput_sum_squares;
    logic [31:0] active_routers;
    
    // Congestion tracking
    logic [7:0]  congestion_duration[NUM_ROUTERS];
    logic [31:0] congestion_cycles[NUM_ROUTERS];
    
    // Performance thresholds
    localparam logic [7:0] MIN_EFFICIENCY = 8'd70;
    localparam logic [7:0] MIN_FAIRNESS = 8'd60;
    localparam logic [7:0] MAX_CONGESTION = 8'd80;
    
    // Cycle counter and window management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 0;
            window_start_cycle <= 0;
        end else begin
            cycle_counter <= cycle_counter + 1;
            
            // Reset monitoring window periodically
            if ((cycle_counter - window_start_cycle) >= MONITOR_WINDOW) begin
                window_start_cycle <= cycle_counter;
            end
        end
    end
    
    // Throughput calculation
    genvar r;
    generate
        for (r = 0; r < NUM_ROUTERS; r++) begin : gen_throughput_calc
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    packets_in_window[r] <= 0;
                    prev_packets_routed[r] <= 0;
                    for (int q = 0; q < 4; q++) begin
                        prev_qos_grants[r][q] <= 0;
                    end
                end else begin
                    if ((cycle_counter - window_start_cycle) == 0) begin
                        // Start of new window
                        packets_in_window[r] <= 0;
                        prev_packets_routed[r] <= packets_routed[r];
                        for (int q = 0; q < 4; q++) begin
                            prev_qos_grants[r][q] <= qos_grants[r][q];
                        end
                    end else begin
                        packets_in_window[r] <= packets_routed[r] - prev_packets_routed[r];
                    end
                end
            end
        end
    endgenerate
    
    // Calculate total throughput
    always_comb begin
        logic [31:0] sum;
        sum = 0;
        for (int router = 0; router < NUM_ROUTERS; router++) begin
            sum += packets_in_window[router];
        end
        total_throughput = sum;
    end
    
    // QoS throughput calculation
    genvar q;
    generate
        for (q = 0; q < 4; q++) begin : gen_qos_throughput
            always_comb begin
                logic [31:0] qos_sum;
                qos_sum = 0;
                for (int router = 0; router < NUM_ROUTERS; router++) begin
                    qos_sum += (qos_grants[router][q] - prev_qos_grants[router][q]);
                end
                qos_throughput[q] = qos_sum;
            end
        end
    endgenerate
    
    // Latency estimation (simplified - based on buffer occupancy)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latency_write_ptr <= 0;
            total_latency_sum <= 0;
            latency_sample_count <= 0;
            packet_latencies <= '{default: 0};
        end else begin
            // Sample latency every 16 cycles (simplified model)
            if (cycle_counter[3:0] == 4'b0000) begin
                logic [31:0] avg_occupancy;
                logic [15:0] estimated_latency;
                int router, port;
                
                avg_occupancy = 0;
                for (router = 0; router < NUM_ROUTERS; router++) begin
                    for (port = 0; port < 5; port++) begin
                        avg_occupancy += buffer_occupancy[router][port];
                    end
                end
                avg_occupancy = avg_occupancy / (NUM_ROUTERS * 5);
                
                // Estimate latency based on occupancy (simplified model)
                estimated_latency = 16'(avg_occupancy + 4);  // Base latency + queuing
                
                // Update circular buffer
                total_latency_sum <= total_latency_sum - 32'(packet_latencies[latency_write_ptr]) + 32'(estimated_latency);
                packet_latencies[latency_write_ptr] <= estimated_latency;
                latency_write_ptr <= latency_write_ptr + 1;
                
                if (latency_sample_count < 16'd256) begin
                    latency_sample_count <= latency_sample_count + 16'd1;
                end
            end
        end
    end
    
    // Calculate average latency
    always_comb begin
        if (latency_sample_count > 0) begin
            average_latency = 16'(total_latency_sum / 32'(latency_sample_count));
        end else begin
            average_latency = 16'd0;
        end
    end
    
    // Network efficiency calculation
    always_comb begin
        logic [31:0] efficiency_sum;
        int router;
        efficiency_sum = 0;
        for (router = 0; router < NUM_ROUTERS; router++) begin
            efficiency_sum += 32'(allocation_efficiency[router]);
        end
        network_efficiency = 8'(efficiency_sum / NUM_ROUTERS);
    end
    
    // Fairness calculations
    always_comb begin
        int router;
        throughput_sum = 0;
        throughput_sum_squares = 0;
        active_routers = 0;
        
        for (router = 0; router < NUM_ROUTERS; router++) begin
            if (packets_in_window[router] > 0) begin
                throughput_sum += 64'(packets_in_window[router]);
                throughput_sum_squares += 64'(packets_in_window[router]) * 64'(packets_in_window[router]);
                active_routers += 1;
            end
        end
    end
    
    // Jain's Fairness Index calculation
    always_comb begin
        logic [63:0] numerator, denominator;
        
        numerator = 64'd0;
        denominator = 64'd0;
        jain_fairness_index = 8'd100;  // Default value
        
        if (active_routers > 0 && throughput_sum_squares > 0) begin
            numerator = throughput_sum * throughput_sum;
            denominator = 64'(active_routers) * throughput_sum_squares;
            
            if (denominator > 0) begin
                jain_fairness_index = 8'((numerator * 64'd100) / denominator);
            end else begin
                jain_fairness_index = 8'd100;
            end
        end else begin
            jain_fairness_index = 8'd100;  // Perfect fairness when no traffic
        end
    end
    
    // Congestion analysis
    generate
        for (r = 0; r < NUM_ROUTERS; r++) begin : gen_congestion_analysis
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    congestion_duration[r] <= 0;
                    congestion_cycles[r] <= 0;
                end else begin
                    if (congestion_detected[r]) begin
                        if (congestion_duration[r] < 8'hFF) begin
                            congestion_duration[r] <= congestion_duration[r] + 1;
                        end
                        congestion_cycles[r] <= congestion_cycles[r] + 1;
                    end else begin
                        congestion_duration[r] <= 0;
                    end
                end
            end
        end
    endgenerate
    
    // Find worst congested router
    always_comb begin
        logic [31:0] max_congestion_cycles;
        int router;
        max_congestion_cycles = 0;
        worst_congested_router = 0;
        
        for (router = 0; router < NUM_ROUTERS; router++) begin
            if (congestion_cycles[router] > max_congestion_cycles) begin
                max_congestion_cycles = congestion_cycles[router];
                worst_congested_router = 4'(router);
            end
        end
    end
    
    // Congestion severity calculation
    always_comb begin
        logic [31:0] total_congestion_cycles;
        int router;
        total_congestion_cycles = 0;
        
        for (router = 0; router < NUM_ROUTERS; router++) begin
            total_congestion_cycles += congestion_cycles[router];
        end
        
        if (cycle_counter > 0) begin
            congestion_severity = 8'((total_congestion_cycles * 100) / (cycle_counter * NUM_ROUTERS));
        end else begin
            congestion_severity = 0;
        end
    end
    
    // Generate congestion map
    generate
        for (r = 0; r < NUM_ROUTERS && r < 16; r++) begin : gen_congestion_map
            assign congestion_map[r] = congestion_detected[r];
        end
        
        // Pad remaining bits if NUM_ROUTERS < 16
        if (NUM_ROUTERS < 16) begin
            assign congestion_map[15:NUM_ROUTERS] = 0;
        end
    endgenerate
    
    // QoS compliance and violation tracking
    generate
        for (q = 0; q < 4; q++) begin : gen_qos_compliance
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    qos_violations[q] <= 0;
                end else begin
                    // Simple violation detection based on expected service levels
                    logic [31:0] expected_grants, actual_grants, total_grants;
                    
                    total_grants = qos_throughput[0] + qos_throughput[1] + 
                                  qos_throughput[2] + qos_throughput[3];
                    
                    if (total_grants > 0) begin
                        case (q)
                            0: expected_grants = total_grants / 32'd10;      // QOS_LOW: 10%
                            1: expected_grants = (total_grants * 32'd3) / 32'd10; // QOS_NORMAL: 30%
                            2: expected_grants = (total_grants * 32'd4) / 32'd10; // QOS_HIGH: 40%
                            3: expected_grants = total_grants / 32'd5;        // QOS_URGENT: 20%
                        endcase
                        
                        actual_grants = qos_throughput[q];
                        
                        if (actual_grants < (expected_grants / 32'd2)) begin
                            qos_violations[q] <= qos_violations[q] + 32'd1;
                        end
                    end
                end
            end
            
            // Calculate violation rate
            always_comb begin
                if (cycle_counter > 0) begin
                    qos_violation_rate[q] = 8'((qos_violations[q] * 32'd100) / cycle_counter);
                end else begin
                    qos_violation_rate[q] = 8'd0;
                end
            end
        end
    endgenerate
    
    // Overall QoS compliance score
    always_comb begin
        logic [31:0] total_violations;
        total_violations = qos_violations[0] + qos_violations[1] + 
                          qos_violations[2] + qos_violations[3];
        
        if (cycle_counter > 0) begin
            qos_compliance_score = 8'd100 - 8'((total_violations * 32'd100) / (cycle_counter * 32'd4));
        end else begin
            qos_compliance_score = 8'd100;
        end
    end
    
    // Fairness index calculation (simplified)
    assign fairness_index = jain_fairness_index;
    
    // Performance alerts
    assign performance_degradation = (network_efficiency < MIN_EFFICIENCY);
    logic any_fairness_violation;
    always_comb begin
        int router;
        any_fairness_violation = 1'b0;
        for (router = 0; router < NUM_ROUTERS; router++) begin
            if (fairness_violation[router]) begin
                any_fairness_violation = 1'b1;
                break;
            end
        end
    end
    
    assign fairness_alert = (fairness_index < MIN_FAIRNESS) || any_fairness_violation;
    assign congestion_alert = (congestion_severity > MAX_CONGESTION);
    
    // Additional outputs
    assign starved_flows = 16'(0);  // Placeholder - would need flow tracking
    assign max_packet_delay = 32'(average_latency);  // Simplified
    
    // QoS average latency (simplified - same as overall for now)
    genvar ql;
    generate
        for (ql = 0; ql < 4; ql++) begin : gen_qos_latency
            assign qos_avg_latency[ql] = average_latency;
        end
    endgenerate

endmodule