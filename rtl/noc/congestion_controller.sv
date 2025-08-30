// Network Congestion Controller
// Monitors network congestion and implements flow control mechanisms

`include "noc_packet.sv"

module congestion_controller #(
    parameter int MESH_SIZE_X = 4,
    parameter int MESH_SIZE_Y = 4,
    parameter int NUM_ROUTERS = MESH_SIZE_X * MESH_SIZE_Y,
    parameter int CONGESTION_THRESHOLD = 75  // Percentage
) (
    input  logic clk,
    input  logic rst_n,
    
    // Router status inputs
    input  logic [31:0] buffer_occupancy[NUM_ROUTERS][5],  // [router][port]
    input  logic [31:0] packets_routed[NUM_ROUTERS],
    input  logic        router_congested[NUM_ROUTERS],
    
    // Flow control outputs
    output logic        global_flow_control,
    output logic [1:0]  throttle_level[NUM_ROUTERS],
    output logic        adaptive_routing_enable,
    
    // Congestion monitoring
    output logic [7:0]  network_utilization,
    output logic [31:0] total_packets_dropped,
    output logic [15:0] congestion_hotspots,
    
    // QoS enforcement
    output logic        qos_enforcement_active,
    output logic [1:0]  priority_boost[NUM_ROUTERS]
);

    // Internal congestion metrics
    logic [31:0] total_buffer_occupancy;
    logic [31:0] max_buffer_capacity;
    logic [7:0]  congestion_percentage;
    logic [15:0] congested_routers;
    
    // Temporal congestion tracking
    logic [7:0]  congestion_history[16];  // 16-cycle history
    logic [3:0]  history_index;
    logic [11:0] avg_congestion;
    
    // Packet drop counters
    logic [31:0] dropped_packets[4];  // Per QoS level
    logic [31:0] total_dropped;
    
    // Hotspot detection
    logic [7:0]  hotspot_threshold;
    logic [15:0] persistent_hotspots;
    logic [7:0]  hotspot_counter[NUM_ROUTERS];
    
    // Calculate total buffer occupancy
    always_comb begin
        total_buffer_occupancy = 0;
        max_buffer_capacity = NUM_ROUTERS * 5 * VC_COUNT * VC_DEPTH;
        
        for (int router = 0; router < NUM_ROUTERS; router++) begin
            for (int port = 0; port < 5; port++) begin
                total_buffer_occupancy += buffer_occupancy[router][port];
            end
        end
        
        // Calculate congestion percentage
        if (max_buffer_capacity > 0) begin
            congestion_percentage = 8'((total_buffer_occupancy * 100) / max_buffer_capacity);
        end else begin
            congestion_percentage = 0;
        end
    end
    
    // Congestion history tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            history_index <= 0;
            for (int i = 0; i < 16; i++) begin
                congestion_history[i] <= 0;
            end
        end else begin
            congestion_history[history_index] <= congestion_percentage;
            history_index <= history_index + 1;
        end
    end
    
    // Calculate average congestion over history window
    always_comb begin
        logic [11:0] sum;
        sum = 0;
        for (int i = 0; i < 16; i++) begin
            sum += {4'b0, congestion_history[i]};
        end
        avg_congestion = sum >> 4;  // Divide by 16
    end
    
    // Identify congested routers
    genvar r;
    generate
        for (r = 0; r < NUM_ROUTERS; r++) begin : gen_congestion_detect
            logic [31:0] router_total_occupancy;
            logic [31:0] router_max_capacity;
            logic [7:0]  router_congestion_pct;
            
            always_comb begin
                router_total_occupancy = 0;
                router_max_capacity = 5 * VC_COUNT * VC_DEPTH;
                
                for (int port = 0; port < 5; port++) begin
                    router_total_occupancy += buffer_occupancy[r][port];
                end
                
                if (router_max_capacity > 0) begin
                    router_congestion_pct = 8'((router_total_occupancy * 100) / router_max_capacity);
                end else begin
                    router_congestion_pct = 0;
                end
            end
            
            assign congested_routers[r] = (router_congestion_pct > 8'(CONGESTION_THRESHOLD)) || 
                                         router_congested[r];
        end
    endgenerate
    
    // Hotspot detection and tracking
    generate
        for (r = 0; r < NUM_ROUTERS; r++) begin : gen_hotspot_detect
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    hotspot_counter[r] <= 0;
                end else begin
                    if (congested_routers[r]) begin
                        if (hotspot_counter[r] < 8'hFF) begin
                            hotspot_counter[r] <= hotspot_counter[r] + 1;
                        end
                    end else begin
                        if (hotspot_counter[r] > 0) begin
                            hotspot_counter[r] <= hotspot_counter[r] - 1;
                        end
                    end
                end
            end
            
            assign persistent_hotspots[r] = (hotspot_counter[r] > hotspot_threshold);
        end
    endgenerate
    
    // Flow control decisions
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            global_flow_control <= 1'b0;
            adaptive_routing_enable <= 1'b0;
            qos_enforcement_active <= 1'b0;
            hotspot_threshold <= 8'd10;  // 10 consecutive cycles
            for (int i = 0; i < NUM_ROUTERS; i++) begin
                throttle_level[i] <= 2'b00;
                priority_boost[i] <= 2'b00;
            end
        end else begin
            // Global flow control activation
            global_flow_control <= (congestion_percentage > 90) || 
                                  (avg_congestion > 80);
            
            // Adaptive routing for moderate congestion
            adaptive_routing_enable <= (congestion_percentage > 60) && 
                                      (congestion_percentage <= 90);
            
            // QoS enforcement during congestion
            qos_enforcement_active <= (congestion_percentage > 50);
            
            // Per-router throttling
            for (int router = 0; router < NUM_ROUTERS; router++) begin
                if (persistent_hotspots[router]) begin
                    throttle_level[router] <= 2'b11;  // Maximum throttling
                    priority_boost[router] <= 2'b00;  // No boost for congested routers
                end else if (congested_routers[router]) begin
                    throttle_level[router] <= 2'b10;  // High throttling
                    priority_boost[router] <= 2'b00;
                end else if (congestion_percentage > 70) begin
                    throttle_level[router] <= 2'b01;  // Light throttling
                    priority_boost[router] <= 2'b01;  // Small boost for non-congested
                end else begin
                    throttle_level[router] <= 2'b00;  // No throttling
                    priority_boost[router] <= 2'b10;  // Normal boost
                end
            end
        end
    end
    
    // Packet drop tracking (simulated - actual implementation would be in routers)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 4; i++) begin
                dropped_packets[i] <= 0;
            end
            total_dropped <= 0;
        end else begin
            // Simulate packet drops during severe congestion
            if (global_flow_control) begin
                // Drop low priority packets more aggressively
                dropped_packets[0] <= dropped_packets[0] + 4;  // QOS_LOW
                dropped_packets[1] <= dropped_packets[1] + 2;  // QOS_NORMAL
                dropped_packets[2] <= dropped_packets[2] + 1;  // QOS_HIGH
                // Never drop QOS_URGENT packets
                
                total_dropped <= total_dropped + 7;
            end
        end
    end
    
    // Output assignments
    assign network_utilization = congestion_percentage;
    assign total_packets_dropped = total_dropped;
    assign congestion_hotspots = persistent_hotspots;

endmodule