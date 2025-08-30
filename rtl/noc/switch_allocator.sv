// Enhanced Switch Allocator for NoC Router
// Handles switch allocation and crossbar control with QoS and congestion awareness

`include "noc_packet.sv"

module switch_allocator (
    input  logic        clk,
    input  logic        rst_n,
    
    // Route requests from routing computation
    input  logic [2:0]  route_request[5][VC_COUNT],
    
    // VC grants from VC allocation
    input  logic [VC_COUNT-1:0] vc_grant[5],
    
    // Buffer status
    input  logic        buffer_empty[5][VC_COUNT],
    input  noc_flit_t   buffer_head_flit[5][VC_COUNT],
    
    // Downstream ready signals
    input  logic        ready_downstream[5],
    
    // QoS and congestion control inputs
    input  logic [1:0]  throttle_level,
    input  logic        qos_enforcement_active,
    input  logic [1:0]  priority_boost,
    input  logic        global_flow_control,
    
    // Output grants and crossbar control
    output logic [4:0]  output_grant[5],
    output logic [2:0]  crossbar_ctrl[5],
    
    // Performance monitoring
    output logic [31:0] qos_grants[4],
    output logic [7:0]  allocation_efficiency,
    output logic        fairness_violation
);

    // Internal signals for switch allocation
    logic [4:0] switch_request[5];  // [output_port][input_port]
    logic [4:0] switch_grant[5];    // [output_port][input_port]
    logic [4:0] qos_switch_request[4][5];  // [qos_level][output_port][input_port]
    
    // QoS-aware request processing
    qos_level_t request_qos[5][VC_COUNT];
    logic [4:0] throttled_request[5];
    
    // Performance counters
    logic [31:0] qos_grant_counters[4];
    logic [31:0] total_requests;
    logic [31:0] successful_grants;
    
    // Fairness tracking
    logic [7:0] qos_service_count[4];
    logic [7:0] expected_service[4];
    logic fairness_check;
    
    // Extract QoS information from buffer head flits
    genvar input_port, vc, output_port, q;
    generate
        for (input_port = 0; input_port < 5; input_port++) begin : gen_qos_extract
            for (vc = 0; vc < VC_COUNT; vc++) begin : gen_vc_qos
                always_comb begin
                    if (!buffer_empty[input_port][vc] && buffer_head_flit[input_port][vc].head) begin
                        request_qos[input_port][vc] = buffer_head_flit[input_port][vc].header.qos;
                    end else begin
                        request_qos[input_port][vc] = QOS_NORMAL;  // Default
                    end
                end
            end
        end
    endgenerate
    
    // Apply throttling based on congestion control
    generate
        for (output_port = 0; output_port < 5; output_port++) begin : gen_throttling
            always_comb begin
                throttled_request[output_port] = 0;
                
                for (int input_port = 0; input_port < 5; input_port++) begin
                    logic should_throttle;
                    logic base_request;
                    
                    // Check if there's a valid request
                    base_request = 1'b0;
                    for (int vc = 0; vc < VC_COUNT; vc++) begin
                        if (vc_grant[input_port][vc] && 
                            !buffer_empty[input_port][vc] &&
                            route_request[input_port][vc] == 3'(output_port) &&
                            ready_downstream[output_port]) begin
                            base_request = 1'b1;
                            break;
                        end
                    end
                    
                    // Apply throttling logic
                    should_throttle = 1'b0;
                    if (global_flow_control) begin
                        should_throttle = 1'b1;  // Throttle all during global flow control
                    end else begin
                        case (throttle_level)
                            2'b01: should_throttle = (input_port[0]);  // Throttle 50%
                            2'b10: should_throttle = (input_port[1:0] != 2'b00);  // Throttle 75%
                            2'b11: should_throttle = 1'b1;  // Throttle all
                            default: should_throttle = 1'b0;  // No throttling
                        endcase
                    end
                    
                    throttled_request[output_port][input_port] = base_request && !should_throttle;
                end
            end
        end
    endgenerate
    
    // Generate QoS-separated switch requests
    generate
        for (q = 0; q < 4; q++) begin : gen_qos_requests
            for (output_port = 0; output_port < 5; output_port++) begin : gen_qos_switch_req
                always_comb begin
                    qos_switch_request[q][output_port] = 0;
                    
                    for (int input_port = 0; input_port < 5; input_port++) begin
                        for (int vc = 0; vc < VC_COUNT; vc++) begin
                            if (throttled_request[output_port][input_port] &&
                                vc_grant[input_port][vc] && 
                                !buffer_empty[input_port][vc] &&
                                route_request[input_port][vc] == 3'(output_port) &&
                                request_qos[input_port][vc] == qos_level_t'(q)) begin
                                qos_switch_request[q][output_port][input_port] = 1'b1;
                            end
                        end
                    end
                end
            end
        end
    endgenerate
    
    // Combine QoS requests with priority (highest QoS first)
    generate
        for (output_port = 0; output_port < 5; output_port++) begin : gen_combined_req
            always_comb begin
                switch_request[output_port] = 0;
                
                if (qos_enforcement_active) begin
                    // Priority order: URGENT > HIGH > NORMAL > LOW
                    for (int q = 3; q >= 0; q--) begin
                        if (|qos_switch_request[q][output_port]) begin
                            switch_request[output_port] = qos_switch_request[q][output_port];
                            break;
                        end
                    end
                end else begin
                    // Fair round-robin among all QoS levels
                    switch_request[output_port] = qos_switch_request[0][output_port] |
                                                qos_switch_request[1][output_port] |
                                                qos_switch_request[2][output_port] |
                                                qos_switch_request[3][output_port];
                end
            end
        end
    endgenerate
    
    // QoS-aware switch arbitration
    generate
        for (output_port = 0; output_port < 5; output_port++) begin : gen_qos_arbiter
            qos_level_t port_qos_levels[5];
            
            // Extract QoS levels for this output port
            always_comb begin
                for (int input_port = 0; input_port < 5; input_port++) begin
                    port_qos_levels[input_port] = QOS_LOW;  // Default
                    
                    for (int vc = 0; vc < VC_COUNT; vc++) begin
                        if (switch_request[output_port][input_port] &&
                            vc_grant[input_port][vc] && 
                            !buffer_empty[input_port][vc]) begin
                            port_qos_levels[input_port] = request_qos[input_port][vc];
                            break;
                        end
                    end
                end
            end
            
            qos_arbiter #(
                .WIDTH(5),
                .QOS_LEVELS(4)
            ) qos_arb_inst (
                .clk(clk),
                .rst_n(rst_n),
                .request(switch_request[output_port]),
                .qos_level(port_qos_levels),
                .aging_threshold(8'd16),  // 16 cycles before aging
                .fairness_enable(!qos_enforcement_active),
                .grant(switch_grant[output_port]),
                .starved_requests(),  // Not used here
                .total_grants(),      // Not used here
                .max_wait_time()      // Not used here
            );
        end
    endgenerate
    
    // Generate output grants and crossbar control
    generate
        for (input_port = 0; input_port < 5; input_port++) begin : gen_output_grants
            always_comb begin
                output_grant[input_port] = 0;
                crossbar_ctrl[input_port] = 0;
                
                for (int output_port = 0; output_port < 5; output_port++) begin
                    if (switch_grant[output_port][input_port]) begin
                        // Find which VC is granted for this input port
                        for (int vc = 0; vc < VC_COUNT; vc++) begin
                            if (vc_grant[input_port][vc] && 
                                !buffer_empty[input_port][vc] &&
                                route_request[input_port][vc] == 3'(output_port)) begin
                                output_grant[input_port][vc] = 1'b1;
                                crossbar_ctrl[input_port] = 3'(output_port);
                                break;
                            end
                        end
                        break;
                    end
                end
            end
        end
    endgenerate
    
    // Deadlock avoidance: Ensure no cyclic dependencies
    // XY routing naturally avoids deadlocks in mesh topology
    // Additional checks for virtual channel dependencies
    
    logic deadlock_detected;
    logic [4:0] dependency_matrix[5];
    
    // Monitor for potential deadlock conditions
    always_comb begin
        deadlock_detected = 1'b0;
        
        // Check for circular wait conditions
        for (int i = 0; i < 5; i++) begin
            dependency_matrix[i] = 0;
            for (int j = 0; j < 5; j++) begin
                if (switch_request[i][j] && !switch_grant[i][j]) begin
                    dependency_matrix[i][j] = 1'b1;
                end
            end
        end
        
        // Simple deadlock detection (can be enhanced)
        if (&switch_request[0] || &switch_request[1] || 
            &switch_request[2] || &switch_request[3] || &switch_request[4]) begin
            deadlock_detected = 1'b1;
        end
    end
    
    // Performance monitoring and QoS tracking
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_requests <= 0;
            successful_grants <= 0;
            for (int q = 0; q < 4; q++) begin
                qos_grant_counters[q] <= 0;
                qos_service_count[q] <= 0;
            end
        end else begin
            // Count total requests
            for (int output_port = 0; output_port < 5; output_port++) begin
                total_requests <= total_requests + $countones(switch_request[output_port]);
                successful_grants <= successful_grants + $countones(switch_grant[output_port]);
            end
            
            // Count QoS-specific grants
            for (int q = 0; q < 4; q++) begin
                logic [4:0] qos_grants_this_cycle;
                qos_grants_this_cycle = 0;
                
                for (int output_port = 0; output_port < 5; output_port++) begin
                    qos_grants_this_cycle |= (switch_grant[output_port] & 
                                            qos_switch_request[q][output_port]);
                end
                
                if (|qos_grants_this_cycle) begin
                    qos_grant_counters[q] <= qos_grant_counters[q] + $countones(qos_grants_this_cycle);
                    qos_service_count[q] <= qos_service_count[q] + 1;
                end
            end
        end
    end
    
    // Calculate allocation efficiency
    always_comb begin
        if (total_requests > 0) begin
            allocation_efficiency = 8'((successful_grants * 100) / total_requests);
        end else begin
            allocation_efficiency = 8'd100;
        end
    end
    
    // Fairness violation detection
    always_comb begin
        int q;
        logic [15:0] actual_percentage;
        logic [15:0] total_service;
        
        fairness_check = 1'b0;
        actual_percentage = 16'd0;
        
        // Calculate expected service ratios (simplified)
        expected_service[0] = 8'd10;  // QOS_LOW: 10%
        expected_service[1] = 8'd30;  // QOS_NORMAL: 30%
        expected_service[2] = 8'd40;  // QOS_HIGH: 40%
        expected_service[3] = 8'd20;  // QOS_URGENT: 20%
        
        total_service = 16'(qos_service_count[0]) + 16'(qos_service_count[1]) + 
                       16'(qos_service_count[2]) + 16'(qos_service_count[3]);
        
        // Check if any QoS level is severely under-served
        for (q = 0; q < 4; q++) begin
            if (qos_service_count[q] > 0 && total_service > 0) begin
                actual_percentage = (16'(qos_service_count[q]) * 16'd100) / total_service;
                
                // Flag violation if actual service is less than 50% of expected
                if (actual_percentage < (16'(expected_service[q]) >> 1)) begin
                    fairness_check = 1'b1;
                end
            end
        end
    end
    
    // Output assignments
    assign qos_grants = qos_grant_counters;
    assign fairness_violation = fairness_check;

endmodule