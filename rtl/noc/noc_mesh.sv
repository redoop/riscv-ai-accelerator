// NoC Mesh Network Topology
// Connects routers in a 2D mesh configuration

`include "noc_packet.sv"

module noc_mesh #(
    parameter int MESH_SIZE_X = 4,
    parameter int MESH_SIZE_Y = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Processing element interfaces
    input  logic [63:0] pe_addr[MESH_SIZE_X*MESH_SIZE_Y],
    input  logic [255:0] pe_wdata[MESH_SIZE_X*MESH_SIZE_Y],
    output logic [255:0] pe_rdata[MESH_SIZE_X*MESH_SIZE_Y],
    input  logic        pe_read[MESH_SIZE_X*MESH_SIZE_Y],
    input  logic        pe_write[MESH_SIZE_X*MESH_SIZE_Y],
    input  logic [2:0]  pe_size[MESH_SIZE_X*MESH_SIZE_Y],
    input  qos_level_t  pe_qos[MESH_SIZE_X*MESH_SIZE_Y],
    output logic        pe_ready[MESH_SIZE_X*MESH_SIZE_Y],
    output logic        pe_valid[MESH_SIZE_X*MESH_SIZE_Y],
    
    // Network monitoring and control
    output logic [31:0] total_packets_routed,
    output logic [31:0] network_utilization,
    output logic        network_congestion,
    output logic [31:0] avg_latency
);

    // Internal router connections - separate input and output
    noc_flit_t router_flit_in[MESH_SIZE_X][MESH_SIZE_Y][5]; // [x][y][port]
    noc_flit_t router_flit_out[MESH_SIZE_X][MESH_SIZE_Y][5]; // [x][y][port]
    logic router_valid_in[MESH_SIZE_X][MESH_SIZE_Y][5];
    logic router_valid_out[MESH_SIZE_X][MESH_SIZE_Y][5];
    logic router_ready_in[MESH_SIZE_X][MESH_SIZE_Y][5];
    logic router_ready_out[MESH_SIZE_X][MESH_SIZE_Y][5];
    
    // Router performance monitoring
    logic [31:0] router_packets[MESH_SIZE_X][MESH_SIZE_Y];
    logic [31:0] router_occupancy[MESH_SIZE_X][MESH_SIZE_Y][5];
    logic router_congestion[MESH_SIZE_X][MESH_SIZE_Y];
    
    // NIC connections
    noc_flit_t nic_to_router[MESH_SIZE_X][MESH_SIZE_Y];
    noc_flit_t router_to_nic[MESH_SIZE_X][MESH_SIZE_Y];
    logic nic_to_router_valid[MESH_SIZE_X][MESH_SIZE_Y];
    logic router_to_nic_valid[MESH_SIZE_X][MESH_SIZE_Y];
    logic nic_to_router_ready[MESH_SIZE_X][MESH_SIZE_Y];
    logic router_to_nic_ready[MESH_SIZE_X][MESH_SIZE_Y];
    
    genvar x, y;
    
    // Generate mesh of routers
    generate
        for (x = 0; x < MESH_SIZE_X; x++) begin : gen_mesh_x
            for (y = 0; y < MESH_SIZE_Y; y++) begin : gen_mesh_y
                
                // Router instance
                noc_router #(
                    .X_COORD(x),
                    .Y_COORD(y),
                    .MESH_SIZE_X(MESH_SIZE_X),
                    .MESH_SIZE_Y(MESH_SIZE_Y)
                ) router_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .flit_in(router_flit_in[x][y]),
                    .valid_in(router_valid_in[x][y]),
                    .ready_out(router_ready_out[x][y]),
                    .flit_out(router_flit_out[x][y]),
                    .valid_out(router_valid_out[x][y]),
                    .ready_in(router_ready_in[x][y]),
                    .packets_routed(router_packets[x][y]),
                    .buffer_occupancy(router_occupancy[x][y]),
                    .congestion_detected(router_congestion[x][y])
                );
                
                // Network Interface Controller
                noc_interface #(
                    .NODE_X(x),
                    .NODE_Y(y)
                ) nic_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .pe_addr(pe_addr[x*MESH_SIZE_Y + y]),
                    .pe_wdata(pe_wdata[x*MESH_SIZE_Y + y]),
                    .pe_rdata(pe_rdata[x*MESH_SIZE_Y + y]),
                    .pe_read(pe_read[x*MESH_SIZE_Y + y]),
                    .pe_write(pe_write[x*MESH_SIZE_Y + y]),
                    .pe_size(pe_size[x*MESH_SIZE_Y + y]),
                    .pe_qos(pe_qos[x*MESH_SIZE_Y + y]),
                    .pe_ready(pe_ready[x*MESH_SIZE_Y + y]),
                    .pe_valid(pe_valid[x*MESH_SIZE_Y + y]),
                    .noc_flit_out(nic_to_router[x][y]),
                    .noc_valid_out(nic_to_router_valid[x][y]),
                    .noc_ready_in(nic_to_router_ready[x][y]),
                    .noc_flit_in(router_to_nic[x][y]),
                    .noc_valid_in(router_to_nic_valid[x][y]),
                    .noc_ready_out(router_to_nic_ready[x][y]),
                    .packets_sent(),
                    .packets_received(),
                    .buffer_overflow(),
                    .buffer_underflow()
                );
                
                // Connect NIC to router local port
                assign router_flit_in[x][y][DIR_LOCAL] = nic_to_router[x][y];
                assign router_valid_in[x][y][DIR_LOCAL] = nic_to_router_valid[x][y];
                assign nic_to_router_ready[x][y] = router_ready_out[x][y][DIR_LOCAL];
                
                assign router_to_nic[x][y] = router_flit_out[x][y][DIR_LOCAL];
                assign router_to_nic_valid[x][y] = router_valid_out[x][y][DIR_LOCAL];
                assign router_ready_in[x][y][DIR_LOCAL] = router_to_nic_ready[x][y];
            end
        end
    endgenerate
    
    // Connect routers in mesh topology
    generate
        for (x = 0; x < MESH_SIZE_X; x++) begin : gen_connections_x
            for (y = 0; y < MESH_SIZE_Y; y++) begin : gen_connections_y
                
                // North connections
                if (y > 0) begin : north_conn
                    assign router_flit_in[x][y][DIR_NORTH] = router_flit_out[x][y-1][DIR_SOUTH];
                    assign router_valid_in[x][y][DIR_NORTH] = router_valid_out[x][y-1][DIR_SOUTH];
                    assign router_ready_in[x][y-1][DIR_SOUTH] = router_ready_out[x][y][DIR_NORTH];
                end else begin : north_boundary
                    assign router_flit_in[x][y][DIR_NORTH] = '0;
                    assign router_valid_in[x][y][DIR_NORTH] = 1'b0;
                    assign router_ready_in[x][y][DIR_NORTH] = 1'b1;
                end
                
                // South connections
                if (y < MESH_SIZE_Y - 1) begin : south_conn
                    assign router_flit_in[x][y][DIR_SOUTH] = router_flit_out[x][y+1][DIR_NORTH];
                    assign router_valid_in[x][y][DIR_SOUTH] = router_valid_out[x][y+1][DIR_NORTH];
                    assign router_ready_in[x][y+1][DIR_NORTH] = router_ready_out[x][y][DIR_SOUTH];
                end else begin : south_boundary
                    assign router_flit_in[x][y][DIR_SOUTH] = '0;
                    assign router_valid_in[x][y][DIR_SOUTH] = 1'b0;
                    assign router_ready_in[x][y][DIR_SOUTH] = 1'b1;
                end
                
                // East connections
                if (x < MESH_SIZE_X - 1) begin : east_conn
                    assign router_flit_in[x][y][DIR_EAST] = router_flit_out[x+1][y][DIR_WEST];
                    assign router_valid_in[x][y][DIR_EAST] = router_valid_out[x+1][y][DIR_WEST];
                    assign router_ready_in[x+1][y][DIR_WEST] = router_ready_out[x][y][DIR_EAST];
                end else begin : east_boundary
                    assign router_flit_in[x][y][DIR_EAST] = '0;
                    assign router_valid_in[x][y][DIR_EAST] = 1'b0;
                    assign router_ready_in[x][y][DIR_EAST] = 1'b1;
                end
                
                // West connections
                if (x > 0) begin : west_conn
                    assign router_flit_in[x][y][DIR_WEST] = router_flit_out[x-1][y][DIR_EAST];
                    assign router_valid_in[x][y][DIR_WEST] = router_valid_out[x-1][y][DIR_EAST];
                    assign router_ready_in[x-1][y][DIR_EAST] = router_ready_out[x][y][DIR_WEST];
                end else begin : west_boundary
                    assign router_flit_in[x][y][DIR_WEST] = '0;
                    assign router_valid_in[x][y][DIR_WEST] = 1'b0;
                    assign router_ready_in[x][y][DIR_WEST] = 1'b1;
                end
            end
        end
    endgenerate
    
    // Network-wide monitoring and statistics
    logic [31:0] total_packets;
    logic [31:0] total_occupancy;
    logic [31:0] congested_routers;
    
    always_comb begin
        total_packets = 0;
        total_occupancy = 0;
        congested_routers = 0;
        
        for (int i = 0; i < MESH_SIZE_X; i++) begin
            for (int j = 0; j < MESH_SIZE_Y; j++) begin
                total_packets += router_packets[i][j];
                
                for (int port = 0; port < 5; port++) begin
                    total_occupancy += router_occupancy[i][j][port];
                end
                
                if (router_congestion[i][j]) begin
                    congested_routers++;
                end
            end
        end
    end
    
    assign total_packets_routed = total_packets;
    assign network_utilization = total_occupancy / (MESH_SIZE_X * MESH_SIZE_Y * 5 * VC_COUNT * VC_DEPTH);
    assign network_congestion = (congested_routers > (MESH_SIZE_X * MESH_SIZE_Y / 4));
    
    // Latency measurement (simplified)
    logic [31:0] latency_accumulator;
    logic [31:0] latency_samples;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latency_accumulator <= 0;
            latency_samples <= 0;
        end else begin
            // Sample latency periodically (simplified implementation)
            if (total_packets > 0) begin
                latency_accumulator <= latency_accumulator + 10; // Placeholder
                latency_samples <= latency_samples + 1;
            end
        end
    end
    
    assign avg_latency = (latency_samples > 0) ? (latency_accumulator / latency_samples) : 0;

endmodule