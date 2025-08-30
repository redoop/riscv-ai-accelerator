// Network Packet Processor Module
// Handles network protocol processing and packet classification
// Supports TCP/UDP/IP packet processing with hardware acceleration

module network_packet_processor #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 32,
    parameter MAX_PACKET_SIZE = 1518
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Input packet interface from Ethernet controller
    input  logic [DATA_WIDTH-1:0]   pkt_data_in,
    input  logic                    pkt_valid_in,
    output logic                    pkt_ready_in,
    input  logic                    pkt_sop,      // Start of packet
    input  logic                    pkt_eop,      // End of packet
    input  logic [15:0]             pkt_length,
    
    // Output packet interface to application
    output logic [DATA_WIDTH-1:0]   pkt_data_out,
    output logic                    pkt_valid_out,
    input  logic                    pkt_ready_out,
    output logic                    pkt_sop_out,
    output logic                    pkt_eop_out,
    output logic [15:0]             pkt_length_out,
    
    // Packet classification results
    output logic [3:0]              pkt_type,     // 0: Unknown, 1: ARP, 2: IP, 3: TCP, 4: UDP
    output logic [31:0]             src_ip,
    output logic [31:0]             dst_ip,
    output logic [15:0]             src_port,
    output logic [15:0]             dst_port,
    output logic [7:0]              protocol,
    
    // Configuration interface
    input  logic [47:0]             local_mac,
    input  logic [31:0]             local_ip,
    input  logic                    promiscuous_mode,
    
    // Statistics and status
    output logic [31:0]             processed_packets,
    output logic [31:0]             dropped_packets,
    output logic [31:0]             error_packets,
    
    // DMA interface for zero-copy processing
    output logic                    dma_req,
    input  logic                    dma_ack,
    output logic [ADDR_WIDTH-1:0]   dma_addr,
    output logic [15:0]             dma_length,
    output logic                    dma_write,
    
    // Interrupt interface
    output logic                    packet_received_irq,
    output logic                    error_irq
);

    // Ethernet header fields
    typedef struct packed {
        logic [47:0] dst_mac;
        logic [47:0] src_mac;
        logic [15:0] ethertype;
    } eth_header_t;
    
    // IP header fields (simplified IPv4)
    typedef struct packed {
        logic [3:0]  version;
        logic [3:0]  ihl;
        logic [7:0]  tos;
        logic [15:0] total_length;
        logic [15:0] identification;
        logic [2:0]  flags;
        logic [12:0] fragment_offset;
        logic [7:0]  ttl;
        logic [7:0]  protocol;
        logic [15:0] header_checksum;
        logic [31:0] src_ip;
        logic [31:0] dst_ip;
    } ip_header_t;
    
    // TCP header fields (simplified)
    typedef struct packed {
        logic [15:0] src_port;
        logic [15:0] dst_port;
        logic [31:0] seq_num;
        logic [31:0] ack_num;
        logic [3:0]  data_offset;
        logic [5:0]  reserved;
        logic [5:0]  flags;
        logic [15:0] window_size;
        logic [15:0] checksum;
        logic [15:0] urgent_ptr;
    } tcp_header_t;
    
    // UDP header fields
    typedef struct packed {
        logic [15:0] src_port;
        logic [15:0] dst_port;
        logic [15:0] length;
        logic [15:0] checksum;
    } udp_header_t;
    
    // Internal signals
    logic [DATA_WIDTH-1:0]  packet_buffer [0:MAX_PACKET_SIZE/8-1];
    logic [15:0]            buffer_write_ptr;
    logic [15:0]            buffer_read_ptr;
    logic                   buffer_full;
    logic                   buffer_empty;
    
    eth_header_t            eth_header;
    ip_header_t             ip_header;
    tcp_header_t            tcp_header;
    udp_header_t            udp_header;
    
    // State machine for packet processing
    typedef enum logic [3:0] {
        IDLE,
        RECEIVE_ETH_HEADER,
        PARSE_ETH_HEADER,
        RECEIVE_IP_HEADER,
        PARSE_IP_HEADER,
        RECEIVE_TCP_UDP_HEADER,
        PARSE_TCP_UDP_HEADER,
        PROCESS_PAYLOAD,
        FORWARD_PACKET,
        DROP_PACKET,
        ERROR_STATE
    } pkt_state_t;
    
    pkt_state_t current_state, next_state;
    
    // Counters and flags
    logic [15:0]            bytes_received;
    logic [15:0]            header_bytes_parsed;
    logic                   valid_packet;
    logic                   checksum_valid;
    
    // Statistics counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            processed_packets <= 32'h0;
            dropped_packets <= 32'h0;
            error_packets <= 32'h0;
        end else begin
            case (current_state)
                FORWARD_PACKET: processed_packets <= processed_packets + 1;
                DROP_PACKET: dropped_packets <= dropped_packets + 1;
                ERROR_STATE: error_packets <= error_packets + 1;
            endcase
        end
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (pkt_valid_in && pkt_sop) begin
                    next_state = RECEIVE_ETH_HEADER;
                end
            end
            
            RECEIVE_ETH_HEADER: begin
                if (bytes_received >= 14) begin  // Ethernet header is 14 bytes
                    next_state = PARSE_ETH_HEADER;
                end
            end
            
            PARSE_ETH_HEADER: begin
                if (eth_header.ethertype == 16'h0800) begin  // IPv4
                    next_state = RECEIVE_IP_HEADER;
                end else if (eth_header.ethertype == 16'h0806) begin  // ARP
                    next_state = FORWARD_PACKET;  // Forward ARP packets
                end else begin
                    next_state = DROP_PACKET;  // Unknown ethertype
                end
            end
            
            RECEIVE_IP_HEADER: begin
                if (bytes_received >= 34) begin  // Eth + IP header (minimum)
                    next_state = PARSE_IP_HEADER;
                end
            end
            
            PARSE_IP_HEADER: begin
                if (ip_header.protocol == 8'h06) begin  // TCP
                    next_state = RECEIVE_TCP_UDP_HEADER;
                end else if (ip_header.protocol == 8'h11) begin  // UDP
                    next_state = RECEIVE_TCP_UDP_HEADER;
                end else begin
                    next_state = FORWARD_PACKET;  // Other IP protocols
                end
            end
            
            RECEIVE_TCP_UDP_HEADER: begin
                if (ip_header.protocol == 8'h06 && bytes_received >= 54) begin  // TCP
                    next_state = PARSE_TCP_UDP_HEADER;
                end else if (ip_header.protocol == 8'h11 && bytes_received >= 42) begin  // UDP
                    next_state = PARSE_TCP_UDP_HEADER;
                end
            end
            
            PARSE_TCP_UDP_HEADER: begin
                next_state = PROCESS_PAYLOAD;
            end
            
            PROCESS_PAYLOAD: begin
                if (pkt_eop) begin
                    if (valid_packet) begin
                        next_state = FORWARD_PACKET;
                    end else begin
                        next_state = DROP_PACKET;
                    end
                end
            end
            
            FORWARD_PACKET: begin
                if (pkt_ready_out) begin
                    next_state = IDLE;
                end
            end
            
            DROP_PACKET: begin
                next_state = IDLE;
            end
            
            ERROR_STATE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Packet reception and buffering
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer_write_ptr <= 16'h0;
            bytes_received <= 16'h0;
        end else begin
            if (pkt_valid_in && pkt_ready_in && !buffer_full) begin
                packet_buffer[buffer_write_ptr] <= pkt_data_in;
                buffer_write_ptr <= buffer_write_ptr + 1;
                bytes_received <= bytes_received + (DATA_WIDTH / 8);
                
                if (pkt_sop) begin
                    bytes_received <= DATA_WIDTH / 8;
                    buffer_write_ptr <= 16'h1;
                end
            end
            
            if (current_state == IDLE) begin
                buffer_write_ptr <= 16'h0;
                bytes_received <= 16'h0;
            end
        end
    end
    
    // Header parsing logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eth_header <= '0;
            ip_header <= '0;
            tcp_header <= '0;
            udp_header <= '0;
            valid_packet <= 1'b0;
        end else begin
            case (current_state)
                PARSE_ETH_HEADER: begin
                    // Extract Ethernet header from buffer
                    eth_header.dst_mac <= {packet_buffer[0][47:0], packet_buffer[1][63:48]};
                    eth_header.src_mac <= {packet_buffer[1][47:0], packet_buffer[2][63:48]};
                    eth_header.ethertype <= packet_buffer[2][47:32];
                    
                    // Check if packet is for us (or promiscuous mode)
                    if (promiscuous_mode || 
                        eth_header.dst_mac == local_mac || 
                        eth_header.dst_mac == 48'hFFFFFFFFFFFF) begin  // Broadcast
                        valid_packet <= 1'b1;
                    end else begin
                        valid_packet <= 1'b0;
                    end
                end
                
                PARSE_IP_HEADER: begin
                    // Extract IP header (assuming it starts at byte 14)
                    ip_header <= packet_buffer[2][31:0], packet_buffer[3], packet_buffer[4][63:32];
                    
                    // Basic IP validation
                    if (ip_header.version == 4'h4 && 
                        (ip_header.dst_ip == local_ip || ip_header.dst_ip == 32'hFFFFFFFF)) begin
                        valid_packet <= valid_packet && 1'b1;
                    end else begin
                        valid_packet <= 1'b0;
                    end
                end
                
                PARSE_TCP_UDP_HEADER: begin
                    if (ip_header.protocol == 8'h06) begin  // TCP
                        tcp_header <= packet_buffer[5], packet_buffer[6][63:32];
                    end else if (ip_header.protocol == 8'h11) begin  // UDP
                        udp_header <= packet_buffer[5][63:0];
                    end
                end
                
                IDLE: begin
                    valid_packet <= 1'b0;
                end
            endcase
        end
    end
    
    // Output packet classification
    always_comb begin
        case (current_state)
            FORWARD_PACKET: begin
                if (eth_header.ethertype == 16'h0806) begin
                    pkt_type = 4'h1;  // ARP
                end else if (eth_header.ethertype == 16'h0800) begin
                    if (ip_header.protocol == 8'h06) begin
                        pkt_type = 4'h3;  // TCP
                    end else if (ip_header.protocol == 8'h11) begin
                        pkt_type = 4'h4;  // UDP
                    end else begin
                        pkt_type = 4'h2;  // IP (other)
                    end
                end else begin
                    pkt_type = 4'h0;  // Unknown
                end
            end
            default: pkt_type = 4'h0;
        endcase
        
        // Output extracted fields
        src_ip = ip_header.src_ip;
        dst_ip = ip_header.dst_ip;
        protocol = ip_header.protocol;
        
        if (ip_header.protocol == 8'h06) begin  // TCP
            src_port = tcp_header.src_port;
            dst_port = tcp_header.dst_port;
        end else if (ip_header.protocol == 8'h11) begin  // UDP
            src_port = udp_header.src_port;
            dst_port = udp_header.dst_port;
        end else begin
            src_port = 16'h0;
            dst_port = 16'h0;
        end
    end
    
    // Output packet forwarding
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buffer_read_ptr <= 16'h0;
            pkt_valid_out <= 1'b0;
            pkt_sop_out <= 1'b0;
            pkt_eop_out <= 1'b0;
        end else begin
            case (current_state)
                FORWARD_PACKET: begin
                    if (pkt_ready_out && buffer_read_ptr < buffer_write_ptr) begin
                        pkt_data_out <= packet_buffer[buffer_read_ptr];
                        pkt_valid_out <= 1'b1;
                        pkt_sop_out <= (buffer_read_ptr == 16'h0);
                        pkt_eop_out <= (buffer_read_ptr == buffer_write_ptr - 1);
                        buffer_read_ptr <= buffer_read_ptr + 1;
                    end else begin
                        pkt_valid_out <= 1'b0;
                        pkt_sop_out <= 1'b0;
                        pkt_eop_out <= 1'b0;
                    end
                end
                
                IDLE: begin
                    buffer_read_ptr <= 16'h0;
                    pkt_valid_out <= 1'b0;
                    pkt_sop_out <= 1'b0;
                    pkt_eop_out <= 1'b0;
                end
                
                default: begin
                    pkt_valid_out <= 1'b0;
                    pkt_sop_out <= 1'b0;
                    pkt_eop_out <= 1'b0;
                end
            endcase
        end
    end
    
    // Flow control
    assign pkt_ready_in = !buffer_full && (current_state == RECEIVE_ETH_HEADER || 
                                          current_state == RECEIVE_IP_HEADER || 
                                          current_state == RECEIVE_TCP_UDP_HEADER || 
                                          current_state == PROCESS_PAYLOAD);
    
    assign buffer_full = (buffer_write_ptr >= (MAX_PACKET_SIZE / 8 - 1));
    assign buffer_empty = (buffer_write_ptr == 0);
    
    assign pkt_length_out = bytes_received;
    
    // DMA interface for zero-copy operations
    assign dma_req = (current_state == FORWARD_PACKET) && valid_packet;
    assign dma_addr = 32'h10000000;  // Base address for packet buffers
    assign dma_length = bytes_received;
    assign dma_write = 1'b1;  // Writing received packet to memory
    
    // Interrupt generation
    assign packet_received_irq = (current_state == FORWARD_PACKET);
    assign error_irq = (current_state == ERROR_STATE);
    
endmodule