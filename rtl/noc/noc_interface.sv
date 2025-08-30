// Network Interface Controller (NIC)
// Provides interface between processing elements and NoC

`include "noc_packet.sv"

module noc_interface #(
    parameter int NODE_X = 0,
    parameter int NODE_Y = 0,
    parameter int BUFFER_DEPTH = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Processor/PE interface
    input  logic [63:0] pe_addr,
    input  logic [255:0] pe_wdata,
    output logic [255:0] pe_rdata,
    input  logic        pe_read,
    input  logic        pe_write,
    input  logic [2:0]  pe_size,
    input  qos_level_t  pe_qos,
    output logic        pe_ready,
    output logic        pe_valid,
    
    // NoC interface
    output noc_flit_t   noc_flit_out,
    output logic        noc_valid_out,
    input  logic        noc_ready_in,
    
    input  noc_flit_t   noc_flit_in,
    input  logic        noc_valid_in,
    output logic        noc_ready_out,
    
    // Status and control
    output logic [31:0] packets_sent,
    output logic [31:0] packets_received,
    output logic        buffer_overflow,
    output logic        buffer_underflow
);

    // Internal buffers
    typedef struct packed {
        logic [63:0]    addr;
        logic [255:0]   data;
        logic [2:0]     size;
        pkt_type_t      pkt_type;
        qos_level_t     qos;
        logic [7:0]     pkt_id;
    } tx_buffer_entry_t;
    
    typedef struct packed {
        logic [255:0]   data;
        logic [7:0]     pkt_id;
        logic           valid;
    } rx_buffer_entry_t;
    
    // Transmit buffer
    tx_buffer_entry_t tx_buffer[BUFFER_DEPTH];
    logic [$clog2(BUFFER_DEPTH)-1:0] tx_head, tx_tail;
    logic [$clog2(BUFFER_DEPTH):0] tx_count;
    logic tx_full, tx_empty;
    
    // Receive buffer
    rx_buffer_entry_t rx_buffer[BUFFER_DEPTH];
    logic [$clog2(BUFFER_DEPTH)-1:0] rx_head, rx_tail;
    logic [$clog2(BUFFER_DEPTH):0] rx_count;
    logic rx_full, rx_empty;
    
    // Packet ID generation
    logic [7:0] next_pkt_id;
    
    // State machines
    typedef enum logic [2:0] {
        TX_IDLE,
        TX_HEADER,
        TX_DATA,
        TX_WAIT
    } tx_state_t;
    
    typedef enum logic [2:0] {
        RX_IDLE,
        RX_HEADER,
        RX_DATA,
        RX_COMPLETE
    } rx_state_t;
    
    tx_state_t tx_state;
    rx_state_t rx_state;
    
    // Current transmission tracking
    tx_buffer_entry_t current_tx;
    logic [5:0] flits_remaining;
    logic [255:0] current_rx_data;
    logic [7:0] current_rx_id;
    
    // Buffer management
    assign tx_full = (tx_count == ($clog2(BUFFER_DEPTH)+1)'(BUFFER_DEPTH));
    assign tx_empty = (tx_count == ($clog2(BUFFER_DEPTH)+1)'(0));
    assign rx_full = (rx_count == ($clog2(BUFFER_DEPTH)+1)'(BUFFER_DEPTH));
    assign rx_empty = (rx_count == ($clog2(BUFFER_DEPTH)+1)'(0));
    
    assign pe_ready = !tx_full;
    assign pe_valid = !rx_empty;
    
    // Address to coordinates conversion
    function automatic logic [3:0] addr_to_x(logic [63:0] addr);
        return addr[7:4]; // Simple mapping - can be enhanced
    endfunction
    
    function automatic logic [3:0] addr_to_y(logic [63:0] addr);
        return addr[11:8]; // Simple mapping - can be enhanced
    endfunction
    
    // Determine packet type from operation
    function automatic pkt_type_t get_packet_type(logic read, logic write);
        if (read) return PKT_READ_REQ;
        else if (write) return PKT_WRITE_REQ;
        else return PKT_RESERVED;
    endfunction
    
    // Calculate number of flits needed
    function automatic logic [5:0] calc_flits(logic [2:0] size);
        case (size)
            3'b000: return 1; // 1 byte
            3'b001: return 1; // 2 bytes
            3'b010: return 1; // 4 bytes
            3'b011: return 1; // 8 bytes
            3'b100: return 1; // 16 bytes
            3'b101: return 1; // 32 bytes
            3'b110: return 2; // 64 bytes (needs 2 flits)
            3'b111: return 4; // 256 bytes (needs 4 flits)
            default: return 1;
        endcase
    endfunction
    
    // Transmit buffer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_head <= 0;
            tx_tail <= 0;
            tx_count <= 0;
        end else begin
            // Write to transmit buffer
            if ((pe_read || pe_write) && pe_ready) begin
                tx_buffer[tx_tail].addr <= pe_addr;
                tx_buffer[tx_tail].data <= pe_wdata;
                tx_buffer[tx_tail].size <= pe_size;
                tx_buffer[tx_tail].pkt_type <= get_packet_type(pe_read, pe_write);
                tx_buffer[tx_tail].qos <= pe_qos;
                tx_buffer[tx_tail].pkt_id <= next_pkt_id;
                
                /* verilator lint_off WIDTHEXPAND */
                tx_tail <= ($clog2(BUFFER_DEPTH))'(({1'b0, tx_tail} + 1) % BUFFER_DEPTH);
                /* verilator lint_on WIDTHEXPAND */
                tx_count <= tx_count + 1;
                next_pkt_id <= next_pkt_id + 1;
            end
            
            // Read from transmit buffer
            if (tx_state == TX_IDLE && !tx_empty && noc_ready_in) begin
                current_tx <= tx_buffer[tx_head];
                /* verilator lint_off WIDTHEXPAND */
                tx_head <= ($clog2(BUFFER_DEPTH))'(({1'b0, tx_head} + 1) % BUFFER_DEPTH);
                /* verilator lint_on WIDTHEXPAND */
                tx_count <= tx_count - 1;
            end
        end
    end
    
    // Transmit state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE;
            noc_valid_out <= 1'b0;
            noc_flit_out <= '0;
            flits_remaining <= 0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    noc_valid_out <= 1'b0;
                    if (!tx_empty && noc_ready_in) begin
                        tx_state <= TX_HEADER;
                        flits_remaining <= calc_flits(tx_buffer[tx_head].size);
                    end
                end
                
                TX_HEADER: begin
                    if (noc_ready_in) begin
                        noc_valid_out <= 1'b1;
                        noc_flit_out.head <= 1'b1;
                        noc_flit_out.tail <= (flits_remaining == 1);
                        noc_flit_out.header.src_x <= 4'(NODE_X);
                        noc_flit_out.header.src_y <= 4'(NODE_Y);
                        noc_flit_out.header.dst_x <= addr_to_x(current_tx.addr);
                        noc_flit_out.header.dst_y <= addr_to_y(current_tx.addr);
                        noc_flit_out.header.pkt_type <= current_tx.pkt_type;
                        noc_flit_out.header.qos <= current_tx.qos;
                        noc_flit_out.header.pkt_id <= current_tx.pkt_id;
                        noc_flit_out.header.length <= flits_remaining;
                        noc_flit_out.header.multicast <= 1'b0;
                        noc_flit_out.data <= current_tx.data;
                        
                        if (flits_remaining == 1) begin
                            tx_state <= TX_IDLE;
                        end else begin
                            tx_state <= TX_DATA;
                            flits_remaining <= flits_remaining - 1;
                        end
                    end
                end
                
                TX_DATA: begin
                    if (noc_ready_in) begin
                        noc_valid_out <= 1'b1;
                        noc_flit_out.head <= 1'b0;
                        noc_flit_out.tail <= (flits_remaining == 1);
                        noc_flit_out.data <= current_tx.data; // Simplified - should handle multi-flit data
                        
                        if (flits_remaining == 1) begin
                            tx_state <= TX_IDLE;
                        end else begin
                            flits_remaining <= flits_remaining - 1;
                        end
                    end
                end
                
                default: tx_state <= TX_IDLE;
            endcase
        end
    end
    
    // Receive buffer management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_head <= 0;
            rx_tail <= 0;
            rx_count <= 0;
        end else begin
            // Write to receive buffer
            if (rx_state == RX_COMPLETE && !rx_full) begin
                rx_buffer[rx_tail].data <= current_rx_data;
                rx_buffer[rx_tail].pkt_id <= current_rx_id;
                rx_buffer[rx_tail].valid <= 1'b1;
                
                /* verilator lint_off WIDTHEXPAND */
                rx_tail <= ($clog2(BUFFER_DEPTH))'(({1'b0, rx_tail} + 1) % BUFFER_DEPTH);
                /* verilator lint_on WIDTHEXPAND */
                rx_count <= rx_count + 1;
            end
            
            // Read from receive buffer
            if (pe_valid && !rx_empty) begin
                pe_rdata <= rx_buffer[rx_head].data;
                /* verilator lint_off WIDTHEXPAND */
                rx_head <= ($clog2(BUFFER_DEPTH))'(({1'b0, rx_head} + 1) % BUFFER_DEPTH);
                /* verilator lint_on WIDTHEXPAND */
                rx_count <= rx_count - 1;
            end
        end
    end
    
    // Receive state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE;
            noc_ready_out <= 1'b1;
            current_rx_data <= '0;
            current_rx_id <= '0;
        end else begin
            case (rx_state)
                RX_IDLE: begin
                    noc_ready_out <= !rx_full;
                    if (noc_valid_in && !rx_full) begin
                        if (noc_flit_in.head) begin
                            current_rx_data <= noc_flit_in.data;
                            current_rx_id <= noc_flit_in.header.pkt_id;
                            
                            if (noc_flit_in.tail) begin
                                rx_state <= RX_COMPLETE;
                            end else begin
                                rx_state <= RX_DATA;
                            end
                        end
                    end
                end
                
                RX_DATA: begin
                    noc_ready_out <= !rx_full;
                    if (noc_valid_in && !rx_full) begin
                        // Accumulate data for multi-flit packets
                        current_rx_data <= noc_flit_in.data; // Simplified
                        
                        if (noc_flit_in.tail) begin
                            rx_state <= RX_COMPLETE;
                        end
                    end
                end
                
                RX_COMPLETE: begin
                    rx_state <= RX_IDLE;
                end
                
                default: rx_state <= RX_IDLE;
            endcase
        end
    end
    
    // Performance counters
    logic [31:0] tx_packet_count, rx_packet_count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_packet_count <= 0;
            rx_packet_count <= 0;
        end else begin
            // Count transmitted packets
            if (noc_valid_out && noc_ready_in && noc_flit_out.tail) begin
                tx_packet_count <= tx_packet_count + 1;
            end
            
            // Count received packets
            if (rx_state == RX_COMPLETE) begin
                rx_packet_count <= rx_packet_count + 1;
            end
        end
    end
    
    assign packets_sent = tx_packet_count;
    assign packets_received = rx_packet_count;
    assign buffer_overflow = tx_full || rx_full;
    assign buffer_underflow = tx_empty || rx_empty;

endmodule