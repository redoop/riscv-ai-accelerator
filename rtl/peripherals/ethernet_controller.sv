// Gigabit Ethernet Controller Module
// Implements 1000BASE-T Ethernet MAC and PHY interface
// Supports packet processing and DMA operations

module ethernet_controller #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 32,
    parameter FIFO_DEPTH = 1024
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // RGMII interface to PHY
    output logic        rgmii_txc,
    output logic        rgmii_tx_ctl,
    output logic [3:0]  rgmii_txd,
    input  logic        rgmii_rxc,
    input  logic        rgmii_rx_ctl,
    input  logic [3:0]  rgmii_rxd,
    
    // MDIO interface for PHY management
    output logic        mdio_mdc,
    inout  logic        mdio_mdio,
    output logic        phy_reset_n,
    
    // AXI interface for configuration
    axi4_if.slave       axi_cfg_if,
    
    // DMA interface for packet data
    output logic                dma_tx_req,
    input  logic                dma_tx_ack,
    output logic [ADDR_WIDTH-1:0] dma_tx_addr,
    output logic [15:0]         dma_tx_length,
    input  logic [DATA_WIDTH-1:0] dma_tx_data,
    input  logic                dma_tx_valid,
    output logic                dma_tx_ready,
    
    output logic                dma_rx_req,
    input  logic                dma_rx_ack,
    output logic [ADDR_WIDTH-1:0] dma_rx_addr,
    output logic [15:0]         dma_rx_length,
    output logic [DATA_WIDTH-1:0] dma_rx_data,
    output logic                dma_rx_valid,
    input  logic                dma_rx_ready,
    
    // Interrupt interface
    output logic        tx_complete_irq,
    output logic        rx_complete_irq,
    output logic        error_irq,
    
    // Status and statistics
    output logic        link_up,
    output logic [1:0]  link_speed,  // 00: 10M, 01: 100M, 10: 1000M
    output logic        full_duplex,
    output logic [31:0] tx_packet_count,
    output logic [31:0] rx_packet_count,
    output logic [31:0] error_count
);

    // Internal signals
    logic [7:0]         tx_data;
    logic               tx_valid;
    logic               tx_ready;
    logic               tx_error;
    logic               tx_last;
    
    logic [7:0]         rx_data;
    logic               rx_valid;
    logic               rx_ready;
    logic               rx_error;
    logic               rx_last;
    
    // Configuration registers
    logic [31:0]        mac_addr_low;
    logic [15:0]        mac_addr_high;
    logic               promiscuous_mode;
    logic               broadcast_enable;
    logic               multicast_enable;
    logic [31:0]        rx_buffer_addr;
    logic [31:0]        tx_buffer_addr;
    
    // FIFO signals
    logic [DATA_WIDTH-1:0] tx_fifo_data_in;
    logic               tx_fifo_write;
    logic               tx_fifo_full;
    logic [DATA_WIDTH-1:0] tx_fifo_data_out;
    logic               tx_fifo_read;
    logic               tx_fifo_empty;
    
    logic [DATA_WIDTH-1:0] rx_fifo_data_in;
    logic               rx_fifo_write;
    logic               rx_fifo_full;
    logic [DATA_WIDTH-1:0] rx_fifo_data_out;
    logic               rx_fifo_read;
    logic               rx_fifo_empty;
    
    // State machines
    typedef enum logic [2:0] {
        TX_IDLE,
        TX_PREAMBLE,
        TX_DATA,
        TX_CRC,
        TX_COMPLETE
    } tx_state_t;
    
    typedef enum logic [2:0] {
        RX_IDLE,
        RX_PREAMBLE,
        RX_DATA,
        RX_CRC_CHECK,
        RX_COMPLETE
    } rx_state_t;
    
    tx_state_t tx_state, tx_next_state;
    rx_state_t rx_state, rx_next_state;
    
    // Counters and timers
    logic [15:0]        tx_byte_count;
    logic [15:0]        rx_byte_count;
    logic [31:0]        crc_value;
    logic [7:0]         preamble_count;
    
    // Statistics counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_packet_count <= 32'h0;
            rx_packet_count <= 32'h0;
            error_count <= 32'h0;
        end else begin
            if (tx_state == TX_COMPLETE) begin
                tx_packet_count <= tx_packet_count + 1;
            end
            if (rx_state == RX_COMPLETE && !rx_error) begin
                rx_packet_count <= rx_packet_count + 1;
            end
            if (tx_error || rx_error) begin
                error_count <= error_count + 1;
            end
        end
    end
    
    // TX State Machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE;
        end else begin
            tx_state <= tx_next_state;
        end
    end
    
    always_comb begin
        tx_next_state = tx_state;
        
        case (tx_state)
            TX_IDLE: begin
                if (!tx_fifo_empty && dma_tx_ack) begin
                    tx_next_state = TX_PREAMBLE;
                end
            end
            TX_PREAMBLE: begin
                if (preamble_count >= 8) begin
                    tx_next_state = TX_DATA;
                end
            end
            TX_DATA: begin
                if (tx_byte_count >= dma_tx_length) begin
                    tx_next_state = TX_CRC;
                end
            end
            TX_CRC: begin
                tx_next_state = TX_COMPLETE;
            end
            TX_COMPLETE: begin
                tx_next_state = TX_IDLE;
            end
        endcase
    end
    
    // RX State Machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE;
        end else begin
            rx_state <= rx_next_state;
        end
    end
    
    always_comb begin
        rx_next_state = rx_state;
        
        case (rx_state)
            RX_IDLE: begin
                if (rx_valid && rx_data == 8'h55) begin  // Preamble detection
                    rx_next_state = RX_PREAMBLE;
                end
            end
            RX_PREAMBLE: begin
                if (rx_valid && rx_data == 8'hD5) begin  // SFD detection
                    rx_next_state = RX_DATA;
                end
            end
            RX_DATA: begin
                if (rx_last || rx_byte_count >= 1518) begin  // Max frame size
                    rx_next_state = RX_CRC_CHECK;
                end
            end
            RX_CRC_CHECK: begin
                rx_next_state = RX_COMPLETE;
            end
            RX_COMPLETE: begin
                rx_next_state = RX_IDLE;
            end
        endcase
    end
    
    // RGMII TX logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rgmii_txc <= 1'b0;
            rgmii_tx_ctl <= 1'b0;
            rgmii_txd <= 4'h0;
            tx_valid <= 1'b0;
            tx_byte_count <= 16'h0;
            preamble_count <= 8'h0;
        end else begin
            rgmii_txc <= ~rgmii_txc;  // Generate TX clock
            
            case (tx_state)
                TX_IDLE: begin
                    rgmii_tx_ctl <= 1'b0;
                    rgmii_txd <= 4'h0;
                    tx_byte_count <= 16'h0;
                    preamble_count <= 8'h0;
                end
                TX_PREAMBLE: begin
                    rgmii_tx_ctl <= 1'b1;
                    if (preamble_count < 7) begin
                        rgmii_txd <= 4'h5;  // Preamble pattern
                        preamble_count <= preamble_count + 1;
                    end else begin
                        rgmii_txd <= 4'hD;  // SFD
                        preamble_count <= preamble_count + 1;
                    end
                end
                TX_DATA: begin
                    rgmii_tx_ctl <= 1'b1;
                    if (tx_fifo_read) begin
                        rgmii_txd <= tx_fifo_data_out[3:0];  // Lower nibble first
                        tx_byte_count <= tx_byte_count + 1;
                    end
                end
                TX_CRC: begin
                    rgmii_tx_ctl <= 1'b1;
                    rgmii_txd <= crc_value[3:0];  // Transmit CRC
                end
                TX_COMPLETE: begin
                    rgmii_tx_ctl <= 1'b0;
                    rgmii_txd <= 4'h0;
                end
            endcase
        end
    end
    
    // RGMII RX logic
    always_ff @(posedge rgmii_rxc or negedge rst_n) begin
        if (!rst_n) begin
            rx_data <= 8'h0;
            rx_valid <= 1'b0;
            rx_byte_count <= 16'h0;
        end else begin
            if (rgmii_rx_ctl) begin
                rx_data <= {rgmii_rxd, rx_data[7:4]};  // Reconstruct byte
                rx_valid <= 1'b1;
                
                if (rx_state == RX_DATA) begin
                    rx_byte_count <= rx_byte_count + 1;
                end
            end else begin
                rx_valid <= 1'b0;
            end
        end
    end
    
    // TX FIFO
    fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) tx_fifo_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(tx_fifo_data_in),
        .write(tx_fifo_write),
        .full(tx_fifo_full),
        .data_out(tx_fifo_data_out),
        .read(tx_fifo_read),
        .empty(tx_fifo_empty)
    );
    
    // RX FIFO
    fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) rx_fifo_inst (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(rx_fifo_data_in),
        .write(rx_fifo_write),
        .full(rx_fifo_full),
        .data_out(rx_fifo_data_out),
        .read(rx_fifo_read),
        .empty(rx_fifo_empty)
    );
    
    // DMA interface logic
    assign dma_tx_req = !tx_fifo_empty && (tx_state == TX_IDLE);
    assign tx_fifo_data_in = dma_tx_data;
    assign tx_fifo_write = dma_tx_valid && dma_tx_ready;
    assign dma_tx_ready = !tx_fifo_full;
    assign tx_fifo_read = (tx_state == TX_DATA) && tx_ready;
    
    assign dma_rx_req = !rx_fifo_full && (rx_state == RX_COMPLETE);
    assign rx_fifo_data_in = {48'h0, rx_data, 8'h0};  // Pack received data
    assign rx_fifo_write = rx_valid && (rx_state == RX_DATA);
    assign dma_rx_data = rx_fifo_data_out;
    assign dma_rx_valid = !rx_fifo_empty;
    assign rx_fifo_read = dma_rx_ready && dma_rx_valid;
    
    // Configuration register access via AXI
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_addr_low <= 32'h12345678;
            mac_addr_high <= 16'h9ABC;
            promiscuous_mode <= 1'b0;
            broadcast_enable <= 1'b1;
            multicast_enable <= 1'b0;
            rx_buffer_addr <= 32'h0;
            tx_buffer_addr <= 32'h0;
        end else if (axi_cfg_if.awvalid && axi_cfg_if.wvalid) begin
            case (axi_cfg_if.awaddr[7:0])
                8'h00: mac_addr_low <= axi_cfg_if.wdata;
                8'h04: mac_addr_high <= axi_cfg_if.wdata[15:0];
                8'h08: begin
                    promiscuous_mode <= axi_cfg_if.wdata[0];
                    broadcast_enable <= axi_cfg_if.wdata[1];
                    multicast_enable <= axi_cfg_if.wdata[2];
                end
                8'h10: rx_buffer_addr <= axi_cfg_if.wdata;
                8'h14: tx_buffer_addr <= axi_cfg_if.wdata;
            endcase
        end
    end
    
    // AXI read response
    always_comb begin
        case (axi_cfg_if.araddr[7:0])
            8'h00: axi_cfg_if.rdata = mac_addr_low;
            8'h04: axi_cfg_if.rdata = {16'h0, mac_addr_high};
            8'h08: axi_cfg_if.rdata = {29'h0, multicast_enable, broadcast_enable, promiscuous_mode};
            8'h10: axi_cfg_if.rdata = rx_buffer_addr;
            8'h14: axi_cfg_if.rdata = tx_buffer_addr;
            8'h20: axi_cfg_if.rdata = tx_packet_count;
            8'h24: axi_cfg_if.rdata = rx_packet_count;
            8'h28: axi_cfg_if.rdata = error_count;
            8'h2C: axi_cfg_if.rdata = {29'h0, full_duplex, link_speed};
            default: axi_cfg_if.rdata = 32'h0;
        endcase
    end
    
    // AXI interface control
    assign axi_cfg_if.awready = 1'b1;
    assign axi_cfg_if.wready = 1'b1;
    assign axi_cfg_if.bvalid = axi_cfg_if.awvalid && axi_cfg_if.wvalid;
    assign axi_cfg_if.bresp = 2'b00;
    
    assign axi_cfg_if.arready = 1'b1;
    assign axi_cfg_if.rvalid = axi_cfg_if.arvalid;
    assign axi_cfg_if.rresp = 2'b00;
    
    // PHY management and status (simplified)
    assign phy_reset_n = rst_n;
    assign link_up = 1'b1;        // Assume link is up
    assign link_speed = 2'b10;    // 1000 Mbps
    assign full_duplex = 1'b1;    // Full duplex
    
    // MDIO interface (simplified)
    assign mdio_mdc = clk;        // Use system clock for simplicity
    
    // Interrupt generation
    assign tx_complete_irq = (tx_state == TX_COMPLETE);
    assign rx_complete_irq = (rx_state == RX_COMPLETE);
    assign error_irq = tx_error || rx_error;
    
    // CRC calculation (simplified placeholder)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            crc_value <= 32'hFFFFFFFF;
        end else if (tx_state == TX_DATA) begin
            // Simplified CRC calculation
            crc_value <= crc_value ^ {24'h0, tx_data};
        end
    end
    
endmodule

// Simple FIFO module for packet buffering
module fifo #(
    parameter DATA_WIDTH = 64,
    parameter DEPTH = 1024
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [DATA_WIDTH-1:0]   data_in,
    input  logic                    write,
    output logic                    full,
    output logic [DATA_WIDTH-1:0]   data_out,
    input  logic                    read,
    output logic                    empty
);

    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];
    logic [$clog2(DEPTH):0] write_ptr, read_ptr;
    logic [$clog2(DEPTH):0] count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            read_ptr <= 0;
            count <= 0;
        end else begin
            if (write && !full) begin
                memory[write_ptr[$clog2(DEPTH)-1:0]] <= data_in;
                write_ptr <= write_ptr + 1;
                count <= count + 1;
            end
            
            if (read && !empty) begin
                read_ptr <= read_ptr + 1;
                count <= count - 1;
            end
            
            if (write && !full && read && !empty) begin
                count <= count;  // No net change
            end
        end
    end
    
    assign data_out = memory[read_ptr[$clog2(DEPTH)-1:0]];
    assign full = (count == DEPTH);
    assign empty = (count == 0);
    
endmodule