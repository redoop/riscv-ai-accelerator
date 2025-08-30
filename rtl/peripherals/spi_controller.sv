// SPI Controller Module
// Implements SPI master controller with configurable modes
// Supports SPI modes 0-3, variable clock rates, and multiple chip selects

module spi_controller #(
    parameter DATA_WIDTH = 32,
    parameter MAX_SLAVES = 8,
    parameter FIFO_DEPTH = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // SPI physical interface
    output logic        spi_sclk,
    output logic        spi_mosi,
    input  logic        spi_miso,
    output logic [MAX_SLAVES-1:0] spi_cs_n,
    
    // AXI interface for configuration and data
    axi4_if.slave       axi_if,
    
    // Interrupt interface
    output logic        tx_complete_irq,
    output logic        rx_complete_irq,
    output logic        error_irq,
    
    // Status
    output logic        spi_busy,
    output logic [7:0]  tx_fifo_level,
    output logic [7:0]  rx_fifo_level
);

    // SPI configuration registers
    logic [31:0] spi_ctrl;          // Control register
    logic [31:0] spi_status;        // Status register
    logic [31:0] spi_clk_div;       // Clock divider
    logic [31:0] spi_slave_sel;     // Slave select
    logic [31:0] spi_tx_data;       // Transmit data
    logic [31:0] spi_rx_data;       // Receive data
    
    // Control register bit definitions
    logic        spi_enable;        // [0] SPI enable
    logic        spi_master;        // [1] Master mode (always 1)
    logic [1:0]  spi_mode;          // [3:2] SPI mode (CPOL, CPHA)
    logic [4:0]  spi_data_bits;     // [8:4] Data bits per transfer (1-32)
    logic        spi_lsb_first;     // [9] LSB first transmission
    logic        spi_auto_cs;       // [10] Automatic chip select
    logic [2:0]  spi_cs_delay;      // [13:11] CS setup/hold delay
    logic        tx_fifo_reset;     // [16] TX FIFO reset
    logic        rx_fifo_reset;     // [17] RX FIFO reset
    logic        loopback_mode;     // [18] Loopback mode for testing
    
    // Status register bit definitions
    logic        tx_fifo_empty;     // [0] TX FIFO empty
    logic        tx_fifo_full;      // [1] TX FIFO full
    logic        rx_fifo_empty;     // [2] RX FIFO empty
    logic        rx_fifo_full;      // [3] RX FIFO full
    logic        transfer_active;   // [4] Transfer in progress
    logic        slave_selected;    // [5] Slave is selected
    
    // FIFO signals
    logic [31:0] tx_fifo_data_in;
    logic        tx_fifo_write;
    logic        tx_fifo_read;
    logic [31:0] tx_fifo_data_out;
    
    logic [31:0] rx_fifo_data_in;
    logic        rx_fifo_write;
    logic        rx_fifo_read;
    logic [31:0] rx_fifo_data_out;
    
    // SPI state machine
    typedef enum logic [3:0] {
        SPI_IDLE,
        SPI_CS_SETUP,
        SPI_TRANSFER,
        SPI_CS_HOLD,
        SPI_COMPLETE
    } spi_state_t;
    
    spi_state_t spi_state, spi_next_state;
    
    // Transfer control
    logic [5:0]  bit_counter;
    logic [31:0] shift_reg_tx;
    logic [31:0] shift_reg_rx;
    logic [15:0] clk_counter;
    logic        sclk_enable;
    logic        sclk_internal;
    logic [7:0]  cs_delay_counter;
    
    // Clock generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clk_counter <= 16'h0;
            sclk_internal <= 1'b0;
        end else if (sclk_enable && spi_enable) begin
            if (clk_counter >= spi_clk_div[15:0]) begin
                clk_counter <= 16'h0;
                sclk_internal <= ~sclk_internal;
            end else begin
                clk_counter <= clk_counter + 1;
            end
        end else begin
            clk_counter <= 16'h0;
            if (spi_mode[1] == 1'b0) begin  // CPOL = 0
                sclk_internal <= 1'b0;
            end else begin  // CPOL = 1
                sclk_internal <= 1'b1;
            end
        end
    end
    
    // SPI clock output
    assign spi_sclk = sclk_enable ? sclk_internal : (spi_mode[1] ? 1'b1 : 1'b0);
    
    // SPI state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_state <= SPI_IDLE;
        end else begin
            spi_state <= spi_next_state;
        end
    end
    
    always_comb begin
        spi_next_state = spi_state;
        
        case (spi_state)
            SPI_IDLE: begin
                if (spi_enable && !tx_fifo_empty) begin
                    spi_next_state = SPI_CS_SETUP;
                end
            end
            
            SPI_CS_SETUP: begin
                if (cs_delay_counter >= spi_cs_delay) begin
                    spi_next_state = SPI_TRANSFER;
                end
            end
            
            SPI_TRANSFER: begin
                if (bit_counter >= spi_data_bits) begin
                    spi_next_state = SPI_CS_HOLD;
                end
            end
            
            SPI_CS_HOLD: begin
                if (cs_delay_counter >= spi_cs_delay) begin
                    spi_next_state = SPI_COMPLETE;
                end
            end
            
            SPI_COMPLETE: begin
                spi_next_state = SPI_IDLE;
            end
        endcase
    end
    
    // Transfer control logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_counter <= 6'h0;
            shift_reg_tx <= 32'h0;
            shift_reg_rx <= 32'h0;
            cs_delay_counter <= 8'h0;
            sclk_enable <= 1'b0;
            tx_fifo_read <= 1'b0;
            rx_fifo_write <= 1'b0;
        end else begin
            case (spi_state)
                SPI_IDLE: begin
                    bit_counter <= 6'h0;
                    cs_delay_counter <= 8'h0;
                    sclk_enable <= 1'b0;
                    tx_fifo_read <= 1'b0;
                    rx_fifo_write <= 1'b0;
                    
                    if (spi_enable && !tx_fifo_empty) begin
                        shift_reg_tx <= tx_fifo_data_out;
                        tx_fifo_read <= 1'b1;
                    end
                end
                
                SPI_CS_SETUP: begin
                    tx_fifo_read <= 1'b0;
                    cs_delay_counter <= cs_delay_counter + 1;
                    
                    if (cs_delay_counter >= spi_cs_delay) begin
                        sclk_enable <= 1'b1;
                        cs_delay_counter <= 8'h0;
                    end
                end
                
                SPI_TRANSFER: begin
                    // Transfer data on appropriate clock edge
                    if (spi_mode[0] == 1'b0) begin  // CPHA = 0, sample on first edge
                        if (clk_counter == 0 && sclk_internal != sclk_enable) begin
                            // Shift data
                            if (spi_lsb_first) begin
                                shift_reg_tx <= {1'b0, shift_reg_tx[31:1]};
                                shift_reg_rx <= {spi_miso, shift_reg_rx[31:1]};
                            end else begin
                                shift_reg_tx <= {shift_reg_tx[30:0], 1'b0};
                                shift_reg_rx <= {shift_reg_rx[30:0], spi_miso};
                            end
                            bit_counter <= bit_counter + 1;
                        end
                    end else begin  // CPHA = 1, sample on second edge
                        if (clk_counter == (spi_clk_div[15:0] >> 1)) begin
                            // Shift data
                            if (spi_lsb_first) begin
                                shift_reg_tx <= {1'b0, shift_reg_tx[31:1]};
                                shift_reg_rx <= {spi_miso, shift_reg_rx[31:1]};
                            end else begin
                                shift_reg_tx <= {shift_reg_tx[30:0], 1'b0};
                                shift_reg_rx <= {shift_reg_rx[30:0], spi_miso};
                            end
                            bit_counter <= bit_counter + 1;
                        end
                    end
                end
                
                SPI_CS_HOLD: begin
                    sclk_enable <= 1'b0;
                    cs_delay_counter <= cs_delay_counter + 1;
                end
                
                SPI_COMPLETE: begin
                    rx_fifo_write <= 1'b1;
                    cs_delay_counter <= 8'h0;
                end
            endcase
        end
    end
    
    // MOSI output
    always_comb begin
        if (loopback_mode) begin
            spi_mosi = spi_miso;  // Loopback for testing
        end else if (spi_lsb_first) begin
            spi_mosi = shift_reg_tx[0];
        end else begin
            spi_mosi = shift_reg_tx[31];
        end
    end
    
    // Chip select control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_cs_n <= {MAX_SLAVES{1'b1}};  // All CS inactive (high)
        end else begin
            if (spi_auto_cs && (spi_state == SPI_CS_SETUP || 
                               spi_state == SPI_TRANSFER || 
                               spi_state == SPI_CS_HOLD)) begin
                spi_cs_n <= ~(1 << spi_slave_sel[2:0]);  // Active low
            end else begin
                spi_cs_n <= spi_slave_sel[MAX_SLAVES-1:0] ^ {MAX_SLAVES{1'b1}};  // Manual control
            end
        end
    end
    
    // TX FIFO
    fifo #(
        .DATA_WIDTH(32),
        .DEPTH(FIFO_DEPTH)
    ) tx_fifo_inst (
        .clk(clk),
        .rst_n(rst_n && !tx_fifo_reset),
        .data_in(tx_fifo_data_in),
        .write(tx_fifo_write),
        .full(tx_fifo_full),
        .data_out(tx_fifo_data_out),
        .read(tx_fifo_read),
        .empty(tx_fifo_empty),
        .count(tx_fifo_level)
    );
    
    // RX FIFO
    fifo #(
        .DATA_WIDTH(32),
        .DEPTH(FIFO_DEPTH)
    ) rx_fifo_inst (
        .clk(clk),
        .rst_n(rst_n && !rx_fifo_reset),
        .data_in(rx_fifo_data_in),
        .write(rx_fifo_write),
        .full(rx_fifo_full),
        .data_out(rx_fifo_data_out),
        .read(rx_fifo_read),
        .empty(rx_fifo_empty),
        .count(rx_fifo_level)
    );
    
    // FIFO connections
    assign tx_fifo_data_in = spi_tx_data;
    assign rx_fifo_data_in = shift_reg_rx;
    assign spi_rx_data = rx_fifo_data_out;
    
    // Control register decoding
    assign spi_enable = spi_ctrl[0];
    assign spi_master = spi_ctrl[1];
    assign spi_mode = spi_ctrl[3:2];
    assign spi_data_bits = spi_ctrl[8:4] == 0 ? 5'd32 : spi_ctrl[8:4];  // 0 means 32 bits
    assign spi_lsb_first = spi_ctrl[9];
    assign spi_auto_cs = spi_ctrl[10];
    assign spi_cs_delay = spi_ctrl[13:11];
    assign tx_fifo_reset = spi_ctrl[16];
    assign rx_fifo_reset = spi_ctrl[17];
    assign loopback_mode = spi_ctrl[18];
    
    // Status register
    assign spi_status = {13'h0, slave_selected, transfer_active, 
                        rx_fifo_full, rx_fifo_empty, tx_fifo_full, tx_fifo_empty};
    assign transfer_active = (spi_state != SPI_IDLE);
    assign slave_selected = |~spi_cs_n;
    
    // Register access via AXI
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_ctrl <= 32'h00000002;      // Master mode enabled by default
            spi_clk_div <= 32'h00000010;   // Default clock divider
            spi_slave_sel <= 32'h00000000;
            tx_fifo_write <= 1'b0;
            rx_fifo_read <= 1'b0;
        end else if (axi_if.awvalid && axi_if.wvalid) begin
            tx_fifo_write <= 1'b0;  // Default
            
            case (axi_if.awaddr[7:0])
                8'h00: spi_ctrl <= axi_if.wdata;
                8'h08: spi_clk_div <= axi_if.wdata;
                8'h0C: spi_slave_sel <= axi_if.wdata;
                8'h10: begin
                    spi_tx_data <= axi_if.wdata;
                    tx_fifo_write <= 1'b1;
                end
            endcase
        end else begin
            tx_fifo_write <= 1'b0;
        end
        
        // Handle RX FIFO read
        if (axi_if.arvalid && axi_if.araddr[7:0] == 8'h14) begin
            rx_fifo_read <= 1'b1;
        end else begin
            rx_fifo_read <= 1'b0;
        end
    end
    
    // AXI read interface
    always_comb begin
        case (axi_if.araddr[7:0])
            8'h00: axi_if.rdata = spi_ctrl;
            8'h04: axi_if.rdata = spi_status;
            8'h08: axi_if.rdata = spi_clk_div;
            8'h0C: axi_if.rdata = spi_slave_sel;
            8'h14: axi_if.rdata = spi_rx_data;
            8'h18: axi_if.rdata = {16'h0, rx_fifo_level, tx_fifo_level};
            default: axi_if.rdata = 32'h0;
        endcase
    end
    
    // AXI interface control
    assign axi_if.awready = 1'b1;
    assign axi_if.wready = 1'b1;
    assign axi_if.bvalid = axi_if.awvalid && axi_if.wvalid;
    assign axi_if.bresp = 2'b00;
    
    assign axi_if.arready = 1'b1;
    assign axi_if.rvalid = axi_if.arvalid;
    assign axi_if.rresp = 2'b00;
    
    // Interrupt generation
    assign tx_complete_irq = (spi_state == SPI_COMPLETE);
    assign rx_complete_irq = rx_fifo_write;
    assign error_irq = tx_fifo_full && tx_fifo_write;  // TX overflow error
    
    // Status output
    assign spi_busy = transfer_active;
    
endmodule

// Enhanced FIFO with count output
module fifo #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [DATA_WIDTH-1:0]   data_in,
    input  logic                    write,
    output logic                    full,
    output logic [DATA_WIDTH-1:0]   data_out,
    input  logic                    read,
    output logic                    empty,
    output logic [7:0]              count
);

    logic [DATA_WIDTH-1:0] memory [0:DEPTH-1];
    logic [$clog2(DEPTH):0] write_ptr, read_ptr;
    logic [$clog2(DEPTH):0] fifo_count;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            read_ptr <= 0;
            fifo_count <= 0;
        end else begin
            if (write && !full) begin
                memory[write_ptr[$clog2(DEPTH)-1:0]] <= data_in;
                write_ptr <= write_ptr + 1;
            end
            
            if (read && !empty) begin
                read_ptr <= read_ptr + 1;
            end
            
            // Update count
            case ({write && !full, read && !empty})
                2'b10: fifo_count <= fifo_count + 1;  // Write only
                2'b01: fifo_count <= fifo_count - 1;  // Read only
                2'b11: fifo_count <= fifo_count;      // Both (no change)
                2'b00: fifo_count <= fifo_count;      // Neither
            endcase
        end
    end
    
    assign data_out = memory[read_ptr[$clog2(DEPTH)-1:0]];
    assign full = (fifo_count == DEPTH);
    assign empty = (fifo_count == 0);
    assign count = fifo_count[7:0];
    
endmodule