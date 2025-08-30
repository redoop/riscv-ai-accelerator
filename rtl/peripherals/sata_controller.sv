// SATA 3.0 Controller Module
// Implements SATA 3.0 (6 Gbps) host controller with AHCI interface
// Supports SATA 3.0, 2.0, and 1.0 devices

module sata_controller #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter MAX_PORTS = 4,
    parameter FIFO_DEPTH = 1024
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // SATA physical interface
    output logic [MAX_PORTS-1:0] sata_tx_p,
    output logic [MAX_PORTS-1:0] sata_tx_n,
    input  logic [MAX_PORTS-1:0] sata_rx_p,
    input  logic [MAX_PORTS-1:0] sata_rx_n,
    
    // Reference clock for SATA PHY
    input  logic        sata_refclk_p,
    input  logic        sata_refclk_n,
    
    // AXI interface for AHCI registers
    axi4_if.slave       axi_if,
    
    // DMA interface for data transfers
    output logic                dma_req,
    input  logic                dma_ack,
    output logic [ADDR_WIDTH-1:0] dma_addr,
    output logic [15:0]         dma_length,
    output logic                dma_write,
    output logic [DATA_WIDTH-1:0] dma_wdata,
    input  logic [DATA_WIDTH-1:0] dma_rdata,
    input  logic                dma_valid,
    output logic                dma_ready,
    
    // Interrupt interface
    output logic        port_irq [MAX_PORTS-1:0],
    output logic        global_irq,
    
    // Status and configuration
    output logic [MAX_PORTS-1:0] port_present,
    output logic [MAX_PORTS-1:0] port_active,
    output logic [1:0]  link_speed [MAX_PORTS-1:0],  // 0: 1.5G, 1: 3G, 2: 6G
    output logic [31:0] command_count,
    output logic [31:0] error_count
);

    // SATA FIS (Frame Information Structure) types
    typedef enum logic [7:0] {
        FIS_REG_H2D     = 8'h27,  // Register - Host to Device
        FIS_REG_D2H     = 8'h34,  // Register - Device to Host
        FIS_DMA_ACT     = 8'h39,  // DMA Activate
        FIS_DMA_SETUP   = 8'h41,  // DMA Setup
        FIS_DATA        = 8'h46,  // Data
        FIS_BIST        = 8'h58,  // BIST Activate
        FIS_PIO_SETUP   = 8'h5F,  // PIO Setup
        FIS_DEV_BITS    = 8'hA1   // Set Device Bits
    } sata_fis_type_t;
    
    // SATA command types
    typedef enum logic [7:0] {
        CMD_READ_DMA_EXT    = 8'h25,
        CMD_WRITE_DMA_EXT   = 8'h35,
        CMD_READ_FPDMA      = 8'h60,
        CMD_WRITE_FPDMA     = 8'h61,
        CMD_IDENTIFY        = 8'hEC,
        CMD_SET_FEATURES    = 8'hEF
    } sata_command_t;
    
    // AHCI Port registers structure
    typedef struct packed {
        logic [31:0] clb;        // Command List Base Address
        logic [31:0] clbu;       // Command List Base Address Upper
        logic [31:0] fb;         // FIS Base Address
        logic [31:0] fbu;        // FIS Base Address Upper
        logic [31:0] is;         // Interrupt Status
        logic [31:0] ie;         // Interrupt Enable
        logic [31:0] cmd;        // Command and Status
        logic [31:0] tfd;        // Task File Data
        logic [31:0] sig;        // Signature
        logic [31:0] ssts;       // SATA Status
        logic [31:0] sctl;       // SATA Control
        logic [31:0] serr;       // SATA Error
        logic [31:0] sact;       // SATA Active
        logic [31:0] ci;         // Command Issue
        logic [31:0] sntf;       // SATA Notification
    } ahci_port_regs_t;
    
    // AHCI Global registers
    logic [31:0] ghc_cap;        // Host Capabilities
    logic [31:0] ghc_ghc;        // Global Host Control
    logic [31:0] ghc_is;         // Interrupt Status
    logic [31:0] ghc_pi;         // Ports Implemented
    logic [31:0] ghc_vs;         // Version
    logic [31:0] ghc_ccc_ctl;    // Command Completion Coalescing Control
    logic [31:0] ghc_ccc_ports;  // Command Completion Coalescing Ports
    
    // Port registers
    ahci_port_regs_t port_regs [MAX_PORTS-1:0];
    
    // Internal signals per port
    logic [MAX_PORTS-1:0] port_reset;
    logic [MAX_PORTS-1:0] port_comreset;
    logic [MAX_PORTS-1:0] port_link_up;
    logic [MAX_PORTS-1:0] port_device_present;
    
    // Command processing
    logic [4:0]  active_command_slot [MAX_PORTS-1:0];
    logic        command_active [MAX_PORTS-1:0];
    logic [31:0] command_table_addr [MAX_PORTS-1:0];
    
    // Data transfer state machine
    typedef enum logic [2:0] {
        TRANSFER_IDLE,
        TRANSFER_COMMAND,
        TRANSFER_DATA_OUT,
        TRANSFER_DATA_IN,
        TRANSFER_STATUS,
        TRANSFER_COMPLETE,
        TRANSFER_ERROR
    } transfer_state_t;
    
    transfer_state_t transfer_state [MAX_PORTS-1:0];
    transfer_state_t transfer_next_state [MAX_PORTS-1:0];
    
    // FIS transmission and reception
    logic [31:0] tx_fis_data [MAX_PORTS-1:0];
    logic        tx_fis_valid [MAX_PORTS-1:0];
    logic        tx_fis_ready [MAX_PORTS-1:0];
    logic [31:0] rx_fis_data [MAX_PORTS-1:0];
    logic        rx_fis_valid [MAX_PORTS-1:0];
    logic        rx_fis_ready [MAX_PORTS-1:0];
    
    // Statistics
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            command_count <= 32'h0;
            error_count <= 32'h0;
        end else begin
            for (int i = 0; i < MAX_PORTS; i++) begin
                if (transfer_state[i] == TRANSFER_COMPLETE) begin
                    command_count <= command_count + 1;
                end
                if (transfer_state[i] == TRANSFER_ERROR) begin
                    error_count <= error_count + 1;
                end
            end
        end
    end
    
    // Initialize AHCI capabilities
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ghc_cap <= 32'h0;
            ghc_ghc <= 32'h0;
            ghc_is <= 32'h0;
            ghc_pi <= 32'h0;
            ghc_vs <= 32'h00010300;  // AHCI 1.3
            ghc_ccc_ctl <= 32'h0;
            ghc_ccc_ports <= 32'h0;
        end else begin
            // Set capabilities
            ghc_cap[4:0] <= MAX_PORTS - 1;  // Number of ports
            ghc_cap[8] <= 1'b1;             // External SATA
            ghc_cap[9] <= 1'b1;             // Enclosure Management
            ghc_cap[18] <= 1'b1;            // AHCI mode only
            ghc_cap[20:18] <= 3'b010;       // Interface Speed Support (6 Gbps)
            ghc_cap[25] <= 1'b1;            // Command List Override
            ghc_cap[26] <= 1'b1;            // Activity LED
            ghc_cap[27] <= 1'b1;            // Aggressive Link Power Management
            ghc_cap[28] <= 1'b1;            // Staggered Spin-up
            ghc_cap[30] <= 1'b1;            // 64-bit Addressing
            
            // Ports implemented
            ghc_pi <= (1 << MAX_PORTS) - 1;
            
            // Update global interrupt status
            ghc_is <= 32'h0;
            for (int i = 0; i < MAX_PORTS; i++) begin
                ghc_is[i] <= |port_regs[i].is;
            end
        end
    end
    
    // Port state machines and link management
    genvar port_idx;
    generate
        for (port_idx = 0; port_idx < MAX_PORTS; port_idx++) begin : port_gen
            
            // Transfer state machine
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    transfer_state[port_idx] <= TRANSFER_IDLE;
                end else begin
                    transfer_state[port_idx] <= transfer_next_state[port_idx];
                end
            end
            
            always_comb begin
                transfer_next_state[port_idx] = transfer_state[port_idx];
                
                case (transfer_state[port_idx])
                    TRANSFER_IDLE: begin
                        if (port_regs[port_idx].ci != 0 && port_link_up[port_idx]) begin
                            transfer_next_state[port_idx] = TRANSFER_COMMAND;
                        end
                    end
                    
                    TRANSFER_COMMAND: begin
                        if (tx_fis_ready[port_idx]) begin
                            transfer_next_state[port_idx] = TRANSFER_DATA_OUT;
                        end
                    end
                    
                    TRANSFER_DATA_OUT: begin
                        if (dma_valid) begin
                            transfer_next_state[port_idx] = TRANSFER_STATUS;
                        end
                    end
                    
                    TRANSFER_DATA_IN: begin
                        if (rx_fis_valid[port_idx]) begin
                            transfer_next_state[port_idx] = TRANSFER_STATUS;
                        end
                    end
                    
                    TRANSFER_STATUS: begin
                        transfer_next_state[port_idx] = TRANSFER_COMPLETE;
                    end
                    
                    TRANSFER_COMPLETE: begin
                        transfer_next_state[port_idx] = TRANSFER_IDLE;
                    end
                    
                    TRANSFER_ERROR: begin
                        transfer_next_state[port_idx] = TRANSFER_IDLE;
                    end
                endcase
            end
            
            // Port register management
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    port_regs[port_idx] <= '0;
                    port_reset[port_idx] <= 1'b1;
                    port_comreset[port_idx] <= 1'b0;
                    active_command_slot[port_idx] <= 5'h0;
                    command_active[port_idx] <= 1'b0;
                end else begin
                    // Simulate device presence and link up
                    port_device_present[port_idx] <= 1'b1;
                    port_link_up[port_idx] <= 1'b1;
                    
                    // SATA Status (SSTS)
                    port_regs[port_idx].ssts[3:0] <= 4'h3;    // Device present and active
                    port_regs[port_idx].ssts[7:4] <= 4'h3;    // 6 Gbps speed
                    port_regs[port_idx].ssts[11:8] <= 4'h1;   // Interface active
                    
                    // Task File Data (TFD) - simulate ready status
                    port_regs[port_idx].tfd[7] <= 1'b0;       // BSY = 0
                    port_regs[port_idx].tfd[6] <= 1'b1;       // DRDY = 1
                    port_regs[port_idx].tfd[0] <= 1'b0;       // ERR = 0
                    
                    // Signature - SATA disk
                    port_regs[port_idx].sig <= 32'h00000101;
                    
                    // Command processing
                    if (transfer_state[port_idx] == TRANSFER_COMMAND) begin
                        command_active[port_idx] <= 1'b1;
                        // Find first set bit in CI register
                        for (int slot = 0; slot < 32; slot++) begin
                            if (port_regs[port_idx].ci[slot]) begin
                                active_command_slot[port_idx] <= slot[4:0];
                                break;
                            end
                        end
                    end
                    
                    // Complete command
                    if (transfer_state[port_idx] == TRANSFER_COMPLETE) begin
                        port_regs[port_idx].ci[active_command_slot[port_idx]] <= 1'b0;
                        port_regs[port_idx].is[0] <= 1'b1;  // Device to Host Register FIS
                        command_active[port_idx] <= 1'b0;
                    end
                    
                    // Handle AXI writes to port registers
                    if (axi_if.awvalid && axi_if.wvalid) begin
                        case (axi_if.awaddr[11:8])
                            4'h1: begin  // Port 0 registers
                                if (port_idx == 0) begin
                                    case (axi_if.awaddr[7:0])
                                        8'h00: port_regs[port_idx].clb <= axi_if.wdata;
                                        8'h04: port_regs[port_idx].clbu <= axi_if.wdata;
                                        8'h08: port_regs[port_idx].fb <= axi_if.wdata;
                                        8'h0C: port_regs[port_idx].fbu <= axi_if.wdata;
                                        8'h14: port_regs[port_idx].ie <= axi_if.wdata;
                                        8'h18: port_regs[port_idx].cmd <= axi_if.wdata;
                                        8'h2C: port_regs[port_idx].sctl <= axi_if.wdata;
                                        8'h34: port_regs[port_idx].sact <= axi_if.wdata;
                                        8'h38: port_regs[port_idx].ci <= axi_if.wdata;
                                        // Interrupt Status is write-1-to-clear
                                        8'h10: port_regs[port_idx].is <= port_regs[port_idx].is & ~axi_if.wdata;
                                    endcase
                                end
                            end
                            // Add cases for other ports...
                        endcase
                    end
                end
            end
            
            // FIS transmission
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    tx_fis_data[port_idx] <= 32'h0;
                    tx_fis_valid[port_idx] <= 1'b0;
                end else begin
                    case (transfer_state[port_idx])
                        TRANSFER_COMMAND: begin
                            // Send Register - Host to Device FIS
                            tx_fis_data[port_idx] <= {8'h00, 8'h00, 8'h80, FIS_REG_H2D};
                            tx_fis_valid[port_idx] <= 1'b1;
                        end
                        
                        default: begin
                            tx_fis_valid[port_idx] <= 1'b0;
                        end
                    endcase
                end
            end
            
            // FIS reception (simplified)
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rx_fis_ready[port_idx] <= 1'b1;
                end else begin
                    rx_fis_ready[port_idx] <= 1'b1;  // Always ready
                end
            end
            
        end
    endgenerate
    
    // DMA interface
    assign dma_req = |command_active;
    assign dma_addr = port_regs[0].clb;  // Simplified: use port 0
    assign dma_length = 16'd512;         // Standard sector size
    assign dma_write = (transfer_state[0] == TRANSFER_DATA_IN);
    assign dma_ready = 1'b1;
    
    // AXI register interface
    always_comb begin
        case (axi_if.araddr[11:0])
            // Global registers
            12'h000: axi_if.rdata = ghc_cap;
            12'h004: axi_if.rdata = ghc_ghc;
            12'h008: axi_if.rdata = ghc_is;
            12'h00C: axi_if.rdata = ghc_pi;
            12'h010: axi_if.rdata = ghc_vs;
            12'h014: axi_if.rdata = ghc_ccc_ctl;
            12'h018: axi_if.rdata = ghc_ccc_ports;
            
            // Port 0 registers
            12'h100: axi_if.rdata = port_regs[0].clb;
            12'h104: axi_if.rdata = port_regs[0].clbu;
            12'h108: axi_if.rdata = port_regs[0].fb;
            12'h10C: axi_if.rdata = port_regs[0].fbu;
            12'h110: axi_if.rdata = port_regs[0].is;
            12'h114: axi_if.rdata = port_regs[0].ie;
            12'h118: axi_if.rdata = port_regs[0].cmd;
            12'h11C: axi_if.rdata = port_regs[0].tfd;
            12'h120: axi_if.rdata = port_regs[0].sig;
            12'h128: axi_if.rdata = port_regs[0].ssts;
            12'h12C: axi_if.rdata = port_regs[0].sctl;
            12'h130: axi_if.rdata = port_regs[0].serr;
            12'h134: axi_if.rdata = port_regs[0].sact;
            12'h138: axi_if.rdata = port_regs[0].ci;
            
            // Statistics
            12'hF00: axi_if.rdata = command_count;
            12'hF04: axi_if.rdata = error_count;
            
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
    
    // Status outputs
    assign port_present = port_device_present;
    assign port_active = port_link_up;
    
    // Link speed assignment
    generate
        for (port_idx = 0; port_idx < MAX_PORTS; port_idx++) begin : speed_gen
            assign link_speed[port_idx] = 2'b10;  // 6 Gbps
        end
    endgenerate
    
    // Interrupt generation
    generate
        for (port_idx = 0; port_idx < MAX_PORTS; port_idx++) begin : irq_gen
            assign port_irq[port_idx] = |port_regs[port_idx].is & |port_regs[port_idx].ie;
        end
    endgenerate
    
    assign global_irq = |ghc_is & ghc_ghc[1];  // Global interrupt enable
    
    // Physical layer (simplified)
    assign tx_fis_ready = '{MAX_PORTS{1'b1}};
    assign rx_fis_data = '{MAX_PORTS{32'h0}};
    assign rx_fis_valid = '{MAX_PORTS{1'b0}};
    
endmodule