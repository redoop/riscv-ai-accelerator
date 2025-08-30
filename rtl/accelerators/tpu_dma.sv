// TPU DMA Controller - High-speed data transfer engine
// Handles bulk data transfers between memory and TPU caches
// Supports scatter-gather operations and multiple channels

`timescale 1ns/1ps

module tpu_dma #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter NUM_CHANNELS = 4,
    parameter FIFO_DEPTH = 64,
    parameter MAX_BURST_SIZE = 16
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic [ADDR_WIDTH-1:0]  ctrl_addr,
    input  logic                    ctrl_read,
    input  logic                    ctrl_write,
    input  logic [DATA_WIDTH-1:0]  ctrl_wdata,
    output logic [DATA_WIDTH-1:0]  ctrl_rdata,
    output logic                    ctrl_ready,
    
    // Memory interface (source)
    output logic [ADDR_WIDTH-1:0]  src_addr,
    output logic                    src_read,
    input  logic [DATA_WIDTH-1:0]  src_rdata,
    input  logic                    src_ready,
    
    // Memory interface (destination)
    output logic [ADDR_WIDTH-1:0]  dst_addr,
    output logic                    dst_write,
    output logic [DATA_WIDTH-1:0]  dst_wdata,
    input  logic                    dst_ready,
    
    // Cache interface
    output logic [ADDR_WIDTH-1:0]  cache_addr,
    output logic                    cache_read,
    output logic                    cache_write,
    output logic [DATA_WIDTH-1:0]  cache_wdata,
    input  logic [DATA_WIDTH-1:0]  cache_rdata,
    input  logic                    cache_ready,
    
    // Interrupt and status
    output logic [NUM_CHANNELS-1:0] channel_done,
    output logic [NUM_CHANNELS-1:0] channel_error,
    output logic                    dma_interrupt
);

    // Register map
    localparam CTRL_REG_BASE     = 32'h0000;
    localparam STATUS_REG_BASE   = 32'h0004;
    localparam SRC_ADDR_BASE     = 32'h0008;
    localparam DST_ADDR_BASE     = 32'h000C;
    localparam SIZE_REG_BASE     = 32'h0010;
    localparam CONFIG_REG_BASE   = 32'h0014;
    
    // DMA descriptor structure
    typedef struct packed {
        logic [31:0] src_addr;
        logic [31:0] dst_addr;
        logic [31:0] size;
        logic [7:0]  burst_size;
        logic [1:0]  transfer_type;  // 00: mem2mem, 01: mem2cache, 10: cache2mem
        logic        interrupt_en;
        logic        valid;
        logic        active;
    } dma_descriptor_t;
    
    // Channel descriptors
    dma_descriptor_t channels [NUM_CHANNELS-1:0];
    
    // Channel state machines
    typedef enum logic [2:0] {
        CH_IDLE,
        CH_READ_SETUP,
        CH_READ_DATA,
        CH_WRITE_SETUP,
        CH_WRITE_DATA,
        CH_COMPLETE,
        CH_ERROR
    } channel_state_t;
    
    channel_state_t channel_state [NUM_CHANNELS-1:0];
    
    // Channel control signals
    logic [NUM_CHANNELS-1:0] channel_enable;
    logic [NUM_CHANNELS-1:0] channel_start;
    logic [NUM_CHANNELS-1:0] channel_busy;
    
    // Current active channel (round-robin arbitration)
    logic [$clog2(NUM_CHANNELS)-1:0] active_channel;
    logic [$clog2(NUM_CHANNELS)-1:0] next_channel;
    
    // Transfer counters
    logic [31:0] current_src_addr [NUM_CHANNELS-1:0];
    logic [31:0] current_dst_addr [NUM_CHANNELS-1:0];
    logic [31:0] remaining_size [NUM_CHANNELS-1:0];
    logic [7:0]  current_burst [NUM_CHANNELS-1:0];
    
    // Data FIFOs for each channel
    logic [DATA_WIDTH-1:0] channel_fifo [NUM_CHANNELS-1:0][FIFO_DEPTH-1:0];
    logic [$clog2(FIFO_DEPTH):0] fifo_write_ptr [NUM_CHANNELS-1:0];
    logic [$clog2(FIFO_DEPTH):0] fifo_read_ptr [NUM_CHANNELS-1:0];
    logic fifo_full [NUM_CHANNELS-1:0];
    logic fifo_empty [NUM_CHANNELS-1:0];
    
    // Control register access
    // Decode channel from address
    logic [$clog2(NUM_CHANNELS)-1:0] ch_sel;
    assign ch_sel = ctrl_addr[7:4];  // Channel select from address bits
    
    always_comb begin
        ctrl_ready = 1'b1;
        ctrl_rdata = '0;
        
        if (ch_sel < NUM_CHANNELS) begin
            case (ctrl_addr[3:0])
                4'h0: ctrl_rdata = {channels[ch_sel].valid, channels[ch_sel].active, 
                                   channel_busy[ch_sel], channel_done[ch_sel], 
                                   channel_error[ch_sel], 27'h0};
                4'h4: ctrl_rdata = {channel_state[ch_sel], 29'h0};
                4'h8: ctrl_rdata = channels[ch_sel].src_addr;
                4'hC: ctrl_rdata = channels[ch_sel].dst_addr;
                default: ctrl_rdata = 32'hDEADBEEF;
            endcase
        end
    end
    
    // Control register writes
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_CHANNELS; i++) begin
                channels[i] <= '0;
                channel_enable[i] <= 1'b0;
                channel_start[i] <= 1'b0;
            end
        end else if (ctrl_write) begin
            logic [$clog2(NUM_CHANNELS)-1:0] ch_sel;
            ch_sel = ctrl_addr[7:4];
            
            if (ch_sel < NUM_CHANNELS) begin
                case (ctrl_addr[3:0])
                    4'h0: begin  // Control register
                        channel_enable[ch_sel] <= ctrl_wdata[0];
                        channel_start[ch_sel] <= ctrl_wdata[1];
                        channels[ch_sel].interrupt_en <= ctrl_wdata[2];
                        channels[ch_sel].transfer_type <= ctrl_wdata[5:4];
                    end
                    4'h8: channels[ch_sel].src_addr <= ctrl_wdata;
                    4'hC: channels[ch_sel].dst_addr <= ctrl_wdata;
                    5'h10: channels[ch_sel].size <= ctrl_wdata;
                    5'h14: begin
                        channels[ch_sel].burst_size <= ctrl_wdata[7:0];
                        channels[ch_sel].valid <= ctrl_wdata[31];
                    end
                endcase
            end
        end
    end
    
    // Channel arbitration (round-robin)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_channel <= '0;
        end else begin
            // Find next active channel
            next_channel = active_channel;
            for (int i = 1; i <= NUM_CHANNELS; i++) begin
                logic [$clog2(NUM_CHANNELS)-1:0] candidate;
                candidate = (active_channel + i) % NUM_CHANNELS;
                if (channel_enable[candidate] && channels[candidate].valid && 
                    !channel_busy[candidate] && channel_start[candidate]) begin
                    next_channel = candidate;
                    break;
                end
            end
            active_channel <= next_channel;
        end
    end
    
    // Channel state machines
    genvar ch;
    generate
        for (ch = 0; ch < NUM_CHANNELS; ch++) begin : gen_channels
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    channel_state[ch] <= CH_IDLE;
                    current_src_addr[ch] <= '0;
                    current_dst_addr[ch] <= '0;
                    remaining_size[ch] <= '0;
                    current_burst[ch] <= '0;
                    channel_busy[ch] <= 1'b0;
                    channel_done[ch] <= 1'b0;
                    channel_error[ch] <= 1'b0;
                end else begin
                    case (channel_state[ch])
                        CH_IDLE: begin
                            channel_done[ch] <= 1'b0;
                            channel_error[ch] <= 1'b0;
                            
                            if (channel_enable[ch] && channels[ch].valid && channel_start[ch]) begin
                                channel_state[ch] <= CH_READ_SETUP;
                                current_src_addr[ch] <= channels[ch].src_addr;
                                current_dst_addr[ch] <= channels[ch].dst_addr;
                                remaining_size[ch] <= channels[ch].size;
                                current_burst[ch] <= (channels[ch].size < channels[ch].burst_size) ? 
                                                   channels[ch].size[7:0] : channels[ch].burst_size;
                                channel_busy[ch] <= 1'b1;
                                channels[ch].active <= 1'b1;
                            end
                        end
                        
                        CH_READ_SETUP: begin
                            if (active_channel == ch) begin
                                channel_state[ch] <= CH_READ_DATA;
                            end
                        end
                        
                        CH_READ_DATA: begin
                            if (active_channel == ch) begin
                                // Handle read based on transfer type
                                case (channels[ch].transfer_type)
                                    2'b00, 2'b01: begin  // mem2mem or mem2cache
                                        if (src_ready && !fifo_full[ch]) begin
                                            // Read data and store in FIFO
                                            current_src_addr[ch] <= current_src_addr[ch] + 4;
                                            current_burst[ch] <= current_burst[ch] - 1;
                                            
                                            if (current_burst[ch] == 1) begin
                                                channel_state[ch] <= CH_WRITE_SETUP;
                                            end
                                        end
                                    end
                                    2'b10: begin  // cache2mem
                                        if (cache_ready && !fifo_full[ch]) begin
                                            current_src_addr[ch] <= current_src_addr[ch] + 4;
                                            current_burst[ch] <= current_burst[ch] - 1;
                                            
                                            if (current_burst[ch] == 1) begin
                                                channel_state[ch] <= CH_WRITE_SETUP;
                                            end
                                        end
                                    end
                                endcase
                            end
                        end
                        
                        CH_WRITE_SETUP: begin
                            if (active_channel == ch) begin
                                channel_state[ch] <= CH_WRITE_DATA;
                                current_burst[ch] <= (remaining_size[ch] < channels[ch].burst_size) ? 
                                                   remaining_size[ch][7:0] : channels[ch].burst_size;
                            end
                        end
                        
                        CH_WRITE_DATA: begin
                            if (active_channel == ch) begin
                                // Handle write based on transfer type
                                case (channels[ch].transfer_type)
                                    2'b00, 2'b10: begin  // mem2mem or cache2mem
                                        if (dst_ready && !fifo_empty[ch]) begin
                                            current_dst_addr[ch] <= current_dst_addr[ch] + 4;
                                            remaining_size[ch] <= remaining_size[ch] - 4;
                                            current_burst[ch] <= current_burst[ch] - 1;
                                            
                                            if (current_burst[ch] == 1) begin
                                                if (remaining_size[ch] <= 4) begin
                                                    channel_state[ch] <= CH_COMPLETE;
                                                end else begin
                                                    channel_state[ch] <= CH_READ_SETUP;
                                                end
                                            end
                                        end
                                    end
                                    2'b01: begin  // mem2cache
                                        if (cache_ready && !fifo_empty[ch]) begin
                                            current_dst_addr[ch] <= current_dst_addr[ch] + 4;
                                            remaining_size[ch] <= remaining_size[ch] - 4;
                                            current_burst[ch] <= current_burst[ch] - 1;
                                            
                                            if (current_burst[ch] == 1) begin
                                                if (remaining_size[ch] <= 4) begin
                                                    channel_state[ch] <= CH_COMPLETE;
                                                end else begin
                                                    channel_state[ch] <= CH_READ_SETUP;
                                                end
                                            end
                                        end
                                    end
                                endcase
                            end
                        end
                        
                        CH_COMPLETE: begin
                            channel_state[ch] <= CH_IDLE;
                            channel_busy[ch] <= 1'b0;
                            channel_done[ch] <= 1'b1;
                            channels[ch].active <= 1'b0;
                            channel_start[ch] <= 1'b0;
                        end
                        
                        CH_ERROR: begin
                            channel_state[ch] <= CH_IDLE;
                            channel_busy[ch] <= 1'b0;
                            channel_error[ch] <= 1'b1;
                            channels[ch].active <= 1'b0;
                            channel_start[ch] <= 1'b0;
                        end
                    endcase
                end
            end
            
            // FIFO management for each channel
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    fifo_write_ptr[ch] <= '0;
                    fifo_read_ptr[ch] <= '0;
                    fifo_full[ch] <= 1'b0;
                    fifo_empty[ch] <= 1'b1;
                end else begin
                    // FIFO write (during read phase)
                    if (channel_state[ch] == CH_READ_DATA && active_channel == ch) begin
                        case (channels[ch].transfer_type)
                            2'b00, 2'b01: begin
                                if (src_ready && !fifo_full[ch]) begin
                                    channel_fifo[ch][fifo_write_ptr[ch]] <= src_rdata;
                                    fifo_write_ptr[ch] <= fifo_write_ptr[ch] + 1;
                                    fifo_empty[ch] <= 1'b0;
                                    
                                    if (fifo_write_ptr[ch] + 1 == fifo_read_ptr[ch]) begin
                                        fifo_full[ch] <= 1'b1;
                                    end
                                end
                            end
                            2'b10: begin
                                if (cache_ready && !fifo_full[ch]) begin
                                    channel_fifo[ch][fifo_write_ptr[ch]] <= cache_rdata;
                                    fifo_write_ptr[ch] <= fifo_write_ptr[ch] + 1;
                                    fifo_empty[ch] <= 1'b0;
                                    
                                    if (fifo_write_ptr[ch] + 1 == fifo_read_ptr[ch]) begin
                                        fifo_full[ch] <= 1'b1;
                                    end
                                end
                            end
                        endcase
                    end
                    
                    // FIFO read (during write phase)
                    if (channel_state[ch] == CH_WRITE_DATA && active_channel == ch) begin
                        case (channels[ch].transfer_type)
                            2'b00, 2'b10: begin
                                if (dst_ready && !fifo_empty[ch]) begin
                                    fifo_read_ptr[ch] <= fifo_read_ptr[ch] + 1;
                                    fifo_full[ch] <= 1'b0;
                                    
                                    if (fifo_read_ptr[ch] + 1 == fifo_write_ptr[ch]) begin
                                        fifo_empty[ch] <= 1'b1;
                                    end
                                end
                            end
                            2'b01: begin
                                if (cache_ready && !fifo_empty[ch]) begin
                                    fifo_read_ptr[ch] <= fifo_read_ptr[ch] + 1;
                                    fifo_full[ch] <= 1'b0;
                                    
                                    if (fifo_read_ptr[ch] + 1 == fifo_write_ptr[ch]) begin
                                        fifo_empty[ch] <= 1'b1;
                                    end
                                end
                            end
                        endcase
                    end
                end
            end
        end
    endgenerate
    
    // Memory interface multiplexing
    always_comb begin
        // Default values
        src_addr = '0;
        src_read = 1'b0;
        dst_addr = '0;
        dst_write = 1'b0;
        dst_wdata = '0;
        cache_addr = '0;
        cache_read = 1'b0;
        cache_write = 1'b0;
        cache_wdata = '0;
        
        if (active_channel < NUM_CHANNELS) begin
            case (channel_state[active_channel])
                CH_READ_DATA: begin
                    case (channels[active_channel].transfer_type)
                        2'b00, 2'b01: begin  // mem2mem or mem2cache
                            src_addr = current_src_addr[active_channel];
                            src_read = !fifo_full[active_channel];
                        end
                        2'b10: begin  // cache2mem
                            cache_addr = current_src_addr[active_channel];
                            cache_read = !fifo_full[active_channel];
                        end
                    endcase
                end
                
                CH_WRITE_DATA: begin
                    case (channels[active_channel].transfer_type)
                        2'b00, 2'b10: begin  // mem2mem or cache2mem
                            dst_addr = current_dst_addr[active_channel];
                            dst_write = !fifo_empty[active_channel];
                            dst_wdata = channel_fifo[active_channel][fifo_read_ptr[active_channel]];
                        end
                        2'b01: begin  // mem2cache
                            cache_addr = current_dst_addr[active_channel];
                            cache_write = !fifo_empty[active_channel];
                            cache_wdata = channel_fifo[active_channel][fifo_read_ptr[active_channel]];
                        end
                    endcase
                end
            endcase
        end
    end
    
    // Interrupt generation
    always_comb begin
        dma_interrupt = 1'b0;
        for (int i = 0; i < NUM_CHANNELS; i++) begin
            if (channels[i].interrupt_en && (channel_done[i] || channel_error[i])) begin
                dma_interrupt = 1'b1;
                break;
            end
        end
    end

endmodule