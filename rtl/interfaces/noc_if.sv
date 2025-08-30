// Network on Chip (NoC) Interface
// Provides packet-based communication interface

interface noc_if #(
    parameter FLIT_WIDTH = 32,
    parameter ADDR_WIDTH = 8
);
    
    // Flit transmission
    logic                       flit_valid;
    logic                       flit_ready;
    logic [FLIT_WIDTH-1:0]      flit_data;
    
    // Addressing
    logic [ADDR_WIDTH-1:0]      src_addr;
    logic [ADDR_WIDTH-1:0]      dst_addr;
    
    // Packet control
    logic                       head_flit;
    logic                       tail_flit;
    
    // Flow control
    logic                       credit_valid;
    logic [3:0]                 credit_count;
    
    // Sender modport
    modport sender (
        output flit_valid, flit_data, src_addr, dst_addr, head_flit, tail_flit,
        input  flit_ready,
        input  credit_valid, credit_count
    );
    
    // Receiver modport
    modport receiver (
        input  flit_valid, flit_data, src_addr, dst_addr, head_flit, tail_flit,
        output flit_ready,
        output credit_valid, credit_count
    );
    
endinterface