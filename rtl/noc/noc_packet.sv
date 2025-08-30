// NoC Packet Definition
// Defines the packet structure for the mesh network

`ifndef NOC_PACKET_SV
`define NOC_PACKET_SV

// Packet types
typedef enum logic [2:0] {
    PKT_READ_REQ    = 3'b000,
    PKT_READ_RESP   = 3'b001,
    PKT_WRITE_REQ   = 3'b010,
    PKT_WRITE_RESP  = 3'b011,
    PKT_COHERENCE   = 3'b100,
    PKT_INTERRUPT   = 3'b101,
    PKT_DMA         = 3'b110,
    PKT_RESERVED    = 3'b111
} pkt_type_t;

// QoS priority levels
typedef enum logic [1:0] {
    QOS_LOW     = 2'b00,
    QOS_NORMAL  = 2'b01,
    QOS_HIGH    = 2'b10,
    QOS_URGENT  = 2'b11
} qos_level_t;

// Routing directions
typedef enum logic [2:0] {
    DIR_LOCAL = 3'b000,
    DIR_NORTH = 3'b001,
    DIR_SOUTH = 3'b010,
    DIR_EAST  = 3'b011,
    DIR_WEST  = 3'b100
} direction_t;

// NoC packet header structure
typedef struct packed {
    logic [3:0]     src_x;          // Source X coordinate
    logic [3:0]     src_y;          // Source Y coordinate
    logic [3:0]     dst_x;          // Destination X coordinate
    logic [3:0]     dst_y;          // Destination Y coordinate
    pkt_type_t      pkt_type;       // Packet type
    qos_level_t     qos;            // QoS priority
    logic [7:0]     pkt_id;         // Packet ID for tracking
    logic [5:0]     length;         // Payload length in flits
    logic           multicast;      // Multicast flag
    logic [2:0]     reserved;       // Reserved bits
} noc_header_t;

// NoC flit structure
typedef struct packed {
    logic           head;           // Head flit indicator
    logic           tail;           // Tail flit indicator
    noc_header_t    header;         // Header (valid only for head flit)
    logic [255:0]   data;           // Data payload (256 bits)
} noc_flit_t;

// Virtual channel parameters
parameter int VC_COUNT = 4;         // Number of virtual channels
parameter int VC_DEPTH = 8;         // Depth of each VC buffer
parameter int FLIT_WIDTH = 288;     // Total flit width (32 + 256)

`endif // NOC_PACKET_SV