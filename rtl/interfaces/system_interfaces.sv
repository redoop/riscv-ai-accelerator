// System-level interfaces for RISC-V AI accelerator chip
// Defines all major bus protocols and communication interfaces

// AXI4 interface for high-performance memory access
interface axi4_if #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,
    parameter ID_WIDTH = 8
);
    // Write address channel
    logic [ID_WIDTH-1:0]    awid;
    logic [ADDR_WIDTH-1:0]  awaddr;
    logic [7:0]             awlen;
    logic [2:0]             awsize;
    logic [1:0]             awburst;
    logic                   awlock;
    logic [3:0]             awcache;
    logic [2:0]             awprot;
    logic [3:0]             awqos;
    logic                   awvalid;
    logic                   awready;
    
    // Write data channel
    logic [DATA_WIDTH-1:0]      wdata;
    logic [DATA_WIDTH/8-1:0]    wstrb;
    logic                       wlast;
    logic                       wvalid;
    logic                       wready;
    
    // Write response channel
    logic [ID_WIDTH-1:0]    bid;
    logic [1:0]             bresp;
    logic                   bvalid;
    logic                   bready;
    
    // Read address channel
    logic [ID_WIDTH-1:0]    arid;
    logic [ADDR_WIDTH-1:0]  araddr;
    logic [7:0]             arlen;
    logic [2:0]             arsize;
    logic [1:0]             arburst;
    logic                   arlock;
    logic [3:0]             arcache;
    logic [2:0]             arprot;
    logic [3:0]             arqos;
    logic                   arvalid;
    logic                   arready;
    
    // Read data channel
    logic [ID_WIDTH-1:0]    rid;
    logic [DATA_WIDTH-1:0]  rdata;
    logic [1:0]             rresp;
    logic                   rlast;
    logic                   rvalid;
    logic                   rready;
    
    modport master (
        output awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot, awqos, awvalid,
        input  awready,
        output wdata, wstrb, wlast, wvalid,
        input  wready,
        input  bid, bresp, bvalid,
        output bready,
        output arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot, arqos, arvalid,
        input  arready,
        input  rid, rdata, rresp, rlast, rvalid,
        output rready
    );
    
    modport slave (
        input  awid, awaddr, awlen, awsize, awburst, awlock, awcache, awprot, awqos, awvalid,
        output awready,
        input  wdata, wstrb, wlast, wvalid,
        output wready,
        output bid, bresp, bvalid,
        input  bready,
        input  arid, araddr, arlen, arsize, arburst, arlock, arcache, arprot, arqos, arvalid,
        output arready,
        output rid, rdata, rresp, rlast, rvalid,
        input  rready
    );
endinterface

// AI Accelerator interface for TPU and VPU communication
interface ai_accel_if #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 64
);
    logic [ADDR_WIDTH-1:0]  addr;
    logic [DATA_WIDTH-1:0]  wdata;
    logic [DATA_WIDTH-1:0]  rdata;
    logic                   req;
    logic                   we;
    logic [7:0]             be;
    logic                   ready;
    logic                   error;
    
    // Task submission interface
    logic                   task_valid;
    logic [31:0]            task_id;
    logic [7:0]             task_type;  // MATMUL, CONV2D, etc.
    logic                   task_ready;
    logic                   task_done;
    
    modport master (
        output addr, wdata, req, we, be, task_valid, task_id, task_type,
        input  rdata, ready, error, task_ready, task_done
    );
    
    modport slave (
        input  addr, wdata, req, we, be, task_valid, task_id, task_type,
        output rdata, ready, error, task_ready, task_done
    );
endinterface

// Network-on-Chip (NoC) interface
interface noc_if #(
    parameter FLIT_WIDTH = 128,
    parameter ADDR_WIDTH = 8
);
    logic [FLIT_WIDTH-1:0]  flit_data;
    logic                   flit_valid;
    logic                   flit_ready;
    logic [ADDR_WIDTH-1:0]  src_addr;
    logic [ADDR_WIDTH-1:0]  dst_addr;
    logic                   head_flit;
    logic                   tail_flit;
    
    modport sender (
        output flit_data, flit_valid, src_addr, dst_addr, head_flit, tail_flit,
        input  flit_ready
    );
    
    modport receiver (
        input  flit_data, flit_valid, src_addr, dst_addr, head_flit, tail_flit,
        output flit_ready
    );
endinterface

// Power management interface
interface power_mgmt_if;
    logic [3:0]     voltage_level;
    logic [7:0]     freq_divider;
    logic           power_gate_en;
    logic           clock_gate_en;
    logic [15:0]    temp_sensor;
    logic           thermal_alert;
    logic           dvfs_req;
    logic           dvfs_ack;
    
    modport controller (
        output voltage_level, freq_divider, power_gate_en, clock_gate_en, dvfs_ack,
        input  temp_sensor, thermal_alert, dvfs_req
    );
    
    modport client (
        input  voltage_level, freq_divider, power_gate_en, clock_gate_en, dvfs_ack,
        output temp_sensor, thermal_alert, dvfs_req
    );
endinterface