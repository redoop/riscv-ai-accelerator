// Placeholder for ysyx_00000000 CPU module
module ysyx_00000000 (
  input         clock,
  input         reset,
  input         io_interrupt,
  input         io_master_awready,
  output        io_master_awvalid,
  output [31:0] io_master_awaddr,
  output [3:0]  io_master_awid,
  output [7:0]  io_master_awlen,
  output [2:0]  io_master_awsize,
  output [1:0]  io_master_awburst,
  output        io_master_awlock,
  output [3:0]  io_master_awcache,
  output [2:0]  io_master_awprot,
  output [3:0]  io_master_awqos,
  input         io_master_wready,
  output        io_master_wvalid,
  output [31:0] io_master_wdata,
  output [3:0]  io_master_wstrb,
  output        io_master_wlast,
  output        io_master_bready,
  input         io_master_bvalid,
  input  [1:0]  io_master_bresp,
  input  [3:0]  io_master_bid,
  input         io_master_arready,
  output        io_master_arvalid,
  output [31:0] io_master_araddr,
  output [3:0]  io_master_arid,
  output [7:0]  io_master_arlen,
  output [2:0]  io_master_arsize,
  output [1:0]  io_master_arburst,
  output        io_master_arlock,
  output [3:0]  io_master_arcache,
  output [2:0]  io_master_arprot,
  output [3:0]  io_master_arqos,
  output        io_master_rready,
  input         io_master_rvalid,
  input  [1:0]  io_master_rresp,
  input  [31:0] io_master_rdata,
  input         io_master_rlast,
  input  [3:0]  io_master_rid,
  output        io_slave_awready,
  input         io_slave_awvalid,
  input  [31:0] io_slave_awaddr,
  input  [3:0]  io_slave_awid,
  input  [7:0]  io_slave_awlen,
  input  [2:0]  io_slave_awsize,
  input  [1:0]  io_slave_awburst,
  input         io_slave_awlock,
  input  [3:0]  io_slave_awcache,
  input  [2:0]  io_slave_awprot,
  input  [3:0]  io_slave_awqos,
  output        io_slave_wready,
  input         io_slave_wvalid,
  input  [31:0] io_slave_wdata,
  input  [3:0]  io_slave_wstrb,
  input         io_slave_wlast,
  input         io_slave_bready,
  output        io_slave_bvalid,
  output [1:0]  io_slave_bresp,
  output [3:0]  io_slave_bid,
  output        io_slave_arready,
  input         io_slave_arvalid,
  input  [31:0] io_slave_araddr,
  input  [3:0]  io_slave_arid,
  input  [7:0]  io_slave_arlen,
  input  [2:0]  io_slave_arsize,
  input  [1:0]  io_slave_arburst,
  input         io_slave_arlock,
  input  [3:0]  io_slave_arcache,
  input  [2:0]  io_slave_arprot,
  input  [3:0]  io_slave_arqos,
  input         io_slave_rready,
  output        io_slave_rvalid,
  output [1:0]  io_slave_rresp,
  output [31:0] io_slave_rdata,
  output        io_slave_rlast,
  output [3:0]  io_slave_rid
);
  assign io_master_awvalid = 1'b0;
  assign io_master_awaddr = 32'h0;
  assign io_master_awid = 4'h0;
  assign io_master_awlen = 8'h0;
  assign io_master_awsize = 3'h0;
  assign io_master_awburst = 2'h0;
  assign io_master_awlock = 1'b0;
  assign io_master_awcache = 4'h0;
  assign io_master_awprot = 3'h0;
  assign io_master_awqos = 4'h0;
  assign io_master_wvalid = 1'b0;
  assign io_master_wdata = 32'h0;
  assign io_master_wstrb = 4'h0;
  assign io_master_wlast = 1'b0;
  assign io_master_bready = 1'b0;
  assign io_master_arvalid = 1'b0;
  assign io_master_araddr = 32'h0;
  assign io_master_arid = 4'h0;
  assign io_master_arlen = 8'h0;
  assign io_master_arsize = 3'h0;
  assign io_master_arburst = 2'h0;
  assign io_master_arlock = 1'b0;
  assign io_master_arcache = 4'h0;
  assign io_master_arprot = 3'h0;
  assign io_master_arqos = 4'h0;
  assign io_master_rready = 1'b0;
  assign io_slave_awready = 1'b0;
  assign io_slave_wready = 1'b0;
  assign io_slave_bvalid = 1'b0;
  assign io_slave_bresp = 2'h0;
  assign io_slave_bid = 4'h0;
  assign io_slave_arready = 1'b0;
  assign io_slave_rvalid = 1'b0;
  assign io_slave_rresp = 2'h0;
  assign io_slave_rdata = 32'h0;
  assign io_slave_rlast = 1'b0;
  assign io_slave_rid = 4'h0;
endmodule
