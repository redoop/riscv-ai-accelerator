// AI Accelerator Interface
// Provides standardized interface for AI operations

interface ai_accel_if #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 64
);
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Control signals
    logic [ADDR_WIDTH-1:0]    addr;
    logic [DATA_WIDTH-1:0]    wdata;
    logic [DATA_WIDTH-1:0]    rdata;
    logic           req;
    logic           we;
    logic           ready;
    logic           error;
    
    // Task management
    logic           task_valid;
    logic [7:0]     task_id;
    logic [7:0]     task_type;
    logic           task_ready;
    logic           task_done;
    
    // Byte enable
    logic [DATA_WIDTH/8-1:0]     be;
    
    // Master modport (for CPU/core)
    modport master (
        input  clk, rst_n,
        output addr, wdata, req, we, be, task_valid, task_id, task_type,
        input  rdata, ready, error, task_ready, task_done
    );
    
    // Slave modport (for accelerators)
    modport slave (
        input  clk, rst_n,
        input  addr, wdata, req, we, be, task_valid, task_id, task_type,
        output rdata, ready, error, task_ready, task_done
    );
    
endinterface