// TPU Controller - Task scheduling and control logic
// Manages TPU operations, task queuing, and resource allocation
// Interfaces with system bus and manages DMA operations

`timescale 1ns/1ps

module tpu_controller #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter TASK_QUEUE_DEPTH = 16,
    parameter NUM_TPU_UNITS = 2
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // System bus interface
    input  logic [ADDR_WIDTH-1:0]  bus_addr,
    input  logic                    bus_read,
    input  logic                    bus_write,
    input  logic [DATA_WIDTH-1:0]  bus_wdata,
    output logic [DATA_WIDTH-1:0]  bus_rdata,
    output logic                    bus_ready,
    output logic                    bus_error,
    
    // TPU unit interfaces
    output logic [NUM_TPU_UNITS-1:0] tpu_enable,
    output logic [NUM_TPU_UNITS-1:0] tpu_start,
    input  logic [NUM_TPU_UNITS-1:0] tpu_done,
    input  logic [NUM_TPU_UNITS-1:0] tpu_busy,
    input  logic [NUM_TPU_UNITS-1:0] tpu_error,
    
    // Configuration outputs to TPU units
    output logic [7:0]              operation,
    output logic [1:0]              data_type,
    output logic [7:0]              matrix_size_m,
    output logic [7:0]              matrix_size_n,
    output logic [7:0]              matrix_size_k,
    
    // Memory interface for DMA
    output logic [ADDR_WIDTH-1:0]   mem_addr,
    output logic                    mem_read,
    output logic                    mem_write,
    output logic [DATA_WIDTH-1:0]   mem_wdata,
    input  logic [DATA_WIDTH-1:0]   mem_rdata,
    input  logic                    mem_ready,
    
    // Cache interface
    output logic                    cache_flush,
    output logic                    cache_invalidate,
    input  logic                    cache_ready,
    
    // Interrupt and status
    output logic                    interrupt,
    output logic [31:0]             status_reg,
    output logic [31:0]             performance_counter
);

    // Register map definitions
    localparam CTRL_REG_ADDR     = 32'h0000;
    localparam STATUS_REG_ADDR   = 32'h0004;
    localparam CONFIG_REG_ADDR   = 32'h0008;
    localparam TASK_REG_ADDR     = 32'h000C;
    localparam PERF_REG_ADDR     = 32'h0010;
    localparam DMA_SRC_ADDR      = 32'h0014;
    localparam DMA_DST_ADDR      = 32'h0018;
    localparam DMA_SIZE_ADDR     = 32'h001C;
    localparam CACHE_CTRL_ADDR   = 32'h0020;
    
    // Control register bits
    localparam CTRL_ENABLE_BIT   = 0;
    localparam CTRL_START_BIT    = 1;
    localparam CTRL_RESET_BIT    = 2;
    localparam CTRL_INT_EN_BIT   = 3;
    
    // Status register bits
    localparam STATUS_BUSY_BIT   = 0;
    localparam STATUS_DONE_BIT   = 1;
    localparam STATUS_ERROR_BIT  = 2;
    localparam STATUS_INT_BIT    = 3;
    
    // Task descriptor structure
    typedef struct packed {
        logic [7:0]  operation;
        logic [1:0]  data_type;
        logic [7:0]  matrix_m;
        logic [7:0]  matrix_n;
        logic [7:0]  matrix_k;
        logic [31:0] src_addr_a;
        logic [31:0] src_addr_b;
        logic [31:0] dst_addr;
        logic [3:0]  tpu_unit;
        logic        valid;
    } task_descriptor_t;
    
    // Internal registers
    logic [31:0] ctrl_reg;
    logic [31:0] config_reg;
    logic [31:0] dma_src_addr;
    logic [31:0] dma_dst_addr;
    logic [31:0] dma_size;
    logic [31:0] cache_ctrl_reg;
    
    // Task queue
    task_descriptor_t task_queue [TASK_QUEUE_DEPTH-1:0];
    logic [$clog2(TASK_QUEUE_DEPTH)-1:0] task_write_ptr;
    logic [$clog2(TASK_QUEUE_DEPTH)-1:0] task_read_ptr;
    logic task_queue_full;
    logic task_queue_empty;
    logic [$clog2(TASK_QUEUE_DEPTH):0] task_count;
    
    // State machine
    typedef enum logic [3:0] {
        IDLE,
        TASK_DISPATCH,
        WAIT_COMPLETION,
        DMA_TRANSFER,
        CACHE_OPERATION,
        ERROR_HANDLING,
        INTERRUPT_SERVICE
    } controller_state_t;
    
    controller_state_t current_state, next_state;
    
    // Task scheduling
    logic [NUM_TPU_UNITS-1:0] tpu_available;
    logic [$clog2(NUM_TPU_UNITS)-1:0] selected_tpu;
    logic task_dispatch_valid;
    task_descriptor_t current_task;
    
    // DMA control
    logic dma_active;
    logic [31:0] dma_current_addr;
    logic [31:0] dma_remaining_size;
    
    // Performance monitoring
    logic [31:0] cycle_counter;
    logic [31:0] task_counter;
    logic [31:0] error_counter;
    
    // Bus interface logic
    always_comb begin
        bus_ready = 1'b1;  // Always ready for simplicity
        bus_error = 1'b0;
        
        case (bus_addr)
            CTRL_REG_ADDR: begin
                if (bus_read)
                    bus_rdata = ctrl_reg;
                else
                    bus_rdata = '0;
            end
            STATUS_REG_ADDR: begin
                bus_rdata = status_reg;
            end
            CONFIG_REG_ADDR: begin
                if (bus_read)
                    bus_rdata = config_reg;
                else
                    bus_rdata = '0;
            end
            PERF_REG_ADDR: begin
                bus_rdata = performance_counter;
            end
            DMA_SRC_ADDR: begin
                if (bus_read)
                    bus_rdata = dma_src_addr;
                else
                    bus_rdata = '0;
            end
            DMA_DST_ADDR: begin
                if (bus_read)
                    bus_rdata = dma_dst_addr;
                else
                    bus_rdata = '0;
            end
            DMA_SIZE_ADDR: begin
                if (bus_read)
                    bus_rdata = dma_size;
                else
                    bus_rdata = '0;
            end
            CACHE_CTRL_ADDR: begin
                if (bus_read)
                    bus_rdata = cache_ctrl_reg;
                else
                    bus_rdata = '0;
            end
            default: begin
                bus_rdata = 32'hDEADBEEF;
            end
        endcase
    end
    
    // Register write logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg <= '0;
            config_reg <= '0;
            dma_src_addr <= '0;
            dma_dst_addr <= '0;
            dma_size <= '0;
            cache_ctrl_reg <= '0;
        end else if (bus_write) begin
            case (bus_addr)
                CTRL_REG_ADDR: begin
                    ctrl_reg <= bus_wdata;
                end
                CONFIG_REG_ADDR: begin
                    config_reg <= bus_wdata;
                end
                DMA_SRC_ADDR: begin
                    dma_src_addr <= bus_wdata;
                end
                DMA_DST_ADDR: begin
                    dma_dst_addr <= bus_wdata;
                end
                DMA_SIZE_ADDR: begin
                    dma_size <= bus_wdata;
                end
                CACHE_CTRL_ADDR: begin
                    cache_ctrl_reg <= bus_wdata;
                end
                TASK_REG_ADDR: begin
                    // Task submission - add to queue
                    if (!task_queue_full) begin
                        task_queue[task_write_ptr].operation <= bus_wdata[7:0];
                        task_queue[task_write_ptr].data_type <= bus_wdata[9:8];
                        task_queue[task_write_ptr].matrix_m <= bus_wdata[17:10];
                        task_queue[task_write_ptr].matrix_n <= bus_wdata[25:18];
                        task_queue[task_write_ptr].matrix_k <= config_reg[7:0];
                        task_queue[task_write_ptr].src_addr_a <= dma_src_addr;
                        task_queue[task_write_ptr].src_addr_b <= dma_dst_addr;
                        task_queue[task_write_ptr].dst_addr <= dma_dst_addr + dma_size;
                        task_queue[task_write_ptr].tpu_unit <= bus_wdata[31:28];
                        task_queue[task_write_ptr].valid <= 1'b1;
                    end
                end
            endcase
        end
    end
    
    // Task queue management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            task_write_ptr <= '0;
            task_read_ptr <= '0;
            task_count <= '0;
            task_queue_full <= 1'b0;
            task_queue_empty <= 1'b1;
        end else begin
            // Task submission
            if (bus_write && bus_addr == TASK_REG_ADDR && !task_queue_full) begin
                task_write_ptr <= task_write_ptr + 1;
                task_count <= task_count + 1;
                task_queue_empty <= 1'b0;
                
                if (task_count + 1 == TASK_QUEUE_DEPTH)
                    task_queue_full <= 1'b1;
            end
            
            // Task completion
            if (task_dispatch_valid && !task_queue_empty) begin
                task_read_ptr <= task_read_ptr + 1;
                task_count <= task_count - 1;
                task_queue_full <= 1'b0;
                
                if (task_count - 1 == 0)
                    task_queue_empty <= 1'b1;
            end
        end
    end
    
    // TPU availability tracking
    always_comb begin
        for (int i = 0; i < NUM_TPU_UNITS; i++) begin
            tpu_available[i] = tpu_enable[i] && !tpu_busy[i] && !tpu_error[i];
        end
    end
    
    // TPU selection logic (round-robin)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            selected_tpu <= '0;
        end else if (task_dispatch_valid) begin
            // Find next available TPU
            for (int i = 0; i < NUM_TPU_UNITS; i++) begin
                if (tpu_available[(selected_tpu + i) % NUM_TPU_UNITS]) begin
                    selected_tpu <= (selected_tpu + i) % NUM_TPU_UNITS;
                    break;
                end
            end
        end
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (ctrl_reg[CTRL_ENABLE_BIT] && !task_queue_empty)
                    next_state = TASK_DISPATCH;
                else if (cache_ctrl_reg[0])
                    next_state = CACHE_OPERATION;
                else if (dma_size > 0)
                    next_state = DMA_TRANSFER;
            end
            
            TASK_DISPATCH: begin
                if (|tpu_available)
                    next_state = WAIT_COMPLETION;
                else
                    next_state = IDLE;
            end
            
            WAIT_COMPLETION: begin
                if (tpu_done[selected_tpu])
                    next_state = IDLE;
                else if (tpu_error[selected_tpu])
                    next_state = ERROR_HANDLING;
            end
            
            DMA_TRANSFER: begin
                if (dma_remaining_size == 0)
                    next_state = IDLE;
            end
            
            CACHE_OPERATION: begin
                if (cache_ready)
                    next_state = IDLE;
            end
            
            ERROR_HANDLING: begin
                next_state = INTERRUPT_SERVICE;
            end
            
            INTERRUPT_SERVICE: begin
                if (!ctrl_reg[CTRL_INT_EN_BIT])
                    next_state = IDLE;
            end
        endcase
    end
    
    // Control signal generation
    always_comb begin
        // Default values
        tpu_enable = '0;
        tpu_start = '0;
        task_dispatch_valid = 1'b0;
        
        case (current_state)
            TASK_DISPATCH: begin
                if (!task_queue_empty && |tpu_available) begin
                    tpu_enable[selected_tpu] = 1'b1;
                    tpu_start[selected_tpu] = 1'b1;
                    task_dispatch_valid = 1'b1;
                end
            end
            
            WAIT_COMPLETION: begin
                tpu_enable[selected_tpu] = 1'b1;
            end
            
            default: begin
                if (ctrl_reg[CTRL_ENABLE_BIT]) begin
                    tpu_enable = {NUM_TPU_UNITS{1'b1}};
                end
            end
        endcase
    end
    
    // Current task assignment
    always_comb begin
        if (!task_queue_empty) begin
            current_task = task_queue[task_read_ptr];
        end else begin
            current_task = '0;
        end
    end
    
    // Configuration outputs
    assign operation = current_task.operation;
    assign data_type = current_task.data_type;
    assign matrix_size_m = current_task.matrix_m;
    assign matrix_size_n = current_task.matrix_n;
    assign matrix_size_k = current_task.matrix_k;
    
    // DMA control
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_active <= 1'b0;
            dma_current_addr <= '0;
            dma_remaining_size <= '0;
        end else begin
            case (current_state)
                DMA_TRANSFER: begin
                    if (!dma_active) begin
                        dma_active <= 1'b1;
                        dma_current_addr <= dma_src_addr;
                        dma_remaining_size <= dma_size;
                    end else if (mem_ready && dma_remaining_size > 0) begin
                        dma_current_addr <= dma_current_addr + 4;
                        dma_remaining_size <= dma_remaining_size - 4;
                    end else if (dma_remaining_size == 0) begin
                        dma_active <= 1'b0;
                    end
                end
                default: begin
                    dma_active <= 1'b0;
                end
            endcase
        end
    end
    
    // Memory interface
    assign mem_addr = dma_current_addr;
    assign mem_read = dma_active && (dma_remaining_size > 0);
    assign mem_write = 1'b0;  // Read-only DMA for now
    assign mem_wdata = '0;
    
    // Cache control
    assign cache_flush = cache_ctrl_reg[0];
    assign cache_invalidate = cache_ctrl_reg[1];
    
    // Status register
    always_comb begin
        status_reg = '0;
        status_reg[STATUS_BUSY_BIT] = |tpu_busy;
        status_reg[STATUS_DONE_BIT] = |tpu_done;
        status_reg[STATUS_ERROR_BIT] = |tpu_error;
        status_reg[STATUS_INT_BIT] = interrupt;
        status_reg[7:4] = task_count[3:0];
        status_reg[15:8] = current_state;
    end
    
    // Performance counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= '0;
            task_counter <= '0;
            error_counter <= '0;
        end else begin
            cycle_counter <= cycle_counter + 1;
            
            if (task_dispatch_valid)
                task_counter <= task_counter + 1;
                
            if (|tpu_error)
                error_counter <= error_counter + 1;
        end
    end
    
    assign performance_counter = {error_counter[7:0], task_counter[11:0], cycle_counter[11:0]};
    
    // Interrupt generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            interrupt <= 1'b0;
        end else begin
            interrupt <= ctrl_reg[CTRL_INT_EN_BIT] && 
                        (|tpu_done || |tpu_error || task_queue_full);
        end
    end

endmodule