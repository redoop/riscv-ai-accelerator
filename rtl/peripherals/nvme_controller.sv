// NVMe Controller Module
// Implements NVMe 1.4 controller with PCIe interface
// Supports NVMe SSDs with multiple namespaces and queues

module nvme_controller #(
    parameter DATA_WIDTH = 64,
    parameter ADDR_WIDTH = 64,
    parameter MAX_QUEUES = 64,
    parameter QUEUE_DEPTH = 1024,
    parameter MAX_NAMESPACES = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // PCIe interface (connects to PCIe controller)
    input  logic [DATA_WIDTH-1:0]   pcie_rx_data,
    input  logic                    pcie_rx_valid,
    output logic                    pcie_rx_ready,
    output logic [DATA_WIDTH-1:0]   pcie_tx_data,
    output logic                    pcie_tx_valid,
    input  logic                    pcie_tx_ready,
    
    // Memory interface for NVMe registers and queues
    output logic                    mem_req,
    output logic [ADDR_WIDTH-1:0]   mem_addr,
    output logic [DATA_WIDTH-1:0]   mem_wdata,
    input  logic [DATA_WIDTH-1:0]   mem_rdata,
    output logic                    mem_write,
    output logic [7:0]              mem_size,
    input  logic                    mem_ready,
    input  logic                    mem_valid,
    
    // DMA interface for data transfers
    output logic                    dma_req,
    input  logic                    dma_ack,
    output logic [ADDR_WIDTH-1:0]   dma_src_addr,
    output logic [ADDR_WIDTH-1:0]   dma_dst_addr,
    output logic [31:0]             dma_length,
    output logic                    dma_write,
    input  logic                    dma_done,
    input  logic                    dma_error,
    
    // Interrupt interface
    output logic [15:0]             msi_vector,
    output logic                    msi_valid,
    input  logic                    msi_ready,
    
    // Status and statistics
    output logic                    controller_ready,
    output logic [15:0]             active_queues,
    output logic [31:0]             commands_processed,
    output logic [31:0]             error_count,
    output logic [63:0]             data_transferred
);

    // NVMe command opcodes
    typedef enum logic [7:0] {
        // Admin commands
        NVME_ADMIN_DELETE_SQ    = 8'h00,
        NVME_ADMIN_CREATE_SQ    = 8'h01,
        NVME_ADMIN_DELETE_CQ    = 8'h04,
        NVME_ADMIN_CREATE_CQ    = 8'h05,
        NVME_ADMIN_IDENTIFY     = 8'h06,
        NVME_ADMIN_SET_FEATURES = 8'h09,
        NVME_ADMIN_GET_FEATURES = 8'h0A,
        
        // I/O commands
        NVME_CMD_FLUSH          = 8'h00,
        NVME_CMD_WRITE          = 8'h01,
        NVME_CMD_READ           = 8'h02,
        NVME_CMD_WRITE_UNCOR    = 8'h04,
        NVME_CMD_COMPARE        = 8'h05,
        NVME_CMD_WRITE_ZEROS    = 8'h08,
        NVME_CMD_DSM            = 8'h09
    } nvme_opcode_t;
    
    // NVMe command structure (64 bytes)
    typedef struct packed {
        logic [7:0]  opcode;
        logic [7:0]  flags;
        logic [15:0] command_id;
        logic [31:0] nsid;          // Namespace ID
        logic [63:0] reserved1;
        logic [63:0] metadata_ptr;
        logic [63:0] prp1;          // Physical Region Page 1
        logic [63:0] prp2;          // Physical Region Page 2
        logic [31:0] cdw10;         // Command Dword 10
        logic [31:0] cdw11;         // Command Dword 11
        logic [31:0] cdw12;         // Command Dword 12
        logic [31:0] cdw13;         // Command Dword 13
        logic [31:0] cdw14;         // Command Dword 14
        logic [31:0] cdw15;         // Command Dword 15
    } nvme_command_t;
    
    // NVMe completion structure (16 bytes)
    typedef struct packed {
        logic [31:0] result;        // Command-specific result
        logic [31:0] reserved;
        logic [15:0] sq_head;       // Submission Queue Head
        logic [15:0] sq_id;         // Submission Queue ID
        logic [15:0] command_id;    // Command ID
        logic [15:0] status;        // Status field
    } nvme_completion_t;
    
    // NVMe Controller Registers
    typedef struct packed {
        logic [63:0] cap;           // Controller Capabilities
        logic [31:0] vs;            // Version
        logic [31:0] intms;         // Interrupt Mask Set
        logic [31:0] intmc;         // Interrupt Mask Clear
        logic [31:0] cc;            // Controller Configuration
        logic [31:0] reserved1;
        logic [31:0] csts;          // Controller Status
        logic [31:0] nssr;          // NVM Subsystem Reset
        logic [63:0] aqa;           // Admin Queue Attributes
        logic [63:0] asq;           // Admin Submission Queue Base
        logic [63:0] acq;           // Admin Completion Queue Base
    } nvme_controller_regs_t;
    
    // Queue structures
    typedef struct packed {
        logic [ADDR_WIDTH-1:0] base_addr;
        logic [15:0] size;
        logic [15:0] head;
        logic [15:0] tail;
        logic        valid;
        logic [15:0] cq_id;         // Associated completion queue
        logic [1:0]  priority;
    } submission_queue_t;
    
    typedef struct packed {
        logic [ADDR_WIDTH-1:0] base_addr;
        logic [15:0] size;
        logic [15:0] head;
        logic [15:0] tail;
        logic        valid;
        logic [15:0] irq_vector;
        logic        irq_enabled;
    } completion_queue_t;
    
    // Internal registers and state
    nvme_controller_regs_t ctrl_regs;
    submission_queue_t submission_queues [MAX_QUEUES-1:0];
    completion_queue_t completion_queues [MAX_QUEUES-1:0];
    
    // Command processing
    nvme_command_t current_command;
    nvme_completion_t current_completion;
    logic [15:0] current_sq_id;
    logic [15:0] current_cq_id;
    logic        command_valid;
    logic        completion_ready;
    
    // State machines
    typedef enum logic [3:0] {
        CTRL_RESET,
        CTRL_READY,
        CTRL_ENABLED,
        CTRL_SHUTDOWN_NORMAL,
        CTRL_SHUTDOWN_ABRUPT,
        CTRL_ERROR
    } controller_state_t;
    
    typedef enum logic [3:0] {
        CMD_IDLE,
        CMD_FETCH,
        CMD_DECODE,
        CMD_EXECUTE,
        CMD_DATA_TRANSFER,
        CMD_COMPLETE,
        CMD_ERROR
    } command_state_t;
    
    controller_state_t ctrl_state, ctrl_next_state;
    command_state_t cmd_state, cmd_next_state;
    
    // Namespace information
    logic [MAX_NAMESPACES-1:0] namespace_active;
    logic [63:0] namespace_size [MAX_NAMESPACES-1:0];  // Size in blocks
    logic [31:0] namespace_block_size [MAX_NAMESPACES-1:0];
    
    // Statistics and counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            commands_processed <= 32'h0;
            error_count <= 32'h0;
            data_transferred <= 64'h0;
        end else begin
            if (cmd_state == CMD_COMPLETE) begin
                commands_processed <= commands_processed + 1;
                
                // Update data transferred for read/write commands
                if (current_command.opcode == NVME_CMD_READ || 
                    current_command.opcode == NVME_CMD_WRITE) begin
                    data_transferred <= data_transferred + 
                                      (current_command.cdw12[15:0] + 1) * 512;  // Assume 512B blocks
                end
            end
            
            if (cmd_state == CMD_ERROR) begin
                error_count <= error_count + 1;
            end
        end
    end
    
    // Controller state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_state <= CTRL_RESET;
        end else begin
            ctrl_state <= ctrl_next_state;
        end
    end
    
    always_comb begin
        ctrl_next_state = ctrl_state;
        
        case (ctrl_state)
            CTRL_RESET: begin
                if (ctrl_regs.cc[0]) begin  // Enable bit set
                    ctrl_next_state = CTRL_READY;
                end
            end
            
            CTRL_READY: begin
                if (ctrl_regs.cc[0]) begin
                    ctrl_next_state = CTRL_ENABLED;
                end
            end
            
            CTRL_ENABLED: begin
                if (!ctrl_regs.cc[0]) begin
                    ctrl_next_state = CTRL_SHUTDOWN_NORMAL;
                end
            end
            
            CTRL_SHUTDOWN_NORMAL: begin
                ctrl_next_state = CTRL_RESET;
            end
            
            CTRL_SHUTDOWN_ABRUPT: begin
                ctrl_next_state = CTRL_RESET;
            end
            
            CTRL_ERROR: begin
                ctrl_next_state = CTRL_RESET;
            end
        endcase
    end
    
    // Command processing state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_state <= CMD_IDLE;
        end else begin
            cmd_state <= cmd_next_state;
        end
    end
    
    always_comb begin
        cmd_next_state = cmd_state;
        
        case (cmd_state)
            CMD_IDLE: begin
                if (ctrl_state == CTRL_ENABLED && command_valid) begin
                    cmd_next_state = CMD_FETCH;
                end
            end
            
            CMD_FETCH: begin
                if (mem_valid) begin
                    cmd_next_state = CMD_DECODE;
                end
            end
            
            CMD_DECODE: begin
                cmd_next_state = CMD_EXECUTE;
            end
            
            CMD_EXECUTE: begin
                if (current_command.opcode == NVME_CMD_READ || 
                    current_command.opcode == NVME_CMD_WRITE) begin
                    cmd_next_state = CMD_DATA_TRANSFER;
                end else begin
                    cmd_next_state = CMD_COMPLETE;
                end
            end
            
            CMD_DATA_TRANSFER: begin
                if (dma_done) begin
                    cmd_next_state = CMD_COMPLETE;
                end else if (dma_error) begin
                    cmd_next_state = CMD_ERROR;
                end
            end
            
            CMD_COMPLETE: begin
                if (completion_ready) begin
                    cmd_next_state = CMD_IDLE;
                end
            end
            
            CMD_ERROR: begin
                cmd_next_state = CMD_IDLE;
            end
        endcase
    end
    
    // Controller register initialization and management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_regs <= '0;
            
            // Initialize capabilities
            ctrl_regs.cap[15:0] <= QUEUE_DEPTH - 1;     // Maximum Queue Entries
            ctrl_regs.cap[16] <= 1'b1;                  // Contiguous Queues Required
            ctrl_regs.cap[17] <= 1'b1;                  // Arbitration Mechanism Supported
            ctrl_regs.cap[19:18] <= 2'b01;              // Memory Page Size Minimum (4KB)
            ctrl_regs.cap[23:20] <= 4'h0;               // Memory Page Size Maximum (4KB)
            ctrl_regs.cap[36:32] <= 5'h0C;              // Command Sets Supported (NVM)
            ctrl_regs.cap[37] <= 1'b1;                  // NVM Subsystem Reset Supported
            ctrl_regs.cap[44:43] <= 2'b00;              // Doorbell Stride (4 bytes)
            ctrl_regs.cap[52:48] <= 5'h01;              // NVMe Specification Version (1.4)
            
            // Version register
            ctrl_regs.vs <= 32'h00010400;               // NVMe 1.4.0
            
        end else begin
            // Update controller status
            case (ctrl_state)
                CTRL_RESET: begin
                    ctrl_regs.csts[0] <= 1'b0;           // Ready = 0
                    ctrl_regs.csts[1] <= 1'b0;           // Controller Fatal Status = 0
                    ctrl_regs.csts[2] <= 2'b00;         // Shutdown Status = Normal
                end
                
                CTRL_READY: begin
                    ctrl_regs.csts[0] <= 1'b1;           // Ready = 1
                end
                
                CTRL_ENABLED: begin
                    ctrl_regs.csts[0] <= 1'b1;           // Ready = 1
                end
                
                CTRL_ERROR: begin
                    ctrl_regs.csts[1] <= 1'b1;           // Controller Fatal Status = 1
                end
            endcase
            
            // Handle PCIe register writes
            if (pcie_rx_valid && pcie_rx_ready) begin
                // Decode register address from PCIe data
                // Simplified: assume lower 32 bits contain register data
                case (pcie_rx_data[63:32])  // Upper 32 bits as address
                    32'h14: ctrl_regs.cc <= pcie_rx_data[31:0];
                    32'h24: ctrl_regs.aqa <= pcie_rx_data;
                    32'h28: ctrl_regs.asq <= pcie_rx_data;
                    32'h30: ctrl_regs.acq <= pcie_rx_data;
                endcase
            end
        end
    end
    
    // Queue management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            submission_queues <= '{default: '0};
            completion_queues <= '{default: '0};
            active_queues <= 16'h0;
        end else begin
            // Initialize admin queues
            if (ctrl_state == CTRL_READY) begin
                // Admin Submission Queue (ID 0)
                submission_queues[0].base_addr <= ctrl_regs.asq;
                submission_queues[0].size <= ctrl_regs.aqa[15:0];
                submission_queues[0].valid <= 1'b1;
                submission_queues[0].cq_id <= 16'h0;
                
                // Admin Completion Queue (ID 0)
                completion_queues[0].base_addr <= ctrl_regs.acq;
                completion_queues[0].size <= ctrl_regs.aqa[31:16];
                completion_queues[0].valid <= 1'b1;
                completion_queues[0].irq_enabled <= 1'b1;
                
                active_queues <= 16'h1;  // Admin queue active
            end
            
            // Handle queue creation commands
            if (cmd_state == CMD_EXECUTE && current_command.opcode == NVME_ADMIN_CREATE_SQ) begin
                logic [15:0] sq_id = current_command.cdw10[15:0];
                if (sq_id < MAX_QUEUES) begin
                    submission_queues[sq_id].base_addr <= current_command.prp1;
                    submission_queues[sq_id].size <= current_command.cdw10[31:16];
                    submission_queues[sq_id].cq_id <= current_command.cdw11[15:0];
                    submission_queues[sq_id].valid <= 1'b1;
                    active_queues[sq_id] <= 1'b1;
                end
            end
            
            if (cmd_state == CMD_EXECUTE && current_command.opcode == NVME_ADMIN_CREATE_CQ) begin
                logic [15:0] cq_id = current_command.cdw10[15:0];
                if (cq_id < MAX_QUEUES) begin
                    completion_queues[cq_id].base_addr <= current_command.prp1;
                    completion_queues[cq_id].size <= current_command.cdw10[31:16];
                    completion_queues[cq_id].irq_vector <= current_command.cdw11[15:0];
                    completion_queues[cq_id].irq_enabled <= current_command.cdw11[16];
                    completion_queues[cq_id].valid <= 1'b1;
                end
            end
        end
    end
    
    // Command fetching and processing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_command <= '0;
            command_valid <= 1'b0;
            current_sq_id <= 16'h0;
        end else begin
            case (cmd_state)
                CMD_IDLE: begin
                    // Check for pending commands in submission queues
                    for (int i = 0; i < MAX_QUEUES; i++) begin
                        if (submission_queues[i].valid && 
                            submission_queues[i].tail != submission_queues[i].head) begin
                            current_sq_id <= i[15:0];
                            command_valid <= 1'b1;
                            break;
                        end
                    end
                end
                
                CMD_FETCH: begin
                    // Fetch command from submission queue
                    if (mem_valid) begin
                        current_command <= mem_rdata;  // Simplified: assume single read
                        command_valid <= 1'b0;
                    end
                end
                
                CMD_DECODE: begin
                    // Command is already fetched, prepare for execution
                    current_cq_id <= submission_queues[current_sq_id].cq_id;
                end
                
                CMD_COMPLETE: begin
                    // Update submission queue head
                    submission_queues[current_sq_id].head <= 
                        submission_queues[current_sq_id].head + 1;
                end
            endcase
        end
    end
    
    // Completion processing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_completion <= '0;
            completion_ready <= 1'b0;
        end else begin
            case (cmd_state)
                CMD_COMPLETE: begin
                    // Prepare completion entry
                    current_completion.command_id <= current_command.command_id;
                    current_completion.sq_id <= current_sq_id;
                    current_completion.sq_head <= submission_queues[current_sq_id].head;
                    current_completion.status <= 16'h0000;  // Success
                    completion_ready <= 1'b1;
                end
                
                CMD_ERROR: begin
                    current_completion.status <= 16'h0002;  // Generic command status error
                    completion_ready <= 1'b1;
                end
                
                default: begin
                    completion_ready <= 1'b0;
                end
            endcase
        end
    end
    
    // Memory interface for queue access
    assign mem_req = (cmd_state == CMD_FETCH) || (cmd_state == CMD_COMPLETE);
    assign mem_write = (cmd_state == CMD_COMPLETE);
    assign mem_size = 8'd64;  // Command/completion size
    
    always_comb begin
        if (cmd_state == CMD_FETCH) begin
            mem_addr = submission_queues[current_sq_id].base_addr + 
                      (submission_queues[current_sq_id].head * 64);
            mem_wdata = '0;
        end else if (cmd_state == CMD_COMPLETE) begin
            mem_addr = completion_queues[current_cq_id].base_addr + 
                      (completion_queues[current_cq_id].tail * 16);
            mem_wdata = current_completion;
        end else begin
            mem_addr = '0;
            mem_wdata = '0;
        end
    end
    
    // DMA interface for data transfers
    assign dma_req = (cmd_state == CMD_DATA_TRANSFER);
    assign dma_src_addr = (current_command.opcode == NVME_CMD_WRITE) ? 
                         current_command.prp1 : {32'h0, current_command.nsid, current_command.cdw10};
    assign dma_dst_addr = (current_command.opcode == NVME_CMD_READ) ? 
                         current_command.prp1 : {32'h0, current_command.nsid, current_command.cdw10};
    assign dma_length = (current_command.cdw12[15:0] + 1) * 512;  // Number of blocks * block size
    assign dma_write = (current_command.opcode == NVME_CMD_READ);
    
    // PCIe interface
    assign pcie_rx_ready = 1'b1;  // Always ready to receive
    assign pcie_tx_data = {32'h0, ctrl_regs.csts};  // Send status as example
    assign pcie_tx_valid = (ctrl_state == CTRL_ENABLED);
    
    // Interrupt generation
    assign msi_vector = completion_queues[current_cq_id].irq_vector;
    assign msi_valid = completion_ready && completion_queues[current_cq_id].irq_enabled;
    
    // Status outputs
    assign controller_ready = (ctrl_state == CTRL_ENABLED);
    
    // Initialize namespaces
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            namespace_active <= '0;
            namespace_size <= '{default: 64'h0};
            namespace_block_size <= '{default: 32'd512};
        end else if (ctrl_state == CTRL_READY) begin
            // Initialize namespace 1
            namespace_active[1] <= 1'b1;
            namespace_size[1] <= 64'h100000;      // 1M blocks
            namespace_block_size[1] <= 32'd512;   // 512 bytes per block
        end
    end
    
endmodule