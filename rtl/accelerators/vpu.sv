// Vector Processing Unit (VPU) Module
// Specialized unit for vector operations and SIMD computations
// Implements configurable vector register file and functional units

module vpu #(
    parameter VECTOR_LANES = 16,
    parameter VECTOR_REGS = 32,
    parameter MAX_VLEN = 512,
    parameter ELEMENT_WIDTH = 64
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Control interface (simplified)
    input  logic [31:0] ctrl_addr,
    input  logic [63:0] ctrl_wdata,
    output logic [63:0] ctrl_rdata,
    input  logic        ctrl_req,
    input  logic        ctrl_we,
    output logic        ctrl_ready,
    
    // Status and error reporting
    output logic [7:0]  status,
    output logic        busy,
    output logic        error
);

    // ========================================
    // VPU Operation Types
    // ========================================
    
    // Local parameters for synthesis compatibility
    localparam STATUS_OK = 8'h00;
    
    // Data type enumeration (local copy for synthesis)
    typedef enum logic [2:0] {
        DATA_TYPE_INT8   = 3'b000,
        DATA_TYPE_INT16  = 3'b001,
        DATA_TYPE_INT32  = 3'b010,
        DATA_TYPE_FP16   = 3'b011,
        DATA_TYPE_FP32   = 3'b100,
        DATA_TYPE_FP64   = 3'b101
    } data_type_e;
    
    typedef enum logic [3:0] {
        VPU_OP_ADD      = 4'b0000,
        VPU_OP_SUB      = 4'b0001,
        VPU_OP_MUL      = 4'b0010,
        VPU_OP_DIV      = 4'b0011,
        VPU_OP_AND      = 4'b0100,
        VPU_OP_OR       = 4'b0101,
        VPU_OP_XOR      = 4'b0110,
        VPU_OP_MIN      = 4'b0111,
        VPU_OP_MAX      = 4'b1000,
        VPU_OP_CONVERT  = 4'b1001,
        VPU_OP_LOAD     = 4'b1010,
        VPU_OP_STORE    = 4'b1011
    } vpu_operation_e;

    // ========================================
    // Vector Register File
    // ========================================
    
    // Enhanced configurable vector register file
    logic [MAX_VLEN-1:0]    vreg_file [VECTOR_REGS-1:0];
    logic [4:0]             vrs1, vrs2, vrd;
    logic [MAX_VLEN-1:0]    vrs1_data, vrs2_data, vrd_data;
    logic                   vreg_we;
    logic [VECTOR_REGS-1:0] vreg_write_mask;    // Write enable per register
    logic [VECTOR_REGS-1:0] vreg_read_mask;     // Read enable per register
    
    // Vector configuration registers
    logic [2:0]             vsew;           // Selected element width (SEW)
    logic [15:0]            vl;             // Vector length
    logic [15:0]            vtype;          // Vector type register
    logic [15:0]            elements_per_reg;
    logic [15:0]            active_elements;
    logic [15:0]            vlmax;          // Maximum vector length
    data_type_e             src_dtype, dst_dtype;
    
    // Vector mask register (v0)
    logic [MAX_VLEN-1:0]    vmask;
    logic                   mask_enabled;
    
    // Enhanced element width decoding and configuration
    always_comb begin
        case (vsew)
            3'b000: begin // SEW = 8
                elements_per_reg = MAX_VLEN / 8;   // 64 elements
                vlmax = MAX_VLEN / 8;
            end
            3'b001: begin // SEW = 16
                elements_per_reg = MAX_VLEN / 16;  // 32 elements
                vlmax = MAX_VLEN / 16;
            end
            3'b010: begin // SEW = 32
                elements_per_reg = MAX_VLEN / 32;  // 16 elements
                vlmax = MAX_VLEN / 32;
            end
            3'b011: begin // SEW = 64
                elements_per_reg = MAX_VLEN / 64;  // 8 elements
                vlmax = MAX_VLEN / 64;
            end
            default: begin
                elements_per_reg = MAX_VLEN / 32;
                vlmax = MAX_VLEN / 32;
            end
        endcase
        
        // Active elements limited by both vl and physical register capacity
        active_elements = (vl > elements_per_reg) ? elements_per_reg : vl;
        
        // Vector mask handling
        vmask = vreg_file[0]; // v0 is the mask register
        mask_enabled = vtype[0]; // VM bit in vtype
    end
    
    // Enhanced register file access with read/write tracking
    assign vrs1_data = vreg_file[vrs1];
    assign vrs2_data = vreg_file[vrs2];
    
    // Generate read masks
    always_comb begin
        vreg_read_mask = '0;
        if (vrs1 != 5'b0) vreg_read_mask[vrs1] = 1'b1;
        if (vrs2 != 5'b0) vreg_read_mask[vrs2] = 1'b1;
    end
    
    // Generate write mask
    always_comb begin
        vreg_write_mask = '0;
        if (vreg_we && (vrd != 5'b0)) begin
            vreg_write_mask[vrd] = 1'b1;
        end
    end
    
    // Enhanced register file write with initialization
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize all vector registers to zero
            for (int i = 0; i < VECTOR_REGS; i++) begin
                vreg_file[i] <= '0;
            end
        end else begin
            if (vreg_we && (vrd != 5'b0)) begin
                // Apply mask if enabled
                if (mask_enabled) begin
                    // Masked write: only update elements where mask bit is 1
                    for (int elem = 0; elem < MAX_VLEN/8; elem++) begin
                        if (vmask[elem] && (elem < active_elements)) begin
                            case (vsew)
                                3'b000: if ((elem*8 + 7) < MAX_VLEN) vreg_file[vrd][elem*8 +: 8] <= vrd_data[elem*8 +: 8];
                                3'b001: if ((elem*16 + 15) < MAX_VLEN) vreg_file[vrd][elem*16 +: 16] <= vrd_data[elem*16 +: 16];
                                3'b010: if ((elem*32 + 31) < MAX_VLEN) vreg_file[vrd][elem*32 +: 32] <= vrd_data[elem*32 +: 32];
                                3'b011: if ((elem*64 + 63) < MAX_VLEN) vreg_file[vrd][elem*64 +: 64] <= vrd_data[elem*64 +: 64];
                            endcase
                        end
                    end
                end else begin
                    // Unmasked write: update entire register
                    vreg_file[vrd] <= vrd_data;
                end
            end
        end
    end

    // ========================================
    // Enhanced Vector Functional Units
    // ========================================
    
    // Vector lane results and status
    logic [ELEMENT_WIDTH-1:0] lane_results [VECTOR_LANES-1:0];
    logic [VECTOR_LANES-1:0]  lane_valid_out;
    logic [VECTOR_LANES-1:0]  lane_overflow;
    logic [VECTOR_LANES-1:0]  lane_underflow;
    logic [VECTOR_LANES-1:0]  lane_error;
    
    // Enhanced arithmetic units per lane
    genvar lane;
    generate
        for (lane = 0; lane < VECTOR_LANES; lane++) begin : gen_vector_lanes
            
            // Lane data extraction and control
            logic [ELEMENT_WIDTH-1:0] lane_a, lane_b;
            logic                     lane_valid;
            logic                     lane_mask_bit;
            
            // Enhanced lane data extraction with proper element mapping
            always_comb begin
                lane_a = '0;
                lane_b = '0;
                lane_valid = 1'b0;
                lane_mask_bit = 1'b1; // Default to enabled
                
                case (vsew)
                    3'b000: begin // 8-bit elements
                        if (lane < MAX_VLEN/8 && lane < active_elements) begin
                            lane_a = {{(ELEMENT_WIDTH-8){vrs1_data[lane*8+7]}}, vrs1_data[lane*8 +: 8]};
                            lane_b = {{(ELEMENT_WIDTH-8){vrs2_data[lane*8+7]}}, vrs2_data[lane*8 +: 8]};
                            lane_valid = 1'b1;
                            if (mask_enabled) lane_mask_bit = vmask[lane];
                        end
                    end
                    3'b001: begin // 16-bit elements
                        if (lane < MAX_VLEN/16 && lane < active_elements && (lane*16 + 15) < MAX_VLEN) begin
                            lane_a = {{(ELEMENT_WIDTH-16){vrs1_data[lane*16+15]}}, vrs1_data[lane*16 +: 16]};
                            lane_b = {{(ELEMENT_WIDTH-16){vrs2_data[lane*16+15]}}, vrs2_data[lane*16 +: 16]};
                            lane_valid = 1'b1;
                            if (mask_enabled) lane_mask_bit = vmask[lane];
                        end
                    end
                    3'b010: begin // 32-bit elements
                        if (lane < MAX_VLEN/32 && lane < active_elements && (lane*32 + 31) < MAX_VLEN) begin
                            lane_a = {{(ELEMENT_WIDTH-32){vrs1_data[lane*32+31]}}, vrs1_data[lane*32 +: 32]};
                            lane_b = {{(ELEMENT_WIDTH-32){vrs2_data[lane*32+31]}}, vrs2_data[lane*32 +: 32]};
                            lane_valid = 1'b1;
                            if (mask_enabled) lane_mask_bit = vmask[lane];
                        end
                    end
                    3'b011: begin // 64-bit elements
                        if (lane < MAX_VLEN/64 && lane < active_elements && (lane*64 + 63) < MAX_VLEN) begin
                            lane_a = vrs1_data[lane*64 +: 64];
                            lane_b = vrs2_data[lane*64 +: 64];
                            lane_valid = 1'b1;
                            if (mask_enabled) lane_mask_bit = vmask[lane];
                        end
                    end
                endcase
                
                // Apply mask to validity
                lane_valid = lane_valid && lane_mask_bit;
            end
            
            // Enhanced vector arithmetic unit instance
            vector_alu #(
                .ELEMENT_WIDTH(ELEMENT_WIDTH)
            ) u_vector_alu (
                .clk(clk),
                .rst_n(rst_n),
                .operation(operation),
                .src_dtype(src_dtype),
                .dst_dtype(dst_dtype),
                .operand_a(lane_a),
                .operand_b(lane_b),
                .valid_in(lane_valid),
                .result(lane_results[lane]),
                .valid_out(lane_valid_out[lane]),
                .overflow(lane_overflow[lane]),
                .underflow(lane_underflow[lane])
            );
            
            // Error detection per lane
            assign lane_error[lane] = lane_overflow[lane] || lane_underflow[lane];
        end
    endgenerate
    
    // Result aggregation and packing
    always_comb begin
        vrd_data = '0; // Initialize to zero
        
        for (int lane_idx = 0; lane_idx < VECTOR_LANES; lane_idx++) begin
            if (lane_valid_out[lane_idx]) begin
                case (vsew)
                    3'b000: begin // 8-bit elements
                        if (lane_idx < MAX_VLEN/8) begin
                            vrd_data[lane_idx*8 +: 8] = lane_results[lane_idx][7:0];
                        end
                    end
                    3'b001: begin // 16-bit elements
                        if (lane_idx < MAX_VLEN/16) begin
                            vrd_data[lane_idx*16 +: 16] = lane_results[lane_idx][15:0];
                        end
                    end
                    3'b010: begin // 32-bit elements
                        if (lane_idx < MAX_VLEN/32) begin
                            vrd_data[lane_idx*32 +: 32] = lane_results[lane_idx][31:0];
                        end
                    end
                    3'b011: begin // 64-bit elements
                        if (lane_idx < MAX_VLEN/64) begin
                            vrd_data[lane_idx*64 +: 64] = lane_results[lane_idx];
                        end
                    end
                endcase
            end
        end
    end
    
    // Vector operation status aggregation
    logic vector_overflow, vector_underflow, vector_error;
    assign vector_overflow = |lane_overflow;
    assign vector_underflow = |lane_underflow;
    assign vector_error = |lane_error;

    // ========================================
    // Control Logic
    // ========================================
    
    vpu_operation_e         operation;
    logic                   operation_start;
    logic                   operation_done;
    logic [3:0]             operation_cycles;
    logic [3:0]             cycle_counter;
    
    // Control interface decoding
    always_comb begin
        vrs1 = ctrl_addr[9:5];
        vrs2 = ctrl_addr[14:10];
        vrd = ctrl_addr[19:15];
        operation = vpu_operation_e'(ctrl_addr[23:20]);
        vsew = ctrl_addr[26:24];
        src_dtype = data_type_e'(ctrl_wdata[2:0]);
        dst_dtype = data_type_e'(ctrl_wdata[5:3]);
        vl = ctrl_wdata[21:6];
        
        operation_start = ctrl_req && ctrl_we;
    end
    
    // Operation timing
    always_comb begin
        case (operation)
            VPU_OP_ADD, VPU_OP_SUB, VPU_OP_AND, VPU_OP_OR, VPU_OP_XOR, VPU_OP_MIN, VPU_OP_MAX: begin
                operation_cycles = 4'd1;  // Single cycle operations
            end
            VPU_OP_MUL: begin
                operation_cycles = 4'd2;  // Two cycle multiply
            end
            VPU_OP_DIV: begin
                operation_cycles = 4'd8;  // Eight cycle divide
            end
            VPU_OP_CONVERT: begin
                operation_cycles = 4'd2;  // Two cycle conversion
            end
            VPU_OP_LOAD, VPU_OP_STORE: begin
                operation_cycles = 4'd4;  // Four cycle memory operations
            end
            default: begin
                operation_cycles = 4'd1;
            end
        endcase
    end
    
    // Operation state machine
    typedef enum logic [1:0] {
        IDLE,
        EXECUTING,
        COMPLETING
    } vpu_state_e;
    
    vpu_state_e current_state, next_state;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            cycle_counter <= 4'b0;
        end else begin
            current_state <= next_state;
            if (current_state == EXECUTING) begin
                cycle_counter <= cycle_counter + 1;
            end else begin
                cycle_counter <= 4'b0;
            end
        end
    end
    
    always_comb begin
        next_state = current_state;
        operation_done = 1'b0;
        vreg_we = 1'b0;
        
        case (current_state)
            IDLE: begin
                if (operation_start) begin
                    next_state = EXECUTING;
                end
            end
            
            EXECUTING: begin
                if (cycle_counter >= (operation_cycles - 1)) begin
                    next_state = COMPLETING;
                end
            end
            
            COMPLETING: begin
                operation_done = 1'b1;
                vreg_we = 1'b1;
                next_state = IDLE;
            end
        endcase
    end

    // ========================================
    // Memory Interface (Simplified)
    // ========================================
    
    // Simplified memory interface - not implemented in this version

    // ========================================
    // Status and Error Reporting
    // ========================================
    
    always_comb begin
        busy = (current_state != IDLE);
        error = 1'b0; // TODO: Add error detection logic
        
        if (operation_done) begin
            status = STATUS_OK;
        end else if (busy) begin
            status = 8'h01; // Busy status
        end else begin
            status = STATUS_OK;
        end
    end
    
    // Control interface response
    assign ctrl_rdata = {48'b0, vl};
    assign ctrl_ready = (current_state == IDLE) || operation_done;

endmodule