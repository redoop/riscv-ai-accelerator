// RISC-V Vector Unit (VU)
// Implements RVV (RISC-V Vector) extension

`timescale 1ns/1ps

module riscv_vector_unit #(
    parameter XLEN = 64,
    parameter VLEN = 512,  // Vector register length in bits
    parameter ELEN = 64    // Maximum element width
) (
    input  logic                clk,
    input  logic                rst_n,
    
    // Control interface
    input  logic                vec_enable,
    input  logic [5:0]          vec_opcode,
    input  logic [2:0]          funct3,
    input  logic [5:0]          funct6,
    input  logic                vm,         // Vector mask bit
    
    // Instruction fields
    input  logic [4:0]          vs1,        // Vector source 1
    input  logic [4:0]          vs2,        // Vector source 2
    input  logic [4:0]          vd,         // Vector destination
    input  logic [4:0]          rs1,        // Scalar source 1
    input  logic [XLEN-1:0]     rs1_data,   // Scalar data
    input  logic [10:0]         imm,        // Immediate value
    
    // Vector configuration
    input  logic [XLEN-1:0]     vtype,      // Vector type register
    input  logic [XLEN-1:0]     vl,         // Vector length
    
    // Memory interface for vector loads/stores
    output logic [XLEN-1:0]     vec_mem_addr,
    output logic [VLEN-1:0]     vec_mem_wdata,
    output logic                vec_mem_req,
    output logic                vec_mem_we,
    output logic [VLEN/8-1:0]   vec_mem_be,
    input  logic [VLEN-1:0]     vec_mem_rdata,
    input  logic                vec_mem_ready,
    
    // Output
    output logic [XLEN-1:0]     vec_result,
    output logic                vec_ready,
    output logic                vec_valid
);

    // Vector instruction categories
    localparam [5:0] VEC_LOAD_UNIT   = 6'b000000;  // Unit-stride loads
    localparam [5:0] VEC_LOAD_STRIDE = 6'b000010;  // Strided loads
    localparam [5:0] VEC_LOAD_INDEX  = 6'b000011;  // Indexed loads
    localparam [5:0] VEC_STORE_UNIT  = 6'b000100;  // Unit-stride stores
    localparam [5:0] VEC_STORE_STRIDE= 6'b000110;  // Strided stores
    localparam [5:0] VEC_STORE_INDEX = 6'b000111;  // Indexed stores
    localparam [5:0] VEC_ARITH_VV    = 6'b010000;  // Vector-vector arithmetic
    localparam [5:0] VEC_ARITH_VX    = 6'b010001;  // Vector-scalar arithmetic
    localparam [5:0] VEC_ARITH_VI    = 6'b010010;  // Vector-immediate arithmetic
    localparam [5:0] VEC_CONFIG      = 6'b110000;  // Vector configuration

    // Vector register file (32 vector registers)
    logic [VLEN-1:0]    vreg_file [31:0];
    logic [VLEN-1:0]    vs1_data, vs2_data, vd_data;
    logic [VLEN-1:0]    vmask;              // Vector mask register (v0)
    
    // Vector configuration registers
    logic [2:0]         vsew;               // Selected element width
    logic [2:0]         vlmul;              // Vector length multiplier
    logic               vta;                // Tail agnostic
    logic               vma;                // Mask agnostic
    
    // Element configuration
    logic [7:0]         element_width;      // Element width in bits
    logic [15:0]        elements_per_reg;   // Number of elements per register
    logic [15:0]        active_elements;    // Number of active elements
    
    // Operation control
    logic [3:0]         operation_cycles;
    logic [3:0]         cycle_counter;
    logic               operation_active;
    logic               operation_complete;
    
    // Arithmetic operation results
    logic [VLEN-1:0]    arith_result;
    logic [VLEN-1:0]    load_store_result;

    // ========================================
    // Vector Configuration Decoding
    // ========================================
    
    always_comb begin
        vsew = vtype[5:3];      // SEW field
        vlmul = vtype[2:0];     // LMUL field
        vta = vtype[6];         // Tail agnostic
        vma = vtype[7];         // Mask agnostic
        
        // Decode element width
        case (vsew)
            3'b000: element_width = 8;   // SEW=8
            3'b001: element_width = 16;  // SEW=16
            3'b010: element_width = 32;  // SEW=32
            3'b011: element_width = 64;  // SEW=64
            default: element_width = 32;
        endcase
        
        // Calculate elements per register
        elements_per_reg = VLEN / element_width;
        
        // Active elements (limited by vl)
        active_elements = (vl > elements_per_reg) ? elements_per_reg : vl[15:0];
    end

    // ========================================
    // Vector Register File Access
    // ========================================
    
    assign vs1_data = vreg_file[vs1];
    assign vs2_data = vreg_file[vs2];
    assign vmask = vreg_file[0];  // v0 is the mask register
    
    // Vector register write
    always_ff @(posedge clk) begin
        if (vec_valid && (vd != 5'b0)) begin
            vreg_file[vd] <= vd_data;
        end
    end

    // ========================================
    // Operation Control
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_counter <= 4'b0;
            operation_active <= 1'b0;
        end else begin
            if (vec_enable && !operation_active) begin
                operation_active <= 1'b1;
                cycle_counter <= 4'b0;
            end else if (operation_active) begin
                cycle_counter <= cycle_counter + 1;
                if (operation_complete) begin
                    operation_active <= 1'b0;
                end
            end
        end
    end
    
    // Operation timing (simplified)
    always_comb begin
        case (vec_opcode)
            VEC_LOAD_UNIT, VEC_LOAD_STRIDE, VEC_LOAD_INDEX,
            VEC_STORE_UNIT, VEC_STORE_STRIDE, VEC_STORE_INDEX: begin
                operation_cycles = 4'd3;  // Memory operations take 3 cycles
            end
            VEC_ARITH_VV, VEC_ARITH_VX, VEC_ARITH_VI: begin
                operation_cycles = 4'd1;  // Arithmetic operations take 1 cycle
            end
            VEC_CONFIG: begin
                operation_cycles = 4'd0;  // Configuration is immediate
            end
            default: begin
                operation_cycles = 4'd1;
            end
        endcase
        
        operation_complete = (cycle_counter >= operation_cycles);
    end

    // ========================================
    // Vector Arithmetic Operations
    // ========================================
    
    genvar i;
    generate
        for (i = 0; i < VLEN/ELEN; i++) begin : gen_vector_lanes
            logic [ELEN-1:0] lane_vs1, lane_vs2, lane_result;
            logic [ELEN-1:0] scalar_operand;
            logic [ELEN-1:0] imm_operand;
            logic            lane_mask;
            
            // Extract lane data
            assign lane_vs1 = vs1_data[i*ELEN +: ELEN];
            assign lane_vs2 = vs2_data[i*ELEN +: ELEN];
            assign lane_mask = vm ? 1'b1 : vmask[i];  // vm=1 means unmasked
            
            // Scalar operand (replicated across all lanes)
            assign scalar_operand = rs1_data[ELEN-1:0];
            
            // Immediate operand (sign-extended)
            assign imm_operand = {{(ELEN-11){imm[10]}}, imm};
            
            // Vector arithmetic unit per lane
            always_comb begin
                lane_result = '0;
                
                if (lane_mask && (i < active_elements)) begin
                    case (vec_opcode)
                        VEC_ARITH_VV: begin
                            case (funct6)
                                6'b000000: lane_result = lane_vs1 + lane_vs2;     // VADD.VV
                                6'b000010: lane_result = lane_vs1 - lane_vs2;     // VSUB.VV
                                6'b000100: lane_result = lane_vs1 & lane_vs2;     // VAND.VV
                                6'b000101: lane_result = lane_vs1 | lane_vs2;     // VOR.VV
                                6'b000110: lane_result = lane_vs1 ^ lane_vs2;     // VXOR.VV
                                6'b001000: lane_result = (lane_vs1 < lane_vs2) ? lane_vs1 : lane_vs2; // VMIN.VV
                                6'b001001: lane_result = (lane_vs1 > lane_vs2) ? lane_vs1 : lane_vs2; // VMAX.VV
                                6'b100000: lane_result = lane_vs1 * lane_vs2;     // VMUL.VV
                                default: lane_result = lane_vs1;
                            endcase
                        end
                        
                        VEC_ARITH_VX: begin
                            case (funct6)
                                6'b000000: lane_result = lane_vs2 + scalar_operand; // VADD.VX
                                6'b000010: lane_result = lane_vs2 - scalar_operand; // VSUB.VX
                                6'b000100: lane_result = lane_vs2 & scalar_operand; // VAND.VX
                                6'b000101: lane_result = lane_vs2 | scalar_operand; // VOR.VX
                                6'b000110: lane_result = lane_vs2 ^ scalar_operand; // VXOR.VX
                                6'b001000: lane_result = (lane_vs2 < scalar_operand) ? lane_vs2 : scalar_operand; // VMIN.VX
                                6'b001001: lane_result = (lane_vs2 > scalar_operand) ? lane_vs2 : scalar_operand; // VMAX.VX
                                6'b100000: lane_result = lane_vs2 * scalar_operand; // VMUL.VX
                                default: lane_result = lane_vs2;
                            endcase
                        end
                        
                        VEC_ARITH_VI: begin
                            case (funct6)
                                6'b000000: lane_result = lane_vs2 + imm_operand;   // VADD.VI
                                6'b000100: lane_result = lane_vs2 & imm_operand;   // VAND.VI
                                6'b000101: lane_result = lane_vs2 | imm_operand;   // VOR.VI
                                6'b000110: lane_result = lane_vs2 ^ imm_operand;   // VXOR.VI
                                default: lane_result = lane_vs2;
                            endcase
                        end
                        
                        default: begin
                            lane_result = lane_vs2;
                        end
                    endcase
                end else begin
                    // Masked out or beyond active elements
                    if (vta) begin
                        lane_result = '1;  // Tail agnostic - set to all 1s
                    end else begin
                        lane_result = lane_vs2;  // Keep old value
                    end
                end
            end
            
            // Assign lane result to output
            assign arith_result[i*ELEN +: ELEN] = lane_result;
        end
    endgenerate

    // ========================================
    // Vector Memory Operations
    // ========================================
    
    logic [XLEN-1:0]    base_addr;
    logic [XLEN-1:0]    stride_value;
    logic [15:0]        mem_element_count;
    
    assign base_addr = rs1_data;
    assign stride_value = vs2_data[XLEN-1:0];  // For strided operations
    assign mem_element_count = active_elements;
    
    always_comb begin
        vec_mem_addr = '0;
        vec_mem_wdata = '0;
        vec_mem_req = 1'b0;
        vec_mem_we = 1'b0;
        vec_mem_be = '0;
        load_store_result = '0;
        
        if (operation_active) begin
            case (vec_opcode)
                VEC_LOAD_UNIT: begin
                    // Unit-stride load
                    vec_mem_addr = base_addr + (cycle_counter * (VLEN/8));
                    vec_mem_req = 1'b1;
                    vec_mem_we = 1'b0;
                    vec_mem_be = '1;  // Load all bytes
                    load_store_result = vec_mem_rdata;
                end
                
                VEC_STORE_UNIT: begin
                    // Unit-stride store
                    vec_mem_addr = base_addr + (cycle_counter * (VLEN/8));
                    vec_mem_wdata = vs2_data;
                    vec_mem_req = 1'b1;
                    vec_mem_we = 1'b1;
                    vec_mem_be = '1;  // Store all bytes
                end
                
                VEC_LOAD_STRIDE: begin
                    // Strided load (simplified)
                    vec_mem_addr = base_addr + (cycle_counter * stride_value);
                    vec_mem_req = 1'b1;
                    vec_mem_we = 1'b0;
                    vec_mem_be = '1;
                    load_store_result = vec_mem_rdata;
                end
                
                VEC_STORE_STRIDE: begin
                    // Strided store (simplified)
                    vec_mem_addr = base_addr + (cycle_counter * stride_value);
                    vec_mem_wdata = vs2_data;
                    vec_mem_req = 1'b1;
                    vec_mem_we = 1'b1;
                    vec_mem_be = '1;
                end
                
                default: begin
                    // No memory operation
                end
            endcase
        end
    end

    // ========================================
    // Result Selection and Output
    // ========================================
    
    always_comb begin
        case (vec_opcode)
            VEC_ARITH_VV, VEC_ARITH_VX, VEC_ARITH_VI: begin
                vd_data = arith_result;
                vec_result = '0;  // No scalar result for vector arithmetic
            end
            
            VEC_LOAD_UNIT, VEC_LOAD_STRIDE, VEC_LOAD_INDEX: begin
                vd_data = load_store_result;
                vec_result = '0;
            end
            
            VEC_STORE_UNIT, VEC_STORE_STRIDE, VEC_STORE_INDEX: begin
                vd_data = '0;  // No vector result for stores
                vec_result = '0;
            end
            
            VEC_CONFIG: begin
                vd_data = '0;
                vec_result = vl;  // Return vector length for vsetvl instructions
            end
            
            default: begin
                vd_data = '0;
                vec_result = '0;
            end
        endcase
    end
    
    assign vec_ready = operation_complete;
    assign vec_valid = operation_active && operation_complete;

endmodule