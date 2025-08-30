// Vector Processing Unit Instruction Pipeline
// Implements vector instruction decode, dispatch, and execution pipeline

`include "chip_config.sv"

/* verilator lint_off SELRANGE */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off CASEINCOMPLETE */

module vpu_instruction_pipeline #(
    parameter VECTOR_LANES = 16,
    parameter MAX_VLEN = 512,
    parameter ELEMENT_WIDTH = 64
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Instruction interface
    input  logic [31:0] instruction,
    input  logic        instruction_valid,
    output logic        instruction_ready,
    
    // Vector register file interface
    output logic [4:0]  vrs1_addr, vrs2_addr, vrd_addr,
    input  logic [MAX_VLEN-1:0] vrs1_data, vrs2_data,
    output logic [MAX_VLEN-1:0] vrd_data,
    output logic        vrd_we,
    
    // Vector configuration
    input  logic [2:0]  vsew,      // Selected element width
    input  logic [15:0] vl,        // Vector length
    input  logic [MAX_VLEN-1:0] vmask, // Vector mask register (v0)
    
    // Memory interface for vector loads/stores
    output logic [63:0] mem_addr,
    output logic [MAX_VLEN-1:0] mem_wdata,
    output logic        mem_req,
    output logic        mem_we,
    output logic [MAX_VLEN/8-1:0] mem_be,
    input  logic [MAX_VLEN-1:0] mem_rdata,
    input  logic        mem_ready,
    
    // Status and control
    output logic        pipeline_busy,
    output logic        pipeline_done,
    output logic        pipeline_error
);

    import chip_config_pkg::*;

    // ========================================
    // Instruction Decode
    // ========================================
    
    // RVV instruction format fields
    logic [6:0]  opcode;
    logic [4:0]  vd, vs1, vs2, rs1;
    logic [2:0]  funct3;
    logic [5:0]  funct6;
    logic        vm;        // Vector mask bit
    logic [10:0] imm;       // Immediate value
    
    // Decoded instruction type
    typedef enum logic [3:0] {
        VEC_ARITH_VV    = 4'b0000,  // Vector-vector arithmetic
        VEC_ARITH_VX    = 4'b0001,  // Vector-scalar arithmetic
        VEC_ARITH_VI    = 4'b0010,  // Vector-immediate arithmetic
        VEC_LOAD_UNIT   = 4'b0011,  // Unit-stride load
        VEC_LOAD_STRIDE = 4'b0100,  // Strided load
        VEC_LOAD_INDEX  = 4'b0101,  // Indexed load
        VEC_STORE_UNIT  = 4'b0110,  // Unit-stride store
        VEC_STORE_STRIDE= 4'b0111,  // Strided store
        VEC_STORE_INDEX = 4'b1000,  // Indexed store
        VEC_MASK_OP     = 4'b1001,  // Vector mask operations
        VEC_PERM_OP     = 4'b1010,  // Vector permutation
        VEC_CONFIG      = 4'b1011,  // Vector configuration
        VEC_INVALID     = 4'b1111   // Invalid instruction
    } vec_instr_type_e;
    
    vec_instr_type_e decoded_instr_type;
    
    // Instruction field extraction
    always_comb begin
        opcode = instruction[6:0];
        vd = instruction[11:7];
        funct3 = instruction[14:12];
        vs1 = instruction[19:15];
        vs2 = instruction[24:20];
        vm = instruction[25];
        funct6 = instruction[31:26];
        rs1 = instruction[19:15];  // Same as vs1 for scalar operations
        imm = instruction[30:20];  // Immediate field
    end
    
    // Enhanced instruction type decoding with better dispatch logic
    always_comb begin
        decoded_instr_type = VEC_INVALID;
        
        if (opcode == 7'b1010111) begin // Vector opcode
            case (funct3)
                3'b000: begin // OPIVV - vector-vector integer
                    decoded_instr_type = VEC_ARITH_VV;
                end
                3'b001: begin // OPFVV - vector-vector floating-point
                    decoded_instr_type = VEC_ARITH_VV;
                end
                3'b010: begin // OPMVV - vector-vector mask
                    decoded_instr_type = VEC_MASK_OP;
                end
                3'b011: begin // OPIVI - vector-immediate integer
                    decoded_instr_type = VEC_ARITH_VI;
                end
                3'b100: begin // OPIVX - vector-scalar integer
                    decoded_instr_type = VEC_ARITH_VX;
                end
                3'b101: begin // OPFVF - vector-scalar floating-point
                    decoded_instr_type = VEC_ARITH_VX;
                end
                3'b110: begin // OPMVX - vector-scalar mask
                    decoded_instr_type = VEC_MASK_OP;
                end
                3'b111: begin // OPCFG - vector configuration
                    decoded_instr_type = VEC_CONFIG;
                end
                default: decoded_instr_type = VEC_INVALID;
            endcase
        end else if (opcode == 7'b0000111) begin // Vector load opcode
            case (funct3[2:0])
                3'b000: decoded_instr_type = VEC_LOAD_UNIT;   // Unit-stride
                3'b010: decoded_instr_type = VEC_LOAD_STRIDE; // Strided
                3'b011: decoded_instr_type = VEC_LOAD_INDEX;  // Indexed (gather)
                3'b110: decoded_instr_type = VEC_LOAD_UNIT;   // Whole register load
                default: decoded_instr_type = VEC_INVALID;
            endcase
        end else if (opcode == 7'b0100111) begin // Vector store opcode
            case (funct3[2:0])
                3'b000: decoded_instr_type = VEC_STORE_UNIT;   // Unit-stride
                3'b010: decoded_instr_type = VEC_STORE_STRIDE; // Strided
                3'b011: decoded_instr_type = VEC_STORE_INDEX;  // Indexed (scatter)
                3'b110: decoded_instr_type = VEC_STORE_UNIT;   // Whole register store
                default: decoded_instr_type = VEC_INVALID;
            endcase
        end
    end
    
    // Dispatch control signals
    logic dispatch_ready;
    logic dispatch_valid;
    logic [3:0] dispatch_unit_id;
    
    // Instruction dispatch logic
    always_comb begin
        dispatch_ready = 1'b1; // Assume ready unless busy
        dispatch_valid = instruction_valid && (decoded_instr_type != VEC_INVALID);
        dispatch_unit_id = 4'b0;
        
        // Determine which execution unit to dispatch to
        case (decoded_instr_type)
            VEC_ARITH_VV, VEC_ARITH_VX, VEC_ARITH_VI: begin
                dispatch_unit_id = 4'b0001; // Arithmetic unit
            end
            VEC_LOAD_UNIT, VEC_LOAD_STRIDE, VEC_LOAD_INDEX,
            VEC_STORE_UNIT, VEC_STORE_STRIDE, VEC_STORE_INDEX: begin
                dispatch_unit_id = 4'b0010; // Memory unit
            end
            VEC_MASK_OP: begin
                dispatch_unit_id = 4'b0100; // Mask unit
            end
            VEC_PERM_OP: begin
                dispatch_unit_id = 4'b1000; // Permutation unit
            end
            VEC_CONFIG: begin
                dispatch_unit_id = 4'b0000; // Configuration (immediate)
            end
            default: begin
                dispatch_unit_id = 4'b0000;
                dispatch_ready = 1'b0;
            end
        endcase
    end

    // ========================================
    // Pipeline Stages
    // ========================================
    
    typedef enum logic [2:0] {
        PIPE_IDLE,
        PIPE_DECODE,
        PIPE_EXECUTE,
        PIPE_MEMORY,
        PIPE_WRITEBACK
    } pipeline_state_e;
    
    pipeline_state_e current_state, next_state;
    
    // Pipeline registers
    logic [31:0] instr_reg;
    vec_instr_type_e instr_type_reg;
    logic [4:0] vd_reg, vs1_reg, vs2_reg;
    logic [2:0] funct3_reg;
    logic [5:0] funct6_reg;
    logic vm_reg;
    logic [10:0] imm_reg;
    
    // Execution control
    logic [3:0] execution_cycles;
    logic [3:0] cycle_counter;
    logic execution_complete;
    
    // Vector mask processing
    logic [MAX_VLEN-1:0] effective_mask;
    logic [15:0] active_elements;
    
    // Calculate active elements based on vector length and element width
    always_comb begin
        case (vsew)
            3'b000: active_elements = (vl > MAX_VLEN/8)  ? MAX_VLEN/8  : vl;  // 8-bit
            3'b001: active_elements = (vl > MAX_VLEN/16) ? MAX_VLEN/16 : vl;  // 16-bit
            3'b010: active_elements = (vl > MAX_VLEN/32) ? MAX_VLEN/32 : vl;  // 32-bit
            3'b011: active_elements = (vl > MAX_VLEN/64) ? MAX_VLEN/64 : vl;  // 64-bit
            default: active_elements = vl;
        endcase
    end
    
    // Enhanced mask processing with tail and prestart handling
    logic [MAX_VLEN-1:0] tail_mask;
    logic [MAX_VLEN-1:0] prestart_mask;
    logic [MAX_VLEN-1:0] body_mask;
    
    // Generate tail mask (elements beyond vl are inactive)
    always_comb begin
        tail_mask = '0;
        for (int i = 0; i < MAX_VLEN; i++) begin
            if (i < active_elements) begin
                tail_mask[i] = 1'b1;
            end
        end
    end
    
    // Generate prestart mask (for fractional LMUL < 1)
    always_comb begin
        prestart_mask = '1; // Simplified - assume no prestart elements for now
    end
    
    // Generate body mask (active elements within vector length)
    always_comb begin
        body_mask = tail_mask & prestart_mask;
    end
    
    // Generate effective mask (considering vm bit and body mask)
    always_comb begin
        if (vm_reg) begin
            // Unmasked operation - use body mask only
            effective_mask = body_mask;
        end else begin
            // Masked operation - combine v0 mask with body mask
            effective_mask = vmask & body_mask;
        end
    end
    
    // Mask policy handling (agnostic vs undisturbed)
    typedef enum logic [1:0] {
        MASK_POLICY_UNDISTURBED = 2'b00,
        MASK_POLICY_AGNOSTIC    = 2'b01
    } mask_policy_e;
    
    mask_policy_e mask_policy;
    mask_policy_e tail_policy;
    
    // Extract policies from vtype register (simplified)
    always_comb begin
        mask_policy = mask_policy_e'(1'b0); // Undisturbed for now
        tail_policy = mask_policy_e'(1'b0); // Undisturbed for now
    end

    // ========================================
    // Pipeline State Machine
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= PIPE_IDLE;
            cycle_counter <= 4'b0;
            instr_reg <= 32'b0;
            instr_type_reg <= VEC_INVALID;
            vd_reg <= 5'b0;
            vs1_reg <= 5'b0;
            vs2_reg <= 5'b0;
            funct3_reg <= 3'b0;
            funct6_reg <= 6'b0;
            vm_reg <= 1'b0;
            imm_reg <= 11'b0;
        end else begin
            current_state <= next_state;
            
            if (current_state == PIPE_DECODE) begin
                // Latch instruction fields
                instr_reg <= instruction;
                instr_type_reg <= decoded_instr_type;
                vd_reg <= vd;
                vs1_reg <= vs1;
                vs2_reg <= vs2;
                funct3_reg <= funct3;
                funct6_reg <= funct6;
                vm_reg <= vm;
                imm_reg <= imm;
                cycle_counter <= 4'b0;
            end else if (current_state == PIPE_EXECUTE || current_state == PIPE_MEMORY) begin
                cycle_counter <= cycle_counter + 1;
            end else begin
                cycle_counter <= 4'b0;
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = current_state;
        
        case (current_state)
            PIPE_IDLE: begin
                if (instruction_valid) begin
                    next_state = PIPE_DECODE;
                end
            end
            
            PIPE_DECODE: begin
                if (decoded_instr_type != VEC_INVALID) begin
                    if (decoded_instr_type == VEC_LOAD_UNIT || 
                        decoded_instr_type == VEC_LOAD_STRIDE || 
                        decoded_instr_type == VEC_LOAD_INDEX ||
                        decoded_instr_type == VEC_STORE_UNIT || 
                        decoded_instr_type == VEC_STORE_STRIDE || 
                        decoded_instr_type == VEC_STORE_INDEX) begin
                        next_state = PIPE_MEMORY;
                    end else begin
                        next_state = PIPE_EXECUTE;
                    end
                end else begin
                    next_state = PIPE_IDLE; // Invalid instruction
                end
            end
            
            PIPE_EXECUTE: begin
                if (execution_complete) begin
                    next_state = PIPE_WRITEBACK;
                end
            end
            
            PIPE_MEMORY: begin
                if (execution_complete && mem_ready && memory_transaction_complete) begin
                    next_state = PIPE_WRITEBACK;
                end
            end
            
            PIPE_WRITEBACK: begin
                next_state = PIPE_IDLE;
            end
        endcase
    end
    
    // Execution timing
    always_comb begin
        case (instr_type_reg)
            VEC_ARITH_VV, VEC_ARITH_VX, VEC_ARITH_VI: begin
                case (funct6_reg)
                    6'b100000, 6'b100001: execution_cycles = 4'd3; // Multiply/divide
                    default: execution_cycles = 4'd1; // Add/sub/logical
                endcase
            end
            VEC_LOAD_UNIT, VEC_LOAD_STRIDE, VEC_LOAD_INDEX,
            VEC_STORE_UNIT, VEC_STORE_STRIDE, VEC_STORE_INDEX: begin
                execution_cycles = 4'd2; // Memory operations
            end
            VEC_MASK_OP, VEC_PERM_OP: begin
                execution_cycles = 4'd2; // Mask/permutation operations
            end
            VEC_CONFIG: begin
                execution_cycles = 4'd0; // Configuration is immediate
            end
            default: begin
                execution_cycles = 4'd1;
            end
        endcase
        
        execution_complete = (cycle_counter >= execution_cycles);
    end

    // ========================================
    // Vector Arithmetic Execution
    // ========================================
    
    logic [MAX_VLEN-1:0] arithmetic_result;
    logic [ELEMENT_WIDTH-1:0] lane_results [VECTOR_LANES-1:0];
    
    genvar lane;
    generate
        for (lane = 0; lane < VECTOR_LANES; lane++) begin : gen_arithmetic_lanes
            logic [ELEMENT_WIDTH-1:0] lane_vs1, lane_vs2, lane_result;
            logic [ELEMENT_WIDTH-1:0] scalar_operand, imm_operand;
            logic lane_mask;
            
            // Extract lane data based on element width
            always_comb begin
                case (vsew)
                    3'b000: begin // 8-bit elements
                        if (lane < MAX_VLEN/8) begin
                            lane_vs1 = {{(ELEMENT_WIDTH-8){1'b0}}, vrs1_data[lane*8 +: 8]};
                            lane_vs2 = {{(ELEMENT_WIDTH-8){1'b0}}, vrs2_data[lane*8 +: 8]};
                        end else begin
                            lane_vs1 = '0;
                            lane_vs2 = '0;
                        end
                    end
                    3'b001: begin // 16-bit elements
                        if (lane < MAX_VLEN/16) begin
                            lane_vs1 = {{(ELEMENT_WIDTH-16){1'b0}}, vrs1_data[lane*16 +: 16]};
                            lane_vs2 = {{(ELEMENT_WIDTH-16){1'b0}}, vrs2_data[lane*16 +: 16]};
                        end else begin
                            lane_vs1 = '0;
                            lane_vs2 = '0;
                        end
                    end
                    3'b010: begin // 32-bit elements
                        if (lane < MAX_VLEN/32) begin
                            lane_vs1 = {{(ELEMENT_WIDTH-32){1'b0}}, vrs1_data[lane*32 +: 32]};
                            lane_vs2 = {{(ELEMENT_WIDTH-32){1'b0}}, vrs2_data[lane*32 +: 32]};
                        end else begin
                            lane_vs1 = '0;
                            lane_vs2 = '0;
                        end
                    end
                    3'b011: begin // 64-bit elements
                        if (lane < MAX_VLEN/64 && (lane*64 + 63) < MAX_VLEN) begin
                            lane_vs1 = vrs1_data[lane*64 +: 64];
                            lane_vs2 = vrs2_data[lane*64 +: 64];
                        end else begin
                            lane_vs1 = '0;
                            lane_vs2 = '0;
                        end
                    end
                    default: begin
                        lane_vs1 = '0;
                        lane_vs2 = '0;
                    end
                endcase
                
                // Scalar and immediate operands
                scalar_operand = {{(ELEMENT_WIDTH-32){1'b0}}, instr_reg[31:0]}; // Simplified
                imm_operand = {{(ELEMENT_WIDTH-11){imm_reg[10]}}, imm_reg};
                
                // Lane mask
                lane_mask = effective_mask[lane] && (lane < active_elements);
            end
            
            // Enhanced arithmetic operation per lane with proper mask handling
            always_comb begin
                lane_result = '0;
                
                if (lane_mask) begin
                    case (instr_type_reg)
                        VEC_ARITH_VV: begin
                            case (funct6_reg)
                                6'b000000: lane_result = lane_vs1 + lane_vs2;     // VADD.VV
                                6'b000010: lane_result = lane_vs1 - lane_vs2;     // VSUB.VV
                                6'b000100: lane_result = lane_vs1 & lane_vs2;     // VAND.VV
                                6'b000101: lane_result = lane_vs1 | lane_vs2;     // VOR.VV
                                6'b000110: lane_result = lane_vs1 ^ lane_vs2;     // VXOR.VV
                                6'b100000: lane_result = lane_vs1 * lane_vs2;     // VMUL.VV
                                6'b001000: begin // VMIN.VV
                                    lane_result = (lane_vs1 < lane_vs2) ? lane_vs1 : lane_vs2;
                                end
                                6'b001001: begin // VMAX.VV
                                    lane_result = (lane_vs1 > lane_vs2) ? lane_vs1 : lane_vs2;
                                end
                                6'b001100: begin // VSLL.VV (shift left logical)
                                    lane_result = lane_vs1 << lane_vs2[5:0];
                                end
                                6'b001101: begin // VSRL.VV (shift right logical)
                                    lane_result = lane_vs1 >> lane_vs2[5:0];
                                end
                                default: lane_result = lane_vs1;
                            endcase
                        end
                        
                        VEC_ARITH_VX: begin
                            case (funct6_reg)
                                6'b000000: lane_result = lane_vs2 + scalar_operand; // VADD.VX
                                6'b000010: lane_result = lane_vs2 - scalar_operand; // VSUB.VX
                                6'b000100: lane_result = lane_vs2 & scalar_operand; // VAND.VX
                                6'b000101: lane_result = lane_vs2 | scalar_operand; // VOR.VX
                                6'b000110: lane_result = lane_vs2 ^ scalar_operand; // VXOR.VX
                                6'b100000: lane_result = lane_vs2 * scalar_operand; // VMUL.VX
                                6'b001000: begin // VMIN.VX
                                    lane_result = (lane_vs2 < scalar_operand) ? lane_vs2 : scalar_operand;
                                end
                                6'b001001: begin // VMAX.VX
                                    lane_result = (lane_vs2 > scalar_operand) ? lane_vs2 : scalar_operand;
                                end
                                6'b001100: begin // VSLL.VX
                                    lane_result = lane_vs2 << scalar_operand[5:0];
                                end
                                6'b001101: begin // VSRL.VX
                                    lane_result = lane_vs2 >> scalar_operand[5:0];
                                end
                                default: lane_result = lane_vs2;
                            endcase
                        end
                        
                        VEC_ARITH_VI: begin
                            case (funct6_reg)
                                6'b000000: lane_result = lane_vs2 + imm_operand;   // VADD.VI
                                6'b000100: lane_result = lane_vs2 & imm_operand;   // VAND.VI
                                6'b000101: lane_result = lane_vs2 | imm_operand;   // VOR.VI
                                6'b000110: lane_result = lane_vs2 ^ imm_operand;   // VXOR.VI
                                6'b001100: begin // VSLL.VI
                                    lane_result = lane_vs2 << imm_operand[5:0];
                                end
                                6'b001101: begin // VSRL.VI
                                    lane_result = lane_vs2 >> imm_operand[5:0];
                                end
                                default: lane_result = lane_vs2;
                            endcase
                        end
                        
                        VEC_MASK_OP: begin
                            // Vector mask operations
                            case (funct6_reg)
                                6'b010000: begin // VMAND.MM
                                    lane_result = lane_vs1 & lane_vs2;
                                end
                                6'b010001: begin // VMNAND.MM
                                    lane_result = ~(lane_vs1 & lane_vs2);
                                end
                                6'b010010: begin // VMOR.MM
                                    lane_result = lane_vs1 | lane_vs2;
                                end
                                6'b010011: begin // VMNOR.MM
                                    lane_result = ~(lane_vs1 | lane_vs2);
                                end
                                6'b010100: begin // VMXOR.MM
                                    lane_result = lane_vs1 ^ lane_vs2;
                                end
                                6'b010101: begin // VMXNOR.MM
                                    lane_result = ~(lane_vs1 ^ lane_vs2);
                                end
                                default: lane_result = lane_vs1;
                            endcase
                        end
                        
                        default: begin
                            lane_result = lane_vs2;
                        end
                    endcase
                end else begin
                    // Handle mask policy for masked-out elements
                    case (mask_policy)
                        MASK_POLICY_UNDISTURBED: begin
                            // Keep old value - simplified to zero for now
                            lane_result = '0;
                        end
                        MASK_POLICY_AGNOSTIC: begin
                            // Can be any value - use 1s for simplicity
                            lane_result = '1;
                        end
                        default: begin
                            lane_result = '0;
                        end
                    endcase
                end
            end
            
            // Store lane result
            assign lane_results[lane] = lane_result;
        end
    endgenerate
    
    // Aggregate lane results into vector result
    always_comb begin
        arithmetic_result = '0;
        
        for (int lane_idx = 0; lane_idx < VECTOR_LANES; lane_idx++) begin
            case (vsew)
                3'b000: begin // 8-bit elements
                    if (lane_idx < MAX_VLEN/8) begin
                        arithmetic_result[lane_idx*8 +: 8] = lane_results[lane_idx][7:0];
                    end
                end
                3'b001: begin // 16-bit elements
                    if (lane_idx < MAX_VLEN/16) begin
                        arithmetic_result[lane_idx*16 +: 16] = lane_results[lane_idx][15:0];
                    end
                end
                3'b010: begin // 32-bit elements
                    if (lane_idx < MAX_VLEN/32) begin
                        arithmetic_result[lane_idx*32 +: 32] = lane_results[lane_idx][31:0];
                    end
                end
                3'b011: begin // 64-bit elements
                    if (lane_idx < MAX_VLEN/64 && (lane_idx*64 + 63) < MAX_VLEN) begin
                        arithmetic_result[lane_idx*64 +: 64] = lane_results[lane_idx];
                    end
                end
                default: begin
                    // No assignment
                end
            endcase
        end
    end

    // ========================================
    // Enhanced Vector Memory Operations with Gather/Scatter
    // ========================================
    
    logic [63:0] base_address;
    logic [63:0] stride_value;
    logic [MAX_VLEN-1:0] memory_result;
    logic [15:0] memory_element_counter;
    logic [15:0] elements_per_transaction;
    logic memory_transaction_complete;
    
    // Calculate elements per memory transaction based on element width
    always_comb begin
        case (vsew)
            3'b000: elements_per_transaction = MAX_VLEN / 8;   // 8-bit elements
            3'b001: elements_per_transaction = MAX_VLEN / 16;  // 16-bit elements  
            3'b010: elements_per_transaction = MAX_VLEN / 32;  // 32-bit elements
            3'b011: elements_per_transaction = MAX_VLEN / 64;  // 64-bit elements
            default: elements_per_transaction = MAX_VLEN / 32;
        endcase
    end
    
    // Memory transaction completion logic
    always_comb begin
        memory_transaction_complete = (memory_element_counter >= active_elements) || 
                                    (memory_element_counter >= elements_per_transaction);
    end
    
    // Enhanced memory address calculation with gather/scatter support
    always_comb begin
        base_address = {{32{1'b0}}, instr_reg[31:0]}; // Simplified base address
        stride_value = vrs2_data[63:0]; // Stride from vs2 for strided operations
        
        mem_addr = base_address;
        mem_wdata = '0;
        mem_req = 1'b0;
        mem_we = 1'b0;
        mem_be = '0;
        memory_result = '0;
        
        if (current_state == PIPE_MEMORY) begin
            case (instr_type_reg)
                VEC_LOAD_UNIT: begin
                    // Unit-stride load
                    mem_addr = base_address + (memory_element_counter * (1 << vsew));
                    mem_req = 1'b1;
                    mem_we = 1'b0;
                    
                    // Generate byte enables based on element width and mask
                    for (int i = 0; i < MAX_VLEN/8; i++) begin
                        if (i < active_elements && effective_mask[i]) begin
                            mem_be[i] = 1'b1;
                        end else begin
                            mem_be[i] = 1'b0;
                        end
                    end
                    
                    memory_result = mem_rdata;
                end
                
                VEC_STORE_UNIT: begin
                    // Unit-stride store
                    mem_addr = base_address + (memory_element_counter * (1 << vsew));
                    mem_req = 1'b1;
                    mem_we = 1'b1;
                    
                    // Pack data based on element width and mask
                    case (vsew)
                        3'b000: begin // 8-bit elements
                            for (int i = 0; i < MAX_VLEN/8; i++) begin
                                if (i < active_elements && effective_mask[i]) begin
                                    mem_wdata[i*8 +: 8] = vrs2_data[i*8 +: 8];
                                    mem_be[i] = 1'b1;
                                end else begin
                                    mem_be[i] = 1'b0;
                                end
                            end
                        end
                        3'b001: begin // 16-bit elements
                            for (int i = 0; i < MAX_VLEN/16; i++) begin
                                if (i < active_elements && effective_mask[i]) begin
                                    mem_wdata[i*16 +: 16] = vrs2_data[i*16 +: 16];
                                    mem_be[i*2 +: 2] = 2'b11;
                                end else begin
                                    mem_be[i*2 +: 2] = 2'b00;
                                end
                            end
                        end
                        3'b010: begin // 32-bit elements
                            for (int i = 0; i < MAX_VLEN/32; i++) begin
                                if (i < active_elements && effective_mask[i]) begin
                                    mem_wdata[i*32 +: 32] = vrs2_data[i*32 +: 32];
                                    mem_be[i*4 +: 4] = 4'b1111;
                                end else begin
                                    mem_be[i*4 +: 4] = 4'b0000;
                                end
                            end
                        end
                        3'b011: begin // 64-bit elements
                            for (int i = 0; i < MAX_VLEN/64; i++) begin
                                if (i < active_elements && effective_mask[i]) begin
                                    mem_wdata[i*64 +: 64] = vrs2_data[i*64 +: 64];
                                    mem_be[i*8 +: 8] = 8'b11111111;
                                end else begin
                                    mem_be[i*8 +: 8] = 8'b00000000;
                                end
                            end
                        end
                    endcase
                end
                
                VEC_LOAD_STRIDE: begin
                    // Strided load
                    mem_addr = base_address + (memory_element_counter * stride_value);
                    mem_req = 1'b1;
                    mem_we = 1'b0;
                    mem_be = (1 << (1 << vsew)) - 1; // Element-sized access
                    memory_result = mem_rdata;
                end
                
                VEC_STORE_STRIDE: begin
                    // Strided store
                    mem_addr = base_address + (memory_element_counter * stride_value);
                    mem_req = 1'b1;
                    mem_we = 1'b1;
                    mem_be = (1 << (1 << vsew)) - 1; // Element-sized access
                    
                    // Extract single element for strided store
                    case (vsew)
                        3'b000: mem_wdata[7:0] = vrs2_data[memory_element_counter*8 +: 8];
                        3'b001: mem_wdata[15:0] = vrs2_data[memory_element_counter*16 +: 16];
                        3'b010: mem_wdata[31:0] = vrs2_data[memory_element_counter*32 +: 32];
                        3'b011: mem_wdata[63:0] = vrs2_data[memory_element_counter*64 +: 64];
                    endcase
                end
                
                VEC_LOAD_INDEX: begin
                    // Indexed load (gather operation)
                    logic [63:0] index_value;
                    
                    // Extract index from vs2 register
                    case (vsew)
                        3'b000: index_value = {{56{1'b0}}, vrs2_data[memory_element_counter*8 +: 8]};
                        3'b001: index_value = {{48{1'b0}}, vrs2_data[memory_element_counter*16 +: 16]};
                        3'b010: index_value = {{32{1'b0}}, vrs2_data[memory_element_counter*32 +: 32]};
                        3'b011: index_value = vrs2_data[memory_element_counter*64 +: 64];
                        default: index_value = 64'b0;
                    endcase
                    
                    mem_addr = base_address + index_value;
                    mem_req = (memory_element_counter < MAX_VLEN) ? effective_mask[memory_element_counter] && (memory_element_counter < active_elements) : 1'b0;
                    mem_we = 1'b0;
                    mem_be = (1 << (1 << vsew)) - 1;
                    memory_result = mem_rdata;
                end
                
                VEC_STORE_INDEX: begin
                    // Indexed store (scatter operation)
                    logic [63:0] index_value;
                    
                    // Extract index from vs2 register
                    case (vsew)
                        3'b000: index_value = {{56{1'b0}}, vrs2_data[memory_element_counter*8 +: 8]};
                        3'b001: index_value = {{48{1'b0}}, vrs2_data[memory_element_counter*16 +: 16]};
                        3'b010: index_value = {{32{1'b0}}, vrs2_data[memory_element_counter*32 +: 32]};
                        3'b011: index_value = vrs2_data[memory_element_counter*64 +: 64];
                        default: index_value = 64'b0;
                    endcase
                    
                    mem_addr = base_address + index_value;
                    mem_req = (memory_element_counter < MAX_VLEN) ? effective_mask[memory_element_counter] && (memory_element_counter < active_elements) : 1'b0;
                    mem_we = 1'b1;
                    mem_be = (1 << (1 << vsew)) - 1;
                    
                    // Extract data element from vs1 register (data source for scatter)
                    case (vsew)
                        3'b000: mem_wdata[7:0] = vrs1_data[memory_element_counter*8 +: 8];
                        3'b001: mem_wdata[15:0] = vrs1_data[memory_element_counter*16 +: 16];
                        3'b010: mem_wdata[31:0] = vrs1_data[memory_element_counter*32 +: 32];
                        3'b011: mem_wdata[63:0] = vrs1_data[memory_element_counter*64 +: 64];
                    endcase
                end
                
                default: begin
                    mem_req = 1'b0;
                end
            endcase
        end
    end
    
    // Memory element counter for multi-element operations
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            memory_element_counter <= 16'b0;
        end else begin
            if (current_state == PIPE_MEMORY) begin
                if (mem_req && mem_ready) begin
                    if (memory_transaction_complete) begin
                        memory_element_counter <= 16'b0;
                    end else begin
                        memory_element_counter <= memory_element_counter + 1;
                    end
                end
            end else begin
                memory_element_counter <= 16'b0;
            end
        end
    end

    // ========================================
    // Output Assignment
    // ========================================
    
    // Register file interface
    assign vrs1_addr = vs1_reg;
    assign vrs2_addr = vs2_reg;
    assign vrd_addr = vd_reg;
    
    // Result selection
    always_comb begin
        case (instr_type_reg)
            VEC_ARITH_VV, VEC_ARITH_VX, VEC_ARITH_VI: begin
                vrd_data = arithmetic_result;
            end
            VEC_LOAD_UNIT, VEC_LOAD_STRIDE, VEC_LOAD_INDEX: begin
                vrd_data = memory_result;
            end
            default: begin
                vrd_data = '0;
            end
        endcase
    end
    
    assign vrd_we = (current_state == PIPE_WRITEBACK) && 
                    (instr_type_reg == VEC_ARITH_VV || 
                     instr_type_reg == VEC_ARITH_VX || 
                     instr_type_reg == VEC_ARITH_VI ||
                     instr_type_reg == VEC_LOAD_UNIT ||
                     instr_type_reg == VEC_LOAD_STRIDE ||
                     instr_type_reg == VEC_LOAD_INDEX);
    
    // Control signals
    assign instruction_ready = (current_state == PIPE_IDLE);
    assign pipeline_busy = (current_state != PIPE_IDLE);
    assign pipeline_done = (current_state == PIPE_WRITEBACK);
    assign pipeline_error = (decoded_instr_type == VEC_INVALID) && instruction_valid;

endmodule