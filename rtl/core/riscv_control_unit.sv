// RISC-V Control Unit
// Generates control signals for instruction execution

`timescale 1ns/1ps

module riscv_control_unit (
    input  logic [6:0]  opcode,
    input  logic [2:0]  funct3,
    input  logic [6:0]  funct7,
    
    output logic        reg_write,
    output logic        mem_read,
    output logic        mem_write,
    output logic        branch,
    output logic        jump,
    output logic        alu_src,
    output logic [3:0]  alu_op,
    output logic [1:0]  wb_sel,
    
    // Extension control signals
    output logic        mdu_enable,     // M extension enable
    output logic        fpu_enable,     // F/D extension enable
    output logic        vec_enable,     // V extension enable
    output logic        ai_enable,      // AI extension enable
    output logic        is_32bit,       // 32-bit operation (W suffix)
    output logic        is_double       // Double precision FP
);

    // RV64I Base Instruction Set Opcodes
    localparam [6:0] OP_LUI     = 7'b0110111;  // Load Upper Immediate
    localparam [6:0] OP_AUIPC   = 7'b0010111;  // Add Upper Immediate to PC
    localparam [6:0] OP_JAL     = 7'b1101111;  // Jump and Link
    localparam [6:0] OP_JALR    = 7'b1100111;  // Jump and Link Register
    localparam [6:0] OP_BRANCH  = 7'b1100011;  // Branch instructions
    localparam [6:0] OP_LOAD    = 7'b0000011;  // Load instructions
    localparam [6:0] OP_STORE   = 7'b0100011;  // Store instructions
    localparam [6:0] OP_IMM     = 7'b0010011;  // Immediate arithmetic
    localparam [6:0] OP_REG     = 7'b0110011;  // Register arithmetic
    localparam [6:0] OP_FENCE   = 7'b0001111;  // Fence instructions
    localparam [6:0] OP_SYSTEM  = 7'b1110011;  // System instructions
    
    // RV64M Extension Opcodes
    localparam [6:0] OP_IMM_32  = 7'b0011011;  // 32-bit immediate arithmetic
    localparam [6:0] OP_REG_32  = 7'b0111011;  // 32-bit register arithmetic
    
    // RV64F/D Extension Opcodes
    localparam [6:0] OP_LOAD_FP = 7'b0000111;  // Floating-point loads
    localparam [6:0] OP_STORE_FP= 7'b0100111;  // Floating-point stores
    localparam [6:0] OP_FMADD   = 7'b1000011;  // Floating-point fused multiply-add
    localparam [6:0] OP_FMSUB   = 7'b1000111;  // Floating-point fused multiply-subtract
    localparam [6:0] OP_FNMSUB  = 7'b1001011;  // Floating-point negated fused multiply-subtract
    localparam [6:0] OP_FNMADD  = 7'b1001111;  // Floating-point negated fused multiply-add
    localparam [6:0] OP_FP      = 7'b1010011;  // Floating-point operations
    
    // RVV Extension Opcodes
    localparam [6:0] OP_VECTOR  = 7'b1010111;  // Vector operations
    
    // AI Extension Opcodes
    localparam [6:0] OP_AI_CUSTOM = 7'b0001011;  // Custom AI operations

    // ALU Operation Codes
    localparam [3:0] ALU_ADD    = 4'b0000;
    localparam [3:0] ALU_SUB    = 4'b0001;
    localparam [3:0] ALU_SLL    = 4'b0010;
    localparam [3:0] ALU_SLT    = 4'b0011;
    localparam [3:0] ALU_SLTU   = 4'b0100;
    localparam [3:0] ALU_XOR    = 4'b0101;
    localparam [3:0] ALU_SRL    = 4'b0110;
    localparam [3:0] ALU_SRA    = 4'b0111;
    localparam [3:0] ALU_OR     = 4'b1000;
    localparam [3:0] ALU_AND    = 4'b1001;
    localparam [3:0] ALU_LUI    = 4'b1010;
    localparam [3:0] ALU_AUIPC  = 4'b1011;

    always_comb begin
        // Default values
        reg_write = 1'b0;
        mem_read = 1'b0;
        mem_write = 1'b0;
        branch = 1'b0;
        jump = 1'b0;
        alu_src = 1'b0;
        alu_op = ALU_ADD;
        wb_sel = 2'b00; // ALU result
        
        // Extension control defaults
        mdu_enable = 1'b0;
        fpu_enable = 1'b0;
        vec_enable = 1'b0;
        ai_enable = 1'b0;
        is_32bit = 1'b0;
        is_double = 1'b0;
        
        case (opcode)
            OP_LUI: begin
                reg_write = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_LUI;
                wb_sel = 2'b00;
            end
            
            OP_AUIPC: begin
                reg_write = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_AUIPC;
                wb_sel = 2'b00;
            end
            
            OP_JAL: begin
                reg_write = 1'b1;
                jump = 1'b1;
                wb_sel = 2'b10; // PC + 4
            end
            
            OP_JALR: begin
                reg_write = 1'b1;
                jump = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_ADD;
                wb_sel = 2'b10; // PC + 4
            end
            
            OP_BRANCH: begin
                branch = 1'b1;
                alu_op = ALU_ADD; // For address calculation
            end
            
            OP_LOAD: begin
                reg_write = 1'b1;
                mem_read = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_ADD;
                wb_sel = 2'b01; // Memory data
            end
            
            OP_STORE: begin
                mem_write = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_ADD;
            end
            
            OP_IMM: begin
                reg_write = 1'b1;
                alu_src = 1'b1;
                case (funct3)
                    3'b000: alu_op = ALU_ADD;   // ADDI
                    3'b010: alu_op = ALU_SLT;   // SLTI
                    3'b011: alu_op = ALU_SLTU;  // SLTIU
                    3'b100: alu_op = ALU_XOR;   // XORI
                    3'b110: alu_op = ALU_OR;    // ORI
                    3'b111: alu_op = ALU_AND;   // ANDI
                    3'b001: alu_op = ALU_SLL;   // SLLI
                    3'b101: begin
                        if (funct7[5]) alu_op = ALU_SRA; // SRAI
                        else alu_op = ALU_SRL;           // SRLI
                    end
                    default: alu_op = ALU_ADD;
                endcase
            end
            
            OP_REG: begin
                reg_write = 1'b1;
                if (funct7[0]) begin
                    // M extension instructions (MUL, DIV, etc.)
                    mdu_enable = 1'b1;
                    wb_sel = 2'b11; // MDU result
                end else begin
                    // Base integer instructions
                    case (funct3)
                        3'b000: begin
                            if (funct7[5]) alu_op = ALU_SUB; // SUB
                            else alu_op = ALU_ADD;           // ADD
                        end
                        3'b001: alu_op = ALU_SLL;   // SLL
                        3'b010: alu_op = ALU_SLT;   // SLT
                        3'b011: alu_op = ALU_SLTU;  // SLTU
                        3'b100: alu_op = ALU_XOR;   // XOR
                        3'b101: begin
                            if (funct7[5]) alu_op = ALU_SRA; // SRA
                            else alu_op = ALU_SRL;           // SRL
                        end
                        3'b110: alu_op = ALU_OR;    // OR
                        3'b111: alu_op = ALU_AND;   // AND
                        default: alu_op = ALU_ADD;
                    endcase
                end
            end
            
            OP_IMM_32: begin
                // 32-bit immediate arithmetic (RV64I)
                reg_write = 1'b1;
                alu_src = 1'b1;
                is_32bit = 1'b1;
                case (funct3)
                    3'b000: alu_op = ALU_ADD;   // ADDIW
                    3'b001: alu_op = ALU_SLL;   // SLLIW
                    3'b101: begin
                        if (funct7[5]) alu_op = ALU_SRA; // SRAIW
                        else alu_op = ALU_SRL;           // SRLIW
                    end
                    default: alu_op = ALU_ADD;
                endcase
            end
            
            OP_REG_32: begin
                // 32-bit register arithmetic (RV64I and RV64M)
                reg_write = 1'b1;
                is_32bit = 1'b1;
                if (funct7[0]) begin
                    // M extension 32-bit instructions
                    mdu_enable = 1'b1;
                    wb_sel = 2'b11; // MDU result
                end else begin
                    // Base 32-bit integer instructions
                    case (funct3)
                        3'b000: begin
                            if (funct7[5]) alu_op = ALU_SUB; // SUBW
                            else alu_op = ALU_ADD;           // ADDW
                        end
                        3'b001: alu_op = ALU_SLL;   // SLLW
                        3'b101: begin
                            if (funct7[5]) alu_op = ALU_SRA; // SRAW
                            else alu_op = ALU_SRL;           // SRLW
                        end
                        default: alu_op = ALU_ADD;
                    endcase
                end
            end
            
            OP_LOAD_FP: begin
                // Floating-point loads
                reg_write = 1'b1;
                mem_read = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_ADD;
                wb_sel = 2'b01; // Memory data
                case (funct3)
                    3'b010: is_double = 1'b0; // FLW (single precision)
                    3'b011: is_double = 1'b1; // FLD (double precision)
                    default: is_double = 1'b0;
                endcase
            end
            
            OP_STORE_FP: begin
                // Floating-point stores
                mem_write = 1'b1;
                alu_src = 1'b1;
                alu_op = ALU_ADD;
                case (funct3)
                    3'b010: is_double = 1'b0; // FSW (single precision)
                    3'b011: is_double = 1'b1; // FSD (double precision)
                    default: is_double = 1'b0;
                endcase
            end
            
            OP_FP: begin
                // Floating-point operations
                reg_write = 1'b1;
                fpu_enable = 1'b1;
                wb_sel = 2'b10; // FPU result
                is_double = (funct7[1:0] == 2'b01); // D extension vs F extension
            end
            
            OP_FMADD, OP_FMSUB, OP_FNMSUB, OP_FNMADD: begin
                // Floating-point fused multiply-add operations
                reg_write = 1'b1;
                fpu_enable = 1'b1;
                wb_sel = 2'b10; // FPU result
                is_double = (funct3[1:0] == 2'b01); // D extension vs F extension
            end
            
            OP_VECTOR: begin
                // Vector operations
                vec_enable = 1'b1;
                // Vector operations may or may not write to scalar registers
                // This depends on the specific vector instruction
                case (funct3)
                    3'b111: begin // Vector configuration instructions
                        reg_write = 1'b1;
                        wb_sel = 2'b01; // Vector unit result
                    end
                    default: begin
                        reg_write = 1'b0; // Most vector ops don't write scalar registers
                    end
                endcase
            end
            
            OP_AI_CUSTOM: begin
                // AI custom instructions
                ai_enable = 1'b1;
                reg_write = 1'b1;
                wb_sel = 2'b01; // AI unit result (reuse vector result path)
            end
            
            OP_FENCE: begin
                // FENCE instructions - no operation for now
                reg_write = 1'b0;
            end
            
            OP_SYSTEM: begin
                // ECALL, EBREAK, CSR instructions - basic implementation
                case (funct3)
                    3'b000: begin // ECALL/EBREAK
                        reg_write = 1'b0;
                    end
                    default: begin // CSR instructions
                        reg_write = 1'b1;
                        alu_op = ALU_ADD; // Pass through for now
                    end
                endcase
            end
            
            default: begin
                // Invalid instruction - all signals remain at default
                reg_write = 1'b0;
            end
        endcase
    end

endmodule