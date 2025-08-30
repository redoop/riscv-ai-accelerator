// RISC-V Core Module
// Main processor core implementing RV64IMAFDV instruction set

`timescale 1ns/1ps

module riscv_core #(
    parameter XLEN = 64,
    parameter VLEN = 512  // Vector length for RVV extension
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Instruction memory interface
    output logic [XLEN-1:0]     imem_addr,
    output logic                imem_req,
    input  logic [31:0]         imem_rdata,
    input  logic                imem_ready,
    
    // Data memory interface
    output logic [XLEN-1:0]     dmem_addr,
    output logic [XLEN-1:0]     dmem_wdata,
    output logic [7:0]          dmem_wmask,
    output logic                dmem_req,
    output logic                dmem_we,
    input  logic [XLEN-1:0]     dmem_rdata,
    input  logic                dmem_ready,
    
    // AI accelerator interface
    output logic [31:0]         ai_addr,
    output logic [63:0]         ai_wdata,
    input  logic [63:0]         ai_rdata,
    output logic                ai_req,
    output logic                ai_we,
    output logic [7:0]          ai_be,
    input  logic                ai_ready,
    input  logic                ai_error,
    output logic                ai_task_valid,
    output logic [31:0]         ai_task_id,
    output logic [7:0]          ai_task_type,
    input  logic                ai_task_ready,
    input  logic                ai_task_done,
    
    // Interrupt interface
    input  logic                ext_irq,
    input  logic                timer_irq,
    input  logic                soft_irq
);

    // Package parameters (inlined for synthesis compatibility)
    localparam STATUS_OK = 8'h00;
    localparam STATUS_ERROR = 8'hFF;
    localparam STATUS_BUSY = 8'h01;

    // ========================================
    // Pipeline Stage Registers
    // ========================================
    
    // Fetch Stage
    logic [XLEN-1:0]    pc_f;
    logic [31:0]        instr_f;
    logic               valid_f;
    
    // Decode Stage  
    logic [XLEN-1:0]    pc_d;
    logic [31:0]        instr_d;
    logic               valid_d;
    
    // Execute Stage
    logic [XLEN-1:0]    pc_e;
    logic [31:0]        instr_e;
    logic               valid_e;
    
    // Memory Stage
    logic [XLEN-1:0]    pc_m;
    logic [31:0]        instr_m;
    logic               valid_m;
    
    // Writeback Stage
    logic [XLEN-1:0]    pc_w;
    logic [31:0]        instr_w;
    logic               valid_w;

    // ========================================
    // Register File
    // ========================================
    
    logic [XLEN-1:0]    regfile [31:0];
    logic [4:0]         rs1_addr, rs2_addr, rd_addr;
    logic [XLEN-1:0]    rs1_data, rs2_data, rd_data;
    logic               reg_we;
    
    // ========================================
    // Program Counter
    // ========================================
    
    logic [XLEN-1:0]    pc_next;
    logic [XLEN-1:0]    pc_plus4;
    logic [XLEN-1:0]    pc_branch;
    logic               pc_sel;
    logic               stall;
    
    // ========================================
    // Instruction Decoder
    // ========================================
    
    // Instruction fields
    logic [6:0]         opcode;
    logic [4:0]         rd, rs1, rs2;
    logic [2:0]         funct3;
    logic [6:0]         funct7;
    logic [XLEN-1:0]    imm;
    
    // Control signals
    logic               reg_write;
    logic               mem_read;
    logic               mem_write;
    logic               branch;
    logic               jump;
    logic               alu_src;
    logic [3:0]         alu_op;
    logic [1:0]         wb_sel;
    
    // Extension control signals
    logic               mdu_enable;
    logic               fpu_enable;
    logic               vec_enable;
    logic               ai_enable;
    logic               is_32bit;
    logic               is_double;
    
    // ========================================
    // ALU
    // ========================================
    
    logic [XLEN-1:0]    alu_a, alu_b;
    logic [XLEN-1:0]    alu_result;
    logic               alu_zero;
    logic               alu_overflow;
    
    // ========================================
    // Multiply/Divide Unit (MDU)
    // ========================================
    
    logic [XLEN-1:0]    mdu_result;
    logic               mdu_ready;
    logic               mdu_valid;
    
    // ========================================
    // Floating Point Unit (FPU)
    // ========================================
    
    logic [XLEN-1:0]    fpu_result;
    logic               fpu_ready;
    logic [4:0]         fpu_flags;
    
    // ========================================
    // Vector Unit (VU)
    // ========================================
    
    logic [XLEN-1:0]    vec_result;
    logic               vec_ready;
    logic               vec_valid;
    
    // ========================================
    // AI Unit
    // ========================================
    
    logic [XLEN-1:0]    ai_result;
    logic               ai_ready;
    logic               ai_valid;
    logic [4:0]         ai_flags;
    
    // Vector configuration registers
    logic [XLEN-1:0]    vtype_reg;
    logic [XLEN-1:0]    vl_reg;
    
    // Vector configuration update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vtype_reg <= '0;
            vl_reg <= '0;
        end else if (vec_enable && valid_e && (instr_e[14:12] == 3'b111)) begin
            // Vector configuration instructions (vsetvl, vsetivli, vsetvli)
            vtype_reg <= rs2_data_e; // vtype comes from rs2 or immediate
            vl_reg <= alu_a;         // requested vl comes from rs1
        end
    end
    
    // ========================================
    // Branch Unit
    // ========================================
    
    logic               branch_taken;
    logic               jump_taken;
    
    // ========================================
    // Hazard Detection and Forwarding
    // ========================================
    
    logic               load_use_hazard;
    logic               data_hazard_rs1;
    logic               data_hazard_rs2;
    logic [1:0]         forward_a;
    logic [1:0]         forward_b;

    // ========================================
    // Fetch Stage Implementation
    // ========================================
    
    assign pc_plus4 = pc_f + 4;
    assign pc_next = (branch_taken | jump_taken) ? pc_branch : pc_plus4;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_f <= 64'h0000_0000_0000_0000;
        end else if (!stall) begin
            pc_f <= pc_next;
        end
    end
    
    // Instruction memory interface
    assign imem_addr = pc_f;
    assign imem_req = !stall;
    assign instr_f = imem_rdata;
    assign valid_f = imem_ready && !stall;

    // ========================================
    // Fetch-Decode Pipeline Register
    // ========================================
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_d <= '0;
            instr_d <= 32'h0000_0013; // NOP (addi x0, x0, 0)
            valid_d <= 1'b0;
        end else if (!stall) begin
            pc_d <= pc_f;
            instr_d <= instr_f;
            valid_d <= valid_f;
        end
    end

    // ========================================
    // Decode Stage Implementation
    // ========================================
    
    // Instruction field extraction
    assign opcode = instr_d[6:0];
    assign rd = instr_d[11:7];
    assign funct3 = instr_d[14:12];
    assign rs1 = instr_d[19:15];
    assign rs2 = instr_d[24:20];
    assign funct7 = instr_d[31:25];
    
    // Register file read
    assign rs1_addr = rs1;
    assign rs2_addr = rs2;
    assign rs1_data = (rs1 == 5'b0) ? 64'b0 : regfile[rs1];
    assign rs2_data = (rs2 == 5'b0) ? 64'b0 : regfile[rs2];
    
    // Immediate generation
    always_comb begin
        case (opcode)
            7'b0010011, 7'b0000011, 7'b1100111: begin // I-type
                imm = {{52{instr_d[31]}}, instr_d[31:20]};
            end
            7'b0100011: begin // S-type
                imm = {{52{instr_d[31]}}, instr_d[31:25], instr_d[11:7]};
            end
            7'b1100011: begin // B-type
                imm = {{51{instr_d[31]}}, instr_d[31], instr_d[7], instr_d[30:25], instr_d[11:8], 1'b0};
            end
            7'b0110111, 7'b0010111: begin // U-type
                imm = {{32{instr_d[31]}}, instr_d[31:12], 12'b0};
            end
            7'b1101111: begin // J-type
                imm = {{43{instr_d[31]}}, instr_d[31], instr_d[19:12], instr_d[20], instr_d[30:21], 1'b0};
            end
            default: imm = 64'b0;
        endcase
    end
    
    // Control unit
    riscv_control_unit control_unit (
        .opcode(opcode),
        .funct3(funct3),
        .funct7(funct7),
        .reg_write(reg_write),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .branch(branch),
        .jump(jump),
        .alu_src(alu_src),
        .alu_op(alu_op),
        .wb_sel(wb_sel),
        .mdu_enable(mdu_enable),
        .fpu_enable(fpu_enable),
        .vec_enable(vec_enable),
        .ai_enable(ai_enable),
        .is_32bit(is_32bit),
        .is_double(is_double)
    );

    // ========================================
    // Decode-Execute Pipeline Register
    // ========================================
    
    logic [XLEN-1:0]    rs1_data_e, rs2_data_e, imm_e;
    logic [4:0]         rd_e, rs1_e, rs2_e;
    logic               reg_write_e, mem_read_e, mem_write_e;
    logic               branch_e, jump_e, alu_src_e;
    logic [3:0]         alu_op_e;
    logic [1:0]         wb_sel_e;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_e <= '0;
            instr_e <= 32'h0000_0013;
            valid_e <= 1'b0;
            rs1_data_e <= '0;
            rs2_data_e <= '0;
            imm_e <= '0;
            rd_e <= '0;
            rs1_e <= '0;
            rs2_e <= '0;
            reg_write_e <= 1'b0;
            mem_read_e <= 1'b0;
            mem_write_e <= 1'b0;
            branch_e <= 1'b0;
            jump_e <= 1'b0;
            alu_src_e <= 1'b0;
            alu_op_e <= 4'b0;
            wb_sel_e <= 2'b0;
        end else if (!stall) begin
            pc_e <= pc_d;
            instr_e <= instr_d;
            valid_e <= valid_d;
            rs1_data_e <= rs1_data;
            rs2_data_e <= rs2_data;
            imm_e <= imm;
            rd_e <= rd;
            rs1_e <= rs1;
            rs2_e <= rs2;
            reg_write_e <= reg_write;
            mem_read_e <= mem_read;
            mem_write_e <= mem_write;
            branch_e <= branch;
            jump_e <= jump;
            alu_src_e <= alu_src;
            alu_op_e <= alu_op;
            wb_sel_e <= wb_sel;
        end
    end

    // ========================================
    // Execute Stage Implementation
    // ========================================
    
    // Forwarding unit
    riscv_forwarding_unit forwarding_unit (
        .rs1_e(rs1_e),
        .rs2_e(rs2_e),
        .rd_m(instr_m[11:7]),
        .rd_w(instr_w[11:7]),
        .reg_write_m(valid_m && (instr_m[6:0] != 7'b0100011)), // Not store
        .reg_write_w(reg_we),
        .forward_a(forward_a),
        .forward_b(forward_b)
    );
    
    // ALU input selection with forwarding
    always_comb begin
        case (forward_a)
            2'b00: alu_a = rs1_data_e;
            2'b01: alu_a = rd_data; // Forward from writeback
            2'b10: alu_a = alu_result; // Forward from memory (not implemented yet)
            default: alu_a = rs1_data_e;
        endcase
    end
    
    always_comb begin
        case (forward_b)
            2'b00: alu_b = alu_src_e ? imm_e : rs2_data_e;
            2'b01: alu_b = alu_src_e ? imm_e : rd_data;
            2'b10: alu_b = alu_src_e ? imm_e : alu_result;
            default: alu_b = alu_src_e ? imm_e : rs2_data_e;
        endcase
    end
    
    // ALU
    riscv_alu alu (
        .a(alu_a),
        .b(alu_b),
        .alu_op(alu_op_e),
        .result(alu_result),
        .zero(alu_zero),
        .overflow(alu_overflow)
    );
    
    // Multiply/Divide Unit
    riscv_mdu mdu (
        .clk(clk),
        .rst_n(rst_n),
        .mdu_enable(mdu_enable && valid_e),
        .funct3(instr_e[14:12]),
        .is_32bit(is_32bit),
        .rs1_data(alu_a),
        .rs2_data(rs2_data_e),
        .mdu_result(mdu_result),
        .mdu_ready(mdu_ready),
        .mdu_valid(mdu_valid)
    );
    
    // Floating Point Unit
    riscv_fpu fpu (
        .clk(clk),
        .rst_n(rst_n),
        .fpu_enable(fpu_enable && valid_e),
        .fpu_op(instr_e[31:25]),
        .funct3(instr_e[14:12]),
        .funct7(instr_e[31:25]),
        .is_double(is_double),
        .rs1_data(alu_a),
        .rs2_data(rs2_data_e),
        .rs3_data(imm_e), // For fused operations, rs3 would be separate
        .fpu_result(fpu_result),
        .fpu_ready(fpu_ready),
        .fpu_flags(fpu_flags)
    );
    
    // Vector Unit
    riscv_vector_unit vector_unit (
        .clk(clk),
        .rst_n(rst_n),
        .vec_enable(vec_enable && valid_e),
        .vec_opcode(instr_e[31:26]),
        .funct3(instr_e[14:12]),
        .funct6(instr_e[31:26]),
        .vm(instr_e[25]),
        .vs1(instr_e[19:15]),
        .vs2(instr_e[24:20]),
        .vd(instr_e[11:7]),
        .rs1(instr_e[19:15]),
        .rs1_data(alu_a),
        .imm(instr_e[30:20]),
        .vtype(vtype_reg),
        .vl(vl_reg),
        .vec_mem_addr(),  // Connect to memory interface
        .vec_mem_wdata(),
        .vec_mem_req(),
        .vec_mem_we(),
        .vec_mem_be(),
        .vec_mem_rdata('0), // Connect to memory interface
        .vec_mem_ready(1'b1),
        .vec_result(vec_result),
        .vec_ready(vec_ready),
        .vec_valid(vec_valid)
    );
    
    // Branch unit
    riscv_branch_unit branch_unit (
        .rs1_data(alu_a),
        .rs2_data(rs2_data_e), // Use original rs2 for comparison
        .funct3(instr_e[14:12]),
        .branch(branch_e),
        .jump(jump_e),
        .branch_taken(branch_taken),
        .jump_taken(jump_taken)
    );
    
    // Branch target calculation
    assign pc_branch = jump_e ? (alu_a + imm_e) : (pc_e + imm_e);
    
    // AI Unit
    riscv_ai_unit ai_unit (
        .clk(clk),
        .rst_n(rst_n),
        .ai_enable(ai_enable && valid_e),
        .ai_opcode(instr_e[6:0]),
        .funct3(instr_e[14:12]),
        .funct7(instr_e[31:25]),
        .rs1(instr_e[19:15]),
        .rs2(instr_e[24:20]),
        .rs3(instr_e[31:27]), // Use upper bits for rs3
        .rd(instr_e[11:7]),
        .rs1_data(alu_a),
        .rs2_data(rs2_data_e),
        .rs3_data(imm_e), // Use immediate field for rs3 data
        .ai_result(ai_result),
        .ai_mem_addr(), // Connect to memory arbiter
        .ai_mem_wdata(),
        .ai_mem_wmask(),
        .ai_mem_req(),
        .ai_mem_we(),
        .ai_mem_rdata(dmem_rdata),
        .ai_mem_ready(dmem_ready),
        .ai_ready(ai_ready),
        .ai_valid(ai_valid),
        .ai_flags(ai_flags)
    );

    // ========================================
    // Execute-Memory Pipeline Register
    // ========================================
    
    logic [XLEN-1:0]    alu_result_m, rs2_data_m;
    logic [4:0]         rd_m;
    logic               reg_write_m, mem_read_m, mem_write_m;
    logic [1:0]         wb_sel_m;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_m <= '0;
            instr_m <= 32'h0000_0013;
            valid_m <= 1'b0;
            alu_result_m <= '0;
            rs2_data_m <= '0;
            rd_m <= '0;
            reg_write_m <= 1'b0;
            mem_read_m <= 1'b0;
            mem_write_m <= 1'b0;
            wb_sel_m <= 2'b0;
        end else begin
            pc_m <= pc_e;
            instr_m <= instr_e;
            valid_m <= valid_e;
            alu_result_m <= alu_result;
            rs2_data_m <= (forward_b == 2'b01) ? rd_data : rs2_data_e;
            rd_m <= rd_e;
            reg_write_m <= reg_write_e;
            mem_read_m <= mem_read_e;
            mem_write_m <= mem_write_e;
            wb_sel_m <= wb_sel_e;
        end
    end

    // ========================================
    // Memory Stage Implementation
    // ========================================
    
    // Data memory interface
    assign dmem_addr = alu_result_m;
    assign dmem_wdata = rs2_data_m;
    assign dmem_req = mem_read_m | mem_write_m;
    assign dmem_we = mem_write_m;
    
    // Byte enable generation based on funct3
    always_comb begin
        case (instr_m[14:12]) // funct3
            3'b000: dmem_wmask = 8'b0000_0001; // SB
            3'b001: dmem_wmask = 8'b0000_0011; // SH
            3'b010: dmem_wmask = 8'b0000_1111; // SW
            3'b011: dmem_wmask = 8'b1111_1111; // SD
            default: dmem_wmask = 8'b1111_1111;
        endcase
    end

    // ========================================
    // Memory-Writeback Pipeline Register
    // ========================================
    
    logic [XLEN-1:0]    alu_result_w, mem_data_w;
    logic [4:0]         rd_w;
    logic               reg_write_w;
    logic [1:0]         wb_sel_w;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc_w <= '0;
            instr_w <= 32'h0000_0013;
            valid_w <= 1'b0;
            alu_result_w <= '0;
            mem_data_w <= '0;
            rd_w <= '0;
            reg_write_w <= 1'b0;
            wb_sel_w <= 2'b0;
        end else begin
            pc_w <= pc_m;
            instr_w <= instr_m;
            valid_w <= valid_m;
            alu_result_w <= alu_result_m;
            mem_data_w <= dmem_rdata;
            rd_w <= rd_m;
            reg_write_w <= reg_write_m;
            wb_sel_w <= wb_sel_m;
        end
    end

    // ========================================
    // Writeback Stage Implementation
    // ========================================
    
    // Writeback data selection
    always_comb begin
        case (wb_sel_w)
            2'b00: rd_data = alu_result_w;      // ALU result
            2'b01: begin
                // Memory data, Vector result, or AI result
                if (ai_valid) begin
                    rd_data = ai_result;
                end else if (vec_valid) begin
                    rd_data = vec_result;
                end else begin
                    rd_data = mem_data_w;
                end
            end
            2'b10: rd_data = fpu_result;        // FPU result
            2'b11: rd_data = mdu_result;        // MDU result
            default: rd_data = alu_result_w;
        endcase
    end
    
    assign rd_addr = rd_w;
    assign reg_we = reg_write_w && valid_w && (rd_w != 5'b0);
    
    // Register file write
    always_ff @(posedge clk) begin
        if (reg_we) begin
            regfile[rd_addr] <= rd_data;
        end
    end

    // ========================================
    // Hazard Detection Unit
    // ========================================
    
    riscv_hazard_unit hazard_unit (
        .rs1_d(rs1),
        .rs2_d(rs2),
        .rd_e(rd_e),
        .mem_read_e(mem_read_e),
        .branch_taken(branch_taken),
        .jump_taken(jump_taken),
        .stall(stall),
        .load_use_hazard(load_use_hazard)
    );

    // ========================================
    // AI Accelerator Interface (Placeholder)
    // ========================================
    
    // AI interface will be implemented in task 2.3
    assign ai_addr = '0;
    assign ai_wdata = '0;
    assign ai_req = 1'b0;
    assign ai_we = 1'b0;
    assign ai_be = '0;
    assign ai_task_valid = 1'b0;
    assign ai_task_id = '0;
    assign ai_task_type = '0;

endmodule