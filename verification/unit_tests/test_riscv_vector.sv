// Test bench for RISC-V Vector Unit
// Tests RVV extension functionality

`timescale 1ns / 1ps

module test_riscv_vector;

    // Parameters
    parameter XLEN = 64;
    parameter VLEN = 512;
    parameter ELEN = 64;
    parameter CLK_PERIOD = 10;

    // Signals
    logic                clk;
    logic                rst_n;
    logic                vec_enable;
    logic [5:0]          vec_opcode;
    logic [2:0]          funct3;
    logic [5:0]          funct6;
    logic                vm;
    logic [4:0]          vs1;
    logic [4:0]          vs2;
    logic [4:0]          vd;
    logic [4:0]          rs1;
    logic [XLEN-1:0]     rs1_data;
    logic [10:0]         imm;
    logic [XLEN-1:0]     vtype;
    logic [XLEN-1:0]     vl;
    logic [XLEN-1:0]     vec_mem_addr;
    logic [VLEN-1:0]     vec_mem_wdata;
    logic                vec_mem_req;
    logic                vec_mem_we;
    logic [VLEN/8-1:0]   vec_mem_be;
    logic [VLEN-1:0]     vec_mem_rdata;
    logic                vec_mem_ready;
    logic [XLEN-1:0]     vec_result;
    logic                vec_ready;
    logic                vec_valid;

    // Test variables
    int                  test_count;
    int                  pass_count;
    int                  fail_count;

    // Memory model for vector loads/stores
    logic [VLEN-1:0]     test_memory [0:1023];

    // DUT instantiation
    riscv_vector_unit #(
        .XLEN(XLEN),
        .VLEN(VLEN),
        .ELEN(ELEN)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .vec_enable(vec_enable),
        .vec_opcode(vec_opcode),
        .funct3(funct3),
        .funct6(funct6),
        .vm(vm),
        .vs1(vs1),
        .vs2(vs2),
        .vd(vd),
        .rs1(rs1),
        .rs1_data(rs1_data),
        .imm(imm),
        .vtype(vtype),
        .vl(vl),
        .vec_mem_addr(vec_mem_addr),
        .vec_mem_wdata(vec_mem_wdata),
        .vec_mem_req(vec_mem_req),
        .vec_mem_we(vec_mem_we),
        .vec_mem_be(vec_mem_be),
        .vec_mem_rdata(vec_mem_rdata),
        .vec_mem_ready(vec_mem_ready),
        .vec_result(vec_result),
        .vec_ready(vec_ready),
        .vec_valid(vec_valid)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Memory model
    always_comb begin
        vec_mem_ready = 1'b1;
        if (vec_mem_req && !vec_mem_we) begin
            // Read operation
            vec_mem_rdata = test_memory[vec_mem_addr[15:6]]; // Simple address mapping
        end else begin
            vec_mem_rdata = '0;
        end
    end

    always_ff @(posedge clk) begin
        if (vec_mem_req && vec_mem_we) begin
            // Write operation
            test_memory[vec_mem_addr[15:6]] <= vec_mem_wdata;
        end
    end

    // Test stimulus
    initial begin
        $display("Starting RISC-V Vector Unit Tests");
        
        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        
        rst_n = 0;
        vec_enable = 0;
        vec_opcode = 6'b0;
        funct3 = 3'b0;
        funct6 = 6'b0;
        vm = 1;
        vs1 = 5'b0;
        vs2 = 5'b0;
        vd = 5'b0;
        rs1 = 5'b0;
        rs1_data = 0;
        imm = 11'b0;
        vtype = 0;
        vl = 0;
        
        // Initialize test memory
        for (int i = 0; i < 1024; i++) begin
            test_memory[i] = i * 64'h0101_0101_0101_0101;
        end
        
        // Reset
        #(CLK_PERIOD * 2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // Configure vector unit for testing
        configure_vector_unit();
        
        // Test vector arithmetic operations
        test_vector_arithmetic();
        
        // Test vector memory operations
        test_vector_memory();
        
        // Test vector configuration
        test_vector_config();
        
        // Summary
        $display("\n=== Vector Unit Test Summary ===");
        $display("Total tests: %0d", test_count);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        
        if (fail_count == 0) begin
            $display("All Vector Unit tests PASSED!");
        end else begin
            $display("Some Vector Unit tests FAILED!");
        end
        
        $finish;
    end

    // Test tasks
    task configure_vector_unit();
        $display("\n--- Configuring Vector Unit ---");
        
        // Set vtype: SEW=32, LMUL=1
        vtype = 64'h0000_0000_0000_0002; // SEW=32 (bits 5:3 = 010), LMUL=1 (bits 2:0 = 000)
        vl = 16; // Vector length = 16 elements
        
        $display("Vector configuration: SEW=32, LMUL=1, VL=16");
    endtask

    task test_vector_arithmetic();
        $display("\n--- Testing Vector Arithmetic Operations ---");
        
        // Test VADD.VV: Vector-vector addition
        test_vector_op(6'b010000, 3'b000, 6'b000000, 1, 5'd1, 5'd2, 5'd3, 
                      "VADD.VV v3,v1,v2");
        
        // Test VADD.VX: Vector-scalar addition
        test_vector_op(6'b010001, 3'b100, 6'b000000, 1, 5'd0, 5'd2, 5'd3,
                      "VADD.VX v3,v2,x0");
        
        // Test VADD.VI: Vector-immediate addition
        test_vector_op(6'b010010, 3'b011, 6'b000000, 1, 5'd0, 5'd2, 5'd3,
                      "VADD.VI v3,v2,5");
        
        // Test VSUB.VV: Vector-vector subtraction
        test_vector_op(6'b010000, 3'b000, 6'b000010, 1, 5'd1, 5'd2, 5'd3,
                      "VSUB.VV v3,v1,v2");
        
        // Test VMUL.VV: Vector-vector multiplication
        test_vector_op(6'b010000, 3'b000, 6'b100000, 1, 5'd1, 5'd2, 5'd3,
                      "VMUL.VV v3,v1,v2");
        
        // Test VAND.VV: Vector-vector bitwise AND
        test_vector_op(6'b010000, 3'b000, 6'b000100, 1, 5'd1, 5'd2, 5'd3,
                      "VAND.VV v3,v1,v2");
        
        // Test VOR.VV: Vector-vector bitwise OR
        test_vector_op(6'b010000, 3'b000, 6'b000101, 1, 5'd1, 5'd2, 5'd3,
                      "VOR.VV v3,v1,v2");
        
        // Test VXOR.VV: Vector-vector bitwise XOR
        test_vector_op(6'b010000, 3'b000, 6'b000110, 1, 5'd1, 5'd2, 5'd3,
                      "VXOR.VV v3,v1,v2");
    endtask

    task test_vector_memory();
        $display("\n--- Testing Vector Memory Operations ---");
        
        // Test VLE: Vector unit-stride load
        test_vector_mem_op(6'b000000, 3'b000, 5'd0, 5'd0, 5'd1, 64'h1000,
                          "VLE32.V v1,0(x0)");
        
        // Test VSE: Vector unit-stride store
        test_vector_mem_op(6'b000100, 3'b000, 5'd0, 5'd1, 5'd0, 64'h2000,
                          "VSE32.V v1,0(x0)");
        
        // Test VLSE: Vector strided load
        test_vector_mem_op(6'b000010, 3'b000, 5'd2, 5'd0, 5'd1, 64'h1000,
                          "VLSE32.V v1,0(x0),x2");
        
        // Test VSSE: Vector strided store
        test_vector_mem_op(6'b000110, 3'b000, 5'd2, 5'd1, 5'd0, 64'h2000,
                          "VSSE32.V v1,0(x0),x2");
    endtask

    task test_vector_config();
        $display("\n--- Testing Vector Configuration ---");
        
        // Test VSETVL: Set vector length
        test_vector_config_op(6'b110000, 3'b111, 5'd1, 5'd2, 5'd3,
                             "VSETVL x3,x1,x2");
    endtask

    task test_vector_op(
        input [5:0] opcode,
        input [2:0] f3,
        input [5:0] f6,
        input mask,
        input [4:0] s1,
        input [4:0] s2,
        input [4:0] dest,
        input string test_name
    );
        test_count++;
        
        // Set inputs
        vec_opcode = opcode;
        funct3 = f3;
        funct6 = f6;
        vm = mask;
        vs1 = s1;
        vs2 = s2;
        vd = dest;
        rs1_data = 64'h5; // Scalar operand for VX operations
        imm = 11'd5;      // Immediate operand for VI operations
        vec_enable = 1;
        
        // Wait for operation to start
        @(posedge clk);
        vec_enable = 0;
        
        // Wait for completion
        wait(vec_valid);
        @(posedge clk);
        
        // For now, just check that operation completed
        if (vec_valid) begin
            $display("PASS: %s - Operation completed", test_name);
            pass_count++;
        end else begin
            $display("FAIL: %s - Operation did not complete", test_name);
            fail_count++;
        end
        
        // Wait a cycle before next test
        @(posedge clk);
    endtask

    task test_vector_mem_op(
        input [5:0] opcode,
        input [2:0] f3,
        input [4:0] s1,
        input [4:0] s2,
        input [4:0] dest,
        input [XLEN-1:0] addr,
        input string test_name
    );
        test_count++;
        
        // Set inputs
        vec_opcode = opcode;
        funct3 = f3;
        vs1 = s1;
        vs2 = s2;
        vd = dest;
        rs1_data = addr;
        vec_enable = 1;
        
        // Wait for operation to start
        @(posedge clk);
        vec_enable = 0;
        
        // Wait for completion
        wait(vec_valid);
        @(posedge clk);
        
        // Check that memory operation completed
        if (vec_valid) begin
            $display("PASS: %s - Memory operation completed", test_name);
            pass_count++;
        end else begin
            $display("FAIL: %s - Memory operation did not complete", test_name);
            fail_count++;
        end
        
        // Wait a cycle before next test
        @(posedge clk);
    endtask

    task test_vector_config_op(
        input [5:0] opcode,
        input [2:0] f3,
        input [4:0] s1,
        input [4:0] s2,
        input [4:0] dest,
        input string test_name
    );
        test_count++;
        
        // Set inputs
        vec_opcode = opcode;
        funct3 = f3;
        vs1 = s1;
        vs2 = s2;
        vd = dest;
        rs1_data = 32; // Requested vector length
        vec_enable = 1;
        
        // Wait for operation
        @(posedge clk);
        vec_enable = 0;
        
        // Wait for completion
        wait(vec_valid);
        @(posedge clk);
        
        // Check that configuration completed and returned a valid VL
        if (vec_valid && (vec_result <= 32)) begin
            $display("PASS: %s - Returned VL: %0d", test_name, vec_result);
            pass_count++;
        end else begin
            $display("FAIL: %s - Invalid VL returned: %0d", test_name, vec_result);
            fail_count++;
        end
        
        // Wait a cycle before next test
        @(posedge clk);
    endtask

endmodule