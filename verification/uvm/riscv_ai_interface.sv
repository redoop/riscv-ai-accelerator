// RISC-V AI Interface
// SystemVerilog interface for AI accelerator verification

`ifndef RISCV_AI_INTERFACE_SV
`define RISCV_AI_INTERFACE_SV

interface riscv_ai_interface(input logic clk, input logic rst_n);
    
    // Basic control signals
    logic valid;
    logic ready;
    
    // Address and data
    logic [63:0] addr;
    logic [63:0] data;
    logic [31:0] size;
    
    // Operation control
    logic [2:0] op_type;
    logic [2:0] data_type;
    
    // Core/Unit identification
    logic [7:0] core_id;
    logic [7:0] tpu_id;
    logic [7:0] vpu_id;
    
    // AI-specific parameters
    logic [31:0] matrix_m, matrix_n, matrix_k;
    logic [31:0] conv_height, conv_width, conv_channels;
    logic [31:0] kernel_size, stride, padding;
    logic [31:0] pool_size;
    
    // Response signals
    logic response_valid;
    logic [63:0] response_data;
    logic error;
    
    // Performance monitoring signals
    logic [31:0] performance_counter;
    logic [15:0] power_consumption;
    logic [7:0] temperature;
    
    // Debug signals
    logic [31:0] debug_state;
    logic [63:0] debug_pc;
    logic debug_halt;
    
    // Clocking blocks for different verification components
    
    // Driver clocking block
    clocking driver_cb @(posedge clk);
        default input #1step output #1step;
        output valid, addr, data, size, op_type, data_type;
        output core_id, tpu_id, vpu_id;
        output matrix_m, matrix_n, matrix_k;
        output conv_height, conv_width, conv_channels;
        output kernel_size, stride, padding, pool_size;
        input ready, response_valid, response_data, error;
        input performance_counter, power_consumption, temperature;
    endclocking
    
    // Monitor clocking block
    clocking monitor_cb @(posedge clk);
        default input #1step;
        input valid, ready, addr, data, size, op_type, data_type;
        input core_id, tpu_id, vpu_id;
        input matrix_m, matrix_n, matrix_k;
        input conv_height, conv_width, conv_channels;
        input kernel_size, stride, padding, pool_size;
        input response_valid, response_data, error;
        input performance_counter, power_consumption, temperature;
        input debug_state, debug_pc, debug_halt;
    endclocking
    
    // Modports for different verification components
    modport driver_mp (
        clocking driver_cb,
        input clk, rst_n
    );
    
    modport monitor_mp (
        clocking monitor_cb,
        input clk, rst_n
    );
    
    modport dut_mp (
        input clk, rst_n,
        input valid, addr, data, size, op_type, data_type,
        input core_id, tpu_id, vpu_id,
        input matrix_m, matrix_n, matrix_k,
        input conv_height, conv_width, conv_channels,
        input kernel_size, stride, padding, pool_size,
        output ready, response_valid, response_data, error,
        output performance_counter, power_consumption, temperature,
        output debug_state, debug_pc, debug_halt
    );
    
    // Assertions for protocol checking
    
    // Valid/Ready handshake assertions
    property valid_ready_handshake;
        @(posedge clk) disable iff (!rst_n)
        valid && !ready |=> valid;
    endproperty
    
    assert property (valid_ready_handshake) else
        $error("Valid deasserted before ready");
    
    // Response valid should follow request
    property response_follows_request;
        @(posedge clk) disable iff (!rst_n)
        valid && ready |-> ##[1:100] response_valid;
    endproperty
    
    assert property (response_follows_request) else
        $warning("No response received within 100 cycles");
    
    // Address alignment assertions
    property addr_alignment_64bit;
        @(posedge clk) disable iff (!rst_n)
        valid && ready && size == 8 |-> addr[2:0] == 3'b000;
    endproperty
    
    assert property (addr_alignment_64bit) else
        $error("64-bit access not 8-byte aligned");
    
    property addr_alignment_32bit;
        @(posedge clk) disable iff (!rst_n)
        valid && ready && size == 4 |-> addr[1:0] == 2'b00;
    endproperty
    
    assert property (addr_alignment_32bit) else
        $error("32-bit access not 4-byte aligned");
    
    // AI operation parameter validation
    property matmul_params_valid;
        @(posedge clk) disable iff (!rst_n)
        valid && ready && op_type == 3'b010 |-> 
        matrix_m > 0 && matrix_n > 0 && matrix_k > 0 &&
        matrix_m <= 1024 && matrix_n <= 1024 && matrix_k <= 1024;
    endproperty
    
    assert property (matmul_params_valid) else
        $error("Invalid matrix multiplication parameters");
    
    property conv2d_params_valid;
        @(posedge clk) disable iff (!rst_n)
        valid && ready && op_type == 3'b011 |-> 
        conv_height > 0 && conv_width > 0 && conv_channels > 0 &&
        kernel_size > 0 && stride > 0;
    endproperty
    
    assert property (conv2d_params_valid) else
        $error("Invalid convolution parameters");
    
    // Coverage for different operation types
    covergroup op_type_cg @(posedge clk);
        option.per_instance = 1;
        
        op_type_cp: coverpoint op_type iff (valid && ready) {
            bins read_op = {3'b000};
            bins write_op = {3'b001};
            bins matmul_op = {3'b010};
            bins conv2d_op = {3'b011};
            bins relu_op = {3'b100};
            bins sigmoid_op = {3'b101};
            bins maxpool_op = {3'b110};
            bins avgpool_op = {3'b111};
        }
        
        data_type_cp: coverpoint data_type iff (valid && ready) {
            bins int8 = {3'b000};
            bins int16 = {3'b001};
            bins int32 = {3'b010};
            bins fp16 = {3'b011};
            bins fp32 = {3'b100};
            bins fp64 = {3'b101};
        }
        
        core_id_cp: coverpoint core_id iff (valid && ready) {
            bins core0 = {0};
            bins core1 = {1};
            bins core2 = {2};
            bins core3 = {3};
        }
        
        tpu_id_cp: coverpoint tpu_id iff (valid && ready && op_type inside {3'b010, 3'b011}) {
            bins tpu0 = {0};
            bins tpu1 = {1};
        }
        
        vpu_id_cp: coverpoint vpu_id iff (valid && ready && op_type inside {3'b100, 3'b101, 3'b110, 3'b111}) {
            bins vpu0 = {0};
            bins vpu1 = {1};
        }
        
        // Cross coverage
        op_data_cross: cross op_type_cp, data_type_cp;
        op_core_cross: cross op_type_cp, core_id_cp;
        tpu_data_cross: cross tpu_id_cp, data_type_cp;
        vpu_data_cross: cross vpu_id_cp, data_type_cp;
    endgroup
    
    // Coverage for matrix dimensions
    covergroup matmul_cg @(posedge clk);
        option.per_instance = 1;
        
        matrix_m_cp: coverpoint matrix_m iff (valid && ready && op_type == 3'b010) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        matrix_n_cp: coverpoint matrix_n iff (valid && ready && op_type == 3'b010) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        matrix_k_cp: coverpoint matrix_k iff (valid && ready && op_type == 3'b010) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        matmul_dims_cross: cross matrix_m_cp, matrix_n_cp, matrix_k_cp;
    endgroup
    
    // Coverage for convolution parameters
    covergroup conv2d_cg @(posedge clk);
        option.per_instance = 1;
        
        conv_height_cp: coverpoint conv_height iff (valid && ready && op_type == 3'b011) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
        }
        
        conv_width_cp: coverpoint conv_width iff (valid && ready && op_type == 3'b011) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
        }
        
        kernel_size_cp: coverpoint kernel_size iff (valid && ready && op_type == 3'b011) {
            bins k1 = {1};
            bins k3 = {3};
            bins k5 = {5};
            bins k7 = {7};
        }
        
        stride_cp: coverpoint stride iff (valid && ready && op_type == 3'b011) {
            bins s1 = {1};
            bins s2 = {2};
            bins s4 = {4};
        }
        
        conv_params_cross: cross kernel_size_cp, stride_cp;
    endgroup
    
    // Instantiate coverage groups
    op_type_cg op_type_cg_inst = new();
    matmul_cg matmul_cg_inst = new();
    conv2d_cg conv2d_cg_inst = new();
    
    // Utility tasks for testbench
    task wait_for_ready();
        @(posedge clk iff ready);
    endtask
    
    task wait_for_response();
        @(posedge clk iff response_valid);
    endtask
    
    task reset_interface();
        valid <= 1'b0;
        addr <= 64'h0;
        data <= 64'h0;
        size <= 32'h0;
        op_type <= 3'b000;
        data_type <= 3'b000;
        core_id <= 8'h0;
        tpu_id <= 8'h0;
        vpu_id <= 8'h0;
        matrix_m <= 32'h0;
        matrix_n <= 32'h0;
        matrix_k <= 32'h0;
        conv_height <= 32'h0;
        conv_width <= 32'h0;
        conv_channels <= 32'h0;
        kernel_size <= 32'h0;
        stride <= 32'h0;
        padding <= 32'h0;
        pool_size <= 32'h0;
    endtask
    
    // Initialize interface
    initial begin
        reset_interface();
    end
    
endinterface : riscv_ai_interface

`endif // RISCV_AI_INTERFACE_SV