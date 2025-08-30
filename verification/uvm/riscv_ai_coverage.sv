// RISC-V AI Coverage Collector
// Comprehensive functional and code coverage collection

`ifndef RISCV_AI_COVERAGE_SV
`define RISCV_AI_COVERAGE_SV

class riscv_ai_coverage extends uvm_subscriber #(riscv_ai_sequence_item);
    
    // Coverage groups
    covergroup operation_coverage;
        option.per_instance = 1;
        option.name = "operation_coverage";
        
        // Operation type coverage
        op_type: coverpoint subscriber_item.op_type {
            bins memory_ops[] = {READ_OP, WRITE_OP};
            bins ai_compute_ops[] = {AI_MATMUL_OP, AI_CONV2D_OP};
            bins ai_activation_ops[] = {AI_RELU_OP, AI_SIGMOID_OP};
            bins ai_pooling_ops[] = {AI_MAXPOOL_OP, AI_AVGPOOL_OP};
            bins ai_norm_ops[] = {AI_BATCHNORM_OP};
        }
        
        // Data type coverage
        data_type: coverpoint subscriber_item.data_type {
            bins integer_types[] = {INT8_TYPE, INT16_TYPE, INT32_TYPE};
            bins float_types[] = {FP16_TYPE, FP32_TYPE, FP64_TYPE};
        }
        
        // Core ID coverage
        core_id: coverpoint subscriber_item.core_id {
            bins cores[] = {[0:3]};
        }
        
        // TPU ID coverage
        tpu_id: coverpoint subscriber_item.tpu_id iff (subscriber_item.op_type inside {AI_MATMUL_OP, AI_CONV2D_OP}) {
            bins tpus[] = {[0:1]};
        }
        
        // VPU ID coverage
        vpu_id: coverpoint subscriber_item.vpu_id iff (subscriber_item.op_type inside {AI_RELU_OP, AI_SIGMOID_OP, AI_MAXPOOL_OP, AI_AVGPOOL_OP, AI_BATCHNORM_OP}) {
            bins vpus[] = {[0:1]};
        }
        
        // Cross coverage
        op_data_cross: cross op_type, data_type {
            // Exclude invalid combinations
            ignore_bins invalid_int_activation = binsof(op_type) intersect {AI_RELU_OP, AI_SIGMOID_OP} &&
                                                 binsof(data_type) intersect {INT8_TYPE, INT16_TYPE, INT32_TYPE};
        }
        
        op_core_cross: cross op_type, core_id;
        op_tpu_cross: cross op_type, tpu_id;
        op_vpu_cross: cross op_type, vpu_id;
    endgroup
    
    covergroup matrix_coverage;
        option.per_instance = 1;
        option.name = "matrix_coverage";
        
        // Matrix dimensions coverage
        matrix_m: coverpoint subscriber_item.matrix_m iff (subscriber_item.op_type == AI_MATMUL_OP) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        matrix_n: coverpoint subscriber_item.matrix_n iff (subscriber_item.op_type == AI_MATMUL_OP) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        matrix_k: coverpoint subscriber_item.matrix_k iff (subscriber_item.op_type == AI_MATMUL_OP) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:512]};
            bins xlarge = {[513:1024]};
        }
        
        // Matrix shape categories
        matrix_shape: coverpoint {subscriber_item.matrix_m, subscriber_item.matrix_n} iff (subscriber_item.op_type == AI_MATMUL_OP) {
            bins square_small = {[1:32], [1:32]};
            bins square_medium = {[33:128], [33:128]};
            bins square_large = {[129:512], [129:512]};
            bins tall_matrix = {[129:1024], [1:128]};
            bins wide_matrix = {[1:128], [129:1024]};
        }
        
        // Cross coverage for matrix dimensions
        matrix_dims_cross: cross matrix_m, matrix_n, matrix_k;
        matrix_data_type_cross: cross matrix_shape, subscriber_item.data_type;
    endgroup
    
    covergroup convolution_coverage;
        option.per_instance = 1;
        option.name = "convolution_coverage";
        
        // Input tensor dimensions
        conv_height: coverpoint subscriber_item.conv_height iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:256]};
            bins xlarge = {[257:512]};
        }
        
        conv_width: coverpoint subscriber_item.conv_width iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins small = {[1:32]};
            bins medium = {[33:128]};
            bins large = {[129:256]};
            bins xlarge = {[257:512]};
        }
        
        conv_channels: coverpoint subscriber_item.conv_channels iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins few = {[1:16]};
            bins moderate = {[17:64]};
            bins many = {[65:256]};
            bins very_many = {[257:512]};
        }
        
        // Kernel parameters
        kernel_size: coverpoint subscriber_item.kernel_size iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins k1 = {1};
            bins k3 = {3};
            bins k5 = {5};
            bins k7 = {7};
            bins k9 = {9};
            bins k11 = {11};
        }
        
        stride: coverpoint subscriber_item.stride iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins s1 = {1};
            bins s2 = {2};
            bins s3 = {3};
            bins s4 = {4};
        }
        
        padding: coverpoint subscriber_item.padding iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins no_pad = {0};
            bins small_pad = {[1:2]};
            bins medium_pad = {[3:5]};
        }
        
        // Common convolution configurations
        conv_config: coverpoint {subscriber_item.kernel_size, subscriber_item.stride, subscriber_item.padding} 
                     iff (subscriber_item.op_type == AI_CONV2D_OP) {
            bins conv3x3_s1_p1 = {3, 1, 1};
            bins conv3x3_s2_p1 = {3, 2, 1};
            bins conv1x1_s1_p0 = {1, 1, 0};
            bins conv5x5_s1_p2 = {5, 1, 2};
            bins conv7x7_s2_p3 = {7, 2, 3};
        }
        
        // Cross coverage
        conv_dims_cross: cross conv_height, conv_width, conv_channels;
        conv_params_cross: cross kernel_size, stride, padding;
        conv_data_type_cross: cross conv_config, subscriber_item.data_type;
    endgroup
    
    covergroup pooling_coverage;
        option.per_instance = 1;
        option.name = "pooling_coverage";
        
        // Pool size coverage
        pool_size: coverpoint subscriber_item.pool_size iff (subscriber_item.op_type inside {AI_MAXPOOL_OP, AI_AVGPOOL_OP}) {
            bins pool2 = {2};
            bins pool3 = {3};
            bins pool4 = {4};
            bins pool8 = {8};
        }
        
        // Pool type coverage
        pool_type: coverpoint subscriber_item.op_type iff (subscriber_item.op_type inside {AI_MAXPOOL_OP, AI_AVGPOOL_OP}) {
            bins max_pool = {AI_MAXPOOL_OP};
            bins avg_pool = {AI_AVGPOOL_OP};
        }
        
        // Input dimensions for pooling
        pool_input_height: coverpoint subscriber_item.conv_height iff (subscriber_item.op_type inside {AI_MAXPOOL_OP, AI_AVGPOOL_OP}) {
            bins small = {[2:32]};
            bins medium = {[33:128]};
            bins large = {[129:256]};
        }
        
        pool_input_width: coverpoint subscriber_item.conv_width iff (subscriber_item.op_type inside {AI_MAXPOOL_OP, AI_AVGPOOL_OP}) {
            bins small = {[2:32]};
            bins medium = {[33:128]};
            bins large = {[129:256]};
        }
        
        // Cross coverage
        pool_config_cross: cross pool_type, pool_size;
        pool_dims_cross: cross pool_input_height, pool_input_width, pool_size;
    endgroup
    
    covergroup memory_coverage;
        option.per_instance = 1;
        option.name = "memory_coverage";
        
        // Address coverage
        address: coverpoint subscriber_item.addr iff (subscriber_item.op_type inside {READ_OP, WRITE_OP}) {
            bins low_mem = {[64'h0000_0000_0000_0000:64'h0000_0000_FFFF_FFFF]};
            bins mid_mem = {[64'h0000_0001_0000_0000:64'h0000_FFFF_FFFF_FFFF]};
            bins high_mem = {[64'h0001_0000_0000_0000:64'hFFFF_FFFF_FFFF_FFFF]};
        }
        
        // Size coverage
        access_size: coverpoint subscriber_item.size iff (subscriber_item.op_type inside {READ_OP, WRITE_OP}) {
            bins byte_access = {1};
            bins halfword_access = {2};
            bins word_access = {4};
            bins doubleword_access = {8};
            bins cache_line = {64};
            bins large_transfer[] = {[128:4096]};
        }
        
        // Address alignment
        alignment: coverpoint subscriber_item.addr[5:0] iff (subscriber_item.op_type inside {READ_OP, WRITE_OP}) {
            bins aligned_64 = {6'b000000};
            bins aligned_32 = {6'b100000};
            bins aligned_16 = {6'b010000, 6'b110000};
            bins aligned_8 = {6'b001000, 6'b011000, 6'b101000, 6'b111000};
            bins unaligned[] = default;
        }
        
        // Memory operation type
        mem_op: coverpoint subscriber_item.op_type iff (subscriber_item.op_type inside {READ_OP, WRITE_OP}) {
            bins read = {READ_OP};
            bins write = {WRITE_OP};
        }
        
        // Cross coverage
        mem_size_align_cross: cross access_size, alignment;
        mem_addr_size_cross: cross address, access_size;
        mem_op_size_cross: cross mem_op, access_size;
    endgroup
    
    covergroup performance_coverage;
        option.per_instance = 1;
        option.name = "performance_coverage";
        
        // Latency coverage
        latency: coverpoint subscriber_item.latency {
            bins very_fast = {[1:10]};
            bins fast = {[11:50]};
            bins moderate = {[51:200]};
            bins slow = {[201:1000]};
            bins very_slow = {[1001:$]};
        }
        
        // Throughput coverage
        throughput: coverpoint subscriber_item.throughput_mbps {
            bins low = {[1:100]};
            bins medium = {[101:1000]};
            bins high = {[1001:10000]};
            bins very_high = {[10001:$]};
        }
        
        // Power consumption coverage
        power: coverpoint subscriber_item.power_consumption_mw {
            bins low_power = {[1:100]};
            bins medium_power = {[101:500]};
            bins high_power = {[501:1000]};
            bins very_high_power = {[1001:$]};
        }
        
        // Cross coverage
        latency_throughput_cross: cross latency, throughput;
        power_throughput_cross: cross power, throughput;
        op_performance_cross: cross subscriber_item.op_type, latency, throughput;
    endgroup
    
    covergroup error_coverage;
        option.per_instance = 1;
        option.name = "error_coverage";
        
        // Error occurrence
        error_status: coverpoint subscriber_item.error {
            bins no_error = {0};
            bins error_occurred = {1};
        }
        
        // Error by operation type
        error_by_op: coverpoint subscriber_item.op_type iff (subscriber_item.error) {
            bins memory_errors[] = {READ_OP, WRITE_OP};
            bins ai_compute_errors[] = {AI_MATMUL_OP, AI_CONV2D_OP};
            bins ai_activation_errors[] = {AI_RELU_OP, AI_SIGMOID_OP};
            bins ai_pooling_errors[] = {AI_MAXPOOL_OP, AI_AVGPOOL_OP};
        }
        
        // Error injection coverage
        error_injection: coverpoint subscriber_item.inject_error {
            bins no_injection = {0};
            bins injection_enabled = {1};
        }
        
        error_type: coverpoint subscriber_item.error_type iff (subscriber_item.inject_error) {
            bins error_types[] = {[1:8]};
        }
        
        // Cross coverage
        error_op_cross: cross error_status, subscriber_item.op_type;
        error_injection_cross: cross error_injection, error_type;
    endgroup
    
    // Coverage group instances
    operation_coverage op_cov;
    matrix_coverage mat_cov;
    convolution_coverage conv_cov;
    pooling_coverage pool_cov;
    memory_coverage mem_cov;
    performance_coverage perf_cov;
    error_coverage err_cov;
    
    // Coverage statistics
    real total_coverage = 0.0;
    real operation_coverage_pct = 0.0;
    real matrix_coverage_pct = 0.0;
    real convolution_coverage_pct = 0.0;
    real pooling_coverage_pct = 0.0;
    real memory_coverage_pct = 0.0;
    real performance_coverage_pct = 0.0;
    real error_coverage_pct = 0.0;
    
    `uvm_component_utils(riscv_ai_coverage)
    
    // Constructor
    function new(string name = "riscv_ai_coverage", uvm_component parent = null);
        super.new(name, parent);
        
        // Create coverage groups
        op_cov = new();
        mat_cov = new();
        conv_cov = new();
        pool_cov = new();
        mem_cov = new();
        perf_cov = new();
        err_cov = new();
    endfunction
    
    // Write method - called for each transaction
    virtual function void write(riscv_ai_sequence_item t);
        subscriber_item = t;
        
        // Sample all coverage groups
        op_cov.sample();
        mat_cov.sample();
        conv_cov.sample();
        pool_cov.sample();
        mem_cov.sample();
        perf_cov.sample();
        err_cov.sample();
        
        `uvm_info(get_type_name(), $sformatf("Coverage sampled for %s operation", t.op_type.name()), UVM_HIGH)
    endfunction
    
    // Extract phase - calculate coverage statistics
    virtual function void extract_phase(uvm_phase phase);
        super.extract_phase(phase);
        
        operation_coverage_pct = op_cov.get_inst_coverage();
        matrix_coverage_pct = mat_cov.get_inst_coverage();
        convolution_coverage_pct = conv_cov.get_inst_coverage();
        pooling_coverage_pct = pool_cov.get_inst_coverage();
        memory_coverage_pct = mem_cov.get_inst_coverage();
        performance_coverage_pct = perf_cov.get_inst_coverage();
        error_coverage_pct = err_cov.get_inst_coverage();
        
        // Calculate total coverage (weighted average)
        total_coverage = (operation_coverage_pct * 0.25 +
                         matrix_coverage_pct * 0.15 +
                         convolution_coverage_pct * 0.15 +
                         pooling_coverage_pct * 0.10 +
                         memory_coverage_pct * 0.15 +
                         performance_coverage_pct * 0.10 +
                         error_coverage_pct * 0.10);
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== COVERAGE REPORT ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Operation Coverage: %.2f%%", operation_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Matrix Coverage: %.2f%%", matrix_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Convolution Coverage: %.2f%%", convolution_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Pooling Coverage: %.2f%%", pooling_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Memory Coverage: %.2f%%", memory_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Performance Coverage: %.2f%%", performance_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Error Coverage: %.2f%%", error_coverage_pct), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("TOTAL COVERAGE: %.2f%%", total_coverage), UVM_LOW)
        
        // Coverage goals check
        if (total_coverage >= 95.0) begin
            `uvm_info(get_type_name(), "COVERAGE GOAL ACHIEVED (95%+)", UVM_LOW)
        end else if (total_coverage >= 90.0) begin
            `uvm_warning(get_type_name(), "Coverage close to goal (90-95%)")
        end else begin
            `uvm_error(get_type_name(), "Coverage below acceptable threshold (<90%)")
        end
    endfunction
    
    // Get coverage summary
    virtual function string get_coverage_summary();
        string summary;
        summary = $sformatf("Total: %.1f%% | Op: %.1f%% | Mat: %.1f%% | Conv: %.1f%% | Pool: %.1f%% | Mem: %.1f%% | Perf: %.1f%% | Err: %.1f%%",
                           total_coverage, operation_coverage_pct, matrix_coverage_pct, convolution_coverage_pct,
                           pooling_coverage_pct, memory_coverage_pct, performance_coverage_pct, error_coverage_pct);
        return summary;
    endfunction
    
endclass : riscv_ai_coverage

`endif // RISCV_AI_COVERAGE_SV