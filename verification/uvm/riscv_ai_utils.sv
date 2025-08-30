// RISC-V AI Utilities
// Utility functions and classes for AI accelerator verification

`ifndef RISCV_AI_UTILS_SV
`define RISCV_AI_UTILS_SV

// Utility class for common verification functions
class riscv_ai_utils;
    
    // Convert operation type to string
    static function string op_type_to_string(operation_type_e op);
        case (op)
            READ_OP: return "READ";
            WRITE_OP: return "WRITE";
            AI_MATMUL_OP: return "MATMUL";
            AI_CONV2D_OP: return "CONV2D";
            AI_RELU_OP: return "RELU";
            AI_SIGMOID_OP: return "SIGMOID";
            AI_MAXPOOL_OP: return "MAXPOOL";
            AI_AVGPOOL_OP: return "AVGPOOL";
            AI_BATCHNORM_OP: return "BATCHNORM";
            default: return "UNKNOWN";
        endcase
    endfunction
    
    // Convert data type to string
    static function string data_type_to_string(data_type_e dtype);
        case (dtype)
            INT8_TYPE: return "INT8";
            INT16_TYPE: return "INT16";
            INT32_TYPE: return "INT32";
            FP16_TYPE: return "FP16";
            FP32_TYPE: return "FP32";
            FP64_TYPE: return "FP64";
            default: return "UNKNOWN";
        endcase
    endfunction
    
    // Get data type size in bytes
    static function int get_data_type_size(data_type_e dtype);
        case (dtype)
            INT8_TYPE: return 1;
            INT16_TYPE: return 2;
            INT32_TYPE: return 4;
            FP16_TYPE: return 2;
            FP32_TYPE: return 4;
            FP64_TYPE: return 8;
            default: return 4;
        endcase
    endfunction
    
    // Calculate matrix multiplication FLOPS
    static function longint calculate_matmul_flops(int m, int n, int k);
        return longint'(m) * longint'(n) * longint'(k) * 2;  // 2 operations per MAC
    endfunction
    
    // Calculate convolution FLOPS
    static function longint calculate_conv2d_flops(int oh, int ow, int ic, int oc, int kh, int kw);
        return longint'(oh) * longint'(ow) * longint'(ic) * longint'(oc) * longint'(kh) * longint'(kw) * 2;
    endfunction
    
    // Generate random matrix dimensions with constraints
    static function void randomize_matrix_dims(ref int m, int n, int k, int min_size = 8, int max_size = 1024);
        m = $urandom_range(min_size, max_size);
        n = $urandom_range(min_size, max_size);
        k = $urandom_range(min_size, max_size);
        
        // Ensure dimensions are multiples of 8 for better performance
        m = (m / 8) * 8;
        n = (n / 8) * 8;
        k = (k / 8) * 8;
        
        // Ensure minimum size
        if (m < min_size) m = min_size;
        if (n < min_size) n = min_size;
        if (k < min_size) k = min_size;
    endfunction
    
    // Generate random convolution parameters
    static function void randomize_conv_params(ref int h, int w, int c, int ksize, int stride, int padding);
        // Common image sizes
        int heights[] = '{32, 64, 128, 224, 256, 512};
        int widths[] = '{32, 64, 128, 224, 256, 512};
        int channels[] = '{3, 16, 32, 64, 128, 256, 512};
        int kernel_sizes[] = '{1, 3, 5, 7};
        int strides[] = '{1, 2, 4};
        int paddings[] = '{0, 1, 2, 3};
        
        h = heights[$urandom_range(0, heights.size()-1)];
        w = widths[$urandom_range(0, widths.size()-1)];
        c = channels[$urandom_range(0, channels.size()-1)];
        ksize = kernel_sizes[$urandom_range(0, kernel_sizes.size()-1)];
        stride = strides[$urandom_range(0, strides.size()-1)];
        padding = paddings[$urandom_range(0, paddings.size()-1)];
    endfunction
    
    // Calculate output dimensions for convolution
    static function void calculate_conv_output_dims(int ih, int iw, int ksize, int stride, int padding, 
                                                   ref int oh, int ow);
        oh = (ih + 2*padding - ksize) / stride + 1;
        ow = (iw + 2*padding - ksize) / stride + 1;
    endfunction
    
    // Validate address alignment
    static function bit is_address_aligned(bit [63:0] addr, int size);
        case (size)
            1: return 1;  // Byte access always aligned
            2: return (addr[0] == 0);  // Halfword alignment
            4: return (addr[1:0] == 0);  // Word alignment
            8: return (addr[2:0] == 0);  // Doubleword alignment
            default: return (addr[5:0] == 0);  // Cache line alignment for larger sizes
        endcase
    endfunction
    
    // Generate aligned address
    static function bit [63:0] generate_aligned_address(bit [63:0] base_addr, int size);
        bit [63:0] aligned_addr = base_addr;
        case (size)
            2: aligned_addr[0] = 0;
            4: aligned_addr[1:0] = 0;
            8: aligned_addr[2:0] = 0;
            default: aligned_addr[5:0] = 0;  // Cache line alignment
        endcase
        return aligned_addr;
    endfunction
    
    // Calculate bandwidth in MB/s
    static function real calculate_bandwidth_mbps(int bytes_transferred, time duration_ns);
        if (duration_ns == 0) return 0.0;
        real seconds = real'(duration_ns) / 1e9;
        real megabytes = real'(bytes_transferred) / (1024.0 * 1024.0);
        return megabytes / seconds;
    endfunction
    
    // Calculate TOPS (Tera Operations Per Second)
    static function real calculate_tops(longint operations, time duration_ns);
        if (duration_ns == 0) return 0.0;
        real seconds = real'(duration_ns) / 1e9;
        real tera_ops = real'(operations) / 1e12;
        return tera_ops / seconds;
    endfunction
    
    // Format large numbers with units
    static function string format_number_with_units(longint number);
        if (number >= 1000000000000) begin  // Tera
            return $sformatf("%.2fT", real'(number) / 1e12);
        end else if (number >= 1000000000) begin  // Giga
            return $sformatf("%.2fG", real'(number) / 1e9);
        end else if (number >= 1000000) begin  // Mega
            return $sformatf("%.2fM", real'(number) / 1e6);
        end else if (number >= 1000) begin  // Kilo
            return $sformatf("%.2fK", real'(number) / 1e3);
        end else begin
            return $sformatf("%0d", number);
        end
    endfunction
    
    // Format time with units
    static function string format_time_with_units(time time_ns);
        if (time_ns >= 1000000000) begin  // Seconds
            return $sformatf("%.3fs", real'(time_ns) / 1e9);
        end else if (time_ns >= 1000000) begin  // Milliseconds
            return $sformatf("%.3fms", real'(time_ns) / 1e6);
        end else if (time_ns >= 1000) begin  // Microseconds
            return $sformatf("%.3fus", real'(time_ns) / 1e3);
        end else begin  // Nanoseconds
            return $sformatf("%0dns", time_ns);
        end
    endfunction
    
    // Generate test data patterns
    static function bit [63:0] generate_test_pattern(int pattern_type, int index = 0);
        case (pattern_type)
            0: return 64'h0;  // All zeros
            1: return 64'hFFFFFFFFFFFFFFFF;  // All ones
            2: return 64'hAAAAAAAAAAAAAAAA;  // Alternating pattern
            3: return 64'h5555555555555555;  // Alternating pattern
            4: return {32'h12345678, 32'h9ABCDEF0};  // Walking pattern
            5: return 64'h0123456789ABCDEF;  // Incremental pattern
            6: return $urandom_range(0, 64'hFFFFFFFFFFFFFFFF);  // Random
            7: return {index[31:0], ~index[31:0]};  // Index-based pattern
            default: return 64'h0;
        endcase
    endfunction
    
    // Compare floating point values with tolerance
    static function bit compare_fp_with_tolerance(bit [63:0] expected, bit [63:0] actual, 
                                                 data_type_e dtype, real tolerance_pct = 0.1);
        real exp_val, act_val, diff, tolerance;
        
        case (dtype)
            FP16_TYPE: begin
                // Simplified FP16 comparison (would need proper IEEE 754 handling)
                exp_val = $itor(expected[15:0]);
                act_val = $itor(actual[15:0]);
            end
            FP32_TYPE: begin
                // Simplified FP32 comparison (would need proper IEEE 754 handling)
                exp_val = $itor(expected[31:0]);
                act_val = $itor(actual[31:0]);
            end
            FP64_TYPE: begin
                // Simplified FP64 comparison (would need proper IEEE 754 handling)
                exp_val = $itor(expected);
                act_val = $itor(actual);
            end
            default: begin
                // Integer comparison - exact match required
                return (expected == actual);
            end
        endcase
        
        diff = (exp_val > act_val) ? (exp_val - act_val) : (act_val - exp_val);
        tolerance = (exp_val * tolerance_pct / 100.0);
        
        return (diff <= tolerance);
    endfunction
    
    // Generate memory access pattern
    static function void generate_memory_pattern(ref bit [63:0] addresses[], int pattern_type, 
                                                bit [63:0] base_addr, int num_accesses, int stride = 64);
        addresses = new[num_accesses];
        
        case (pattern_type)
            0: begin  // Sequential
                for (int i = 0; i < num_accesses; i++) begin
                    addresses[i] = base_addr + (i * stride);
                end
            end
            1: begin  // Random
                for (int i = 0; i < num_accesses; i++) begin
                    addresses[i] = base_addr + ($urandom_range(0, num_accesses-1) * stride);
                end
            end
            2: begin  // Strided
                int large_stride = stride * 16;
                for (int i = 0; i < num_accesses; i++) begin
                    addresses[i] = base_addr + (i * large_stride);
                end
            end
            3: begin  // Reverse sequential
                for (int i = 0; i < num_accesses; i++) begin
                    addresses[i] = base_addr + ((num_accesses - 1 - i) * stride);
                end
            end
            default: begin  // Sequential (default)
                for (int i = 0; i < num_accesses; i++) begin
                    addresses[i] = base_addr + (i * stride);
                end
            end
        endcase
    endfunction
    
    // Print transaction summary
    static function void print_transaction_summary(riscv_ai_sequence_item item);
        string summary;
        summary = $sformatf("Op: %s, DataType: %s, Core: %0d", 
                           op_type_to_string(item.op_type), 
                           data_type_to_string(item.data_type), 
                           item.core_id);
        
        case (item.op_type)
            AI_MATMUL_OP: begin
                summary = {summary, $sformatf(", Matrix: %0dx%0dx%0d", item.matrix_m, item.matrix_n, item.matrix_k)};
            end
            AI_CONV2D_OP: begin
                summary = {summary, $sformatf(", Conv: %0dx%0dx%0d, K:%0d, S:%0d", 
                          item.conv_height, item.conv_width, item.conv_channels, item.kernel_size, item.stride)};
            end
            READ_OP, WRITE_OP: begin
                summary = {summary, $sformatf(", Addr: 0x%016h, Size: %0d", item.addr, item.size)};
            end
        endcase
        
        if (item.latency > 0) begin
            summary = {summary, $sformatf(", Latency: %s", format_time_with_units(item.latency))};
        end
        
        $display("Transaction: %s", summary);
    endfunction
    
endclass : riscv_ai_utils

// Performance analysis utility class
class riscv_ai_perf_analyzer;
    
    // Performance counters
    longint total_operations = 0;
    longint total_bytes_transferred = 0;
    time total_execution_time = 0;
    
    // Operation-specific counters
    longint matmul_operations = 0;
    longint conv2d_operations = 0;
    longint activation_operations = 0;
    longint memory_operations = 0;
    
    // Performance metrics
    real peak_tops = 0.0;
    real average_tops = 0.0;
    real peak_bandwidth_mbps = 0.0;
    real average_bandwidth_mbps = 0.0;
    
    // Add performance sample
    function void add_sample(riscv_ai_sequence_item item);
        longint ops = 0;
        int bytes = 0;
        
        // Calculate operations and bytes for this transaction
        case (item.op_type)
            AI_MATMUL_OP: begin
                ops = riscv_ai_utils::calculate_matmul_flops(item.matrix_m, item.matrix_n, item.matrix_k);
                bytes = (item.matrix_m * item.matrix_k + item.matrix_k * item.matrix_n + item.matrix_m * item.matrix_n) * 
                        riscv_ai_utils::get_data_type_size(item.data_type);
                matmul_operations++;
            end
            AI_CONV2D_OP: begin
                int oh, ow;
                riscv_ai_utils::calculate_conv_output_dims(item.conv_height, item.conv_width, 
                                                          item.kernel_size, item.stride, item.padding, oh, ow);
                ops = riscv_ai_utils::calculate_conv2d_flops(oh, ow, item.conv_channels, item.conv_channels, 
                                                            item.kernel_size, item.kernel_size);
                bytes = (item.conv_height * item.conv_width * item.conv_channels + oh * ow * item.conv_channels) * 
                        riscv_ai_utils::get_data_type_size(item.data_type);
                conv2d_operations++;
            end
            AI_RELU_OP, AI_SIGMOID_OP: begin
                ops = item.size / riscv_ai_utils::get_data_type_size(item.data_type);
                bytes = item.size * 2;  // Input + output
                activation_operations++;
            end
            READ_OP, WRITE_OP: begin
                ops = 1;  // One memory operation
                bytes = item.size;
                memory_operations++;
            end
        endcase
        
        total_operations += ops;
        total_bytes_transferred += bytes;
        total_execution_time += item.latency;
        
        // Calculate instantaneous performance
        if (item.latency > 0) begin
            real inst_tops = riscv_ai_utils::calculate_tops(ops, item.latency);
            real inst_bandwidth = riscv_ai_utils::calculate_bandwidth_mbps(bytes, item.latency);
            
            if (inst_tops > peak_tops) peak_tops = inst_tops;
            if (inst_bandwidth > peak_bandwidth_mbps) peak_bandwidth_mbps = inst_bandwidth;
        end
        
        // Update averages
        if (total_execution_time > 0) begin
            average_tops = riscv_ai_utils::calculate_tops(total_operations, total_execution_time);
            average_bandwidth_mbps = riscv_ai_utils::calculate_bandwidth_mbps(total_bytes_transferred, total_execution_time);
        end
    endfunction
    
    // Print performance report
    function void print_report();
        $display("=== PERFORMANCE ANALYSIS REPORT ===");
        $display("Total Operations: %s", riscv_ai_utils::format_number_with_units(total_operations));
        $display("Total Bytes Transferred: %s", riscv_ai_utils::format_number_with_units(total_bytes_transferred));
        $display("Total Execution Time: %s", riscv_ai_utils::format_time_with_units(total_execution_time));
        $display("");
        $display("Operation Breakdown:");
        $display("  Matrix Multiplications: %0d", matmul_operations);
        $display("  Convolutions: %0d", conv2d_operations);
        $display("  Activations: %0d", activation_operations);
        $display("  Memory Operations: %0d", memory_operations);
        $display("");
        $display("Performance Metrics:");
        $display("  Peak TOPS: %.3f", peak_tops);
        $display("  Average TOPS: %.3f", average_tops);
        $display("  Peak Bandwidth: %.2f MB/s", peak_bandwidth_mbps);
        $display("  Average Bandwidth: %.2f MB/s", average_bandwidth_mbps);
    endfunction
    
endclass : riscv_ai_perf_analyzer

`endif // RISCV_AI_UTILS_SV