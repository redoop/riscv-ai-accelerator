// AI Benchmarks Testbench
// Top-level testbench for AI workload benchmarking

`timescale 1ns/1ps

`include "uvm_macros.svh"
import uvm_pkg::*;
import ai_benchmark_pkg::*;

module tb_ai_benchmarks;

    // Clock and reset generation
    logic clk = 0;
    logic rst_n = 0;
    
    // Clock generation - 1GHz system clock
    always #0.5 clk = ~clk;
    
    // Reset generation
    initial begin
        rst_n = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        `uvm_info("TB", "Reset released", UVM_LOW)
    end
    
    // Interface instantiation (reuse from UVM testbench)
    riscv_ai_interface ai_if(clk, rst_n);
    
    // Simplified DUT for benchmark testing
    riscv_ai_benchmark_dut dut (
        .clk(clk),
        .rst_n(rst_n),
        .ai_if(ai_if.dut_mp)
    );
    
    // UVM test execution
    initial begin
        // Set interface in config DB
        uvm_config_db#(virtual riscv_ai_interface)::set(null, "*", "vif", ai_if);
        
        // Configure benchmark suite from command line
        string benchmark_suite;
        if ($value$plusargs("BENCHMARK_SUITE=%s", benchmark_suite)) begin
            uvm_config_db#(string)::set(null, "*", "benchmark_suite", benchmark_suite);
            `uvm_info("TB", $sformatf("Running benchmark suite: %s", benchmark_suite), UVM_LOW)
        end else begin
            uvm_config_db#(string)::set(null, "*", "benchmark_suite", "mlperf_inference");
            `uvm_info("TB", "Using default benchmark suite: mlperf_inference", UVM_LOW)
        end
        
        // Start UVM test
        run_test("ai_benchmark_test");
    end
    
    // Extended timeout for benchmarks
    initial begin
        #1s;  // 1 second timeout
        `uvm_fatal("TB", "Benchmark timeout - simulation ran too long")
    end
    
    // Waveform dumping
    initial begin
        if ($test$plusargs("DUMP_WAVES")) begin
            $dumpfile("tb_ai_benchmarks.vcd");
            $dumpvars(0, tb_ai_benchmarks);
        end
    end

endmodule

// Simplified DUT for benchmark testing
module riscv_ai_benchmark_dut (
    input logic clk,
    input logic rst_n,
    riscv_ai_interface.dut_mp ai_if
);

    // Enhanced DUT with more realistic AI processing simulation
    logic [31:0] cycle_count = 0;
    logic [63:0] response_data_reg = 0;
    logic response_valid_reg = 0;
    logic error_reg = 0;
    logic ready_reg = 1;
    
    // Performance counters
    logic [31:0] perf_counter = 0;
    logic [15:0] power_consumption = 1000;  // 1W baseline
    logic [7:0] temperature = 45;  // 45°C baseline
    
    // AI processing state machine
    typedef enum logic [2:0] {
        IDLE,
        PREPROCESSING,
        INFERENCE,
        POSTPROCESSING,
        RESPONSE
    } ai_state_t;
    
    ai_state_t current_state = IDLE;
    logic [15:0] processing_cycles = 0;
    logic [15:0] target_cycles = 0;
    
    // Cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
        end
    end
    
    // AI processing state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            response_valid_reg <= 0;
            response_data_reg <= 0;
            error_reg <= 0;
            ready_reg <= 1;
            processing_cycles <= 0;
            target_cycles <= 0;
            perf_counter <= 0;
        end else begin
            case (current_state)
                IDLE: begin
                    response_valid_reg <= 0;
                    ready_reg <= 1;
                    
                    if (ai_if.valid && ai_if.ready) begin
                        ready_reg <= 0;
                        current_state <= PREPROCESSING;
                        processing_cycles <= 0;
                        
                        // Set target processing cycles based on operation
                        case (ai_if.op_type)
                            3'b000, 3'b001: target_cycles <= $urandom_range(1, 5);      // Memory ops
                            3'b010: target_cycles <= calculate_matmul_cycles();          // MATMUL
                            3'b011: target_cycles <= calculate_conv2d_cycles();          // CONV2D
                            3'b100, 3'b101: target_cycles <= $urandom_range(2, 8);      // Activations
                            3'b110, 3'b111: target_cycles <= $urandom_range(3, 12);     // Pooling
                            default: target_cycles <= $urandom_range(5, 20);
                        endcase
                    end
                end
                
                PREPROCESSING: begin
                    processing_cycles <= processing_cycles + 1;
                    if (processing_cycles >= (target_cycles >> 3)) begin  // 12.5% of total time
                        current_state <= INFERENCE;
                        processing_cycles <= 0;
                    end
                end
                
                INFERENCE: begin
                    processing_cycles <= processing_cycles + 1;
                    if (processing_cycles >= (target_cycles * 3 >> 2)) begin  // 75% of total time
                        current_state <= POSTPROCESSING;
                        processing_cycles <= 0;
                        
                        // Generate result based on operation type
                        generate_ai_result();
                    end
                end
                
                POSTPROCESSING: begin
                    processing_cycles <= processing_cycles + 1;
                    if (processing_cycles >= (target_cycles >> 3)) begin  // 12.5% of total time
                        current_state <= RESPONSE;
                    end
                end
                
                RESPONSE: begin
                    response_valid_reg <= 1;
                    ready_reg <= 1;
                    current_state <= IDLE;
                    perf_counter <= perf_counter + 1;
                end
            endcase
        end
    end
    
    // Calculate processing cycles for matrix multiplication
    function logic [15:0] calculate_matmul_cycles();
        logic [31:0] ops = ai_if.matrix_m * ai_if.matrix_n * ai_if.matrix_k;
        logic [15:0] base_cycles;
        
        // Scale based on data type
        case (ai_if.data_type)
            3'b000: base_cycles = ops >> 12;  // INT8 - fastest
            3'b011: base_cycles = ops >> 10;  // FP16 - medium
            3'b100: base_cycles = ops >> 8;   // FP32 - slower
            default: base_cycles = ops >> 9;
        endcase
        
        // Add some randomness and ensure minimum cycles
        base_cycles = base_cycles + $urandom_range(0, base_cycles >> 2);
        return (base_cycles < 10) ? 10 : base_cycles;
    endfunction
    
    // Calculate processing cycles for convolution
    function logic [15:0] calculate_conv2d_cycles();
        logic [31:0] ops = ai_if.conv_height * ai_if.conv_width * ai_if.conv_channels * 
                          ai_if.kernel_size * ai_if.kernel_size;
        logic [15:0] base_cycles;
        
        // Scale based on data type
        case (ai_if.data_type)
            3'b000: base_cycles = ops >> 10;  // INT8
            3'b011: base_cycles = ops >> 8;   // FP16
            3'b100: base_cycles = ops >> 6;   // FP32
            default: base_cycles = ops >> 7;
        endcase
        
        // Add randomness and ensure minimum
        base_cycles = base_cycles + $urandom_range(0, base_cycles >> 2);
        return (base_cycles < 15) ? 15 : base_cycles;
    endfunction
    
    // Generate AI processing result
    task generate_ai_result();
        case (ai_if.op_type)
            3'b000: begin // READ_OP
                response_data_reg <= ai_if.addr + 64'h1234567890ABCDEF;
                error_reg <= (ai_if.addr[2:0] != 3'b000 && ai_if.size == 8) ? 1 : 0;
            end
            
            3'b001: begin // WRITE_OP
                response_data_reg <= ai_if.data;
                error_reg <= (ai_if.addr[2:0] != 3'b000 && ai_if.size == 8) ? 1 : 0;
            end
            
            3'b010: begin // AI_MATMUL_OP
                // Simulate matrix multiplication with some accuracy variation
                logic [63:0] base_result = ai_if.matrix_m * ai_if.matrix_n * ai_if.matrix_k;
                logic [15:0] accuracy_noise = $urandom_range(0, 1000);
                response_data_reg <= base_result + accuracy_noise;
                error_reg <= (ai_if.matrix_m == 0 || ai_if.matrix_n == 0 || ai_if.matrix_k == 0) ? 1 : 0;
                
                // Update power consumption based on operation complexity
                power_consumption <= 2000 + (ai_if.matrix_m >> 4) + (ai_if.matrix_n >> 4);
            end
            
            3'b011: begin // AI_CONV2D_OP
                // Simulate convolution result
                logic [63:0] base_result = ai_if.conv_height * ai_if.conv_width * ai_if.conv_channels;
                logic [15:0] accuracy_noise = $urandom_range(0, 500);
                response_data_reg <= base_result + accuracy_noise;
                error_reg <= (ai_if.conv_height == 0 || ai_if.conv_width == 0 || ai_if.conv_channels == 0) ? 1 : 0;
                
                power_consumption <= 1800 + (ai_if.conv_channels >> 3) + (ai_if.kernel_size << 2);
            end
            
            3'b100: begin // AI_RELU_OP
                // ReLU: max(0, x) with some processing variation
                if (ai_if.data[63] == 1'b1) begin
                    response_data_reg <= 64'h0;
                end else begin
                    // Add small processing variation to simulate real hardware
                    response_data_reg <= ai_if.data + $urandom_range(0, 10);
                end
                error_reg <= 0;
                power_consumption <= 800 + $urandom_range(0, 200);
            end
            
            3'b101: begin // AI_SIGMOID_OP
                // Sigmoid approximation with processing variation
                response_data_reg <= (ai_if.data >> 1) + $urandom_range(0, 100);
                error_reg <= 0;
                power_consumption <= 1200 + $urandom_range(0, 300);
            end
            
            3'b110: begin // AI_MAXPOOL_OP
                // Max pooling simulation
                response_data_reg <= ai_if.data | (64'h1 << ai_if.pool_size) + $urandom_range(0, 50);
                error_reg <= (ai_if.pool_size == 0) ? 1 : 0;
                power_consumption <= 600 + (ai_if.pool_size << 3);
            end
            
            3'b111: begin // AI_AVGPOOL_OP
                // Average pooling simulation
                response_data_reg <= (ai_if.data >> ai_if.pool_size) + $urandom_range(0, 25);
                error_reg <= (ai_if.pool_size == 0) ? 1 : 0;
                power_consumption <= 600 + (ai_if.pool_size << 3);
            end
            
            default: begin
                response_data_reg <= 64'hDEADBEEFCAFEBABE;
                error_reg <= 1;
            end
        endcase
    endtask
    
    // Temperature simulation based on power consumption and workload
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temperature <= 45;
        end else begin
            // Thermal model based on power and activity
            if (current_state == INFERENCE && power_consumption > 1500) begin
                if (temperature < 85) temperature <= temperature + 1;
            end else if (current_state == IDLE && power_consumption < 1000) begin
                if (temperature > 25) temperature <= temperature - 1;
            end
            
            // Add some thermal noise
            if (cycle_count % 1000 == 0) begin
                temperature <= temperature + $urandom_range(0, 2) - 1;
                if (temperature > 85) temperature <= 85;
                if (temperature < 25) temperature <= 25;
            end
        end
    end
    
    // Connect interface signals
    assign ai_if.ready = ready_reg;
    assign ai_if.response_valid = response_valid_reg;
    assign ai_if.response_data = response_data_reg;
    assign ai_if.error = error_reg;
    assign ai_if.performance_counter = perf_counter;
    assign ai_if.power_consumption = power_consumption;
    assign ai_if.temperature = temperature;
    assign ai_if.debug_state = {29'b0, current_state};
    assign ai_if.debug_pc = {48'b0, processing_cycles};
    assign ai_if.debug_halt = error_reg;
    
    // Enhanced assertions for benchmark validation
    property valid_ready_handshake;
        @(posedge clk) disable iff (!rst_n)
        ai_if.valid && !ai_if.ready |=> ai_if.valid;
    endproperty
    
    assert property (valid_ready_handshake) else
        $error("Valid deasserted before ready at time %0t", $time);
    
    property response_after_request;
        @(posedge clk) disable iff (!rst_n)
        ai_if.valid && ai_if.ready |-> ##[1:1000] ai_if.response_valid;
    endproperty
    
    assert property (response_after_request) else
        $warning("No response within 1000 cycles at time %0t", $time);
    
    // Performance monitoring assertions
    property power_consumption_reasonable;
        @(posedge clk) disable iff (!rst_n)
        power_consumption inside {[500:5000]};
    endproperty
    
    assert property (power_consumption_reasonable) else
        $warning("Power consumption out of expected range: %0d mW", power_consumption);
    
    property temperature_within_limits;
        @(posedge clk) disable iff (!rst_n)
        temperature inside {[20:90]};
    endproperty
    
    assert property (temperature_within_limits) else
        $error("Temperature out of safe range: %0d°C", temperature);
    
    // Coverage collection for benchmark operations
    covergroup benchmark_op_coverage @(posedge clk);
        option.per_instance = 1;
        
        op_type: coverpoint ai_if.op_type iff (ai_if.valid && ai_if.ready) {
            bins memory_ops[] = {3'b000, 3'b001};
            bins ai_compute_ops[] = {3'b010, 3'b011};
            bins ai_activation_ops[] = {3'b100, 3'b101};
            bins ai_pooling_ops[] = {3'b110, 3'b111};
        }
        
        data_type: coverpoint ai_if.data_type iff (ai_if.valid && ai_if.ready) {
            bins integer_types[] = {3'b000, 3'b001, 3'b010};
            bins float_types[] = {3'b011, 3'b100, 3'b101};
        }
        
        processing_state: coverpoint current_state {
            bins processing_states[] = {PREPROCESSING, INFERENCE, POSTPROCESSING};
            bins idle_response[] = {IDLE, RESPONSE};
        }
        
        power_range: coverpoint power_consumption {
            bins low_power = {[500:1000]};
            bins medium_power = {[1001:2000]};
            bins high_power = {[2001:3000]};
            bins very_high_power = {[3001:5000]};
        }
        
        // Cross coverage
        op_data_cross: cross op_type, data_type;
        op_power_cross: cross op_type, power_range;
        state_power_cross: cross processing_state, power_range;
    endgroup
    
    benchmark_op_coverage bench_cov = new();

endmodule

// AI Benchmark Test Class
class ai_benchmark_test extends uvm_test;
    
    benchmark_runner runner;
    benchmark_analyzer analyzer;
    string benchmark_suite = "mlperf_inference";
    
    `uvm_component_utils(ai_benchmark_test)
    
    function new(string name = "ai_benchmark_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Get benchmark suite from config
        if (!uvm_config_db#(string)::get(this, "", "benchmark_suite", benchmark_suite)) begin
            `uvm_info(get_type_name(), "Using default benchmark suite: mlperf_inference", UVM_MEDIUM)
        end
        
        // Create benchmark runner
        runner = benchmark_runner::type_id::create("runner", this);
        
        // Configure runner based on benchmark suite
        configure_benchmark_runner();
        
        // Create analyzer
        analyzer = benchmark_analyzer::type_id::create("analyzer", this);
    endfunction
    
    virtual function void configure_benchmark_runner();
        case (benchmark_suite)
            "mlperf_inference": begin
                runner.selected_benchmarks.push_back("MLPerf-ResNet50-Inference");
                runner.selected_benchmarks.push_back("MLPerf-BERT-Inference");
                runner.selected_benchmarks.push_back("MLPerf-SSD-MobileNet-Inference");
                runner.run_all_benchmarks = 0;
            end
            "image_classification": begin
                runner.selected_benchmarks.push_back("ResNet50-FP32-ImageNet");
                runner.selected_benchmarks.push_back("ResNet50-INT8-ImageNet");
                runner.selected_benchmarks.push_back("MobileNet-V2-FP16-ImageNet");
                runner.selected_benchmarks.push_back("EfficientNet-B0-FP32-ImageNet");
                runner.run_all_benchmarks = 0;
            end
            "object_detection": begin
                runner.selected_benchmarks.push_back("YOLOv5-FP32-COCO");
                runner.selected_benchmarks.push_back("SSD-MobileNet-FP16-COCO");
                runner.selected_benchmarks.push_back("Faster-RCNN-FP32-COCO");
                runner.run_all_benchmarks = 0;
            end
            "nlp_benchmarks": begin
                runner.selected_benchmarks.push_back("BERT-Base-TextClassification");
                runner.selected_benchmarks.push_back("BERT-Large-QuestionAnswering");
                runner.selected_benchmarks.push_back("GPT2-Small-LanguageModeling");
                runner.run_all_benchmarks = 0;
            end
            "recommendation_systems": begin
                runner.selected_benchmarks.push_back("Wide-Deep-MovieLens");
                runner.selected_benchmarks.push_back("DeepFM-MovieLens");
                runner.selected_benchmarks.push_back("NeuralCF-MovieLens");
                runner.run_all_benchmarks = 0;
            end
            "all_benchmarks": begin
                runner.run_all_benchmarks = 1;
            end
            default: begin
                `uvm_warning(get_type_name(), $sformatf("Unknown benchmark suite: %s, using mlperf_inference", benchmark_suite))
                benchmark_suite = "mlperf_inference";
                configure_benchmark_runner();
            end
        endcase
        
        `uvm_info(get_type_name(), $sformatf("Configured for benchmark suite: %s", benchmark_suite), UVM_MEDIUM)
    endfunction
    
    virtual function void end_of_elaboration_phase(uvm_phase phase);
        super.end_of_elaboration_phase(phase);
        uvm_top.print_topology();
    endfunction
    
    virtual task run_phase(uvm_phase phase);
        phase.raise_objection(this, "Running AI benchmarks");
        
        `uvm_info(get_type_name(), $sformatf("Starting AI benchmark suite: %s", benchmark_suite), UVM_LOW)
        
        // Wait for reset
        wait(tb_ai_benchmarks.rst_n);
        repeat(100) @(posedge tb_ai_benchmarks.clk);
        
        `uvm_info(get_type_name(), "AI benchmark execution completed", UVM_LOW)
        
        phase.drop_objection(this, "Benchmarks completed");
    endtask
    
    virtual function void extract_phase(uvm_phase phase);
        super.extract_phase(phase);
        
        // Analyze results if available
        if (runner.all_results.size() > 0) begin
            analyzer.analyze_results(runner.all_results, runner.benchmark_names);
        end
    endfunction
    
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), $sformatf("=== AI BENCHMARK TEST SUMMARY ==="), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Benchmark Suite: %s", benchmark_suite), UVM_LOW)
        
        if (runner.completed_benchmarks == runner.total_benchmarks && runner.failed_benchmarks == 0) begin
            `uvm_info(get_type_name(), "*** ALL BENCHMARKS PASSED ***", UVM_LOW)
        end else begin
            `uvm_error(get_type_name(), $sformatf("*** BENCHMARK FAILURES: %0d failed out of %0d total ***", 
                     runner.failed_benchmarks, runner.total_benchmarks))
        end
    endfunction
    
endclass : ai_benchmark_test