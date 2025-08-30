// RISC-V AI UVM Testbench
// Top-level testbench for UVM-based verification

`timescale 1ns/1ps

`include "uvm_macros.svh"
import uvm_pkg::*;
import riscv_ai_pkg::*;

module tb_riscv_ai_uvm;

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
    
    // Interface instantiation
    riscv_ai_interface ai_if(clk, rst_n);
    
    // DUT instantiation (simplified for UVM testing)
    riscv_ai_dut_wrapper dut (
        .clk(clk),
        .rst_n(rst_n),
        .ai_if(ai_if.dut_mp)
    );
    
    // UVM test execution
    initial begin
        // Set interface in config DB
        uvm_config_db#(virtual riscv_ai_interface)::set(null, "*", "vif", ai_if);
        
        // Set default sequence for the sequencer
        uvm_config_db#(uvm_object_wrapper)::set(null, "*", "default_sequence", riscv_ai_random_sequence::type_id::get());
        
        // Enable UVM command line processor
        uvm_cmdline_processor clp = uvm_cmdline_processor::get_inst();
        
        // Start UVM test
        run_test();
    end
    
    // Timeout watchdog
    initial begin
        #100ms;
        `uvm_fatal("TB", "Test timeout - simulation ran too long")
    end
    
    // Waveform dumping
    initial begin
        if ($test$plusargs("DUMP_WAVES")) begin
            $dumpfile("tb_riscv_ai_uvm.vcd");
            $dumpvars(0, tb_riscv_ai_uvm);
        end
    end
    
    // Performance monitoring
    initial begin
        forever begin
            @(posedge clk);
            if (ai_if.valid && ai_if.ready) begin
                `uvm_info("PERF", $sformatf("Transaction at time %0t: op=%s", $time, ai_if.op_type.name()), UVM_HIGH)
            end
        end
    end

endmodule

// DUT wrapper for UVM testing
// This wrapper provides a simplified interface for the actual DUT
module riscv_ai_dut_wrapper (
    input logic clk,
    input logic rst_n,
    riscv_ai_interface.dut_mp ai_if
);

    // Internal signals
    logic [31:0] cycle_count = 0;
    logic [63:0] response_data_reg = 0;
    logic response_valid_reg = 0;
    logic error_reg = 0;
    logic ready_reg = 1;
    
    // Performance counters
    logic [31:0] perf_counter = 0;
    logic [15:0] power_consumption = 1000;  // 1W baseline
    logic [7:0] temperature = 45;  // 45°C baseline
    
    // Debug signals
    logic [31:0] debug_state = 0;
    logic [63:0] debug_pc = 0;
    logic debug_halt = 0;
    
    // Cycle counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
        end
    end
    
    // Simple transaction processing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            response_valid_reg <= 0;
            response_data_reg <= 0;
            error_reg <= 0;
            ready_reg <= 1;
            perf_counter <= 0;
        end else begin
            // Default values
            response_valid_reg <= 0;
            ready_reg <= 1;
            
            // Process incoming transactions
            if (ai_if.valid && ai_if.ready) begin
                // Simulate processing delay
                ready_reg <= 0;
                
                // Process different operation types
                case (ai_if.op_type)
                    3'b000: begin // READ_OP
                        response_data_reg <= ai_if.addr + 64'h1234567890ABCDEF;
                        error_reg <= (ai_if.addr[2:0] != 3'b000 && ai_if.size == 8) ? 1 : 0;  // Alignment check
                    end
                    
                    3'b001: begin // WRITE_OP
                        response_data_reg <= ai_if.data;
                        error_reg <= (ai_if.addr[2:0] != 3'b000 && ai_if.size == 8) ? 1 : 0;  // Alignment check
                    end
                    
                    3'b010: begin // AI_MATMUL_OP
                        // Simulate matrix multiplication result
                        response_data_reg <= ai_if.matrix_m * ai_if.matrix_n * ai_if.matrix_k;
                        error_reg <= (ai_if.matrix_m == 0 || ai_if.matrix_n == 0 || ai_if.matrix_k == 0) ? 1 : 0;
                        power_consumption <= 2000 + (ai_if.matrix_m >> 4);  // Higher power for larger matrices
                    end
                    
                    3'b011: begin // AI_CONV2D_OP
                        // Simulate convolution result
                        response_data_reg <= ai_if.conv_height * ai_if.conv_width * ai_if.conv_channels;
                        error_reg <= (ai_if.conv_height == 0 || ai_if.conv_width == 0 || ai_if.conv_channels == 0) ? 1 : 0;
                        power_consumption <= 1800 + (ai_if.conv_channels >> 3);
                    end
                    
                    3'b100: begin // AI_RELU_OP
                        // ReLU: max(0, x)
                        response_data_reg <= (ai_if.data[63] == 1'b1) ? 64'h0 : ai_if.data;
                        error_reg <= 0;
                        power_consumption <= 800;
                    end
                    
                    3'b101: begin // AI_SIGMOID_OP
                        // Simplified sigmoid
                        response_data_reg <= ai_if.data >> 1;  // Simplified approximation
                        error_reg <= 0;
                        power_consumption <= 1200;
                    end
                    
                    3'b110: begin // AI_MAXPOOL_OP
                        // Simplified max pooling
                        response_data_reg <= ai_if.data | (64'h1 << ai_if.pool_size);
                        error_reg <= (ai_if.pool_size == 0) ? 1 : 0;
                        power_consumption <= 600;
                    end
                    
                    3'b111: begin // AI_AVGPOOL_OP
                        // Simplified average pooling
                        response_data_reg <= ai_if.data >> ai_if.pool_size;
                        error_reg <= (ai_if.pool_size == 0) ? 1 : 0;
                        power_consumption <= 600;
                    end
                    
                    default: begin
                        response_data_reg <= 64'hDEADBEEFCAFEBABE;
                        error_reg <= 1;  // Unknown operation
                    end
                endcase
                
                perf_counter <= perf_counter + 1;
            end
            
            // Generate response after processing delay
            if (!ready_reg) begin
                // Simulate variable processing latency based on operation
                case (ai_if.op_type)
                    3'b010: begin // MATMUL - longer latency
                        repeat($urandom_range(10, 50)) @(posedge clk);
                    end
                    3'b011: begin // CONV2D - longer latency
                        repeat($urandom_range(15, 60)) @(posedge clk);
                    end
                    default: begin // Other operations - shorter latency
                        repeat($urandom_range(1, 10)) @(posedge clk);
                    end
                endcase
                
                response_valid_reg <= 1;
                ready_reg <= 1;
            end
        end
    end
    
    // Temperature simulation based on power consumption
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            temperature <= 45;
        end else begin
            // Simple thermal model
            if (power_consumption > 1500) begin
                temperature <= temperature + 1;
            end else if (power_consumption < 1000 && temperature > 45) begin
                temperature <= temperature - 1;
            end
            
            // Thermal limits
            if (temperature > 85) temperature <= 85;
            if (temperature < 25) temperature <= 25;
        end
    end
    
    // Debug state machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            debug_state <= 0;
            debug_pc <= 0;
            debug_halt <= 0;
        end else begin
            debug_state <= debug_state + 1;
            if (ai_if.valid && ai_if.ready) begin
                debug_pc <= debug_pc + 4;  // Simulate PC increment
            end
            
            // Simulate debug halt on error
            debug_halt <= error_reg;
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
    assign ai_if.debug_state = debug_state;
    assign ai_if.debug_pc = debug_pc;
    assign ai_if.debug_halt = debug_halt;
    
    // Assertions for basic protocol checking
    property valid_ready_handshake;
        @(posedge clk) disable iff (!rst_n)
        ai_if.valid && !ai_if.ready |=> ai_if.valid;
    endproperty
    
    assert property (valid_ready_handshake) else
        $error("Valid deasserted before ready at time %0t", $time);
    
    property response_after_request;
        @(posedge clk) disable iff (!rst_n)
        ai_if.valid && ai_if.ready |-> ##[1:100] ai_if.response_valid;
    endproperty
    
    assert property (response_after_request) else
        $warning("No response within 100 cycles at time %0t", $time);
    
    // Coverage collection
    covergroup op_coverage @(posedge clk);
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
        
        op_data_cross: cross op_type, data_type;
    endgroup
    
    op_coverage op_cov = new();

endmodule