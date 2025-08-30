/**
 * Hardware Error Detection and Reporting System
 * Monitors various error sources and provides centralized error reporting
 */

module error_detector #(
    parameter NUM_CORES = 4,
    parameter NUM_TPUS = 2,
    parameter NUM_VPUS = 2
) (
    input  logic clk,
    input  logic rst_n,
    
    // ECC error inputs from memory subsystem
    input  logic [NUM_CORES-1:0]    l1_cache_single_error,
    input  logic [NUM_CORES-1:0]    l1_cache_double_error,
    input  logic                    l2_cache_single_error,
    input  logic                    l2_cache_double_error,
    input  logic                    l3_cache_single_error,
    input  logic                    l3_cache_double_error,
    input  logic                    memory_single_error,
    input  logic                    memory_double_error,
    
    // Compute unit error inputs
    input  logic [NUM_CORES-1:0]    core_arithmetic_error,
    input  logic [NUM_CORES-1:0]    core_pipeline_error,
    input  logic [NUM_TPUS-1:0]     tpu_compute_error,
    input  logic [NUM_TPUS-1:0]     tpu_overflow_error,
    input  logic [NUM_VPUS-1:0]     vpu_compute_error,
    input  logic [NUM_VPUS-1:0]     vpu_overflow_error,
    
    // System error inputs
    input  logic                    noc_deadlock_error,
    input  logic                    noc_timeout_error,
    input  logic                    power_domain_error,
    input  logic                    thermal_error,
    input  logic                    clock_error,
    
    // Error status outputs
    output logic                    error_interrupt,
    output logic [31:0]             error_status,
    output logic [31:0]             error_mask,
    output logic [7:0]              error_severity,
    
    // Error logging interface
    output logic                    error_log_valid,
    output logic [63:0]             error_log_data,
    output logic [31:0]             error_timestamp,
    
    // Control interface
    input  logic                    error_clear,
    input  logic [31:0]             error_mask_set,
    input  logic                    error_inject_enable,
    input  logic [4:0]              error_inject_type
);

    // Error type definitions
    typedef enum logic [4:0] {
        ERR_NONE           = 5'b00000,
        ERR_L1_SINGLE      = 5'b00001,
        ERR_L1_DOUBLE      = 5'b00010,
        ERR_L2_SINGLE      = 5'b00011,
        ERR_L2_DOUBLE      = 5'b00100,
        ERR_L3_SINGLE      = 5'b00101,
        ERR_L3_DOUBLE      = 5'b00110,
        ERR_MEM_SINGLE     = 5'b00111,
        ERR_MEM_DOUBLE     = 5'b01000,
        ERR_CORE_ARITH     = 5'b01001,
        ERR_CORE_PIPE      = 5'b01010,
        ERR_TPU_COMPUTE    = 5'b01011,
        ERR_TPU_OVERFLOW   = 5'b01100,
        ERR_VPU_COMPUTE    = 5'b01101,
        ERR_VPU_OVERFLOW   = 5'b01110,
        ERR_NOC_DEADLOCK   = 5'b01111,
        ERR_NOC_TIMEOUT    = 5'b10000,
        ERR_POWER_DOMAIN   = 5'b10001,
        ERR_THERMAL        = 5'b10010,
        ERR_CLOCK          = 5'b10011
    } error_type_t;

    // Error severity levels
    typedef enum logic [2:0] {
        SEV_INFO     = 3'b000,  // Informational
        SEV_WARNING  = 3'b001,  // Warning
        SEV_MINOR    = 3'b010,  // Minor error
        SEV_MAJOR    = 3'b011,  // Major error
        SEV_CRITICAL = 3'b100,  // Critical error
        SEV_FATAL    = 3'b101   // Fatal error
    } severity_t;

    // Internal registers
    logic [31:0] error_status_reg;
    logic [31:0] error_mask_reg;
    logic [31:0] error_count [32];
    logic [31:0] timestamp_counter;
    
    // Error detection logic
    logic [31:0] current_errors;
    logic [31:0] new_errors;
    logic [31:0] masked_errors;
    
    // Error injection
    logic [31:0] injected_errors;
    
    // Timestamp counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            timestamp_counter <= 32'b0;
        end else begin
            timestamp_counter <= timestamp_counter + 1;
        end
    end
    
    // Collect all error signals
    always_comb begin
        current_errors = 32'b0;
        
        // Memory ECC errors
        current_errors[0] = |l1_cache_single_error;
        current_errors[1] = |l1_cache_double_error;
        current_errors[2] = l2_cache_single_error;
        current_errors[3] = l2_cache_double_error;
        current_errors[4] = l3_cache_single_error;
        current_errors[5] = l3_cache_double_error;
        current_errors[6] = memory_single_error;
        current_errors[7] = memory_double_error;
        
        // Compute unit errors
        current_errors[8] = |core_arithmetic_error;
        current_errors[9] = |core_pipeline_error;
        current_errors[10] = |tpu_compute_error;
        current_errors[11] = |tpu_overflow_error;
        current_errors[12] = |vpu_compute_error;
        current_errors[13] = |vpu_overflow_error;
        
        // System errors
        current_errors[14] = noc_deadlock_error;
        current_errors[15] = noc_timeout_error;
        current_errors[16] = power_domain_error;
        current_errors[17] = thermal_error;
        current_errors[18] = clock_error;
        
        // Reserved for future use
        current_errors[31:19] = 13'b0;
    end
    
    // Error injection logic
    always_comb begin
        injected_errors = 32'b0;
        if (error_inject_enable) begin
            injected_errors[error_inject_type] = 1'b1;
        end
    end
    
    // Combine real and injected errors
    assign new_errors = current_errors | injected_errors;
    
    // Error status register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_status_reg <= 32'b0;
        end else if (error_clear) begin
            error_status_reg <= 32'b0;
        end else begin
            error_status_reg <= error_status_reg | new_errors;
        end
    end
    
    // Error mask register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error_mask_reg <= 32'hFFFFFFFF; // All errors masked by default
        end else begin
            error_mask_reg <= error_mask_set;
        end
    end
    
    // Masked errors
    assign masked_errors = error_status_reg & ~error_mask_reg;
    
    // Error interrupt generation
    assign error_interrupt = |masked_errors;
    
    // Error severity assignment
    always_comb begin
        error_severity = 8'b0;
        
        // Fatal errors
        if (masked_errors[1] || masked_errors[3] || masked_errors[5] || 
            masked_errors[7] || masked_errors[14] || masked_errors[17] || 
            masked_errors[18]) begin
            error_severity = {5'b0, SEV_FATAL};
        end
        // Critical errors
        else if (masked_errors[9] || masked_errors[11] || masked_errors[13] || 
                 masked_errors[15] || masked_errors[16]) begin
            error_severity = {5'b0, SEV_CRITICAL};
        end
        // Major errors
        else if (masked_errors[8] || masked_errors[10] || masked_errors[12]) begin
            error_severity = {5'b0, SEV_MAJOR};
        end
        // Minor errors (single-bit ECC errors)
        else if (masked_errors[0] || masked_errors[2] || masked_errors[4] || 
                 masked_errors[6]) begin
            error_severity = {5'b0, SEV_MINOR};
        end
    end
    
    // Error counting
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : error_counters
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    error_count[i] <= 32'b0;
                end else if (error_clear) begin
                    error_count[i] <= 32'b0;
                end else if (new_errors[i] && !error_status_reg[i]) begin
                    // Increment on new error occurrence
                    error_count[i] <= error_count[i] + 1;
                end
            end
        end
    endgenerate
    
    // Error logging
    logic [4:0] error_encode;
    logic error_valid;
    
    // Priority encoder for error logging
    always_comb begin
        error_encode = 5'b0;
        error_valid = 1'b0;
        
        for (int j = 31; j >= 0; j--) begin
            if (new_errors[j] && !error_status_reg[j]) begin
                error_encode = j[4:0];
                error_valid = 1'b1;
            end
        end
    end
    
    assign error_log_valid = error_valid;
    assign error_log_data = {
        timestamp_counter,      // [63:32] Timestamp
        3'b0,                  // [31:29] Reserved
        error_severity[2:0],   // [28:26] Severity
        3'b0,                  // [25:23] Reserved
        error_encode,          // [22:18] Error type
        2'b0,                  // [17:16] Reserved
        error_count[error_encode][15:0] // [15:0] Error count
    };
    assign error_timestamp = timestamp_counter;
    
    // Output assignments
    assign error_status = error_status_reg;
    assign error_mask = error_mask_reg;

endmodule