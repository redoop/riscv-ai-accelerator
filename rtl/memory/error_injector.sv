/**
 * Error Injection Module for Testing and Validation
 * Provides controlled error injection capabilities for system testing
 */

module error_injector #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 64
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    inject_enable,
    input  logic [3:0]              inject_mode,
    input  logic [ADDR_WIDTH-1:0]  inject_addr,
    input  logic [15:0]             inject_mask,
    input  logic [31:0]             inject_count,
    input  logic                    inject_trigger,
    
    // Memory access monitoring
    input  logic                    mem_access,
    input  logic [ADDR_WIDTH-1:0]  mem_addr,
    input  logic                    mem_we,
    input  logic [DATA_WIDTH-1:0]  mem_wdata,
    output logic [DATA_WIDTH-1:0]  mem_wdata_out,
    input  logic [DATA_WIDTH-1:0]  mem_rdata,
    output logic [DATA_WIDTH-1:0]  mem_rdata_out,
    
    // Error injection outputs
    output logic                    single_error_inject,
    output logic                    double_error_inject,
    output logic                    burst_error_inject,
    output logic                    address_error_inject,
    output logic                    control_error_inject,
    
    // Status outputs
    output logic [31:0]             injection_count,
    output logic                    injection_active,
    output logic [ADDR_WIDTH-1:0]  last_inject_addr
);

    // Injection modes
    typedef enum logic [3:0] {
        INJECT_NONE         = 4'b0000,
        INJECT_SINGLE_BIT   = 4'b0001,
        INJECT_DOUBLE_BIT   = 4'b0010,
        INJECT_BURST_ERROR  = 4'b0011,
        INJECT_ADDR_ERROR   = 4'b0100,
        INJECT_CTRL_ERROR   = 4'b0101,
        INJECT_RANDOM       = 4'b0110,
        INJECT_PERIODIC     = 4'b0111,
        INJECT_TARGETED     = 4'b1000
    } inject_mode_t;

    // Internal registers
    logic [31:0] inject_counter;
    logic [31:0] access_counter;
    logic [15:0] lfsr_state;
    logic        inject_active_reg;
    logic [ADDR_WIDTH-1:0] last_addr_reg;
    
    // LFSR for random error generation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_state <= 16'hACE1; // Non-zero seed
        end else begin
            lfsr_state <= {lfsr_state[14:0], lfsr_state[15] ^ lfsr_state[13] ^ lfsr_state[12] ^ lfsr_state[10]};
        end
    end
    
    // Access counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            access_counter <= 32'b0;
        end else if (mem_access) begin
            access_counter <= access_counter + 1;
        end
    end
    
    // Injection counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inject_counter <= 32'b0;
        end else if (injection_active) begin
            inject_counter <= inject_counter + 1;
        end
    end
    
    // Injection trigger logic
    logic should_inject;
    always_comb begin
        should_inject = 1'b0;
        
        if (inject_enable) begin
            case (inject_mode_t'(inject_mode))
                INJECT_TARGETED: begin
                    should_inject = mem_access && (mem_addr == inject_addr);
                end
                INJECT_PERIODIC: begin
                    should_inject = mem_access && (access_counter % inject_count == 0);
                end
                INJECT_RANDOM: begin
                    should_inject = mem_access && (lfsr_state[7:0] < inject_mask[7:0]);
                end
                default: begin
                    should_inject = inject_trigger && mem_access;
                end
            endcase
        end
    end
    
    // Error injection logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            single_error_inject <= 1'b0;
            double_error_inject <= 1'b0;
            burst_error_inject <= 1'b0;
            address_error_inject <= 1'b0;
            control_error_inject <= 1'b0;
            inject_active_reg <= 1'b0;
            last_addr_reg <= '0;
        end else begin
            // Clear previous injections
            single_error_inject <= 1'b0;
            double_error_inject <= 1'b0;
            burst_error_inject <= 1'b0;
            address_error_inject <= 1'b0;
            control_error_inject <= 1'b0;
            inject_active_reg <= 1'b0;
            
            if (should_inject) begin
                inject_active_reg <= 1'b1;
                last_addr_reg <= mem_addr;
                
                case (inject_mode_t'(inject_mode))
                    INJECT_SINGLE_BIT: begin
                        single_error_inject <= 1'b1;
                    end
                    INJECT_DOUBLE_BIT: begin
                        double_error_inject <= 1'b1;
                    end
                    INJECT_BURST_ERROR: begin
                        burst_error_inject <= 1'b1;
                    end
                    INJECT_ADDR_ERROR: begin
                        address_error_inject <= 1'b1;
                    end
                    INJECT_CTRL_ERROR: begin
                        control_error_inject <= 1'b1;
                    end
                    INJECT_RANDOM: begin
                        // Random error type based on LFSR
                        case (lfsr_state[1:0])
                            2'b00: single_error_inject <= 1'b1;
                            2'b01: double_error_inject <= 1'b1;
                            2'b10: burst_error_inject <= 1'b1;
                            2'b11: address_error_inject <= 1'b1;
                        endcase
                    end
                    default: begin
                        single_error_inject <= 1'b1;
                    end
                endcase
            end
        end
    end
    
    // Data corruption for write operations
    always_comb begin
        mem_wdata_out = mem_wdata;
        
        if (inject_active_reg && mem_we) begin
            case (inject_mode_t'(inject_mode))
                INJECT_SINGLE_BIT: begin
                    // Flip single bit based on mask
                    mem_wdata_out = mem_wdata ^ (64'b1 << (inject_mask[5:0] % DATA_WIDTH));
                end
                INJECT_DOUBLE_BIT: begin
                    // Flip two bits
                    mem_wdata_out = mem_wdata ^ (64'b1 << (inject_mask[5:0] % DATA_WIDTH)) ^
                                               (64'b1 << ((inject_mask[11:6] % DATA_WIDTH)));
                end
                INJECT_BURST_ERROR: begin
                    // Corrupt multiple consecutive bits
                    mem_wdata_out = mem_wdata ^ (inject_mask << (inject_mask[5:0] % (DATA_WIDTH-16)));
                end
                default: begin
                    mem_wdata_out = mem_wdata;
                end
            endcase
        end
    end
    
    // Data corruption for read operations
    always_comb begin
        mem_rdata_out = mem_rdata;
        
        if (inject_active_reg && !mem_we) begin
            case (inject_mode_t'(inject_mode))
                INJECT_SINGLE_BIT: begin
                    // Flip single bit based on mask
                    mem_rdata_out = mem_rdata ^ (64'b1 << (inject_mask[5:0] % DATA_WIDTH));
                end
                INJECT_DOUBLE_BIT: begin
                    // Flip two bits
                    mem_rdata_out = mem_rdata ^ (64'b1 << (inject_mask[5:0] % DATA_WIDTH)) ^
                                               (64'b1 << ((inject_mask[11:6] % DATA_WIDTH)));
                end
                INJECT_BURST_ERROR: begin
                    // Corrupt multiple consecutive bits
                    mem_rdata_out = mem_rdata ^ (inject_mask << (inject_mask[5:0] % (DATA_WIDTH-16)));
                end
                default: begin
                    mem_rdata_out = mem_rdata;
                end
            endcase
        end
    end
    
    // Output assignments
    assign injection_count = inject_counter;
    assign injection_active = inject_active_reg;
    assign last_inject_addr = last_addr_reg;

endmodule

/**
 * Error Pattern Generator
 * Generates various error patterns for comprehensive testing
 */
module error_pattern_generator (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        enable,
    input  logic [2:0]  pattern_select,
    output logic [63:0] error_pattern
);

    logic [31:0] counter;
    logic [15:0] lfsr;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 32'b0;
            lfsr <= 16'hACE1;
        end else if (enable) begin
            counter <= counter + 1;
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        end
    end
    
    always_comb begin
        case (pattern_select)
            3'b000: error_pattern = 64'h0000000000000001; // Single bit
            3'b001: error_pattern = 64'h0000000000000003; // Two adjacent bits
            3'b010: error_pattern = 64'h000000000000000F; // Four adjacent bits
            3'b011: error_pattern = 64'h00000000000000FF; // Byte error
            3'b100: error_pattern = 64'h0000000000000101; // Two separated bits
            3'b101: error_pattern = {lfsr, lfsr, lfsr, lfsr}; // Random pattern
            3'b110: error_pattern = {32'b0, counter}; // Counter pattern
            3'b111: error_pattern = 64'hFFFFFFFFFFFFFFFF; // All bits
            default: error_pattern = 64'b0;
        endcase
    end

endmodule