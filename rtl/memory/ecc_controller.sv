/**
 * ECC Controller for Memory and Cache Protection
 * Implements Single Error Correction, Double Error Detection (SECDED)
 */

module ecc_controller #(
    parameter DATA_WIDTH = 64,
    parameter ECC_WIDTH = 8,
    parameter ADDR_WIDTH = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Memory interface
    input  logic                    mem_req,
    input  logic                    mem_we,
    input  logic [ADDR_WIDTH-1:0]  mem_addr,
    input  logic [DATA_WIDTH-1:0]  mem_wdata,
    output logic [DATA_WIDTH-1:0]  mem_rdata,
    output logic                    mem_ready,
    
    // ECC status and control
    output logic                    single_error,
    output logic                    double_error,
    output logic [ADDR_WIDTH-1:0]  error_addr,
    input  logic                    error_inject_en,
    input  logic [1:0]              error_inject_type, // 0: none, 1: single, 2: double
    
    // Memory array interface
    output logic                    array_req,
    output logic                    array_we,
    output logic [ADDR_WIDTH-1:0]  array_addr,
    output logic [DATA_WIDTH+ECC_WIDTH-1:0] array_wdata,
    input  logic [DATA_WIDTH+ECC_WIDTH-1:0] array_rdata,
    input  logic                    array_ready
);

    // ECC encoding/decoding logic
    logic [ECC_WIDTH-1:0] ecc_encode_out;
    logic [ECC_WIDTH-1:0] ecc_syndrome;
    logic [DATA_WIDTH-1:0] corrected_data;
    logic single_err_detected;
    logic double_err_detected;
    
    // Error injection
    logic [DATA_WIDTH+ECC_WIDTH-1:0] injected_data;
    
    // ECC encoder for write operations
    ecc_encoder #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) u_encoder (
        .data_in(mem_wdata),
        .ecc_out(ecc_encode_out)
    );
    
    // ECC decoder for read operations
    ecc_decoder #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) u_decoder (
        .data_in(array_rdata[DATA_WIDTH-1:0]),
        .ecc_in(array_rdata[DATA_WIDTH+ECC_WIDTH-1:DATA_WIDTH]),
        .data_out(corrected_data),
        .syndrome(ecc_syndrome),
        .single_error(single_err_detected),
        .double_error(double_err_detected)
    );
    
    // Error injection logic
    always_comb begin
        injected_data = {ecc_encode_out, mem_wdata};
        
        if (error_inject_en && mem_we) begin
            case (error_inject_type)
                2'b01: begin // Single bit error
                    injected_data[0] = ~injected_data[0];
                end
                2'b10: begin // Double bit error
                    injected_data[0] = ~injected_data[0];
                    injected_data[1] = ~injected_data[1];
                end
                default: begin
                    // No error injection
                end
            endcase
        end
    end
    
    // Memory interface control
    assign array_req = mem_req;
    assign array_we = mem_we;
    assign array_addr = mem_addr;
    assign array_wdata = injected_data;
    assign mem_ready = array_ready;
    
    // Output data selection
    assign mem_rdata = corrected_data;
    
    // Error status registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            single_error <= 1'b0;
            double_error <= 1'b0;
            error_addr <= '0;
        end else if (array_ready && !mem_we) begin
            single_error <= single_err_detected;
            double_error <= double_err_detected;
            if (single_err_detected || double_err_detected) begin
                error_addr <= mem_addr;
            end
        end
    end

endmodule

/**
 * Hamming ECC Encoder
 * Generates ECC bits for SECDED protection
 */
module ecc_encoder #(
    parameter DATA_WIDTH = 64,
    parameter ECC_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0] data_in,
    output logic [ECC_WIDTH-1:0]  ecc_out
);

    // Hamming code generation matrix
    // For 64-bit data, we need 8 ECC bits (including parity)
    always_comb begin
        // P1: covers bits 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63
        ecc_out[0] = ^(data_in & 64'hAAAAAAAAAAAAAAAA);
        
        // P2: covers bits 2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35,38,39,42,43,46,47,50,51,54,55,58,59,62,63
        ecc_out[1] = ^(data_in & 64'hCCCCCCCCCCCCCCCC);
        
        // P4: covers bits 4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31,36,37,38,39,44,45,46,47,52,53,54,55,60,61,62,63
        ecc_out[2] = ^(data_in & 64'hF0F0F0F0F0F0F0F0);
        
        // P8: covers bits 8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31,40,41,42,43,44,45,46,47,56,57,58,59,60,61,62,63
        ecc_out[3] = ^(data_in & 64'hFF00FF00FF00FF00);
        
        // P16: covers bits 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
        ecc_out[4] = ^(data_in & 64'hFFFF0000FFFF0000);
        
        // P32: covers bits 32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
        ecc_out[5] = ^(data_in & 64'hFFFFFFFF00000000);
        
        // P64: covers all 64 data bits
        ecc_out[6] = ^data_in;
        
        // Overall parity bit
        ecc_out[7] = ^{data_in, ecc_out[6:0]};
    end

endmodule

/**
 * Hamming ECC Decoder
 * Detects and corrects single-bit errors, detects double-bit errors
 */
module ecc_decoder #(
    parameter DATA_WIDTH = 64,
    parameter ECC_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic [ECC_WIDTH-1:0]  ecc_in,
    output logic [DATA_WIDTH-1:0] data_out,
    output logic [ECC_WIDTH-1:0]  syndrome,
    output logic                  single_error,
    output logic                  double_error
);

    logic [ECC_WIDTH-1:0] calculated_ecc;
    logic overall_parity;
    
    // Calculate expected ECC for received data
    ecc_encoder #(
        .DATA_WIDTH(DATA_WIDTH),
        .ECC_WIDTH(ECC_WIDTH)
    ) u_calc_ecc (
        .data_in(data_in),
        .ecc_out(calculated_ecc)
    );
    
    // Calculate syndrome
    assign syndrome = ecc_in ^ calculated_ecc;
    assign overall_parity = ^{data_in, ecc_in};
    
    // Error detection logic
    always_comb begin
        single_error = 1'b0;
        double_error = 1'b0;
        data_out = data_in;
        
        if (syndrome != 8'b0) begin
            if (overall_parity) begin
                // Single bit error - correct it
                single_error = 1'b1;
                if (syndrome[6:0] != 7'b0 && syndrome[6:0] <= DATA_WIDTH) begin
                    // Error in data bit
                    data_out = data_in;
                    data_out[syndrome[6:0]-1] = ~data_in[syndrome[6:0]-1];
                end
                // If error is in ECC bits, data is already correct
            end else begin
                // Double bit error detected
                double_error = 1'b1;
            end
        end
    end

endmodule