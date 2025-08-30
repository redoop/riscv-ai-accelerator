// GPIO Controller Module
// Implements configurable GPIO pins with interrupt support
// Supports input, output, and bidirectional modes

module gpio_controller #(
    parameter NUM_PINS = 32,
    parameter DATA_WIDTH = 32
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // GPIO physical pins
    inout  logic [NUM_PINS-1:0] gpio_pins,
    
    // AXI interface for configuration
    axi4_if.slave       axi_if,
    
    // Interrupt output
    output logic        gpio_irq,
    
    // Status
    output logic [NUM_PINS-1:0] pin_status
);

    // GPIO registers
    logic [NUM_PINS-1:0] gpio_data_out;     // Output data register
    logic [NUM_PINS-1:0] gpio_data_in;      // Input data register
    logic [NUM_PINS-1:0] gpio_direction;    // Direction register (1=output, 0=input)
    logic [NUM_PINS-1:0] gpio_pullup_en;    // Pull-up enable
    logic [NUM_PINS-1:0] gpio_pulldown_en;  // Pull-down enable
    logic [NUM_PINS-1:0] gpio_irq_enable;   // Interrupt enable
    logic [NUM_PINS-1:0] gpio_irq_type;     // Interrupt type (1=edge, 0=level)
    logic [NUM_PINS-1:0] gpio_irq_polarity; // Interrupt polarity (1=rising/high, 0=falling/low)
    logic [NUM_PINS-1:0] gpio_irq_status;   // Interrupt status (write 1 to clear)
    
    // Internal signals
    logic [NUM_PINS-1:0] gpio_in_sync;      // Synchronized input
    logic [NUM_PINS-1:0] gpio_in_prev;      // Previous input state for edge detection
    logic [NUM_PINS-1:0] gpio_out_enable;   // Output enable
    
    // Synchronize inputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpio_in_sync <= '0;
            gpio_in_prev <= '0;
        end else begin
            gpio_in_sync <= gpio_data_in;
            gpio_in_prev <= gpio_in_sync;
        end
    end
    
    // GPIO pin control
    genvar i;
    generate
        for (i = 0; i < NUM_PINS; i++) begin : gpio_pin_gen
            
            // Output enable logic
            assign gpio_out_enable[i] = gpio_direction[i];
            
            // Bidirectional pin control
            assign gpio_pins[i] = gpio_out_enable[i] ? gpio_data_out[i] : 1'bz;
            
            // Input data capture
            always_comb begin
                if (gpio_direction[i] == 1'b0) begin  // Input mode
                    gpio_data_in[i] = gpio_pins[i];
                end else begin  // Output mode
                    gpio_data_in[i] = gpio_data_out[i];  // Read back output value
                end
            end
            
        end
    endgenerate
    
    // Interrupt detection
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpio_irq_status <= '0;
        end else begin
            for (int j = 0; j < NUM_PINS; j++) begin
                if (gpio_irq_enable[j]) begin
                    if (gpio_irq_type[j]) begin  // Edge-triggered
                        if (gpio_irq_polarity[j]) begin  // Rising edge
                            if (!gpio_in_prev[j] && gpio_in_sync[j]) begin
                                gpio_irq_status[j] <= 1'b1;
                            end
                        end else begin  // Falling edge
                            if (gpio_in_prev[j] && !gpio_in_sync[j]) begin
                                gpio_irq_status[j] <= 1'b1;
                            end
                        end
                    end else begin  // Level-triggered
                        if (gpio_irq_polarity[j]) begin  // High level
                            if (gpio_in_sync[j]) begin
                                gpio_irq_status[j] <= 1'b1;
                            end
                        end else begin  // Low level
                            if (!gpio_in_sync[j]) begin
                                gpio_irq_status[j] <= 1'b1;
                            end
                        end
                    end
                end
                
                // Clear interrupt on write-1-to-clear
                if (axi_if.awvalid && axi_if.wvalid && axi_if.awaddr[7:0] == 8'h20) begin
                    if (axi_if.wdata[j]) begin
                        gpio_irq_status[j] <= 1'b0;
                    end
                end
            end
        end
    end
    
    // Register access via AXI
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gpio_data_out <= '0;
            gpio_direction <= '0;
            gpio_pullup_en <= '0;
            gpio_pulldown_en <= '0;
            gpio_irq_enable <= '0;
            gpio_irq_type <= '0;
            gpio_irq_polarity <= '0;
        end else if (axi_if.awvalid && axi_if.wvalid) begin
            case (axi_if.awaddr[7:0])
                8'h00: gpio_data_out <= axi_if.wdata[NUM_PINS-1:0];
                8'h08: gpio_direction <= axi_if.wdata[NUM_PINS-1:0];
                8'h0C: gpio_pullup_en <= axi_if.wdata[NUM_PINS-1:0];
                8'h10: gpio_pulldown_en <= axi_if.wdata[NUM_PINS-1:0];
                8'h14: gpio_irq_enable <= axi_if.wdata[NUM_PINS-1:0];
                8'h18: gpio_irq_type <= axi_if.wdata[NUM_PINS-1:0];
                8'h1C: gpio_irq_polarity <= axi_if.wdata[NUM_PINS-1:0];
                // Note: gpio_irq_status is handled in interrupt detection logic
            endcase
        end
    end
    
    // AXI read interface
    always_comb begin
        case (axi_if.araddr[7:0])
            8'h00: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_data_out};
            8'h04: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_data_in};
            8'h08: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_direction};
            8'h0C: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_pullup_en};
            8'h10: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_pulldown_en};
            8'h14: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_irq_enable};
            8'h18: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_irq_type};
            8'h1C: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_irq_polarity};
            8'h20: axi_if.rdata = {{(32-NUM_PINS){1'b0}}, gpio_irq_status};
            default: axi_if.rdata = 32'h0;
        endcase
    end
    
    // AXI interface control
    assign axi_if.awready = 1'b1;
    assign axi_if.wready = 1'b1;
    assign axi_if.bvalid = axi_if.awvalid && axi_if.wvalid;
    assign axi_if.bresp = 2'b00;
    
    assign axi_if.arready = 1'b1;
    assign axi_if.rvalid = axi_if.arvalid;
    assign axi_if.rresp = 2'b00;
    
    // Interrupt output
    assign gpio_irq = |gpio_irq_status;
    
    // Status output
    assign pin_status = gpio_data_in;
    
endmodule