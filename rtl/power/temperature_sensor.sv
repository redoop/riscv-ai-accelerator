/*
 * Temperature Sensor Model
 * 
 * This module models a temperature sensor for the RISC-V AI chip.
 * It simulates realistic temperature behavior based on power consumption
 * and thermal characteristics.
 */

module temperature_sensor #(
    parameter SENSOR_ID = 0,
    parameter TEMP_SENSOR_WIDTH = 16,
    parameter THERMAL_TIME_CONSTANT = 1000  // Thermal time constant in clock cycles
) (
    input  logic                           clk,
    input  logic                           rst_n,
    
    // Power inputs (affecting temperature)
    input  logic [15:0]                    local_power,      // Power dissipated near this sensor
    input  logic [15:0]                    ambient_power,    // Background power from other sources
    
    // Environmental inputs
    input  logic [7:0]                     ambient_temp,     // Ambient temperature (°C)
    input  logic [7:0]                     cooling_factor,   // Cooling effectiveness (0-255)
    
    // Sensor outputs
    output logic [TEMP_SENSOR_WIDTH-1:0] temp_reading,     // Raw ADC reading
    output logic                           sensor_valid,     // Sensor validity
    output logic [7:0]                     temp_celsius     // Temperature in Celsius
);

    // Thermal model parameters
    localparam real THERMAL_RESISTANCE = 0.1;  // °C/mW thermal resistance
    localparam real THERMAL_CAPACITANCE = 1000.0; // Thermal capacitance
    
    // Internal temperature state
    logic [15:0] current_temp_raw;
    logic [7:0] target_temp;
    logic [7:0] current_temp;
    logic [31:0] thermal_accumulator;
    logic [15:0] power_total;
    
    // Sensor characteristics
    logic [7:0] sensor_offset;
    logic [7:0] sensor_gain;
    logic sensor_fault;
    logic [15:0] noise_lfsr;

    // Initialize sensor characteristics
    initial begin
        case (SENSOR_ID)
            0: begin // CPU core sensor
                sensor_offset = 8'd2;   // +2°C offset
                sensor_gain = 8'd255;   // Normal gain
            end
            1: begin // AI unit sensor
                sensor_offset = 8'd1;   // +1°C offset
                sensor_gain = 8'd250;   // Slightly lower gain
            end
            2: begin // Memory sensor
                sensor_offset = 8'd0;   // No offset
                sensor_gain = 8'd255;   // Normal gain
            end
            3: begin // NoC sensor
                sensor_offset = 8'd1;   // +1°C offset
                sensor_gain = 8'd248;   // Lower gain
            end
            default: begin
                sensor_offset = 8'd0;
                sensor_gain = 8'd255;
            end
        endcase
        sensor_fault = 1'b0;
        noise_lfsr = 16'hACE1 + SENSOR_ID; // Different seed per sensor
    end

    // Power calculation
    always_comb begin
        power_total = local_power + (ambient_power >> 2); // Ambient has less effect
    end

    // Thermal model - simplified first-order thermal response
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_temp <= ambient_temp;
            thermal_accumulator <= '0;
            sensor_fault <= 1'b0;
        end else begin
            // Calculate target temperature based on power dissipation
            logic [15:0] power_temp_rise = (power_total >> 4); // Simplified thermal model
            target_temp = ambient_temp + power_temp_rise[7:0];
            
            // Apply cooling factor
            if (cooling_factor > 0) begin
                target_temp = target_temp - ((target_temp - ambient_temp) * cooling_factor >> 8);
            end
            
            // First-order thermal response
            if (current_temp < target_temp) begin
                thermal_accumulator <= thermal_accumulator + {{24{1'b0}}, (target_temp - current_temp)};
            end else if (current_temp > target_temp) begin
                thermal_accumulator <= thermal_accumulator - {{24{1'b0}}, (current_temp - target_temp)};
            end
            
            // Update current temperature based on thermal time constant
            if (thermal_accumulator >= THERMAL_TIME_CONSTANT) begin
                if (current_temp < target_temp) begin
                    current_temp <= current_temp + 1;
                end
                thermal_accumulator <= '0;
            end else if (thermal_accumulator <= -THERMAL_TIME_CONSTANT) begin
                if (current_temp > target_temp) begin
                    current_temp <= current_temp - 1;
                end
                thermal_accumulator <= '0;
            end
            
            // Simulate occasional sensor faults (very rare)
            if (noise_lfsr[15:0] == 16'hFFFF) begin
                sensor_fault <= 1'b1;
            end else if (noise_lfsr[7:0] == 8'h00) begin
                sensor_fault <= 1'b0;
            end
        end
    end

    // Noise generation (LFSR for realistic sensor noise)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            noise_lfsr <= 16'hACE1 + SENSOR_ID;
        end else begin
            // 16-bit LFSR with taps at positions 16, 14, 13, 11
            noise_lfsr <= {noise_lfsr[14:0], 
                          noise_lfsr[15] ^ noise_lfsr[13] ^ noise_lfsr[12] ^ noise_lfsr[10]};
        end
    end

    // Convert temperature to ADC reading
    always_comb begin
        logic [7:0] temp_with_offset;
        logic [7:0] temp_with_noise;
        logic [15:0] adc_value;
        
        // Apply sensor offset and gain
        temp_with_offset = current_temp + sensor_offset;
        
        // Add small amount of noise (±1°C)
        temp_with_noise = temp_with_offset + {{7{1'b0}}, noise_lfsr[0]};
        
        // Convert to ADC reading: ADC = (Temperature + 40) * 256 / 165
        // This assumes a temperature range of -40°C to +125°C mapped to 0-65535
        adc_value = (({{8{1'b0}}, temp_with_noise} + 16'd40) * {{8{1'b0}}, sensor_gain}) >> 8;
        
        if (sensor_fault) begin
            temp_reading = 16'hFFFF; // Invalid reading
            sensor_valid = 1'b0;
            temp_celsius = 8'd0;
        end else begin
            temp_reading = adc_value;
            sensor_valid = 1'b1;
            temp_celsius = temp_with_noise;
        end
    end

endmodule