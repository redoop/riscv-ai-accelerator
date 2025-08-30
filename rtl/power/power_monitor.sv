/*
 * Power Monitor
 * 
 * This module monitors power consumption across different domains
 * of the RISC-V AI chip and provides real-time power reporting.
 */

module power_monitor #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter POWER_SAMPLE_PERIOD = 1000  // Sample period in clock cycles
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Activity inputs
    input  logic [NUM_CORES-1:0]          core_activity,
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_activity,
    input  logic                           memory_activity,
    input  logic                           noc_activity,
    
    // Voltage and frequency inputs
    input  logic [2:0]                     voltage_level,
    input  logic [2:0]                     frequency_level,
    
    // Power enable inputs
    input  logic [NUM_CORES-1:0]          core_power_enable,
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_power_enable,
    input  logic                           memory_power_enable,
    input  logic                           noc_power_enable,
    
    // Load monitoring inputs
    input  logic [15:0]                    core_load [NUM_CORES-1:0],
    input  logic [15:0]                    memory_load,
    input  logic [15:0]                    ai_accel_load,
    
    // Power consumption outputs
    output logic [15:0]                    core_power [NUM_CORES-1:0],
    output logic [15:0]                    ai_unit_power [NUM_AI_UNITS-1:0],
    output logic [15:0]                    memory_power,
    output logic [15:0]                    noc_power,
    output logic [15:0]                    total_power,
    
    // Power statistics
    output logic [15:0]                    avg_power,
    output logic [15:0]                    peak_power,
    output logic [31:0]                    energy_consumed,  // Accumulated energy in mJ
    
    // Configuration interface
    input  logic [31:0]                    pm_config_addr,
    input  logic [31:0]                    pm_config_wdata,
    output logic [31:0]                    pm_config_rdata,
    input  logic                           pm_config_req,
    input  logic                           pm_config_we,
    output logic                           pm_config_ready
);

    // Power model parameters (in mW at nominal voltage/frequency)
    localparam logic [15:0] CORE_BASE_POWER = 16'd500;      // 500mW per core base
    localparam logic [15:0] CORE_DYNAMIC_POWER = 16'd1500;  // 1500mW per core at full load
    localparam logic [15:0] AI_UNIT_BASE_POWER = 16'd1000;  // 1000mW per AI unit base
    localparam logic [15:0] AI_UNIT_DYNAMIC_POWER = 16'd5000; // 5000mW per AI unit at full load
    localparam logic [15:0] MEMORY_BASE_POWER = 16'd800;    // 800mW memory base
    localparam logic [15:0] MEMORY_DYNAMIC_POWER = 16'd2000; // 2000mW memory at full load
    localparam logic [15:0] NOC_BASE_POWER = 16'd300;       // 300mW NoC base
    localparam logic [15:0] NOC_DYNAMIC_POWER = 16'd700;    // 700mW NoC at full load

    // Internal registers
    logic [31:0] sample_counter;
    logic [31:0] power_accumulator;
    logic [15:0] power_samples;
    logic [15:0] current_total_power;
    logic [15:0] max_power_seen;
    
    // Voltage and frequency scaling factors
    logic [7:0] voltage_scale_factor;
    logic [7:0] frequency_scale_factor;
    
    // Configuration registers
    logic power_monitor_enable;
    logic [15:0] power_sample_period_reg;

    // Calculate voltage scaling factor (power scales with V^2)
    always_comb begin
        case (voltage_level)
            3'd0: voltage_scale_factor = 8'd36;   // 0.6V -> (0.6)^2 = 0.36
            3'd1: voltage_scale_factor = 8'd49;   // 0.7V -> (0.7)^2 = 0.49
            3'd2: voltage_scale_factor = 8'd64;   // 0.8V -> (0.8)^2 = 0.64
            3'd3: voltage_scale_factor = 8'd81;   // 0.9V -> (0.9)^2 = 0.81
            3'd4: voltage_scale_factor = 8'd100;  // 1.0V -> (1.0)^2 = 1.00
            3'd5: voltage_scale_factor = 8'd121;  // 1.1V -> (1.1)^2 = 1.21
            3'd6: voltage_scale_factor = 8'd144;  // 1.2V -> (1.2)^2 = 1.44
            3'd7: voltage_scale_factor = 8'd169;  // 1.3V -> (1.3)^2 = 1.69
        endcase
    end

    // Calculate frequency scaling factor (dynamic power scales linearly with frequency)
    always_comb begin
        case (frequency_level)
            3'd0: frequency_scale_factor = 8'd25;   // 200MHz -> 0.25x
            3'd1: frequency_scale_factor = 8'd50;   // 400MHz -> 0.50x
            3'd2: frequency_scale_factor = 8'd75;   // 600MHz -> 0.75x
            3'd3: frequency_scale_factor = 8'd100;  // 800MHz -> 1.00x
            3'd4: frequency_scale_factor = 8'd125;  // 1000MHz -> 1.25x
            3'd5: frequency_scale_factor = 8'd150;  // 1200MHz -> 1.50x
            3'd6: frequency_scale_factor = 8'd175;  // 1400MHz -> 1.75x
            3'd7: frequency_scale_factor = 8'd200;  // 1600MHz -> 2.00x
        endcase
    end

    // Core power calculation
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : core_power_calc
            always_comb begin
                automatic logic [15:0] base_power = 16'd0;
                automatic logic [15:0] dynamic_power = 16'd0;
                
                if (core_power_enable[i]) begin
                    // Base power (leakage) scales with voltage^2
                    base_power = (CORE_BASE_POWER * voltage_scale_factor) >> 7;
                    
                    // Dynamic power scales with voltage^2 * frequency * activity
                    dynamic_power = (CORE_DYNAMIC_POWER * voltage_scale_factor) >> 7;
                    dynamic_power = (dynamic_power * frequency_scale_factor) >> 7;
                    dynamic_power = (dynamic_power * core_load[i]) >> 16;
                    
                    core_power[i] = base_power + dynamic_power;
                end else begin
                    core_power[i] = 16'd0; // No power when gated
                end
            end
        end
    endgenerate

    // AI unit power calculation
    generate
        for (i = 0; i < NUM_AI_UNITS; i++) begin : ai_power_calc
            always_comb begin
                automatic logic [15:0] base_power = 16'd0;
                automatic logic [15:0] dynamic_power = 16'd0;
                
                if (ai_unit_power_enable[i]) begin
                    base_power = (AI_UNIT_BASE_POWER * voltage_scale_factor) >> 7;
                    dynamic_power = (AI_UNIT_DYNAMIC_POWER * voltage_scale_factor) >> 7;
                    dynamic_power = (dynamic_power * frequency_scale_factor) >> 7;
                    dynamic_power = (dynamic_power * ai_accel_load) >> 16;
                    
                    ai_unit_power[i] = base_power + dynamic_power;
                end else begin
                    ai_unit_power[i] = 16'd0;
                end
            end
        end
    endgenerate

    // Memory power calculation
    always_comb begin
        automatic logic [15:0] base_power = 16'd0;
        automatic logic [15:0] dynamic_power = 16'd0;
        
        if (memory_power_enable) begin
            base_power = (MEMORY_BASE_POWER * voltage_scale_factor) >> 7;
            dynamic_power = (MEMORY_DYNAMIC_POWER * voltage_scale_factor) >> 7;
            dynamic_power = (dynamic_power * frequency_scale_factor) >> 7;
            dynamic_power = (dynamic_power * memory_load) >> 16;
            
            memory_power = base_power + dynamic_power;
        end else begin
            memory_power = 16'd0;
        end
    end

    // NoC power calculation
    always_comb begin
        automatic logic [15:0] base_power = 16'd0;
        automatic logic [15:0] dynamic_power = 16'd0;
        automatic logic [15:0] noc_load_estimate;
        
        // Estimate NoC load based on memory and AI activity
        noc_load_estimate = (memory_load + ai_accel_load) >> 1;
        
        if (noc_power_enable) begin
            base_power = (NOC_BASE_POWER * voltage_scale_factor) >> 7;
            dynamic_power = (NOC_DYNAMIC_POWER * voltage_scale_factor) >> 7;
            dynamic_power = (dynamic_power * frequency_scale_factor) >> 7;
            dynamic_power = (dynamic_power * noc_load_estimate) >> 16;
            
            noc_power = base_power + dynamic_power;
        end else begin
            noc_power = 16'd0;
        end
    end

    // Total power calculation
    always_comb begin
        logic [19:0] power_sum;
        
        power_sum = '0;
        for (int j = 0; j < NUM_CORES; j++) begin
            power_sum += {{4{1'b0}}, core_power[j]};
        end
        for (int j = 0; j < NUM_AI_UNITS; j++) begin
            power_sum += {{4{1'b0}}, ai_unit_power[j]};
        end
        power_sum += {{4{1'b0}}, memory_power};
        power_sum += {{4{1'b0}}, noc_power};
        
        current_total_power = power_sum[15:0];
        total_power = current_total_power;
    end

    // Power statistics calculation
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_counter <= '0;
            power_accumulator <= '0;
            power_samples <= '0;
            avg_power <= '0;
            peak_power <= '0;
            max_power_seen <= '0;
            energy_consumed <= '0;
        end else if (power_monitor_enable) begin
            sample_counter <= sample_counter + 1;
            
            // Track peak power
            if (current_total_power > max_power_seen) begin
                max_power_seen <= current_total_power;
                peak_power <= current_total_power;
            end
            
            // Accumulate energy (integrate power over time)
            // Energy in mJ = Power in mW * Time in ms
            // Assuming 1MHz clock, each cycle = 1us, so divide by 1000 for mJ
            if (sample_counter[9:0] == 10'd0) begin // Every 1024 cycles ≈ 1ms at 1MHz
                energy_consumed <= energy_consumed + {{26{1'b0}}, current_total_power[15:10]};
            end
            
            // Calculate average power over sampling period
            if (sample_counter >= power_sample_period_reg) begin
                power_accumulator <= power_accumulator + {{16{1'b0}}, current_total_power};
                power_samples <= power_samples + 1;
                
                if (power_samples > 0) begin
                    logic [31:0] avg_calc = power_accumulator / {{16{1'b0}}, power_samples};
                    avg_power <= avg_calc[15:0];
                end
                
                // Reset for next sampling period
                sample_counter <= '0;
                power_accumulator <= '0;
                power_samples <= '0;
            end else begin
                power_accumulator <= power_accumulator + {{16{1'b0}}, current_total_power};
            end
        end
    end

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            power_monitor_enable <= 1'b1;
            power_sample_period_reg <= POWER_SAMPLE_PERIOD;
            pm_config_rdata <= '0;
            pm_config_ready <= 1'b0;
        end else begin
            pm_config_ready <= pm_config_req;
            
            if (pm_config_req && pm_config_we) begin
                case (pm_config_addr[7:0])
                    8'h00: power_monitor_enable <= pm_config_wdata[0];
                    8'h04: power_sample_period_reg <= pm_config_wdata[15:0];
                    8'h08: begin
                        // Reset statistics
                        if (pm_config_wdata[0]) begin
                            power_accumulator <= '0;
                            power_samples <= '0;
                            max_power_seen <= '0;
                            energy_consumed <= '0;
                        end
                    end
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (pm_config_req && !pm_config_we) begin
                case (pm_config_addr[7:0])
                    8'h00: pm_config_rdata <= {31'b0, power_monitor_enable};
                    8'h04: pm_config_rdata <= {16'b0, power_sample_period_reg};
                    8'h08: pm_config_rdata <= {16'b0, current_total_power};
                    8'h0C: pm_config_rdata <= {16'b0, avg_power};
                    8'h10: pm_config_rdata <= {16'b0, peak_power};
                    8'h14: pm_config_rdata <= energy_consumed;
                    8'h18: pm_config_rdata <= {16'b0, power_samples};
                    8'h1C: pm_config_rdata <= sample_counter;
                    // Individual domain power readings
                    8'h20: pm_config_rdata <= {16'b0, core_power[0]};
                    8'h24: pm_config_rdata <= {16'b0, core_power[1]};
                    8'h28: pm_config_rdata <= {16'b0, core_power[2]};
                    8'h2C: pm_config_rdata <= {16'b0, core_power[3]};
                    8'h30: pm_config_rdata <= {16'b0, ai_unit_power[0]};
                    8'h34: pm_config_rdata <= {16'b0, ai_unit_power[1]};
                    8'h38: pm_config_rdata <= {16'b0, memory_power};
                    8'h3C: pm_config_rdata <= {16'b0, noc_power};
                    default: pm_config_rdata <= 32'h0;
                endcase
            end
        end
    end

endmodule