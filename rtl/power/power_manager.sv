/*
 * Power Management Unit (PMU)
 * 
 * Top-level power management module that coordinates DVFS, power gating,
 * and thermal management for the RISC-V AI chip.
 */

module power_manager #(
    parameter NUM_CORES = 4,
    parameter NUM_AI_UNITS = 2,
    parameter LOAD_MONITOR_WIDTH = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    
    // Power management control outputs (simplified for synthesis)
    output logic [3:0]         core_voltage_level_0,
    output logic [3:0]         core_voltage_level_1,
    output logic [3:0]         core_voltage_level_2,
    output logic [3:0]         core_voltage_level_3,
    output logic [7:0]         core_freq_divider_0,
    output logic [7:0]         core_freq_divider_1,
    output logic [7:0]         core_freq_divider_2,
    output logic [7:0]         core_freq_divider_3,
    output logic               core_power_gate_en_0,
    output logic               core_power_gate_en_1,
    output logic               core_power_gate_en_2,
    output logic               core_power_gate_en_3,
    output logic               core_clock_gate_en_0,
    output logic               core_clock_gate_en_1,
    output logic               core_clock_gate_en_2,
    output logic               core_clock_gate_en_3,
    
    output logic [3:0]         tpu_voltage_level_0,
    output logic [3:0]         tpu_voltage_level_1,
    output logic [7:0]         tpu_freq_divider_0,
    output logic [7:0]         tpu_freq_divider_1,
    output logic               tpu_power_gate_en_0,
    output logic               tpu_power_gate_en_1,
    
    output logic [3:0]         vpu_voltage_level_0,
    output logic [3:0]         vpu_voltage_level_1,
    output logic [7:0]         vpu_freq_divider_0,
    output logic [7:0]         vpu_freq_divider_1,
    output logic               vpu_power_gate_en_0,
    output logic               vpu_power_gate_en_1,
    
    // DVFS control inputs
    input  logic                           dvfs_enable,
    input  logic [LOAD_MONITOR_WIDTH-1:0] core_load_0,
    input  logic [LOAD_MONITOR_WIDTH-1:0] core_load_1,
    input  logic [LOAD_MONITOR_WIDTH-1:0] core_load_2,
    input  logic [LOAD_MONITOR_WIDTH-1:0] core_load_3,
    input  logic                           core_active_0,
    input  logic                           core_active_1,
    input  logic                           core_active_2,
    input  logic                           core_active_3,
    input  logic [LOAD_MONITOR_WIDTH-1:0] memory_load,
    input  logic [LOAD_MONITOR_WIDTH-1:0] noc_load,
    input  logic [LOAD_MONITOR_WIDTH-1:0] ai_accel_load,
    
    // Activity monitoring
    input  logic [NUM_CORES-1:0]          core_activity,
    input  logic                           memory_activity,
    input  logic [NUM_AI_UNITS-1:0]       ai_unit_activity,
    
    // Global power control
    output logic [3:0]          global_voltage,
    output logic [7:0]          global_freq_div,
    
    // Enhanced power domain control
    output logic [NUM_CORES-1:0]          core_power_enable,
    output logic                           l1_cache_power_enable,
    output logic                           l2_cache_power_enable,
    output logic                           memory_ctrl_power_enable,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_power_enable,
    output logic                           noc_power_enable,
    
    // Isolation control
    output logic [NUM_CORES-1:0]          core_isolation_enable,
    output logic                           memory_isolation_enable,
    output logic [NUM_AI_UNITS-1:0]       ai_unit_isolation_enable,
    
    // Thermal sensors
    input  logic [15:0]         temp_sensor_0,
    input  logic [15:0]         temp_sensor_1,
    input  logic [15:0]         temp_sensor_2,
    input  logic [15:0]         temp_sensor_3,
    input  logic [15:0]         temp_sensor_4,
    input  logic [15:0]         temp_sensor_5,
    input  logic [15:0]         temp_sensor_6,
    input  logic [15:0]         temp_sensor_7,
    
    // External voltage regulator interface
    output logic                           vreg_scl,
    inout  logic                           vreg_sda,
    
    // Clock generation
    input  logic                           ref_clk,
    output logic                           cpu_clk,
    output logic                           ai_accel_clk,
    output logic                           memory_clk,
    output logic                           noc_clk,
    
    // Configuration interface
    input  logic [31:0]         pm_config_addr,
    input  logic [31:0]         pm_config_wdata,
    output logic [31:0]         pm_config_rdata,
    input  logic                pm_config_req,
    input  logic                pm_config_we,
    output logic                pm_config_ready
);

    // Internal signals
    logic [7:0] average_temperature;
    logic thermal_alert;
    logic [2:0] voltage_level_req;
    logic [2:0] frequency_level_req;
    logic [2:0] current_voltage_level;
    logic [2:0] current_frequency_level;
    logic [NUM_CORES-1:0] core_power_gate_req;
    logic memory_power_gate_req;
    logic ai_accel_power_gate_req;
    logic voltage_ready;
    logic frequency_ready;
    logic pll_locked;
    logic dvfs_transition_busy;
    logic power_transition_busy;
    logic [7:0] power_domain_status;
    
    // Configuration registers
    logic [2:0] min_voltage_level;
    logic [2:0] max_voltage_level;
    logic dvfs_enable_reg;

    // Calculate average temperature from sensors
    always_comb begin
        logic [18:0] temp_sum;
        temp_sum = {{11{1'b0}}, temp_sensor_0[15:8]} + 
                   {{11{1'b0}}, temp_sensor_1[15:8]} + 
                   {{11{1'b0}}, temp_sensor_2[15:8]} + 
                   {{11{1'b0}}, temp_sensor_3[15:8]} + 
                   {{11{1'b0}}, temp_sensor_4[15:8]} + 
                   {{11{1'b0}}, temp_sensor_5[15:8]} + 
                   {{11{1'b0}}, temp_sensor_6[15:8]} + 
                   {{11{1'b0}}, temp_sensor_7[15:8]};
        average_temperature = temp_sum[10:3];  // Divide by 8, proper width
        thermal_alert = (average_temperature > 8'd85);  // 85°C threshold
    end

    // DVFS Controller
    dvfs_controller #(
        .NUM_CORES(NUM_CORES),
        .LOAD_MONITOR_WIDTH(LOAD_MONITOR_WIDTH)
    ) u_dvfs_controller (
        .clk(clk),
        .rst_n(rst_n),
        .core_load(core_load),
        .core_active(core_active),
        .memory_load(memory_load),
        .noc_load(noc_load),
        .ai_accel_load(ai_accel_load),
        .temperature(average_temperature),
        .thermal_alert(thermal_alert),
        .voltage_level(voltage_level_req),
        .frequency_level(frequency_level_req),
        .core_power_gate(core_power_gate_req),
        .memory_power_gate(memory_power_gate_req),
        .ai_accel_power_gate(ai_accel_power_gate_req),
        .dvfs_enable(dvfs_enable_reg),
        .min_voltage_level(min_voltage_level),
        .max_voltage_level(max_voltage_level),
        .dvfs_transition_busy(dvfs_transition_busy),
        .power_state()  // Not used in top level
    );

    // Voltage Regulator Controller
    voltage_regulator u_voltage_regulator (
        .clk(clk),
        .rst_n(rst_n),
        .voltage_level_req(voltage_level_req),
        .voltage_enable(dvfs_enable_reg),
        .vreg_scl(vreg_scl),
        .vreg_sda(vreg_sda),
        .voltage_ready(voltage_ready),
        .current_voltage_level(current_voltage_level),
        .voltage_fault()  // Not used
    );

    // Frequency Controller
    frequency_controller u_frequency_controller (
        .ref_clk(ref_clk),
        .rst_n(rst_n),
        .frequency_level_req(frequency_level_req),
        .frequency_enable(dvfs_enable_reg),
        .cpu_clk(cpu_clk),
        .ai_accel_clk(ai_accel_clk),
        .memory_clk(memory_clk),
        .noc_clk(noc_clk),
        .frequency_ready(frequency_ready),
        .current_frequency_level(current_frequency_level),
        .pll_locked(pll_locked)
    );

    // Power Domain Controller
    power_domain_controller #(
        .NUM_CORES(NUM_CORES),
        .NUM_AI_UNITS(NUM_AI_UNITS)
    ) u_power_domain_controller (
        .clk(clk),
        .rst_n(rst_n),
        .core_power_gate_req(core_power_gate_req),
        .memory_power_gate_req(memory_power_gate_req),
        .ai_accel_power_gate_req(ai_accel_power_gate_req),
        .core_activity(core_activity),
        .memory_activity(memory_activity),
        .ai_unit_activity(ai_unit_activity),
        .core_power_enable(core_power_enable),
        .l1_cache_power_enable(l1_cache_power_enable),
        .l2_cache_power_enable(l2_cache_power_enable),
        .memory_ctrl_power_enable(memory_ctrl_power_enable),
        .ai_unit_power_enable(ai_unit_power_enable),
        .noc_power_enable(noc_power_enable),
        .core_isolation_enable(core_isolation_enable),
        .memory_isolation_enable(memory_isolation_enable),
        .ai_unit_isolation_enable(ai_unit_isolation_enable),
        .power_domain_status(power_domain_status),
        .power_transition_busy(power_transition_busy)
    );

    // Configuration register interface
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            min_voltage_level <= 3'd1;
            max_voltage_level <= 3'd7;
            dvfs_enable_reg <= 1'b0;
            pm_config_rdata <= '0;
            pm_config_ready <= 1'b0;
        end else begin
            pm_config_ready <= pm_config_req;
            
            if (pm_config_req && pm_config_we) begin
                case (pm_config_addr[7:0])
                    8'h00: dvfs_enable_reg <= pm_config_wdata[0];
                    8'h04: min_voltage_level <= pm_config_wdata[2:0];
                    8'h08: max_voltage_level <= pm_config_wdata[2:0];
                    default: begin
                        // Invalid address - do nothing
                    end
                endcase
            end
            
            if (pm_config_req && !pm_config_we) begin
                case (pm_config_addr[7:0])
                    8'h00: pm_config_rdata <= {31'b0, dvfs_enable_reg};
                    8'h04: pm_config_rdata <= {29'b0, min_voltage_level};
                    8'h08: pm_config_rdata <= {29'b0, max_voltage_level};
                    8'h0C: pm_config_rdata <= {29'b0, current_voltage_level};
                    8'h10: pm_config_rdata <= {29'b0, current_frequency_level};
                    8'h14: pm_config_rdata <= {24'b0, average_temperature};
                    8'h18: pm_config_rdata <= {24'b0, power_domain_status};
                    8'h1C: pm_config_rdata <= {30'b0, dvfs_transition_busy, power_transition_busy};
                    default: pm_config_rdata <= 32'h0;
                endcase
            end
        end
    end

    // Legacy interface compatibility
    assign global_voltage = {1'b0, current_voltage_level};
    assign global_freq_div = {5'b0, current_frequency_level};

endmodule