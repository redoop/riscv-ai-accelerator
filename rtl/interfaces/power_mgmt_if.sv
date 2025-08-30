/*
 * Power Management Interface
 * 
 * SystemVerilog interface for power management communication
 * between the power manager and various processing units.
 */

interface power_mgmt_if;
    
    // Power control signals
    logic        power_enable;      // Enable power to the unit
    logic        clock_enable;      // Enable clock to the unit
    logic        reset_n;           // Reset signal (active low)
    
    // Power state requests
    logic [2:0]  power_state_req;   // Requested power state
    logic [2:0]  power_state_ack;   // Acknowledged power state
    
    // Load and activity monitoring
    logic [15:0] load_level;        // Current load level (0-65535)
    logic        activity;          // Activity indicator
    logic        idle;              // Idle indicator
    
    // Voltage and frequency control
    logic [2:0]  voltage_level;     // Voltage level (0-7)
    logic [2:0]  frequency_level;   // Frequency level (0-7)
    
    // Status and control
    logic        power_good;        // Power is stable and good
    logic        clock_stable;      // Clock is stable
    logic        transition_busy;   // Power transition in progress
    
    // Temperature monitoring
    logic [7:0]  temperature;       // Temperature reading
    logic        thermal_alert;     // Thermal alert
    
    // Power gating control
    logic        isolation_enable;  // Isolation enable
    logic        retention_enable;  // Retention enable
    
    // Modport for power manager (controller)
    modport controller (
        output power_enable,
        output clock_enable,
        output reset_n,
        output power_state_req,
        input  power_state_ack,
        input  load_level,
        input  activity,
        input  idle,
        output voltage_level,
        output frequency_level,
        input  power_good,
        input  clock_stable,
        input  transition_busy,
        input  temperature,
        input  thermal_alert,
        output isolation_enable,
        output retention_enable
    );
    
    // Modport for processing units (target)
    modport target (
        input  power_enable,
        input  clock_enable,
        input  reset_n,
        input  power_state_req,
        output power_state_ack,
        output load_level,
        output activity,
        output idle,
        input  voltage_level,
        input  frequency_level,
        output power_good,
        output clock_stable,
        output transition_busy,
        output temperature,
        output thermal_alert,
        input  isolation_enable,
        input  retention_enable
    );

endinterface