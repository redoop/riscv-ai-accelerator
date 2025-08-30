// System Monitor
// Monitors system-wide state and performance metrics

`ifndef SYSTEM_MONITOR_SV
`define SYSTEM_MONITOR_SV

class system_monitor extends uvm_component;
    
    // Virtual interface for monitoring
    virtual riscv_ai_interface vif;
    
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Monitoring configuration
    bit enable_performance_monitoring = 1;
    bit enable_power_monitoring = 1;
    bit enable_thermal_monitoring = 1;
    bit enable_error_monitoring = 1;
    
    // Performance counters
    int total_transactions = 0;
    int successful_transactions = 0;
    int failed_transactions = 0;
    time total_latency = 0;
    
    // System health metrics
    real current_power_watts = 10.0;
    real current_temperature_c = 45.0;
    real peak_power_watts = 10.0;
    real peak_temperature_c = 45.0;
    
    // Error tracking
    int error_count = 0;
    string error_log[$];
    
    `uvm_component_utils_begin(system_monitor)
        `uvm_field_int(enable_performance_monitoring, UVM_ALL_ON)
        `uvm_field_int(enable_power_monitoring, UVM_ALL_ON)
        `uvm_field_int(enable_thermal_monitoring, UVM_ALL_ON)
        `uvm_field_int(total_transactions, UVM_ALL_ON)
        `uvm_field_real(current_power_watts, UVM_ALL_ON)
        `uvm_field_real(current_temperature_c, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "system_monitor", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        // Get virtual interface
        if (!uvm_config_db#(virtual riscv_ai_interface)::get(this, "", "vif", vif)) begin
            `uvm_warning(get_type_name(), "Virtual interface not found, using internal clock")
        end else begin
            clk = vif.clk;
            rst_n = vif.rst_n;
        end
    endfunction
    
    virtual task run_phase(uvm_phase phase);
        fork
            if (enable_performance_monitoring) monitor_performance();
            if (enable_power_monitoring) monitor_power();
            if (enable_thermal_monitoring) monitor_thermal();
            if (enable_error_monitoring) monitor_errors();
        join_none
    endtask
    
    // Monitor performance metrics
    virtual task monitor_performance();
        forever begin
            @(posedge clk);
            
            if (vif != null && vif.valid && vif.ready) begin
                total_transactions++;
                time start_time = $time;
                
                // Wait for response
                @(posedge vif.response_valid);
                time end_time = $time;
                
                total_latency += (end_time - start_time);
                
                if (vif.error) begin
                    failed_transactions++;
                end else begin
                    successful_transactions++;
                end
            end
        end
    endtask
    
    // Monitor power consumption
    virtual task monitor_power();
        forever begin
            repeat(100) @(posedge clk);
            
            // Simulate power monitoring
            current_power_watts = 10.0 + $urandom_range(0, 50);
            if (current_power_watts > peak_power_watts) begin
                peak_power_watts = current_power_watts;
            end
            
            // Check power limits
            if (current_power_watts > 100.0) begin
                log_error($sformatf("Power consumption too high: %.1fW", current_power_watts));
            end
        end
    endtask
    
    // Monitor thermal conditions
    virtual task monitor_thermal();
        forever begin
            repeat(200) @(posedge clk);
            
            // Thermal follows power with inertia
            real target_temp = 45.0 + (current_power_watts - 10.0) * 0.8;
            current_temperature_c = current_temperature_c * 0.95 + target_temp * 0.05;
            
            if (current_temperature_c > peak_temperature_c) begin
                peak_temperature_c = current_temperature_c;
            end
            
            // Check thermal limits
            if (current_temperature_c > 85.0) begin
                log_error($sformatf("Temperature too high: %.1f°C", current_temperature_c));
            end
        end
    endtask
    
    // Monitor system errors
    virtual task monitor_errors();
        forever begin
            @(posedge clk);
            
            if (vif != null && vif.error) begin
                error_count++;
                log_error($sformatf("System error detected at time %0t", $time));
            end
        end
    endtask
    
    // Log error with timestamp
    virtual function void log_error(string message);
        string timestamped_msg = $sformatf("[%0t] %s", $time, message);
        error_log.push_back(timestamped_msg);
        `uvm_error(get_type_name(), timestamped_msg);
    endfunction
    
    // Get performance summary
    virtual function string get_performance_summary();
        real avg_latency = (total_transactions > 0) ? real'(total_latency) / real'(total_transactions) / 1e6 : 0.0;
        real success_rate = (total_transactions > 0) ? real'(successful_transactions) / real'(total_transactions) * 100.0 : 0.0;
        
        return $sformatf("Transactions: %0d, Success Rate: %.1f%%, Avg Latency: %.2fms, Peak Power: %.1fW, Peak Temp: %.1f°C", 
                        total_transactions, success_rate, avg_latency, peak_power_watts, peak_temperature_c);
    endfunction
    
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== SYSTEM MONITOR REPORT ===", UVM_LOW)
        `uvm_info(get_type_name(), get_performance_summary(), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Errors: %0d", error_count), UVM_LOW)
        
        if (error_log.size() > 0) begin
            `uvm_info(get_type_name(), "Recent Errors:", UVM_LOW)
            for (int i = error_log.size() - 5; i < error_log.size(); i++) begin
                if (i >= 0) `uvm_info(get_type_name(), error_log[i], UVM_LOW);
            end
        end
    endfunction
    
endclass : system_monitor

`endif // SYSTEM_MONITOR_SV