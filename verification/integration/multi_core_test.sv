// Multi-Core Coordination Test
// Tests coordination and synchronization between multiple CPU cores

`ifndef MULTI_CORE_TEST_SV
`define MULTI_CORE_TEST_SV

class multi_core_coordination_test extends system_integration_base;
    
    // Multi-core specific configuration
    int num_parallel_tasks = 8;
    int synchronization_points = 4;
    bit enable_load_balancing = 1;
    bit enable_work_stealing = 1;
    
    // Task tracking
    typedef struct {
        int task_id;
        int assigned_core;
        int actual_core;
        time start_time;
        time end_time;
        bit completed;
        int dependencies[];
    } task_info_t;
    
    task_info_t active_tasks[];
    task_info_t completed_tasks[];
    
    // Core synchronization
    bit core_sync_points[4];
    int core_task_counts[4];
    real core_load_balance[4];
    
    `uvm_component_utils_begin(multi_core_coordination_test)
        `uvm_field_int(num_parallel_tasks, UVM_ALL_ON)
        `uvm_field_int(synchronization_points, UVM_ALL_ON)
        `uvm_field_int(enable_load_balancing, UVM_ALL_ON)
        `uvm_field_int(enable_work_stealing, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "multi_core_coordination_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    virtual function void configure_test();
        super.configure_test();
        
        config.test_type = MULTI_CORE_COORDINATION;
        config.complexity = INTERMEDIATE_INTEGRATION;
        config.num_cores_active = 4;
        config.num_tpus_active = 0;
        config.num_vpus_active = 0;
        config.test_duration_ms = 200.0;
        config.max_concurrent_transactions = 128;
        config.workload_pattern = "parallel_compute";
        
        // Initialize core synchronization
        core_sync_points = '{0, 0, 0, 0};
        core_task_counts = '{0, 0, 0, 0};
        core_load_balance = '{0.0, 0.0, 0.0, 0.0};
        
        `uvm_info(get_type_name(), "Configured multi-core coordination test", UVM_MEDIUM)
    endfunction
    
    virtual task run_integration_test();
        `uvm_info(get_type_name(), "Running multi-core coordination test", UVM_MEDIUM)
        
        fork
            monitor_system_state();
            run_parallel_workload();
            monitor_core_synchronization();
            test_load_balancing();
            if (enable_work_stealing) test_work_stealing();
        join
    endtask
    
    // Run parallel workload across multiple cores
    virtual task run_parallel_workload();
        `uvm_info(get_type_name(), $sformatf("Starting %0d parallel tasks on %0d cores", 
                 num_parallel_tasks, config.num_cores_active), UVM_MEDIUM)
        
        // Initialize task array
        active_tasks = new[num_parallel_tasks];
        
        // Create and assign tasks
        for (int i = 0; i < num_parallel_tasks; i++) begin
            active_tasks[i].task_id = i;
            active_tasks[i].assigned_core = i % config.num_cores_active;
            active_tasks[i].actual_core = active_tasks[i].assigned_core;
            active_tasks[i].completed = 0;
            active_tasks[i].start_time = 0;
            active_tasks[i].end_time = 0;
            
            // Create task dependencies (some tasks depend on others)
            if (i > 0 && $urandom_range(0, 2) == 0) begin
                active_tasks[i].dependencies = new[1];
                active_tasks[i].dependencies[0] = $urandom_range(0, i-1);
            end else begin
                active_tasks[i].dependencies = new[0];
            end
        end
        
        // Execute tasks in parallel
        fork
            for (int core = 0; core < config.num_cores_active; core++) begin
                automatic int core_id = core;
                fork
                    execute_core_tasks(core_id);
                join_none
            end
        join
        
        // Wait for all tasks to complete
        wait_for_all_tasks_completion();
        
        `uvm_info(get_type_name(), "Parallel workload completed", UVM_MEDIUM)
    endtask
    
    // Execute tasks assigned to a specific core
    virtual task execute_core_tasks(int core_id);
        `uvm_info(get_type_name(), $sformatf("Core %0d starting task execution", core_id), UVM_HIGH)
        
        while (test_running) begin
            // Find next available task for this core
            int task_idx = find_next_task_for_core(core_id);
            
            if (task_idx >= 0) begin
                // Execute the task
                execute_single_task(task_idx, core_id);
                core_task_counts[core_id]++;
            end else begin
                // No tasks available, try work stealing if enabled
                if (enable_work_stealing) begin
                    task_idx = steal_task_from_other_core(core_id);
                    if (task_idx >= 0) begin
                        execute_single_task(task_idx, core_id);
                        core_task_counts[core_id]++;
                    end
                end
                
                // If still no task, wait a bit
                if (task_idx < 0) begin
                    repeat(10) @(posedge monitor.clk);
                end
            end
        end
    endtask
    
    // Find next available task for a core
    virtual function int find_next_task_for_core(int core_id);
        for (int i = 0; i < active_tasks.size(); i++) begin
            if (!active_tasks[i].completed && 
                active_tasks[i].assigned_core == core_id &&
                active_tasks[i].start_time == 0 &&
                check_task_dependencies(i)) begin
                return i;
            end
        end
        return -1;
    endfunction
    
    // Check if task dependencies are satisfied
    virtual function bit check_task_dependencies(int task_idx);
        for (int i = 0; i < active_tasks[task_idx].dependencies.size(); i++) begin
            int dep_task = active_tasks[task_idx].dependencies[i];
            if (!active_tasks[dep_task].completed) begin
                return 0;
            end
        end
        return 1;
    endfunction
    
    // Steal task from another core (work stealing)
    virtual function int steal_task_from_other_core(int stealing_core);
        // Find the most loaded core
        int max_load_core = 0;
        int max_pending_tasks = 0;
        
        for (int core = 0; core < config.num_cores_active; core++) begin
            if (core != stealing_core) begin
                int pending_tasks = count_pending_tasks_for_core(core);
                if (pending_tasks > max_pending_tasks) begin
                    max_pending_tasks = pending_tasks;
                    max_load_core = core;
                end
            end
        end
        
        // Steal a task from the most loaded core
        if (max_pending_tasks > 1) begin
            for (int i = 0; i < active_tasks.size(); i++) begin
                if (!active_tasks[i].completed && 
                    active_tasks[i].assigned_core == max_load_core &&
                    active_tasks[i].start_time == 0 &&
                    check_task_dependencies(i)) begin
                    
                    // Reassign task to stealing core
                    active_tasks[i].assigned_core = stealing_core;
                    `uvm_info(get_type_name(), $sformatf("Core %0d stole task %0d from core %0d", 
                             stealing_core, i, max_load_core), UVM_HIGH)
                    return i;
                end
            end
        end
        
        return -1;
    endfunction
    
    // Count pending tasks for a core
    virtual function int count_pending_tasks_for_core(int core_id);
        int count = 0;
        for (int i = 0; i < active_tasks.size(); i++) begin
            if (!active_tasks[i].completed && 
                active_tasks[i].assigned_core == core_id &&
                active_tasks[i].start_time == 0) begin
                count++;
            end
        end
        return count;
    endfunction
    
    // Execute a single task
    virtual task execute_single_task(int task_idx, int executing_core);
        active_tasks[task_idx].start_time = $time;
        active_tasks[task_idx].actual_core = executing_core;
        
        `uvm_info(get_type_name(), $sformatf("Core %0d executing task %0d", executing_core, task_idx), UVM_HIGH)
        
        // Simulate task execution time (variable based on task complexity)
        int execution_cycles = $urandom_range(100, 1000);
        repeat(execution_cycles) @(posedge monitor.clk);
        
        // Mark task as completed
        active_tasks[task_idx].end_time = $time;
        active_tasks[task_idx].completed = 1;
        
        // Move to completed tasks
        completed_tasks = new[completed_tasks.size() + 1](completed_tasks);
        completed_tasks[completed_tasks.size() - 1] = active_tasks[task_idx];
        
        `uvm_info(get_type_name(), $sformatf("Task %0d completed on core %0d", task_idx, executing_core), UVM_HIGH)
    endtask
    
    // Wait for all tasks to complete
    virtual task wait_for_all_tasks_completion();
        while (completed_tasks.size() < num_parallel_tasks) begin
            repeat(100) @(posedge monitor.clk);
        end
        `uvm_info(get_type_name(), "All parallel tasks completed", UVM_MEDIUM)
    endtask
    
    // Monitor core synchronization
    virtual task monitor_core_synchronization();
        for (int sync_point = 0; sync_point < synchronization_points; sync_point++) begin
            // Wait for synchronization interval
            repeat(1000) @(posedge monitor.clk);
            
            // Simulate synchronization barrier
            `uvm_info(get_type_name(), $sformatf("Synchronization point %0d", sync_point), UVM_MEDIUM)
            
            // All cores reach synchronization point
            fork
                for (int core = 0; core < config.num_cores_active; core++) begin
                    automatic int core_id = core;
                    fork
                        begin
                            // Simulate core reaching sync point
                            repeat($urandom_range(10, 100)) @(posedge monitor.clk);
                            core_sync_points[core_id] = 1;
                            `uvm_info(get_type_name(), $sformatf("Core %0d reached sync point %0d", core_id, sync_point), UVM_HIGH)
                        end
                    join_none
                end
            join
            
            // Wait for all cores to reach sync point
            while (!all_cores_synchronized()) begin
                repeat(10) @(posedge monitor.clk);
            end
            
            `uvm_info(get_type_name(), $sformatf("All cores synchronized at point %0d", sync_point), UVM_MEDIUM)
            
            // Reset sync points for next iteration
            core_sync_points = '{0, 0, 0, 0};
        end
    endtask
    
    // Check if all cores are synchronized
    virtual function bit all_cores_synchronized();
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (!core_sync_points[i]) return 0;
        end
        return 1;
    endfunction
    
    // Test load balancing
    virtual task test_load_balancing();
        if (!enable_load_balancing) return;
        
        `uvm_info(get_type_name(), "Testing load balancing", UVM_MEDIUM)
        
        // Monitor load balance periodically
        while (test_running) begin
            repeat(2000) @(posedge monitor.clk);
            
            // Calculate load balance metrics
            calculate_load_balance();
            
            // Check if load is reasonably balanced
            check_load_balance_quality();
        end
    endtask
    
    // Calculate load balance across cores
    virtual function void calculate_load_balance();
        int total_tasks = 0;
        for (int i = 0; i < config.num_cores_active; i++) begin
            total_tasks += core_task_counts[i];
        end
        
        if (total_tasks > 0) begin
            real average_load = real'(total_tasks) / real'(config.num_cores_active);
            for (int i = 0; i < config.num_cores_active; i++) begin
                core_load_balance[i] = real'(core_task_counts[i]) / average_load;
            end
        end
    endfunction
    
    // Check load balance quality
    virtual function void check_load_balance_quality();
        real max_load = 0.0;
        real min_load = 999.0;
        
        for (int i = 0; i < config.num_cores_active; i++) begin
            if (core_load_balance[i] > max_load) max_load = core_load_balance[i];
            if (core_load_balance[i] < min_load) min_load = core_load_balance[i];
        end
        
        real load_imbalance = max_load - min_load;
        
        if (load_imbalance > 0.5) begin  // More than 50% imbalance
            `uvm_warning(get_type_name(), $sformatf("Load imbalance detected: %.2f", load_imbalance))
            error_count++;
            error_messages.push_back($sformatf("Load imbalance: %.2f", load_imbalance));
        end
        
        `uvm_info(get_type_name(), $sformatf("Load balance - Max: %.2f, Min: %.2f, Imbalance: %.2f", 
                 max_load, min_load, load_imbalance), UVM_HIGH)
    endfunction
    
    // Test work stealing mechanism
    virtual task test_work_stealing();
        `uvm_info(get_type_name(), "Testing work stealing mechanism", UVM_MEDIUM)
        
        // Create artificial load imbalance
        create_load_imbalance();
        
        // Monitor work stealing effectiveness
        monitor_work_stealing_effectiveness();
    endtask
    
    // Create artificial load imbalance for testing
    virtual task create_load_imbalance();
        // Assign more tasks to core 0
        for (int i = 0; i < num_parallel_tasks / 2; i++) begin
            if (i < active_tasks.size()) begin
                active_tasks[i].assigned_core = 0;
            end
        end
        
        `uvm_info(get_type_name(), "Created artificial load imbalance for work stealing test", UVM_HIGH)
    endtask
    
    // Monitor work stealing effectiveness
    virtual task monitor_work_stealing_effectiveness();
        int initial_imbalance = count_pending_tasks_for_core(0);
        
        // Wait for work stealing to take effect
        repeat(5000) @(posedge monitor.clk);
        
        int final_imbalance = count_pending_tasks_for_core(0);
        
        if (final_imbalance < initial_imbalance) begin
            `uvm_info(get_type_name(), $sformatf("Work stealing effective: reduced imbalance from %0d to %0d", 
                     initial_imbalance, final_imbalance), UVM_MEDIUM)
        end else begin
            `uvm_warning(get_type_name(), "Work stealing not effective")
            error_count++;
            error_messages.push_back("Work stealing ineffective");
        end
    endtask
    
    // Additional checks specific to multi-core test
    virtual function void perform_additional_checks();
        super.perform_additional_checks();
        
        // Check task completion
        if (completed_tasks.size() != num_parallel_tasks) begin
            test_passed = 0;
            `uvm_error(get_type_name(), $sformatf("Not all tasks completed: %0d/%0d", 
                     completed_tasks.size(), num_parallel_tasks))
        end
        
        // Check synchronization effectiveness
        check_synchronization_effectiveness();
        
        // Check core utilization balance
        check_core_utilization_balance();
    endfunction
    
    // Check synchronization effectiveness
    virtual function void check_synchronization_effectiveness();
        // Analyze task completion times to verify synchronization
        time max_completion_time = 0;
        time min_completion_time = 999999999;
        
        foreach (completed_tasks[i]) begin
            time completion_time = completed_tasks[i].end_time - completed_tasks[i].start_time;
            if (completion_time > max_completion_time) max_completion_time = completion_time;
            if (completion_time < min_completion_time) min_completion_time = completion_time;
        end
        
        real sync_efficiency = real'(min_completion_time) / real'(max_completion_time);
        
        if (sync_efficiency < 0.5) begin
            `uvm_warning(get_type_name(), $sformatf("Poor synchronization efficiency: %.2f", sync_efficiency))
        end
        
        `uvm_info(get_type_name(), $sformatf("Synchronization efficiency: %.2f", sync_efficiency), UVM_MEDIUM)
    endfunction
    
    // Check core utilization balance
    virtual function void check_core_utilization_balance();
        real total_utilization = 0.0;
        for (int i = 0; i < config.num_cores_active; i++) begin
            total_utilization += current_state.cpu_utilization[i];
        end
        
        real avg_utilization = total_utilization / real'(config.num_cores_active);
        real max_deviation = 0.0;
        
        for (int i = 0; i < config.num_cores_active; i++) begin
            real deviation = $abs(current_state.cpu_utilization[i] - avg_utilization);
            if (deviation > max_deviation) max_deviation = deviation;
        end
        
        if (max_deviation > 0.3) begin  // More than 30% deviation
            `uvm_warning(get_type_name(), $sformatf("High core utilization imbalance: %.2f", max_deviation))
        end
        
        `uvm_info(get_type_name(), $sformatf("Core utilization balance - Avg: %.2f, Max deviation: %.2f", 
                 avg_utilization, max_deviation), UVM_MEDIUM)
    endfunction
    
    // Enhanced reporting for multi-core test
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== MULTI-CORE COORDINATION DETAILS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Parallel Tasks: %0d", num_parallel_tasks), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Synchronization Points: %0d", synchronization_points), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Load Balancing: %s", enable_load_balancing ? "Enabled" : "Disabled"), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Work Stealing: %s", enable_work_stealing ? "Enabled" : "Disabled"), UVM_LOW)
        
        `uvm_info(get_type_name(), "Core Task Distribution:", UVM_LOW)
        for (int i = 0; i < config.num_cores_active; i++) begin
            `uvm_info(get_type_name(), $sformatf("  Core %0d: %0d tasks (Load: %.2f)", 
                     i, core_task_counts[i], core_load_balance[i]), UVM_LOW)
        end
        
        `uvm_info(get_type_name(), $sformatf("Completed Tasks: %0d/%0d", completed_tasks.size(), num_parallel_tasks), UVM_LOW)
    endfunction
    
endclass : multi_core_coordination_test

`endif // MULTI_CORE_TEST_SV