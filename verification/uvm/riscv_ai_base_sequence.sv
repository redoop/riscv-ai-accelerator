// RISC-V AI Base Sequence
// Base sequence class for all AI accelerator test sequences

`ifndef RISCV_AI_BASE_SEQUENCE_SV
`define RISCV_AI_BASE_SEQUENCE_SV

class riscv_ai_base_sequence extends uvm_sequence #(riscv_ai_sequence_item);
    
    // Sequence configuration
    rand int num_transactions;
    rand int sequence_priority;
    
    // Performance tracking
    int total_cycles;
    int successful_transactions;
    int failed_transactions;
    
    // Constraints
    constraint num_trans_c {
        num_transactions inside {[10:1000]};
    }
    
    constraint priority_c {
        sequence_priority inside {[100:500]};
    }
    
    // UVM automation
    `uvm_object_utils_begin(riscv_ai_base_sequence)
        `uvm_field_int(num_transactions, UVM_ALL_ON)
        `uvm_field_int(sequence_priority, UVM_ALL_ON)
    `uvm_object_utils_end
    
    // Constructor
    function new(string name = "riscv_ai_base_sequence");
        super.new(name);
    endfunction
    
    // Pre-body task
    virtual task pre_body();
        super.pre_body();
        `uvm_info(get_type_name(), $sformatf("Starting sequence with %0d transactions", num_transactions), UVM_MEDIUM)
        total_cycles = 0;
        successful_transactions = 0;
        failed_transactions = 0;
    endtask
    
    // Body task - to be overridden by derived classes
    virtual task body();
        `uvm_error(get_type_name(), "Base sequence body() called - should be overridden")
    endtask
    
    // Post-body task
    virtual task post_body();
        super.post_body();
        `uvm_info(get_type_name(), 
                 $sformatf("Sequence completed: %0d successful, %0d failed, %0d total cycles", 
                          successful_transactions, failed_transactions, total_cycles), UVM_MEDIUM)
    endtask
    
    // Utility function to create and send transaction
    virtual task send_transaction(riscv_ai_sequence_item item);
        start_item(item);
        if (!item.randomize()) begin
            `uvm_error(get_type_name(), "Failed to randomize sequence item")
            failed_transactions++;
            return;
        end
        
        item.start_time = $time;
        finish_item(item);
        item.end_time = $time;
        item.latency = item.end_time - item.start_time;
        
        if (item.error) begin
            failed_transactions++;
            `uvm_warning(get_type_name(), $sformatf("Transaction failed: %s", item.convert2string()))
        end else begin
            successful_transactions++;
        end
        
        total_cycles += item.latency;
    endtask
    
    // Wait for specified number of clock cycles
    virtual task wait_cycles(int cycles);
        repeat(cycles) @(posedge m_sequencer.vif.clk);
    endtask
    
endclass : riscv_ai_base_sequence

`endif // RISCV_AI_BASE_SEQUENCE_SV