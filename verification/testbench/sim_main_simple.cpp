// Simple Verilator simulation main file
// Provides basic C++ testbench driver

#include <verilated.h>
#include <verilated_vcd_c.h>
#include <iostream>

// Include the Verilated model header (will be generated)
#include "Vtb_riscv_ai_chip_simple.h"

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Create instance of our module under test
    Vtb_riscv_ai_chip_simple* tb = new Vtb_riscv_ai_chip_simple;
    
    // Initialize trace dump
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("tb_riscv_ai_chip_simple.vcd");
    
    // Simulation variables
    vluint64_t sim_time = 0;
    const vluint64_t max_sim_time = 100000; // Maximum simulation cycles
    
    std::cout << "Starting RISC-V AI Chip simulation..." << std::endl;
    
    // Run simulation
    while (sim_time < max_sim_time && !Verilated::gotFinish()) {
        // Evaluate model
        tb->eval();
        
        // Dump trace data
        tfp->dump(sim_time);
        
        // Advance simulation time
        sim_time++;
    }
    
    // Final evaluation
    tb->final();
    
    // Close trace file
    tfp->close();
    
    // Cleanup
    delete tb;
    delete tfp;
    
    std::cout << "Simulation completed at time " << sim_time << std::endl;
    
    return 0;
}