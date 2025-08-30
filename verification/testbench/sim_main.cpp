// Verilator simulation main file
// Provides C++ testbench driver for SystemVerilog testbench

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vtb_riscv_ai_chip.h"

#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Create DUT instance
    auto dut = std::make_unique<Vtb_riscv_ai_chip>();
    
    // Initialize tracing
    Verilated::traceEverOn(true);
    auto tfp = std::make_unique<VerilatedVcdC>();
    dut->trace(tfp.get(), 99);
    tfp->open("tb_riscv_ai_chip.vcd");
    
    // Simulation parameters
    const int MAX_CYCLES = 100000;
    int cycle = 0;
    
    std::cout << "Starting RISC-V AI Chip simulation..." << std::endl;
    
    // Main simulation loop
    while (cycle < MAX_CYCLES && !Verilated::gotFinish()) {
        // Evaluate model
        dut->eval();
        
        // Dump trace
        tfp->dump(cycle);
        
        cycle++;
        
        // Print progress every 10000 cycles
        if (cycle % 10000 == 0) {
            std::cout << "Cycle: " << cycle << std::endl;
        }
    }
    
    // Cleanup
    tfp->close();
    
    std::cout << "Simulation completed after " << cycle << " cycles" << std::endl;
    
    return 0;
}