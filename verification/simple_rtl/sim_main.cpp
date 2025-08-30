#include <verilated.h>
#include <verilated_vcd_c.h>
#include <iostream>
#include "Vsimple_mac_test.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    Vsimple_mac_test* tb = new Vsimple_mac_test;
    
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    tb->trace(tfp, 99);
    tfp->open("simple_mac_test.vcd");
    
    vluint64_t sim_time = 0;
    const vluint64_t max_sim_time = 50000;
    
    std::cout << "Starting simple MAC RTL simulation..." << std::endl;
    
    while (sim_time < max_sim_time && !Verilated::gotFinish()) {
        tb->eval();
        tfp->dump(sim_time);
        sim_time++;
    }
    
    tb->final();
    tfp->close();
    
    delete tb;
    delete tfp;
    
    std::cout << "RTL simulation completed at time " << sim_time << std::endl;
    return 0;
}