// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vtest_tpu_mac_simple__pch.h"
#include "Vtest_tpu_mac_simple.h"
#include "Vtest_tpu_mac_simple___024root.h"

// FUNCTIONS
Vtest_tpu_mac_simple__Syms::~Vtest_tpu_mac_simple__Syms()
{
}

Vtest_tpu_mac_simple__Syms::Vtest_tpu_mac_simple__Syms(VerilatedContext* contextp, const char* namep, Vtest_tpu_mac_simple* modelp)
    : VerilatedSyms{contextp}
    // Setup internal state of the Syms class
    , __Vm_modelp{modelp}
    // Setup module instances
    , TOP{this, namep}
{
        // Check resources
        Verilated::stackCheck(54);
    // Configure time unit / time precision
    _vm_contextp__->timeunit(-9);
    _vm_contextp__->timeprecision(-12);
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOP.__Vconfigure(true);
}
