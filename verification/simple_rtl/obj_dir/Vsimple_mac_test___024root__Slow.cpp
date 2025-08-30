// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vsimple_mac_test.h for the primary calling header

#include "Vsimple_mac_test__pch.h"
#include "Vsimple_mac_test__Syms.h"
#include "Vsimple_mac_test___024root.h"

void Vsimple_mac_test___024root___ctor_var_reset(Vsimple_mac_test___024root* vlSelf);

Vsimple_mac_test___024root::Vsimple_mac_test___024root(Vsimple_mac_test__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vsimple_mac_test___024root___ctor_var_reset(this);
}

void Vsimple_mac_test___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vsimple_mac_test___024root::~Vsimple_mac_test___024root() {
}
