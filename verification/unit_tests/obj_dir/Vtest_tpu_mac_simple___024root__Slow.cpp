// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_tpu_mac_simple.h for the primary calling header

#include "Vtest_tpu_mac_simple__pch.h"
#include "Vtest_tpu_mac_simple__Syms.h"
#include "Vtest_tpu_mac_simple___024root.h"

void Vtest_tpu_mac_simple___024root___ctor_var_reset(Vtest_tpu_mac_simple___024root* vlSelf);

Vtest_tpu_mac_simple___024root::Vtest_tpu_mac_simple___024root(Vtest_tpu_mac_simple__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtest_tpu_mac_simple___024root___ctor_var_reset(this);
}

void Vtest_tpu_mac_simple___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtest_tpu_mac_simple___024root::~Vtest_tpu_mac_simple___024root() {
}
