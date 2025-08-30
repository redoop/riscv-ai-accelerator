// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_simple_tpu_mac.h for the primary calling header

#include "Vtest_simple_tpu_mac__pch.h"
#include "Vtest_simple_tpu_mac__Syms.h"
#include "Vtest_simple_tpu_mac___024root.h"

void Vtest_simple_tpu_mac___024root___ctor_var_reset(Vtest_simple_tpu_mac___024root* vlSelf);

Vtest_simple_tpu_mac___024root::Vtest_simple_tpu_mac___024root(Vtest_simple_tpu_mac__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtest_simple_tpu_mac___024root___ctor_var_reset(this);
}

void Vtest_simple_tpu_mac___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtest_simple_tpu_mac___024root::~Vtest_simple_tpu_mac___024root() {
}
