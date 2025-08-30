// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_ecc_controller.h for the primary calling header

#include "Vtest_ecc_controller__pch.h"
#include "Vtest_ecc_controller__Syms.h"
#include "Vtest_ecc_controller___024root.h"

void Vtest_ecc_controller___024root___ctor_var_reset(Vtest_ecc_controller___024root* vlSelf);

Vtest_ecc_controller___024root::Vtest_ecc_controller___024root(Vtest_ecc_controller__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , __VdlySched{*symsp->_vm_contextp__}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtest_ecc_controller___024root___ctor_var_reset(this);
}

void Vtest_ecc_controller___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vtest_ecc_controller___024root::~Vtest_ecc_controller___024root() {
}
