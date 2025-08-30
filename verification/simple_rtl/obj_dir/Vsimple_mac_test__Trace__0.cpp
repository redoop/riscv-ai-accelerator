// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vsimple_mac_test__Syms.h"


void Vsimple_mac_test___024root__trace_chg_0_sub_0(Vsimple_mac_test___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vsimple_mac_test___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_chg_0\n"); );
    // Init
    Vsimple_mac_test___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vsimple_mac_test___024root*>(voidSelf);
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vsimple_mac_test___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vsimple_mac_test___024root__trace_chg_0_sub_0(Vsimple_mac_test___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_chg_0_sub_0\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(((vlSelfRef.__Vm_traceActivity[1U] 
                      | vlSelfRef.__Vm_traceActivity
                      [2U])))) {
        bufp->chgBit(oldp+0,(vlSelfRef.simple_mac_test__DOT__rst_n));
        bufp->chgSData(oldp+1,(vlSelfRef.simple_mac_test__DOT__a),16);
        bufp->chgSData(oldp+2,(vlSelfRef.simple_mac_test__DOT__b),16);
        bufp->chgIData(oldp+3,(vlSelfRef.simple_mac_test__DOT__c),32);
        bufp->chgBit(oldp+4,(vlSelfRef.simple_mac_test__DOT__valid));
    }
    bufp->chgBit(oldp+5,(vlSelfRef.simple_mac_test__DOT__clk));
    bufp->chgIData(oldp+6,(vlSelfRef.simple_mac_test__DOT__result),32);
    bufp->chgBit(oldp+7,(vlSelfRef.simple_mac_test__DOT__ready));
}

void Vsimple_mac_test___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_cleanup\n"); );
    // Init
    Vsimple_mac_test___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vsimple_mac_test___024root*>(voidSelf);
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
}
