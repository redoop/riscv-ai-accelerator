// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vsimple_mac_test.h for the primary calling header

#include "Vsimple_mac_test__pch.h"
#include "Vsimple_mac_test__Syms.h"
#include "Vsimple_mac_test___024root.h"

VL_INLINE_OPT VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__2(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__2\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x1388ULL, 
                                             nullptr, 
                                             "simple_mac_test.sv", 
                                             22);
        vlSelfRef.simple_mac_test__DOT__clk = (1U & 
                                               (~ (IData)(vlSelfRef.simple_mac_test__DOT__clk)));
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vsimple_mac_test___024root___dump_triggers__act(Vsimple_mac_test___024root* vlSelf);
#endif  // VL_DEBUG

void Vsimple_mac_test___024root___eval_triggers__act(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_triggers__act\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.simple_mac_test__DOT__clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((~ (IData)(vlSelfRef.simple_mac_test__DOT__rst_n)) 
                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__rst_n__0)));
    vlSelfRef.__VactTriggered.setBit(2U, ((IData)(vlSelfRef.simple_mac_test__DOT__ready) 
                                          != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__ready__0)));
    vlSelfRef.__VactTriggered.setBit(3U, vlSelfRef.__VdlySched.awaitingCurrentTime());
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__clk__0 
        = vlSelfRef.simple_mac_test__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__rst_n__0 
        = vlSelfRef.simple_mac_test__DOT__rst_n;
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__ready__0 
        = vlSelfRef.simple_mac_test__DOT__ready;
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.__VactDidInit)))))) {
        vlSelfRef.__VactDidInit = 1U;
        vlSelfRef.__VactTriggered.setBit(2U, 1U);
    }
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vsimple_mac_test___024root___dump_triggers__act(vlSelf);
    }
#endif
}
