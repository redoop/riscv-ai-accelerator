// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_simple_tpu_mac.h for the primary calling header

#include "Vtest_simple_tpu_mac__pch.h"
#include "Vtest_simple_tpu_mac__Syms.h"
#include "Vtest_simple_tpu_mac___024root.h"

VL_INLINE_OPT VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__2(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__2\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x1388ULL, 
                                             nullptr, 
                                             "test_simple_tpu_mac.sv", 
                                             24);
        vlSelfRef.test_simple_tpu_mac__DOT__clk = (1U 
                                                   & (~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__clk)));
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___dump_triggers__act(Vtest_simple_tpu_mac___024root* vlSelf);
#endif  // VL_DEBUG

void Vtest_simple_tpu_mac___024root___eval_triggers__act(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_triggers__act\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.test_simple_tpu_mac__DOT__clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__rst_n)) 
                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__rst_n__0)));
    vlSelfRef.__VactTriggered.setBit(2U, ((IData)(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid) 
                                          != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__dut__DOT__result_valid__0)));
    vlSelfRef.__VactTriggered.setBit(3U, vlSelfRef.__VdlySched.awaitingCurrentTime());
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__clk__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__rst_n__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__rst_n;
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__dut__DOT__result_valid__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid;
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.__VactDidInit)))))) {
        vlSelfRef.__VactDidInit = 1U;
        vlSelfRef.__VactTriggered.setBit(2U, 1U);
    }
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtest_simple_tpu_mac___024root___dump_triggers__act(vlSelf);
    }
#endif
}
