// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vsimple_mac_test.h for the primary calling header

#include "Vsimple_mac_test__pch.h"
#include "Vsimple_mac_test___024root.h"

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_static__TOP(Vsimple_mac_test___024root* vlSelf);
VL_ATTR_COLD void Vsimple_mac_test___024root____Vm_traceActivitySetAll(Vsimple_mac_test___024root* vlSelf);

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_static(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_static\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vsimple_mac_test___024root___eval_static__TOP(vlSelf);
    Vsimple_mac_test___024root____Vm_traceActivitySetAll(vlSelf);
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__clk__0 
        = vlSelfRef.simple_mac_test__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__rst_n__0 
        = vlSelfRef.simple_mac_test__DOT__rst_n;
    vlSelfRef.__Vtrigprevexpr___TOP__simple_mac_test__DOT__ready__0 
        = vlSelfRef.simple_mac_test__DOT__ready;
}

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_static__TOP(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_static__TOP\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.simple_mac_test__DOT__a = 0U;
    vlSelfRef.simple_mac_test__DOT__b = 0U;
    vlSelfRef.simple_mac_test__DOT__c = 0U;
    vlSelfRef.simple_mac_test__DOT__valid = 0U;
}

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_final(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_final\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_settle(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_settle\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vsimple_mac_test___024root___dump_triggers__act(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___dump_triggers__act\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge simple_mac_test.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge simple_mac_test.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @( simple_mac_test.ready)\n");
    }
    if ((8ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 3 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vsimple_mac_test___024root___dump_triggers__nba(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___dump_triggers__nba\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge simple_mac_test.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge simple_mac_test.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @( simple_mac_test.ready)\n");
    }
    if ((8ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 3 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vsimple_mac_test___024root____Vm_traceActivitySetAll(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root____Vm_traceActivitySetAll\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[0U] = 1U;
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
}

VL_ATTR_COLD void Vsimple_mac_test___024root___ctor_var_reset(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___ctor_var_reset\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->simple_mac_test__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6677692212104075320ull);
    vlSelf->simple_mac_test__DOT__rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6012916863153435610ull);
    vlSelf->simple_mac_test__DOT__a = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2601104680900566550ull);
    vlSelf->simple_mac_test__DOT__b = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4804730112150433571ull);
    vlSelf->simple_mac_test__DOT__c = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 10882496897001744015ull);
    vlSelf->simple_mac_test__DOT__valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14875550836186479611ull);
    vlSelf->simple_mac_test__DOT__result = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5102029062360637688ull);
    vlSelf->simple_mac_test__DOT__ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9557281798338305373ull);
    vlSelf->__Vtrigprevexpr___TOP__simple_mac_test__DOT__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17271852668114202870ull);
    vlSelf->__Vtrigprevexpr___TOP__simple_mac_test__DOT__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17550873537924375721ull);
    vlSelf->__Vtrigprevexpr___TOP__simple_mac_test__DOT__ready__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5436433244490208862ull);
    vlSelf->__VactDidInit = 0;
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
