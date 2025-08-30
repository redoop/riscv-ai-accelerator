// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_simple_tpu_mac.h for the primary calling header

#include "Vtest_simple_tpu_mac__pch.h"
#include "Vtest_simple_tpu_mac___024root.h"

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___eval_static(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_static\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__clk__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__rst_n__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__rst_n;
    vlSelfRef.__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__dut__DOT__result_valid__0 
        = vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid;
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___eval_final(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_final\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___eval_settle(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_settle\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___dump_triggers__act(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___dump_triggers__act\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge test_simple_tpu_mac.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge test_simple_tpu_mac.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @( test_simple_tpu_mac.dut.result_valid)\n");
    }
    if ((8ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 3 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___dump_triggers__nba(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___dump_triggers__nba\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge test_simple_tpu_mac.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge test_simple_tpu_mac.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @( test_simple_tpu_mac.dut.result_valid)\n");
    }
    if ((8ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 3 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___ctor_var_reset(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___ctor_var_reset\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->test_simple_tpu_mac__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12862440027256351891ull);
    vlSelf->test_simple_tpu_mac__DOT__rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16179831554763543810ull);
    vlSelf->test_simple_tpu_mac__DOT__enable = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 5583435213000694276ull);
    vlSelf->test_simple_tpu_mac__DOT__data_type = VL_SCOPED_RAND_RESET_I(3, __VscopeHash, 8434882147153485684ull);
    vlSelf->test_simple_tpu_mac__DOT__a_data = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17506659220194829802ull);
    vlSelf->test_simple_tpu_mac__DOT__b_data = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10448431401894000387ull);
    vlSelf->test_simple_tpu_mac__DOT__c_data = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7473362684674687241ull);
    vlSelf->test_simple_tpu_mac__DOT__valid_in = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 10689196115811041835ull);
    vlSelf->test_simple_tpu_mac__DOT__dut__DOT__mac_result = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12636176111987663197ull);
    vlSelf->test_simple_tpu_mac__DOT__dut__DOT__result_valid = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 1181817681323840418ull);
    vlSelf->__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 15548285422459580889ull);
    vlSelf->__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2699800701056164135ull);
    vlSelf->__Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__dut__DOT__result_valid__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16218018900103784080ull);
    vlSelf->__VactDidInit = 0;
    for (int __Vi0 = 0; __Vi0 < 3; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
