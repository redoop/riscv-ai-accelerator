// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_tpu_mac_simple.h for the primary calling header

#include "Vtest_tpu_mac_simple__pch.h"
#include "Vtest_tpu_mac_simple___024root.h"

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_static__TOP(Vtest_tpu_mac_simple___024root* vlSelf);
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root____Vm_traceActivitySetAll(Vtest_tpu_mac_simple___024root* vlSelf);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_static(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_static\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtest_tpu_mac_simple___024root___eval_static__TOP(vlSelf);
    Vtest_tpu_mac_simple___024root____Vm_traceActivitySetAll(vlSelf);
    vlSelfRef.__Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__clk__0 
        = vlSelfRef.test_tpu_mac_simple__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__rst_n__0 
        = vlSelfRef.test_tpu_mac_simple__DOT__rst_n;
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_static__TOP(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_static__TOP\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_tpu_mac_simple__DOT__test_count = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__pass_count = 0U;
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_final(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_final\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__stl(Vtest_tpu_mac_simple___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtest_tpu_mac_simple___024root___eval_phase__stl(Vtest_tpu_mac_simple___024root* vlSelf);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_settle(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_settle\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY(((0x64U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtest_tpu_mac_simple___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("test_tpu_mac_simple.sv", 6, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtest_tpu_mac_simple___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelfRef.__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__stl(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___dump_triggers__stl\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VstlTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

void Vtest_tpu_mac_simple___024root___act_sequent__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_stl(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_stl\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        Vtest_tpu_mac_simple___024root___act_sequent__TOP__0(vlSelf);
        Vtest_tpu_mac_simple___024root____Vm_traceActivitySetAll(vlSelf);
    }
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_triggers__stl(Vtest_tpu_mac_simple___024root* vlSelf);

VL_ATTR_COLD bool Vtest_tpu_mac_simple___024root___eval_phase__stl(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_phase__stl\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtest_tpu_mac_simple___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelfRef.__VstlTriggered.any();
    if (__VstlExecute) {
        Vtest_tpu_mac_simple___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__act(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___dump_triggers__act\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge test_tpu_mac_simple.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge test_tpu_mac_simple.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__nba(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___dump_triggers__nba\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge test_tpu_mac_simple.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge test_tpu_mac_simple.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root____Vm_traceActivitySetAll(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root____Vm_traceActivitySetAll\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[0U] = 1U;
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.__Vm_traceActivity[3U] = 1U;
    vlSelfRef.__Vm_traceActivity[4U] = 1U;
    vlSelfRef.__Vm_traceActivity[5U] = 1U;
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___ctor_var_reset(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___ctor_var_reset\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->test_tpu_mac_simple__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3106928980426954982ull);
    vlSelf->test_tpu_mac_simple__DOT__rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9824808928492984551ull);
    vlSelf->test_tpu_mac_simple__DOT__enable = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 7166068480107700685ull);
    vlSelf->test_tpu_mac_simple__DOT__data_type = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 3670313710479759976ull);
    vlSelf->test_tpu_mac_simple__DOT__a_in = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17754357246874639424ull);
    vlSelf->test_tpu_mac_simple__DOT__b_in = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 9237012762411056950ull);
    vlSelf->test_tpu_mac_simple__DOT__c_in = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13100132068978778266ull);
    vlSelf->test_tpu_mac_simple__DOT__a_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 6798594749826349341ull);
    vlSelf->test_tpu_mac_simple__DOT__b_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16539095964847241473ull);
    vlSelf->test_tpu_mac_simple__DOT__c_out = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12422122132870568159ull);
    vlSelf->test_tpu_mac_simple__DOT__load_weight = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17416735176248017606ull);
    vlSelf->test_tpu_mac_simple__DOT__accumulate = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13413129032249692543ull);
    vlSelf->test_tpu_mac_simple__DOT__overflow = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16692160926436128351ull);
    vlSelf->test_tpu_mac_simple__DOT__underflow = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 17120804212795394450ull);
    vlSelf->test_tpu_mac_simple__DOT__test_count = 0;
    vlSelf->test_tpu_mac_simple__DOT__pass_count = 0;
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__weight_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 16344037441103661976ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__mult_result = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15213955737138617069ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__acc_result = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 13885937008045282662ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__a_int8 = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 11248886389996212552ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__b_int8 = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 17427375306230435483ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__a_fp16 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9079481445569582491ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__b_fp16 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2862132657873491573ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__a_fp32 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 4778344372977392931ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__b_fp32 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 18315585280649881568ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__mult_int8 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 14493414296633800565ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__mult_fp16 = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 6084021351801075921ull);
    vlSelf->test_tpu_mac_simple__DOT__dut__DOT__mult_fp32 = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 5079690093321742774ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 11713297359540633151ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 2634609444271968678ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 5583553028855725043ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 7979945077250870692ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11990922920678763708ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 1698572643749504518ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12812928247794736470ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 2628105712633874478ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17352504379579501322ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 15027686329874791921ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 10468796718807658477ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 15514714390444132747ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 14184925447596167387ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 3791232542799848005ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13620220261824061621ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 9067775406400434715ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14681721694003806242ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17815876762231761239ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11710873171146972259ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12837179664177357554ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13502018179460985436ull);
    vlSelf->__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 18422024110134342963ull);
    vlSelf->__Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 16778160960571451836ull);
    vlSelf->__Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6873598818877482606ull);
    for (int __Vi0 = 0; __Vi0 < 6; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
