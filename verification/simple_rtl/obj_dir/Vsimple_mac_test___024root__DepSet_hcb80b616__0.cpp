// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vsimple_mac_test.h for the primary calling header

#include "Vsimple_mac_test__pch.h"
#include "Vsimple_mac_test___024root.h"

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_initial__TOP(Vsimple_mac_test___024root* vlSelf);
VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__0(Vsimple_mac_test___024root* vlSelf);
VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__1(Vsimple_mac_test___024root* vlSelf);
VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__2(Vsimple_mac_test___024root* vlSelf);

void Vsimple_mac_test___024root___eval_initial(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_initial\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vsimple_mac_test___024root___eval_initial__TOP(vlSelf);
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__2(vlSelf);
}

VL_INLINE_OPT VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__0(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_WRITEF_NX("Starting simple MAC RTL test...\n",0);
    vlSelfRef.simple_mac_test__DOT__rst_n = 0U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         44);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__rst_n = 1U;
    VL_WRITEF_NX("Reset released\n",0);
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         49);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         49);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__a = 2U;
    vlSelfRef.simple_mac_test__DOT__b = 3U;
    vlSelfRef.simple_mac_test__DOT__c = 4U;
    vlSelfRef.simple_mac_test__DOT__valid = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         54);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__valid = 0U;
    while ((1U & (~ (IData)(vlSelfRef.simple_mac_test__DOT__ready)))) {
        co_await vlSelfRef.__VtrigSched_h5ca73cd1__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( simple_mac_test.ready)", 
                                                             "simple_mac_test.sv", 
                                                             58);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         59);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    if ((0xaU == vlSelfRef.simple_mac_test__DOT__result)) {
        VL_WRITEF_NX("\342\234\205 Test 1 PASSED: 2 * 3 + 4 = %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 1 FAILED: Expected 10, got %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__a = 5U;
    vlSelfRef.simple_mac_test__DOT__b = 7U;
    vlSelfRef.simple_mac_test__DOT__c = 1U;
    vlSelfRef.simple_mac_test__DOT__valid = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         73);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__valid = 0U;
    while ((1U & (~ (IData)(vlSelfRef.simple_mac_test__DOT__ready)))) {
        co_await vlSelfRef.__VtrigSched_h5ca73cd1__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( simple_mac_test.ready)", 
                                                             "simple_mac_test.sv", 
                                                             77);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         78);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    if ((0x24U == vlSelfRef.simple_mac_test__DOT__result)) {
        VL_WRITEF_NX("\342\234\205 Test 2 PASSED: 5 * 7 + 1 = %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 2 FAILED: Expected 36, got %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         87);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         87);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__a = 0U;
    vlSelfRef.simple_mac_test__DOT__b = 0x64U;
    vlSelfRef.simple_mac_test__DOT__c = 0x32U;
    vlSelfRef.simple_mac_test__DOT__valid = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         92);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.simple_mac_test__DOT__valid = 0U;
    while ((1U & (~ (IData)(vlSelfRef.simple_mac_test__DOT__ready)))) {
        co_await vlSelfRef.__VtrigSched_h5ca73cd1__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( simple_mac_test.ready)", 
                                                             "simple_mac_test.sv", 
                                                             96);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         97);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    if ((0x32U == vlSelfRef.simple_mac_test__DOT__result)) {
        VL_WRITEF_NX("\342\234\205 Test 3 PASSED: 0 * 100 + 50 = %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 3 FAILED: Expected 50, got %0#\n",0,
                     32,vlSelfRef.simple_mac_test__DOT__result);
    }
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_haa7fc2ca__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge simple_mac_test.clk)", 
                                                         "simple_mac_test.sv", 
                                                         105);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("Simple MAC RTL test completed successfully!\n",0);
    VL_FINISH_MT("simple_mac_test.sv", 107, "");
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
}

VL_INLINE_OPT VlCoroutine Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__1(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    co_await vlSelfRef.__VdlySched.delay(0x989680ULL, 
                                         nullptr, "simple_mac_test.sv", 
                                         112);
    VL_WRITEF_NX("ERROR: Test timeout!\n",0);
    VL_FINISH_MT("simple_mac_test.sv", 114, "");
}

void Vsimple_mac_test___024root___eval_act(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_act\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vsimple_mac_test___024root___nba_sequent__TOP__0(Vsimple_mac_test___024root* vlSelf);

void Vsimple_mac_test___024root___eval_nba(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_nba\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vsimple_mac_test___024root___nba_sequent__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vsimple_mac_test___024root___nba_sequent__TOP__0(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___nba_sequent__TOP__0\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.simple_mac_test__DOT__ready = ((IData)(vlSelfRef.simple_mac_test__DOT__rst_n) 
                                             && (IData)(vlSelfRef.simple_mac_test__DOT__valid));
    if (vlSelfRef.simple_mac_test__DOT__rst_n) {
        if (vlSelfRef.simple_mac_test__DOT__valid) {
            vlSelfRef.simple_mac_test__DOT__result 
                = (((IData)(vlSelfRef.simple_mac_test__DOT__a) 
                    * (IData)(vlSelfRef.simple_mac_test__DOT__b)) 
                   + vlSelfRef.simple_mac_test__DOT__c);
        }
    } else {
        vlSelfRef.simple_mac_test__DOT__result = 0U;
    }
}

void Vsimple_mac_test___024root___timing_commit(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___timing_commit\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((! (1ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_haa7fc2ca__0.commit(
                                                   "@(posedge simple_mac_test.clk)");
    }
    if ((! (4ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_h5ca73cd1__0.commit(
                                                   "@( simple_mac_test.ready)");
    }
}

void Vsimple_mac_test___024root___timing_resume(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___timing_resume\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_haa7fc2ca__0.resume(
                                                   "@(posedge simple_mac_test.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_h5ca73cd1__0.resume(
                                                   "@( simple_mac_test.ready)");
    }
    if ((8ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vsimple_mac_test___024root___eval_triggers__act(Vsimple_mac_test___024root* vlSelf);

bool Vsimple_mac_test___024root___eval_phase__act(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_phase__act\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<4> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vsimple_mac_test___024root___eval_triggers__act(vlSelf);
    Vsimple_mac_test___024root___timing_commit(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vsimple_mac_test___024root___timing_resume(vlSelf);
        Vsimple_mac_test___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vsimple_mac_test___024root___eval_phase__nba(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_phase__nba\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vsimple_mac_test___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vsimple_mac_test___024root___dump_triggers__nba(Vsimple_mac_test___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vsimple_mac_test___024root___dump_triggers__act(Vsimple_mac_test___024root* vlSelf);
#endif  // VL_DEBUG

void Vsimple_mac_test___024root___eval(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY(((0x64U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vsimple_mac_test___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("simple_mac_test.sv", 4, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vsimple_mac_test___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("simple_mac_test.sv", 4, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vsimple_mac_test___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vsimple_mac_test___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vsimple_mac_test___024root___eval_debug_assertions(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_debug_assertions\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
