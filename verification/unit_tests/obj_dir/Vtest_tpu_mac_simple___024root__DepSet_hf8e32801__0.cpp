// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_tpu_mac_simple.h for the primary calling header

#include "Vtest_tpu_mac_simple__pch.h"
#include "Vtest_tpu_mac_simple___024root.h"

VlCoroutine Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__0(Vtest_tpu_mac_simple___024root* vlSelf);
VlCoroutine Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__1(Vtest_tpu_mac_simple___024root* vlSelf);

void Vtest_tpu_mac_simple___024root___eval_initial(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_initial\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

VL_INLINE_OPT VlCoroutine Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__0(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_tpu_mac_simple__DOT__clk = 0U;
    while (1U) {
        co_await vlSelfRef.__VdlySched.delay(0x1388ULL, 
                                             nullptr, 
                                             "test_tpu_mac_simple.sv", 
                                             31);
        vlSelfRef.test_tpu_mac_simple__DOT__clk = (1U 
                                                   & (~ (IData)(vlSelfRef.test_tpu_mac_simple__DOT__clk)));
    }
}

VL_INLINE_OPT VlCoroutine Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__1(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_3__DOT____Vrepeat2;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_3__DOT____Vrepeat2 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_5__DOT____Vrepeat4;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_5__DOT____Vrepeat4 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_6__DOT____Vrepeat5;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_6__DOT____Vrepeat5 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_8__DOT____Vrepeat7;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_8__DOT____Vrepeat7 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_9__DOT____Vrepeat8;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_9__DOT____Vrepeat8 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9 = 0;
    IData/*31:0*/ __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_11__DOT____Vrepeat10;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_11__DOT____Vrepeat10 = 0;
    // Body
    VL_WRITEF_NX("=== Simple TPU MAC Unit Test ===\n",0);
    vlSelfRef.test_tpu_mac_simple__DOT__rst_n = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__data_type = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__a_in = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__b_in = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__c_in = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__accumulate = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         68);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__rst_n = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         70);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         70);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         70);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         70);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_3__DOT____Vrepeat2 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_5__DOT____Vrepeat4 = 0;
    vlSelfRef.test_tpu_mac_simple__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    VL_WRITEF_NX("\nTest %0d: Basic INT8 multiplication\n",0,
                 32,vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    vlSelfRef.test_tpu_mac_simple__DOT__data_type = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__accumulate = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__b_in = 0x5050505U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         107);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_3__DOT____Vrepeat2 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         107);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_3__DOT____Vrepeat2 = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__a_in = 0x3030303U;
    vlSelfRef.test_tpu_mac_simple__DOT__c_in = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3 = 2U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_4__DOT____Vrepeat3 = 0U;
    if ((0xfU == (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out))) {
        VL_WRITEF_NX("  PASS: 3 * 5 = %0#\n",0,8,(0xffU 
                                                  & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
        vlSelfRef.test_tpu_mac_simple__DOT__pass_count 
            = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__pass_count);
    } else {
        VL_WRITEF_NX("  FAIL: Expected 15, got %0#\n",0,
                     8,(0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
    }
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         124);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_5__DOT____Vrepeat4 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         124);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_basic_int8__0__test_tpu_mac_simple__DOT__unnamedblk1_5__DOT____Vrepeat4 = 0U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_6__DOT____Vrepeat5 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_8__DOT____Vrepeat7 = 0;
    vlSelfRef.test_tpu_mac_simple__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    VL_WRITEF_NX("\nTest %0d: Weight loading\n",0,32,
                 vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    vlSelfRef.test_tpu_mac_simple__DOT__data_type = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__b_in = 0x7070707U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         138);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_6__DOT____Vrepeat5 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         138);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_6__DOT____Vrepeat5 = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__a_in = 0x2020202U;
    vlSelfRef.test_tpu_mac_simple__DOT__c_in = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         144);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6 = 2U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         144);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         144);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_7__DOT____Vrepeat6 = 0U;
    if ((0xeU == (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out))) {
        VL_WRITEF_NX("  PASS: 2 * 7 = %0#\n",0,8,(0xffU 
                                                  & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
        vlSelfRef.test_tpu_mac_simple__DOT__pass_count 
            = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__pass_count);
    } else {
        VL_WRITEF_NX("  FAIL: Expected 14, got %0#\n",0,
                     8,(0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
    }
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         155);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_8__DOT____Vrepeat7 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         155);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_weight_loading__1__test_tpu_mac_simple__DOT__unnamedblk1_8__DOT____Vrepeat7 = 0U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_9__DOT____Vrepeat8 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9 = 0;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_11__DOT____Vrepeat10 = 0;
    vlSelfRef.test_tpu_mac_simple__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    VL_WRITEF_NX("\nTest %0d: Accumulation\n",0,32,
                 vlSelfRef.test_tpu_mac_simple__DOT__test_count);
    vlSelfRef.test_tpu_mac_simple__DOT__data_type = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__accumulate = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 1U;
    vlSelfRef.test_tpu_mac_simple__DOT__b_in = 0x4040404U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         170);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_9__DOT____Vrepeat8 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         170);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_9__DOT____Vrepeat8 = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__load_weight = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__a_in = 0x3030303U;
    vlSelfRef.test_tpu_mac_simple__DOT__c_in = 0x5050505U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         176);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9 = 2U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         176);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         176);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_10__DOT____Vrepeat9 = 0U;
    if ((0x11U == (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out))) {
        VL_WRITEF_NX("  PASS: 3 * 4 + 5 = %0#\n",0,
                     8,(0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
        vlSelfRef.test_tpu_mac_simple__DOT__pass_count 
            = ((IData)(1U) + vlSelfRef.test_tpu_mac_simple__DOT__pass_count);
    } else {
        VL_WRITEF_NX("  FAIL: Expected 17, got %0#\n",0,
                     8,(0xffU & vlSelfRef.test_tpu_mac_simple__DOT__c_out));
    }
    vlSelfRef.test_tpu_mac_simple__DOT__accumulate = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__enable = 0U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         188);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_11__DOT____Vrepeat10 = 1U;
    co_await vlSelfRef.__VtrigSched_h3f83c5f4__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_tpu_mac_simple.clk)", 
                                                         "test_tpu_mac_simple.sv", 
                                                         188);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_tpu_mac_simple__DOT__test_accumulation__2__test_tpu_mac_simple__DOT__unnamedblk1_11__DOT____Vrepeat10 = 0U;
    VL_WRITEF_NX("\n=== Test Summary ===\nTests run: %0d\nTests passed: %0d\n",0,
                 32,vlSelfRef.test_tpu_mac_simple__DOT__test_count,
                 32,vlSelfRef.test_tpu_mac_simple__DOT__pass_count);
    if ((vlSelfRef.test_tpu_mac_simple__DOT__pass_count 
         == vlSelfRef.test_tpu_mac_simple__DOT__test_count)) {
        VL_WRITEF_NX("All tests PASSED!\n",0);
    } else {
        VL_WRITEF_NX("Some tests FAILED!\n",0);
    }
    VL_FINISH_MT("test_tpu_mac_simple.sv", 92, "");
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
}

void Vtest_tpu_mac_simple___024root___act_sequent__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf);

void Vtest_tpu_mac_simple___024root___eval_act(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_act\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        Vtest_tpu_mac_simple___024root___act_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[3U] = 1U;
    }
}

VL_INLINE_OPT void Vtest_tpu_mac_simple___024root___act_sequent__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___act_sequent__TOP__0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((0U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16 
                = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16 
                = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__a_in);
        }
        if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32 
                    = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg;
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32 
                    = vlSelfRef.test_tpu_mac_simple__DOT__a_in;
            }
        }
    }
    if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8 
            = (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg);
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8 
            = (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__a_in);
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8 
            = (0xffffU & VL_MULS_III(16, (0xffffU & 
                                          VL_EXTENDS_II(16,8, (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8))), 
                                     (0xffffU & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8)))));
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
            = (((- (IData)((1U & ((IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8) 
                                  >> 0xfU)))) << 0x10U) 
               | (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8));
    } else {
        if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            if ((2U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8 = 0U;
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8 = 0U;
            }
        }
        if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result 
                = ((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a) 
                   * (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b));
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout 
                = (0xffffU & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout;
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16;
        } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result 
                = ((QData)((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a)) 
                   * (QData)((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b)));
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout 
                = (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout;
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32;
        } else {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result = 0U;
        }
    }
    if (vlSelfRef.test_tpu_mac_simple__DOT__accumulate) {
        if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = (vlSelfRef.test_tpu_mac_simple__DOT__c_in 
                   + vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result);
        } else if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = VL_EXTEND_II(32,16, ([&]() {
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b 
                            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result);
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a 
                            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__c_in);
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout 
                            = (0xffffU & ((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a) 
                                          + (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b)));
                    }(), (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout)));
        } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__c_in;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout 
                = (vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a 
                   + vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout;
        } else {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
        }
    } else {
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
    }
    vlSelfRef.test_tpu_mac_simple__DOT__overflow = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__underflow = 0U;
    if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = VL_LTS_III(32, 0x7fffffffU, vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = VL_GTS_III(32, 0x80000000U, vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
    } else if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val 
            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout 
            = (IData)((0x7c00U == (0x7fffU & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val))));
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val 
            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout 
            = (IData)(((0U == (0x7c00U & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val))) 
                       & (0U != (0x3ffU & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val)))));
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout;
    } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout 
            = (IData)((0x7f800000U == (0x7fffffffU 
                                       & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val)));
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout 
            = (IData)(((0U == (0x7f800000U & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val)) 
                       & (0U != (0x7fffffU & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val))));
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout;
    } else {
        vlSelfRef.test_tpu_mac_simple__DOT__overflow = 0U;
        vlSelfRef.test_tpu_mac_simple__DOT__underflow = 0U;
    }
}

void Vtest_tpu_mac_simple___024root___nba_sequent__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf);
void Vtest_tpu_mac_simple___024root___nba_sequent__TOP__1(Vtest_tpu_mac_simple___024root* vlSelf);
void Vtest_tpu_mac_simple___024root___nba_comb__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf);

void Vtest_tpu_mac_simple___024root___eval_nba(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_nba\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_tpu_mac_simple___024root___nba_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[4U] = 1U;
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_tpu_mac_simple___024root___nba_sequent__TOP__1(vlSelf);
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_tpu_mac_simple___024root___nba_comb__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[5U] = 1U;
    }
}

VL_INLINE_OPT void Vtest_tpu_mac_simple___024root___nba_sequent__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___nba_sequent__TOP__0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (vlSelfRef.test_tpu_mac_simple__DOT__rst_n) {
        if (vlSelfRef.test_tpu_mac_simple__DOT__enable) {
            vlSelfRef.test_tpu_mac_simple__DOT__a_out 
                = vlSelfRef.test_tpu_mac_simple__DOT__a_in;
            vlSelfRef.test_tpu_mac_simple__DOT__b_out 
                = vlSelfRef.test_tpu_mac_simple__DOT__b_in;
            vlSelfRef.test_tpu_mac_simple__DOT__c_out 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result;
        }
        if (vlSelfRef.test_tpu_mac_simple__DOT__load_weight) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg 
                = vlSelfRef.test_tpu_mac_simple__DOT__b_in;
        }
    } else {
        vlSelfRef.test_tpu_mac_simple__DOT__a_out = 0U;
        vlSelfRef.test_tpu_mac_simple__DOT__b_out = 0U;
        vlSelfRef.test_tpu_mac_simple__DOT__c_out = 0U;
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg = 0U;
    }
}

VL_INLINE_OPT void Vtest_tpu_mac_simple___024root___nba_sequent__TOP__1(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___nba_sequent__TOP__1\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8 
            = (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__a_in);
    } else if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        if ((2U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8 = 0U;
        }
    }
    if ((0U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16 
                = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__a_in);
        }
        if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32 
                    = vlSelfRef.test_tpu_mac_simple__DOT__a_in;
            }
        }
    }
}

VL_INLINE_OPT void Vtest_tpu_mac_simple___024root___nba_comb__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___nba_comb__TOP__0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((0U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16 
                = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg);
        }
        if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32 
                    = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg;
            }
        }
    }
    if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8 
            = (0xffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg);
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8 
            = (0xffffU & VL_MULS_III(16, (0xffffU & 
                                          VL_EXTENDS_II(16,8, (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8))), 
                                     (0xffffU & VL_EXTENDS_II(16,8, (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8)))));
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
            = (((- (IData)((1U & ((IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8) 
                                  >> 0xfU)))) << 0x10U) 
               | (IData)(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8));
    } else {
        if ((1U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            if ((2U != (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
                vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8 = 0U;
            }
        }
        if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result 
                = ((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a) 
                   * (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b));
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout 
                = (0xffffU & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout;
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16;
        } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result 
                = ((QData)((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a)) 
                   * (QData)((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b)));
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout 
                = (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout;
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32;
        } else {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result = 0U;
        }
    }
    if (vlSelfRef.test_tpu_mac_simple__DOT__accumulate) {
        if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = (vlSelfRef.test_tpu_mac_simple__DOT__c_in 
                   + vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result);
        } else if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = VL_EXTEND_II(32,16, ([&]() {
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b 
                            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result);
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a 
                            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__c_in);
                        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout 
                            = (0xffffU & ((IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a) 
                                          + (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b)));
                    }(), (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout)));
        } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a 
                = vlSelfRef.test_tpu_mac_simple__DOT__c_in;
            vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout 
                = (vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a 
                   + vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b);
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout;
        } else {
            vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
                = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
        }
    } else {
        vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result;
    }
    vlSelfRef.test_tpu_mac_simple__DOT__overflow = 0U;
    vlSelfRef.test_tpu_mac_simple__DOT__underflow = 0U;
    if ((0U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = VL_LTS_III(32, 0x7fffffffU, vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = VL_GTS_III(32, 0x80000000U, vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
    } else if ((1U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val 
            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout 
            = (IData)((0x7c00U == (0x7fffU & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val))));
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val 
            = (0xffffU & vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result);
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout 
            = (IData)(((0U == (0x7c00U & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val))) 
                       & (0U != (0x3ffU & (IData)(vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val)))));
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout;
    } else if ((2U == (IData)(vlSelfRef.test_tpu_mac_simple__DOT__data_type))) {
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout 
            = (IData)((0x7f800000U == (0x7fffffffU 
                                       & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val)));
        vlSelfRef.test_tpu_mac_simple__DOT__overflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val 
            = vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result;
        vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout 
            = (IData)(((0U == (0x7f800000U & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val)) 
                       & (0U != (0x7fffffU & vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val))));
        vlSelfRef.test_tpu_mac_simple__DOT__underflow 
            = vlSelfRef.__Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout;
    } else {
        vlSelfRef.test_tpu_mac_simple__DOT__overflow = 0U;
        vlSelfRef.test_tpu_mac_simple__DOT__underflow = 0U;
    }
}

void Vtest_tpu_mac_simple___024root___timing_commit(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___timing_commit\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((! (1ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_h3f83c5f4__0.commit(
                                                   "@(posedge test_tpu_mac_simple.clk)");
    }
}

void Vtest_tpu_mac_simple___024root___timing_resume(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___timing_resume\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_h3f83c5f4__0.resume(
                                                   "@(posedge test_tpu_mac_simple.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtest_tpu_mac_simple___024root___eval_triggers__act(Vtest_tpu_mac_simple___024root* vlSelf);

bool Vtest_tpu_mac_simple___024root___eval_phase__act(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_phase__act\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<3> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtest_tpu_mac_simple___024root___eval_triggers__act(vlSelf);
    Vtest_tpu_mac_simple___024root___timing_commit(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vtest_tpu_mac_simple___024root___timing_resume(vlSelf);
        Vtest_tpu_mac_simple___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtest_tpu_mac_simple___024root___eval_phase__nba(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_phase__nba\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtest_tpu_mac_simple___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__nba(Vtest_tpu_mac_simple___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__act(Vtest_tpu_mac_simple___024root* vlSelf);
#endif  // VL_DEBUG

void Vtest_tpu_mac_simple___024root___eval(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vtest_tpu_mac_simple___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("test_tpu_mac_simple.sv", 6, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtest_tpu_mac_simple___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("test_tpu_mac_simple.sv", 6, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vtest_tpu_mac_simple___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vtest_tpu_mac_simple___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtest_tpu_mac_simple___024root___eval_debug_assertions(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_debug_assertions\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
