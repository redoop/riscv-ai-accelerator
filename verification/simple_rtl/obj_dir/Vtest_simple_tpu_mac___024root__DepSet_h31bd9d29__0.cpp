// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_simple_tpu_mac.h for the primary calling header

#include "Vtest_simple_tpu_mac__pch.h"
#include "Vtest_simple_tpu_mac___024root.h"

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___eval_initial__TOP(Vtest_simple_tpu_mac___024root* vlSelf);
VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__0(Vtest_simple_tpu_mac___024root* vlSelf);
VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__1(Vtest_simple_tpu_mac___024root* vlSelf);
VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__2(Vtest_simple_tpu_mac___024root* vlSelf);

void Vtest_simple_tpu_mac___024root___eval_initial(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_initial\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtest_simple_tpu_mac___024root___eval_initial__TOP(vlSelf);
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__2(vlSelf);
}

VL_INLINE_OPT VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__0(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_WRITEF_NX("\360\237\224\254 Testing Simplified TPU MAC Unit RTL Code\n==========================================\n",0);
    vlSelfRef.test_simple_tpu_mac__DOT__rst_n = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__enable = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__data_type = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__a_data = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__b_data = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__c_data = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 0U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         58);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__rst_n = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__enable = 1U;
    VL_WRITEF_NX("\342\234\205 Reset released, TPU MAC unit enabled\n",0);
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         64);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         64);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__data_type = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__a_data = 0xaU;
    vlSelfRef.test_simple_tpu_mac__DOT__b_data = 0x14U;
    vlSelfRef.test_simple_tpu_mac__DOT__c_data = 5U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         70);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 0U;
    while ((1U & (~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid)))) {
        co_await vlSelfRef.__VtrigSched_he43d38c9__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( test_simple_tpu_mac.dut.result_valid)", 
                                                             "test_simple_tpu_mac.sv", 
                                                             74);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         75);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("\360\237\247\256 Test 1 (INT8): 10 * 20 + 5 = %0#\n",0,
                 32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    if ((0xcdU == vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result)) {
        VL_WRITEF_NX("\342\234\205 Test 1 PASSED\n",0);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 1 FAILED: Expected 205, got %0#\n",0,
                     32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         85);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         85);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         85);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         85);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         85);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__data_type = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__a_data = 7U;
    vlSelfRef.test_simple_tpu_mac__DOT__b_data = 8U;
    vlSelfRef.test_simple_tpu_mac__DOT__c_data = 0x64U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         91);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 0U;
    while ((1U & (~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid)))) {
        co_await vlSelfRef.__VtrigSched_he43d38c9__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( test_simple_tpu_mac.dut.result_valid)", 
                                                             "test_simple_tpu_mac.sv", 
                                                             95);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         96);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("\360\237\247\256 Test 2 (INT16): 7 * 8 + 100 = %0#\n",0,
                 32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    if ((0x9cU == vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result)) {
        VL_WRITEF_NX("\342\234\205 Test 2 PASSED\n",0);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 2 FAILED: Expected 156, got %0#\n",0,
                     32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         106);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         106);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         106);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         106);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         106);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__data_type = 2U;
    vlSelfRef.test_simple_tpu_mac__DOT__a_data = 0xfU;
    vlSelfRef.test_simple_tpu_mac__DOT__b_data = 4U;
    vlSelfRef.test_simple_tpu_mac__DOT__c_data = 0x19U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         112);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 0U;
    while ((1U & (~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid)))) {
        co_await vlSelfRef.__VtrigSched_he43d38c9__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( test_simple_tpu_mac.dut.result_valid)", 
                                                             "test_simple_tpu_mac.sv", 
                                                             116);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("\360\237\247\256 Test 3 (INT32): 15 * 4 + 25 = %0#\n",0,
                 32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    if ((0x55U == vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result)) {
        VL_WRITEF_NX("\342\234\205 Test 3 PASSED\n",0);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 3 FAILED: Expected 85, got %0#\n",0,
                     32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         127);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         127);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         127);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         127);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         127);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__a_data = 0U;
    vlSelfRef.test_simple_tpu_mac__DOT__b_data = 0x3e7U;
    vlSelfRef.test_simple_tpu_mac__DOT__c_data = 0x2aU;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         132);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_simple_tpu_mac__DOT__valid_in = 0U;
    while ((1U & (~ (IData)(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid)))) {
        co_await vlSelfRef.__VtrigSched_he43d38c9__0.trigger(1U, 
                                                             nullptr, 
                                                             "@( test_simple_tpu_mac.dut.result_valid)", 
                                                             "test_simple_tpu_mac.sv", 
                                                             136);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         137);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("\360\237\247\256 Test 4 (Zero): 0 * 999 + 42 = %0#\n",0,
                 32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    if ((0x2aU == vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result)) {
        VL_WRITEF_NX("\342\234\205 Test 4 PASSED\n",0);
    } else {
        VL_WRITEF_NX("\342\235\214 Test 4 FAILED: Expected 42, got %0#\n",0,
                     32,vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result);
    }
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    co_await vlSelfRef.__VtrigSched_h111eea42__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_simple_tpu_mac.clk)", 
                                                         "test_simple_tpu_mac.sv", 
                                                         146);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    VL_WRITEF_NX("\n\360\237\216\211 TPU MAC RTL \346\265\213\350\257\225\345\256\214\346\210\220!\n\342\234\250 \346\210\220\345\212\237\346\211\247\350\241\214\344\272\206 RTL \347\241\254\344\273\266\346\217\217\350\277\260\344\273\243\347\240\201!\n\360\237\224\247 \350\277\231\346\230\257\347\234\237\346\255\243\347\232\204\347\241\254\344\273\266\351\200\273\350\276\221\344\273\277\347\234\237\357\274\214\344\270\215\346\230\257\350\275\257\344\273\266\346\250\241\346\213\237!\n",0);
    VL_FINISH_MT("test_simple_tpu_mac.sv", 151, "");
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
}

VL_INLINE_OPT VlCoroutine Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__1(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    co_await vlSelfRef.__VdlySched.delay(0x5f5e100ULL, 
                                         nullptr, "test_simple_tpu_mac.sv", 
                                         156);
    VL_WRITEF_NX("\342\235\214 ERROR: Test timeout!\n",0);
    VL_FINISH_MT("test_simple_tpu_mac.sv", 158, "");
}

void Vtest_simple_tpu_mac___024root___eval_act(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_act\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

void Vtest_simple_tpu_mac___024root___nba_sequent__TOP__0(Vtest_simple_tpu_mac___024root* vlSelf);

void Vtest_simple_tpu_mac___024root___eval_nba(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_nba\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_simple_tpu_mac___024root___nba_sequent__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vtest_simple_tpu_mac___024root___nba_sequent__TOP__0(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___nba_sequent__TOP__0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid 
        = ((IData)(vlSelfRef.test_simple_tpu_mac__DOT__rst_n) 
           && ((IData)(vlSelfRef.test_simple_tpu_mac__DOT__enable) 
               & (IData)(vlSelfRef.test_simple_tpu_mac__DOT__valid_in)));
    if (vlSelfRef.test_simple_tpu_mac__DOT__rst_n) {
        if (((IData)(vlSelfRef.test_simple_tpu_mac__DOT__enable) 
             & (IData)(vlSelfRef.test_simple_tpu_mac__DOT__valid_in))) {
            vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result 
                = ((0U == (IData)(vlSelfRef.test_simple_tpu_mac__DOT__data_type))
                    ? (((0xffU & (IData)(vlSelfRef.test_simple_tpu_mac__DOT__a_data)) 
                        * (0xffU & (IData)(vlSelfRef.test_simple_tpu_mac__DOT__b_data))) 
                       + vlSelfRef.test_simple_tpu_mac__DOT__c_data)
                    : ((1U == (IData)(vlSelfRef.test_simple_tpu_mac__DOT__data_type))
                        ? (((IData)(vlSelfRef.test_simple_tpu_mac__DOT__a_data) 
                            * (IData)(vlSelfRef.test_simple_tpu_mac__DOT__b_data)) 
                           + vlSelfRef.test_simple_tpu_mac__DOT__c_data)
                        : ((2U == (IData)(vlSelfRef.test_simple_tpu_mac__DOT__data_type))
                            ? (((IData)(vlSelfRef.test_simple_tpu_mac__DOT__a_data) 
                                * (IData)(vlSelfRef.test_simple_tpu_mac__DOT__b_data)) 
                               + vlSelfRef.test_simple_tpu_mac__DOT__c_data)
                            : (((IData)(vlSelfRef.test_simple_tpu_mac__DOT__a_data) 
                                * (IData)(vlSelfRef.test_simple_tpu_mac__DOT__b_data)) 
                               + vlSelfRef.test_simple_tpu_mac__DOT__c_data))));
        }
    } else {
        vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result = 0U;
    }
}

void Vtest_simple_tpu_mac___024root___timing_commit(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___timing_commit\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((! (1ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_h111eea42__0.commit(
                                                   "@(posedge test_simple_tpu_mac.clk)");
    }
    if ((! (4ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_he43d38c9__0.commit(
                                                   "@( test_simple_tpu_mac.dut.result_valid)");
    }
}

void Vtest_simple_tpu_mac___024root___timing_resume(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___timing_resume\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_h111eea42__0.resume(
                                                   "@(posedge test_simple_tpu_mac.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_he43d38c9__0.resume(
                                                   "@( test_simple_tpu_mac.dut.result_valid)");
    }
    if ((8ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtest_simple_tpu_mac___024root___eval_triggers__act(Vtest_simple_tpu_mac___024root* vlSelf);

bool Vtest_simple_tpu_mac___024root___eval_phase__act(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_phase__act\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<4> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtest_simple_tpu_mac___024root___eval_triggers__act(vlSelf);
    Vtest_simple_tpu_mac___024root___timing_commit(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vtest_simple_tpu_mac___024root___timing_resume(vlSelf);
        Vtest_simple_tpu_mac___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtest_simple_tpu_mac___024root___eval_phase__nba(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_phase__nba\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtest_simple_tpu_mac___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___dump_triggers__nba(Vtest_simple_tpu_mac___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___dump_triggers__act(Vtest_simple_tpu_mac___024root* vlSelf);
#endif  // VL_DEBUG

void Vtest_simple_tpu_mac___024root___eval(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vtest_simple_tpu_mac___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("test_simple_tpu_mac.sv", 4, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtest_simple_tpu_mac___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("test_simple_tpu_mac.sv", 4, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vtest_simple_tpu_mac___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vtest_simple_tpu_mac___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtest_simple_tpu_mac___024root___eval_debug_assertions(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_debug_assertions\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
