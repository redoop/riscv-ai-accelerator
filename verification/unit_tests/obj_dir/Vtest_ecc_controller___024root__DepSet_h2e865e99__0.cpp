// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_ecc_controller.h for the primary calling header

#include "Vtest_ecc_controller__pch.h"
#include "Vtest_ecc_controller___024root.h"

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_initial__TOP(Vtest_ecc_controller___024root* vlSelf);
VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__0(Vtest_ecc_controller___024root* vlSelf);
VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__1(Vtest_ecc_controller___024root* vlSelf);
VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__2(Vtest_ecc_controller___024root* vlSelf);

void Vtest_ecc_controller___024root___eval_initial(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_initial\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtest_ecc_controller___024root___eval_initial__TOP(vlSelf);
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__1(vlSelf);
    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__2(vlSelf);
}

VL_INLINE_OPT VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__0(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_ecc_controller__DOT__clk = 0U;
    while (1U) {
        co_await vlSelfRef.__VdlySched.delay(0x1388ULL, 
                                             nullptr, 
                                             "test_ecc_controller.sv", 
                                             70);
        vlSelfRef.test_ecc_controller__DOT__clk = (1U 
                                                   & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__clk)));
    }
}

VL_INLINE_OPT VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__1(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_2__DOT____Vrepeat1;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_2__DOT____Vrepeat1 = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__1__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__1__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__1__data;
    __Vtask_test_ecc_controller__DOT__write_memory__1__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__2__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__2__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__2__data;
    __Vtask_test_ecc_controller__DOT__read_memory__2__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__3__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__3__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__3__data;
    __Vtask_test_ecc_controller__DOT__write_memory__3__data = 0;
    CData/*1:0*/ __Vtask_test_ecc_controller__DOT__inject_error__4__error_type;
    __Vtask_test_ecc_controller__DOT__inject_error__4__error_type = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__5__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__5__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__5__data;
    __Vtask_test_ecc_controller__DOT__read_memory__5__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__6__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__6__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__6__data;
    __Vtask_test_ecc_controller__DOT__write_memory__6__data = 0;
    CData/*1:0*/ __Vtask_test_ecc_controller__DOT__inject_error__7__error_type;
    __Vtask_test_ecc_controller__DOT__inject_error__7__error_type = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__8__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__8__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__8__data;
    __Vtask_test_ecc_controller__DOT__read_memory__8__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    __Vtask_test_ecc_controller__DOT__read_memory__10__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__11__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__11__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__11__data;
    __Vtask_test_ecc_controller__DOT__write_memory__11__data = 0;
    CData/*1:0*/ __Vtask_test_ecc_controller__DOT__inject_error__12__error_type;
    __Vtask_test_ecc_controller__DOT__inject_error__12__error_type = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__13__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__13__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__13__data;
    __Vtask_test_ecc_controller__DOT__read_memory__13__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__write_memory__14__addr;
    __Vtask_test_ecc_controller__DOT__write_memory__14__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__write_memory__14__data;
    __Vtask_test_ecc_controller__DOT__write_memory__14__data = 0;
    IData/*31:0*/ __Vtask_test_ecc_controller__DOT__read_memory__15__addr;
    __Vtask_test_ecc_controller__DOT__read_memory__15__addr = 0;
    QData/*63:0*/ __Vtask_test_ecc_controller__DOT__read_memory__15__data;
    __Vtask_test_ecc_controller__DOT__read_memory__15__data = 0;
    // Body
    VL_WRITEF_NX("Starting ECC Controller Test\n",0);
    vlSelfRef.test_ecc_controller__DOT__error_count = 0U;
    vlSelfRef.test_ecc_controller__DOT__test_count = 0U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 0;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_2__DOT____Vrepeat1 = 0;
    vlSelfRef.test_ecc_controller__DOT__rst_n = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = 0ULL;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 0U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_type = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         98);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 4U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         98);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 3U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         98);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 2U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         98);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 1U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         98);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_1__DOT____Vrepeat0 = 0U;
    vlSelfRef.test_ecc_controller__DOT__rst_n = 1U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         100);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_2__DOT____Vrepeat1 = 1U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         100);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__reset_system__0__test_ecc_controller__DOT__unnamedblk1_2__DOT____Vrepeat1 = 0U;
    VL_WRITEF_NX("Test 1: Basic write/read without errors\n",0);
    vlSelfRef.test_ecc_controller__DOT__test_data = 0xdeadbeefcafebabeULL;
    __Vtask_test_ecc_controller__DOT__write_memory__1__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__1__addr = 0x100U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__1__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__1__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__read_memory__2__addr = 0x100U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__2__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__2__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__2__data;
    if ((vlSelfRef.test_ecc_controller__DOT__read_data 
         == vlSelfRef.test_ecc_controller__DOT__test_data)) {
        VL_WRITEF_NX("PASS: Basic write/read test\n",0);
    } else {
        VL_WRITEF_NX("FAIL: Basic write/read test - Expected: %x, Got: %x\n",0,
                     64,vlSelfRef.test_ecc_controller__DOT__test_data,
                     64,vlSelfRef.test_ecc_controller__DOT__read_data);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("Test 2: Single bit error injection and correction\n",0);
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdef0ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__3__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__3__addr = 0x200U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__3__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__3__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__inject_error__4__error_type = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_type 
        = __Vtask_test_ecc_controller__DOT__inject_error__4__error_type;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         131);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 0U;
    __Vtask_test_ecc_controller__DOT__read_memory__5__addr = 0x200U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__5__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__5__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__5__data;
    if (((vlSelfRef.test_ecc_controller__DOT__read_data 
          == vlSelfRef.test_ecc_controller__DOT__test_data) 
         & (IData)(vlSelfRef.test_ecc_controller__DOT__single_error))) {
        VL_WRITEF_NX("PASS: Single bit error correction\n",0);
    } else {
        VL_WRITEF_NX("FAIL: Single bit error correction - Expected: %x, Got: %x, Single Error: %b\n",0,
                     64,vlSelfRef.test_ecc_controller__DOT__test_data,
                     64,vlSelfRef.test_ecc_controller__DOT__read_data,
                     1,(IData)(vlSelfRef.test_ecc_controller__DOT__single_error));
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("Test 3: Double bit error detection\n",0);
    vlSelfRef.test_ecc_controller__DOT__test_data = 0xfedcba9876543210ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__6__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__6__addr = 0x300U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__6__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__6__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__inject_error__7__error_type = 2U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_type 
        = __Vtask_test_ecc_controller__DOT__inject_error__7__error_type;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         131);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 0U;
    __Vtask_test_ecc_controller__DOT__read_memory__8__addr = 0x300U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__8__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__8__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__8__data;
    if (vlSelfRef.test_ecc_controller__DOT__double_error) {
        VL_WRITEF_NX("PASS: Double bit error detection\n",0);
    } else {
        VL_WRITEF_NX("FAIL: Double bit error detection - Double Error: %b\n",0,
                     1,vlSelfRef.test_ecc_controller__DOT__double_error);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("Test 4: Multiple memory locations\n",0);
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdefULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x400U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 1U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf0ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x408U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 2U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf1ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x410U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 3U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf2ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x418U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 4U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf3ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x420U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 5U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf4ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x428U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 6U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf5ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x430U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 7U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf6ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x438U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 8U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf7ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x440U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 9U;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf8ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x448U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xaU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdf9ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x450U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xbU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdfaULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x458U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xcU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdfbULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x460U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xdU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdfcULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x468U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xeU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdfdULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x470U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0xfU;
    vlSelfRef.test_ecc_controller__DOT__test_data = 0x123456789abcdfeULL;
    __Vtask_test_ecc_controller__DOT__write_memory__9__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__9__addr = 0x478U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__9__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__9__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i = 0x10U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x400U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdefULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000400\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 1U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x408U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf0ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000408\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 2U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x410U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf1ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000410\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 3U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x418U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf2ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000418\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 4U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x420U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf3ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000420\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 5U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x428U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf4ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000428\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 6U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x430U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf5ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000430\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 7U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x438U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf6ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000438\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 8U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x440U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf7ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000440\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 9U;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x448U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf8ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000448\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xaU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x450U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdf9ULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000450\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xbU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x458U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdfaULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000458\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xcU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x460U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdfbULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000460\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xdU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x468U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdfcULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000468\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xeU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x470U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdfdULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000470\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0xfU;
    __Vtask_test_ecc_controller__DOT__read_memory__10__addr = 0x478U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__10__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__10__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__10__data;
    if (VL_UNLIKELY(((0x123456789abcdfeULL != vlSelfRef.test_ecc_controller__DOT__read_data)))) {
        VL_WRITEF_NX("FAIL: Multiple locations test at address 00000478\n",0);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i = 0x10U;
    if (VL_UNLIKELY(((0U == vlSelfRef.test_ecc_controller__DOT__error_count)))) {
        VL_WRITEF_NX("PASS: Multiple memory locations test\n",0);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("Test 5: Error address reporting\n",0);
    vlSelfRef.test_ecc_controller__DOT__test_data = 0xa5a5a5a5a5a5a5a5ULL;
    __Vtask_test_ecc_controller__DOT__write_memory__11__data 
        = vlSelfRef.test_ecc_controller__DOT__test_data;
    __Vtask_test_ecc_controller__DOT__write_memory__11__addr = 0x500U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         104);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__write_memory__11__addr;
    vlSelfRef.test_ecc_controller__DOT__mem_wdata = __Vtask_test_ecc_controller__DOT__write_memory__11__data;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         109);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             110);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         113);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    __Vtask_test_ecc_controller__DOT__inject_error__12__error_type = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_type 
        = __Vtask_test_ecc_controller__DOT__inject_error__12__error_type;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         131);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__error_inject_en = 0U;
    __Vtask_test_ecc_controller__DOT__read_memory__13__addr = 0x500U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         117);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
    vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
    vlSelfRef.test_ecc_controller__DOT__mem_addr = __Vtask_test_ecc_controller__DOT__read_memory__13__addr;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         121);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
        co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                             nullptr, 
                                                             "@(posedge test_ecc_controller.clk)", 
                                                             "test_ecc_controller.sv", 
                                                             122);
        vlSelfRef.__Vm_traceActivity[2U] = 1U;
    }
    __Vtask_test_ecc_controller__DOT__read_memory__13__data 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
    vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
    co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                         nullptr, 
                                                         "@(posedge test_ecc_controller.clk)", 
                                                         "test_ecc_controller.sv", 
                                                         125);
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.test_ecc_controller__DOT__read_data = __Vtask_test_ecc_controller__DOT__read_memory__13__data;
    if (((0x500U == vlSelfRef.test_ecc_controller__DOT__error_addr) 
         & (IData)(vlSelfRef.test_ecc_controller__DOT__single_error))) {
        VL_WRITEF_NX("PASS: Error address reporting\n",0);
    } else {
        VL_WRITEF_NX("FAIL: Error address reporting - Expected: 00000500, Got: %x\n",0,
                     32,vlSelfRef.test_ecc_controller__DOT__error_addr);
        vlSelfRef.test_ecc_controller__DOT__error_count 
            = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("Test 6: Stress test with random data\n",0);
    vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i = 0U;
    {
        while (VL_GTS_III(32, 0x64U, vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i)) {
            vlSelfRef.test_ecc_controller__DOT__test_data 
                = VL_EXTENDS_QI(64,32, VL_RANDOM_I());
            __Vtask_test_ecc_controller__DOT__write_memory__14__data 
                = vlSelfRef.test_ecc_controller__DOT__test_data;
            __Vtask_test_ecc_controller__DOT__write_memory__14__addr 
                = ((IData)(0x600U) + VL_SHIFTL_III(32,32,32, vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i, 3U));
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 104);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
            vlSelfRef.test_ecc_controller__DOT__mem_we = 1U;
            vlSelfRef.test_ecc_controller__DOT__mem_addr 
                = __Vtask_test_ecc_controller__DOT__write_memory__14__addr;
            vlSelfRef.test_ecc_controller__DOT__mem_wdata 
                = __Vtask_test_ecc_controller__DOT__write_memory__14__data;
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 109);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
                co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                     nullptr, 
                                                                     "@(posedge test_ecc_controller.clk)", 
                                                                     "test_ecc_controller.sv", 
                                                                     110);
                vlSelfRef.__Vm_traceActivity[2U] = 1U;
            }
            vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
            vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 113);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            __Vtask_test_ecc_controller__DOT__read_memory__15__addr 
                = ((IData)(0x600U) + VL_SHIFTL_III(32,32,32, vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i, 3U));
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 117);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            vlSelfRef.test_ecc_controller__DOT__mem_req = 1U;
            vlSelfRef.test_ecc_controller__DOT__mem_we = 0U;
            vlSelfRef.test_ecc_controller__DOT__mem_addr 
                = __Vtask_test_ecc_controller__DOT__read_memory__15__addr;
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 121);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            while ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__array_ready)))) {
                co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                     nullptr, 
                                                                     "@(posedge test_ecc_controller.clk)", 
                                                                     "test_ecc_controller.sv", 
                                                                     122);
                vlSelfRef.__Vm_traceActivity[2U] = 1U;
            }
            __Vtask_test_ecc_controller__DOT__read_memory__15__data 
                = vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data;
            vlSelfRef.test_ecc_controller__DOT__mem_req = 0U;
            co_await vlSelfRef.__VtrigSched_hc35d34f7__0.trigger(0U, 
                                                                 nullptr, 
                                                                 "@(posedge test_ecc_controller.clk)", 
                                                                 "test_ecc_controller.sv", 
                                                                 125);
            vlSelfRef.__Vm_traceActivity[2U] = 1U;
            vlSelfRef.test_ecc_controller__DOT__read_data 
                = __Vtask_test_ecc_controller__DOT__read_memory__15__data;
            if (VL_UNLIKELY(((vlSelfRef.test_ecc_controller__DOT__read_data 
                              != vlSelfRef.test_ecc_controller__DOT__test_data)))) {
                VL_WRITEF_NX("FAIL: Stress test at iteration %11d\n",0,
                             32,vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i);
                vlSelfRef.test_ecc_controller__DOT__error_count 
                    = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__error_count);
                goto __Vlabel1;
            }
            vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i 
                = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i);
        }
        __Vlabel1: ;
    }
    if (VL_UNLIKELY(((0U == vlSelfRef.test_ecc_controller__DOT__error_count)))) {
        VL_WRITEF_NX("PASS: Stress test with random data\n",0);
    }
    vlSelfRef.test_ecc_controller__DOT__test_count 
        = ((IData)(1U) + vlSelfRef.test_ecc_controller__DOT__test_count);
    VL_WRITEF_NX("\n=== ECC Controller Test Summary ===\nTotal tests: %11d\nFailed tests: %11d\nSuccess rate: %.1f%%\n",0,
                 32,vlSelfRef.test_ecc_controller__DOT__test_count,
                 32,vlSelfRef.test_ecc_controller__DOT__error_count,
                 64,((100.0 * VL_ISTOR_D_I(32, (vlSelfRef.test_ecc_controller__DOT__test_count 
                                                - vlSelfRef.test_ecc_controller__DOT__error_count))) 
                     / VL_ISTOR_D_I(32, vlSelfRef.test_ecc_controller__DOT__test_count)));
    if ((0U == vlSelfRef.test_ecc_controller__DOT__error_count)) {
        VL_WRITEF_NX("All tests PASSED!\n",0);
    } else {
        VL_WRITEF_NX("Some tests FAILED!\n",0);
    }
    VL_FINISH_MT("test_ecc_controller.sv", 260, "");
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
}

VL_INLINE_OPT VlCoroutine Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__2(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_initial__TOP__Vtiming__2\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    co_await vlSelfRef.__VdlySched.delay(0x3b9aca00ULL, 
                                         nullptr, "test_ecc_controller.sv", 
                                         265);
    VL_WRITEF_NX("ERROR: Test timeout!\n",0);
    VL_FINISH_MT("test_ecc_controller.sv", 267, "");
}

void Vtest_ecc_controller___024root___act_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf);

void Vtest_ecc_controller___024root___eval_act(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_act\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        Vtest_ecc_controller___024root___act_sequent__TOP__0(vlSelf);
    }
}

VL_INLINE_OPT void Vtest_ecc_controller___024root___act_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___act_sequent__TOP__0\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xfcU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((2U & (VL_REDXOR_64((0xccccccccccccccccULL 
                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                     << 1U)) | (1U & VL_REDXOR_64((0xaaaaaaaaaaaaaaaaULL 
                                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xf3U & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((8U & (VL_REDXOR_64((0xff00ff00ff00ff00ULL 
                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                     << 3U)) | (4U & (VL_REDXOR_64(
                                                   (0xf0f0f0f0f0f0f0f0ULL 
                                                    & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                                      << 2U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xcfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((0x20U & (VL_REDXOR_64((0xffffffff00000000ULL 
                                      & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                        << 5U)) | (0x10U & (VL_REDXOR_64(
                                                         (0xffff0000ffff0000ULL 
                                                          & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                                            << 4U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xbfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | (0x40U & (VL_REDXOR_64(vlSelfRef.test_ecc_controller__DOT__mem_wdata) 
                       << 6U)));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | (0x80U & ((VL_REDXOR_64(vlSelfRef.test_ecc_controller__DOT__mem_wdata) 
                        ^ VL_REDXOR_32((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)))) 
                       << 7U)));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
        = (IData)(vlSelfRef.test_ecc_controller__DOT__mem_wdata);
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[1U] 
        = (IData)((vlSelfRef.test_ecc_controller__DOT__mem_wdata 
                   >> 0x20U));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[2U] 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out;
    if (((IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_en) 
         & (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we))) {
        if ((1U == (IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_type))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffeU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (1U & (~ vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U])));
        } else if ((2U == (IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_type))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffeU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (1U & (~ vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U])));
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffdU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (2U & ((~ (vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                                >> 1U)) << 1U)));
        }
    }
}

void Vtest_ecc_controller___024root___nba_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf);
void Vtest_ecc_controller___024root___nba_sequent__TOP__1(Vtest_ecc_controller___024root* vlSelf);
void Vtest_ecc_controller___024root___nba_sequent__TOP__2(Vtest_ecc_controller___024root* vlSelf);

void Vtest_ecc_controller___024root___eval_nba(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_nba\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_ecc_controller___024root___nba_sequent__TOP__0(vlSelf);
        vlSelfRef.__Vm_traceActivity[3U] = 1U;
    }
    if ((3ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_ecc_controller___024root___nba_sequent__TOP__1(vlSelf);
        vlSelfRef.__Vm_traceActivity[4U] = 1U;
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        Vtest_ecc_controller___024root___nba_sequent__TOP__2(vlSelf);
        vlSelfRef.__Vm_traceActivity[5U] = 1U;
    }
}

VL_INLINE_OPT void Vtest_ecc_controller___024root___nba_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___nba_sequent__TOP__0\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<3>/*71:0*/ __VdlyVal__test_ecc_controller__DOT__memory_array__v0;
    VL_ZERO_W(72, __VdlyVal__test_ecc_controller__DOT__memory_array__v0);
    SData/*9:0*/ __VdlyDim0__test_ecc_controller__DOT__memory_array__v0;
    __VdlyDim0__test_ecc_controller__DOT__memory_array__v0 = 0;
    CData/*0:0*/ __VdlySet__test_ecc_controller__DOT__memory_array__v0;
    __VdlySet__test_ecc_controller__DOT__memory_array__v0 = 0;
    // Body
    __VdlySet__test_ecc_controller__DOT__memory_array__v0 = 0U;
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xfcU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((2U & (VL_REDXOR_64((0xccccccccccccccccULL 
                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                     << 1U)) | (1U & VL_REDXOR_64((0xaaaaaaaaaaaaaaaaULL 
                                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xf3U & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((8U & (VL_REDXOR_64((0xff00ff00ff00ff00ULL 
                                   & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                     << 3U)) | (4U & (VL_REDXOR_64(
                                                   (0xf0f0f0f0f0f0f0f0ULL 
                                                    & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                                      << 2U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xcfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | ((0x20U & (VL_REDXOR_64((0xffffffff00000000ULL 
                                      & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                        << 5U)) | (0x10U & (VL_REDXOR_64(
                                                         (0xffff0000ffff0000ULL 
                                                          & vlSelfRef.test_ecc_controller__DOT__mem_wdata)) 
                                            << 4U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0xbfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | (0x40U & (VL_REDXOR_64(vlSelfRef.test_ecc_controller__DOT__mem_wdata) 
                       << 6U)));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out 
        = ((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)) 
           | (0x80U & ((VL_REDXOR_64(vlSelfRef.test_ecc_controller__DOT__mem_wdata) 
                        ^ VL_REDXOR_32((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out)))) 
                       << 7U)));
    if (((IData)(vlSelfRef.test_ecc_controller__DOT__mem_req) 
         & (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we))) {
        __VdlyVal__test_ecc_controller__DOT__memory_array__v0[0U] 
            = vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U];
        __VdlyVal__test_ecc_controller__DOT__memory_array__v0[1U] 
            = vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[1U];
        __VdlyVal__test_ecc_controller__DOT__memory_array__v0[2U] 
            = vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[2U];
        __VdlyDim0__test_ecc_controller__DOT__memory_array__v0 
            = (0x3ffU & vlSelfRef.test_ecc_controller__DOT__mem_addr);
        __VdlySet__test_ecc_controller__DOT__memory_array__v0 = 1U;
    }
    if ((1U & (~ ((IData)(vlSelfRef.test_ecc_controller__DOT__mem_req) 
                  & (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we))))) {
        if (((IData)(vlSelfRef.test_ecc_controller__DOT__mem_req) 
             & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we)))) {
            vlSelfRef.test_ecc_controller__DOT__array_rdata[0U] 
                = vlSelfRef.test_ecc_controller__DOT__memory_array
                [(0x3ffU & vlSelfRef.test_ecc_controller__DOT__mem_addr)][0U];
            vlSelfRef.test_ecc_controller__DOT__array_rdata[1U] 
                = vlSelfRef.test_ecc_controller__DOT__memory_array
                [(0x3ffU & vlSelfRef.test_ecc_controller__DOT__mem_addr)][1U];
            vlSelfRef.test_ecc_controller__DOT__array_rdata[2U] 
                = vlSelfRef.test_ecc_controller__DOT__memory_array
                [(0x3ffU & vlSelfRef.test_ecc_controller__DOT__mem_addr)][2U];
        }
    }
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
        = (IData)(vlSelfRef.test_ecc_controller__DOT__mem_wdata);
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[1U] 
        = (IData)((vlSelfRef.test_ecc_controller__DOT__mem_wdata 
                   >> 0x20U));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[2U] 
        = vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out;
    if (((IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_en) 
         & (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we))) {
        if ((1U == (IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_type))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffeU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (1U & (~ vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U])));
        } else if ((2U == (IData)(vlSelfRef.test_ecc_controller__DOT__error_inject_type))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffeU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (1U & (~ vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U])));
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                = ((0xfffffffdU & vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U]) 
                   | (2U & ((~ (vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data[0U] 
                                >> 1U)) << 1U)));
        }
    }
    if (__VdlySet__test_ecc_controller__DOT__memory_array__v0) {
        vlSelfRef.test_ecc_controller__DOT__memory_array[__VdlyDim0__test_ecc_controller__DOT__memory_array__v0][0U] 
            = __VdlyVal__test_ecc_controller__DOT__memory_array__v0[0U];
        vlSelfRef.test_ecc_controller__DOT__memory_array[__VdlyDim0__test_ecc_controller__DOT__memory_array__v0][1U] 
            = __VdlyVal__test_ecc_controller__DOT__memory_array__v0[1U];
        vlSelfRef.test_ecc_controller__DOT__memory_array[__VdlyDim0__test_ecc_controller__DOT__memory_array__v0][2U] 
            = __VdlyVal__test_ecc_controller__DOT__memory_array__v0[2U];
    }
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity 
        = (1U & (VL_REDXOR_32(vlSelfRef.test_ecc_controller__DOT__array_rdata[2U]) 
                 ^ VL_REDXOR_64((((QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                  << 0x20U) | (QData)((IData)(
                                                              vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc 
        = ((0xfcU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)) 
           | ((2U & (VL_REDXOR_64((0xccccccccccccccccULL 
                                   & (((QData)((IData)(
                                                       vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                       << 0x20U) | (QData)((IData)(
                                                                   vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))) 
                     << 1U)) | (1U & VL_REDXOR_64((0xaaaaaaaaaaaaaaaaULL 
                                                   & (((QData)((IData)(
                                                                       vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                                       << 0x20U) 
                                                      | (QData)((IData)(
                                                                        vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc 
        = ((0xf3U & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)) 
           | ((8U & (VL_REDXOR_64((0xff00ff00ff00ff00ULL 
                                   & (((QData)((IData)(
                                                       vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                       << 0x20U) | (QData)((IData)(
                                                                   vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))) 
                     << 3U)) | (4U & (VL_REDXOR_64(
                                                   (0xf0f0f0f0f0f0f0f0ULL 
                                                    & (((QData)((IData)(
                                                                        vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                                        << 0x20U) 
                                                       | (QData)((IData)(
                                                                         vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))) 
                                      << 2U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc 
        = ((0xcfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)) 
           | ((0x20U & (VL_REDXOR_64((0xffffffff00000000ULL 
                                      & (((QData)((IData)(
                                                          vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                          << 0x20U) 
                                         | (QData)((IData)(
                                                           vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))) 
                        << 5U)) | (0x10U & (VL_REDXOR_64(
                                                         (0xffff0000ffff0000ULL 
                                                          & (((QData)((IData)(
                                                                              vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                                              << 0x20U) 
                                                             | (QData)((IData)(
                                                                               vlSelfRef.test_ecc_controller__DOT__array_rdata[0U]))))) 
                                            << 4U))));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc 
        = ((0xbfU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)) 
           | (0x40U & (VL_REDXOR_64((((QData)((IData)(
                                                      vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                      << 0x20U) | (QData)((IData)(
                                                                  vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])))) 
                       << 6U)));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc 
        = ((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)) 
           | (0x80U & ((VL_REDXOR_64((((QData)((IData)(
                                                       vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                       << 0x20U) | (QData)((IData)(
                                                                   vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])))) 
                        ^ VL_REDXOR_32((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc)))) 
                       << 7U)));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome 
        = (0xffU & ((IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc) 
                    ^ vlSelfRef.test_ecc_controller__DOT__array_rdata[2U]));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data 
        = (((QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
            << 0x20U) | (QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])));
    if ((0U != (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome))) {
        if (vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity) {
            if (((0U != (0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome))) 
                 & (0x40U >= (0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome))))) {
                vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data 
                    = (((QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                        << 0x20U) | (QData)((IData)(
                                                    vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])));
                vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data 
                    = (((~ (1ULL << (0x3fU & ((0x7fU 
                                               & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome)) 
                                              - (IData)(1U))))) 
                        & vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data) 
                       | ((QData)((IData)((1U & (~ 
                                                 (vlSelfRef.test_ecc_controller__DOT__array_rdata[
                                                  (1U 
                                                   & (((0x7fU 
                                                        & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome)) 
                                                       - (IData)(1U)) 
                                                      >> 5U))] 
                                                  >> 
                                                  (0x1fU 
                                                   & ((0x7fU 
                                                       & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome)) 
                                                      - (IData)(1U)))))))) 
                          << (0x3fU & ((0x7fU & (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome)) 
                                       - (IData)(1U)))));
            }
        }
    }
}

VL_INLINE_OPT void Vtest_ecc_controller___024root___nba_sequent__TOP__1(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___nba_sequent__TOP__1\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (vlSelfRef.test_ecc_controller__DOT__rst_n) {
        if (((IData)(vlSelfRef.test_ecc_controller__DOT__array_ready) 
             & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__mem_we)))) {
            vlSelfRef.test_ecc_controller__DOT__single_error 
                = vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected;
            vlSelfRef.test_ecc_controller__DOT__double_error 
                = vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected;
            if (((IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected) 
                 | (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected))) {
                vlSelfRef.test_ecc_controller__DOT__error_addr 
                    = vlSelfRef.test_ecc_controller__DOT__mem_addr;
            }
        }
    } else {
        vlSelfRef.test_ecc_controller__DOT__single_error = 0U;
        vlSelfRef.test_ecc_controller__DOT__double_error = 0U;
        vlSelfRef.test_ecc_controller__DOT__error_addr = 0U;
    }
}

VL_INLINE_OPT void Vtest_ecc_controller___024root___nba_sequent__TOP__2(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___nba_sequent__TOP__2\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected = 0U;
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected = 0U;
    if ((0U != (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome))) {
        if (vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected = 1U;
        }
        if ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity)))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected = 1U;
        }
    }
    vlSelfRef.test_ecc_controller__DOT__array_ready 
        = vlSelfRef.test_ecc_controller__DOT__mem_req;
}

void Vtest_ecc_controller___024root___timing_commit(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___timing_commit\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((! (1ULL & vlSelfRef.__VactTriggered.word(0U)))) {
        vlSelfRef.__VtrigSched_hc35d34f7__0.commit(
                                                   "@(posedge test_ecc_controller.clk)");
    }
}

void Vtest_ecc_controller___024root___timing_resume(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___timing_resume\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VtrigSched_hc35d34f7__0.resume(
                                                   "@(posedge test_ecc_controller.clk)");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        vlSelfRef.__VdlySched.resume();
    }
}

void Vtest_ecc_controller___024root___eval_triggers__act(Vtest_ecc_controller___024root* vlSelf);

bool Vtest_ecc_controller___024root___eval_phase__act(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_phase__act\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlTriggerVec<3> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    Vtest_ecc_controller___024root___eval_triggers__act(vlSelf);
    Vtest_ecc_controller___024root___timing_commit(vlSelf);
    __VactExecute = vlSelfRef.__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelfRef.__VactTriggered, vlSelfRef.__VnbaTriggered);
        vlSelfRef.__VnbaTriggered.thisOr(vlSelfRef.__VactTriggered);
        Vtest_ecc_controller___024root___timing_resume(vlSelf);
        Vtest_ecc_controller___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool Vtest_ecc_controller___024root___eval_phase__nba(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_phase__nba\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelfRef.__VnbaTriggered.any();
    if (__VnbaExecute) {
        Vtest_ecc_controller___024root___eval_nba(vlSelf);
        vlSelfRef.__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__nba(Vtest_ecc_controller___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__act(Vtest_ecc_controller___024root* vlSelf);
#endif  // VL_DEBUG

void Vtest_ecc_controller___024root___eval(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vtest_ecc_controller___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("test_ecc_controller.sv", 8, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelfRef.__VactIterCount = 0U;
        vlSelfRef.__VactContinue = 1U;
        while (vlSelfRef.__VactContinue) {
            if (VL_UNLIKELY(((0x64U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                Vtest_ecc_controller___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("test_ecc_controller.sv", 8, "", "Active region did not converge.");
            }
            vlSelfRef.__VactIterCount = ((IData)(1U) 
                                         + vlSelfRef.__VactIterCount);
            vlSelfRef.__VactContinue = 0U;
            if (Vtest_ecc_controller___024root___eval_phase__act(vlSelf)) {
                vlSelfRef.__VactContinue = 1U;
            }
        }
        if (Vtest_ecc_controller___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void Vtest_ecc_controller___024root___eval_debug_assertions(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_debug_assertions\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
