// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_simple_tpu_mac.h for the primary calling header

#include "Vtest_simple_tpu_mac__pch.h"
#include "Vtest_simple_tpu_mac__Syms.h"
#include "Vtest_simple_tpu_mac___024root.h"

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root___eval_initial__TOP(Vtest_simple_tpu_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root___eval_initial__TOP\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<6>/*191:0*/ __Vtemp_1;
    // Body
    vlSelfRef.test_simple_tpu_mac__DOT__clk = 0U;
    __Vtemp_1[0U] = 0x2e766364U;
    __Vtemp_1[1U] = 0x5f6d6163U;
    __Vtemp_1[2U] = 0x5f747075U;
    __Vtemp_1[3U] = 0x6d706c65U;
    __Vtemp_1[4U] = 0x745f7369U;
    __Vtemp_1[5U] = 0x746573U;
    vlSymsp->_vm_contextp__->dumpfile(VL_CVT_PACK_STR_NW(6, __Vtemp_1));
    vlSymsp->_traceDumpOpen();
}
