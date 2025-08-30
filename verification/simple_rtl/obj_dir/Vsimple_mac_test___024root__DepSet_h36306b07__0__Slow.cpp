// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vsimple_mac_test.h for the primary calling header

#include "Vsimple_mac_test__pch.h"
#include "Vsimple_mac_test__Syms.h"
#include "Vsimple_mac_test___024root.h"

VL_ATTR_COLD void Vsimple_mac_test___024root___eval_initial__TOP(Vsimple_mac_test___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root___eval_initial__TOP\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<5>/*159:0*/ __Vtemp_1;
    // Body
    vlSelfRef.simple_mac_test__DOT__clk = 0U;
    __Vtemp_1[0U] = 0x2e766364U;
    __Vtemp_1[1U] = 0x74657374U;
    __Vtemp_1[2U] = 0x6d61635fU;
    __Vtemp_1[3U] = 0x706c655fU;
    __Vtemp_1[4U] = 0x73696dU;
    vlSymsp->_vm_contextp__->dumpfile(VL_CVT_PACK_STR_NW(5, __Vtemp_1));
    vlSymsp->_traceDumpOpen();
}
