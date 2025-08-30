// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_ecc_controller.h for the primary calling header

#include "Vtest_ecc_controller__pch.h"
#include "Vtest_ecc_controller__Syms.h"
#include "Vtest_ecc_controller___024root.h"

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_initial__TOP(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_initial__TOP\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    VlWide<6>/*191:0*/ __Vtemp_1;
    // Body
    __Vtemp_1[0U] = 0x2e766364U;
    __Vtemp_1[1U] = 0x6c6c6572U;
    __Vtemp_1[2U] = 0x6e74726fU;
    __Vtemp_1[3U] = 0x635f636fU;
    __Vtemp_1[4U] = 0x745f6563U;
    __Vtemp_1[5U] = 0x746573U;
    vlSymsp->_vm_contextp__->dumpfile(VL_CVT_PACK_STR_NW(6, __Vtemp_1));
    vlSymsp->_traceDumpOpen();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__stl(Vtest_ecc_controller___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_triggers__stl(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_triggers__stl\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered.setBit(0U, (IData)(vlSelfRef.__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtest_ecc_controller___024root___dump_triggers__stl(vlSelf);
    }
#endif
}
