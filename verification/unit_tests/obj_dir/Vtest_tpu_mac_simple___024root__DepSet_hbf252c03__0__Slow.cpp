// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_tpu_mac_simple.h for the primary calling header

#include "Vtest_tpu_mac_simple__pch.h"
#include "Vtest_tpu_mac_simple__Syms.h"
#include "Vtest_tpu_mac_simple___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___dump_triggers__stl(Vtest_tpu_mac_simple___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root___eval_triggers__stl(Vtest_tpu_mac_simple___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root___eval_triggers__stl\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered.setBit(0U, (IData)(vlSelfRef.__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtest_tpu_mac_simple___024root___dump_triggers__stl(vlSelf);
    }
#endif
}
