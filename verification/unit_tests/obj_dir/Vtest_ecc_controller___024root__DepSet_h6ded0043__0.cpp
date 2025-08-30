// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_ecc_controller.h for the primary calling header

#include "Vtest_ecc_controller__pch.h"
#include "Vtest_ecc_controller__Syms.h"
#include "Vtest_ecc_controller___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__act(Vtest_ecc_controller___024root* vlSelf);
#endif  // VL_DEBUG

void Vtest_ecc_controller___024root___eval_triggers__act(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_triggers__act\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.setBit(0U, ((IData)(vlSelfRef.test_ecc_controller__DOT__clk) 
                                          & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__clk__0))));
    vlSelfRef.__VactTriggered.setBit(1U, ((~ (IData)(vlSelfRef.test_ecc_controller__DOT__rst_n)) 
                                          & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__rst_n__0)));
    vlSelfRef.__VactTriggered.setBit(2U, vlSelfRef.__VdlySched.awaitingCurrentTime());
    vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__clk__0 
        = vlSelfRef.test_ecc_controller__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__rst_n__0 
        = vlSelfRef.test_ecc_controller__DOT__rst_n;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtest_ecc_controller___024root___dump_triggers__act(vlSelf);
    }
#endif
}
