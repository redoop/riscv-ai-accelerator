// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtest_ecc_controller.h for the primary calling header

#include "Vtest_ecc_controller__pch.h"
#include "Vtest_ecc_controller___024root.h"

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_static(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_static\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__clk__0 
        = vlSelfRef.test_ecc_controller__DOT__clk;
    vlSelfRef.__Vtrigprevexpr___TOP__test_ecc_controller__DOT__rst_n__0 
        = vlSelfRef.test_ecc_controller__DOT__rst_n;
}

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_final(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_final\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__stl(Vtest_ecc_controller___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtest_ecc_controller___024root___eval_phase__stl(Vtest_ecc_controller___024root* vlSelf);

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_settle(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_settle\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
            Vtest_ecc_controller___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("test_ecc_controller.sv", 8, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (Vtest_ecc_controller___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelfRef.__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__stl(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___dump_triggers__stl\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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

VL_ATTR_COLD void Vtest_ecc_controller___024root___stl_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf);
VL_ATTR_COLD void Vtest_ecc_controller___024root____Vm_traceActivitySetAll(Vtest_ecc_controller___024root* vlSelf);

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_stl(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_stl\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1ULL & vlSelfRef.__VstlTriggered.word(0U))) {
        Vtest_ecc_controller___024root___stl_sequent__TOP__0(vlSelf);
        Vtest_ecc_controller___024root____Vm_traceActivitySetAll(vlSelf);
    }
}

VL_ATTR_COLD void Vtest_ecc_controller___024root___stl_sequent__TOP__0(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___stl_sequent__TOP__0\n"); );
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
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome 
        = (0xffU & ((IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc) 
                    ^ vlSelfRef.test_ecc_controller__DOT__array_rdata[2U]));
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected = 0U;
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected = 0U;
    vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data 
        = (((QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
            << 0x20U) | (QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])));
    if ((0U != (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome))) {
        if (vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected = 1U;
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
        if ((1U & (~ (IData)(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity)))) {
            vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected = 1U;
        }
    }
}

VL_ATTR_COLD void Vtest_ecc_controller___024root___eval_triggers__stl(Vtest_ecc_controller___024root* vlSelf);

VL_ATTR_COLD bool Vtest_ecc_controller___024root___eval_phase__stl(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___eval_phase__stl\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    Vtest_ecc_controller___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelfRef.__VstlTriggered.any();
    if (__VstlExecute) {
        Vtest_ecc_controller___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__act(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___dump_triggers__act\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VactTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge test_ecc_controller.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 1 is active: @(negedge test_ecc_controller.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtest_ecc_controller___024root___dump_triggers__nba(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___dump_triggers__nba\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((1U & (~ vlSelfRef.__VnbaTriggered.any()))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge test_ecc_controller.clk)\n");
    }
    if ((2ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 1 is active: @(negedge test_ecc_controller.rst_n)\n");
    }
    if ((4ULL & vlSelfRef.__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtest_ecc_controller___024root____Vm_traceActivitySetAll(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root____Vm_traceActivitySetAll\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__Vm_traceActivity[0U] = 1U;
    vlSelfRef.__Vm_traceActivity[1U] = 1U;
    vlSelfRef.__Vm_traceActivity[2U] = 1U;
    vlSelfRef.__Vm_traceActivity[3U] = 1U;
    vlSelfRef.__Vm_traceActivity[4U] = 1U;
    vlSelfRef.__Vm_traceActivity[5U] = 1U;
}

VL_ATTR_COLD void Vtest_ecc_controller___024root___ctor_var_reset(Vtest_ecc_controller___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root___ctor_var_reset\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->name());
    vlSelf->test_ecc_controller__DOT__clk = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 8902595878825024355ull);
    vlSelf->test_ecc_controller__DOT__rst_n = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 3828115807704565521ull);
    vlSelf->test_ecc_controller__DOT__mem_req = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 2164992045029713014ull);
    vlSelf->test_ecc_controller__DOT__mem_we = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9438867986872514671ull);
    vlSelf->test_ecc_controller__DOT__mem_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 8723684448323310174ull);
    vlSelf->test_ecc_controller__DOT__mem_wdata = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 16584744625072097480ull);
    vlSelf->test_ecc_controller__DOT__single_error = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 4485971987476787361ull);
    vlSelf->test_ecc_controller__DOT__double_error = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 11662462208458744721ull);
    vlSelf->test_ecc_controller__DOT__error_addr = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 11152611066768960553ull);
    vlSelf->test_ecc_controller__DOT__error_inject_en = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14063291196695025016ull);
    vlSelf->test_ecc_controller__DOT__error_inject_type = VL_SCOPED_RAND_RESET_I(2, __VscopeHash, 1365082749981577231ull);
    VL_SCOPED_RAND_RESET_W(72, vlSelf->test_ecc_controller__DOT__array_rdata, __VscopeHash, 831035962174953190ull);
    vlSelf->test_ecc_controller__DOT__array_ready = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6456273684268006391ull);
    for (int __Vi0 = 0; __Vi0 < 1024; ++__Vi0) {
        VL_SCOPED_RAND_RESET_W(72, vlSelf->test_ecc_controller__DOT__memory_array[__Vi0], __VscopeHash, 17565184035934785632ull);
    }
    vlSelf->test_ecc_controller__DOT__test_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 15250893850125588637ull);
    vlSelf->test_ecc_controller__DOT__read_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 2378260722527488942ull);
    vlSelf->test_ecc_controller__DOT__error_count = 0;
    vlSelf->test_ecc_controller__DOT__test_count = 0;
    vlSelf->test_ecc_controller__DOT__unnamedblk1__DOT__i = 0;
    vlSelf->test_ecc_controller__DOT__unnamedblk2__DOT__i = 0;
    vlSelf->test_ecc_controller__DOT__unnamedblk3__DOT__i = 0;
    vlSelf->test_ecc_controller__DOT__dut__DOT__ecc_encode_out = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 10609092059097488459ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__ecc_syndrome = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 15633628988640289207ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__corrected_data = VL_SCOPED_RAND_RESET_Q(64, __VscopeHash, 4648379153432131919ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__single_err_detected = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 867005668108463385ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__double_err_detected = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 543408104021881043ull);
    VL_SCOPED_RAND_RESET_W(72, vlSelf->test_ecc_controller__DOT__dut__DOT__injected_data, __VscopeHash, 12396592143382044433ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc = VL_SCOPED_RAND_RESET_I(8, __VscopeHash, 16236837299456514561ull);
    vlSelf->test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 6668781637726740319ull);
    vlSelf->__Vtrigprevexpr___TOP__test_ecc_controller__DOT__clk__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 13734698036877037883ull);
    vlSelf->__Vtrigprevexpr___TOP__test_ecc_controller__DOT__rst_n__0 = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 9780983795256113498ull);
    for (int __Vi0 = 0; __Vi0 < 6; ++__Vi0) {
        vlSelf->__Vm_traceActivity[__Vi0] = 0;
    }
}
