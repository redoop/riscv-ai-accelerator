// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtest_simple_tpu_mac__Syms.h"


void Vtest_simple_tpu_mac___024root__trace_chg_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtest_simple_tpu_mac___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_chg_0\n"); );
    // Init
    Vtest_simple_tpu_mac___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_simple_tpu_mac___024root*>(voidSelf);
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtest_simple_tpu_mac___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtest_simple_tpu_mac___024root__trace_chg_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_chg_0_sub_0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(((vlSelfRef.__Vm_traceActivity[1U] 
                      | vlSelfRef.__Vm_traceActivity
                      [2U])))) {
        bufp->chgBit(oldp+0,(vlSelfRef.test_simple_tpu_mac__DOT__rst_n));
        bufp->chgBit(oldp+1,(vlSelfRef.test_simple_tpu_mac__DOT__enable));
        bufp->chgCData(oldp+2,(vlSelfRef.test_simple_tpu_mac__DOT__data_type),3);
        bufp->chgSData(oldp+3,(vlSelfRef.test_simple_tpu_mac__DOT__a_data),16);
        bufp->chgSData(oldp+4,(vlSelfRef.test_simple_tpu_mac__DOT__b_data),16);
        bufp->chgIData(oldp+5,(vlSelfRef.test_simple_tpu_mac__DOT__c_data),32);
        bufp->chgBit(oldp+6,(vlSelfRef.test_simple_tpu_mac__DOT__valid_in));
    }
    bufp->chgBit(oldp+7,(vlSelfRef.test_simple_tpu_mac__DOT__clk));
    bufp->chgIData(oldp+8,(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result),32);
    bufp->chgBit(oldp+9,(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid));
}

void Vtest_simple_tpu_mac___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_cleanup\n"); );
    // Init
    Vtest_simple_tpu_mac___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_simple_tpu_mac___024root*>(voidSelf);
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
}
