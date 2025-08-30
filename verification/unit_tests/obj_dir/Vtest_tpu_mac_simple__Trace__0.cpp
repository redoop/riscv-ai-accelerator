// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtest_tpu_mac_simple__Syms.h"


void Vtest_tpu_mac_simple___024root__trace_chg_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtest_tpu_mac_simple___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_chg_0\n"); );
    // Init
    Vtest_tpu_mac_simple___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_tpu_mac_simple___024root*>(voidSelf);
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtest_tpu_mac_simple___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtest_tpu_mac_simple___024root__trace_chg_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_chg_0_sub_0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(((vlSelfRef.__Vm_traceActivity[1U] 
                      | vlSelfRef.__Vm_traceActivity
                      [2U])))) {
        bufp->chgBit(oldp+0,(vlSelfRef.test_tpu_mac_simple__DOT__rst_n));
        bufp->chgBit(oldp+1,(vlSelfRef.test_tpu_mac_simple__DOT__enable));
        bufp->chgCData(oldp+2,(vlSelfRef.test_tpu_mac_simple__DOT__data_type),2);
        bufp->chgIData(oldp+3,(vlSelfRef.test_tpu_mac_simple__DOT__a_in),32);
        bufp->chgIData(oldp+4,(vlSelfRef.test_tpu_mac_simple__DOT__b_in),32);
        bufp->chgIData(oldp+5,(vlSelfRef.test_tpu_mac_simple__DOT__c_in),32);
        bufp->chgBit(oldp+6,(vlSelfRef.test_tpu_mac_simple__DOT__load_weight));
        bufp->chgBit(oldp+7,(vlSelfRef.test_tpu_mac_simple__DOT__accumulate));
        bufp->chgIData(oldp+8,(vlSelfRef.test_tpu_mac_simple__DOT__test_count),32);
        bufp->chgIData(oldp+9,(vlSelfRef.test_tpu_mac_simple__DOT__pass_count),32);
    }
    if (VL_UNLIKELY(((vlSelfRef.__Vm_traceActivity[3U] 
                      | vlSelfRef.__Vm_traceActivity
                      [5U])))) {
        bufp->chgBit(oldp+10,(vlSelfRef.test_tpu_mac_simple__DOT__overflow));
        bufp->chgBit(oldp+11,(vlSelfRef.test_tpu_mac_simple__DOT__underflow));
        bufp->chgIData(oldp+12,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result),32);
        bufp->chgIData(oldp+13,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result),32);
        bufp->chgCData(oldp+14,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8),8);
        bufp->chgSData(oldp+15,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16),16);
        bufp->chgIData(oldp+16,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32),32);
        bufp->chgSData(oldp+17,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8),16);
        bufp->chgSData(oldp+18,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16),16);
        bufp->chgIData(oldp+19,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32),32);
    }
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[4U]))) {
        bufp->chgIData(oldp+20,(vlSelfRef.test_tpu_mac_simple__DOT__a_out),32);
        bufp->chgIData(oldp+21,(vlSelfRef.test_tpu_mac_simple__DOT__b_out),32);
        bufp->chgIData(oldp+22,(vlSelfRef.test_tpu_mac_simple__DOT__c_out),32);
        bufp->chgIData(oldp+23,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg),32);
    }
    bufp->chgBit(oldp+24,(vlSelfRef.test_tpu_mac_simple__DOT__clk));
    bufp->chgCData(oldp+25,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8),8);
    bufp->chgSData(oldp+26,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16),16);
    bufp->chgIData(oldp+27,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32),32);
}

void Vtest_tpu_mac_simple___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_cleanup\n"); );
    // Init
    Vtest_tpu_mac_simple___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_tpu_mac_simple___024root*>(voidSelf);
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[3U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[4U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[5U] = 0U;
}
