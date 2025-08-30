// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtest_ecc_controller__Syms.h"


void Vtest_ecc_controller___024root__trace_chg_0_sub_0(Vtest_ecc_controller___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vtest_ecc_controller___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root__trace_chg_0\n"); );
    // Init
    Vtest_ecc_controller___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_ecc_controller___024root*>(voidSelf);
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vtest_ecc_controller___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vtest_ecc_controller___024root__trace_chg_0_sub_0(Vtest_ecc_controller___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root__trace_chg_0_sub_0\n"); );
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(((vlSelfRef.__Vm_traceActivity[1U] 
                      | vlSelfRef.__Vm_traceActivity
                      [2U])))) {
        bufp->chgBit(oldp+0,(vlSelfRef.test_ecc_controller__DOT__rst_n));
        bufp->chgBit(oldp+1,(vlSelfRef.test_ecc_controller__DOT__mem_req));
        bufp->chgBit(oldp+2,(vlSelfRef.test_ecc_controller__DOT__mem_we));
        bufp->chgIData(oldp+3,(vlSelfRef.test_ecc_controller__DOT__mem_addr),32);
        bufp->chgQData(oldp+4,(vlSelfRef.test_ecc_controller__DOT__mem_wdata),64);
        bufp->chgBit(oldp+6,(vlSelfRef.test_ecc_controller__DOT__error_inject_en));
        bufp->chgCData(oldp+7,(vlSelfRef.test_ecc_controller__DOT__error_inject_type),2);
        bufp->chgQData(oldp+8,(vlSelfRef.test_ecc_controller__DOT__test_data),64);
        bufp->chgQData(oldp+10,(vlSelfRef.test_ecc_controller__DOT__read_data),64);
        bufp->chgIData(oldp+12,(vlSelfRef.test_ecc_controller__DOT__error_count),32);
        bufp->chgIData(oldp+13,(vlSelfRef.test_ecc_controller__DOT__test_count),32);
        bufp->chgIData(oldp+14,(vlSelfRef.test_ecc_controller__DOT__unnamedblk1__DOT__i),32);
        bufp->chgIData(oldp+15,(vlSelfRef.test_ecc_controller__DOT__unnamedblk2__DOT__i),32);
        bufp->chgIData(oldp+16,(vlSelfRef.test_ecc_controller__DOT__unnamedblk3__DOT__i),32);
    }
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[3U]))) {
        bufp->chgQData(oldp+17,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__corrected_data),64);
        bufp->chgWData(oldp+19,(vlSelfRef.test_ecc_controller__DOT__array_rdata),72);
        bufp->chgCData(oldp+22,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_syndrome),8);
        bufp->chgQData(oldp+23,((((QData)((IData)(vlSelfRef.test_ecc_controller__DOT__array_rdata[1U])) 
                                  << 0x20U) | (QData)((IData)(
                                                              vlSelfRef.test_ecc_controller__DOT__array_rdata[0U])))),64);
        bufp->chgCData(oldp+25,((0xffU & vlSelfRef.test_ecc_controller__DOT__array_rdata[2U])),8);
        bufp->chgCData(oldp+26,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc),8);
        bufp->chgBit(oldp+27,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity));
    }
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[4U]))) {
        bufp->chgBit(oldp+28,(vlSelfRef.test_ecc_controller__DOT__single_error));
        bufp->chgBit(oldp+29,(vlSelfRef.test_ecc_controller__DOT__double_error));
        bufp->chgIData(oldp+30,(vlSelfRef.test_ecc_controller__DOT__error_addr),32);
    }
    if (VL_UNLIKELY((vlSelfRef.__Vm_traceActivity[5U]))) {
        bufp->chgBit(oldp+31,(vlSelfRef.test_ecc_controller__DOT__array_ready));
        bufp->chgBit(oldp+32,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__single_err_detected));
        bufp->chgBit(oldp+33,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__double_err_detected));
    }
    bufp->chgBit(oldp+34,(vlSelfRef.test_ecc_controller__DOT__clk));
    bufp->chgWData(oldp+35,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__injected_data),72);
    bufp->chgCData(oldp+38,(vlSelfRef.test_ecc_controller__DOT__dut__DOT__ecc_encode_out),8);
}

void Vtest_ecc_controller___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_ecc_controller___024root__trace_cleanup\n"); );
    // Init
    Vtest_ecc_controller___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_ecc_controller___024root*>(voidSelf);
    Vtest_ecc_controller__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[3U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[4U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[5U] = 0U;
}
