// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtest_tpu_mac_simple__Syms.h"


VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_init_sub__TOP__0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_init_sub__TOP__0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("test_tpu_mac_simple", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+29,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+30,0,"CLK_PERIOD",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+25,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+1,0,"rst_n",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"enable",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+3,0,"data_type",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+4,0,"a_in",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+5,0,"b_in",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+6,0,"c_in",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+21,0,"a_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+22,0,"b_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+23,0,"c_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+7,0,"load_weight",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+8,0,"accumulate",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+11,0,"overflow",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+12,0,"underflow",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+9,0,"test_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->declBus(c+10,0,"pass_count",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INT, false,-1, 31,0);
    tracep->pushPrefix("dut", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+29,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+25,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+1,0,"rst_n",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+3,0,"data_type",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 1,0);
    tracep->declBus(c+4,0,"a_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+5,0,"b_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+6,0,"c_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+21,0,"a_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+22,0,"b_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+23,0,"c_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+7,0,"load_weight",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+8,0,"accumulate",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+11,0,"overflow",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+12,0,"underflow",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+24,0,"weight_reg",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+13,0,"mult_result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+14,0,"acc_result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+26,0,"a_int8",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+15,0,"b_int8",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 7,0);
    tracep->declBus(c+27,0,"a_fp16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+16,0,"b_fp16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+28,0,"a_fp32",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+17,0,"b_fp32",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+18,0,"mult_int8",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+19,0,"mult_fp16",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+20,0,"mult_fp32",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_init_top(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_init_top\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtest_tpu_mac_simple___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtest_tpu_mac_simple___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtest_tpu_mac_simple___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_register(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_register\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    tracep->addConstCb(&Vtest_tpu_mac_simple___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vtest_tpu_mac_simple___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vtest_tpu_mac_simple___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vtest_tpu_mac_simple___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_const_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_const_0\n"); );
    // Init
    Vtest_tpu_mac_simple___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_tpu_mac_simple___024root*>(voidSelf);
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtest_tpu_mac_simple___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_const_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_const_0_sub_0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+29,(0x20U),32);
    bufp->fullIData(oldp+30,(0xaU),32);
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_full_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_full_0\n"); );
    // Init
    Vtest_tpu_mac_simple___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_tpu_mac_simple___024root*>(voidSelf);
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtest_tpu_mac_simple___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtest_tpu_mac_simple___024root__trace_full_0_sub_0(Vtest_tpu_mac_simple___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_tpu_mac_simple___024root__trace_full_0_sub_0\n"); );
    Vtest_tpu_mac_simple__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullBit(oldp+1,(vlSelfRef.test_tpu_mac_simple__DOT__rst_n));
    bufp->fullBit(oldp+2,(vlSelfRef.test_tpu_mac_simple__DOT__enable));
    bufp->fullCData(oldp+3,(vlSelfRef.test_tpu_mac_simple__DOT__data_type),2);
    bufp->fullIData(oldp+4,(vlSelfRef.test_tpu_mac_simple__DOT__a_in),32);
    bufp->fullIData(oldp+5,(vlSelfRef.test_tpu_mac_simple__DOT__b_in),32);
    bufp->fullIData(oldp+6,(vlSelfRef.test_tpu_mac_simple__DOT__c_in),32);
    bufp->fullBit(oldp+7,(vlSelfRef.test_tpu_mac_simple__DOT__load_weight));
    bufp->fullBit(oldp+8,(vlSelfRef.test_tpu_mac_simple__DOT__accumulate));
    bufp->fullIData(oldp+9,(vlSelfRef.test_tpu_mac_simple__DOT__test_count),32);
    bufp->fullIData(oldp+10,(vlSelfRef.test_tpu_mac_simple__DOT__pass_count),32);
    bufp->fullBit(oldp+11,(vlSelfRef.test_tpu_mac_simple__DOT__overflow));
    bufp->fullBit(oldp+12,(vlSelfRef.test_tpu_mac_simple__DOT__underflow));
    bufp->fullIData(oldp+13,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_result),32);
    bufp->fullIData(oldp+14,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__acc_result),32);
    bufp->fullCData(oldp+15,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_int8),8);
    bufp->fullSData(oldp+16,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp16),16);
    bufp->fullIData(oldp+17,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__b_fp32),32);
    bufp->fullSData(oldp+18,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_int8),16);
    bufp->fullSData(oldp+19,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp16),16);
    bufp->fullIData(oldp+20,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__mult_fp32),32);
    bufp->fullIData(oldp+21,(vlSelfRef.test_tpu_mac_simple__DOT__a_out),32);
    bufp->fullIData(oldp+22,(vlSelfRef.test_tpu_mac_simple__DOT__b_out),32);
    bufp->fullIData(oldp+23,(vlSelfRef.test_tpu_mac_simple__DOT__c_out),32);
    bufp->fullIData(oldp+24,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__weight_reg),32);
    bufp->fullBit(oldp+25,(vlSelfRef.test_tpu_mac_simple__DOT__clk));
    bufp->fullCData(oldp+26,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_int8),8);
    bufp->fullSData(oldp+27,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp16),16);
    bufp->fullIData(oldp+28,(vlSelfRef.test_tpu_mac_simple__DOT__dut__DOT__a_fp32),32);
}
