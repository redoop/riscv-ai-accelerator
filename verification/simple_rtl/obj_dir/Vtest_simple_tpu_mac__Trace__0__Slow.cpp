// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vtest_simple_tpu_mac__Syms.h"


VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_init_sub__TOP__0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_init_sub__TOP__0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("test_simple_tpu_mac", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+8,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+1,0,"rst_n",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"enable",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+3,0,"data_type",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+4,0,"a_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+5,0,"b_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+6,0,"c_data",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+7,0,"valid_in",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+9,0,"result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+10,0,"valid_out",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"ready",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("dut", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBus(c+11,0,"DATA_WIDTH",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::PARAMETER, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+8,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+1,0,"rst_n",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+3,0,"data_type",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 2,0);
    tracep->declBus(c+4,0,"a_data",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+5,0,"b_data",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+6,0,"c_data",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+7,0,"valid_in",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+9,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+10,0,"valid_out",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+2,0,"ready",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+9,0,"mac_result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+10,0,"result_valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_init_top(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_init_top\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtest_simple_tpu_mac___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtest_simple_tpu_mac___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vtest_simple_tpu_mac___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_register(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_register\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    tracep->addConstCb(&Vtest_simple_tpu_mac___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vtest_simple_tpu_mac___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vtest_simple_tpu_mac___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vtest_simple_tpu_mac___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_const_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_const_0\n"); );
    // Init
    Vtest_simple_tpu_mac___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_simple_tpu_mac___024root*>(voidSelf);
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtest_simple_tpu_mac___024root__trace_const_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_const_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_const_0_sub_0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullIData(oldp+11,(0x20U),32);
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_full_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_full_0\n"); );
    // Init
    Vtest_simple_tpu_mac___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vtest_simple_tpu_mac___024root*>(voidSelf);
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vtest_simple_tpu_mac___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vtest_simple_tpu_mac___024root__trace_full_0_sub_0(Vtest_simple_tpu_mac___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtest_simple_tpu_mac___024root__trace_full_0_sub_0\n"); );
    Vtest_simple_tpu_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullBit(oldp+1,(vlSelfRef.test_simple_tpu_mac__DOT__rst_n));
    bufp->fullBit(oldp+2,(vlSelfRef.test_simple_tpu_mac__DOT__enable));
    bufp->fullCData(oldp+3,(vlSelfRef.test_simple_tpu_mac__DOT__data_type),3);
    bufp->fullSData(oldp+4,(vlSelfRef.test_simple_tpu_mac__DOT__a_data),16);
    bufp->fullSData(oldp+5,(vlSelfRef.test_simple_tpu_mac__DOT__b_data),16);
    bufp->fullIData(oldp+6,(vlSelfRef.test_simple_tpu_mac__DOT__c_data),32);
    bufp->fullBit(oldp+7,(vlSelfRef.test_simple_tpu_mac__DOT__valid_in));
    bufp->fullBit(oldp+8,(vlSelfRef.test_simple_tpu_mac__DOT__clk));
    bufp->fullIData(oldp+9,(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__mac_result),32);
    bufp->fullBit(oldp+10,(vlSelfRef.test_simple_tpu_mac__DOT__dut__DOT__result_valid));
}
