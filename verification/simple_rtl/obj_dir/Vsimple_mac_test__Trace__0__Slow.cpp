// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vsimple_mac_test__Syms.h"


VL_ATTR_COLD void Vsimple_mac_test___024root__trace_init_sub__TOP__0(Vsimple_mac_test___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_init_sub__TOP__0\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->pushPrefix("simple_mac_test", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+6,0,"clk",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+1,0,"rst_n",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+2,0,"a",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+3,0,"b",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+4,0,"c",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+5,0,"valid",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBus(c+7,0,"result",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBit(c+8,0,"ready",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_init_top(Vsimple_mac_test___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_init_top\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vsimple_mac_test___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vsimple_mac_test___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vsimple_mac_test___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vsimple_mac_test___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_register(Vsimple_mac_test___024root* vlSelf, VerilatedVcd* tracep) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_register\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    tracep->addConstCb(&Vsimple_mac_test___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vsimple_mac_test___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vsimple_mac_test___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vsimple_mac_test___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_const_0\n"); );
    // Init
    Vsimple_mac_test___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vsimple_mac_test___024root*>(voidSelf);
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
}

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_full_0_sub_0(Vsimple_mac_test___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_full_0\n"); );
    // Init
    Vsimple_mac_test___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vsimple_mac_test___024root*>(voidSelf);
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vsimple_mac_test___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vsimple_mac_test___024root__trace_full_0_sub_0(Vsimple_mac_test___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vsimple_mac_test___024root__trace_full_0_sub_0\n"); );
    Vsimple_mac_test__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullBit(oldp+1,(vlSelfRef.simple_mac_test__DOT__rst_n));
    bufp->fullSData(oldp+2,(vlSelfRef.simple_mac_test__DOT__a),16);
    bufp->fullSData(oldp+3,(vlSelfRef.simple_mac_test__DOT__b),16);
    bufp->fullIData(oldp+4,(vlSelfRef.simple_mac_test__DOT__c),32);
    bufp->fullBit(oldp+5,(vlSelfRef.simple_mac_test__DOT__valid));
    bufp->fullBit(oldp+6,(vlSelfRef.simple_mac_test__DOT__clk));
    bufp->fullIData(oldp+7,(vlSelfRef.simple_mac_test__DOT__result),32);
    bufp->fullBit(oldp+8,(vlSelfRef.simple_mac_test__DOT__ready));
}
