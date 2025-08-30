// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vsimple_mac_test.h for the primary calling header

#ifndef VERILATED_VSIMPLE_MAC_TEST___024ROOT_H_
#define VERILATED_VSIMPLE_MAC_TEST___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vsimple_mac_test__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vsimple_mac_test___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ simple_mac_test__DOT__clk;
    CData/*0:0*/ simple_mac_test__DOT__rst_n;
    CData/*0:0*/ simple_mac_test__DOT__valid;
    CData/*0:0*/ simple_mac_test__DOT__ready;
    CData/*0:0*/ __Vtrigprevexpr___TOP__simple_mac_test__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__simple_mac_test__DOT__rst_n__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__simple_mac_test__DOT__ready__0;
    CData/*0:0*/ __VactDidInit;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ simple_mac_test__DOT__a;
    SData/*15:0*/ simple_mac_test__DOT__b;
    IData/*31:0*/ simple_mac_test__DOT__c;
    IData/*31:0*/ simple_mac_test__DOT__result;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*0:0*/, 3> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_haa7fc2ca__0;
    VlTriggerScheduler __VtrigSched_h5ca73cd1__0;
    VlTriggerVec<4> __VactTriggered;
    VlTriggerVec<4> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vsimple_mac_test__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vsimple_mac_test___024root(Vsimple_mac_test__Syms* symsp, const char* v__name);
    ~Vsimple_mac_test___024root();
    VL_UNCOPYABLE(Vsimple_mac_test___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
