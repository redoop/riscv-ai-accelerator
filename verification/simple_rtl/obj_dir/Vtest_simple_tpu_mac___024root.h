// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtest_simple_tpu_mac.h for the primary calling header

#ifndef VERILATED_VTEST_SIMPLE_TPU_MAC___024ROOT_H_
#define VERILATED_VTEST_SIMPLE_TPU_MAC___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtest_simple_tpu_mac__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtest_simple_tpu_mac___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ test_simple_tpu_mac__DOT__clk;
    CData/*0:0*/ test_simple_tpu_mac__DOT__rst_n;
    CData/*0:0*/ test_simple_tpu_mac__DOT__enable;
    CData/*2:0*/ test_simple_tpu_mac__DOT__data_type;
    CData/*0:0*/ test_simple_tpu_mac__DOT__valid_in;
    CData/*0:0*/ test_simple_tpu_mac__DOT__dut__DOT__result_valid;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__rst_n__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_simple_tpu_mac__DOT__dut__DOT__result_valid__0;
    CData/*0:0*/ __VactDidInit;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ test_simple_tpu_mac__DOT__a_data;
    SData/*15:0*/ test_simple_tpu_mac__DOT__b_data;
    IData/*31:0*/ test_simple_tpu_mac__DOT__c_data;
    IData/*31:0*/ test_simple_tpu_mac__DOT__dut__DOT__mac_result;
    IData/*31:0*/ __VactIterCount;
    VlUnpacked<CData/*0:0*/, 3> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_h111eea42__0;
    VlTriggerScheduler __VtrigSched_he43d38c9__0;
    VlTriggerVec<4> __VactTriggered;
    VlTriggerVec<4> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtest_simple_tpu_mac__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtest_simple_tpu_mac___024root(Vtest_simple_tpu_mac__Syms* symsp, const char* v__name);
    ~Vtest_simple_tpu_mac___024root();
    VL_UNCOPYABLE(Vtest_simple_tpu_mac___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
