// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtest_tpu_mac_simple.h for the primary calling header

#ifndef VERILATED_VTEST_TPU_MAC_SIMPLE___024ROOT_H_
#define VERILATED_VTEST_TPU_MAC_SIMPLE___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtest_tpu_mac_simple__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtest_tpu_mac_simple___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ test_tpu_mac_simple__DOT__clk;
    CData/*0:0*/ test_tpu_mac_simple__DOT__rst_n;
    CData/*0:0*/ test_tpu_mac_simple__DOT__enable;
    CData/*1:0*/ test_tpu_mac_simple__DOT__data_type;
    CData/*0:0*/ test_tpu_mac_simple__DOT__load_weight;
    CData/*0:0*/ test_tpu_mac_simple__DOT__accumulate;
    CData/*0:0*/ test_tpu_mac_simple__DOT__overflow;
    CData/*0:0*/ test_tpu_mac_simple__DOT__underflow;
    CData/*7:0*/ test_tpu_mac_simple__DOT__dut__DOT__a_int8;
    CData/*7:0*/ test_tpu_mac_simple__DOT__dut__DOT__b_int8;
    CData/*0:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__Vfuncout;
    CData/*0:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__Vfuncout;
    CData/*0:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__Vfuncout;
    CData/*0:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__Vfuncout;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_tpu_mac_simple__DOT__rst_n__0;
    CData/*0:0*/ __VactContinue;
    SData/*15:0*/ test_tpu_mac_simple__DOT__dut__DOT__a_fp16;
    SData/*15:0*/ test_tpu_mac_simple__DOT__dut__DOT__b_fp16;
    SData/*15:0*/ test_tpu_mac_simple__DOT__dut__DOT__mult_int8;
    SData/*15:0*/ test_tpu_mac_simple__DOT__dut__DOT__mult_fp16;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__Vfuncout;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__a;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__b;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__Vfuncout;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__a;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_add__5__b;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_overflow__7__val;
    SData/*15:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_is_underflow__8__val;
    IData/*31:0*/ test_tpu_mac_simple__DOT__a_in;
    IData/*31:0*/ test_tpu_mac_simple__DOT__b_in;
    IData/*31:0*/ test_tpu_mac_simple__DOT__c_in;
    IData/*31:0*/ test_tpu_mac_simple__DOT__a_out;
    IData/*31:0*/ test_tpu_mac_simple__DOT__b_out;
    IData/*31:0*/ test_tpu_mac_simple__DOT__c_out;
    IData/*31:0*/ test_tpu_mac_simple__DOT__test_count;
    IData/*31:0*/ test_tpu_mac_simple__DOT__pass_count;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__weight_reg;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__mult_result;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__acc_result;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__a_fp32;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__b_fp32;
    IData/*31:0*/ test_tpu_mac_simple__DOT__dut__DOT__mult_fp32;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp16_multiply__3__temp_result;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__Vfuncout;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__a;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__b;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__Vfuncout;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__a;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_add__6__b;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_overflow__9__val;
    IData/*31:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_is_underflow__10__val;
    IData/*31:0*/ __VactIterCount;
    QData/*63:0*/ __Vfunc_test_tpu_mac_simple__DOT__dut__DOT__fp32_multiply__4__temp_result;
    VlUnpacked<CData/*0:0*/, 6> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_h3f83c5f4__0;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<3> __VactTriggered;
    VlTriggerVec<3> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtest_tpu_mac_simple__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtest_tpu_mac_simple___024root(Vtest_tpu_mac_simple__Syms* symsp, const char* v__name);
    ~Vtest_tpu_mac_simple___024root();
    VL_UNCOPYABLE(Vtest_tpu_mac_simple___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
