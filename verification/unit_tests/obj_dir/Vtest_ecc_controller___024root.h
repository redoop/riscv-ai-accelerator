// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtest_ecc_controller.h for the primary calling header

#ifndef VERILATED_VTEST_ECC_CONTROLLER___024ROOT_H_
#define VERILATED_VTEST_ECC_CONTROLLER___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtest_ecc_controller__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtest_ecc_controller___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ test_ecc_controller__DOT__clk;
    CData/*0:0*/ test_ecc_controller__DOT__rst_n;
    CData/*0:0*/ test_ecc_controller__DOT__mem_req;
    CData/*0:0*/ test_ecc_controller__DOT__mem_we;
    CData/*0:0*/ test_ecc_controller__DOT__single_error;
    CData/*0:0*/ test_ecc_controller__DOT__double_error;
    CData/*0:0*/ test_ecc_controller__DOT__error_inject_en;
    CData/*1:0*/ test_ecc_controller__DOT__error_inject_type;
    CData/*0:0*/ test_ecc_controller__DOT__array_ready;
    CData/*7:0*/ test_ecc_controller__DOT__dut__DOT__ecc_encode_out;
    CData/*7:0*/ test_ecc_controller__DOT__dut__DOT__ecc_syndrome;
    CData/*0:0*/ test_ecc_controller__DOT__dut__DOT__single_err_detected;
    CData/*0:0*/ test_ecc_controller__DOT__dut__DOT__double_err_detected;
    CData/*7:0*/ test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__calculated_ecc;
    CData/*0:0*/ test_ecc_controller__DOT__dut__DOT__u_decoder__DOT__overall_parity;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_ecc_controller__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__test_ecc_controller__DOT__rst_n__0;
    CData/*0:0*/ __VactContinue;
    IData/*31:0*/ test_ecc_controller__DOT__mem_addr;
    IData/*31:0*/ test_ecc_controller__DOT__error_addr;
    VlWide<3>/*71:0*/ test_ecc_controller__DOT__array_rdata;
    IData/*31:0*/ test_ecc_controller__DOT__error_count;
    IData/*31:0*/ test_ecc_controller__DOT__test_count;
    IData/*31:0*/ test_ecc_controller__DOT__unnamedblk1__DOT__i;
    IData/*31:0*/ test_ecc_controller__DOT__unnamedblk2__DOT__i;
    IData/*31:0*/ test_ecc_controller__DOT__unnamedblk3__DOT__i;
    VlWide<3>/*71:0*/ test_ecc_controller__DOT__dut__DOT__injected_data;
    IData/*31:0*/ __VactIterCount;
    QData/*63:0*/ test_ecc_controller__DOT__mem_wdata;
    QData/*63:0*/ test_ecc_controller__DOT__test_data;
    QData/*63:0*/ test_ecc_controller__DOT__read_data;
    QData/*63:0*/ test_ecc_controller__DOT__dut__DOT__corrected_data;
    VlUnpacked<VlWide<3>/*71:0*/, 1024> test_ecc_controller__DOT__memory_array;
    VlUnpacked<CData/*0:0*/, 6> __Vm_traceActivity;
    VlDelayScheduler __VdlySched;
    VlTriggerScheduler __VtrigSched_hc35d34f7__0;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<3> __VactTriggered;
    VlTriggerVec<3> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vtest_ecc_controller__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vtest_ecc_controller___024root(Vtest_ecc_controller__Syms* symsp, const char* v__name);
    ~Vtest_ecc_controller___024root();
    VL_UNCOPYABLE(Vtest_ecc_controller___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
