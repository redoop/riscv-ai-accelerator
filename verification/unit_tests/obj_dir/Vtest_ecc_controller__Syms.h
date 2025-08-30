// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTEST_ECC_CONTROLLER__SYMS_H_
#define VERILATED_VTEST_ECC_CONTROLLER__SYMS_H_  // guard

#include "verilated.h"
#include "verilated_vcd_c.h"

// INCLUDE MODEL CLASS

#include "Vtest_ecc_controller.h"

// INCLUDE MODULE CLASSES
#include "Vtest_ecc_controller___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vtest_ecc_controller__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtest_ecc_controller* const __Vm_modelp;
    bool __Vm_dumping = false;  // Dumping is active
    VerilatedMutex __Vm_dumperMutex;  // Protect __Vm_dumperp
    VerilatedVcdC* __Vm_dumperp VL_GUARDED_BY(__Vm_dumperMutex) = nullptr;  /// Trace class for $dump*
    bool __Vm_activity = false;  ///< Used by trace routines to determine change occurred
    uint32_t __Vm_baseCode = 0;  ///< Used by trace routines when tracing multiple models
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtest_ecc_controller___024root TOP;

    // CONSTRUCTORS
    Vtest_ecc_controller__Syms(VerilatedContext* contextp, const char* namep, Vtest_ecc_controller* modelp);
    ~Vtest_ecc_controller__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
    void _traceDump();
    void _traceDumpOpen();
    void _traceDumpClose();
};

#endif  // guard
