/*
 * JTAG Interface for RISC-V AI Accelerator
 * Provides JTAG debugging capabilities and GDB integration
 */

#ifndef JTAG_INTERFACE_H
#define JTAG_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// JTAG TAP states
typedef enum {
    JTAG_STATE_TEST_LOGIC_RESET = 0,
    JTAG_STATE_RUN_TEST_IDLE,
    JTAG_STATE_SELECT_DR_SCAN,
    JTAG_STATE_CAPTURE_DR,
    JTAG_STATE_SHIFT_DR,
    JTAG_STATE_EXIT1_DR,
    JTAG_STATE_PAUSE_DR,
    JTAG_STATE_EXIT2_DR,
    JTAG_STATE_UPDATE_DR,
    JTAG_STATE_SELECT_IR_SCAN,
    JTAG_STATE_CAPTURE_IR,
    JTAG_STATE_SHIFT_IR,
    JTAG_STATE_EXIT1_IR,
    JTAG_STATE_PAUSE_IR,
    JTAG_STATE_EXIT2_IR,
    JTAG_STATE_UPDATE_IR
} jtag_state_t;

// JTAG instruction register values
#define JTAG_IR_BYPASS      0x00
#define JTAG_IR_IDCODE      0x01
#define JTAG_IR_DEBUG       0x02
#define JTAG_IR_DTMCONTROL  0x10
#define JTAG_IR_DBUS        0x11

// Debug module registers
#define DMI_DMCONTROL       0x10
#define DMI_DMSTATUS        0x11
#define DMI_HARTINFO        0x12
#define DMI_ABSTRACTCS      0x16
#define DMI_COMMAND         0x17
#define DMI_ABSTRACTAUTO    0x18
#define DMI_DATA0           0x04
#define DMI_PROGBUF0        0x20

// JTAG interface structure
typedef struct {
    int tck_pin;        // Test Clock
    int tms_pin;        // Test Mode Select
    int tdi_pin;        // Test Data In
    int tdo_pin;        // Test Data Out
    int trst_pin;       // Test Reset (optional)
    jtag_state_t state; // Current TAP state
    bool initialized;
} jtag_interface_t;

// Debug target information
typedef struct {
    uint32_t hart_count;
    uint32_t selected_hart;
    uint32_t idcode;
    bool debug_module_present;
    bool abstract_commands_supported;
} debug_target_t;

// Function prototypes

// JTAG interface management
int jtag_init(jtag_interface_t* jtag, int tck, int tms, int tdi, int tdo, int trst);
void jtag_cleanup(jtag_interface_t* jtag);
int jtag_reset(jtag_interface_t* jtag);

// JTAG TAP state machine
int jtag_goto_state(jtag_interface_t* jtag, jtag_state_t target_state);
jtag_state_t jtag_get_state(jtag_interface_t* jtag);

// JTAG data transfer
int jtag_shift_ir(jtag_interface_t* jtag, uint32_t instruction, int ir_length);
int jtag_shift_dr(jtag_interface_t* jtag, uint64_t data_in, uint64_t* data_out, int dr_length);

// Debug module interface
int debug_init(jtag_interface_t* jtag, debug_target_t* target);
int debug_halt_hart(jtag_interface_t* jtag, uint32_t hart_id);
int debug_resume_hart(jtag_interface_t* jtag, uint32_t hart_id);
int debug_step_hart(jtag_interface_t* jtag, uint32_t hart_id);

// Register access
int debug_read_register(jtag_interface_t* jtag, uint32_t reg_num, uint64_t* value);
int debug_write_register(jtag_interface_t* jtag, uint32_t reg_num, uint64_t value);
int debug_read_csr(jtag_interface_t* jtag, uint32_t csr_addr, uint64_t* value);
int debug_write_csr(jtag_interface_t* jtag, uint32_t csr_addr, uint64_t value);

// Memory access
int debug_read_memory(jtag_interface_t* jtag, uint64_t address, uint8_t* data, size_t length);
int debug_write_memory(jtag_interface_t* jtag, uint64_t address, const uint8_t* data, size_t length);

// Breakpoint management
int debug_set_breakpoint(jtag_interface_t* jtag, uint64_t address, uint32_t* bp_id);
int debug_clear_breakpoint(jtag_interface_t* jtag, uint32_t bp_id);
int debug_list_breakpoints(jtag_interface_t* jtag, uint32_t* bp_list, size_t max_count);

// Watchpoint management
int debug_set_watchpoint(jtag_interface_t* jtag, uint64_t address, size_t size, 
                        bool read, bool write, uint32_t* wp_id);
int debug_clear_watchpoint(jtag_interface_t* jtag, uint32_t wp_id);

// AI accelerator specific debugging
int debug_read_tpu_state(jtag_interface_t* jtag, uint32_t tpu_id, uint32_t* state);
int debug_read_vpu_registers(jtag_interface_t* jtag, uint32_t vpu_id, uint64_t* regs, size_t count);
int debug_dump_cache_state(jtag_interface_t* jtag, uint32_t cache_level, void* dump_buffer);

// Utility functions
const char* jtag_state_name(jtag_state_t state);
int jtag_scan_chain(jtag_interface_t* jtag, uint32_t* device_count, uint32_t* idcodes);

// Error codes
#define JTAG_SUCCESS            0
#define JTAG_ERROR_INIT         -1
#define JTAG_ERROR_INVALID_ARG  -2
#define JTAG_ERROR_TIMEOUT      -3
#define JTAG_ERROR_PROTOCOL     -4
#define JTAG_ERROR_NOT_HALTED   -5
#define JTAG_ERROR_NO_DEBUG_MODULE -6

#ifdef __cplusplus
}
#endif

#endif // JTAG_INTERFACE_H