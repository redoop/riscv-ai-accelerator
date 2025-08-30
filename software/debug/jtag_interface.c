/*
 * JTAG Interface Implementation for RISC-V AI Accelerator
 */

#include "jtag_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>

// GPIO access for JTAG pins (platform-specific)
#ifdef __linux__
#include <linux/gpio.h>
#endif

// JTAG timing parameters (microseconds)
#define JTAG_TCK_PERIOD     1000    // 1MHz clock
#define JTAG_SETUP_TIME     100     // Setup time
#define JTAG_HOLD_TIME      100     // Hold time

// Debug module timeout (milliseconds)
#define DEBUG_TIMEOUT_MS    1000

// Static functions
static void jtag_clock_pulse(jtag_interface_t* jtag);
static void jtag_set_tms(jtag_interface_t* jtag, bool value);
static void jtag_set_tdi(jtag_interface_t* jtag, bool value);
static bool jtag_get_tdo(jtag_interface_t* jtag);
static int dmi_read(jtag_interface_t* jtag, uint32_t address, uint32_t* data);
static int dmi_write(jtag_interface_t* jtag, uint32_t address, uint32_t data);

// JTAG state transition table
static const jtag_state_t state_transitions[16][2] = {
    // Current state                    TMS=0                    TMS=1
    {JTAG_STATE_RUN_TEST_IDLE,         JTAG_STATE_TEST_LOGIC_RESET}, // TEST_LOGIC_RESET
    {JTAG_STATE_RUN_TEST_IDLE,         JTAG_STATE_SELECT_DR_SCAN},   // RUN_TEST_IDLE
    {JTAG_STATE_CAPTURE_DR,            JTAG_STATE_SELECT_IR_SCAN},   // SELECT_DR_SCAN
    {JTAG_STATE_SHIFT_DR,              JTAG_STATE_EXIT1_DR},         // CAPTURE_DR
    {JTAG_STATE_SHIFT_DR,              JTAG_STATE_EXIT1_DR},         // SHIFT_DR
    {JTAG_STATE_PAUSE_DR,              JTAG_STATE_UPDATE_DR},        // EXIT1_DR
    {JTAG_STATE_PAUSE_DR,              JTAG_STATE_EXIT2_DR},         // PAUSE_DR
    {JTAG_STATE_SHIFT_DR,              JTAG_STATE_UPDATE_DR},        // EXIT2_DR
    {JTAG_STATE_RUN_TEST_IDLE,         JTAG_STATE_SELECT_DR_SCAN},   // UPDATE_DR
    {JTAG_STATE_CAPTURE_IR,            JTAG_STATE_TEST_LOGIC_RESET}, // SELECT_IR_SCAN
    {JTAG_STATE_SHIFT_IR,              JTAG_STATE_EXIT1_IR},         // CAPTURE_IR
    {JTAG_STATE_SHIFT_IR,              JTAG_STATE_EXIT1_IR},         // SHIFT_IR
    {JTAG_STATE_PAUSE_IR,              JTAG_STATE_UPDATE_IR},        // EXIT1_IR
    {JTAG_STATE_PAUSE_IR,              JTAG_STATE_EXIT2_IR},         // PAUSE_IR
    {JTAG_STATE_SHIFT_IR,              JTAG_STATE_UPDATE_IR},        // EXIT2_IR
    {JTAG_STATE_RUN_TEST_IDLE,         JTAG_STATE_SELECT_DR_SCAN}    // UPDATE_IR
};

int jtag_init(jtag_interface_t* jtag, int tck, int tms, int tdi, int tdo, int trst) {
    if (!jtag) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    memset(jtag, 0, sizeof(jtag_interface_t));
    
    jtag->tck_pin = tck;
    jtag->tms_pin = tms;
    jtag->tdi_pin = tdi;
    jtag->tdo_pin = tdo;
    jtag->trst_pin = trst;
    jtag->state = JTAG_STATE_TEST_LOGIC_RESET;
    
    // Initialize GPIO pins (platform-specific implementation)
    // This is a simplified version - real implementation would configure GPIO
    
    jtag->initialized = true;
    
    // Reset JTAG TAP
    return jtag_reset(jtag);
}

void jtag_cleanup(jtag_interface_t* jtag) {
    if (jtag && jtag->initialized) {
        // Cleanup GPIO resources
        jtag->initialized = false;
    }
}

int jtag_reset(jtag_interface_t* jtag) {
    if (!jtag || !jtag->initialized) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Assert TRST if available
    if (jtag->trst_pin >= 0) {
        // Set TRST low, wait, then high
        usleep(1000);
    }
    
    // Send 5 TMS=1 to ensure we're in TEST_LOGIC_RESET
    for (int i = 0; i < 5; i++) {
        jtag_set_tms(jtag, true);
        jtag_clock_pulse(jtag);
    }
    
    jtag->state = JTAG_STATE_TEST_LOGIC_RESET;
    
    // Go to RUN_TEST_IDLE
    return jtag_goto_state(jtag, JTAG_STATE_RUN_TEST_IDLE);
}

int jtag_goto_state(jtag_interface_t* jtag, jtag_state_t target_state) {
    if (!jtag || !jtag->initialized) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Simple path finding - could be optimized
    while (jtag->state != target_state) {
        bool tms_value;
        
        // Determine TMS value to get closer to target
        if (target_state == JTAG_STATE_TEST_LOGIC_RESET) {
            tms_value = true;
        } else if (jtag->state == JTAG_STATE_TEST_LOGIC_RESET) {
            tms_value = false;
        } else {
            // Use state transition table
            jtag_state_t next_0 = state_transitions[jtag->state][0];
            jtag_state_t next_1 = state_transitions[jtag->state][1];
            
            // Simple heuristic: prefer TMS=0 path unless TMS=1 gets us there directly
            tms_value = (next_1 == target_state) ? true : false;
        }
        
        jtag_set_tms(jtag, tms_value);
        jtag_clock_pulse(jtag);
        jtag->state = state_transitions[jtag->state][tms_value ? 1 : 0];
    }
    
    return JTAG_SUCCESS;
}

jtag_state_t jtag_get_state(jtag_interface_t* jtag) {
    return jtag ? jtag->state : JTAG_STATE_TEST_LOGIC_RESET;
}

int jtag_shift_ir(jtag_interface_t* jtag, uint32_t instruction, int ir_length) {
    if (!jtag || !jtag->initialized || ir_length <= 0) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Go to SHIFT_IR state
    int ret = jtag_goto_state(jtag, JTAG_STATE_SHIFT_IR);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // Shift instruction bits
    for (int i = 0; i < ir_length; i++) {
        bool bit = (instruction >> i) & 1;
        bool is_last = (i == ir_length - 1);
        
        jtag_set_tdi(jtag, bit);
        jtag_set_tms(jtag, is_last); // TMS=1 on last bit to exit
        jtag_clock_pulse(jtag);
        
        if (is_last) {
            jtag->state = JTAG_STATE_EXIT1_IR;
        }
    }
    
    // Go to UPDATE_IR
    return jtag_goto_state(jtag, JTAG_STATE_UPDATE_IR);
}

int jtag_shift_dr(jtag_interface_t* jtag, uint64_t data_in, uint64_t* data_out, int dr_length) {
    if (!jtag || !jtag->initialized || dr_length <= 0) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Go to SHIFT_DR state
    int ret = jtag_goto_state(jtag, JTAG_STATE_SHIFT_DR);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    uint64_t result = 0;
    
    // Shift data bits
    for (int i = 0; i < dr_length; i++) {
        bool bit_in = (data_in >> i) & 1;
        bool is_last = (i == dr_length - 1);
        
        jtag_set_tdi(jtag, bit_in);
        jtag_set_tms(jtag, is_last); // TMS=1 on last bit to exit
        
        bool bit_out = jtag_get_tdo(jtag);
        jtag_clock_pulse(jtag);
        
        if (bit_out) {
            result |= (1ULL << i);
        }
        
        if (is_last) {
            jtag->state = JTAG_STATE_EXIT1_DR;
        }
    }
    
    if (data_out) {
        *data_out = result;
    }
    
    // Go to UPDATE_DR
    return jtag_goto_state(jtag, JTAG_STATE_UPDATE_DR);
}

int debug_init(jtag_interface_t* jtag, debug_target_t* target) {
    if (!jtag || !target) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Read IDCODE
    int ret = jtag_shift_ir(jtag, JTAG_IR_IDCODE, 5);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    uint64_t idcode;
    ret = jtag_shift_dr(jtag, 0, &idcode, 32);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    target->idcode = (uint32_t)idcode;
    
    // Check for debug module
    ret = jtag_shift_ir(jtag, JTAG_IR_DTMCONTROL, 5);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    uint64_t dtmcontrol;
    ret = jtag_shift_dr(jtag, 0, &dtmcontrol, 32);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    target->debug_module_present = (dtmcontrol & 0x1) != 0;
    
    if (target->debug_module_present) {
        // Read DMSTATUS to get hart count
        uint32_t dmstatus;
        ret = dmi_read(jtag, DMI_DMSTATUS, &dmstatus);
        if (ret == JTAG_SUCCESS) {
            target->hart_count = 1; // Simplified - real implementation would parse DMSTATUS
            target->selected_hart = 0;
            target->abstract_commands_supported = true;
        }
    }
    
    return JTAG_SUCCESS;
}

int debug_halt_hart(jtag_interface_t* jtag, uint32_t hart_id) {
    // Set DMCONTROL.haltreq for specified hart
    uint32_t dmcontrol = (1 << 31) | (hart_id << 16) | (1 << 0); // haltreq | hartsel | dmactive
    return dmi_write(jtag, DMI_DMCONTROL, dmcontrol);
}

int debug_resume_hart(jtag_interface_t* jtag, uint32_t hart_id) {
    // Set DMCONTROL.resumereq for specified hart
    uint32_t dmcontrol = (1 << 30) | (hart_id << 16) | (1 << 0); // resumereq | hartsel | dmactive
    return dmi_write(jtag, DMI_DMCONTROL, dmcontrol);
}

int debug_read_register(jtag_interface_t* jtag, uint32_t reg_num, uint64_t* value) {
    if (!value) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Use abstract command to read register
    uint32_t command = (2 << 24) | (1 << 17) | reg_num; // regno | transfer | cmdtype=0 (register)
    
    int ret = dmi_write(jtag, DMI_COMMAND, command);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // Read result from DATA0
    uint32_t data_low, data_high = 0;
    ret = dmi_read(jtag, DMI_DATA0, &data_low);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // For 64-bit registers, read DATA1 as well
    if (reg_num >= 32) { // Assume 64-bit for CSRs and some registers
        ret = dmi_read(jtag, DMI_DATA0 + 1, &data_high);
        if (ret != JTAG_SUCCESS) {
            return ret;
        }
    }
    
    *value = ((uint64_t)data_high << 32) | data_low;
    return JTAG_SUCCESS;
}

int debug_write_register(jtag_interface_t* jtag, uint32_t reg_num, uint64_t value) {
    // Write value to DATA0/DATA1
    int ret = dmi_write(jtag, DMI_DATA0, (uint32_t)value);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    if (reg_num >= 32) { // 64-bit register
        ret = dmi_write(jtag, DMI_DATA0 + 1, (uint32_t)(value >> 32));
        if (ret != JTAG_SUCCESS) {
            return ret;
        }
    }
    
    // Use abstract command to write register
    uint32_t command = (2 << 24) | (1 << 18) | (1 << 17) | reg_num; // write | transfer | cmdtype=0
    return dmi_write(jtag, DMI_COMMAND, command);
}

int debug_read_memory(jtag_interface_t* jtag, uint64_t address, uint8_t* data, size_t length) {
    if (!data || length == 0) {
        return JTAG_ERROR_INVALID_ARG;
    }
    
    // Use abstract command for memory access
    // This is simplified - real implementation would handle different access sizes
    for (size_t i = 0; i < length; i += 4) {
        uint32_t word_addr = (uint32_t)(address + i);
        
        // Write address to DATA1
        int ret = dmi_write(jtag, DMI_DATA0 + 1, word_addr);
        if (ret != JTAG_SUCCESS) {
            return ret;
        }
        
        // Memory read command
        uint32_t command = (2 << 24) | (1 << 17) | (1 << 16); // transfer | postexec | cmdtype=2 (memory)
        ret = dmi_write(jtag, DMI_COMMAND, command);
        if (ret != JTAG_SUCCESS) {
            return ret;
        }
        
        // Read result
        uint32_t word_data;
        ret = dmi_read(jtag, DMI_DATA0, &word_data);
        if (ret != JTAG_SUCCESS) {
            return ret;
        }
        
        // Copy to output buffer
        size_t copy_len = (length - i < 4) ? (length - i) : 4;
        memcpy(data + i, &word_data, copy_len);
    }
    
    return JTAG_SUCCESS;
}

// Static helper functions
static void jtag_clock_pulse(jtag_interface_t* jtag) {
    // Set TCK low
    usleep(JTAG_TCK_PERIOD / 2);
    
    // Set TCK high
    usleep(JTAG_TCK_PERIOD / 2);
}

static void jtag_set_tms(jtag_interface_t* jtag, bool value) {
    // Platform-specific GPIO write
    (void)jtag; (void)value; // Suppress warnings
}

static void jtag_set_tdi(jtag_interface_t* jtag, bool value) {
    // Platform-specific GPIO write
    (void)jtag; (void)value; // Suppress warnings
}

static bool jtag_get_tdo(jtag_interface_t* jtag) {
    // Platform-specific GPIO read
    (void)jtag; // Suppress warning
    return false; // Placeholder
}

static int dmi_read(jtag_interface_t* jtag, uint32_t address, uint32_t* data) {
    // Select DBUS instruction
    int ret = jtag_shift_ir(jtag, JTAG_IR_DBUS, 5);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // DMI read operation: op=1 (read), address, data=0
    uint64_t dmi_request = (1ULL << 34) | ((uint64_t)address << 2);
    uint64_t dmi_response;
    
    ret = jtag_shift_dr(jtag, dmi_request, &dmi_response, 41);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // Check operation status
    uint32_t op_status = dmi_response & 0x3;
    if (op_status != 0) {
        return JTAG_ERROR_PROTOCOL;
    }
    
    if (data) {
        *data = (uint32_t)(dmi_response >> 2);
    }
    
    return JTAG_SUCCESS;
}

static int dmi_write(jtag_interface_t* jtag, uint32_t address, uint32_t data) {
    // Select DBUS instruction
    int ret = jtag_shift_ir(jtag, JTAG_IR_DBUS, 5);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // DMI write operation: op=2 (write), address, data
    uint64_t dmi_request = (2ULL << 34) | ((uint64_t)address << 2) | ((uint64_t)data << 2);
    uint64_t dmi_response;
    
    ret = jtag_shift_dr(jtag, dmi_request, &dmi_response, 41);
    if (ret != JTAG_SUCCESS) {
        return ret;
    }
    
    // Check operation status
    uint32_t op_status = dmi_response & 0x3;
    if (op_status != 0) {
        return JTAG_ERROR_PROTOCOL;
    }
    
    return JTAG_SUCCESS;
}

const char* jtag_state_name(jtag_state_t state) {
    static const char* state_names[] = {
        "TEST_LOGIC_RESET", "RUN_TEST_IDLE", "SELECT_DR_SCAN", "CAPTURE_DR",
        "SHIFT_DR", "EXIT1_DR", "PAUSE_DR", "EXIT2_DR", "UPDATE_DR",
        "SELECT_IR_SCAN", "CAPTURE_IR", "SHIFT_IR", "EXIT1_IR", "PAUSE_IR",
        "EXIT2_IR", "UPDATE_IR"
    };
    
    if (state < sizeof(state_names) / sizeof(state_names[0])) {
        return state_names[state];
    }
    return "UNKNOWN";
}