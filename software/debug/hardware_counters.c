/*
 * Hardware Event Counters for RISC-V AI Accelerator
 * Implementation of performance monitoring using hardware counters
 */

#include "performance_analyzer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>

// Memory-mapped register base addresses (platform-specific)
#define PERF_COUNTER_BASE       0x10000000
#define AI_ACCELERATOR_BASE     0x20000000
#define NOC_MONITOR_BASE        0x30000000
#define POWER_MONITOR_BASE      0x40000000

// Register offsets
#define PERF_CTRL_REG           0x000
#define PERF_STATUS_REG         0x004
#define PERF_COUNTER_REG(n)     (0x100 + (n) * 8)
#define PERF_CONFIG_REG(n)      (0x200 + (n) * 4)

// AI accelerator performance registers
#define TPU_PERF_CTRL           0x000
#define TPU_CYCLE_COUNT         0x010
#define TPU_OP_COUNT            0x018
#define TPU_CACHE_HITS          0x020
#define TPU_CACHE_MISSES        0x028
#define VPU_PERF_CTRL           0x100
#define VPU_CYCLE_COUNT         0x110
#define VPU_OP_COUNT            0x118

// NoC performance registers
#define NOC_PACKET_COUNT        0x000
#define NOC_STALL_CYCLES        0x008
#define NOC_BANDWIDTH_UTIL      0x010

// Power monitoring registers
#define POWER_CURRENT           0x000
#define POWER_VOLTAGE           0x004
#define POWER_ENERGY            0x008

// Global state
static void* perf_base_addr = NULL;
static void* ai_base_addr = NULL;
static void* noc_base_addr = NULL;
static void* power_base_addr = NULL;
static int mem_fd = -1;
static bool counters_initialized = false;

// Counter definitions
static const struct {
    uint32_t id;
    const char* name;
    const char* description;
    uint32_t reg_offset;
    bool is_ai_counter;
} counter_definitions[] = {
    {PERF_COUNTER_CYCLES, "cycles", "CPU cycles", PERF_COUNTER_REG(0), false},
    {PERF_COUNTER_INSTRUCTIONS, "instructions", "Instructions executed", PERF_COUNTER_REG(1), false},
    {PERF_COUNTER_CACHE_MISSES, "cache_misses", "Cache misses", PERF_COUNTER_REG(2), false},
    {PERF_COUNTER_CACHE_HITS, "cache_hits", "Cache hits", PERF_COUNTER_REG(3), false},
    {PERF_COUNTER_BRANCH_MISSES, "branch_misses", "Branch mispredictions", PERF_COUNTER_REG(4), false},
    {PERF_COUNTER_BRANCH_TAKEN, "branch_taken", "Branches taken", PERF_COUNTER_REG(5), false},
    {PERF_COUNTER_TLB_MISSES, "tlb_misses", "TLB misses", PERF_COUNTER_REG(6), false},
    {PERF_COUNTER_MEMORY_ACCESSES, "memory_accesses", "Memory accesses", PERF_COUNTER_REG(7), false},
    {PERF_COUNTER_TPU_CYCLES, "tpu_cycles", "TPU cycles", TPU_CYCLE_COUNT, true},
    {PERF_COUNTER_TPU_OPERATIONS, "tpu_operations", "TPU operations", TPU_OP_COUNT, true},
    {PERF_COUNTER_TPU_CACHE_MISSES, "tpu_cache_misses", "TPU cache misses", TPU_CACHE_MISSES, true},
    {PERF_COUNTER_VPU_CYCLES, "vpu_cycles", "VPU cycles", VPU_CYCLE_COUNT, true},
    {PERF_COUNTER_VPU_OPERATIONS, "vpu_operations", "VPU operations", VPU_OP_COUNT, true},
    {PERF_COUNTER_NOC_PACKETS, "noc_packets", "NoC packets", NOC_PACKET_COUNT, true},
    {PERF_COUNTER_NOC_STALLS, "noc_stalls", "NoC stall cycles", NOC_STALL_CYCLES, true},
};

#define NUM_COUNTER_DEFINITIONS (sizeof(counter_definitions) / sizeof(counter_definitions[0]))

// Helper functions
static uint64_t read_reg64(void* base, uint32_t offset);
static void write_reg64(void* base, uint32_t offset, uint64_t value);
static uint32_t read_reg32(void* base, uint32_t offset);
static void write_reg32(void* base, uint32_t offset, uint32_t value);
static int find_counter_definition(uint32_t counter_id);

int perf_init_counters(void) {
    if (counters_initialized) {
        return PERF_SUCCESS;
    }
    
    // Open /dev/mem for memory-mapped register access
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        fprintf(stderr, "Failed to open /dev/mem: %s\n", strerror(errno));
        return PERF_ERROR_PERMISSION;
    }
    
    // Map performance counter registers
    perf_base_addr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED,
                         mem_fd, PERF_COUNTER_BASE);
    if (perf_base_addr == MAP_FAILED) {
        fprintf(stderr, "Failed to map performance counter registers: %s\n", strerror(errno));
        close(mem_fd);
        return PERF_ERROR_INIT;
    }
    
    // Map AI accelerator registers
    ai_base_addr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED,
                       mem_fd, AI_ACCELERATOR_BASE);
    if (ai_base_addr == MAP_FAILED) {
        fprintf(stderr, "Failed to map AI accelerator registers: %s\n", strerror(errno));
        munmap(perf_base_addr, 0x1000);
        close(mem_fd);
        return PERF_ERROR_INIT;
    }
    
    // Map NoC monitor registers
    noc_base_addr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED,
                        mem_fd, NOC_MONITOR_BASE);
    if (noc_base_addr == MAP_FAILED) {
        fprintf(stderr, "Failed to map NoC monitor registers: %s\n", strerror(errno));
        munmap(perf_base_addr, 0x1000);
        munmap(ai_base_addr, 0x1000);
        close(mem_fd);
        return PERF_ERROR_INIT;
    }
    
    // Map power monitor registers
    power_base_addr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED,
                          mem_fd, POWER_MONITOR_BASE);
    if (power_base_addr == MAP_FAILED) {
        fprintf(stderr, "Failed to map power monitor registers: %s\n", strerror(errno));
        munmap(perf_base_addr, 0x1000);
        munmap(ai_base_addr, 0x1000);
        munmap(noc_base_addr, 0x1000);
        close(mem_fd);
        return PERF_ERROR_INIT;
    }
    
    // Initialize performance counters
    write_reg32(perf_base_addr, PERF_CTRL_REG, 0x1); // Enable performance monitoring
    
    // Initialize AI accelerator performance monitoring
    write_reg32(ai_base_addr, TPU_PERF_CTRL, 0x1);   // Enable TPU monitoring
    write_reg32(ai_base_addr, VPU_PERF_CTRL, 0x1);   // Enable VPU monitoring
    
    counters_initialized = true;
    return PERF_SUCCESS;
}

int perf_enable_counter(uint32_t counter_id) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    int def_idx = find_counter_definition(counter_id);
    if (def_idx < 0) {
        return PERF_ERROR_INVALID_COUNTER;
    }
    
    // Enable the specific counter
    if (counter_definitions[def_idx].is_ai_counter) {
        // AI accelerator counters are always enabled when the module is active
        return PERF_SUCCESS;
    } else {
        // Enable CPU performance counter
        uint32_t config = read_reg32(perf_base_addr, PERF_CONFIG_REG(counter_id));
        config |= 0x1; // Enable bit
        write_reg32(perf_base_addr, PERF_CONFIG_REG(counter_id), config);
    }
    
    return PERF_SUCCESS;
}

int perf_disable_counter(uint32_t counter_id) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    int def_idx = find_counter_definition(counter_id);
    if (def_idx < 0) {
        return PERF_ERROR_INVALID_COUNTER;
    }
    
    if (!counter_definitions[def_idx].is_ai_counter) {
        // Disable CPU performance counter
        uint32_t config = read_reg32(perf_base_addr, PERF_CONFIG_REG(counter_id));
        config &= ~0x1; // Clear enable bit
        write_reg32(perf_base_addr, PERF_CONFIG_REG(counter_id), config);
    }
    
    return PERF_SUCCESS;
}

int perf_read_counter(uint32_t counter_id, uint64_t* value) {
    if (!counters_initialized || !value) {
        return PERF_ERROR_INIT;
    }
    
    int def_idx = find_counter_definition(counter_id);
    if (def_idx < 0) {
        return PERF_ERROR_INVALID_COUNTER;
    }
    
    void* base_addr;
    if (counter_definitions[def_idx].is_ai_counter) {
        if (counter_id >= PERF_COUNTER_NOC_PACKETS) {
            base_addr = noc_base_addr;
        } else {
            base_addr = ai_base_addr;
        }
    } else {
        base_addr = perf_base_addr;
    }
    
    *value = read_reg64(base_addr, counter_definitions[def_idx].reg_offset);
    return PERF_SUCCESS;
}

int perf_reset_counter(uint32_t counter_id) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    int def_idx = find_counter_definition(counter_id);
    if (def_idx < 0) {
        return PERF_ERROR_INVALID_COUNTER;
    }
    
    void* base_addr;
    if (counter_definitions[def_idx].is_ai_counter) {
        if (counter_id >= PERF_COUNTER_NOC_PACKETS) {
            base_addr = noc_base_addr;
        } else {
            base_addr = ai_base_addr;
        }
    } else {
        base_addr = perf_base_addr;
    }
    
    write_reg64(base_addr, counter_definitions[def_idx].reg_offset, 0);
    return PERF_SUCCESS;
}

int perf_reset_all_counters(void) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    // Reset all CPU performance counters
    for (int i = 0; i < 8; i++) {
        write_reg64(perf_base_addr, PERF_COUNTER_REG(i), 0);
    }
    
    // Reset AI accelerator counters
    write_reg64(ai_base_addr, TPU_CYCLE_COUNT, 0);
    write_reg64(ai_base_addr, TPU_OP_COUNT, 0);
    write_reg64(ai_base_addr, TPU_CACHE_HITS, 0);
    write_reg64(ai_base_addr, TPU_CACHE_MISSES, 0);
    write_reg64(ai_base_addr, VPU_CYCLE_COUNT, 0);
    write_reg64(ai_base_addr, VPU_OP_COUNT, 0);
    
    // Reset NoC counters
    write_reg64(noc_base_addr, NOC_PACKET_COUNT, 0);
    write_reg64(noc_base_addr, NOC_STALL_CYCLES, 0);
    
    return PERF_SUCCESS;
}

int perf_monitor_tpu_utilization(uint32_t tpu_id, double* utilization) {
    if (!counters_initialized || !utilization) {
        return PERF_ERROR_INIT;
    }
    
    // Read TPU cycle count and total cycles
    uint64_t tpu_cycles = read_reg64(ai_base_addr, TPU_CYCLE_COUNT);
    uint64_t total_cycles = read_reg64(perf_base_addr, PERF_COUNTER_REG(0));
    
    if (total_cycles > 0) {
        *utilization = (double)tpu_cycles / (double)total_cycles * 100.0;
    } else {
        *utilization = 0.0;
    }
    
    return PERF_SUCCESS;
}

int perf_monitor_vpu_utilization(uint32_t vpu_id, double* utilization) {
    if (!counters_initialized || !utilization) {
        return PERF_ERROR_INIT;
    }
    
    // Read VPU cycle count and total cycles
    uint64_t vpu_cycles = read_reg64(ai_base_addr, VPU_CYCLE_COUNT);
    uint64_t total_cycles = read_reg64(perf_base_addr, PERF_COUNTER_REG(0));
    
    if (total_cycles > 0) {
        *utilization = (double)vpu_cycles / (double)total_cycles * 100.0;
    } else {
        *utilization = 0.0;
    }
    
    return PERF_SUCCESS;
}

int perf_monitor_noc_traffic(uint64_t* packets_sent, uint64_t* packets_received) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    if (packets_sent) {
        *packets_sent = read_reg64(noc_base_addr, NOC_PACKET_COUNT);
    }
    
    if (packets_received) {
        // For simplicity, assume sent == received in this implementation
        *packets_received = read_reg64(noc_base_addr, NOC_PACKET_COUNT);
    }
    
    return PERF_SUCCESS;
}

int perf_monitor_power_consumption(double* current_power, double* average_power) {
    if (!counters_initialized) {
        return PERF_ERROR_INIT;
    }
    
    if (current_power) {
        // Read current and voltage to calculate power
        uint32_t current_ma = read_reg32(power_base_addr, POWER_CURRENT);
        uint32_t voltage_mv = read_reg32(power_base_addr, POWER_VOLTAGE);
        *current_power = (double)(current_ma * voltage_mv) / 1000000.0; // Convert to watts
    }
    
    if (average_power) {
        // Read accumulated energy and calculate average
        uint64_t energy_uj = read_reg64(power_base_addr, POWER_ENERGY);
        uint64_t time_cycles = read_reg64(perf_base_addr, PERF_COUNTER_REG(0));
        
        // Assume 1GHz clock for time calculation
        double time_seconds = (double)time_cycles / 1000000000.0;
        if (time_seconds > 0) {
            *average_power = (double)energy_uj / 1000000.0 / time_seconds; // Convert to watts
        } else {
            *average_power = 0.0;
        }
    }
    
    return PERF_SUCCESS;
}

const char* perf_counter_name(uint32_t counter_id) {
    int def_idx = find_counter_definition(counter_id);
    if (def_idx >= 0) {
        return counter_definitions[def_idx].name;
    }
    return "unknown";
}

const char* perf_counter_description(uint32_t counter_id) {
    int def_idx = find_counter_definition(counter_id);
    if (def_idx >= 0) {
        return counter_definitions[def_idx].description;
    }
    return "Unknown counter";
}

double perf_calculate_ipc(uint64_t instructions, uint64_t cycles) {
    if (cycles > 0) {
        return (double)instructions / (double)cycles;
    }
    return 0.0;
}

double perf_calculate_cache_hit_rate(uint64_t hits, uint64_t misses) {
    uint64_t total = hits + misses;
    if (total > 0) {
        return (double)hits / (double)total * 100.0;
    }
    return 0.0;
}

uint64_t perf_cycles_to_nanoseconds(uint64_t cycles, uint64_t frequency) {
    if (frequency > 0) {
        return (cycles * 1000000000ULL) / frequency;
    }
    return 0;
}

double perf_calculate_bandwidth(uint64_t bytes, uint64_t time_ns) {
    if (time_ns > 0) {
        return (double)bytes * 1000000000.0 / (double)time_ns; // bytes per second
    }
    return 0.0;
}

// Helper functions
static uint64_t read_reg64(void* base, uint32_t offset) {
    volatile uint64_t* reg = (volatile uint64_t*)((char*)base + offset);
    return *reg;
}

static void write_reg64(void* base, uint32_t offset, uint64_t value) {
    volatile uint64_t* reg = (volatile uint64_t*)((char*)base + offset);
    *reg = value;
}

static uint32_t read_reg32(void* base, uint32_t offset) {
    volatile uint32_t* reg = (volatile uint32_t*)((char*)base + offset);
    return *reg;
}

static void write_reg32(void* base, uint32_t offset, uint32_t value) {
    volatile uint32_t* reg = (volatile uint32_t*)((char*)base + offset);
    *reg = value;
}

static int find_counter_definition(uint32_t counter_id) {
    for (int i = 0; i < NUM_COUNTER_DEFINITIONS; i++) {
        if (counter_definitions[i].id == counter_id) {
            return i;
        }
    }
    return -1;
}