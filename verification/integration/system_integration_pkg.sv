// System Integration Test Package
// Comprehensive system-level integration testing framework

`ifndef SYSTEM_INTEGRATION_PKG_SV
`define SYSTEM_INTEGRATION_PKG_SV

package system_integration_pkg;

    import uvm_pkg::*;
    `include "uvm_macros.svh"
    
    // Integration test types
    typedef enum {
        MULTI_CORE_COORDINATION,
        CACHE_COHERENCY,
        MEMORY_CONSISTENCY,
        NOC_COMMUNICATION,
        POWER_MANAGEMENT,
        THERMAL_MANAGEMENT,
        ERROR_RECOVERY,
        PERFORMANCE_SCALING,
        MIXED_WORKLOAD,
        STRESS_TEST,
        ENDURANCE_TEST,
        SYSTEM_BOOT,
        HARDWARE_SOFTWARE_CODESIGN
    } integration_test_type_e;
    
    // Test complexity levels
    typedef enum {
        BASIC_INTEGRATION,
        INTERMEDIATE_INTEGRATION,
        ADVANCED_INTEGRATION,
        COMPREHENSIVE_INTEGRATION
    } test_complexity_e;
    
    // System components
    typedef enum {
        CPU_CORE_0,
        CPU_CORE_1,
        CPU_CORE_2,
        CPU_CORE_3,
        TPU_UNIT_0,
        TPU_UNIT_1,
        VPU_UNIT_0,
        VPU_UNIT_1,
        L1_ICACHE,
        L1_DCACHE,
        L2_CACHE,
        L3_CACHE,
        MEMORY_CONTROLLER,
        NOC_ROUTER,
        POWER_MANAGER,
        THERMAL_CONTROLLER
    } system_component_e;
    
    // Integration test configuration
    typedef struct {
        integration_test_type_e test_type;
        test_complexity_e complexity;
        system_component_e active_components[];
        int num_cores_active;
        int num_tpus_active;
        int num_vpus_active;
        bit enable_cache_coherency;
        bit enable_power_management;
        bit enable_thermal_management;
        bit enable_error_injection;
        real test_duration_ms;
        int max_concurrent_transactions;
        string workload_pattern;
    } integration_test_config_t;
    
    // System state monitoring
    typedef struct {
        real cpu_utilization[4];
        real tpu_utilization[2];
        real vpu_utilization[2];
        real cache_hit_rates[4];  // L1I, L1D, L2, L3
        real memory_bandwidth_utilization;
        real noc_bandwidth_utilization;
        real power_consumption_watts;
        real temperature_celsius;
        int active_transactions;
        int completed_transactions;
        int error_count;
        time current_time;
    } system_state_t;
    
    // Performance metrics
    typedef struct {
        real overall_throughput;
        real latency_p50;
        real latency_p95;
        real latency_p99;
        real power_efficiency;
        real thermal_efficiency;
        real scalability_factor;
        real reliability_score;
        int total_operations;
        time total_execution_time;
    } integration_performance_t;
    
    // Include integration test classes
    `include "system_integration_base.sv"
    `include "multi_core_test.sv"
    `include "cache_coherency_test.sv"
    `include "noc_communication_test.sv"
    `include "power_thermal_test.sv"
    `include "stress_endurance_test.sv"
    `include "mixed_workload_test.sv"
    `include "system_monitor.sv"
    `include "integration_analyzer.sv"

endpackage : system_integration_pkg

`endif // SYSTEM_INTEGRATION_PKG_SV