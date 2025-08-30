/*
 * Test Suite for RISC-V AI Accelerator Debug Tools
 * Comprehensive tests for JTAG, GDB server, and performance analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "jtag_interface.h"
#include "gdb_server.h"
#include "performance_analyzer.h"

// Test framework macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "FAIL: %s - %s\n", __func__, message); \
            return -1; \
        } \
    } while(0)

#define TEST_PASS() \
    do { \
        printf("PASS: %s\n", __func__); \
        return 0; \
    } while(0)

// Global test state
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

// Test helper functions
static void run_test(int (*test_func)(void), const char* test_name);
static int create_mock_jtag_interface(jtag_interface_t* jtag);
static int create_test_gdb_client(int port);

// JTAG interface tests
int test_jtag_initialization(void) {
    jtag_interface_t jtag;
    
    // Test successful initialization
    int result = jtag_init(&jtag, 1, 2, 3, 4, 5);
    TEST_ASSERT(result == JTAG_SUCCESS, "JTAG initialization should succeed");
    TEST_ASSERT(jtag.initialized == true, "JTAG should be marked as initialized");
    TEST_ASSERT(jtag.state == JTAG_STATE_RUN_TEST_IDLE, "JTAG should be in RUN_TEST_IDLE state");
    
    // Test invalid arguments
    result = jtag_init(NULL, 1, 2, 3, 4, 5);
    TEST_ASSERT(result == JTAG_ERROR_INVALID_ARG, "NULL pointer should return error");
    
    jtag_cleanup(&jtag);
    TEST_PASS();
}

int test_jtag_state_machine(void) {
    jtag_interface_t jtag;
    create_mock_jtag_interface(&jtag);
    
    // Test state transitions
    int result = jtag_goto_state(&jtag, JTAG_STATE_SHIFT_DR);
    TEST_ASSERT(result == JTAG_SUCCESS, "State transition should succeed");
    TEST_ASSERT(jtag.state == JTAG_STATE_SHIFT_DR, "Should be in SHIFT_DR state");
    
    result = jtag_goto_state(&jtag, JTAG_STATE_SHIFT_IR);
    TEST_ASSERT(result == JTAG_SUCCESS, "State transition should succeed");
    TEST_ASSERT(jtag.state == JTAG_STATE_SHIFT_IR, "Should be in SHIFT_IR state");
    
    // Test reset
    result = jtag_reset(&jtag);
    TEST_ASSERT(result == JTAG_SUCCESS, "Reset should succeed");
    TEST_ASSERT(jtag.state == JTAG_STATE_RUN_TEST_IDLE, "Should be in RUN_TEST_IDLE after reset");
    
    jtag_cleanup(&jtag);
    TEST_PASS();
}

int test_jtag_data_transfer(void) {
    jtag_interface_t jtag;
    create_mock_jtag_interface(&jtag);
    
    // Test instruction register shift
    int result = jtag_shift_ir(&jtag, JTAG_IR_IDCODE, 5);
    TEST_ASSERT(result == JTAG_SUCCESS, "IR shift should succeed");
    
    // Test data register shift
    uint64_t data_out;
    result = jtag_shift_dr(&jtag, 0x12345678, &data_out, 32);
    TEST_ASSERT(result == JTAG_SUCCESS, "DR shift should succeed");
    
    jtag_cleanup(&jtag);
    TEST_PASS();
}

int test_debug_module_interface(void) {
    jtag_interface_t jtag;
    debug_target_t target;
    
    create_mock_jtag_interface(&jtag);
    
    // Test debug module initialization
    int result = debug_init(&jtag, &target);
    TEST_ASSERT(result == JTAG_SUCCESS, "Debug init should succeed");
    TEST_ASSERT(target.idcode != 0, "IDCODE should be read");
    
    // Test hart control
    result = debug_halt_hart(&jtag, 0);
    TEST_ASSERT(result == JTAG_SUCCESS, "Hart halt should succeed");
    
    result = debug_resume_hart(&jtag, 0);
    TEST_ASSERT(result == JTAG_SUCCESS, "Hart resume should succeed");
    
    // Test register access
    uint64_t reg_value;
    result = debug_read_register(&jtag, 1, &reg_value); // Read x1
    TEST_ASSERT(result == JTAG_SUCCESS, "Register read should succeed");
    
    result = debug_write_register(&jtag, 1, 0xDEADBEEF);
    TEST_ASSERT(result == JTAG_SUCCESS, "Register write should succeed");
    
    jtag_cleanup(&jtag);
    TEST_PASS();
}

int test_gdb_server_initialization(void) {
    gdb_server_t server;
    gdb_server_config_t config = gdb_server_default_config();
    jtag_interface_t jtag;
    debug_target_t target;
    
    create_mock_jtag_interface(&jtag);
    debug_init(&jtag, &target);
    
    // Test server initialization
    int result = gdb_server_init(&server, &config, &jtag, &target);
    TEST_ASSERT(result == GDB_SUCCESS, "GDB server init should succeed");
    TEST_ASSERT(server.jtag == &jtag, "JTAG interface should be set");
    TEST_ASSERT(server.target == &target, "Target should be set");
    
    gdb_server_cleanup(&server);
    jtag_cleanup(&jtag);
    TEST_PASS();
}

int test_gdb_packet_handling(void) {
    // Test packet checksum calculation
    uint8_t checksum = gdb_checksum("qSupported");
    TEST_ASSERT(checksum != 0, "Checksum should be calculated");
    
    // Test hex conversion
    uint8_t bin_data[4];
    int result = gdb_hex_to_bin("DEADBEEF", bin_data, sizeof(bin_data));
    TEST_ASSERT(result == 4, "Hex to binary conversion should succeed");
    TEST_ASSERT(bin_data[0] == 0xDE, "First byte should be correct");
    TEST_ASSERT(bin_data[3] == 0xEF, "Last byte should be correct");
    
    // Test binary to hex conversion
    char hex_data[16];
    result = gdb_bin_to_hex(bin_data, 4, hex_data, sizeof(hex_data));
    TEST_ASSERT(result == 8, "Binary to hex conversion should succeed");
    TEST_ASSERT(strncmp(hex_data, "deadbeef", 8) == 0, "Hex string should be correct");
    
    TEST_PASS();
}

int test_performance_counters(void) {
    // Test counter initialization
    int result = perf_init_counters();
    TEST_ASSERT(result == PERF_SUCCESS, "Performance counter init should succeed");
    
    // Test counter enable/disable
    result = perf_enable_counter(PERF_COUNTER_CYCLES);
    TEST_ASSERT(result == PERF_SUCCESS, "Counter enable should succeed");
    
    result = perf_disable_counter(PERF_COUNTER_CYCLES);
    TEST_ASSERT(result == PERF_SUCCESS, "Counter disable should succeed");
    
    // Test counter read
    uint64_t counter_value;
    result = perf_read_counter(PERF_COUNTER_CYCLES, &counter_value);
    TEST_ASSERT(result == PERF_SUCCESS, "Counter read should succeed");
    
    // Test counter reset
    result = perf_reset_counter(PERF_COUNTER_CYCLES);
    TEST_ASSERT(result == PERF_SUCCESS, "Counter reset should succeed");
    
    result = perf_read_counter(PERF_COUNTER_CYCLES, &counter_value);
    TEST_ASSERT(result == PERF_SUCCESS && counter_value == 0, "Counter should be reset to zero");
    
    TEST_PASS();
}

int test_performance_session(void) {
    perf_session_t session;
    
    // Test session creation
    int result = perf_session_create(&session, "test_session");
    TEST_ASSERT(result == PERF_SUCCESS, "Session creation should succeed");
    TEST_ASSERT(strcmp(session.session_name, "test_session") == 0, "Session name should be set");
    
    // Test adding counters
    result = perf_session_add_counter(&session, PERF_COUNTER_CYCLES);
    TEST_ASSERT(result == PERF_SUCCESS, "Adding counter should succeed");
    TEST_ASSERT(session.counter_count == 1, "Counter count should be incremented");
    
    result = perf_session_add_counter(&session, PERF_COUNTER_INSTRUCTIONS);
    TEST_ASSERT(result == PERF_SUCCESS, "Adding second counter should succeed");
    TEST_ASSERT(session.counter_count == 2, "Counter count should be 2");
    
    // Test session start/stop
    result = perf_session_start(&session);
    TEST_ASSERT(result == PERF_SUCCESS, "Session start should succeed");
    TEST_ASSERT(session.active == true, "Session should be active");
    
    // Simulate some work
    usleep(1000);
    
    result = perf_session_stop(&session);
    TEST_ASSERT(result == PERF_SUCCESS, "Session stop should succeed");
    TEST_ASSERT(session.active == false, "Session should be inactive");
    
    perf_session_destroy(&session);
    TEST_PASS();
}

int test_profiling_functionality(void) {
    profiling_data_t profiling_data;
    memset(&profiling_data, 0, sizeof(profiling_data));
    profiling_data.max_samples = 100;
    
    // Test profiling start
    int result = profiling_start(&profiling_data, 1000); // Sample every 1000 cycles
    TEST_ASSERT(result == PERF_SUCCESS, "Profiling start should succeed");
    
    // Simulate collecting samples
    for (int i = 0; i < 10; i++) {
        result = profiling_collect_sample(&profiling_data);
        TEST_ASSERT(result == PERF_SUCCESS, "Sample collection should succeed");
        usleep(100);
    }
    
    TEST_ASSERT(profiling_data.sample_count == 10, "Should have collected 10 samples");
    
    // Test profiling analysis
    result = profiling_analyze(&profiling_data);
    TEST_ASSERT(result == PERF_SUCCESS, "Profiling analysis should succeed");
    
    // Test profiling stop
    result = profiling_stop(&profiling_data);
    TEST_ASSERT(result == PERF_SUCCESS, "Profiling stop should succeed");
    
    profiling_cleanup(&profiling_data);
    TEST_PASS();
}

int test_ai_accelerator_monitoring(void) {
    // Test TPU utilization monitoring
    double tpu_utilization;
    int result = perf_monitor_tpu_utilization(0, &tpu_utilization);
    TEST_ASSERT(result == PERF_SUCCESS, "TPU utilization monitoring should succeed");
    TEST_ASSERT(tpu_utilization >= 0.0 && tpu_utilization <= 100.0, "TPU utilization should be valid percentage");
    
    // Test VPU utilization monitoring
    double vpu_utilization;
    result = perf_monitor_vpu_utilization(0, &vpu_utilization);
    TEST_ASSERT(result == PERF_SUCCESS, "VPU utilization monitoring should succeed");
    TEST_ASSERT(vpu_utilization >= 0.0 && vpu_utilization <= 100.0, "VPU utilization should be valid percentage");
    
    // Test NoC traffic monitoring
    uint64_t packets_sent, packets_received;
    result = perf_monitor_noc_traffic(&packets_sent, &packets_received);
    TEST_ASSERT(result == PERF_SUCCESS, "NoC traffic monitoring should succeed");
    
    // Test power consumption monitoring
    double current_power, average_power;
    result = perf_monitor_power_consumption(&current_power, &average_power);
    TEST_ASSERT(result == PERF_SUCCESS, "Power monitoring should succeed");
    TEST_ASSERT(current_power >= 0.0, "Current power should be non-negative");
    TEST_ASSERT(average_power >= 0.0, "Average power should be non-negative");
    
    TEST_PASS();
}

int test_utility_functions(void) {
    // Test IPC calculation
    double ipc = perf_calculate_ipc(1000, 2000);
    TEST_ASSERT(ipc == 0.5, "IPC calculation should be correct");
    
    // Test cache hit rate calculation
    double hit_rate = perf_calculate_cache_hit_rate(900, 100);
    TEST_ASSERT(hit_rate == 90.0, "Cache hit rate should be 90%");
    
    // Test cycles to nanoseconds conversion
    uint64_t ns = perf_cycles_to_nanoseconds(1000000, 1000000000); // 1M cycles at 1GHz
    TEST_ASSERT(ns == 1000000, "Should be 1ms in nanoseconds");
    
    // Test bandwidth calculation
    double bandwidth = perf_calculate_bandwidth(1000000, 1000000000); // 1MB in 1 second
    TEST_ASSERT(bandwidth == 1000000.0, "Bandwidth should be 1MB/s");
    
    // Test counter name lookup
    const char* name = perf_counter_name(PERF_COUNTER_CYCLES);
    TEST_ASSERT(strcmp(name, "cycles") == 0, "Counter name should be 'cycles'");
    
    TEST_PASS();
}

int test_report_generation(void) {
    perf_report_t report;
    memset(&report, 0, sizeof(report));
    
    // Fill in some test data
    strcpy(report.title, "Test Performance Report");
    report.total_cycles = 1000000;
    report.total_instructions = 800000;
    report.average_ipc = 0.8;
    report.l1_cache_hits = 900000;
    report.l1_cache_misses = 100000;
    report.l1_hit_rate = 90.0;
    
    // Test report generation
    int result = perf_generate_report(&report, "/tmp/test_report.txt");
    TEST_ASSERT(result == PERF_SUCCESS, "Report generation should succeed");
    
    // Test JSON export
    result = perf_export_json(&report, "/tmp/test_report.json");
    TEST_ASSERT(result == PERF_SUCCESS, "JSON export should succeed");
    
    // Cleanup test files
    unlink("/tmp/test_report.txt");
    unlink("/tmp/test_report.json");
    
    TEST_PASS();
}

// Integration tests
int test_end_to_end_debugging(void) {
    jtag_interface_t jtag;
    debug_target_t target;
    gdb_server_t server;
    gdb_server_config_t config = gdb_server_default_config();
    
    // Initialize JTAG and debug target
    create_mock_jtag_interface(&jtag);
    int result = debug_init(&jtag, &target);
    TEST_ASSERT(result == JTAG_SUCCESS, "Debug initialization should succeed");
    
    // Initialize GDB server
    config.port = 3333; // Use non-standard port for testing
    result = gdb_server_init(&server, &config, &jtag, &target);
    TEST_ASSERT(result == GDB_SUCCESS, "GDB server initialization should succeed");
    
    // Start GDB server (in background)
    result = gdb_server_start(&server);
    TEST_ASSERT(result == GDB_SUCCESS, "GDB server start should succeed");
    
    // Give server time to start
    usleep(100000);
    
    // Test connection (simplified)
    int client_socket = create_test_gdb_client(config.port);
    TEST_ASSERT(client_socket >= 0, "Should be able to connect to GDB server");
    
    if (client_socket >= 0) {
        close(client_socket);
    }
    
    // Stop GDB server
    result = gdb_server_stop(&server);
    TEST_ASSERT(result == GDB_SUCCESS, "GDB server stop should succeed");
    
    // Cleanup
    gdb_server_cleanup(&server);
    jtag_cleanup(&jtag);
    
    TEST_PASS();
}

// Helper functions
static void run_test(int (*test_func)(void), const char* test_name) {
    tests_run++;
    printf("Running %s...\n", test_name);
    
    int result = test_func();
    if (result == 0) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("FAILED: %s\n", test_name);
    }
}

static int create_mock_jtag_interface(jtag_interface_t* jtag) {
    // Create a mock JTAG interface for testing
    memset(jtag, 0, sizeof(jtag_interface_t));
    jtag->tck_pin = 1;
    jtag->tms_pin = 2;
    jtag->tdi_pin = 3;
    jtag->tdo_pin = 4;
    jtag->trst_pin = 5;
    jtag->state = JTAG_STATE_RUN_TEST_IDLE;
    jtag->initialized = true;
    return 0;
}

static int create_test_gdb_client(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        return -1;
    }
    
    return sock;
}

// Main test runner
int main(int argc, char* argv[]) {
    printf("RISC-V AI Accelerator Debug Tools Test Suite\n");
    printf("=============================================\n\n");
    
    // Initialize performance counters for testing
    perf_init_counters();
    
    // Run JTAG tests
    printf("JTAG Interface Tests:\n");
    run_test(test_jtag_initialization, "JTAG Initialization");
    run_test(test_jtag_state_machine, "JTAG State Machine");
    run_test(test_jtag_data_transfer, "JTAG Data Transfer");
    run_test(test_debug_module_interface, "Debug Module Interface");
    
    // Run GDB server tests
    printf("\nGDB Server Tests:\n");
    run_test(test_gdb_server_initialization, "GDB Server Initialization");
    run_test(test_gdb_packet_handling, "GDB Packet Handling");
    
    // Run performance analysis tests
    printf("\nPerformance Analysis Tests:\n");
    run_test(test_performance_counters, "Performance Counters");
    run_test(test_performance_session, "Performance Session");
    run_test(test_profiling_functionality, "Profiling Functionality");
    run_test(test_ai_accelerator_monitoring, "AI Accelerator Monitoring");
    run_test(test_utility_functions, "Utility Functions");
    run_test(test_report_generation, "Report Generation");
    
    // Run integration tests
    printf("\nIntegration Tests:\n");
    run_test(test_end_to_end_debugging, "End-to-End Debugging");
    
    // Print summary
    printf("\n=============================================\n");
    printf("Test Summary:\n");
    printf("  Total tests: %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("  Success rate: %.1f%%\n", (double)tests_passed / tests_run * 100.0);
    
    return (tests_failed == 0) ? 0 : 1;
}