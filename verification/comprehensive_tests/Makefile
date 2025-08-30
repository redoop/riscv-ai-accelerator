# Comprehensive Test Suite Makefile
# Runs all test suites for RISC-V AI Accelerator

# Simulator settings
SIM = verilator
WAVES = --trace --trace-structs
CFLAGS = -Wall -Wno-UNUSED -Wno-UNOPTFLAT -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-LATCH -Wno-CASEINCOMPLETE
VFLAGS = --cc --exe --build -j 4 --timing $(WAVES) $(CFLAGS)

# Directories
RTL_DIR = ../../rtl
CORE_DIR = $(RTL_DIR)/core
ACCEL_DIR = $(RTL_DIR)/accelerators
CONFIG_DIR = $(RTL_DIR)/config
INTERFACE_DIR = $(RTL_DIR)/interfaces
MEMORY_DIR = $(RTL_DIR)/memory
NOC_DIR = $(RTL_DIR)/noc
POWER_DIR = $(RTL_DIR)/power
TOP_DIR = $(RTL_DIR)/top

# Include paths
INCLUDES = -I$(RTL_DIR) -I$(CORE_DIR) -I$(ACCEL_DIR) -I$(CONFIG_DIR) \
           -I$(INTERFACE_DIR) -I$(MEMORY_DIR) -I$(NOC_DIR) -I$(POWER_DIR) -I$(TOP_DIR)

# RTL source files
RTL_SOURCES = \
    $(CONFIG_DIR)/chip_config_pkg.sv \
    $(INTERFACE_DIR)/ai_accel_if.sv \
    $(INTERFACE_DIR)/axi4_if.sv \
    $(INTERFACE_DIR)/noc_if.sv \
    $(CORE_DIR)/riscv_alu.sv \
    $(CORE_DIR)/riscv_control_unit.sv \
    $(CORE_DIR)/riscv_forwarding_unit.sv \
    $(CORE_DIR)/riscv_hazard_unit.sv \
    $(CORE_DIR)/riscv_branch_unit.sv \
    $(CORE_DIR)/riscv_mdu.sv \
    $(CORE_DIR)/riscv_fpu.sv \
    $(CORE_DIR)/riscv_vector_unit.sv \
    $(CORE_DIR)/riscv_ai_unit.sv \
    $(CORE_DIR)/riscv_core.sv \
    $(ACCEL_DIR)/tpu_mac_unit.sv \
    $(ACCEL_DIR)/tpu_compute_array.sv \
    $(ACCEL_DIR)/tpu.sv \
    $(ACCEL_DIR)/vector_alu.sv \
    $(ACCEL_DIR)/vpu.sv

# Test files
COMPREHENSIVE_TEST = test_riscv_ai_comprehensive.sv
AI_INSTRUCTIONS_TEST = test_ai_instructions_detailed.sv
PERFORMANCE_TEST = test_performance_benchmarks.sv
INTEGRATION_TEST = test_system_integration.sv

# Test executables
COMPREHENSIVE_EXE = obj_dir/Vtest_riscv_ai_comprehensive
AI_INSTRUCTIONS_EXE = obj_dir/Vtest_ai_instructions_detailed
PERFORMANCE_EXE = obj_dir/Vtest_performance_benchmarks
INTEGRATION_EXE = obj_dir/Vtest_system_integration

# Default target
all: comprehensive ai_instructions performance integration

# Individual test targets
comprehensive: $(COMPREHENSIVE_EXE)
	@echo "Running Comprehensive Test Suite..."
	@./$(COMPREHENSIVE_EXE)

ai_instructions: $(AI_INSTRUCTIONS_EXE)
	@echo "Running AI Instructions Test Suite..."
	@./$(AI_INSTRUCTIONS_EXE)

performance: $(PERFORMANCE_EXE)
	@echo "Running Performance Benchmarks..."
	@./$(PERFORMANCE_EXE)

integration: $(INTEGRATION_EXE)
	@echo "Running System Integration Tests..."
	@./$(INTEGRATION_EXE)

# Build comprehensive test
$(COMPREHENSIVE_EXE): $(COMPREHENSIVE_TEST) $(RTL_SOURCES) sim_main.cpp
	@echo "Building Comprehensive Test..."
	@$(SIM) $(VFLAGS) $(INCLUDES) --top-module test_riscv_ai_comprehensive \
		$(RTL_SOURCES) $(COMPREHENSIVE_TEST) sim_main.cpp

# Build AI instructions test
$(AI_INSTRUCTIONS_EXE): $(AI_INSTRUCTIONS_TEST) $(RTL_SOURCES) sim_main.cpp
	@echo "Building AI Instructions Test..."
	@$(SIM) $(VFLAGS) $(INCLUDES) --top-module test_ai_instructions_detailed \
		$(RTL_SOURCES) $(AI_INSTRUCTIONS_TEST) sim_main.cpp

# Build performance test
$(PERFORMANCE_EXE): $(PERFORMANCE_TEST) $(RTL_SOURCES) sim_main.cpp
	@echo "Building Performance Test..."
	@$(SIM) $(VFLAGS) $(INCLUDES) --top-module test_performance_benchmarks \
		$(RTL_SOURCES) $(PERFORMANCE_TEST) sim_main.cpp

# Build integration test
$(INTEGRATION_EXE): $(INTEGRATION_TEST) $(RTL_SOURCES) sim_main.cpp
	@echo "Building Integration Test..."
	@$(SIM) $(VFLAGS) $(INCLUDES) --top-module test_system_integration \
		$(RTL_SOURCES) $(INTEGRATION_TEST) sim_main.cpp

# Run all tests in sequence
test_all: clean all
	@echo "========================================="
	@echo "Running Complete Test Suite"
	@echo "========================================="
	@$(MAKE) comprehensive
	@echo ""
	@$(MAKE) ai_instructions
	@echo ""
	@$(MAKE) performance
	@echo ""
	@$(MAKE) integration
	@echo ""
	@echo "========================================="
	@echo "All Tests Completed"
	@echo "========================================="

# Run tests with coverage
coverage: VFLAGS += --coverage
coverage: clean all
	@echo "Running tests with coverage analysis..."
	@$(MAKE) test_all
	@echo "Coverage reports generated in obj_dir/coverage/"

# Syntax check only
syntax_check:
	@echo "Performing syntax check..."
	@$(SIM) --lint-only --timing --Wno-fatal $(INCLUDES) $(RTL_SOURCES) $(COMPREHENSIVE_TEST)
	@$(SIM) --lint-only --timing --Wno-fatal $(INCLUDES) $(RTL_SOURCES) $(AI_INSTRUCTIONS_TEST)
	@$(SIM) --lint-only --timing --Wno-fatal $(INCLUDES) $(RTL_SOURCES) $(PERFORMANCE_TEST)
	@$(SIM) --lint-only --timing --Wno-fatal $(INCLUDES) $(RTL_SOURCES) $(INTEGRATION_TEST)
	@echo "Syntax check completed successfully"

# Quick test (reduced simulation time)
quick_test: CFLAGS += -DQUICK_TEST
quick_test: clean all

# Debug build with more verbose output
debug: CFLAGS += -DDEBUG -g
debug: VFLAGS += --debug --gdbbt
debug: clean all

# Performance optimized build
optimized: CFLAGS += -O3 -DNDEBUG
optimized: clean all

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf obj_dir/
	@rm -f *.vcd *.fst *.log

# Create missing RTL files with stubs if they don't exist
create_stubs:
	@echo "Creating RTL stubs for missing files..."
	@mkdir -p $(CONFIG_DIR) $(INTERFACE_DIR) $(CORE_DIR) $(ACCEL_DIR)
	@for file in $(RTL_SOURCES); do \
		if [ ! -f $$file ]; then \
			echo "Creating stub for $$file"; \
			echo "// Stub file for $$file" > $$file; \
			echo "module $$(basename $$file .sv) ();" >> $$file; \
			echo "endmodule" >> $$file; \
		fi \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build all test suites"
	@echo "  comprehensive - Run comprehensive functionality tests"
	@echo "  ai_instructions - Run AI instruction tests"
	@echo "  performance   - Run performance benchmarks"
	@echo "  integration   - Run system integration tests"
	@echo "  test_all      - Run complete test suite"
	@echo "  coverage      - Run tests with coverage analysis"
	@echo "  syntax_check  - Perform syntax check only"
	@echo "  quick_test    - Run quick tests (reduced simulation)"
	@echo "  debug         - Build with debug information"
	@echo "  optimized     - Build with optimizations"
	@echo "  create_stubs  - Create stub files for missing RTL"
	@echo "  clean         - Clean build artifacts"
	@echo "  help          - Show this help message"

# Phony targets
.PHONY: all comprehensive ai_instructions performance integration test_all \
        coverage syntax_check quick_test debug optimized clean create_stubs help

# Default shell
SHELL := /bin/bash