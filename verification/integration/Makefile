# Makefile for System Integration Testing
# Comprehensive system-level integration test execution

# Tool configuration
SIMULATOR ?= questa
UVM_VERSION ?= 1.2
WAVES ?= 0
COVERAGE ?= 1
VERBOSITY ?= UVM_MEDIUM

# Directories
RTL_DIR = ../../rtl
UVM_DIR = ../uvm
INTEGRATION_DIR = .
WORK_DIR = work
LOG_DIR = logs
REPORT_DIR = reports

# Integration test source files
INTEGRATION_SOURCES = \
	$(INTEGRATION_DIR)/system_integration_pkg.sv \
	$(INTEGRATION_DIR)/tb_system_integration.sv

# UVM source files
UVM_SOURCES = \
	$(UVM_DIR)/riscv_ai_interface.sv \
	$(UVM_DIR)/riscv_ai_pkg.sv

# RTL source files (comprehensive set for integration)
RTL_SOURCES = \
	$(RTL_DIR)/config/chip_config.sv \
	$(RTL_DIR)/interfaces/system_interfaces.sv \
	$(RTL_DIR)/interfaces/power_mgmt_if.sv \
	$(RTL_DIR)/core/riscv_core.sv \
	$(RTL_DIR)/core/riscv_alu.sv \
	$(RTL_DIR)/core/riscv_fpu.sv \
	$(RTL_DIR)/core/riscv_mdu.sv \
	$(RTL_DIR)/core/riscv_vector_unit.sv \
	$(RTL_DIR)/core/ai_matmul_unit.sv \
	$(RTL_DIR)/core/ai_conv2d_unit.sv \
	$(RTL_DIR)/accelerators/tpu.sv \
	$(RTL_DIR)/accelerators/vpu.sv \
	$(RTL_DIR)/memory/l1_icache.sv \
	$(RTL_DIR)/memory/l1_dcache.sv \
	$(RTL_DIR)/memory/l2_cache.sv \
	$(RTL_DIR)/memory/l3_cache.sv \
	$(RTL_DIR)/memory/memory_subsystem.sv \
	$(RTL_DIR)/memory/hbm_controller.sv \
	$(RTL_DIR)/noc/noc_mesh.sv \
	$(RTL_DIR)/noc/noc_router.sv \
	$(RTL_DIR)/power/power_manager.sv \
	$(RTL_DIR)/power/dvfs_controller.sv \
	$(RTL_DIR)/power/thermal_controller.sv \
	$(RTL_DIR)/top/riscv_ai_chip.sv

# Available integration tests
INTEGRATION_TESTS = \
	multi_core_coordination_test \
	cache_coherency_test \
	noc_communication_test \
	power_thermal_test \
	stress_endurance_test \
	mixed_workload_test \
	system_boot_test

# Default test
INTEGRATION_TEST ?= multi_core_coordination_test

# Simulator-specific settings
ifeq ($(SIMULATOR), questa)
    VLOG = vlog
    VSIM = vsim
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(INTEGRATION_DIR) +incdir+$(RTL_DIR) -sv -timescale 1ns/1ps
    VSIM_OPTS = -c -do "run -all; quit -f"
    UVM_OPTS = +UVM_TESTNAME=$(INTEGRATION_TEST) +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -wlf $(REPORT_DIR)/integration_waves.wlf
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += +cover=bcfst
        VSIM_OPTS += -coverage -coverstore $(REPORT_DIR)/integration_coverage
    endif
endif

# Targets
.PHONY: all clean compile run test regression help

all: compile

# Create directories
$(WORK_DIR) $(LOG_DIR) $(REPORT_DIR):
	mkdir -p $@

# Compile integration test framework
compile: $(WORK_DIR) $(LOG_DIR)
	@echo "Compiling system integration test framework..."
	$(VLOG) $(VLOG_OPTS) -work $(WORK_DIR) $(RTL_SOURCES) $(UVM_SOURCES) $(INTEGRATION_SOURCES) 2>&1 | tee $(LOG_DIR)/compile.log
	@echo "Compilation complete."

# Run integration test
run: compile $(REPORT_DIR)
	@echo "Running integration test: $(INTEGRATION_TEST)"
	$(VSIM) $(VSIM_OPTS) $(UVM_OPTS) -work $(WORK_DIR) tb_system_integration 2>&1 | tee $(LOG_DIR)/$(INTEGRATION_TEST).log
	@echo "Integration test $(INTEGRATION_TEST) complete."

# Run specific integration tests
multi_core:
	$(MAKE) INTEGRATION_TEST=multi_core_coordination_test run

cache_coherency:
	$(MAKE) INTEGRATION_TEST=cache_coherency_test run

noc_comm:
	$(MAKE) INTEGRATION_TEST=noc_communication_test run

power_thermal:
	$(MAKE) INTEGRATION_TEST=power_thermal_test run

stress:
	$(MAKE) INTEGRATION_TEST=stress_endurance_test run

mixed_workload:
	$(MAKE) INTEGRATION_TEST=mixed_workload_test run

# Run integration test regression
regression: compile $(REPORT_DIR)
	@echo "Running integration test regression..."
	@for test in $(INTEGRATION_TESTS); do \
		echo "Running integration test: $$test"; \
		$(MAKE) INTEGRATION_TEST=$$test run; \
		if [ $$? -ne 0 ]; then \
			echo "Integration test $$test FAILED"; \
			exit 1; \
		fi; \
	done
	@echo "Integration test regression complete."

# Generate integration reports
reports: $(REPORT_DIR)
	@echo "Generating integration test reports..."
	@if [ -f $(REPORT_DIR)/integration_results.csv ]; then \
		echo "Analyzing integration results..."; \
		python3 scripts/analyze_integration.py $(REPORT_DIR)/integration_results.csv $(REPORT_DIR)/; \
	else \
		echo "No integration results found."; \
	fi

# Quick smoke test
smoke:
	$(MAKE) INTEGRATION_TEST=multi_core_coordination_test VERBOSITY=UVM_LOW run

# Performance integration test
perf:
	$(MAKE) INTEGRATION_TEST=stress_endurance_test VERBOSITY=UVM_LOW run

# Debug with waves
debug:
	$(MAKE) INTEGRATION_TEST=$(INTEGRATION_TEST) WAVES=1 VERBOSITY=UVM_HIGH run

# Clean build artifacts
clean:
	rm -rf $(WORK_DIR) $(LOG_DIR) $(REPORT_DIR)
	rm -f transcript vsim.wlf *.log
	rm -f ucli.key vc_hdrs.h
	rm -rf csrc simv* *.daidir

# Help
help:
	@echo "System Integration Testing Makefile"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Compile integration framework (default)"
	@echo "  compile          - Compile RTL, UVM, and integration sources"
	@echo "  run              - Run integration test (specify with INTEGRATION_TEST=)"
	@echo "  multi_core       - Run multi-core coordination test"
	@echo "  cache_coherency  - Run cache coherency test"
	@echo "  noc_comm         - Run NoC communication test"
	@echo "  power_thermal    - Run power and thermal management test"
	@echo "  stress           - Run stress and endurance test"
	@echo "  mixed_workload   - Run mixed workload test"
	@echo "  regression       - Run all integration tests"
	@echo "  reports          - Generate integration test reports"
	@echo "  smoke            - Run quick smoke test"
	@echo "  perf             - Run performance integration test"
	@echo "  debug            - Run with waveforms and high verbosity"
	@echo "  clean            - Clean build artifacts"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  SIMULATOR={questa|vcs}     - Choose simulator (default: questa)"
	@echo "  INTEGRATION_TEST=test_name - Specify integration test to run"
	@echo "  WAVES={0|1}                - Enable waveform generation (default: 0)"
	@echo "  COVERAGE={0|1}             - Enable coverage collection (default: 1)"
	@echo "  VERBOSITY={UVM_LOW|UVM_MEDIUM|UVM_HIGH} - Set UVM verbosity"
	@echo ""
	@echo "Available Integration Tests:"
	@for test in $(INTEGRATION_TESTS); do echo "  $$test"; done