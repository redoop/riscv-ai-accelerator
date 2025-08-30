# Makefile for RISC-V AI Accelerator UVM Verification
# Comprehensive verification environment build and run system

# Tool configuration
SIMULATOR ?= questa
UVM_VERSION ?= 1.2
WAVES ?= 1
COVERAGE ?= 1
VERBOSITY ?= UVM_MEDIUM

# Directories
RTL_DIR = ../../rtl
TB_DIR = .
UVM_DIR = .
WORK_DIR = work
LOG_DIR = logs
COV_DIR = coverage
WAVE_DIR = waves

# RTL source files
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
	$(RTL_DIR)/core/ai_activation_unit.sv \
	$(RTL_DIR)/accelerators/tpu.sv \
	$(RTL_DIR)/accelerators/tpu_compute_array.sv \
	$(RTL_DIR)/accelerators/tpu_mac_unit.sv \
	$(RTL_DIR)/accelerators/vpu.sv \
	$(RTL_DIR)/accelerators/vector_alu.sv \
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
	$(RTL_DIR)/top/riscv_ai_chip.sv

# UVM source files
UVM_SOURCES = \
	$(UVM_DIR)/riscv_ai_interface.sv \
	$(UVM_DIR)/riscv_ai_pkg.sv

# Testbench files
TB_SOURCES = \
	$(TB_DIR)/tb_riscv_ai_uvm.sv

# Test list
TESTS = \
	riscv_ai_smoke_test \
	riscv_ai_random_test \
	riscv_ai_matmul_test \
	riscv_ai_conv2d_test \
	riscv_ai_activation_test \
	riscv_ai_memory_test \
	riscv_ai_stress_test \
	riscv_ai_power_test \
	riscv_ai_error_test

# Default test
TEST ?= riscv_ai_smoke_test

# Simulator-specific settings
ifeq ($(SIMULATOR), questa)
    VLOG = vlog
    VSIM = vsim
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(RTL_DIR) -sv -timescale 1ns/1ps
    VSIM_OPTS = -c -do "run -all; quit -f"
    UVM_OPTS = +UVM_TESTNAME=$(TEST) +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -wlf $(WAVE_DIR)/$(TEST).wlf
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += +cover=bcfst
        VSIM_OPTS += -coverage -coverstore $(COV_DIR)/$(TEST)
    endif
    
else ifeq ($(SIMULATOR), vcs)
    VLOG = vlogan
    VSIM = vcs
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(RTL_DIR) -sverilog -timescale=1ns/1ps
    VSIM_OPTS = -R +UVM_TESTNAME=$(TEST) +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -debug_access+all +vcs+dumpvars+$(WAVE_DIR)/$(TEST).vpd
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += -cm line+cond+fsm+branch+tgl
        VSIM_OPTS += -cm line+cond+fsm+branch+tgl -cm_dir $(COV_DIR)/$(TEST)
    endif
    
else ifeq ($(SIMULATOR), xcelium)
    VLOG = xmvlog
    VSIM = xmsim
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(RTL_DIR) -sv -timescale 1ns/1ps
    VSIM_OPTS = -R +UVM_TESTNAME=$(TEST) +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -input $(TB_DIR)/xcelium_waves.tcl
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += -coverage all
        VSIM_OPTS += -coverage all -covdut tb_riscv_ai_uvm
    endif
endif

# Targets
.PHONY: all clean compile run test regression coverage waves help

all: compile

# Create directories
$(WORK_DIR) $(LOG_DIR) $(COV_DIR) $(WAVE_DIR):
	mkdir -p $@

# Compile RTL and testbench
compile: $(WORK_DIR) $(LOG_DIR)
	@echo "Compiling RTL and UVM testbench..."
	$(VLOG) $(VLOG_OPTS) -work $(WORK_DIR) $(RTL_SOURCES) $(UVM_SOURCES) $(TB_SOURCES) 2>&1 | tee $(LOG_DIR)/compile.log
	@echo "Compilation complete."

# Run single test
run: compile $(WAVE_DIR) $(COV_DIR)
	@echo "Running test: $(TEST)"
	$(VSIM) $(VSIM_OPTS) -work $(WORK_DIR) tb_riscv_ai_uvm 2>&1 | tee $(LOG_DIR)/$(TEST).log
	@echo "Test $(TEST) complete. Check $(LOG_DIR)/$(TEST).log for results."

# Run specific test (alias for run)
test: run

# Run regression (all tests)
regression: compile $(WAVE_DIR) $(COV_DIR)
	@echo "Running regression with $(words $(TESTS)) tests..."
	@for test in $(TESTS); do \
		echo "Running $$test..."; \
		$(MAKE) TEST=$$test run; \
		if [ $$? -ne 0 ]; then \
			echo "Test $$test FAILED"; \
			exit 1; \
		fi; \
	done
	@echo "Regression complete. All tests passed."

# Generate coverage report
coverage: $(COV_DIR)
ifeq ($(SIMULATOR), questa)
	@echo "Generating coverage report..."
	vcover merge $(COV_DIR)/merged_coverage $(COV_DIR)/*
	vcover report -html -htmldir $(COV_DIR)/html $(COV_DIR)/merged_coverage
	@echo "Coverage report generated in $(COV_DIR)/html/"
else ifeq ($(SIMULATOR), vcs)
	@echo "Generating coverage report..."
	urg -dir $(COV_DIR)/* -report $(COV_DIR)/html
	@echo "Coverage report generated in $(COV_DIR)/html/"
else ifeq ($(SIMULATOR), xcelium)
	@echo "Generating coverage report..."
	imc -load $(COV_DIR)/* -execcmd "report -html -out $(COV_DIR)/html"
	@echo "Coverage report generated in $(COV_DIR)/html/"
endif

# Open waveforms
waves: $(WAVE_DIR)
ifeq ($(SIMULATOR), questa)
	@if [ -f $(WAVE_DIR)/$(TEST).wlf ]; then \
		vsim -view $(WAVE_DIR)/$(TEST).wlf; \
	else \
		echo "No waveform file found for $(TEST). Run with WAVES=1"; \
	fi
else ifeq ($(SIMULATOR), vcs)
	@if [ -f $(WAVE_DIR)/$(TEST).vpd ]; then \
		dve -vpd $(WAVE_DIR)/$(TEST).vpd; \
	else \
		echo "No waveform file found for $(TEST). Run with WAVES=1"; \
	fi
else ifeq ($(SIMULATOR), xcelium)
	@if [ -f $(WAVE_DIR)/$(TEST).shm ]; then \
		simvision $(WAVE_DIR)/$(TEST).shm; \
	else \
		echo "No waveform file found for $(TEST). Run with WAVES=1"; \
	fi
endif

# Clean build artifacts
clean:
	rm -rf $(WORK_DIR) $(LOG_DIR) $(COV_DIR) $(WAVE_DIR)
	rm -f transcript vsim.wlf *.log
	rm -f ucli.key vc_hdrs.h
	rm -rf csrc simv* *.daidir
	rm -rf xcelium.d xmsim.key waves.shm

# Quick smoke test
smoke:
	$(MAKE) TEST=riscv_ai_smoke_test run

# Performance test
perf:
	$(MAKE) TEST=riscv_ai_stress_test VERBOSITY=UVM_LOW run

# Debug test with waves
debug:
	$(MAKE) TEST=$(TEST) WAVES=1 VERBOSITY=UVM_HIGH run

# Help
help:
	@echo "RISC-V AI Accelerator UVM Verification Makefile"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Compile RTL and testbench (default)"
	@echo "  compile     - Compile RTL and testbench"
	@echo "  run         - Run single test (specify with TEST=test_name)"
	@echo "  test        - Alias for run"
	@echo "  regression  - Run all tests"
	@echo "  coverage    - Generate coverage report"
	@echo "  waves       - Open waveform viewer"
	@echo "  smoke       - Run quick smoke test"
	@echo "  perf        - Run performance stress test"
	@echo "  debug       - Run test with waves and high verbosity"
	@echo "  clean       - Clean build artifacts"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  SIMULATOR={questa|vcs|xcelium}  - Choose simulator (default: questa)"
	@echo "  TEST=test_name                  - Specify test to run (default: riscv_ai_smoke_test)"
	@echo "  WAVES={0|1}                     - Enable waveform generation (default: 1)"
	@echo "  COVERAGE={0|1}                  - Enable coverage collection (default: 1)"
	@echo "  VERBOSITY={UVM_LOW|UVM_MEDIUM|UVM_HIGH|UVM_FULL} - Set UVM verbosity (default: UVM_MEDIUM)"
	@echo ""
	@echo "Available Tests:"
	@for test in $(TESTS); do echo "  $$test"; done
	@echo ""
	@echo "Examples:"
	@echo "  make compile                    - Compile everything"
	@echo "  make run TEST=riscv_ai_matmul_test - Run matrix multiplication test"
	@echo "  make regression                 - Run all tests"
	@echo "  make debug TEST=riscv_ai_conv2d_test - Debug convolution test with waves"
	@echo "  make coverage                   - Generate coverage report"

# Print configuration
config:
	@echo "Configuration:"
	@echo "  SIMULATOR: $(SIMULATOR)"
	@echo "  TEST: $(TEST)"
	@echo "  WAVES: $(WAVES)"
	@echo "  COVERAGE: $(COVERAGE)"
	@echo "  VERBOSITY: $(VERBOSITY)"
	@echo "  UVM_VERSION: $(UVM_VERSION)"