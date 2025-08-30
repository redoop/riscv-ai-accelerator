# Makefile for AI Benchmark Testing
# Comprehensive AI workload benchmark execution system

# Tool configuration
SIMULATOR ?= questa
UVM_VERSION ?= 1.2
WAVES ?= 0
COVERAGE ?= 0
VERBOSITY ?= UVM_MEDIUM

# Directories
RTL_DIR = ../../rtl
UVM_DIR = ../uvm
BENCH_DIR = .
WORK_DIR = work
LOG_DIR = logs
REPORT_DIR = reports
DATA_DIR = data

# Benchmark source files
BENCH_SOURCES = \
	$(BENCH_DIR)/ai_benchmark_pkg.sv \
	$(BENCH_DIR)/tb_ai_benchmarks.sv

# UVM source files (from UVM directory)
UVM_SOURCES = \
	$(UVM_DIR)/riscv_ai_interface.sv \
	$(UVM_DIR)/riscv_ai_pkg.sv

# RTL source files (subset needed for benchmarks)
RTL_SOURCES = \
	$(RTL_DIR)/config/chip_config.sv \
	$(RTL_DIR)/interfaces/system_interfaces.sv \
	$(RTL_DIR)/core/riscv_core.sv \
	$(RTL_DIR)/accelerators/tpu.sv \
	$(RTL_DIR)/accelerators/vpu.sv \
	$(RTL_DIR)/top/riscv_ai_chip.sv

# Available benchmark suites
BENCHMARK_SUITES = \
	mlperf_inference \
	image_classification \
	object_detection \
	nlp_benchmarks \
	recommendation_systems \
	all_benchmarks

# Default benchmark suite
BENCHMARK_SUITE ?= mlperf_inference

# Simulator-specific settings
ifeq ($(SIMULATOR), questa)
    VLOG = vlog
    VSIM = vsim
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(BENCH_DIR) +incdir+$(RTL_DIR) -sv -timescale 1ns/1ps
    VSIM_OPTS = -c -do "run -all; quit -f"
    UVM_OPTS = +UVM_TESTNAME=ai_benchmark_test +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -wlf $(REPORT_DIR)/benchmark_waves.wlf
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += +cover=bcfst
        VSIM_OPTS += -coverage -coverstore $(REPORT_DIR)/benchmark_coverage
    endif
    
else ifeq ($(SIMULATOR), vcs)
    VLOG = vlogan
    VSIM = vcs
    VLOG_OPTS = +incdir+$(UVM_DIR) +incdir+$(BENCH_DIR) +incdir+$(RTL_DIR) -sverilog -timescale=1ns/1ps
    VSIM_OPTS = -R +UVM_TESTNAME=ai_benchmark_test +UVM_VERBOSITY=$(VERBOSITY)
    
    ifeq ($(WAVES), 1)
        VSIM_OPTS += -debug_access+all +vcs+dumpvars+$(REPORT_DIR)/benchmark_waves.vpd
    endif
    
    ifeq ($(COVERAGE), 1)
        VLOG_OPTS += -cm line+cond+fsm+branch+tgl
        VSIM_OPTS += -cm line+cond+fsm+branch+tgl -cm_dir $(REPORT_DIR)/benchmark_coverage
    endif
endif

# Targets
.PHONY: all clean compile run benchmarks help

all: compile

# Create directories
$(WORK_DIR) $(LOG_DIR) $(REPORT_DIR) $(DATA_DIR):
	mkdir -p $@

# Compile benchmark framework
compile: $(WORK_DIR) $(LOG_DIR)
	@echo "Compiling AI benchmark framework..."
	$(VLOG) $(VLOG_OPTS) -work $(WORK_DIR) $(RTL_SOURCES) $(UVM_SOURCES) $(BENCH_SOURCES) 2>&1 | tee $(LOG_DIR)/compile.log
	@echo "Compilation complete."

# Run benchmark suite
run: compile $(REPORT_DIR) $(DATA_DIR)
	@echo "Running benchmark suite: $(BENCHMARK_SUITE)"
	$(VSIM) $(VSIM_OPTS) -work $(WORK_DIR) tb_ai_benchmarks +BENCHMARK_SUITE=$(BENCHMARK_SUITE) 2>&1 | tee $(LOG_DIR)/$(BENCHMARK_SUITE).log
	@echo "Benchmark suite $(BENCHMARK_SUITE) complete. Check $(LOG_DIR)/$(BENCHMARK_SUITE).log for results."

# Run specific benchmark suites
mlperf: 
	$(MAKE) BENCHMARK_SUITE=mlperf_inference run

image_classification:
	$(MAKE) BENCHMARK_SUITE=image_classification run

object_detection:
	$(MAKE) BENCHMARK_SUITE=object_detection run

nlp:
	$(MAKE) BENCHMARK_SUITE=nlp_benchmarks run

recommendation:
	$(MAKE) BENCHMARK_SUITE=recommendation_systems run

all_benchmarks:
	$(MAKE) BENCHMARK_SUITE=all_benchmarks run

# Run comprehensive benchmark regression
regression: compile $(REPORT_DIR) $(DATA_DIR)
	@echo "Running comprehensive benchmark regression..."
	@for suite in $(BENCHMARK_SUITES); do \
		echo "Running benchmark suite: $$suite"; \
		$(MAKE) BENCHMARK_SUITE=$$suite run; \
		if [ $$? -ne 0 ]; then \
			echo "Benchmark suite $$suite FAILED"; \
			exit 1; \
		fi; \
	done
	@echo "Benchmark regression complete. All suites passed."

# Generate benchmark reports
reports: $(REPORT_DIR)
	@echo "Generating benchmark analysis reports..."
	@if [ -f $(REPORT_DIR)/benchmark_results.csv ]; then \
		echo "Analyzing benchmark results..."; \
		python3 scripts/analyze_benchmarks.py $(REPORT_DIR)/benchmark_results.csv $(REPORT_DIR)/; \
	else \
		echo "No benchmark results found. Run benchmarks first."; \
	fi

# Performance analysis
analyze: $(REPORT_DIR)
	@echo "Performing detailed performance analysis..."
	@if [ -f $(REPORT_DIR)/benchmark_results.csv ]; then \
		python3 scripts/performance_analysis.py $(REPORT_DIR)/benchmark_results.csv; \
	else \
		echo "No benchmark results found. Run benchmarks first."; \
	fi

# Compare with baseline
compare: $(REPORT_DIR)
	@echo "Comparing with baseline results..."
	@if [ -f $(DATA_DIR)/baseline_results.csv ] && [ -f $(REPORT_DIR)/benchmark_results.csv ]; then \
		python3 scripts/compare_results.py $(DATA_DIR)/baseline_results.csv $(REPORT_DIR)/benchmark_results.csv; \
	else \
		echo "Baseline or current results not found."; \
	fi

# Set current results as baseline
set_baseline: $(DATA_DIR)
	@if [ -f $(REPORT_DIR)/benchmark_results.csv ]; then \
		cp $(REPORT_DIR)/benchmark_results.csv $(DATA_DIR)/baseline_results.csv; \
		echo "Current results set as baseline."; \
	else \
		echo "No current results found. Run benchmarks first."; \
	fi

# Quick smoke test
smoke:
	$(MAKE) BENCHMARK_SUITE=mlperf_inference VERBOSITY=UVM_LOW run

# Performance test with detailed analysis
perf:
	$(MAKE) BENCHMARK_SUITE=all_benchmarks VERBOSITY=UVM_LOW run
	$(MAKE) analyze

# Debug with waves
debug:
	$(MAKE) BENCHMARK_SUITE=$(BENCHMARK_SUITE) WAVES=1 VERBOSITY=UVM_HIGH run

# Clean build artifacts
clean:
	rm -rf $(WORK_DIR) $(LOG_DIR) $(REPORT_DIR)
	rm -f transcript vsim.wlf *.log
	rm -f ucli.key vc_hdrs.h
	rm -rf csrc simv* *.daidir
	rm -rf xcelium.d xmsim.key waves.shm

# Create benchmark data directory structure
setup_data: $(DATA_DIR)
	mkdir -p $(DATA_DIR)/models
	mkdir -p $(DATA_DIR)/datasets
	mkdir -p $(DATA_DIR)/baselines
	@echo "Data directory structure created."

# Download sample datasets (placeholder)
download_data: setup_data
	@echo "Downloading sample benchmark datasets..."
	@echo "Note: In a real implementation, this would download actual datasets"
	@echo "Creating placeholder dataset files..."
	touch $(DATA_DIR)/datasets/imagenet_sample.txt
	touch $(DATA_DIR)/datasets/coco_sample.txt
	touch $(DATA_DIR)/datasets/squad_sample.txt
	touch $(DATA_DIR)/datasets/movielens_sample.txt

# Validate benchmark setup
validate:
	@echo "Validating benchmark setup..."
	@echo "Checking directories..."
	@test -d $(BENCH_DIR) || (echo "Benchmark directory missing" && exit 1)
	@test -d $(UVM_DIR) || (echo "UVM directory missing" && exit 1)
	@test -d $(RTL_DIR) || (echo "RTL directory missing" && exit 1)
	@echo "Checking source files..."
	@test -f $(BENCH_DIR)/ai_benchmark_pkg.sv || (echo "Benchmark package missing" && exit 1)
	@test -f $(UVM_DIR)/riscv_ai_pkg.sv || (echo "UVM package missing" && exit 1)
	@echo "Checking simulator..."
	@which $(VLOG) > /dev/null || (echo "Simulator not found: $(VLOG)" && exit 1)
	@echo "Benchmark setup validation passed."

# Help
help:
	@echo "AI Benchmark Testing Makefile"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Compile benchmark framework (default)"
	@echo "  compile          - Compile RTL, UVM, and benchmark sources"
	@echo "  run              - Run benchmark suite (specify with BENCHMARK_SUITE=)"
	@echo "  mlperf           - Run MLPerf inference benchmarks"
	@echo "  image_classification - Run image classification benchmarks"
	@echo "  object_detection - Run object detection benchmarks"
	@echo "  nlp              - Run NLP benchmarks"
	@echo "  recommendation   - Run recommendation system benchmarks"
	@echo "  all_benchmarks   - Run all benchmark suites"
	@echo "  regression       - Run comprehensive benchmark regression"
	@echo "  reports          - Generate benchmark analysis reports"
	@echo "  analyze          - Perform detailed performance analysis"
	@echo "  compare          - Compare with baseline results"
	@echo "  set_baseline     - Set current results as baseline"
	@echo "  smoke            - Run quick smoke test"
	@echo "  perf             - Run performance test with analysis"
	@echo "  debug            - Run with waveforms and high verbosity"
	@echo "  setup_data       - Create data directory structure"
	@echo "  download_data    - Download sample datasets"
	@echo "  validate         - Validate benchmark setup"
	@echo "  clean            - Clean build artifacts"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  SIMULATOR={questa|vcs}     - Choose simulator (default: questa)"
	@echo "  BENCHMARK_SUITE=suite_name - Specify benchmark suite to run"
	@echo "  WAVES={0|1}                - Enable waveform generation (default: 0)"
	@echo "  COVERAGE={0|1}             - Enable coverage collection (default: 0)"
	@echo "  VERBOSITY={UVM_LOW|UVM_MEDIUM|UVM_HIGH} - Set UVM verbosity (default: UVM_MEDIUM)"
	@echo ""
	@echo "Available Benchmark Suites:"
	@for suite in $(BENCHMARK_SUITES); do echo "  $$suite"; done
	@echo ""
	@echo "Examples:"
	@echo "  make compile                    - Compile everything"
	@echo "  make mlperf                     - Run MLPerf benchmarks"
	@echo "  make run BENCHMARK_SUITE=nlp_benchmarks - Run NLP benchmarks"
	@echo "  make regression                 - Run all benchmark suites"
	@echo "  make debug BENCHMARK_SUITE=image_classification - Debug image classification"
	@echo "  make compare                    - Compare with baseline"

# Print configuration
config:
	@echo "Configuration:"
	@echo "  SIMULATOR: $(SIMULATOR)"
	@echo "  BENCHMARK_SUITE: $(BENCHMARK_SUITE)"
	@echo "  WAVES: $(WAVES)"
	@echo "  COVERAGE: $(COVERAGE)"
	@echo "  VERBOSITY: $(VERBOSITY)"
	@echo "  UVM_VERSION: $(UVM_VERSION)"