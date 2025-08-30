# Makefile for RISC-V Core Unit Tests
# Supports multiple simulators: Verilator, ModelSim, VCS

# Default simulator
SIM ?= verilator

# Source directories
RTL_DIR = ../../rtl
CORE_DIR = $(RTL_DIR)/core
CONFIG_DIR = $(RTL_DIR)/config
INTERFACE_DIR = $(RTL_DIR)/interfaces

# Test files
TESTS = test_riscv_alu test_riscv_control_unit test_riscv_core_basic \
        test_riscv_mdu test_riscv_fpu test_riscv_vector test_riscv_extensions \
        test_riscv_ai_instructions test_ai_integration test_l1_cache test_l2_l3_cache \
        test_memory_subsystem test_vector_alu test_vpu_functional_units test_vpu_instruction_pipeline

# Common source files
COMMON_SRCS = $(CONFIG_DIR)/chip_config.sv \
              $(INTERFACE_DIR)/system_interfaces.sv \
              $(CORE_DIR)/riscv_alu.sv \
              $(CORE_DIR)/riscv_control_unit.sv \
              $(CORE_DIR)/riscv_branch_unit.sv \
              $(CORE_DIR)/riscv_forwarding_unit.sv \
              $(CORE_DIR)/riscv_hazard_unit.sv \
              $(CORE_DIR)/riscv_mdu.sv \
              $(CORE_DIR)/riscv_fpu.sv \
              $(CORE_DIR)/riscv_vector_unit.sv \
              $(CORE_DIR)/riscv_ai_unit.sv \
              $(CORE_DIR)/ai_matmul_unit.sv \
              $(CORE_DIR)/ai_conv2d_unit.sv \
              $(CORE_DIR)/ai_activation_unit.sv \
              $(CORE_DIR)/ai_pooling_unit.sv \
              $(CORE_DIR)/ai_batchnorm_unit.sv \
              $(CORE_DIR)/riscv_core.sv

# Memory subsystem source files
MEMORY_DIR = $(RTL_DIR)/memory
MEMORY_SRCS = $(MEMORY_DIR)/cache_controller.sv \
              $(MEMORY_DIR)/l1_icache.sv \
              $(MEMORY_DIR)/l1_dcache.sv \
              $(MEMORY_DIR)/l2_cache.sv \
              $(MEMORY_DIR)/l3_cache.sv \
              $(MEMORY_DIR)/hbm_controller.sv \
              $(MEMORY_DIR)/memory_subsystem.sv

# Accelerator source files
ACCEL_DIR = $(RTL_DIR)/accelerators
ACCEL_SRCS = $(ACCEL_DIR)/vpu.sv \
             $(ACCEL_DIR)/vector_alu.sv \
             $(ACCEL_DIR)/vpu_instruction_pipeline.sv \
             $(ACCEL_DIR)/tpu.sv \
             $(ACCEL_DIR)/tpu_compute_array.sv \
             $(ACCEL_DIR)/tpu_mac_unit.sv \
             $(ACCEL_DIR)/tpu_systolic_array.sv

# Verilator settings
VERILATOR_FLAGS = --cc --exe --build --trace --timing -I$(CONFIG_DIR) -I$(INTERFACE_DIR)
VERILATOR_CFLAGS = -CFLAGS "-std=c++14 -Wall"

# ModelSim settings
MODELSIM_FLAGS = -work work +incdir+$(CONFIG_DIR) +incdir+$(INTERFACE_DIR)

# VCS settings
VCS_FLAGS = +incdir+$(CONFIG_DIR) +incdir+$(INTERFACE_DIR) -sverilog

.PHONY: all clean test_% run_all_tests

all: run_all_tests

# Run all tests
run_all_tests: $(TESTS)
	@echo "All unit tests completed"

# Individual test targets
test_riscv_alu: test_riscv_alu.sv $(COMMON_SRCS)
	@echo "Running ALU unit test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_alu \
		test_riscv_alu.sv $(CORE_DIR)/riscv_alu.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_alu
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_alu.sv $(CORE_DIR)/riscv_alu.sv
	vsim -c -do "run -all; quit" test_riscv_alu
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_alu_sim test_riscv_alu.sv $(CORE_DIR)/riscv_alu.sv
	./test_riscv_alu_sim
endif

test_riscv_control_unit: test_riscv_control_unit.sv $(COMMON_SRCS)
	@echo "Running Control Unit test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_control_unit \
		test_riscv_control_unit.sv $(CORE_DIR)/riscv_control_unit.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_control_unit
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_control_unit.sv $(CORE_DIR)/riscv_control_unit.sv
	vsim -c -do "run -all; quit" test_riscv_control_unit
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_control_unit_sim test_riscv_control_unit.sv $(CORE_DIR)/riscv_control_unit.sv
	./test_riscv_control_unit_sim
endif

test_riscv_core_basic: test_riscv_core_basic.sv $(COMMON_SRCS)
	@echo "Running Core Integration test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_core_basic \
		test_riscv_core_basic.sv $(COMMON_SRCS) \
		--exe /dev/null
	./obj_dir/Vtest_riscv_core_basic
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_core_basic.sv $(COMMON_SRCS)
	vsim -c -do "run -all; quit" test_riscv_core_basic
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_core_basic_sim test_riscv_core_basic.sv $(COMMON_SRCS)
	./test_riscv_core_basic_sim
endif

test_riscv_mdu: test_riscv_mdu.sv $(COMMON_SRCS)
	@echo "Running MDU (M Extension) test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_mdu \
		test_riscv_mdu.sv $(CORE_DIR)/riscv_mdu.sv $(CONFIG_DIR)/chip_config.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_mdu
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_mdu.sv $(CORE_DIR)/riscv_mdu.sv
	vsim -c -do "run -all; quit" test_riscv_mdu
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_mdu_sim test_riscv_mdu.sv $(CORE_DIR)/riscv_mdu.sv
	./test_riscv_mdu_sim
endif

test_riscv_fpu: test_riscv_fpu.sv $(COMMON_SRCS)
	@echo "Running FPU (F/D Extension) test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_fpu \
		test_riscv_fpu.sv $(CORE_DIR)/riscv_fpu.sv $(CONFIG_DIR)/chip_config.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_fpu
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_fpu.sv $(CORE_DIR)/riscv_fpu.sv
	vsim -c -do "run -all; quit" test_riscv_fpu
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_fpu_sim test_riscv_fpu.sv $(CORE_DIR)/riscv_fpu.sv
	./test_riscv_fpu_sim
endif

test_riscv_vector: test_riscv_vector.sv $(COMMON_SRCS)
	@echo "Running Vector Unit (V Extension) test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_vector \
		test_riscv_vector.sv $(CORE_DIR)/riscv_vector_unit.sv $(CONFIG_DIR)/chip_config.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_vector
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_vector.sv $(CORE_DIR)/riscv_vector_unit.sv
	vsim -c -do "run -all; quit" test_riscv_vector
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_vector_sim test_riscv_vector.sv $(CORE_DIR)/riscv_vector_unit.sv
	./test_riscv_vector_sim
endif

test_riscv_extensions: test_riscv_extensions.sv $(COMMON_SRCS)
	@echo "Running Extensions Integration test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_extensions \
		test_riscv_extensions.sv $(COMMON_SRCS) \
		--exe /dev/null
	./obj_dir/Vtest_riscv_extensions
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_extensions.sv $(COMMON_SRCS)
	vsim -c -do "run -all; quit" test_riscv_extensions
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_extensions_sim test_riscv_extensions.sv $(COMMON_SRCS)
	./test_riscv_extensions_sim
endif

test_riscv_ai_instructions: test_riscv_ai_instructions.sv $(COMMON_SRCS)
	@echo "Running AI Instructions Unit test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_riscv_ai_instructions \
		test_riscv_ai_instructions.sv \
		$(CORE_DIR)/riscv_ai_unit.sv \
		$(CORE_DIR)/ai_matmul_unit.sv \
		$(CORE_DIR)/ai_conv2d_unit.sv \
		$(CORE_DIR)/ai_activation_unit.sv \
		$(CORE_DIR)/ai_pooling_unit.sv \
		$(CORE_DIR)/ai_batchnorm_unit.sv \
		$(CONFIG_DIR)/chip_config.sv \
		--exe /dev/null
	./obj_dir/Vtest_riscv_ai_instructions
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_riscv_ai_instructions.sv \
		$(CORE_DIR)/riscv_ai_unit.sv \
		$(CORE_DIR)/ai_matmul_unit.sv \
		$(CORE_DIR)/ai_conv2d_unit.sv \
		$(CORE_DIR)/ai_activation_unit.sv \
		$(CORE_DIR)/ai_pooling_unit.sv \
		$(CORE_DIR)/ai_batchnorm_unit.sv
	vsim -c -do "run -all; quit" test_riscv_ai_instructions
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_riscv_ai_instructions_sim test_riscv_ai_instructions.sv \
		$(CORE_DIR)/riscv_ai_unit.sv \
		$(CORE_DIR)/ai_matmul_unit.sv \
		$(CORE_DIR)/ai_conv2d_unit.sv \
		$(CORE_DIR)/ai_activation_unit.sv \
		$(CORE_DIR)/ai_pooling_unit.sv \
		$(CORE_DIR)/ai_batchnorm_unit.sv
	./test_riscv_ai_instructions_sim
endif

test_ai_integration: test_ai_integration.sv $(COMMON_SRCS)
	@echo "Running AI Integration test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_ai_integration \
		test_ai_integration.sv $(COMMON_SRCS) \
		--exe /dev/null
	./obj_dir/Vtest_ai_integration
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_ai_integration.sv $(COMMON_SRCS)
	vsim -c -do "run -all; quit" test_ai_integration
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_ai_integration_sim test_ai_integration.sv $(COMMON_SRCS)
	./test_ai_integration_sim
endif

test_l1_cache: test_l1_cache.sv $(MEMORY_SRCS)
	@echo "Running L1 Cache test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_l1_cache \
		test_l1_cache.sv $(MEMORY_SRCS) $(INTERFACE_DIR)/system_interfaces.sv \
		--exe /dev/null
	./obj_dir/Vtest_l1_cache
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_l1_cache.sv $(MEMORY_SRCS)
	vsim -c -do "run -all; quit" test_l1_cache
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_l1_cache_sim test_l1_cache.sv $(MEMORY_SRCS)
	./test_l1_cache_sim
endif

test_l2_l3_cache: test_l2_l3_cache.sv $(MEMORY_SRCS)
	@echo "Running L2/L3 Cache test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_l2_l3_cache \
		test_l2_l3_cache.sv $(MEMORY_SRCS) $(INTERFACE_DIR)/system_interfaces.sv \
		--exe /dev/null
	./obj_dir/Vtest_l2_l3_cache
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_l2_l3_cache.sv $(MEMORY_SRCS)
	vsim -c -do "run -all; quit" test_l2_l3_cache
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_l2_l3_cache_sim test_l2_l3_cache.sv $(MEMORY_SRCS)
	./test_l2_l3_cache_sim
endif

test_memory_subsystem: test_memory_subsystem.sv $(MEMORY_SRCS)
	@echo "Running Memory Subsystem test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_memory_subsystem \
		test_memory_subsystem.sv $(MEMORY_SRCS) $(INTERFACE_DIR)/system_interfaces.sv \
		--exe /dev/null
	./obj_dir/Vtest_memory_subsystem
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_memory_subsystem.sv $(MEMORY_SRCS)
	vsim -c -do "run -all; quit" test_memory_subsystem
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_memory_subsystem_sim test_memory_subsystem.sv $(MEMORY_SRCS)
	./test_memory_subsystem_sim
endif

test_vector_alu: test_vector_alu.sv $(ACCEL_SRCS)
	@echo "Running Vector ALU test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_vector_alu \
		test_vector_alu.sv $(ACCEL_DIR)/vector_alu.sv \
		--exe /dev/null
	./obj_dir/Vtest_vector_alu
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_vector_alu.sv $(ACCEL_DIR)/vector_alu.sv
	vsim -c -do "run -all; quit" test_vector_alu
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_vector_alu_sim test_vector_alu.sv $(ACCEL_DIR)/vector_alu.sv
	./test_vector_alu_sim
endif

test_vpu_functional_units: test_vpu_functional_units.sv $(ACCEL_SRCS)
	@echo "Running VPU Functional Units test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_vpu_functional_units \
		test_vpu_functional_units.sv $(ACCEL_DIR)/vpu.sv $(ACCEL_DIR)/vector_alu.sv \
		$(CONFIG_DIR)/chip_config.sv $(INTERFACE_DIR)/system_interfaces.sv \
		--exe /dev/null
	./obj_dir/Vtest_vpu_functional_units
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_vpu_functional_units.sv $(ACCEL_DIR)/vpu.sv $(ACCEL_DIR)/vector_alu.sv
	vsim -c -do "run -all; quit" test_vpu_functional_units
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_vpu_functional_units_sim test_vpu_functional_units.sv \
		$(ACCEL_DIR)/vpu.sv $(ACCEL_DIR)/vector_alu.sv
	./test_vpu_functional_units_sim
endif

test_vpu_instruction_pipeline: test_vpu_instruction_pipeline.sv $(ACCEL_SRCS)
	@echo "Running VPU Instruction Pipeline test with $(SIM)"
ifeq ($(SIM),verilator)
	verilator $(VERILATOR_FLAGS) $(VERILATOR_CFLAGS) \
		--top-module test_vpu_instruction_pipeline \
		test_vpu_instruction_pipeline.sv $(ACCEL_DIR)/vpu_instruction_pipeline.sv \
		$(CONFIG_DIR)/chip_config.sv \
		--exe /dev/null
	./obj_dir/Vtest_vpu_instruction_pipeline
else ifeq ($(SIM),modelsim)
	vlog $(MODELSIM_FLAGS) test_vpu_instruction_pipeline.sv $(ACCEL_DIR)/vpu_instruction_pipeline.sv
	vsim -c -do "run -all; quit" test_vpu_instruction_pipeline
else ifeq ($(SIM),vcs)
	vcs $(VCS_FLAGS) -o test_vpu_instruction_pipeline_sim test_vpu_instruction_pipeline.sv \
		$(ACCEL_DIR)/vpu_instruction_pipeline.sv
	./test_vpu_instruction_pipeline_sim
endif

# Clean build artifacts
clean:
	rm -rf obj_dir/
	rm -rf work/
	rm -rf *.log *.wlf *.vstf
	rm -rf *_sim
	rm -rf csrc/
	rm -rf simv*
	rm -rf ucli.key
	rm -rf *.vpd

# Help target
help:
	@echo "Available targets:"
	@echo "  all                     - Run all unit tests"
	@echo "  test_riscv_alu          - Run ALU unit test"
	@echo "  test_riscv_control_unit - Run Control Unit test"
	@echo "  test_riscv_core_basic   - Run Core Integration test"
	@echo "  test_riscv_mdu          - Run MDU (M Extension) test"
	@echo "  test_riscv_fpu          - Run FPU (F/D Extension) test"
	@echo "  test_riscv_vector       - Run Vector Unit (V Extension) test"
	@echo "  test_riscv_extensions   - Run Extensions Integration test"
	@echo "  test_riscv_ai_instructions - Run AI Instructions Unit test"
	@echo "  test_ai_integration     - Run AI Integration test"
	@echo "  test_l1_cache           - Run L1 Cache test"
	@echo "  test_l2_l3_cache        - Run L2/L3 Cache test"
	@echo "  test_memory_subsystem   - Run Memory Subsystem test"
	@echo "  test_vector_alu         - Run Vector ALU test"
	@echo "  test_vpu_functional_units - Run VPU Functional Units test"
	@echo "  test_vpu_instruction_pipeline - Run VPU Instruction Pipeline test"
	@echo "  clean                   - Clean build artifacts"
	@echo ""
	@echo "Simulator selection (default: verilator):"
	@echo "  make SIM=verilator  - Use Verilator"
	@echo "  make SIM=modelsim   - Use ModelSim"
	@echo "  make SIM=vcs        - Use Synopsys VCS"