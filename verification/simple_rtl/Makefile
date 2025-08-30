# Simple RTL Test Makefile
# Tests basic RTL functionality without complex dependencies

# Simulation tool
SIM = verilator

# Test files
SIMPLE_TESTS = simple_mac_test.sv

# Compiler flags
VFLAGS = --cc --exe --build -Wall -Wno-UNUSED -Wno-UNOPTFLAT -Wno-EOFNEWLINE -Wno-PROCASSINIT
VFLAGS += --timing --trace --trace-depth 2

# Default target
all: test_mac

# Simple MAC test
test_mac: simple_mac_test.sv
	@echo "🔨 Building simple MAC RTL test..."
	$(SIM) $(VFLAGS) --top-module simple_mac_test simple_mac_test.sv --exe sim_main.cpp
	@echo "🚀 Running simple MAC RTL test..."
	./obj_dir/Vsimple_mac_test

# Create simple C++ main file
sim_main.cpp:
	@echo "📝 Creating simple simulation main..."

# Clean build artifacts
clean:
	rm -rf obj_dir/
	rm -f *.vcd *.fst
	rm -f sim_main.cpp

# Help target
help:
	@echo "Simple RTL Test Makefile"
	@echo "========================"
	@echo "Targets:"
	@echo "  all       - Run all tests"
	@echo "  test_mac  - Test simple MAC unit"
	@echo "  clean     - Clean build artifacts"
	@echo "  help      - Show this help"

.PHONY: all test_mac clean help