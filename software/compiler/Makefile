# RISC-V AI Compiler Support Makefile
# Builds and tests GCC integration for AI instructions

# Compiler configuration
CC = riscv64-unknown-linux-gnu-gcc
CXX = riscv64-unknown-linux-gnu-g++
AS = riscv64-unknown-linux-gnu-as
LD = riscv64-unknown-linux-gnu-ld
OBJDUMP = riscv64-unknown-linux-gnu-objdump

# Compiler flags
CFLAGS = -march=rv64imafdv -mabi=lp64d -mai -O2 -g
CXXFLAGS = $(CFLAGS) -std=c++17
LDFLAGS = -static
INCLUDES = -I. -I../lib -I../drivers

# Test configuration
TEST_CFLAGS = $(CFLAGS) -DTEST_MODE -Wall -Wextra
TEST_LDFLAGS = $(LDFLAGS) -lm

# Source files
INTRINSIC_HEADERS = riscv_ai_intrinsics.h
PATCH_FILES = gcc-patches/riscv-ai-extension.patch \
              gcc-patches/riscv-ai.md \
              gcc-patches/riscv-ai-constants.h

TEST_SOURCES = tests/test_ai_intrinsics.c
TEST_TARGETS = $(TEST_SOURCES:.c=)

# Build targets
.PHONY: all clean test install patch-gcc check-compiler benchmark

all: test

# Test compilation and execution
test: $(TEST_TARGETS)
	@echo "Running AI intrinsics tests..."
	@for test in $(TEST_TARGETS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "All tests passed!"

# Build test executables
tests/test_ai_intrinsics: tests/test_ai_intrinsics.c $(INTRINSIC_HEADERS)
	@echo "Building AI intrinsics test..."
	$(CC) $(TEST_CFLAGS) $(INCLUDES) -o $@ $< $(TEST_LDFLAGS)

# Check compiler support
check-compiler:
	@echo "Checking RISC-V AI compiler support..."
	@$(CC) --version
	@echo "Checking AI extension support..."
	@$(CC) -march=rv64imafdv -mai -dM -E - < /dev/null | grep -i ai || echo "AI extension not found"
	@echo "Checking available intrinsics..."
	@echo '#include "riscv_ai_intrinsics.h"' | $(CC) $(CFLAGS) $(INCLUDES) -E - > /dev/null && echo "Intrinsics header OK" || echo "Intrinsics header ERROR"

# Generate assembly output for inspection
assembly: tests/test_ai_intrinsics.c
	@echo "Generating assembly output..."
	$(CC) $(CFLAGS) $(INCLUDES) -S -o tests/test_ai_intrinsics.s $<
	@echo "Assembly saved to tests/test_ai_intrinsics.s"

# Disassemble compiled binary
disasm: tests/test_ai_intrinsics
	@echo "Disassembling compiled binary..."
	$(OBJDUMP) -d $< > tests/test_ai_intrinsics.disasm
	@echo "Disassembly saved to tests/test_ai_intrinsics.disasm"

# Performance benchmark
benchmark: tests/test_ai_intrinsics
	@echo "Running performance benchmarks..."
	./tests/test_ai_intrinsics | grep -E "(GFLOPS|elem/s|ms)"

# Install intrinsics header
install: $(INTRINSIC_HEADERS)
	@echo "Installing AI intrinsics header..."
	@mkdir -p $(DESTDIR)/usr/include/riscv
	@cp $(INTRINSIC_HEADERS) $(DESTDIR)/usr/include/riscv/
	@echo "Header installed to $(DESTDIR)/usr/include/riscv/"

# Apply GCC patches (requires GCC source tree)
patch-gcc:
	@if [ -z "$(GCC_SOURCE_DIR)" ]; then \
		echo "Error: GCC_SOURCE_DIR not set"; \
		echo "Usage: make patch-gcc GCC_SOURCE_DIR=/path/to/gcc/source"; \
		exit 1; \
	fi
	@echo "Applying AI extension patches to GCC..."
	@cd $(GCC_SOURCE_DIR) && patch -p1 < $(PWD)/gcc-patches/riscv-ai-extension.patch
	@cp gcc-patches/riscv-ai.md $(GCC_SOURCE_DIR)/gcc/config/riscv/
	@cp gcc-patches/riscv-ai-constants.h $(GCC_SOURCE_DIR)/gcc/config/riscv/
	@echo "Patches applied successfully!"
	@echo "Now rebuild GCC with: cd $(GCC_SOURCE_DIR) && make"

# Code quality checks
lint:
	@echo "Running code quality checks..."
	@which cppcheck > /dev/null && cppcheck --enable=all --std=c99 tests/ || echo "cppcheck not found"
	@which clang-tidy > /dev/null && clang-tidy tests/*.c -- $(CFLAGS) $(INCLUDES) || echo "clang-tidy not found"

# Memory leak detection (requires valgrind)
memcheck: tests/test_ai_intrinsics
	@echo "Running memory leak detection..."
	@which valgrind > /dev/null && valgrind --leak-check=full --error-exitcode=1 ./tests/test_ai_intrinsics || echo "valgrind not found"

# Generate documentation
docs:
	@echo "Generating documentation..."
	@which doxygen > /dev/null && doxygen Doxyfile || echo "doxygen not found"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(TEST_TARGETS)
	@rm -f tests/*.s tests/*.disasm tests/*.o
	@rm -rf docs/html docs/latex

# Help target
help:
	@echo "RISC-V AI Compiler Support Makefile"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all tests (default)"
	@echo "  test         - Build and run all tests"
	@echo "  check-compiler - Check compiler AI support"
	@echo "  assembly     - Generate assembly output"
	@echo "  disasm       - Disassemble compiled binary"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  install      - Install intrinsics header"
	@echo "  patch-gcc    - Apply patches to GCC source"
	@echo "  lint         - Run code quality checks"
	@echo "  memcheck     - Run memory leak detection"
	@echo "  docs         - Generate documentation"
	@echo "  clean        - Clean build artifacts"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CC           - C compiler ($(CC))"
	@echo "  CFLAGS       - C compiler flags ($(CFLAGS))"
	@echo "  GCC_SOURCE_DIR - GCC source directory for patching"
	@echo ""
	@echo "Examples:"
	@echo "  make test                    # Run all tests"
	@echo "  make benchmark              # Run performance tests"
	@echo "  make patch-gcc GCC_SOURCE_DIR=/path/to/gcc"
	@echo "  make install DESTDIR=/usr/local"

# Continuous integration target
ci: check-compiler test benchmark lint
	@echo "Continuous integration checks passed!"

# Development target with verbose output
dev: CFLAGS += -DDEBUG -v
dev: test assembly disasm

# Cross-compilation support
cross-%:
	$(MAKE) CC=riscv64-unknown-elf-gcc LD=riscv64-unknown-elf-ld $*

# Dependency tracking
-include tests/*.d

# Pattern rule for dependency generation
tests/%.d: tests/%.c
	@$(CC) $(CFLAGS) $(INCLUDES) -MM -MT $(@:.d=) $< > $@