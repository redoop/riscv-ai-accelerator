# Makefile for TPU Integration Tests

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -I../drivers -I../lib
LIBS = -lm -lpthread

# Source files
DRIVER_SOURCES = ../drivers/ai_accel_driver.c ../drivers/tpu_interface.c
LIB_SOURCES = ../lib/libtpu.c
TEST_SOURCES = test_tpu_integration.c

# Object files
DRIVER_OBJECTS = $(DRIVER_SOURCES:.c=.o)
LIB_OBJECTS = $(LIB_SOURCES:.c=.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

# Executables
TEST_EXECUTABLE = test_tpu_integration

# Default target
all: $(TEST_EXECUTABLE)

# Build test executable
$(TEST_EXECUTABLE): $(DRIVER_OBJECTS) $(LIB_OBJECTS) $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "Built TPU integration test executable"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Run tests
test: $(TEST_EXECUTABLE)
	@echo "Running TPU integration tests..."
	./$(TEST_EXECUTABLE)

# Run tests with valgrind for memory checking
test-valgrind: $(TEST_EXECUTABLE)
	@echo "Running TPU integration tests with valgrind..."
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_EXECUTABLE)

# Clean build artifacts
clean:
	rm -f $(DRIVER_OBJECTS) $(LIB_OBJECTS) $(TEST_OBJECTS) $(TEST_EXECUTABLE)
	@echo "Cleaned build artifacts"

# Create mock device file for testing (requires root)
setup-mock-device:
	@echo "Setting up mock TPU device..."
	@if [ ! -c /dev/ai_accel ]; then \
		echo "Creating mock device file (requires root)"; \
		sudo mknod /dev/ai_accel c 240 0; \
		sudo chmod 666 /dev/ai_accel; \
	fi

# Remove mock device file
cleanup-mock-device:
	@echo "Cleaning up mock TPU device..."
	@if [ -c /dev/ai_accel ]; then \
		sudo rm -f /dev/ai_accel; \
	fi

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing test dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential valgrind

# Static analysis with cppcheck
analyze:
	@echo "Running static analysis..."
	cppcheck --enable=all --std=c99 $(DRIVER_SOURCES) $(LIB_SOURCES) $(TEST_SOURCES)

# Format code
format:
	@echo "Formatting code..."
	clang-format -i $(DRIVER_SOURCES) $(LIB_SOURCES) $(TEST_SOURCES)

# Generate documentation
docs:
	@echo "Generating documentation..."
	doxygen Doxyfile

# Help target
help:
	@echo "Available targets:"
	@echo "  all                 - Build all executables"
	@echo "  test                - Run integration tests"
	@echo "  test-valgrind       - Run tests with memory checking"
	@echo "  clean               - Clean build artifacts"
	@echo "  setup-mock-device   - Create mock device file (requires root)"
	@echo "  cleanup-mock-device - Remove mock device file"
	@echo "  install-deps        - Install build dependencies"
	@echo "  analyze             - Run static analysis"
	@echo "  format              - Format source code"
	@echo "  docs                - Generate documentation"
	@echo "  help                - Show this help message"

.PHONY: all test test-valgrind clean setup-mock-device cleanup-mock-device install-deps analyze format docs help