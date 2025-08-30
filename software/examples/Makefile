# Makefile for TPU Examples

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -I../drivers -I../lib
LIBS = -lm -lpthread

# Source files
DRIVER_SOURCES = ../drivers/ai_accel_driver.c ../drivers/tpu_interface.c
LIB_SOURCES = ../lib/libtpu.c
EXAMPLE_SOURCES = tpu_example.c

# Object files
DRIVER_OBJECTS = $(DRIVER_SOURCES:.c=.o)
LIB_OBJECTS = $(LIB_SOURCES:.c=.o)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:.c=.o)

# Executables
EXAMPLE_EXECUTABLE = tpu_example

# Default target
all: $(EXAMPLE_EXECUTABLE)

# Build example executable
$(EXAMPLE_EXECUTABLE): $(DRIVER_OBJECTS) $(LIB_OBJECTS) $(EXAMPLE_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "Built TPU example executable"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Run example
run: $(EXAMPLE_EXECUTABLE)
	@echo "Running TPU examples..."
	./$(EXAMPLE_EXECUTABLE)

# Clean build artifacts
clean:
	rm -f $(DRIVER_OBJECTS) $(LIB_OBJECTS) $(EXAMPLE_OBJECTS) $(EXAMPLE_EXECUTABLE)
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "Available targets:"
	@echo "  all   - Build example executable"
	@echo "  run   - Run TPU examples"
	@echo "  clean - Clean build artifacts"
	@echo "  help  - Show this help message"

.PHONY: all run clean help