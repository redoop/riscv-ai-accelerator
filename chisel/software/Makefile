# Makefile for RISC-V AI SoC Software
# Requires RISC-V GCC toolchain

# Toolchain
# Default to riscv64-elf (Homebrew on macOS)
# Override with: make PREFIX=riscv32-unknown-elf-
PREFIX ?= riscv64-elf-
CC = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
SIZE = $(PREFIX)size

# Compiler flags
ARCH = rv32i
ABI = ilp32
CFLAGS = -march=$(ARCH) -mabi=$(ABI) -O2 -g
CFLAGS += -Wall -Wextra
CFLAGS += -nostdlib -nostartfiles
CFLAGS += -ffunction-sections -fdata-sections

# Linker flags
LDFLAGS = -T linker.ld
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$@.map
# Link libgcc for mul/div support
LIBGCC = $(shell $(CC) $(CFLAGS) -print-libgcc-file-name)
LDFLAGS += -L$(dir $(LIBGCC)) -lgcc

# Directories
LIB_DIR = lib
BOOT_DIR = bootloader
EXAMPLES_DIR = examples
BUILD_DIR = build

# Library sources
LIB_SRCS = $(LIB_DIR)/hal.c \
           $(LIB_DIR)/graphics.c \
           $(LIB_DIR)/font_8x8.c

# Startup code
STARTUP = $(LIB_DIR)/start.S

# Library objects
LIB_OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
STARTUP_OBJ = $(BUILD_DIR)/$(LIB_DIR)/start.o

# Examples
EXAMPLES = hello_lcd ai_demo benchmark system_monitor

# Bootloader
BOOTLOADER = bootloader

# Default target
all: $(EXAMPLES) $(BOOTLOADER)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)/$(LIB_DIR)
	@mkdir -p $(BUILD_DIR)/$(BOOT_DIR)
	@mkdir -p $(BUILD_DIR)/$(EXAMPLES_DIR)

# Compile library
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "CC $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Compile startup code
$(STARTUP_OBJ): $(STARTUP) | $(BUILD_DIR)
	@echo "AS $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Build examples
$(EXAMPLES): %: $(EXAMPLES_DIR)/%.c $(STARTUP_OBJ) $(LIB_OBJS)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/$@.elf \
		$(STARTUP_OBJ) $(EXAMPLES_DIR)/$@.c $(LIB_OBJS) \
		$(LDFLAGS) $(LIBGCC)
	@$(OBJCOPY) -O binary $(BUILD_DIR)/$@.elf $(BUILD_DIR)/$@.bin
	@$(SIZE) $(BUILD_DIR)/$@.elf
	@echo "Created $(BUILD_DIR)/$@.bin"

# Build bootloader
$(BOOTLOADER): $(BOOT_DIR)/$(BOOTLOADER).c $(STARTUP_OBJ) $(LIB_OBJS)
	@echo "Building $@..."
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/$@.elf \
		$(STARTUP_OBJ) $(BOOT_DIR)/$@.c $(LIB_OBJS) \
		$(LDFLAGS) $(LIBGCC)
	@$(OBJCOPY) -O binary $(BUILD_DIR)/$@.elf $(BUILD_DIR)/$@.bin
	@$(SIZE) $(BUILD_DIR)/$@.elf
	@echo "Created $(BUILD_DIR)/$@.bin"

# Generate disassembly
disasm: $(EXAMPLES) $(BOOTLOADER)
	@for ex in $(EXAMPLES) $(BOOTLOADER); do \
		echo "Disassembling $$ex..."; \
		$(OBJDUMP) -d $(BUILD_DIR)/$$ex.elf > $(BUILD_DIR)/$$ex.dis; \
	done

# Upload program
upload: $(BUILD_DIR)/$(PROG).bin
	@echo "Uploading $(PROG)..."
	@python3 tools/upload.py $(PORT) $(BUILD_DIR)/$(PROG).bin --run

# Upload and run specific program
# Usage: make run PROG=hello_lcd PORT=/dev/ttyUSB0
run: $(BUILD_DIR)/$(PROG).bin
	@echo "Uploading and running $(PROG)..."
	@python3 tools/upload.py $(PORT) $(BUILD_DIR)/$(PROG).bin --run

# Test LCD
test-lcd:
	@echo "Running LCD test..."
	@python3 tools/upload.py $(PORT) --test-lcd

# Get info
info:
	@echo "Getting bootloader info..."
	@python3 tools/upload.py $(PORT) --info

# Clean
clean:
	@echo "Cleaning..."
	@rm -rf $(BUILD_DIR)

# Help
help:
	@echo "RISC-V AI SoC Software Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all examples and bootloader"
	@echo "  hello_lcd    - Build hello_lcd example"
	@echo "  ai_demo      - Build ai_demo example"
	@echo "  benchmark    - Build benchmark example"
	@echo "  system_monitor - Build system_monitor example"
	@echo "  bootloader   - Build bootloader"
	@echo "  disasm       - Generate disassembly files"
	@echo "  clean        - Remove build directory"
	@echo ""
	@echo "Upload targets:"
	@echo "  run PROG=<name> PORT=<port> - Upload and run program"
	@echo "  test-lcd PORT=<port>         - Run LCD test"
	@echo "  info PORT=<port>             - Get bootloader info"
	@echo ""
	@echo "Examples:"
	@echo "  make all"
	@echo "  make hello_lcd"
	@echo "  make run PROG=hello_lcd PORT=/dev/ttyUSB0"
	@echo "  make test-lcd PORT=/dev/ttyUSB0"

.PHONY: all clean help disasm upload run test-lcd info $(EXAMPLES) $(BOOTLOADER)
