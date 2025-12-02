CROSS = riscv64-unknown-elf-
CC = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
OBJDUMP = $(CROSS)objdump

CFLAGS = -march=rv32i -mabi=ilp32 -nostdlib -O2
LDFLAGS = -T link.ld

all: test.hex

test.elf: start.S test.c link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ start.S test.c

test.bin: test.elf
	$(OBJCOPY) -O binary $< $@

test.hex: test.elf
	$(OBJCOPY) -O verilog $< $@

test.dis: test.elf
	$(OBJDUMP) -d $< > $@

clean:
	rm -f test.elf test.bin test.hex test.dis

.PHONY: all clean
