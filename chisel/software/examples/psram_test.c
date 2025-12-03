#include "../lib/hal.h"

// Helper function to print hex
static void print_hex(uint32_t val) {
    const char hex[] = "0123456789ABCDEF";
    uart_putc('0');
    uart_putc('x');
    for (int i = 7; i >= 0; i--) {
        uart_putc(hex[(val >> (i * 4)) & 0xF]);
    }
}

// Helper function to print decimal
static void print_dec(uint32_t val) {
    if (val == 0) {
        uart_putc('0');
        return;
    }
    char buf[10];
    int i = 0;
    while (val > 0) {
        buf[i++] = '0' + (val % 10);
        val /= 10;
    }
    while (i > 0) {
        uart_putc(buf[--i]);
    }
}

void psram_test(void) {
    uart_puts("\n=== PSRAM Test ===\n");
    
    // Test 1: Initialize
    uart_puts("Test 1: Initialize PSRAM\n");
    psram_init();
    uart_puts("  Init complete\n");
    
    // Test 2: Write and read single word
    uart_puts("\nTest 2: Write 0xDEADBEEF to 0x001000\n");
    psram_write(0x001000, 0xDEADBEEF);
    uart_puts("  Write complete\n");
    
    uart_puts("Test 3: Read back from 0x001000\n");
    uint32_t data = psram_read(0x001000);
    uart_puts("  Data: ");
    print_hex(data);
    if (data == 0xDEADBEEF) {
        uart_puts(" [PASS]\n");
    } else {
        uart_puts(" [FAIL]\n");
    }
    
    // Test 4: Block write
    uart_puts("\nTest 4: Block write (16 bytes)\n");
    uint8_t write_buf[16] = {
        0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0x0C,
        0x0D, 0x0E, 0x0F, 0x10
    };
    psram_write_block(0x002000, write_buf, 16);
    uart_puts("  Block write complete\n");
    
    // Test 5: Block read
    uart_puts("\nTest 5: Block read (16 bytes)\n");
    uint8_t read_buf[16] = {0};
    psram_read_block(0x002000, read_buf, 16);
    uart_puts("  Data: ");
    bool block_pass = true;
    for (int i = 0; i < 16; i++) {
        if (read_buf[i] != write_buf[i]) {
            block_pass = false;
        }
    }
    if (block_pass) {
        uart_puts("[PASS]\n");
    } else {
        uart_puts("[FAIL]\n");
    }
    
    // Test 6: QPI mode
    uart_puts("\nTest 6: QPI mode test\n");
    uart_puts("  Current mode: ");
    if (psram_is_qpi_mode()) {
        uart_puts("QPI\n");
    } else {
        uart_puts("SPI\n");
    }
    
    uart_puts("  Enabling QPI mode...\n");
    psram_enable_qpi();
    if (psram_is_qpi_mode()) {
        uart_puts("  QPI mode enabled [PASS]\n");
    } else {
        uart_puts("  QPI mode failed [FAIL]\n");
    }
    
    uart_puts("  Disabling QPI mode...\n");
    psram_disable_qpi();
    if (!psram_is_qpi_mode()) {
        uart_puts("  QPI mode disabled [PASS]\n");
    } else {
        uart_puts("  QPI mode failed [FAIL]\n");
    }
    
    // Test 7: Performance test
    uart_puts("\nTest 7: Performance test\n");
    uart_puts("  Writing 1KB...\n");
    for (uint32_t i = 0; i < 256; i++) {
        psram_write(0x003000 + i * 4, i);
    }
    
    uart_puts("  Reading 1KB...\n");
    for (uint32_t i = 0; i < 256; i++) {
        volatile uint32_t d = psram_read(0x003000 + i * 4);
        (void)d;
    }
    uart_puts("  Performance test complete\n");
    
    uart_puts("\n=== All Tests Complete ===\n");
}

int main(void) {
    uart_init(115200);
    uart_puts("\n\n*** PSRAM Test Program ***\n");
    
    psram_test();
    
    uart_puts("\nTest finished.\n");
    
    while(1);
    return 0;
}
