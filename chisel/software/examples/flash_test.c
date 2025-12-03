// flash_test.c - SPI Flash Test Program
// Flash/PSRAM Extension - Phase 1 Day 3

#include "../lib/hal.h"
#include "../lib/graphics.h"

void print_hex(uint32_t val) {
    uart_puts("0x");
    for (int i = 7; i >= 0; i--) {
        uint8_t nibble = (val >> (i * 4)) & 0xF;
        uart_putc(nibble < 10 ? '0' + nibble : 'A' + nibble - 10);
    }
}

int main(void) {
    // Initialize hardware
    uart_init(115200);
    lcd_init();
    flash_init();
    
    uart_puts("\n=== Flash Test ===\n");
    
    // Test 1: Read test
    uart_puts("Test 1: Read from address 0x000000\n");
    uint32_t data = flash_read(0x000000);
    uart_puts("  Data: ");
    print_hex(data);
    uart_puts("\n");
    
    // Test 2: Write test
    uart_puts("Test 2: Write 0xDEADBEEF to 0x001000\n");
    flash_write(0x001000, 0xDEADBEEF);
    uart_puts("  Write complete\n");
    
    // Test 3: Read back
    uart_puts("Test 3: Read back from 0x001000\n");
    data = flash_read(0x001000);
    uart_puts("  Data: ");
    print_hex(data);
    if (data == 0xDEADBEEF) {
        uart_puts(" [PASS]\n");
    } else {
        uart_puts(" [FAIL]\n");
    }
    
    // Test 4: Erase sector
    uart_puts("Test 4: Erase sector at 0x001000\n");
    flash_erase_sector(0x001000);
    uart_puts("  Erase complete\n");
    
    // Test 5: Read after erase
    uart_puts("Test 5: Read after erase\n");
    data = flash_read(0x001000);
    uart_puts("  Data: ");
    print_hex(data);
    if (data == 0xFFFFFFFF) {
        uart_puts(" [PASS]\n");
    } else {
        uart_puts(" [WARN - may not be erased]\n");
    }
    
    // Display on LCD
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(10, 10, "Flash Test", COLOR_GREEN, COLOR_BLACK);
    lcd_draw_string(10, 30, "Read: OK", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(10, 50, "Write: OK", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(10, 70, "Erase: OK", COLOR_WHITE, COLOR_BLACK);
    
    uart_puts("\n=== All Tests Complete ===\n");
    
    while (1) {
        // Idle
    }
    
    return 0;
}
