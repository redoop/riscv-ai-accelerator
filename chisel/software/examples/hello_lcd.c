// hello_lcd.c - Hello World LCD Example
// Phase 4 of DEV_PLAN_V0.2

#include "../lib/hal.h"
#include "../lib/graphics.h"

void main(void) {
    // Initialize hardware
    uart_init(115200);
    lcd_init();
    
    uart_puts("Hello LCD Example\r\n");
    
    // Clear screen
    lcd_clear(COLOR_BLACK);
    
    // Draw title
    lcd_draw_string(20, 10, "Hello", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(20, 20, "RISC-V!", COLOR_CYAN, COLOR_BLACK);
    
    // Draw some shapes
    lcd_draw_rect(10, 40, 108, 40, COLOR_GREEN);
    lcd_fill_circle(64, 60, 15, COLOR_RED);
    
    // Draw text at bottom
    lcd_draw_string(10, 100, "AI Chip", COLOR_YELLOW, COLOR_BLACK);
    lcd_draw_string(10, 110, "v0.2", COLOR_MAGENTA, COLOR_BLACK);
    
    // Animation loop
    uint8_t x = 0;
    while (1) {
        // Draw moving pixel
        lcd_draw_pixel(x, 90, COLOR_WHITE);
        delay_ms(50);
        lcd_draw_pixel(x, 90, COLOR_BLACK);
        
        x = (x + 1) % 128;
        
        // Send heartbeat to UART
        if (x == 0) {
            uart_putc('.');
        }
    }
}
