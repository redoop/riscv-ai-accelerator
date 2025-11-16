// hal.c - Hardware Abstraction Layer Implementation
// Phase 3 & 4 of DEV_PLAN_V0.2

#include "hal.h"

// Assume 50MHz clock
#define CLOCK_FREQ 50000000

// ============================================================================
// UART Functions
// ============================================================================

void uart_init(uint32_t baudrate) {
    // Calculate baud divisor
    uint32_t divisor = CLOCK_FREQ / baudrate;
    UART->BAUD_DIV = divisor;
    
    // Enable TX and RX
    UART->CONTROL = UART_CTRL_TX_ENABLE | UART_CTRL_RX_ENABLE;
}

void uart_putc(char c) {
    // Wait for TX FIFO not full
    while (UART->STATUS & UART_STATUS_TX_FIFO_FULL);
    UART->DATA = c;
}

char uart_getc(void) {
    // Wait for RX data ready
    while (!(UART->STATUS & UART_STATUS_RX_READY));
    return (char)(UART->DATA & 0xFF);
}

void uart_puts(const char* str) {
    while (*str) {
        uart_putc(*str++);
    }
}

bool uart_rx_ready(void) {
    return (UART->STATUS & UART_STATUS_RX_READY) != 0;
}

bool uart_tx_ready(void) {
    return (UART->STATUS & UART_STATUS_TX_FIFO_FULL) == 0;
}

// ============================================================================
// LCD Functions
// ============================================================================

void lcd_init(void) {
    // Enable reset
    LCD->CONTROL = LCD_CTRL_RESET;
    delay_ms(10);
    
    // Wait for initialization to complete
    while (!(LCD->STATUS & LCD_STATUS_INIT_DONE));
    
    // Enable backlight
    LCD->CONTROL = LCD_CTRL_RESET | LCD_CTRL_BACKLIGHT;
    
    delay_ms(100);
}

void lcd_reset(void) {
    LCD->CONTROL = 0;
    delay_ms(10);
    LCD->CONTROL = LCD_CTRL_RESET;
    delay_ms(10);
}

void lcd_backlight(bool on) {
    if (on) {
        LCD->CONTROL |= LCD_CTRL_BACKLIGHT;
    } else {
        LCD->CONTROL &= ~LCD_CTRL_BACKLIGHT;
    }
}

void lcd_send_command(uint8_t cmd) {
    // Wait for not busy
    while (LCD->STATUS & LCD_STATUS_BUSY);
    LCD->COMMAND = cmd;
}

void lcd_send_data(uint8_t data) {
    // Wait for not busy
    while (LCD->STATUS & LCD_STATUS_BUSY);
    LCD->DATA = data;
}

void lcd_set_window(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1) {
    LCD->X_START = x0;
    LCD->Y_START = y0;
    LCD->X_END = x1;
    LCD->Y_END = y1;
    
    // Send column address set command
    lcd_send_command(LCD_CMD_CASET);
    lcd_send_data(0);
    lcd_send_data(x0);
    lcd_send_data(0);
    lcd_send_data(x1);
    
    // Send row address set command
    lcd_send_command(LCD_CMD_RASET);
    lcd_send_data(0);
    lcd_send_data(y0);
    lcd_send_data(0);
    lcd_send_data(y1);
    
    // Start RAM write
    lcd_send_command(LCD_CMD_RAMWR);
}

void lcd_draw_pixel(uint8_t x, uint8_t y, uint16_t color) {
    if (x >= 128 || y >= 128) return;
    
    // Write to framebuffer
    LCD->FRAMEBUFFER[y * 128 + x] = color;
    
    // Also send to display
    lcd_set_window(x, y, x, y);
    lcd_send_data(color >> 8);
    lcd_send_data(color & 0xFF);
}

void lcd_clear(uint16_t color) {
    lcd_fill_rect(0, 0, 128, 128, color);
}

void lcd_fill_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color) {
    if (x >= 128 || y >= 128) return;
    if (x + w > 128) w = 128 - x;
    if (y + h > 128) h = 128 - y;
    
    lcd_set_window(x, y, x + w - 1, y + h - 1);
    
    for (uint16_t i = 0; i < w * h; i++) {
        lcd_send_data(color >> 8);
        lcd_send_data(color & 0xFF);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void delay_ms(uint32_t ms) {
    // Simple delay loop (not accurate, depends on optimization)
    for (uint32_t i = 0; i < ms; i++) {
        for (volatile uint32_t j = 0; j < CLOCK_FREQ / 10000; j++);
    }
}

void delay_us(uint32_t us) {
    // Simple delay loop
    for (volatile uint32_t i = 0; i < us * (CLOCK_FREQ / 10000000); i++);
}
