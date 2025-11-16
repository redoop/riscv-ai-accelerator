// system_monitor.c - Real-time System Monitor
// Displays system status and resource usage

#include "../lib/hal.h"
#include "../lib/graphics.h"

// System status
typedef struct {
    uint32_t cpu_freq;
    uint32_t uptime;
    uint32_t uart_rx_count;
    uint32_t uart_tx_count;
    uint32_t lcd_updates;
    bool ai_busy;
} SystemStatus;

void update_status(SystemStatus* status);
void display_status(SystemStatus* status);
void draw_bar(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t value, uint16_t color);

void main(void) {
    SystemStatus status = {
        .cpu_freq = 50,  // 50 MHz
        .uptime = 0,
        .uart_rx_count = 0,
        .uart_tx_count = 0,
        .lcd_updates = 0,
        .ai_busy = false
    };
    
    // Initialize hardware
    uart_init(115200);
    lcd_init();
    
    uart_puts("System Monitor Started\r\n");
    
    // Main loop
    while(1) {
        update_status(&status);
        display_status(&status);
        
        // Update every 100ms
        delay_ms(100);
        status.uptime++;
        status.lcd_updates++;
    }
}

void update_status(SystemStatus* status) {
    // Check UART status
    if (uart_rx_ready()) {
        status->uart_rx_count++;
        uart_getc();  // Consume byte
    }
    
    // Check AI accelerator status
    volatile uint32_t* compact_status = (uint32_t*)0x10000004;
    status->ai_busy = (*compact_status & 0x1) != 0;
    
    // Check for UART commands
    if (uart_rx_ready()) {
        char cmd = uart_getc();
        if (cmd == 'r') {
            // Reset counters
            status->uart_rx_count = 0;
            status->uart_tx_count = 0;
            status->lcd_updates = 0;
            status->uptime = 0;
        }
    }
}

void display_status(SystemStatus* status) {
    lcd_clear(COLOR_BLACK);
    
    // Title
    lcd_draw_string(5, 2, "System Monitor", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_line(5, 14, 123, 14, COLOR_CYAN);
    
    // CPU Info
    lcd_draw_string(5, 18, "CPU:", COLOR_GREEN, COLOR_BLACK);
    lcd_printf(40, 18, COLOR_WHITE, COLOR_BLACK, "%d MHz", status->cpu_freq);
    
    // Uptime
    lcd_draw_string(5, 28, "Time:", COLOR_GREEN, COLOR_BLACK);
    lcd_printf(40, 28, COLOR_WHITE, COLOR_BLACK, "%d.%ds", 
               status->uptime / 10, status->uptime % 10);
    
    // UART Stats
    lcd_draw_string(5, 42, "UART RX:", COLOR_YELLOW, COLOR_BLACK);
    lcd_printf(5, 52, COLOR_WHITE, COLOR_BLACK, "%d", status->uart_rx_count);
    
    lcd_draw_string(5, 62, "UART TX:", COLOR_YELLOW, COLOR_BLACK);
    lcd_printf(5, 72, COLOR_WHITE, COLOR_BLACK, "%d", status->uart_tx_count);
    
    // LCD Updates
    lcd_draw_string(5, 86, "LCD Upd:", COLOR_YELLOW, COLOR_BLACK);
    lcd_printf(5, 96, COLOR_WHITE, COLOR_BLACK, "%d", status->lcd_updates);
    
    // AI Status
    lcd_draw_string(5, 110, "AI:", COLOR_MAGENTA, COLOR_BLACK);
    if (status->ai_busy) {
        lcd_fill_circle(30, 114, 4, COLOR_RED);
        lcd_draw_string(40, 110, "BUSY", COLOR_RED, COLOR_BLACK);
    } else {
        lcd_fill_circle(30, 114, 4, COLOR_GREEN);
        lcd_draw_string(40, 110, "IDLE", COLOR_GREEN, COLOR_BLACK);
    }
    
    // CPU usage bar (simulated)
    uint8_t cpu_usage = (status->uptime % 100);
    draw_bar(80, 20, 40, 8, cpu_usage, COLOR_CYAN);
}

void draw_bar(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint8_t value, uint16_t color) {
    // Draw border
    lcd_draw_rect(x, y, w, h, COLOR_WHITE);
    
    // Draw fill
    uint8_t fill_w = (value * (w - 2)) / 100;
    if (fill_w > 0) {
        lcd_fill_rect(x + 1, y + 1, fill_w, h - 2, color);
    }
}
