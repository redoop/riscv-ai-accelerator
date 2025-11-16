// bootloader.c - Simple Bootloader for RISC-V AI SoC
// Phase 3 of DEV_PLAN_V0.2

#include "../lib/hal.h"
#include "../lib/graphics.h"

// Bootloader commands
#define CMD_UPLOAD    'U'  // Upload program
#define CMD_RUN       'R'  // Run program
#define CMD_READ_MEM  'M'  // Read memory
#define CMD_WRITE_REG 'W'  // Write register
#define CMD_LCD_TEST  'L'  // LCD test
#define CMD_PING      'P'  // Ping
#define CMD_INFO      'I'  // Get info

// Program load address
#define PROGRAM_ADDR  0x00010000

// Function prototypes
void display_boot_screen(void);
void handle_command(uint8_t cmd);
void cmd_upload(void);
void cmd_run(void);
void cmd_read_mem(void);
void cmd_write_reg(void);
void cmd_lcd_test(void);
void cmd_ping(void);
void cmd_info(void);

// ============================================================================
// Main
// ============================================================================

int main(void) {
    // Initialize hardware
    uart_init(115200);
    lcd_init();
    
    // Display boot screen
    display_boot_screen();
    
    // Send ready message
    uart_puts("\r\nRISC-V AI Bootloader v0.2\r\n");
    uart_puts("Ready for commands...\r\n");
    
    // Main loop
    while (1) {
        if (uart_rx_ready()) {
            uint8_t cmd = uart_getc();
            handle_command(cmd);
        }
    }
    
    return 0;
}

// ============================================================================
// Display Functions
// ============================================================================

void display_boot_screen(void) {
    lcd_clear(COLOR_BLACK);
    
    // Title
    lcd_draw_string(20, 10, "RISC-V AI", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(20, 20, "Bootloader", COLOR_CYAN, COLOR_BLACK);
    
    // Version
    lcd_draw_string(40, 40, "v0.2", COLOR_GREEN, COLOR_BLACK);
    
    // Status
    lcd_draw_rect(10, 60, 108, 20, COLOR_WHITE);
    lcd_draw_string(20, 65, "Ready...", COLOR_YELLOW, COLOR_BLACK);
    
    // Draw decorative elements
    lcd_draw_line(0, 90, 127, 90, COLOR_BLUE);
    lcd_fill_circle(64, 110, 10, COLOR_RED);
}

// ============================================================================
// Command Handler
// ============================================================================

void handle_command(uint8_t cmd) {
    switch (cmd) {
        case CMD_UPLOAD:
            cmd_upload();
            break;
        case CMD_RUN:
            cmd_run();
            break;
        case CMD_READ_MEM:
            cmd_read_mem();
            break;
        case CMD_WRITE_REG:
            cmd_write_reg();
            break;
        case CMD_LCD_TEST:
            cmd_lcd_test();
            break;
        case CMD_PING:
            cmd_ping();
            break;
        case CMD_INFO:
            cmd_info();
            break;
        default:
            uart_puts("ERR: Unknown command\r\n");
            break;
    }
}

// ============================================================================
// Command Implementations
// ============================================================================

void cmd_upload(void) {
    // Read program size (4 bytes, little endian)
    uint32_t size = 0;
    for (int i = 0; i < 4; i++) {
        size |= ((uint32_t)uart_getc()) << (i * 8);
    }
    
    // Update display
    lcd_fill_rect(10, 60, 108, 20, COLOR_BLACK);
    lcd_draw_rect(10, 60, 108, 20, COLOR_WHITE);
    lcd_draw_string(20, 65, "Uploading", COLOR_YELLOW, COLOR_BLACK);
    
    // Receive program data
    uint8_t* dest = (uint8_t*)PROGRAM_ADDR;
    for (uint32_t i = 0; i < size; i++) {
        dest[i] = uart_getc();
        
        // Update progress every 256 bytes
        if ((i & 0xFF) == 0) {
            uint32_t progress = (i * 100) / size;
            lcd_fill_rect(11, 71, progress, 8, COLOR_GREEN);
        }
    }
    
    // Send acknowledgment
    uart_putc('K');
    uart_puts("\r\nUpload complete\r\n");
    
    // Update display
    lcd_fill_rect(10, 60, 108, 20, COLOR_BLACK);
    lcd_draw_rect(10, 60, 108, 20, COLOR_WHITE);
    lcd_draw_string(20, 65, "Complete!", COLOR_GREEN, COLOR_BLACK);
}

void cmd_run(void) {
    uart_puts("Running program...\r\n");
    
    // Update display
    lcd_fill_rect(10, 60, 108, 20, COLOR_BLACK);
    lcd_draw_rect(10, 60, 108, 20, COLOR_WHITE);
    lcd_draw_string(20, 65, "Running", COLOR_GREEN, COLOR_BLACK);
    
    // Jump to program
    void (*program)(void) = (void (*)(void))PROGRAM_ADDR;
    program();
}

void cmd_read_mem(void) {
    // Read address (4 bytes)
    uint32_t addr = 0;
    for (int i = 0; i < 4; i++) {
        addr |= ((uint32_t)uart_getc()) << (i * 8);
    }
    
    // Read length (4 bytes)
    uint32_t len = 0;
    for (int i = 0; i < 4; i++) {
        len |= ((uint32_t)uart_getc()) << (i * 8);
    }
    
    // Send data
    uint8_t* src = (uint8_t*)addr;
    for (uint32_t i = 0; i < len; i++) {
        uart_putc(src[i]);
    }
}

void cmd_write_reg(void) {
    // Read address (4 bytes)
    uint32_t addr = 0;
    for (int i = 0; i < 4; i++) {
        addr |= ((uint32_t)uart_getc()) << (i * 8);
    }
    
    // Read value (4 bytes)
    uint32_t value = 0;
    for (int i = 0; i < 4; i++) {
        value |= ((uint32_t)uart_getc()) << (i * 8);
    }
    
    // Write to register
    *((volatile uint32_t*)addr) = value;
    
    uart_putc('K');
}

void cmd_lcd_test(void) {
    uart_puts("LCD Test\r\n");
    
    // Test 1: Color bars
    lcd_clear(COLOR_BLACK);
    lcd_fill_rect(0, 0, 128, 18, COLOR_RED);
    lcd_fill_rect(0, 18, 128, 18, COLOR_GREEN);
    lcd_fill_rect(0, 36, 128, 18, COLOR_BLUE);
    lcd_fill_rect(0, 54, 128, 18, COLOR_YELLOW);
    lcd_fill_rect(0, 72, 128, 18, COLOR_CYAN);
    lcd_fill_rect(0, 90, 128, 18, COLOR_MAGENTA);
    lcd_fill_rect(0, 108, 128, 20, COLOR_WHITE);
    delay_ms(1000);
    
    // Test 2: Shapes
    lcd_clear(COLOR_BLACK);
    lcd_draw_rect(10, 10, 50, 50, COLOR_RED);
    lcd_fill_rect(70, 10, 50, 50, COLOR_GREEN);
    lcd_draw_circle(35, 90, 20, COLOR_BLUE);
    lcd_fill_circle(93, 90, 20, COLOR_YELLOW);
    delay_ms(1000);
    
    // Test 3: Text
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(10, 10, "Hello!", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(10, 30, "RISC-V", COLOR_RED, COLOR_BLACK);
    lcd_draw_string(10, 50, "AI Chip", COLOR_GREEN, COLOR_BLACK);
    lcd_draw_string(10, 70, "v0.2", COLOR_CYAN, COLOR_BLACK);
    
    uart_putc('K');
}

void cmd_ping(void) {
    uart_putc('K');
}

void cmd_info(void) {
    uart_puts("RISC-V AI SoC Bootloader\r\n");
    uart_puts("Version: 0.2\r\n");
    uart_puts("CPU: PicoRV32 @ 50MHz\r\n");
    uart_puts("RAM: 256MB\r\n");
    uart_puts("LCD: ST7735 128x128\r\n");
    uart_puts("UART: 115200 bps\r\n");
}
