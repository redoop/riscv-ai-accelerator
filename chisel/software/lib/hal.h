// hal.h - Hardware Abstraction Layer
// Phase 3 & 4 of DEV_PLAN_V0.2

#ifndef HAL_H
#define HAL_H

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Memory Map
// ============================================================================

#define RAM_BASE        0x00000000
#define COMPACT_BASE    0x10000000
#define BITNET_BASE     0x10001000
#define UART_BASE       0x20000000
#define LCD_BASE        0x20010000
#define GPIO_BASE       0x20020000

// ============================================================================
// UART Registers
// ============================================================================

typedef struct {
    volatile uint32_t DATA;       // 0x00: Data register
    volatile uint32_t STATUS;     // 0x04: Status register
    volatile uint32_t CONTROL;    // 0x08: Control register
    volatile uint32_t BAUD_DIV;   // 0x0C: Baud rate divisor
} UART_TypeDef;

#define UART ((UART_TypeDef*)UART_BASE)

// UART Status bits
#define UART_STATUS_TX_BUSY       (1 << 0)
#define UART_STATUS_RX_READY      (1 << 1)
#define UART_STATUS_TX_FIFO_FULL  (1 << 2)
#define UART_STATUS_RX_FIFO_EMPTY (1 << 3)

// UART Control bits
#define UART_CTRL_TX_ENABLE       (1 << 0)
#define UART_CTRL_RX_ENABLE       (1 << 1)
#define UART_CTRL_TX_IRQ_ENABLE   (1 << 2)
#define UART_CTRL_RX_IRQ_ENABLE   (1 << 3)

// ============================================================================
// LCD Registers
// ============================================================================

typedef struct {
    volatile uint32_t COMMAND;    // 0x00: Command register
    volatile uint32_t DATA;       // 0x04: Data register
    volatile uint32_t STATUS;     // 0x08: Status register
    volatile uint32_t CONTROL;    // 0x0C: Control register
    volatile uint32_t X_START;    // 0x10: X start coordinate
    volatile uint32_t Y_START;    // 0x14: Y start coordinate
    volatile uint32_t X_END;      // 0x18: X end coordinate
    volatile uint32_t Y_END;      // 0x1C: Y end coordinate
    volatile uint32_t COLOR;      // 0x20: Color data (RGB565)
    uint32_t _reserved[1015];     // Reserved
    volatile uint16_t FRAMEBUFFER[16384]; // 0x1000: Framebuffer (32KB)
} LCD_TypeDef;

#define LCD ((LCD_TypeDef*)LCD_BASE)

// LCD Status bits
#define LCD_STATUS_BUSY      (1 << 0)
#define LCD_STATUS_INIT_DONE (1 << 1)

// LCD Control bits
#define LCD_CTRL_BACKLIGHT   (1 << 0)
#define LCD_CTRL_RESET       (1 << 1)

// LCD Commands (ST7735)
#define LCD_CMD_NOP          0x00
#define LCD_CMD_SWRESET      0x01
#define LCD_CMD_SLPOUT       0x11
#define LCD_CMD_NORON        0x13
#define LCD_CMD_INVOFF       0x20
#define LCD_CMD_DISPON       0x29
#define LCD_CMD_CASET        0x2A
#define LCD_CMD_RASET        0x2B
#define LCD_CMD_RAMWR        0x2C
#define LCD_CMD_COLMOD       0x3A

// ============================================================================
// GPIO Registers
// ============================================================================

typedef struct {
    volatile uint32_t DATA;       // 0x00: GPIO data
} GPIO_TypeDef;

#define GPIO ((GPIO_TypeDef*)GPIO_BASE)

// ============================================================================
// Color Definitions (RGB565)
// ============================================================================

#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF
#define COLOR_RED     0xF800
#define COLOR_GREEN   0x07E0
#define COLOR_BLUE    0x001F
#define COLOR_YELLOW  0xFFE0
#define COLOR_CYAN    0x07FF
#define COLOR_MAGENTA 0xF81F
#define COLOR_ORANGE  0xFD20
#define COLOR_PURPLE  0x780F
#define COLOR_GRAY    0x8410

// RGB to RGB565 conversion
#define RGB565(r, g, b) ((((r) & 0xF8) << 8) | (((g) & 0xFC) << 3) | (((b) & 0xF8) >> 3))

// ============================================================================
// UART Functions
// ============================================================================

void uart_init(uint32_t baudrate);
void uart_putc(char c);
char uart_getc(void);
void uart_puts(const char* str);
bool uart_rx_ready(void);
bool uart_tx_ready(void);

// ============================================================================
// LCD Functions
// ============================================================================

void lcd_init(void);
void lcd_reset(void);
void lcd_backlight(bool on);
void lcd_send_command(uint8_t cmd);
void lcd_send_data(uint8_t data);
void lcd_set_window(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1);
void lcd_draw_pixel(uint8_t x, uint8_t y, uint16_t color);
void lcd_clear(uint16_t color);
void lcd_fill_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color);

// ============================================================================
// Utility Functions
// ============================================================================

void delay_ms(uint32_t ms);
void delay_us(uint32_t us);

#endif // HAL_H
