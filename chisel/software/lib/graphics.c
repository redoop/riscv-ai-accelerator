// graphics.c - Graphics Library Implementation
// Phase 4 of DEV_PLAN_V0.2

#include "graphics.h"
#include <stdarg.h>
#include <stdio.h>

// ============================================================================
// Helper Functions
// ============================================================================

static inline void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

static inline int abs(int x) {
    return x < 0 ? -x : x;
}

// ============================================================================
// Basic Graphics Functions
// ============================================================================

void lcd_draw_line(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, uint16_t color) {
    // Bresenham's line algorithm
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    
    while (1) {
        lcd_draw_pixel(x0, y0, color);
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void lcd_draw_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color) {
    lcd_draw_line(x, y, x + w - 1, y, color);
    lcd_draw_line(x + w - 1, y, x + w - 1, y + h - 1, color);
    lcd_draw_line(x + w - 1, y + h - 1, x, y + h - 1, color);
    lcd_draw_line(x, y + h - 1, x, y, color);
}

void lcd_draw_circle(uint8_t x0, uint8_t y0, uint8_t r, uint16_t color) {
    // Midpoint circle algorithm
    int x = r;
    int y = 0;
    int err = 0;
    
    while (x >= y) {
        lcd_draw_pixel(x0 + x, y0 + y, color);
        lcd_draw_pixel(x0 + y, y0 + x, color);
        lcd_draw_pixel(x0 - y, y0 + x, color);
        lcd_draw_pixel(x0 - x, y0 + y, color);
        lcd_draw_pixel(x0 - x, y0 - y, color);
        lcd_draw_pixel(x0 - y, y0 - x, color);
        lcd_draw_pixel(x0 + y, y0 - x, color);
        lcd_draw_pixel(x0 + x, y0 - y, color);
        
        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

void lcd_fill_circle(uint8_t x0, uint8_t y0, uint8_t r, uint16_t color) {
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            if (x * x + y * y <= r * r) {
                lcd_draw_pixel(x0 + x, y0 + y, color);
            }
        }
    }
}

// ============================================================================
// Text Functions
// ============================================================================

void lcd_draw_char(uint8_t x, uint8_t y, char c, uint16_t fg, uint16_t bg) {
    if (c < 0 || c >= 128) c = '?';
    
    for (uint8_t i = 0; i < 8; i++) {
        uint8_t line = font_8x8[(uint8_t)c][i];
        for (uint8_t j = 0; j < 8; j++) {
            if (line & (1 << j)) {
                lcd_draw_pixel(x + j, y + i, fg);
            } else {
                lcd_draw_pixel(x + j, y + i, bg);
            }
        }
    }
}

void lcd_draw_string(uint8_t x, uint8_t y, const char* str, uint16_t fg, uint16_t bg) {
    uint8_t cx = x;
    while (*str) {
        if (*str == '\n') {
            cx = x;
            y += 8;
        } else {
            lcd_draw_char(cx, y, *str, fg, bg);
            cx += 8;
            if (cx >= 128) {
                cx = x;
                y += 8;
            }
        }
        str++;
    }
}

void lcd_printf(uint8_t x, uint8_t y, uint16_t fg, uint16_t bg, const char* fmt, ...) {
    char buffer[128];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    lcd_draw_string(x, y, buffer, fg, bg);
}

// ============================================================================
// Image Functions
// ============================================================================

void lcd_draw_image(uint8_t x, uint8_t y, uint8_t w, uint8_t h, const uint16_t* data) {
    for (uint8_t j = 0; j < h; j++) {
        for (uint8_t i = 0; i < w; i++) {
            if (x + i < 128 && y + j < 128) {
                lcd_draw_pixel(x + i, y + j, data[j * w + i]);
            }
        }
    }
}
