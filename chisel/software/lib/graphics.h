// graphics.h - Graphics Library
// Phase 4 of DEV_PLAN_V0.2

#ifndef GRAPHICS_H
#define GRAPHICS_H

#include "hal.h"

// ============================================================================
// Basic Graphics Functions
// ============================================================================

void lcd_draw_line(uint8_t x0, uint8_t y0, uint8_t x1, uint8_t y1, uint16_t color);
void lcd_draw_rect(uint8_t x, uint8_t y, uint8_t w, uint8_t h, uint16_t color);
void lcd_draw_circle(uint8_t x0, uint8_t y0, uint8_t r, uint16_t color);
void lcd_fill_circle(uint8_t x0, uint8_t y0, uint8_t r, uint16_t color);

// ============================================================================
// Text Functions
// ============================================================================

void lcd_draw_char(uint8_t x, uint8_t y, char c, uint16_t fg, uint16_t bg);
void lcd_draw_string(uint8_t x, uint8_t y, const char* str, uint16_t fg, uint16_t bg);
void lcd_printf(uint8_t x, uint8_t y, uint16_t fg, uint16_t bg, const char* fmt, ...);

// ============================================================================
// Image Functions
// ============================================================================

void lcd_draw_image(uint8_t x, uint8_t y, uint8_t w, uint8_t h, const uint16_t* data);

// ============================================================================
// Font Data (8x8 ASCII)
// ============================================================================

extern const uint8_t font_8x8[128][8];

#endif // GRAPHICS_H
