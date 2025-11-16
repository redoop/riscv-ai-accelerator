// ai_demo.c - AI Inference Demo
// Phase 4 of DEV_PLAN_V0.2

#include "../lib/hal.h"
#include "../lib/graphics.h"

// AI Accelerator registers
#define COMPACT_CTRL   ((volatile uint32_t*)(COMPACT_BASE + 0x000))
#define COMPACT_STATUS ((volatile uint32_t*)(COMPACT_BASE + 0x004))
#define COMPACT_CYCLES ((volatile uint32_t*)(COMPACT_BASE + 0x028))

void display_inference_result(const char* class_name, uint32_t confidence, uint32_t fps) {
    lcd_clear(COLOR_BLACK);
    
    // Title
    lcd_draw_string(10, 10, "AI Inference", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_line(10, 25, 118, 25, COLOR_CYAN);
    
    // Result
    lcd_draw_string(10, 35, "Class:", COLOR_GREEN, COLOR_BLACK);
    lcd_draw_string(10, 45, class_name, COLOR_YELLOW, COLOR_BLACK);
    
    // Confidence
    lcd_printf(10, 60, COLOR_CYAN, COLOR_BLACK, "Conf: %d%%", confidence);
    
    // Progress bar
    lcd_draw_rect(10, 75, 108, 12, COLOR_WHITE);
    uint8_t bar_width = (confidence * 106) / 100;
    lcd_fill_rect(11, 76, bar_width, 10, COLOR_GREEN);
    
    // FPS
    lcd_printf(10, 95, COLOR_YELLOW, COLOR_BLACK, "FPS: %d", fps);
    
    // Status indicator
    lcd_fill_circle(110, 110, 8, COLOR_GREEN);
}

void main(void) {
    // Initialize hardware
    uart_init(115200);
    lcd_init();
    
    uart_puts("AI Demo Starting...\r\n");
    
    // Display initial screen
    lcd_clear(COLOR_BLACK);
    lcd_draw_string(20, 50, "AI Demo", COLOR_WHITE, COLOR_BLACK);
    lcd_draw_string(20, 60, "Loading...", COLOR_CYAN, COLOR_BLACK);
    delay_ms(1000);
    
    // Simulated inference loop
    const char* classes[] = {"Cat", "Dog", "Bird", "Car", "Tree"};
    uint8_t class_idx = 0;
    uint32_t frame_count = 0;
    uint32_t start_time = 0;
    
    while (1) {
        // Simulate AI inference
        // In real application, this would trigger the AI accelerator
        *COMPACT_CTRL = 0x1;  // Start computation
        
        // Wait for completion
        while ((*COMPACT_STATUS & 0x2) == 0);
        
        // Get performance metrics
        uint32_t cycles = *COMPACT_CYCLES;
        uint32_t fps = 50000000 / cycles;  // Assuming 50MHz clock
        
        // Simulated confidence (varies between 75-95%)
        uint32_t confidence = 75 + (frame_count % 20);
        
        // Display result
        display_inference_result(classes[class_idx], confidence, fps);
        
        // Update for next frame
        frame_count++;
        if (frame_count % 30 == 0) {
            class_idx = (class_idx + 1) % 5;
        }
        
        // Send status to UART
        uart_puts("Frame: ");
        uart_puts(classes[class_idx]);
        uart_puts("\r\n");
        
        delay_ms(100);
    }
}
