// 简单的 GPIO 测试程序
#define GPIO_BASE 0x20020000

volatile unsigned int *gpio_out = (unsigned int *)GPIO_BASE;

void delay(int count) {
    for (int i = 0; i < count; i++) {
        asm volatile("nop");
    }
}

int main() {
    while (1) {
        *gpio_out = 0xAAAAAAAA;
        delay(100);
        *gpio_out = 0x55555555;
        delay(100);
    }
    return 0;
}
