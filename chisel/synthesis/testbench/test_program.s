# 简单的 RISC-V 测试程序
# 测试 GPIO 输出

.section .text
.globl _start

_start:
    # 设置 GPIO 基地址 (假设在 0x10000000)
    lui  x5, 0x10000      # x5 = 0x10000000
    
loop:
    # 写入 GPIO 输出 (0xAAAAAAAA)
    lui  x6, 0xAAAAB      # x6 = 0xAAAAB000
    addi x6, x6, -1366    # x6 = 0xAAAAAAAA
    sw   x6, 0(x5)        # GPIO[0] = 0xAAAAAAAA
    
    # 延迟
    li   x7, 1000
delay:
    addi x7, x7, -1
    bnez x7, delay
    
    # 写入 GPIO 输出 (0x55555555)
    lui  x6, 0x55555      # x6 = 0x55555000
    addi x6, x6, 1365     # x6 = 0x55555555
    sw   x6, 0(x5)        # GPIO[0] = 0x55555555
    
    # 延迟
    li   x7, 1000
delay2:
    addi x7, x7, -1
    bnez x7, delay2
    
    # 循环
    j loop
