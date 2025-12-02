.section .text.start
.globl _start

_start:
    # 设置栈指针
    lui sp, 0x80010
    
    # 跳转到 main
    call main
    
    # 如果返回，死循环
1:
    j 1b
