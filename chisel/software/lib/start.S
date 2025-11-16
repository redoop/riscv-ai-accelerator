/* start.S - Startup code for RISC-V AI SoC */

.section .text.start
.global _start

_start:
    /* Set up stack pointer */
    la sp, __stack_top
    
    /* Clear BSS section */
    la t0, __bss_start
    la t1, __bss_end
clear_bss:
    beq t0, t1, bss_done
    sw zero, 0(t0)
    addi t0, t0, 4
    j clear_bss
bss_done:
    
    /* Call main */
    call main
    
    /* If main returns, loop forever */
loop:
    j loop

/* Trap handler (optional) */
.global trap_handler
trap_handler:
    j trap_handler
