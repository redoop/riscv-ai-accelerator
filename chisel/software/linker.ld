/* Linker script for RISC-V AI SoC */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00010000, LENGTH = 256M - 64K
}

SECTIONS
{
    . = 0x00010000;
    
    .text : {
        *(.text.start)
        *(.text*)
        *(.rodata*)
    } > RAM
    
    .data : {
        *(.data*)
        *(.sdata*)
    } > RAM
    
    .bss : {
        __bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        __bss_end = .;
    } > RAM
    
    . = ALIGN(16);
    __stack_top = 0x00100000;  /* 1MB for stack */
    
    /DISCARD/ : {
        *(.comment)
        *(.eh_frame)
    }
}
