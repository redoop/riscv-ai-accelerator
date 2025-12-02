OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY {
    RAM : ORIGIN = 0x80000000, LENGTH = 64K
}

SECTIONS {
    .text : {
        *(.text.start)
        *(.text*)
    } > RAM
    
    .data : {
        *(.data*)
    } > RAM
    
    .bss : {
        *(.bss*)
    } > RAM
}
