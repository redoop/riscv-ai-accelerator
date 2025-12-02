// 外部 Boot ROM 模型
// 在 0x80000000 地址提供简单的测试程序

module boot_rom (
    input  logic        clk,
    input  logic [31:0] addr,
    output logic [31:0] data,
    output logic        valid
);

    // 简单的测试程序 (不会 trap)
    always_comb begin
        valid = (addr >= 32'h80000000 && addr < 32'h80000100);
        
        case (addr)
            // NOP 循环 - 不会 trap
            32'h80000000: data = 32'h00000013; // nop
            32'h80000004: data = 32'h00000013; // nop
            32'h80000008: data = 32'h00000013; // nop
            32'h8000000c: data = 32'hff9ff06f; // j -8 (跳回 0x80000008)
            default:      data = 32'h00000013; // nop
        endcase
    end

endmodule
