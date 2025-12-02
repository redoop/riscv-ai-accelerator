// 简单的引导内存模型
// 用于后综合仿真加载程序

module boot_mem #(
    parameter ADDR_WIDTH = 14,  // 16KB
    parameter DATA_WIDTH = 32
) (
    input  logic clk,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wdata,
    input  logic [3:0] wmask,
    input  logic wen,
    output logic [DATA_WIDTH-1:0] rdata
);

    logic [7:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    // 初始化内存
    initial begin
        $readmemh("testbench/test.hex", mem);
    end
    
    // 读操作
    always_comb begin
        rdata = {mem[addr+3], mem[addr+2], mem[addr+1], mem[addr]};
    end
    
    // 写操作
    always_ff @(posedge clk) begin
        if (wen) begin
            if (wmask[0]) mem[addr+0] <= wdata[7:0];
            if (wmask[1]) mem[addr+1] <= wdata[15:8];
            if (wmask[2]) mem[addr+2] <= wdata[23:16];
            if (wmask[3]) mem[addr+3] <= wdata[31:24];
        end
    end

endmodule
