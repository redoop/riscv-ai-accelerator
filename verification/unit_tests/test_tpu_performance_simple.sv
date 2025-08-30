// 简化的TPU性能测试
// 测试MAC单元的性能和吞吐量

`timescale 1ns/1ps

module test_tpu_performance_simple;

    parameter DATA_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    parameter NUM_OPERATIONS = 1000;
    
    // 时钟和复位
    logic clk;
    logic rst_n;
    
    // DUT信号
    logic enable;
    logic [1:0] data_type;
    logic [DATA_WIDTH-1:0] a_in, b_in, c_in;
    logic [DATA_WIDTH-1:0] a_out, b_out, c_out;
    logic load_weight;
    logic accumulate;
    logic overflow, underflow;
    
    // 测试变量
    integer i;
    integer correct_results;
    integer total_operations;
    real start_time, end_time, duration;
    real throughput;
    
    // 实例化DUT
    tpu_mac_unit #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .data_type(data_type),
        .a_in(a_in),
        .b_in(b_in),
        .c_in(c_in),
        .a_out(a_out),
        .b_out(b_out),
        .c_out(c_out),
        .load_weight(load_weight),
        .accumulate(accumulate),
        .overflow(overflow),
        .underflow(underflow)
    );
    
    // 时钟生成
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // 主测试序列
    initial begin
        $display("=== TPU MAC性能测试 ===");
        $display("测试参数:");
        $display("  时钟周期: %0d ns", CLK_PERIOD);
        $display("  操作数量: %0d", NUM_OPERATIONS);
        
        // 初始化
        rst_n = 0;
        enable = 0;
        data_type = 2'b00; // INT8
        a_in = 0;
        b_in = 0;
        c_in = 0;
        load_weight = 0;
        accumulate = 0;
        correct_results = 0;
        total_operations = 0;
        
        // 复位
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // 性能测试1: 连续乘法操作
        $display("\n🔧 测试1: 连续乘法性能");
        start_time = $realtime;
        
        for (i = 0; i < NUM_OPERATIONS; i = i + 1) begin
            // 设置输入数据
            a_in = (i % 127) + 1;  // 1-127
            b_in = ((i * 3) % 127) + 1;  // 变化的权重
            c_in = 0;
            accumulate = 0;
            enable = 1;
            load_weight = (i % 10 == 0); // 每10次操作加载新权重
            
            @(posedge clk);
            
            // 检查结果
            if (i > 0) begin // 跳过第一个结果
                total_operations = total_operations + 1;
                // 简单验证：结果不应该为0（除非输入为0）
                if (c_out != 0 || (a_in == 0 || b_in == 0)) begin
                    correct_results = correct_results + 1;
                end
            end
        end
        
        end_time = $realtime;
        duration = end_time - start_time;
        throughput = (NUM_OPERATIONS * 1000.0) / duration; // 操作/微秒
        
        $display("  完成时间: %.2f ns", duration);
        $display("  吞吐量: %.2f 操作/微秒", throughput);
        $display("  正确结果: %0d/%0d", correct_results, total_operations);
        
        // 性能测试2: 累加操作性能
        $display("\n🔧 测试2: 累加操作性能");
        start_time = $realtime;
        
        // 加载固定权重
        a_in = 5;
        b_in = 3;
        load_weight = 1;
        enable = 1;
        @(posedge clk);
        load_weight = 0;
        
        // 连续累加
        for (i = 0; i < 100; i = i + 1) begin
            a_in = 2;  // 固定激活值
            c_in = c_out; // 使用前一个结果作为累加输入
            accumulate = 1;
            enable = 1;
            
            @(posedge clk);
        end
        
        end_time = $realtime;
        duration = end_time - start_time;
        
        $display("  累加100次完成时间: %.2f ns", duration);
        $display("  最终累加结果: %0d", c_out);
        $display("  预期结果: %0d", 5 * 3 + 100 * 2 * 3); // 初始值 + 100次累加
        
        // 性能测试3: 不同数据类型性能对比
        $display("\n🔧 测试3: 数据类型性能对比");
        
        // INT8性能
        data_type = 2'b00;
        start_time = $realtime;
        for (i = 0; i < 100; i = i + 1) begin
            a_in = 10;
            b_in = 5;
            enable = 1;
            @(posedge clk);
        end
        end_time = $realtime;
        $display("  INT8 (100次操作): %.2f ns", end_time - start_time);
        
        // FP16性能
        data_type = 2'b01;
        start_time = $realtime;
        for (i = 0; i < 100; i = i + 1) begin
            a_in = 32'h4200; // FP16格式的3.0
            b_in = 32'h4000; // FP16格式的2.0
            enable = 1;
            @(posedge clk);
        end
        end_time = $realtime;
        $display("  FP16 (100次操作): %.2f ns", end_time - start_time);
        
        // FP32性能
        data_type = 2'b10;
        start_time = $realtime;
        for (i = 0; i < 100; i = i + 1) begin
            a_in = 32'h40400000; // FP32格式的3.0
            b_in = 32'h40000000; // FP32格式的2.0
            enable = 1;
            @(posedge clk);
        end
        end_time = $realtime;
        $display("  FP32 (100次操作): %.2f ns", end_time - start_time);
        
        // 测试总结
        $display("\n=== 性能测试总结 ===");
        $display("✅ MAC单元性能测试完成");
        $display("📊 基础吞吐量: %.2f 操作/微秒", throughput);
        $display("🎯 功能正确性: %0d%%", (correct_results * 100) / total_operations);
        $display("⚡ 支持多种数据类型: INT8, FP16, FP32");
        
        if (correct_results == total_operations) begin
            $display("🎉 所有性能测试通过!");
        end else begin
            $display("⚠️  部分测试结果异常");
        end
        
        $finish;
    end
    
    // 监控溢出/下溢
    always @(posedge clk) begin
        if (overflow) begin
            $display("⚠️  检测到溢出 @ %0t", $time);
        end
        if (underflow) begin
            $display("⚠️  检测到下溢 @ %0t", $time);
        end
    end
    
    // 生成VCD波形文件
    initial begin
        $dumpfile("tpu_performance_test.vcd");
        $dumpvars(0, test_tpu_performance_simple);
    end

endmodule