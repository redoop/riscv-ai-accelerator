#!/usr/bin/env python3
"""
手动插入时钟缓冲器
将 1 个时钟分成多个分支，每个分支驱动约 100 个触发器
"""

import re
import sys

def insert_clock_buffers(input_file, output_file):
    """在网表中插入时钟缓冲器"""
    
    print("读取网表...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 找到所有连接到 sys_clk 的触发器
    print("查找时钟连接...")
    ff_lines = []
    for i, line in enumerate(lines):
        if '.CK(sys_clk)' in line:
            ff_lines.append(i)
    
    print(f"找到 {len(ff_lines)} 个触发器连接到 sys_clk")
    
    # 计算需要的缓冲器数量
    ffs_per_buffer = 100
    num_buffers = (len(ff_lines) + ffs_per_buffer - 1) // ffs_per_buffer
    
    print(f"需要 {num_buffers} 个时钟缓冲器")
    
    # 生成时钟缓冲器定义
    clock_buffers = []
    clock_buffers.append("  // 时钟树缓冲器\n")
    clock_buffers.append("  wire sys_clk_root;\n")
    
    # 根缓冲器
    clock_buffers.append("  BUFX8H7L clk_root_buf (\n")
    clock_buffers.append("    .A(sys_clk),\n")
    clock_buffers.append("    .Y(sys_clk_root)\n")
    clock_buffers.append("  );\n\n")
    
    # 分支缓冲器
    for i in range(num_buffers):
        clock_buffers.append(f"  wire sys_clk_buf{i};\n")
        clock_buffers.append(f"  BUFX4H7L clk_buf{i} (\n")
        clock_buffers.append(f"    .A(sys_clk_root),\n")
        clock_buffers.append(f"    .Y(sys_clk_buf{i})\n")
        clock_buffers.append(f"  );\n\n")
    
    # 修改触发器连接
    print("修改触发器连接...")
    for idx, line_num in enumerate(ff_lines):
        buffer_idx = idx // ffs_per_buffer
        lines[line_num] = lines[line_num].replace(
            '.CK(sys_clk)',
            f'.CK(sys_clk_buf{buffer_idx})'
        )
    
    # 插入时钟缓冲器定义（在 module 定义后）
    for i, line in enumerate(lines):
        if 'module asic_top' in line:
            # 找到第一个 wire 定义的位置
            for j in range(i, min(i+100, len(lines))):
                if 'wire' in lines[j]:
                    lines.insert(j, ''.join(clock_buffers))
                    break
            break
    
    # 写入输出文件
    print(f"写入修改后的网表到 {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print("✅ 完成！")
    print(f"插入了 {num_buffers + 1} 个时钟缓冲器")
    print(f"每个缓冲器驱动约 {ffs_per_buffer} 个触发器")

if __name__ == '__main__':
    input_file = 'project/netlist/asic_top_ics55.v'
    output_file = 'project/netlist/asic_top_ics55_clkbuf.v'
    
    print("=" * 50)
    print("手动插入时钟缓冲器")
    print("=" * 50)
    
    insert_clock_buffers(input_file, output_file)
    
    print("\n下一步：")
    print("1. 使用新网表运行 STA")
    print("2. 验证时序改进")
