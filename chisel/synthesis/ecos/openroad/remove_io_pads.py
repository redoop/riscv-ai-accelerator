#!/usr/bin/env python3
"""彻底移除网表中的 IO PAD"""

import re

input_file = "../project/netlist/asic_top_ics55.v"
output_file = "asic_top_core.v"

print(f"读取: {input_file}")
with open(input_file, 'r') as f:
    lines = f.readlines()

print(f"原始行数: {len(lines)}")

# 移除包含 IO PAD 的行
filtered_lines = []
removed = 0
for line in lines:
    # 跳过 IO PAD 实例定义
    if re.search(r'P65_1233_\w+\s+u_\w*pad', line):
        removed += 1
        continue
    # 跳过 latch 实例
    if re.search(r'\$_DLATCH_P_', line):
        removed += 1
        continue
    # 跳过 IO PAD 端口连接
    if re.search(r'\.io_pad\d+\(', line):
        removed += 1
        continue
    if re.search(r'\.sys_clk_[io]_pad\(', line):
        removed += 1
        continue
    if re.search(r'\.rst_n_pad\(', line):
        removed += 1
        continue
    if re.search(r'\.ip_sel_pad\d+\(', line):
        removed += 1
        continue
    
    filtered_lines.append(line)

print(f"移除行数: {removed}")
print(f"保留行数: {len(filtered_lines)}")

# 修改模块端口列表，移除 PAD 端口
content = ''.join(filtered_lines)

# 简化端口列表
content = re.sub(
    r'module asic_top\([^)]+\);',
    'module asic_top(sys_clk_i_pad, rst_n_pad);',
    content
)

# 添加端口声明
port_decl = """  input sys_clk_i_pad;
  input rst_n_pad;
"""
content = re.sub(r'(module asic_top[^;]+;)', r'\1\n' + port_decl, content)

print(f"写入: {output_file}")
with open(output_file, 'w') as f:
    f.write(content)

print("✅ 完成")
