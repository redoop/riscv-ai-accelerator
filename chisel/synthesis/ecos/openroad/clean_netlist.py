#!/usr/bin/env python3
"""彻底清理网表，移除所有 IO PAD"""

import re

input_file = "../project/netlist/asic_top_ics55.v"
output_file = "asic_top_core_final.v"

with open(input_file, 'r') as f:
    content = f.read()

# 1. 替换模块声明
module_decl = """module asic_top();
  // 内部信号
  wire sys_clk;
  wire rst_n;
"""

content = re.sub(
    r'module asic_top\([^;]+;[^w]*wire[^;]+;[^w]*wire[^;]+;',
    module_decl,
    content,
    flags=re.DOTALL
)

# 2. 移除 IO PAD 实例
content = re.sub(r'P65_1233_\w+[^;]+;', '', content)

# 3. 移除 latch 实例
content = re.sub(r'\$_DLATCH_P_[^;]+;', '', content)

# 4. 移除 wire 声明
for pad_type in ['io_pad', 'ip_sel_pad', 'sys_clk_i_pad', 'sys_clk_o_pad', 'rst_n_pad']:
    content = re.sub(rf'wire\s+{pad_type}\w*\s*;', '', content)

# 5. 清理空行
content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

with open(output_file, 'w') as f:
    f.write(content)

print(f"✅ 生成: {output_file}")
print(f"行数: {content.count(chr(10))}")
