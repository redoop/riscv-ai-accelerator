#!/usr/bin/env python3
"""
预处理 Verilog 文件以兼容 Yosys 综合
- 移除 automatic 变量声明（将其转换为简单赋值）
- 注释 DPI-C 导入
"""

import re
import sys

def preprocess_verilog(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 1. 注释 DPI-C 导入
    content = re.sub(r'^(\s*)import "DPI-C"', r'\1// import "DPI-C"', content, flags=re.MULTILINE)
    
    # 2. 替换 DPI 函数调用
    content = re.sub(r'mrom_read\(raddr,\s*rdata\)', r'rdata = 32\'h0', content)
    
    # 3. 移除 automatic 变量声明（包括多行）
    # 匹配 "automatic logic [x:y] var = ... ;" (可能跨多行)
    content = re.sub(
        r'automatic\s+logic\s+(\[[^\]]+\])?\s+(\w+)\s*=',
        r'/* automatic logic \1 */ \2 =',
        content
    )
    
    # 4. 移除剩余的 automatic 关键字
    content = re.sub(r'\bautomatic\s+', '', content)
    
    with open(output_file, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.v> <output.v>")
        sys.exit(1)
    
    preprocess_verilog(sys.argv[1], sys.argv[2])
