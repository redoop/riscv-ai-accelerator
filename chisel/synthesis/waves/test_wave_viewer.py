#!/usr/bin/env python3
"""
测试波形查看器的性能和功能
"""

import time
from pathlib import Path
from wave_viewer import VCDParser

def test_vcd_parser():
    """测试 VCD 解析器性能"""
    
    vcd_file = Path("waves/post_syn.vcd")
    
    if not vcd_file.exists():
        print(f"❌ VCD 文件不存在: {vcd_file}")
        return
    
    file_size_mb = vcd_file.stat().st_size / (1024 * 1024)
    print(f"📁 VCD 文件: {vcd_file}")
    print(f"📊 文件大小: {file_size_mb:.2f} MB")
    print()
    
    # 测试 1: 快速加载（只解析头部和时间范围）
    print("测试 1: 快速加载模式")
    print("-" * 60)
    parser = VCDParser(vcd_file)
    
    start = time.time()
    parser.parse_header()
    header_time = time.time() - start
    print(f"✓ 解析头部: {header_time:.2f}s")
    print(f"  - 信号数量: {len(parser.signals)}")
    
    start = time.time()
    parser.parse_values_fast()
    fast_time = time.time() - start
    print(f"✓ 快速扫描: {fast_time:.2f}s")
    print(f"  - 时间范围: 0 - {parser.max_time} {parser.timescale}")
    print(f"  - 总加载时间: {header_time + fast_time:.2f}s")
    print()
    
    # 测试 2: 范围加载（加载部分信号数据）
    print("测试 2: 范围加载模式")
    print("-" * 60)
    
    # 选择前 5 个信号
    test_symbols = list(parser.signals.keys())[:5]
    print(f"测试信号: {len(test_symbols)} 个")
    
    # 测试不同的时间范围
    test_ranges = [
        (0, parser.max_time, "全范围"),
        (0, parser.max_time // 2, "前半段"),
        (parser.max_time // 4, parser.max_time // 2, "1/4 到 1/2"),
    ]
    
    for start_time, end_time, desc in test_ranges:
        start = time.time()
        data = parser.parse_signal_range(test_symbols, start_time, end_time, max_points=2000)
        load_time = time.time() - start
        
        total_points = sum(len(data[s]['values']) for s in data)
        avg_points = total_points // len(test_symbols) if test_symbols else 0
        
        print(f"  {desc}:")
        print(f"    - 时间: {load_time:.2f}s")
        print(f"    - 数据点: {total_points} (平均 {avg_points}/信号)")
        print(f"    - 压缩比: 1:{(end_time - start_time) // avg_points if avg_points > 0 else 0}")
    
    print()
    
    # 测试 3: 不同采样率
    print("测试 3: 不同采样率")
    print("-" * 60)
    
    max_points_list = [500, 1000, 2000, 5000]
    
    for max_points in max_points_list:
        start = time.time()
        data = parser.parse_signal_range(test_symbols, 0, parser.max_time, max_points=max_points)
        load_time = time.time() - start
        
        total_points = sum(len(data[s]['values']) for s in data)
        avg_points = total_points // len(test_symbols) if test_symbols else 0
        
        print(f"  最大点数 {max_points}:")
        print(f"    - 时间: {load_time:.2f}s")
        print(f"    - 实际点数: {total_points} (平均 {avg_points}/信号)")
    
    print()
    print("=" * 60)
    print("✓ 所有测试完成")
    print()
    print("性能总结:")
    print(f"  - 文件大小: {file_size_mb:.2f} MB")
    print(f"  - 快速加载: {header_time + fast_time:.2f}s")
    print(f"  - 信号数量: {len(parser.signals)}")
    print(f"  - 适合 Web 查看: {'是' if file_size_mb < 500 else '建议使用更强大的工具'}")

if __name__ == '__main__':
    test_vcd_parser()
