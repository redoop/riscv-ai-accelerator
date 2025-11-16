#!/usr/bin/env python3
"""
测试图片渲染功能
"""

import time
from pathlib import Path
from wave_viewer import VCDParser
from wave_renderer import WaveformRenderer

def test_image_rendering():
    """测试波形图片渲染"""
    
    vcd_file = Path("waves/post_syn.vcd")
    
    if not vcd_file.exists():
        print(f"❌ VCD 文件不存在: {vcd_file}")
        return
    
    file_size_mb = vcd_file.stat().st_size / (1024 * 1024)
    print(f"📁 VCD 文件: {vcd_file}")
    print(f"📊 文件大小: {file_size_mb:.2f} MB")
    print()
    
    # 解析 VCD
    print("步骤 1: 加载 VCD 文件")
    print("-" * 60)
    parser = VCDParser(vcd_file)
    
    start = time.time()
    parser.parse_header()
    parser.parse_values_fast()
    load_time = time.time() - start
    
    print(f"✓ 加载完成: {load_time:.2f}s")
    print(f"  - 信号数量: {len(parser.signals)}")
    print(f"  - 时间范围: 0 - {parser.max_time} {parser.timescale}")
    print()
    
    # 选择测试信号
    print("步骤 2: 选择测试信号")
    print("-" * 60)
    test_symbols = list(parser.signals.keys())[:10]  # 选择前 10 个信号
    print(f"选择信号数: {len(test_symbols)}")
    for i, symbol in enumerate(test_symbols[:5]):
        info = parser.signals[symbol]
        print(f"  {i+1}. {info['full_name']} [{info['width']}]")
    if len(test_symbols) > 5:
        print(f"  ... 还有 {len(test_symbols) - 5} 个信号")
    print()
    
    # 加载信号数据
    print("步骤 3: 加载信号数据")
    print("-" * 60)
    start = time.time()
    signal_data = parser.parse_signal_range(test_symbols, 0, parser.max_time, max_points=3000)
    data_time = time.time() - start
    
    total_points = sum(len(signal_data[s]['values']) for s in signal_data)
    print(f"✓ 数据加载完成: {data_time:.2f}s")
    print(f"  - 总数据点: {total_points}")
    print(f"  - 平均点数: {total_points // len(test_symbols)}/信号")
    print()
    
    # 渲染图片
    print("步骤 4: 渲染波形图片")
    print("-" * 60)
    renderer = WaveformRenderer(width=1600, height=800, dpi=100)
    
    start = time.time()
    img_buf = renderer.render_to_png(
        signal_data,
        parser.max_time,
        parser.timescale,
        0,
        parser.max_time
    )
    render_time = time.time() - start
    
    img_size_kb = len(img_buf.getvalue()) / 1024
    print(f"✓ 渲染完成: {render_time:.2f}s")
    print(f"  - 图片大小: {img_size_kb:.2f} KB")
    print()
    
    # 保存测试图片
    output_file = Path("waves/test_waveform.png")
    with open(output_file, 'wb') as f:
        f.write(img_buf.getvalue())
    print(f"✓ 测试图片已保存: {output_file}")
    print()
    
    # 性能总结
    print("=" * 60)
    print("性能总结")
    print("=" * 60)
    print(f"VCD 文件大小: {file_size_mb:.2f} MB")
    print(f"加载时间: {load_time:.2f}s")
    print(f"数据提取: {data_time:.2f}s")
    print(f"图片渲染: {render_time:.2f}s")
    print(f"总时间: {load_time + data_time + render_time:.2f}s")
    print()
    print(f"✓ 图片模式可以处理 {file_size_mb:.0f}MB 的 VCD 文件")
    print(f"✓ 用户只需等待 ~{render_time:.0f}秒 即可看到波形")
    print()

if __name__ == '__main__':
    test_image_rendering()
