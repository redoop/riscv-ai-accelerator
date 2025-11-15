#!/usr/bin/env python3
"""
生成静态波形 HTML 页面
将波形图直接嵌入 HTML，无需服务器即可查看
"""

import base64
import time
from pathlib import Path
from wave_viewer import VCDParser
from wave_renderer import WaveformRenderer
import argparse

def generate_static_html(vcd_file, output_file, signals=None, max_signals=20, max_points=3000):
    """
    生成静态 HTML 页面
    
    Args:
        vcd_file: VCD 文件路径
        output_file: 输出 HTML 文件路径
        signals: 要显示的信号列表（None 表示自动选择）
        max_signals: 最大信号数量
        max_points: 每个信号的最大采样点数
    """
    
    print("=" * 70)
    print("生成静态波形 HTML 页面")
    print("=" * 70)
    print(f"VCD 文件: {vcd_file}")
    print(f"输出文件: {output_file}")
    print()
    
    # 解析 VCD 文件
    print("步骤 1: 解析 VCD 文件...")
    print("-" * 70)
    start_time = time.time()
    
    parser = VCDParser(vcd_file)
    parser.parse_header()
    parser.parse_values_fast()
    
    parse_time = time.time() - start_time
    file_size_mb = vcd_file.stat().st_size / (1024 * 1024)
    
    print(f"✓ 解析完成: {parse_time:.2f}s")
    print(f"  - 文件大小: {file_size_mb:.2f} MB")
    print(f"  - 信号数量: {len(parser.signals)}")
    print(f"  - 时间范围: 0 - {parser.max_time} {parser.timescale}")
    print()
    
    # 选择信号
    print("步骤 2: 选择信号...")
    print("-" * 70)
    
    if signals is None:
        # 自动选择关键信号
        all_symbols = list(parser.signals.keys())
        
        # 优先选择顶层信号和关键信号
        priority_keywords = ['clk', 'clock', 'reset', 'trap', 'valid', 'ready', 'irq']
        priority_signals = []
        other_signals = []
        
        for symbol in all_symbols:
            info = parser.signals[symbol]
            name_lower = info['full_name'].lower()
            
            if any(kw in name_lower for kw in priority_keywords):
                priority_signals.append(symbol)
            else:
                other_signals.append(symbol)
        
        # 组合信号列表
        selected_symbols = priority_signals[:max_signals]
        if len(selected_symbols) < max_signals:
            selected_symbols.extend(other_signals[:max_signals - len(selected_symbols)])
    else:
        selected_symbols = signals[:max_signals]
    
    print(f"选择信号数: {len(selected_symbols)}")
    for i, symbol in enumerate(selected_symbols[:10]):
        info = parser.signals[symbol]
        print(f"  {i+1}. {info['full_name']} [{info['width']}]")
    if len(selected_symbols) > 10:
        print(f"  ... 还有 {len(selected_symbols) - 10} 个信号")
    print()
    
    # 加载信号数据
    print("步骤 3: 加载信号数据...")
    print("-" * 70)
    start_time = time.time()
    
    signal_data = parser.parse_signal_range(
        selected_symbols,
        0,
        parser.max_time,
        max_points=max_points
    )
    
    load_time = time.time() - start_time
    total_points = sum(len(signal_data[s]['values']) for s in signal_data)
    
    print(f"✓ 数据加载完成: {load_time:.2f}s")
    print(f"  - 总数据点: {total_points}")
    print(f"  - 平均点数: {total_points // len(selected_symbols)}/信号")
    print()
    
    # 渲染波形图
    print("步骤 4: 渲染波形图...")
    print("-" * 70)
    start_time = time.time()
    
    renderer = WaveformRenderer(width=1600, height=800, dpi=100)
    img_buf = renderer.render_to_png(
        signal_data,
        parser.max_time,
        parser.timescale,
        0,
        parser.max_time
    )
    
    render_time = time.time() - start_time
    img_size_kb = len(img_buf.getvalue()) / 1024
    
    print(f"✓ 渲染完成: {render_time:.2f}s")
    print(f"  - 图片大小: {img_size_kb:.2f} KB")
    print()
    
    # 将图片转换为 base64
    print("步骤 5: 生成 HTML...")
    print("-" * 70)
    
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    
    # 生成信号列表 HTML
    signal_list_html = ""
    for symbol in selected_symbols:
        info = parser.signals[symbol]
        signal_list_html += f"""
        <tr>
            <td>{info['name']}</td>
            <td>{info['full_name']}</td>
            <td>{info['width']}</td>
        </tr>
        """
    
    # 生成完整 HTML
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>波形查看 - {vcd_file.name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .header {{
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #3e3e42;
        }}
        
        .header h1 {{
            color: #4ec9b0;
            margin-bottom: 10px;
            font-size: 24px;
        }}
        
        .info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .info-item {{
            background: #252526;
            padding: 10px 15px;
            border-radius: 4px;
            border: 1px solid #3e3e42;
        }}
        
        .info-label {{
            color: #858585;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            color: #d4d4d4;
            font-size: 16px;
            font-weight: 500;
        }}
        
        .waveform {{
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #3e3e42;
        }}
        
        .waveform h2 {{
            color: #4ec9b0;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        .waveform img {{
            width: 100%;
            display: block;
            border-radius: 4px;
            background: #1e1e1e;
        }}
        
        .signals {{
            background: #2d2d30;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #3e3e42;
        }}
        
        .signals h2 {{
            color: #4ec9b0;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #3e3e42;
        }}
        
        th {{
            background: #252526;
            color: #858585;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        
        td {{
            color: #d4d4d4;
            font-size: 13px;
        }}
        
        tr:hover {{
            background: #252526;
        }}
        
        .footer {{
            text-align: center;
            color: #858585;
            margin-top: 30px;
            padding: 20px;
            font-size: 12px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .badge-success {{
            background: #4ec9b0;
            color: #1e1e1e;
        }}
        
        .badge-info {{
            background: #007acc;
            color: white;
        }}
        
        .controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }}
        
        button {{
            padding: 8px 15px;
            background: #3c3c3c;
            color: #d4d4d4;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        
        button:hover {{
            background: #505050;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 VCD 波形查看器 <span class="badge badge-success">静态页面</span></h1>
            <p style="color: #858585; margin-top: 5px;">
                此页面为独立静态 HTML，无需服务器即可查看波形
            </p>
            
            <div class="info">
                <div class="info-item">
                    <div class="info-label">VCD 文件</div>
                    <div class="info-value">{vcd_file.name}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">文件大小</div>
                    <div class="info-value">{file_size_mb:.2f} MB</div>
                </div>
                <div class="info-item">
                    <div class="info-label">时间范围</div>
                    <div class="info-value">0 - {parser.max_time} {parser.timescale}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">显示信号</div>
                    <div class="info-value">{len(selected_symbols)} / {len(parser.signals)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">生成时间</div>
                    <div class="info-value">{time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </div>
            
            <div class="controls">
                <button onclick="downloadImage()">💾 下载波形图</button>
                <button onclick="window.print()">🖨️ 打印</button>
            </div>
        </div>
        
        <div class="waveform">
            <h2>波形图</h2>
            <img id="waveformImage" src="data:image/png;base64,{img_base64}" alt="Waveform">
        </div>
        
        <div class="signals">
            <h2>信号列表</h2>
            <table>
                <thead>
                    <tr>
                        <th>信号名称</th>
                        <th>完整路径</th>
                        <th>位宽</th>
                    </tr>
                </thead>
                <tbody>
                    {signal_list_html}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 5px;">
                解析时间: {parse_time:.2f}s | 
                数据加载: {load_time:.2f}s | 
                图片渲染: {render_time:.2f}s | 
                总计: {parse_time + load_time + render_time:.2f}s
            </p>
        </div>
    </div>
    
    <script>
        function downloadImage() {{
            const img = document.getElementById('waveformImage');
            const link = document.createElement('a');
            link.href = img.src;
            link.download = 'waveform_{vcd_file.stem}.png';
            link.click();
        }}
    </script>
</body>
</html>
"""
    
    # 写入文件
    output_file.write_text(html_content, encoding='utf-8')
    html_size_kb = output_file.stat().st_size / 1024
    
    print(f"✓ HTML 生成完成")
    print(f"  - HTML 大小: {html_size_kb:.2f} KB")
    print()
    
    # 总结
    print("=" * 70)
    print("✓ 静态页面生成成功！")
    print("=" * 70)
    print(f"输出文件: {output_file.absolute()}")
    print(f"文件大小: {html_size_kb:.2f} KB")
    print()
    print("使用方法:")
    print(f"  1. 直接在浏览器中打开: file://{output_file.absolute()}")
    print(f"  2. 或双击文件: {output_file.name}")
    print()
    print("性能统计:")
    print(f"  - VCD 文件: {file_size_mb:.2f} MB")
    print(f"  - 处理时间: {parse_time + load_time + render_time:.2f}s")
    print(f"  - 输出大小: {html_size_kb:.2f} KB")
    print(f"  - 压缩比: {file_size_mb * 1024 / html_size_kb:.1f}:1")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='生成静态波形 HTML 页面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动选择信号生成
  python generate_static_wave.py waves/post_syn.vcd
  
  # 指定输出文件
  python generate_static_wave.py waves/post_syn.vcd -o waveform.html
  
  # 限制信号数量
  python generate_static_wave.py waves/post_syn.vcd --max-signals 10
  
  # 提高采样率
  python generate_static_wave.py waves/post_syn.vcd --max-points 5000
        """
    )
    
    parser.add_argument(
        'vcd_file',
        type=Path,
        help='VCD 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='输出 HTML 文件路径（默认: waveform.html）'
    )
    
    parser.add_argument(
        '--max-signals',
        type=int,
        default=20,
        help='最大信号数量（默认: 20）'
    )
    
    parser.add_argument(
        '--max-points',
        type=int,
        default=3000,
        help='每个信号的最大采样点数（默认: 3000）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.vcd_file.exists():
        print(f"❌ 错误: VCD 文件不存在: {args.vcd_file}")
        return 1
    
    # 确定输出文件
    if args.output:
        output_file = args.output
    else:
        output_file = Path(f"waveform_{args.vcd_file.stem}.html")
    
    # 生成静态页面
    try:
        generate_static_html(
            args.vcd_file,
            output_file,
            max_signals=args.max_signals,
            max_points=args.max_points
        )
        return 0
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
