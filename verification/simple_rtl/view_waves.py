#!/usr/bin/env python3
"""
简单波形查看器 - 查看当前目录的VCD文件
"""

import os
import sys
import subprocess
from pathlib import Path

def find_vcd_files():
    """查找当前目录的VCD文件"""
    vcd_files = list(Path(".").glob("*.vcd"))
    print(f"📊 找到 {len(vcd_files)} 个波形文件:")
    for vcd_file in vcd_files:
        file_size = vcd_file.stat().st_size
        print(f"  - {vcd_file.name} ({file_size} bytes)")
    return vcd_files

def analyze_vcd_file(vcd_file):
    """分析VCD文件内容"""
    try:
        print(f"\n🔍 分析波形文件: {vcd_file.name}")
        print("=" * 50)
        
        with open(vcd_file, 'r') as f:
            lines = f.readlines()
        
        # 统计信息
        total_lines = len(lines)
        signal_count = 0
        time_steps = 0
        
        # 分析VCD内容
        for line in lines[:20]:  # 显示前20行
            line = line.strip()
            if line.startswith('$var'):
                signal_count += 1
            elif line.startswith('#'):
                time_steps += 1
            print(f"  {line}")
        
        if total_lines > 20:
            print(f"  ... (还有 {total_lines - 20} 行)")
        
        print(f"\n📈 波形统计:")
        print(f"  总行数: {total_lines}")
        print(f"  信号数量: {signal_count}")
        print(f"  时间步数: {time_steps}")
        
    except Exception as e:
        print(f"❌ 分析VCD文件失败: {e}")

def create_html_viewer(vcd_file):
    """创建HTML波形查看器"""
    try:
        print(f"\n🌐 创建HTML波形查看器: {vcd_file.name}")
        
        html_file = Path(f"{vcd_file.stem}_waveform.html")
        
        # 读取VCD文件内容
        with open(vcd_file, 'r') as f:
            vcd_content = f.read()
        
        # 创建HTML内容
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RTL波形查看器 - {vcd_file.name}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: #2c3e50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .vcd-content {{ background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 12px; overflow-x: auto; white-space: pre-wrap; max-height: 600px; overflow-y: auto; border: 1px solid #ddd; }}
        .info {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 RTL波形查看器</h1>
            <p><strong>文件:</strong> {vcd_file.name}</p>
            <p><strong>路径:</strong> {vcd_file.absolute()}</p>
            <p><strong>大小:</strong> {vcd_file.stat().st_size} bytes</p>
        </div>
        
        <div class="info">
            <h3>📊 VCD文件内容</h3>
            <div class="vcd-content">{vcd_content}</div>
        </div>
        
        <div class="info">
            <h3>💡 使用说明</h3>
            <ul>
                <li>这是一个简单的VCD文件查看器，用于快速预览波形信息</li>
                <li>推荐使用专业工具如GTKWave查看完整的时序波形图</li>
                <li>安装GTKWave: <code>brew install --cask gtkwave</code></li>
                <li>或者使用在线波形查看器: <a href="https://wavedrom.com/" target="_blank">WaveDrom</a></li>
            </ul>
        </div>
    </div>
</body>
</html>"""
        
        # 写入HTML文件
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML波形查看器已创建: {html_file}")
        print(f"💡 在浏览器中打开: open {html_file}")
        
        return html_file
        
    except Exception as e:
        print(f"❌ 创建HTML查看器失败: {e}")
        return None

def main():
    """主函数"""
    print("🌊 RTL波形查看器")
    print("=" * 40)
    
    # 查找VCD文件
    vcd_files = find_vcd_files()
    
    if not vcd_files:
        print("❌ 未找到VCD波形文件")
        return
    
    # 交互式选择
    while True:
        print(f"\n📊 可用的波形文件:")
        for i, vcd_file in enumerate(vcd_files):
            print(f"  {i+1}. {vcd_file.name}")
        
        print(f"\n🛠️ 操作选项:")
        print("  a) 分析所有波形文件")
        print("  h) 创建HTML波形查看器")
        print("  q) 退出")
        
        choice = input("\n请选择操作: ").strip().lower()
        
        if choice == 'q':
            print("👋 再见!")
            break
        elif choice == 'a':
            for vcd_file in vcd_files:
                analyze_vcd_file(vcd_file)
        elif choice == 'h':
            for vcd_file in vcd_files:
                create_html_viewer(vcd_file)
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()