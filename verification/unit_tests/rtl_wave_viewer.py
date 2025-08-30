#!/usr/bin/env python3
"""
RTL波形查看器 - 专门用于查看RTL目录生成的波形文件
支持TPU、VPU和AI加速器的波形分析
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

class RTLWaveformViewer:
    """RTL波形查看器"""
    
    def __init__(self):
        self.vcd_files = []
        self._find_vcd_files()
    
    def _find_vcd_files(self):
        """查找VCD波形文件"""
        self.vcd_files = list(Path(".").glob("*.vcd"))
        print(f"📊 找到 {len(self.vcd_files)} 个RTL波形文件:")
        for vcd_file in self.vcd_files:
            file_size = vcd_file.stat().st_size
            print(f"  📈 {vcd_file.name} ({file_size} bytes)")
    
    def parse_rtl_vcd(self, vcd_file):
        """解析RTL VCD文件并进行深度分析"""
        try:
            print(f"\n🔍 深度分析RTL波形文件: {vcd_file.name}")
            
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            # 解析VCD结构
            signals = {}
            time_values = {}
            current_time = 0
            
            # 第一遍：提取信号定义
            for line in lines:
                line = line.strip()
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_type = parts[1]
                        signal_width = parts[2]
                        signal_id = parts[3]
                        signal_name = parts[4]
                        signals[signal_id] = {
                            'name': signal_name,
                            'type': signal_type,
                            'width': signal_width,
                            'values': []
                        }
            
            # 第二遍：提取时间和值变化
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    current_time = int(line[1:])
                elif line and not line.startswith('$'):
                    # 解析信号值变化
                    if line[0] in '01xz':
                        # 单bit信号
                        value = line[0]
                        signal_id = line[1:]
                        if signal_id in signals:
                            signals[signal_id]['values'].append((current_time, value))
                    elif line[0] == 'b':
                        # 多bit信号
                        parts = line.split()
                        if len(parts) >= 2:
                            value = parts[0][1:]  # 去掉'b'前缀
                            signal_id = parts[1]
                            if signal_id in signals:
                                # 转换二进制到十六进制显示
                                try:
                                    hex_value = hex(int(value, 2))
                                    signals[signal_id]['values'].append((current_time, hex_value))
                                except:
                                    signals[signal_id]['values'].append((current_time, value))
            
            return signals
            
        except Exception as e:
            print(f"❌ 解析RTL VCD文件失败: {e}")
            return {}
    
    def create_rtl_html_viewer(self, vcd_file):
        """创建RTL专用HTML波形查看器"""
        try:
            print(f"\n🌐 创建RTL HTML波形查看器: {vcd_file.name}")
            
            # 解析VCD数据
            signals = self.parse_rtl_vcd(vcd_file)
            
            if not signals:
                print("❌ 无法解析RTL VCD数据")
                return None
            
            # 分类信号
            clock_signals = []
            control_signals = []
            data_signals = []
            status_signals = []
            
            for signal_id, signal_data in signals.items():
                name = signal_data['name'].lower()
                if 'clk' in name or 'clock' in name:
                    clock_signals.append((signal_id, signal_data))
                elif any(x in name for x in ['rst', 'reset', 'enable', 'valid', 'ready']):
                    control_signals.append((signal_id, signal_data))
                elif any(x in name for x in ['data', 'addr', 'wdata', 'rdata']):
                    data_signals.append((signal_id, signal_data))
                else:
                    status_signals.append((signal_id, signal_data))
            
            # 转换为JavaScript数据格式
            js_data = {
                'clock_signals': [{'id': sid, **sdata} for sid, sdata in clock_signals],
                'control_signals': [{'id': sid, **sdata} for sid, sdata in control_signals],
                'data_signals': [{'id': sid, **sdata} for sid, sdata in data_signals],
                'status_signals': [{'id': sid, **sdata} for sid, sdata in status_signals]
            }
            
            html_file = Path(f"{vcd_file.stem}_rtl_waveform.html")
            
            # 创建RTL专用HTML内容
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RTL波形查看器 - {vcd_file.name}</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1600px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white; 
            padding: 25px; 
            text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .controls {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{ 
            display: flex; 
            align-items: center; 
            gap: 10px;
        }}
        .control-group label {{ 
            font-weight: bold; 
            color: #495057;
        }}
        .control-group input, .control-group select {{ 
            padding: 8px 12px; 
            border: 1px solid #ced4da; 
            border-radius: 5px;
            font-size: 14px;
        }}
        .btn {{ 
            padding: 10px 20px; 
            background: #1e3c72; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .btn:hover {{ background: #2a5298; }}
        .btn.secondary {{ background: #6c757d; }}
        .btn.secondary:hover {{ background: #545b62; }}
        .signal-category {{ 
            margin: 20px; 
            border: 1px solid #dee2e6; 
            border-radius: 8px;
            overflow: hidden;
        }}
        .category-header {{ 
            background: #e9ecef; 
            padding: 15px; 
            font-weight: bold; 
            color: #2c3e50;
            border-bottom: 1px solid #dee2e6;
        }}
        .signal-row {{ 
            display: flex; 
            align-items: center; 
            padding: 10px 15px; 
            border-bottom: 1px solid #f8f9fa;
        }}
        .signal-row:hover {{ background: #f8f9fa; }}
        .signal-name {{ 
            width: 200px; 
            font-weight: bold; 
            color: #2c3e50;
            font-family: monospace;
        }}
        .signal-canvas {{ 
            flex: 1; 
            height: 60px; 
            border: 1px solid #dee2e6; 
            border-radius: 5px;
            background: white;
            margin: 0 10px;
        }}
        .signal-info {{ 
            width: 150px; 
            font-size: 12px; 
            color: #6c757d;
            text-align: right;
        }}
        .timeline {{ 
            height: 40px; 
            background: #2c3e50; 
            color: white; 
            display: flex; 
            align-items: center; 
            padding: 0 20px;
            font-family: monospace;
            font-size: 14px;
        }}
        .stats {{ 
            background: #e9ecef; 
            padding: 20px; 
            border-top: 1px solid #dee2e6;
        }}
        .stats-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px;
        }}
        .stat-card {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-card h4 {{ 
            margin: 0 0 10px 0; 
            color: #2c3e50;
        }}
        .stat-value {{ 
            font-size: 1.5em; 
            font-weight: bold; 
            color: #1e3c72;
        }}
        .clock {{ color: #e74c3c; }}
        .control {{ color: #f39c12; }}
        .data {{ color: #3498db; }}
        .status {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 RTL波形查看器</h1>
            <p>文件: {vcd_file.name} | 大小: {vcd_file.stat().st_size} bytes</p>
            <p>RTL模块: TPU/VPU AI加速器</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>时间范围:</label>
                <input type="number" id="timeStart" value="0" placeholder="开始时间(ps)">
                <span>-</span>
                <input type="number" id="timeEnd" value="1000000" placeholder="结束时间(ps)">
            </div>
            <div class="control-group">
                <label>缩放:</label>
                <select id="zoomLevel">
                    <option value="1">1x</option>
                    <option value="2">2x</option>
                    <option value="4">4x</option>
                    <option value="8">8x</option>
                    <option value="16">16x</option>
                </select>
            </div>
            <button class="btn" onclick="updateWaveforms()">🔄 更新波形</button>
            <button class="btn secondary" onclick="exportData()">💾 导出数据</button>
            <button class="btn secondary" onclick="analyzeSignals()">📊 信号分析</button>
        </div>
        
        <div class="timeline" id="timeline">
            时间轴将在这里显示
        </div>
        
        <div class="signal-category">
            <div class="category-header clock">🕐 时钟信号</div>
            <div id="clockSignals"></div>
        </div>
        
        <div class="signal-category">
            <div class="category-header control">🎛️ 控制信号</div>
            <div id="controlSignals"></div>
        </div>
        
        <div class="signal-category">
            <div class="category-header data">📊 数据信号</div>
            <div id="dataSignals"></div>
        </div>
        
        <div class="signal-category">
            <div class="category-header status">📈 状态信号</div>
            <div id="statusSignals"></div>
        </div>
        
        <div class="stats">
            <h3>📊 RTL波形统计信息</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>时钟信号</h4>
                    <div class="stat-value" id="clockCount">{len(js_data['clock_signals'])}</div>
                </div>
                <div class="stat-card">
                    <h4>控制信号</h4>
                    <div class="stat-value" id="controlCount">{len(js_data['control_signals'])}</div>
                </div>
                <div class="stat-card">
                    <h4>数据信号</h4>
                    <div class="stat-value" id="dataCount">{len(js_data['data_signals'])}</div>
                </div>
                <div class="stat-card">
                    <h4>状态信号</h4>
                    <div class="stat-value" id="statusCount">{len(js_data['status_signals'])}</div>
                </div>
                <div class="stat-card">
                    <h4>总信号数</h4>
                    <div class="stat-value">{len(signals)}</div>
                </div>
                <div class="stat-card">
                    <h4>文件大小</h4>
                    <div class="stat-value">{vcd_file.stat().st_size} B</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // RTL波形数据
        const rtlData = {json.dumps(js_data, indent=2)};
        
        // 全局变量
        let currentTimeStart = 0;
        let currentTimeEnd = 1000000;
        let currentZoom = 1;
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            calculateTimeRange();
            renderAllSignals();
        }});
        
        function calculateTimeRange() {{
            let maxTime = 0;
            
            // 计算所有信号的最大时间
            Object.values(rtlData).forEach(category => {{
                category.forEach(signal => {{
                    signal.values.forEach(([time, value]) => {{
                        maxTime = Math.max(maxTime, time);
                    }});
                }});
            }});
            
            document.getElementById('timeEnd').value = maxTime;
            currentTimeEnd = maxTime;
        }}
        
        function renderAllSignals() {{
            renderSignalCategory('clockSignals', rtlData.clock_signals, '#e74c3c');
            renderSignalCategory('controlSignals', rtlData.control_signals, '#f39c12');
            renderSignalCategory('dataSignals', rtlData.data_signals, '#3498db');
            renderSignalCategory('statusSignals', rtlData.status_signals, '#27ae60');
            updateTimeline();
        }}
        
        function renderSignalCategory(containerId, signals, color) {{
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            signals.forEach(signal => {{
                const signalRow = document.createElement('div');
                signalRow.className = 'signal-row';
                
                const signalName = document.createElement('div');
                signalName.className = 'signal-name';
                signalName.textContent = signal.name;
                
                const canvas = document.createElement('canvas');
                canvas.className = 'signal-canvas';
                canvas.width = 1000;
                canvas.height = 60;
                
                const signalInfo = document.createElement('div');
                signalInfo.className = 'signal-info';
                signalInfo.innerHTML = `
                    类型: ${{signal.type}}<br>
                    宽度: ${{signal.width}}<br>
                    变化: ${{signal.values.length}}
                `;
                
                signalRow.appendChild(signalName);
                signalRow.appendChild(canvas);
                signalRow.appendChild(signalInfo);
                container.appendChild(signalRow);
                
                // 绘制波形
                drawRTLWaveform(canvas, signal, color);
            }});
        }}
        
        function drawRTLWaveform(canvas, signal, color) {{
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            
            // 清空画布
            ctx.clearRect(0, 0, width, height);
            
            // 绘制背景网格
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                const x = (i / 10) * width;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }}
            
            // 绘制波形
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.font = '10px monospace';
            
            if (signal.values.length === 0) return;
            
            let lastValue = '0';
            let lastX = 0;
            let lastY = height - 10;
            
            ctx.beginPath();
            
            signal.values.forEach(([time, value], index) => {{
                const x = ((time - currentTimeStart) / (currentTimeEnd - currentTimeStart)) * width;
                
                if (x >= 0 && x <= width) {{
                    let y;
                    
                    // 根据信号类型确定Y位置
                    if (signal.width === '1') {{
                        // 单bit信号
                        y = value === '1' || value === '0x1' ? 10 : height - 10;
                    }} else {{
                        // 多bit信号，显示为数字波形
                        y = height / 2;
                    }}
                    
                    if (index === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        // 绘制从上一个值到当前时间的水平线
                        ctx.lineTo(x, lastY);
                        // 绘制垂直跳变线
                        ctx.lineTo(x, y);
                    }}
                    
                    // 绘制值标签
                    if (signal.width !== '1' || (x - lastX) > 50) {{
                        ctx.fillStyle = color;
                        let displayValue = value;
                        if (typeof value === 'string' && value.startsWith('0x')) {{
                            displayValue = value;
                        }} else if (signal.width !== '1') {{
                            displayValue = `0x${{value}}`;
                        }}
                        ctx.fillText(displayValue, x + 2, y - 5);
                    }}
                    
                    lastX = x;
                    lastY = y;
                    lastValue = value;
                }}
            }});
            
            // 延伸到画布末尾
            ctx.lineTo(width, lastY);
            ctx.stroke();
        }}
        
        function updateTimeline() {{
            const timeline = document.getElementById('timeline');
            const timeRange = currentTimeEnd - currentTimeStart;
            const step = Math.pow(10, Math.floor(Math.log10(timeRange / 10)));
            
            let timelineText = '';
            for (let t = currentTimeStart; t <= currentTimeEnd; t += step) {{
                timelineText += `${{t}}ps `;
            }}
            
            timeline.textContent = timelineText;
        }}
        
        function updateWaveforms() {{
            currentTimeStart = parseInt(document.getElementById('timeStart').value) || 0;
            currentTimeEnd = parseInt(document.getElementById('timeEnd').value) || 1000000;
            currentZoom = parseInt(document.getElementById('zoomLevel').value) || 1;
            
            renderAllSignals();
        }}
        
        function exportData() {{
            const dataStr = JSON.stringify(rtlData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = '{vcd_file.stem}_rtl_waveform_data.json';
            link.click();
            
            URL.revokeObjectURL(url);
        }}
        
        function analyzeSignals() {{
            let analysis = "RTL信号分析报告\\n";
            analysis += "==================\\n\\n";
            
            Object.entries(rtlData).forEach(([category, signals]) => {{
                analysis += `${{category.replace('_', ' ').toUpperCase()}}: ${{signals.length}} 个信号\\n`;
                signals.forEach(signal => {{
                    analysis += `  - ${{signal.name}} (${{signal.type}}, ${{signal.width}}bit, ${{signal.values.length}} 变化)\\n`;
                }});
                analysis += "\\n";
            }});
            
            alert(analysis);
        }}
    </script>
</body>
</html>"""
            
            # 写入HTML文件
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ RTL HTML波形查看器已创建: {html_file}")
            print(f"🌐 在浏览器中打开: open {html_file}")
            
            return html_file
            
        except Exception as e:
            print(f"❌ 创建RTL波形查看器失败: {e}")
            return None
    
    def run(self):
        """运行RTL波形查看器"""
        print("🚀 启动RTL波形查看器")
        print("=" * 50)
        
        if not self.vcd_files:
            print("❌ 没有找到RTL VCD文件")
            return
        
        # 为每个VCD文件创建RTL查看器
        for vcd_file in self.vcd_files:
            self.create_rtl_html_viewer(vcd_file)
        
        print(f"\n🎉 已为 {len(self.vcd_files)} 个RTL波形文件创建查看器!")
        print("💡 这些波形文件来自真正的RTL硬件描述代码")
        print("🔧 包含TPU MAC单元、计算数组和控制器的硬件级仿真结果")

if __name__ == "__main__":
    viewer = RTLWaveformViewer()
    viewer.run()