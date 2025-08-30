#!/usr/bin/env python3
"""
高级JavaScript波形查看器 - 当前目录版本
支持更多交互功能和波形分析
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

class AdvancedJSWaveformViewer:
    """高级JavaScript波形查看器"""
    
    def __init__(self):
        self.vcd_files = []
        self._find_vcd_files()
    
    def _find_vcd_files(self):
        """查找VCD波形文件"""
        self.vcd_files = list(Path(".").glob("*.vcd"))
        print(f"📊 找到 {len(self.vcd_files)} 个波形文件:")
        for vcd_file in self.vcd_files:
            file_size = vcd_file.stat().st_size
            print(f"  📈 {vcd_file.name} ({file_size} bytes)")
    
    def parse_vcd_with_analysis(self, vcd_file):
        """解析VCD文件并进行深度分析"""
        try:
            print(f"\n🔍 深度分析波形文件: {vcd_file.name}")
            
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
                                signals[signal_id]['values'].append((current_time, value))
            
            return signals
            
        except Exception as e:
            print(f"❌ 解析VCD文件失败: {e}")
            return {}
    
    def create_advanced_js_viewer(self, vcd_file):
        """创建高级JavaScript波形查看器"""
        try:
            print(f"\n🌐 创建高级JavaScript波形查看器: {vcd_file.name}")
            
            # 解析VCD数据
            signals = self.parse_vcd_with_analysis(vcd_file)
            
            if not signals:
                print("❌ 无法解析VCD数据")
                return None
            
            # 转换为JavaScript数据格式
            js_signals = []
            for signal_id, signal_data in signals.items():
                js_signal = {
                    'name': signal_data['name'],
                    'type': signal_data['type'],
                    'width': signal_data['width'],
                    'values': signal_data['values']
                }
                js_signals.append(js_signal)
            
            html_file = Path(f"{vcd_file.stem}_js_waveform.html")
            
            # 创建高级HTML内容
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>高级RTL波形查看器 - {vcd_file.name}</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{ 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
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
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .btn:hover {{ background: #0056b3; }}
        .btn.secondary {{ background: #6c757d; }}
        .btn.secondary:hover {{ background: #545b62; }}
        .waveform-container {{ 
            padding: 20px; 
            background: #ffffff;
        }}
        .signal-row {{ 
            display: flex; 
            align-items: center; 
            margin-bottom: 15px; 
            padding: 10px; 
            background: #f8f9fa; 
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
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
        }}
        .signal-info {{ 
            width: 150px; 
            font-size: 12px; 
            color: #6c757d;
            text-align: right;
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
            color: #007bff;
        }}
        .timeline {{ 
            height: 40px; 
            background: #2c3e50; 
            color: white; 
            display: flex; 
            align-items: center; 
            padding: 0 20px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 高级RTL波形查看器</h1>
            <p>文件: {vcd_file.name} | 大小: {vcd_file.stat().st_size} bytes</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>时间范围:</label>
                <input type="number" id="timeStart" value="0" placeholder="开始时间">
                <span>-</span>
                <input type="number" id="timeEnd" value="1000" placeholder="结束时间">
            </div>
            <div class="control-group">
                <label>缩放:</label>
                <select id="zoomLevel">
                    <option value="1">1x</option>
                    <option value="2">2x</option>
                    <option value="4">4x</option>
                    <option value="8">8x</option>
                </select>
            </div>
            <button class="btn" onclick="updateWaveforms()">🔄 更新波形</button>
            <button class="btn secondary" onclick="exportData()">💾 导出数据</button>
            <button class="btn secondary" onclick="toggleSignals()">👁️ 切换信号</button>
        </div>
        
        <div class="timeline" id="timeline">
            时间轴将在这里显示
        </div>
        
        <div class="waveform-container" id="waveformContainer">
            <!-- 波形将在这里动态生成 -->
        </div>
        
        <div class="stats">
            <h3>📊 波形统计信息</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>信号总数</h4>
                    <div class="stat-value" id="signalCount">{len(js_signals)}</div>
                </div>
                <div class="stat-card">
                    <h4>时间范围</h4>
                    <div class="stat-value" id="timeRange">计算中...</div>
                </div>
                <div class="stat-card">
                    <h4>状态变化</h4>
                    <div class="stat-value" id="stateChanges">计算中...</div>
                </div>
                <div class="stat-card">
                    <h4>文件大小</h4>
                    <div class="stat-value">{vcd_file.stat().st_size} B</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 波形数据
        const signalData = {json.dumps(js_signals, indent=2)};
        
        // 全局变量
        let currentTimeStart = 0;
        let currentTimeEnd = 1000;
        let currentZoom = 1;
        let visibleSignals = new Set();
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initializeSignals();
            calculateStats();
            renderWaveforms();
        }});
        
        function initializeSignals() {{
            // 默认显示所有信号
            signalData.forEach((signal, index) => {{
                visibleSignals.add(index);
            }});
        }}
        
        function calculateStats() {{
            let totalChanges = 0;
            let maxTime = 0;
            
            signalData.forEach(signal => {{
                totalChanges += signal.values.length;
                signal.values.forEach(([time, value]) => {{
                    maxTime = Math.max(maxTime, time);
                }});
            }});
            
            document.getElementById('timeRange').textContent = `0 - ${{maxTime}} ps`;
            document.getElementById('stateChanges').textContent = totalChanges;
            
            // 更新时间范围输入框
            document.getElementById('timeEnd').value = maxTime;
            currentTimeEnd = maxTime;
        }}
        
        function renderWaveforms() {{
            const container = document.getElementById('waveformContainer');
            container.innerHTML = '';
            
            signalData.forEach((signal, index) => {{
                if (!visibleSignals.has(index)) return;
                
                const signalRow = document.createElement('div');
                signalRow.className = 'signal-row';
                
                const signalName = document.createElement('div');
                signalName.className = 'signal-name';
                signalName.textContent = signal.name;
                
                const canvas = document.createElement('canvas');
                canvas.className = 'signal-canvas';
                canvas.width = 800;
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
                drawSignalWaveform(canvas, signal);
            }});
            
            updateTimeline();
        }}
        
        function drawSignalWaveform(canvas, signal) {{
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            
            // 清空画布
            ctx.clearRect(0, 0, width, height);
            
            // 设置样式
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.font = '12px monospace';
            
            // 绘制背景网格
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                const x = (i / 10) * width;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }}
            
            // 绘制波形
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            let lastValue = '0';
            let lastX = 0;
            
            signal.values.forEach(([time, value], index) => {{
                const x = ((time - currentTimeStart) / (currentTimeEnd - currentTimeStart)) * width;
                
                if (x >= 0 && x <= width) {{
                    if (index === 0) {{
                        ctx.moveTo(x, height - 10);
                    }} else {{
                        // 绘制从上一个值到当前时间的水平线
                        ctx.lineTo(x, lastValue === '1' ? 10 : height - 10);
                        // 绘制垂直跳变线
                        ctx.lineTo(x, value === '1' ? 10 : height - 10);
                    }}
                    
                    lastX = x;
                    lastValue = value;
                }}
            }});
            
            // 延伸到画布末尾
            ctx.lineTo(width, lastValue === '1' ? 10 : height - 10);
            ctx.stroke();
            
            // 绘制值标签
            ctx.fillStyle = '#2c3e50';
            signal.values.forEach(([time, value]) => {{
                const x = ((time - currentTimeStart) / (currentTimeEnd - currentTimeStart)) * width;
                if (x >= 0 && x <= width) {{
                    ctx.fillText(value, x + 5, value === '1' ? 25 : height - 15);
                }}
            }});
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
            currentTimeEnd = parseInt(document.getElementById('timeEnd').value) || 1000;
            currentZoom = parseInt(document.getElementById('zoomLevel').value) || 1;
            
            renderWaveforms();
        }}
        
        function exportData() {{
            const dataStr = JSON.stringify(signalData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = '{vcd_file.stem}_waveform_data.json';
            link.click();
            
            URL.revokeObjectURL(url);
        }}
        
        function toggleSignals() {{
            const allVisible = visibleSignals.size === signalData.length;
            
            if (allVisible) {{
                visibleSignals.clear();
            }} else {{
                visibleSignals.clear();
                signalData.forEach((signal, index) => {{
                    visibleSignals.add(index);
                }});
            }}
            
            renderWaveforms();
        }}
    </script>
</body>
</html>"""
            
            # 写入HTML文件
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ 高级JavaScript波形查看器已创建: {html_file}")
            print(f"🌐 在浏览器中打开: open {html_file}")
            
            return html_file
            
        except Exception as e:
            print(f"❌ 创建高级波形查看器失败: {e}")
            return None
    
    def run(self):
        """运行波形查看器"""
        print("🚀 启动高级JavaScript波形查看器")
        print("=" * 50)
        
        if not self.vcd_files:
            print("❌ 没有找到VCD文件")
            return
        
        # 为每个VCD文件创建高级查看器
        for vcd_file in self.vcd_files:
            self.create_advanced_js_viewer(vcd_file)
        
        print(f"\n🎉 已为 {len(self.vcd_files)} 个波形文件创建高级查看器!")

if __name__ == "__main__":
    viewer = AdvancedJSWaveformViewer()
    viewer.run()