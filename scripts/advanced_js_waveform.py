#!/usr/bin/env python3
"""
高级JavaScript波形查看器 - 增强版本
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
        self.rtl_path = Path("verification/simple_rtl")
        self.vcd_files = []
        self._find_vcd_files()
    
    def _find_vcd_files(self):
        """查找VCD波形文件"""
        self.vcd_files = list(self.rtl_path.glob("*.vcd"))
        print(f"📊 找到 {len(self.vcd_files)} 个波形文件:")
        for vcd_file in self.vcd_files:
            file_size = vcd_file.stat().st_size
            print(f"  📈 {vcd_file.name} ({file_size} bytes)")
    
    def parse_vcd_with_analysis(self, vcd_file):
        """解析VCD文件并进行波形分析"""
        print(f"🔍 深度解析VCD文件: {vcd_file.name}")
        
        signals = {}
        signal_values = {}
        time_values = []
        current_time = 0
        signal_stats = {}
        
        try:
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            # 解析信号定义
            for line in lines:
                line = line.strip()
                
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_type = parts[1]
                        signal_width = int(parts[2])
                        signal_id = parts[3]
                        signal_name = parts[4]
                        
                        signals[signal_id] = {
                            'name': signal_name,
                            'type': signal_type,
                            'width': signal_width,
                            'values': [],
                            'transitions': 0,
                            'high_time': 0,
                            'low_time': 0
                        }
                        signal_values[signal_id] = '0'
                        signal_stats[signal_id] = {
                            'last_transition': 0,
                            'last_value': '0'
                        }
                
                elif line.startswith('$enddefinitions'):
                    break
            
            # 解析信号变化并统计
            last_time = 0
            for line in lines:
                line = line.strip()
                
                if line.startswith('#'):
                    try:
                        current_time = int(line[1:])
                        time_values.append(current_time)
                        
                        # 更新统计信息
                        time_delta = current_time - last_time
                        for sig_id, sig_info in signals.items():
                            current_value = signal_values.get(sig_id, '0')
                            sig_info['values'].append({
                                'time': current_time,
                                'value': current_value
                            })
                            
                            # 统计高低电平时间
                            if sig_info['width'] == 1:  # 只对单bit信号统计
                                if current_value == '1':
                                    sig_info['high_time'] += time_delta
                                else:
                                    sig_info['low_time'] += time_delta
                        
                        last_time = current_time
                    except ValueError:
                        continue
                
                elif line and not line.startswith('$'):
                    # 信号变化
                    if line[0] in '01xzXZ':
                        if len(line) > 1:
                            value = line[0]
                            signal_id = line[1:]
                            if signal_id in signal_values:
                                old_value = signal_values[signal_id]
                                signal_values[signal_id] = value
                                
                                # 统计跳变次数
                                if old_value != value:
                                    signals[signal_id]['transitions'] += 1
                    
                    elif line[0] == 'b':
                        parts = line.split()
                        if len(parts) >= 2:
                            value = parts[0][1:]
                            signal_id = parts[1]
                            if signal_id in signal_values:
                                old_value = signal_values[signal_id]
                                signal_values[signal_id] = value
                                
                                if old_value != value:
                                    signals[signal_id]['transitions'] += 1
            
            # 计算频率和占空比
            total_time = max(time_values) - min(time_values) if time_values else 1
            for sig_id, sig_info in signals.items():
                if sig_info['width'] == 1 and sig_info['transitions'] > 0:
                    sig_info['frequency'] = sig_info['transitions'] / (2 * total_time) * 1e12  # Hz
                    sig_info['duty_cycle'] = sig_info['high_time'] / (sig_info['high_time'] + sig_info['low_time']) * 100
                else:
                    sig_info['frequency'] = 0
                    sig_info['duty_cycle'] = 0
            
            print(f"✅ 深度解析完成: {len(signals)} 个信号, {len(time_values)} 个时间点")
            
            return {
                'signals': signals,
                'time_range': [min(time_values), max(time_values)] if time_values else [0, 0],
                'time_unit': 'ps',
                'total_transitions': sum(sig['transitions'] for sig in signals.values()),
                'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ 解析VCD文件失败: {e}")
            return None
    
    def create_advanced_js_viewer(self, vcd_file):
        """创建高级JavaScript波形查看器"""
        print(f"\n🌊 创建高级JavaScript波形查看器: {vcd_file.name}")
        
        vcd_data = self.parse_vcd_with_analysis(vcd_file)
        if not vcd_data:
            return None
        
        html_file = vcd_file.parent / f"{vcd_file.stem}_advanced_waveform.html"
        
        # 准备JavaScript数据
        js_signals = []
        for sig_id, sig_info in vcd_data['signals'].items():
            js_signals.append({
                'id': sig_id,
                'name': sig_info['name'],
                'type': sig_info['type'],
                'width': sig_info['width'],
                'values': sig_info['values'][:200],  # 限制数据量
                'transitions': sig_info['transitions'],
                'frequency': sig_info.get('frequency', 0),
                'duty_cycle': sig_info.get('duty_cycle', 0),
                'high_time': sig_info.get('high_time', 0),
                'low_time': sig_info.get('low_time', 0)
            })
        
        # 创建增强版HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>高级波形查看器 - {vcd_file.name}</title>
    <meta charset="UTF-8">
    <style>
        * {{ box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .app-container {{
            display: flex;
            height: 100vh;
        }}
        
        .sidebar {{
            width: 300px;
            background: #2c3e50;
            color: white;
            overflow-y: auto;
            padding: 20px;
        }}
        
        .main-content {{
            flex: 1;
            background: white;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .toolbar {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .waveform-area {{
            flex: 1;
            overflow: auto;
            padding: 20px;
        }}
        
        .signal-item {{
            background: #34495e;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .signal-item:hover {{
            background: #4a6741;
            transform: translateX(5px);
        }}
        
        .signal-item.selected {{
            background: #27ae60;
        }}
        
        .signal-name {{
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
        }}
        
        .signal-info {{
            font-size: 12px;
            opacity: 0.8;
        }}
        
        .signal-stats {{
            margin-top: 10px;
            font-size: 11px;
        }}
        
        .waveform-canvas {{
            border: 1px solid #dee2e6;
            margin: 10px 0;
            background: white;
            cursor: crosshair;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .btn {{
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        
        .btn:hover {{ background: #0056b3; }}
        .btn.secondary {{ background: #6c757d; }}
        .btn.success {{ background: #28a745; }}
        .btn.danger {{ background: #dc3545; }}
        
        .input-field {{
            padding: 6px 10px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .measurement-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            display: none;
            z-index: 1000;
        }}
        
        .analysis-panel {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }}
        
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .search-box {{
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #495057;
            border-radius: 4px;
            background: #495057;
            color: white;
        }}
        
        .search-box::placeholder {{
            color: #adb5bd;
        }}
    </style>
</head>
<body>
    <div class="app-container">
        <div class="sidebar">
            <h2>🌊 信号列表</h2>
            
            <input type="text" class="search-box" id="signalSearch" placeholder="搜索信号...">
            
            <div class="analysis-panel">
                <h4>📊 统计信息</h4>
                <div>总信号数: {len(js_signals)}</div>
                <div>总跳变数: {vcd_data['total_transitions']}</div>
                <div>仿真时间: {vcd_data['time_range'][1] - vcd_data['time_range'][0]:,} ps</div>
            </div>
            
            <div id="signalList"></div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>🔬 高级RTL波形分析器</h1>
                <p>{vcd_file.name} | 分析时间: {vcd_data['analysis_time']}</p>
            </div>
            
            <div class="toolbar">
                <div class="controls">
                    <div class="control-group">
                        <label>时间:</label>
                        <input type="number" class="input-field" id="timeStart" value="{vcd_data['time_range'][0]}" style="width: 100px;">
                        <span>-</span>
                        <input type="number" class="input-field" id="timeEnd" value="{vcd_data['time_range'][1]}" style="width: 100px;">
                        <span>ps</span>
                    </div>
                    
                    <div class="control-group">
                        <label>缩放:</label>
                        <input type="range" id="zoomSlider" min="0.1" max="20" value="1" step="0.1" style="width: 150px;">
                        <span id="zoomValue">1.0x</span>
                    </div>
                    
                    <button class="btn" onclick="resetView()">重置</button>
                    <button class="btn secondary" onclick="fitToWindow()">适应窗口</button>
                    <button class="btn success" onclick="measureMode()">测量模式</button>
                    <button class="btn danger" onclick="exportData()">导出数据</button>
                </div>
            </div>
            
            <div class="waveform-area">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="selectedSignals">0</div>
                        <div class="stat-label">选中信号</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="visibleTime">0</div>
                        <div class="stat-label">可见时间 (ps)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="cursorTime">-</div>
                        <div class="stat-label">光标时间 (ps)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="measurementDelta">-</div>
                        <div class="stat-label">测量间隔 (ps)</div>
                    </div>
                </div>
                
                <canvas id="mainCanvas" class="waveform-canvas" width="1200" height="600"></canvas>
            </div>
        </div>
    </div>
    
    <div class="measurement-panel" id="measurementPanel">
        <div id="measurementInfo"></div>
    </div>
    
    <script>
        // 全局数据和状态
        const waveformData = {json.dumps(js_signals, indent=2)};
        const timeRange = {json.dumps(vcd_data['time_range'])};
        
        let selectedSignals = new Set();
        let currentZoom = 1;
        let currentOffset = 0;
        let measurementMode = false;
        let measurementStart = null;
        let measurementEnd = null;
        let cursorPosition = null;
        
        // 画布设置
        const canvas = document.getElementById('mainCanvas');
        const ctx = canvas.getContext('2d');
        const signalHeight = 40;
        const signalSpacing = 50;
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initializeSignalList();
            setupEventListeners();
            renderWaveforms();
            updateStats();
        }});
        
        function initializeSignalList() {{
            const signalList = document.getElementById('signalList');
            
            waveformData.forEach((signal, index) => {{
                const signalItem = document.createElement('div');
                signalItem.className = 'signal-item';
                signalItem.dataset.signalId = signal.id;
                
                const freqText = signal.frequency > 0 ? `${{(signal.frequency/1e6).toFixed(2)}} MHz` : 'N/A';
                const dutyText = signal.duty_cycle > 0 ? `${{signal.duty_cycle.toFixed(1)}}%` : 'N/A';
                
                signalItem.innerHTML = `
                    <div class="signal-name">${{signal.name}}</div>
                    <div class="signal-info">${{signal.type}}, ${{signal.width}} bit</div>
                    <div class="signal-stats">
                        跳变: ${{signal.transitions}} | 频率: ${{freqText}}<br>
                        占空比: ${{dutyText}} | 数据点: ${{signal.values.length}}
                    </div>
                `;
                
                signalItem.addEventListener('click', () => toggleSignal(signal.id, signalItem));
                signalList.appendChild(signalItem);
            }});
        }}
        
        function setupEventListeners() {{
            // 搜索功能
            document.getElementById('signalSearch').addEventListener('input', function() {{
                const searchTerm = this.value.toLowerCase();
                const signalItems = document.querySelectorAll('.signal-item');
                
                signalItems.forEach(item => {{
                    const signalName = item.querySelector('.signal-name').textContent.toLowerCase();
                    item.style.display = signalName.includes(searchTerm) ? 'block' : 'none';
                }});
            }});
            
            // 缩放控制
            document.getElementById('zoomSlider').addEventListener('input', function() {{
                currentZoom = parseFloat(this.value);
                document.getElementById('zoomValue').textContent = currentZoom.toFixed(1) + 'x';
                renderWaveforms();
                updateStats();
            }});
            
            // 时间范围控制
            document.getElementById('timeStart').addEventListener('change', () => {{
                renderWaveforms();
                updateStats();
            }});
            document.getElementById('timeEnd').addEventListener('change', () => {{
                renderWaveforms();
                updateStats();
            }});
            
            // 画布事件
            canvas.addEventListener('mousemove', handleCanvasMouseMove);
            canvas.addEventListener('click', handleCanvasClick);
            canvas.addEventListener('wheel', handleCanvasWheel);
        }}
        
        function toggleSignal(signalId, element) {{
            if (selectedSignals.has(signalId)) {{
                selectedSignals.delete(signalId);
                element.classList.remove('selected');
            }} else {{
                selectedSignals.add(signalId);
                element.classList.add('selected');
            }}
            
            renderWaveforms();
            updateStats();
        }}
        
        function renderWaveforms() {{
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            
            // 清空画布
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 绘制背景网格
            drawGrid(timeStart, timeEnd);
            
            // 绘制选中的信号
            let yOffset = 30;
            selectedSignals.forEach(signalId => {{
                const signal = waveformData.find(s => s.id === signalId);
                if (signal) {{
                    drawSignalWaveform(signal, yOffset, timeStart, timeEnd);
                    yOffset += signalSpacing;
                }}
            }});
            
            // 绘制测量线
            if (measurementStart !== null) {{
                drawMeasurementLine(measurementStart, 'green');
            }}
            if (measurementEnd !== null) {{
                drawMeasurementLine(measurementEnd, 'red');
            }}
            
            // 绘制光标
            if (cursorPosition !== null) {{
                drawCursor(cursorPosition);
            }}
        }}
        
        function drawGrid(timeStart, timeEnd) {{
            const timeDuration = timeEnd - timeStart;
            const gridLines = 20;
            
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            
            // 垂直网格线
            for (let i = 0; i <= gridLines; i++) {{
                const x = (i / gridLines) * canvas.width;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
                
                // 时间标签
                const time = timeStart + (timeDuration * i / gridLines);
                ctx.fillStyle = '#6c757d';
                ctx.font = '10px Arial';
                ctx.fillText(time.toLocaleString() + 'ps', x + 2, 15);
            }}
            
            // 水平网格线
            const signalCount = selectedSignals.size;
            for (let i = 0; i <= signalCount; i++) {{
                const y = 30 + i * signalSpacing;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }}
        }}
        
        function drawSignalWaveform(signal, yOffset, timeStart, timeEnd) {{
            if (signal.values.length === 0) return;
            
            // 绘制信号名称
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(signal.name, 5, yOffset - 10);
            
            if (signal.width === 1) {{
                drawDigitalSignal(signal, yOffset, timeStart, timeEnd);
            }} else {{
                drawBusSignal(signal, yOffset, timeStart, timeEnd);
            }}
        }}
        
        function drawDigitalSignal(signal, yOffset, timeStart, timeEnd) {{
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            let lastValue = '0';
            let lastX = 0;
            
            signal.values.forEach((point, index) => {{
                if (point.time < timeStart || point.time > timeEnd) return;
                
                const x = ((point.time - timeStart) / (timeEnd - timeStart)) * canvas.width;
                const y = point.value === '1' ? yOffset - 15 : yOffset + 15;
                
                if (index === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    if (point.value !== lastValue) {{
                        ctx.lineTo(x, lastValue === '1' ? yOffset - 15 : yOffset + 15);
                        ctx.lineTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}
                
                lastValue = point.value;
                lastX = x;
            }});
            
            ctx.stroke();
            
            // 绘制高低电平标签
            ctx.fillStyle = '#495057';
            ctx.font = '10px Arial';
            ctx.fillText('1', 10, yOffset - 20);
            ctx.fillText('0', 10, yOffset + 20);
        }}
        
        function drawBusSignal(signal, yOffset, timeStart, timeEnd) {{
            ctx.strokeStyle = '#28a745';
            ctx.lineWidth = 2;
            
            signal.values.forEach((point, index) => {{
                if (point.time < timeStart || point.time > timeEnd) return;
                
                const x = ((point.time - timeStart) / (timeEnd - timeStart)) * canvas.width;
                const nextPoint = signal.values[index + 1];
                const width = nextPoint ? 
                    ((nextPoint.time - point.time) / (timeEnd - timeStart)) * canvas.width : 
                    100;
                
                // 绘制总线波形
                ctx.beginPath();
                ctx.moveTo(x, yOffset - 10);
                ctx.lineTo(x + 8, yOffset - 15);
                ctx.lineTo(x + width - 8, yOffset - 15);
                ctx.lineTo(x + width, yOffset - 10);
                ctx.lineTo(x + width - 8, yOffset + 15);
                ctx.lineTo(x + 8, yOffset + 15);
                ctx.closePath();
                ctx.stroke();
                
                // 绘制数值
                ctx.fillStyle = '#495057';
                ctx.font = '9px Courier New';
                const value = point.value.length > 6 ? point.value.substring(0, 6) + '...' : point.value;
                ctx.fillText(value, x + 10, yOffset + 3);
            }});
        }}
        
        function drawMeasurementLine(time, color) {{
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            const x = ((time - timeStart) / (timeEnd - timeStart)) * canvas.width;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
            ctx.setLineDash([]);
        }}
        
        function drawCursor(time) {{
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            const x = ((time - timeStart) / (timeEnd - timeStart)) * canvas.width;
            
            ctx.strokeStyle = '#ff6b6b';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }}
        
        function handleCanvasMouseMove(event) {{
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            const time = timeStart + (x / canvas.width) * (timeEnd - timeStart);
            
            cursorPosition = time;
            document.getElementById('cursorTime').textContent = Math.round(time).toLocaleString();
            
            renderWaveforms();
        }}
        
        function handleCanvasClick(event) {{
            if (!measurementMode) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            const time = timeStart + (x / canvas.width) * (timeEnd - timeStart);
            
            if (measurementStart === null) {{
                measurementStart = time;
            }} else if (measurementEnd === null) {{
                measurementEnd = time;
                const delta = Math.abs(measurementEnd - measurementStart);
                document.getElementById('measurementDelta').textContent = Math.round(delta).toLocaleString();
                
                // 显示详细测量信息
                showMeasurementInfo(measurementStart, measurementEnd);
            }} else {{
                // 重新开始测量
                measurementStart = time;
                measurementEnd = null;
                document.getElementById('measurementDelta').textContent = '-';
            }}
            
            renderWaveforms();
        }}
        
        function handleCanvasWheel(event) {{
            event.preventDefault();
            const delta = event.deltaY > 0 ? 0.9 : 1.1;
            currentZoom = Math.max(0.1, Math.min(20, currentZoom * delta));
            
            document.getElementById('zoomSlider').value = currentZoom;
            document.getElementById('zoomValue').textContent = currentZoom.toFixed(1) + 'x';
            renderWaveforms();
            updateStats();
        }}
        
        function showMeasurementInfo(start, end) {{
            const panel = document.getElementById('measurementPanel');
            const info = document.getElementById('measurementInfo');
            
            const delta = Math.abs(end - start);
            const frequency = delta > 0 ? 1e12 / (2 * delta) : 0;
            
            info.innerHTML = `
                <h4>📏 测量结果</h4>
                <div>起始时间: ${{Math.round(start).toLocaleString()}} ps</div>
                <div>结束时间: ${{Math.round(end).toLocaleString()}} ps</div>
                <div>时间间隔: ${{Math.round(delta).toLocaleString()}} ps</div>
                <div>等效频率: ${{(frequency/1e6).toFixed(2)}} MHz</div>
                <div>等效周期: ${{(delta*2/1000).toFixed(2)}} ns</div>
            `;
            
            panel.style.display = 'block';
        }}
        
        function updateStats() {{
            document.getElementById('selectedSignals').textContent = selectedSignals.size;
            
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            document.getElementById('visibleTime').textContent = Math.round(timeEnd - timeStart).toLocaleString();
        }}
        
        function resetView() {{
            currentZoom = 1;
            currentOffset = 0;
            measurementStart = null;
            measurementEnd = null;
            measurementMode = false;
            
            document.getElementById('timeStart').value = timeRange[0];
            document.getElementById('timeEnd').value = timeRange[1];
            document.getElementById('zoomSlider').value = 1;
            document.getElementById('zoomValue').textContent = '1.0x';
            document.getElementById('measurementDelta').textContent = '-';
            document.getElementById('measurementPanel').style.display = 'none';
            
            renderWaveforms();
            updateStats();
        }}
        
        function fitToWindow() {{
            canvas.width = canvas.parentElement.clientWidth - 40;
            renderWaveforms();
        }}
        
        function measureMode() {{
            measurementMode = !measurementMode;
            const btn = event.target;
            
            if (measurementMode) {{
                btn.textContent = '退出测量';
                btn.className = 'btn danger';
                canvas.style.cursor = 'crosshair';
            }} else {{
                btn.textContent = '测量模式';
                btn.className = 'btn success';
                canvas.style.cursor = 'default';
                measurementStart = null;
                measurementEnd = null;
                document.getElementById('measurementPanel').style.display = 'none';
                renderWaveforms();
            }}
        }}
        
        function exportData() {{
            const exportData = {{
                file: '{vcd_file.name}',
                signals: Array.from(selectedSignals).map(id => {{
                    const signal = waveformData.find(s => s.id === id);
                    return {{
                        name: signal.name,
                        type: signal.type,
                        width: signal.width,
                        values: signal.values,
                        stats: {{
                            transitions: signal.transitions,
                            frequency: signal.frequency,
                            duty_cycle: signal.duty_cycle
                        }}
                    }};
                }}),
                timeRange: [
                    parseInt(document.getElementById('timeStart').value),
                    parseInt(document.getElementById('timeEnd').value)
                ],
                measurement: measurementStart !== null && measurementEnd !== null ? {{
                    start: measurementStart,
                    end: measurementEnd,
                    delta: Math.abs(measurementEnd - measurementStart)
                }} : null,
                exportTime: new Date().toISOString()
            }};
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{vcd_file.stem}_waveform_data.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // 自动选择前几个信号进行显示
        setTimeout(() => {{
            const firstFewSignals = waveformData.slice(0, Math.min(5, waveformData.length));
            firstFewSignals.forEach(signal => {{
                const element = document.querySelector(`[data-signal-id="${{signal.id}}"]`);
                if (element) {{
                    toggleSignal(signal.id, element);
                }}
            }});
        }}, 100);
    </script>
</body>
</html>"""
        
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ 高级JavaScript波形查看器已创建: {html_file}")
            
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(html_file)])
                print("🌐 已在浏览器中打开")
            
            return html_file
            
        except Exception as e:
            print(f"❌ 创建高级波形查看器失败: {e}")
            return None

def main():
    """主程序"""
    print("🚀 启动高级JavaScript波形查看器")
    print("=" * 50)
    
    viewer = AdvancedJSWaveformViewer()
    
    if viewer.vcd_files:
        for vcd_file in viewer.vcd_files:
            print(f"\n📊 为 {vcd_file.name} 创建高级波形查看器...")
            html_file = viewer.create_advanced_js_viewer(vcd_file)
            if html_file:
                print(f"✅ 高级HTML文件: {html_file}")
                print(f"🌐 功能特性:")
                print(f"   • 交互式信号选择和搜索")
                print(f"   • 实时波形测量和分析")
                print(f"   • 缩放和平移功能")
                print(f"   • 频率和占空比统计")
                print(f"   • 数据导出功能")
    else:
        print("❌ 没有找到VCD文件")

if __name__ == "__main__":
    main()