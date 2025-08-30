#!/usr/bin/env python3
"""
JavaScript波形查看器 - 使用Canvas和JavaScript绘制RTL波形
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

class JSWaveformViewer:
    """JavaScript波形查看器"""
    
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
    
    def parse_vcd_file(self, vcd_file):
        """解析VCD文件，提取波形数据"""
        print(f"🔍 解析VCD文件: {vcd_file.name}")
        
        signals = {}
        signal_values = {}
        time_values = []
        current_time = 0
        
        try:
            with open(vcd_file, 'r') as f:
                lines = f.readlines()
            
            # 第一遍：解析信号定义
            in_definitions = False
            for line in lines:
                line = line.strip()
                
                if line.startswith('$var'):
                    # $var wire 1 ! clk $end
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_type = parts[1]  # wire, reg, etc.
                        signal_width = int(parts[2])
                        signal_id = parts[3]
                        signal_name = parts[4]
                        
                        signals[signal_id] = {
                            'name': signal_name,
                            'type': signal_type,
                            'width': signal_width,
                            'values': []
                        }
                        signal_values[signal_id] = '0'  # 初始值
                
                elif line.startswith('$enddefinitions'):
                    in_definitions = False
                    break
            
            # 第二遍：解析信号变化
            for line in lines:
                line = line.strip()
                
                if line.startswith('#'):
                    # 时间戳
                    try:
                        current_time = int(line[1:])
                        time_values.append(current_time)
                        
                        # 记录当前时间点所有信号的值
                        for sig_id, sig_info in signals.items():
                            current_value = signal_values.get(sig_id, '0')
                            sig_info['values'].append({
                                'time': current_time,
                                'value': current_value
                            })
                    except ValueError:
                        continue
                
                elif line and not line.startswith('$'):
                    # 信号变化
                    if line[0] in '01xzXZ':
                        # 单bit信号: 0!, 1!, x!, z!
                        if len(line) > 1:
                            value = line[0]
                            signal_id = line[1:]
                            if signal_id in signal_values:
                                signal_values[signal_id] = value
                    
                    elif line[0] == 'b':
                        # 多bit信号: b1010 "
                        parts = line.split()
                        if len(parts) >= 2:
                            value = parts[0][1:]  # 去掉'b'
                            signal_id = parts[1]
                            if signal_id in signal_values:
                                signal_values[signal_id] = value
            
            print(f"✅ 解析完成: {len(signals)} 个信号, {len(time_values)} 个时间点")
            
            return {
                'signals': signals,
                'time_range': [min(time_values), max(time_values)] if time_values else [0, 0],
                'time_unit': 'ps'
            }
            
        except Exception as e:
            print(f"❌ 解析VCD文件失败: {e}")
            return None
    
    def create_js_waveform_viewer(self, vcd_file):
        """创建JavaScript波形查看器"""
        print(f"\n🌊 创建JavaScript波形查看器: {vcd_file.name}")
        
        # 解析VCD数据
        vcd_data = self.parse_vcd_file(vcd_file)
        if not vcd_data:
            return None
        
        html_file = vcd_file.parent / f"{vcd_file.stem}_js_waveform.html"
        
        # 准备JavaScript数据
        js_signals = []
        for sig_id, sig_info in vcd_data['signals'].items():
            js_signals.append({
                'id': sig_id,
                'name': sig_info['name'],
                'type': sig_info['type'],
                'width': sig_info['width'],
                'values': sig_info['values'][:100]  # 限制数据量
            })
        
        # 创建HTML内容
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>JavaScript波形查看器 - {vcd_file.name}</title>
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
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
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
        
        .btn:hover {{
            background: #0056b3;
        }}
        
        .waveform-container {{
            padding: 20px;
            background: white;
        }}
        
        .signal-list {{
            display: flex;
            flex-direction: column;
            gap: 2px;
            margin-bottom: 20px;
        }}
        
        .signal-row {{
            display: flex;
            align-items: center;
            height: 40px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .signal-name {{
            width: 200px;
            padding: 0 15px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            background: #f8f9fa;
            border-right: 1px solid #dee2e6;
            display: flex;
            align-items: center;
        }}
        
        .signal-wave {{
            flex: 1;
            height: 100%;
            position: relative;
        }}
        
        canvas {{
            border: 1px solid #dee2e6;
            background: white;
        }}
        
        .time-ruler {{
            height: 30px;
            background: #e9ecef;
            border: 1px solid #dee2e6;
            margin-bottom: 10px;
            position: relative;
        }}
        
        .info-panel {{
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .stat-box h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        
        .cursor-info {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 JavaScript波形查看器</h1>
            <p>文件: {vcd_file.name} | 大小: {vcd_file.stat().st_size:,} bytes</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>时间范围:</label>
                <input type="number" id="timeStart" value="{vcd_data['time_range'][0]}" placeholder="开始时间">
                <span>-</span>
                <input type="number" id="timeEnd" value="{vcd_data['time_range'][1]}" placeholder="结束时间">
                <span>{vcd_data['time_unit']}</span>
            </div>
            
            <div class="control-group">
                <label>缩放:</label>
                <input type="range" id="zoomSlider" min="1" max="10" value="1" step="0.1">
                <span id="zoomValue">1x</span>
            </div>
            
            <button class="btn" onclick="resetView()">重置视图</button>
            <button class="btn" onclick="fitToWindow()">适应窗口</button>
            <button class="btn" onclick="exportImage()">导出图片</button>
        </div>
        
        <div class="waveform-container">
            <div class="time-ruler" id="timeRuler"></div>
            <div class="signal-list" id="signalList"></div>
        </div>
        
        <div class="info-panel">
            <div class="stats">
                <div class="stat-box">
                    <h4>📊 文件信息</h4>
                    <p>信号数量: {len(js_signals)}</p>
                    <p>时间范围: {vcd_data['time_range'][0]:,} - {vcd_data['time_range'][1]:,} {vcd_data['time_unit']}</p>
                    <p>持续时间: {vcd_data['time_range'][1] - vcd_data['time_range'][0]:,} {vcd_data['time_unit']}</p>
                </div>
                <div class="stat-box">
                    <h4>🎛️ 控制说明</h4>
                    <p>• 鼠标滚轮: 缩放时间轴</p>
                    <p>• 拖拽: 平移视图</p>
                    <p>• 点击信号: 查看详细信息</p>
                </div>
                <div class="stat-box">
                    <h4>⚡ 性能信息</h4>
                    <p>渲染引擎: HTML5 Canvas</p>
                    <p>数据点: <span id="dataPoints">计算中...</span></p>
                    <p>刷新率: <span id="fps">60 FPS</span></p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="cursor-info" id="cursorInfo"></div>
    
    <script>
        // 波形数据
        const waveformData = {json.dumps(js_signals, indent=2)};
        const timeRange = {json.dumps(vcd_data['time_range'])};
        
        // 全局变量
        let currentZoom = 1;
        let currentOffset = 0;
        let canvasWidth = 1000;
        let canvasHeight = 40;
        let isDragging = false;
        let lastMouseX = 0;
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initializeWaveforms();
            setupEventListeners();
            renderWaveforms();
        }});
        
        function initializeWaveforms() {{
            const signalList = document.getElementById('signalList');
            
            waveformData.forEach((signal, index) => {{
                const signalRow = document.createElement('div');
                signalRow.className = 'signal-row';
                signalRow.innerHTML = `
                    <div class="signal-name" title="${{signal.name}} (${{signal.type}}, ${{signal.width}}bit)">
                        ${{signal.name}}
                    </div>
                    <div class="signal-wave">
                        <canvas id="canvas_${{index}}" width="${{canvasWidth}}" height="${{canvasHeight}}"></canvas>
                    </div>
                `;
                signalList.appendChild(signalRow);
            }});
            
            // 更新数据点统计
            const totalPoints = waveformData.reduce((sum, signal) => sum + signal.values.length, 0);
            document.getElementById('dataPoints').textContent = totalPoints.toLocaleString();
        }}
        
        function setupEventListeners() {{
            // 缩放控制
            const zoomSlider = document.getElementById('zoomSlider');
            zoomSlider.addEventListener('input', function() {{
                currentZoom = parseFloat(this.value);
                document.getElementById('zoomValue').textContent = currentZoom.toFixed(1) + 'x';
                renderWaveforms();
            }});
            
            // 时间范围控制
            document.getElementById('timeStart').addEventListener('change', renderWaveforms);
            document.getElementById('timeEnd').addEventListener('change', renderWaveforms);
            
            // 鼠标事件
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mousedown', handleMouseDown);
            document.addEventListener('mouseup', handleMouseUp);
            document.addEventListener('wheel', handleWheel);
        }}
        
        function renderWaveforms() {{
            const timeStart = parseInt(document.getElementById('timeStart').value) || timeRange[0];
            const timeEnd = parseInt(document.getElementById('timeEnd').value) || timeRange[1];
            const timeDuration = timeEnd - timeStart;
            
            waveformData.forEach((signal, index) => {{
                const canvas = document.getElementById(`canvas_${{index}}`);
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvasWidth, canvasHeight);
                
                // 绘制网格
                drawGrid(ctx, timeStart, timeEnd);
                
                // 绘制波形
                if (signal.width === 1) {{
                    drawDigitalWave(ctx, signal, timeStart, timeEnd);
                }} else {{
                    drawBusWave(ctx, signal, timeStart, timeEnd);
                }}
            }});
            
            // 绘制时间标尺
            drawTimeRuler(timeStart, timeEnd);
        }}
        
        function drawGrid(ctx, timeStart, timeEnd) {{
            ctx.strokeStyle = '#e9ecef';
            ctx.lineWidth = 1;
            
            // 垂直网格线
            const gridLines = 10;
            for (let i = 0; i <= gridLines; i++) {{
                const x = (i / gridLines) * canvasWidth;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvasHeight);
                ctx.stroke();
            }}
            
            // 水平中线
            ctx.beginPath();
            ctx.moveTo(0, canvasHeight / 2);
            ctx.lineTo(canvasWidth, canvasHeight / 2);
            ctx.stroke();
        }}
        
        function drawDigitalWave(ctx, signal, timeStart, timeEnd) {{
            if (signal.values.length === 0) return;
            
            ctx.strokeStyle = '#007bff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            let lastValue = '0';
            let lastX = 0;
            
            signal.values.forEach((point, index) => {{
                if (point.time < timeStart || point.time > timeEnd) return;
                
                const x = ((point.time - timeStart) / (timeEnd - timeStart)) * canvasWidth;
                const y = point.value === '1' ? 10 : canvasHeight - 10;
                
                if (index === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    // 绘制垂直跳变
                    if (point.value !== lastValue) {{
                        ctx.lineTo(x, lastValue === '1' ? 10 : canvasHeight - 10);
                        ctx.lineTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}
                
                lastValue = point.value;
                lastX = x;
            }});
            
            ctx.stroke();
            
            // 绘制信号值标签
            ctx.fillStyle = '#495057';
            ctx.font = '12px Courier New';
            signal.values.forEach((point, index) => {{
                if (point.time < timeStart || point.time > timeEnd) return;
                const x = ((point.time - timeStart) / (timeEnd - timeStart)) * canvasWidth;
                if (index % 3 === 0) {{ // 每3个点显示一个标签
                    ctx.fillText(point.value, x + 2, 15);
                }}
            }});
        }}
        
        function drawBusWave(ctx, signal, timeStart, timeEnd) {{
            if (signal.values.length === 0) return;
            
            ctx.strokeStyle = '#28a745';
            ctx.lineWidth = 2;
            
            signal.values.forEach((point, index) => {{
                if (point.time < timeStart || point.time > timeEnd) return;
                
                const x = ((point.time - timeStart) / (timeEnd - timeStart)) * canvasWidth;
                const nextPoint = signal.values[index + 1];
                const width = nextPoint ? 
                    ((nextPoint.time - point.time) / (timeEnd - timeStart)) * canvasWidth : 
                    50;
                
                // 绘制总线波形 (梯形)
                ctx.beginPath();
                ctx.moveTo(x, 15);
                ctx.lineTo(x + 5, 10);
                ctx.lineTo(x + width - 5, 10);
                ctx.lineTo(x + width, 15);
                ctx.lineTo(x + width - 5, canvasHeight - 10);
                ctx.lineTo(x + 5, canvasHeight - 10);
                ctx.closePath();
                ctx.stroke();
                
                // 绘制数值
                ctx.fillStyle = '#495057';
                ctx.font = '10px Courier New';
                const value = point.value.length > 8 ? point.value.substring(0, 8) + '...' : point.value;
                ctx.fillText(value, x + 8, canvasHeight / 2 + 3);
            }});
        }}
        
        function drawTimeRuler(timeStart, timeEnd) {{
            const ruler = document.getElementById('timeRuler');
            ruler.innerHTML = '';
            
            const timeDuration = timeEnd - timeStart;
            const steps = 10;
            
            for (let i = 0; i <= steps; i++) {{
                const time = timeStart + (timeDuration * i / steps);
                const x = (i / steps) * 100;
                
                const marker = document.createElement('div');
                marker.style.position = 'absolute';
                marker.style.left = x + '%';
                marker.style.top = '5px';
                marker.style.fontSize = '12px';
                marker.style.color = '#495057';
                marker.textContent = time.toLocaleString() + 'ps';
                ruler.appendChild(marker);
            }}
        }}
        
        function handleMouseMove(event) {{
            // 显示光标信息
            const cursorInfo = document.getElementById('cursorInfo');
            cursorInfo.style.display = 'block';
            cursorInfo.style.left = event.clientX + 10 + 'px';
            cursorInfo.style.top = event.clientY - 50 + 'px';
            cursorInfo.innerHTML = `
                X: ${{event.clientX}}px<br>
                Y: ${{event.clientY}}px<br>
                缩放: ${{currentZoom.toFixed(1)}}x
            `;
        }}
        
        function handleMouseDown(event) {{
            isDragging = true;
            lastMouseX = event.clientX;
        }}
        
        function handleMouseUp(event) {{
            isDragging = false;
        }}
        
        function handleWheel(event) {{
            event.preventDefault();
            const delta = event.deltaY > 0 ? 0.9 : 1.1;
            currentZoom = Math.max(0.1, Math.min(10, currentZoom * delta));
            
            document.getElementById('zoomSlider').value = currentZoom;
            document.getElementById('zoomValue').textContent = currentZoom.toFixed(1) + 'x';
            renderWaveforms();
        }}
        
        function resetView() {{
            currentZoom = 1;
            currentOffset = 0;
            document.getElementById('timeStart').value = timeRange[0];
            document.getElementById('timeEnd').value = timeRange[1];
            document.getElementById('zoomSlider').value = 1;
            document.getElementById('zoomValue').textContent = '1x';
            renderWaveforms();
        }}
        
        function fitToWindow() {{
            // 自动调整时间范围以适应窗口
            const container = document.querySelector('.waveform-container');
            canvasWidth = container.clientWidth - 220; // 减去信号名称宽度
            
            waveformData.forEach((signal, index) => {{
                const canvas = document.getElementById(`canvas_${{index}}`);
                if (canvas) {{
                    canvas.width = canvasWidth;
                }}
            }});
            
            renderWaveforms();
        }}
        
        function exportImage() {{
            // 创建一个大画布来合并所有波形
            const exportCanvas = document.createElement('canvas');
            const exportCtx = exportCanvas.getContext('2d');
            
            exportCanvas.width = canvasWidth;
            exportCanvas.height = waveformData.length * (canvasHeight + 5) + 50;
            
            // 绘制背景
            exportCtx.fillStyle = 'white';
            exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
            
            // 绘制标题
            exportCtx.fillStyle = 'black';
            exportCtx.font = '16px Arial';
            exportCtx.fillText('RTL波形图 - {vcd_file.name}', 10, 25);
            
            // 复制所有波形
            waveformData.forEach((signal, index) => {{
                const canvas = document.getElementById(`canvas_${{index}}`);
                if (canvas) {{
                    exportCtx.drawImage(canvas, 0, 40 + index * (canvasHeight + 5));
                    
                    // 添加信号名称
                    exportCtx.fillStyle = 'black';
                    exportCtx.font = '12px Courier New';
                    exportCtx.fillText(signal.name, 5, 55 + index * (canvasHeight + 5));
                }}
            }});
            
            // 下载图片
            const link = document.createElement('a');
            link.download = '{vcd_file.stem}_waveform.png';
            link.href = exportCanvas.toDataURL();
            link.click();
        }}
        
        // 性能监控
        let frameCount = 0;
        let lastTime = performance.now();
        
        function updateFPS() {{
            frameCount++;
            const currentTime = performance.now();
            if (currentTime - lastTime >= 1000) {{
                const fps = Math.round(frameCount * 1000 / (currentTime - lastTime));
                document.getElementById('fps').textContent = fps + ' FPS';
                frameCount = 0;
                lastTime = currentTime;
            }}
            requestAnimationFrame(updateFPS);
        }}
        
        updateFPS();
    </script>
</body>
</html>"""
        
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ JavaScript波形查看器已创建: {html_file}")
            
            # 在浏览器中打开
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(html_file)])
                print("🌐 已在浏览器中打开")
            
            return html_file
            
        except Exception as e:
            print(f"❌ 创建JavaScript波形查看器失败: {e}")
            return None

def test_js_waveform_viewer():
    """测试JavaScript波形查看器"""
    print("🌊 JavaScript波形查看器测试")
    print("=" * 50)
    
    viewer = JSWaveformViewer()
    
    if viewer.vcd_files:
        for vcd_file in viewer.vcd_files:
            print(f"\n📊 为 {vcd_file.name} 创建JavaScript波形查看器...")
            html_file = viewer.create_js_waveform_viewer(vcd_file)
            if html_file:
                print(f"✅ HTML文件: {html_file}")
                print(f"🌐 在浏览器中查看: file://{html_file.absolute()}")
    else:
        print("❌ 没有找到VCD文件")
        print("💡 请先运行RTL仿真生成VCD文件:")
        print("   python3 rtl_hardware_backend.py")

def main():
    """主程序"""
    print("🚀 JavaScript波形查看器套件")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--basic":
            print("📊 启动基础JavaScript波形查看器")
            test_js_waveform_viewer()
        elif sys.argv[1] == "--advanced":
            print("🔬 启动高级JavaScript波形查看器")
            from advanced_js_waveform import AdvancedJSWaveformViewer
            viewer = AdvancedJSWaveformViewer()
            for vcd_file in viewer.vcd_files:
                viewer.create_advanced_js_viewer(vcd_file)
        elif sys.argv[1] == "--compare":
            print("⚖️ 创建所有类型的波形查看器进行比较")
            test_js_waveform_viewer()
            from advanced_js_waveform import AdvancedJSWaveformViewer
            advanced_viewer = AdvancedJSWaveformViewer()
            for vcd_file in advanced_viewer.vcd_files:
                advanced_viewer.create_advanced_js_viewer(vcd_file)
        elif sys.argv[1] == "--help":
            print("JavaScript波形查看器使用说明:")
            print("=" * 40)
            print("  python3 test_html_viewer.py                # 基础版本")
            print("  python3 test_html_viewer.py --basic        # 基础版本")
            print("  python3 test_html_viewer.py --advanced     # 高级版本")
            print("  python3 test_html_viewer.py --compare      # 创建所有版本")
            print("  python3 test_html_viewer.py --help         # 显示帮助")
            print("")
            print("功能对比:")
            print("📊 基础版本:")
            print("   • Canvas绘制波形")
            print("   • 基本缩放和控制")
            print("   • 信号值显示")
            print("   • 时间标尺")
            print("")
            print("🔬 高级版本:")
            print("   • 交互式信号选择")
            print("   • 实时波形测量")
            print("   • 频率和占空比分析")
            print("   • 搜索和过滤功能")
            print("   • 数据导出")
            print("   • 专业级界面")
        else:
            print("❌ 未知参数，使用 --help 查看帮助")
    else:
        # 默认运行基础版本
        test_js_waveform_viewer()

if __name__ == "__main__":
    main()