#!/usr/bin/env python3
"""
Web 波形查看器
使用 Flask 提供 Web 界面查看 VCD 波形文件
"""

import os
import re
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, send_file, request
import argparse
from wave_renderer import WaveformRenderer

app = Flask(__name__)

class VCDParser:
    """优化的 VCD 文件解析器 - 支持大文件和智能抽样"""
    
    def __init__(self, vcd_file):
        self.vcd_file = Path(vcd_file)
        self.signals = {}
        self.timescale = "1ps"
        self.date = ""
        self.version = ""
        self.max_time = 0
        self.file_size = 0
        self.is_large_file = False
        
    def parse_header(self):
        """解析 VCD 文件头部信息"""
        with open(self.vcd_file, 'r') as f:
            in_scope = False
            current_scope = []
            
            for line in f:
                line = line.strip()
                
                if line.startswith('$date'):
                    self.date = next(f).strip()
                elif line.startswith('$version'):
                    self.version = next(f).strip()
                elif line.startswith('$timescale'):
                    self.timescale = next(f).strip()
                elif line.startswith('$scope'):
                    parts = line.split()
                    if len(parts) >= 3:
                        current_scope.append(parts[2])
                    in_scope = True
                elif line.startswith('$upscope'):
                    if current_scope:
                        current_scope.pop()
                elif line.startswith('$var'):
                    # $var wire 1 ! uart_tx $end
                    parts = line.split()
                    if len(parts) >= 5:
                        var_type = parts[1]
                        width = int(parts[2])
                        symbol = parts[3]
                        name = parts[4]
                        
                        full_name = '.'.join(current_scope + [name])
                        self.signals[symbol] = {
                            'name': name,
                            'full_name': full_name,
                            'type': var_type,
                            'width': width,
                            'values': []
                        }
                elif line.startswith('$enddefinitions'):
                    break
    
    def parse_values_fast(self):
        """快速解析 - 只获取时间范围，不加载所有数据"""
        self.file_size = self.vcd_file.stat().st_size
        self.is_large_file = self.file_size > 50 * 1024 * 1024  # 50MB
        
        with open(self.vcd_file, 'r') as f:
            # 跳到数据部分
            for line in f:
                if line.strip().startswith('$enddefinitions'):
                    break
            
            # 只扫描时间戳，不加载数据
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    current_time = int(line[1:])
                    self.max_time = max(self.max_time, current_time)
    
    def parse_signal_range(self, symbols, start_time=0, end_time=None, max_points=2000):
        """解析指定信号在指定时间范围内的数据（智能抽样）"""
        if end_time is None:
            end_time = self.max_time
        
        # 计算抽样间隔
        time_range = end_time - start_time
        if time_range <= 0:
            return {}
        
        # 根据时间范围和目标点数计算抽样间隔
        sample_interval = max(1, time_range // max_points)
        
        result = {}
        for symbol in symbols:
            if symbol in self.signals:
                result[symbol] = {
                    'info': {
                        'name': self.signals[symbol]['name'],
                        'full_name': self.signals[symbol]['full_name'],
                        'width': self.signals[symbol]['width']
                    },
                    'values': []
                }
        
        with open(self.vcd_file, 'r') as f:
            # 跳到数据部分
            for line in f:
                if line.strip().startswith('$enddefinitions'):
                    break
            
            current_time = 0
            last_sample_time = -sample_interval
            signal_last_values = {s: None for s in symbols}
            
            for line in f:
                line = line.strip()
                
                if line.startswith('#'):
                    current_time = int(line[1:])
                    
                    # 超出范围则停止
                    if current_time > end_time:
                        break
                    
                elif line and not line.startswith('$'):
                    # 只处理在时间范围内的数据
                    if current_time < start_time:
                        continue
                    
                    symbol = None
                    value = None
                    
                    if line[0] in '01xzXZ':
                        value = line[0]
                        symbol = line[1:]
                    elif line[0] == 'b':
                        parts = line.split()
                        if len(parts) >= 2:
                            value = parts[0][1:]
                            symbol = parts[1]
                    
                    if symbol and symbol in result:
                        # 智能抽样：保留值变化点 + 定期采样
                        should_sample = False
                        
                        # 1. 值发生变化时必须采样
                        if signal_last_values[symbol] != value:
                            should_sample = True
                            signal_last_values[symbol] = value
                        
                        # 2. 距离上次采样超过间隔时采样
                        elif current_time - last_sample_time >= sample_interval:
                            should_sample = True
                        
                        if should_sample:
                            result[symbol]['values'].append({
                                'time': current_time,
                                'value': value
                            })
                            last_sample_time = current_time
        
        # 压缩数据：合并连续相同值
        for symbol in result:
            result[symbol]['values'] = self._compress_values(result[symbol]['values'])
        
        return result
    
    def _compress_values(self, values):
        """压缩值序列 - 移除冗余的中间点"""
        if len(values) <= 2:
            return values
        
        compressed = [values[0]]
        
        for i in range(1, len(values) - 1):
            # 保留值变化点
            if values[i]['value'] != values[i-1]['value'] or values[i]['value'] != values[i+1]['value']:
                compressed.append(values[i])
        
        compressed.append(values[-1])
        return compressed
    
    def get_signal_list(self):
        """获取信号列表"""
        signal_list = []
        for symbol, info in self.signals.items():
            signal_list.append({
                'symbol': symbol,
                'name': info['name'],
                'full_name': info['full_name'],
                'type': info['type'],
                'width': info['width'],
                'value_count': len(info['values'])
            })
        return sorted(signal_list, key=lambda x: x['full_name'])
    
    def get_signal_data(self, symbols):
        """获取指定信号的数据"""
        data = {}
        for symbol in symbols:
            if symbol in self.signals:
                data[symbol] = {
                    'info': {
                        'name': self.signals[symbol]['name'],
                        'full_name': self.signals[symbol]['full_name'],
                        'width': self.signals[symbol]['width']
                    },
                    'values': self.signals[symbol]['values']
                }
        return data

# 全局变量
vcd_parser = None
wave_dir = None
wave_renderer = WaveformRenderer()

@app.route('/')
def index():
    """主页 - 流式加载模式（默认，适合大文件）"""
    return render_template('wave_viewer_streaming.html')

@app.route('/image')
def image_mode():
    """图片模式（一次性渲染）"""
    return render_template('wave_viewer_image.html')

@app.route('/canvas')
def canvas_mode():
    """Canvas 模式（适合小文件）"""
    return render_template('wave_viewer.html')

@app.route('/api/files')
def list_files():
    """列出可用的波形文件"""
    files = []
    if wave_dir and wave_dir.exists():
        for f in wave_dir.glob('*.vcd'):
            size_mb = f.stat().st_size / (1024 * 1024)
            files.append({
                'name': f.name,
                'path': str(f),
                'size_mb': round(size_mb, 2)
            })
    return jsonify(files)

@app.route('/api/load/<filename>')
def load_file(filename):
    """加载 VCD 文件（优化版 - 快速加载）"""
    global vcd_parser
    
    if not wave_dir:
        return jsonify({'error': '波形目录未设置'}), 400
    
    vcd_file = wave_dir / filename
    if not vcd_file.exists():
        return jsonify({'error': '文件不存在'}), 404
    
    try:
        vcd_parser = VCDParser(vcd_file)
        vcd_parser.parse_header()
        vcd_parser.parse_values_fast()  # 只解析时间范围，不加载所有数据
        
        return jsonify({
            'success': True,
            'info': {
                'date': vcd_parser.date,
                'version': vcd_parser.version,
                'timescale': vcd_parser.timescale,
                'max_time': vcd_parser.max_time,
                'signal_count': len(vcd_parser.signals),
                'file_size_mb': round(vcd_parser.file_size / (1024 * 1024), 2),
                'is_large_file': vcd_parser.is_large_file
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals')
def get_signals():
    """获取信号列表"""
    if not vcd_parser:
        return jsonify({'error': '未加载 VCD 文件'}), 400
    
    return jsonify(vcd_parser.get_signal_list())

@app.route('/api/data')
def get_data():
    """获取信号数据（支持时间范围和抽样）"""
    if not vcd_parser:
        return jsonify({'error': '未加载 VCD 文件'}), 400
    
    symbols = request.args.get('symbols', '').split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    
    if not symbols:
        return jsonify({'error': '未指定信号'}), 400
    
    # 获取时间范围参数
    start_time = int(request.args.get('start_time', 0))
    end_time = int(request.args.get('end_time', vcd_parser.max_time))
    max_points = int(request.args.get('max_points', 2000))
    
    # 使用优化的范围解析
    data = vcd_parser.parse_signal_range(symbols, start_time, end_time, max_points)
    
    return jsonify({
        'max_time': vcd_parser.max_time,
        'timescale': vcd_parser.timescale,
        'start_time': start_time,
        'end_time': end_time,
        'signals': data
    })

@app.route('/api/render')
def render_waveform():
    """渲染波形为图片（后端渲染）"""
    if not vcd_parser:
        return jsonify({'error': '未加载 VCD 文件'}), 400
    
    symbols = request.args.get('symbols', '').split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    
    if not symbols:
        return jsonify({'error': '未指定信号'}), 400
    
    # 获取参数
    start_time = int(request.args.get('start_time', 0))
    end_time = int(request.args.get('end_time', vcd_parser.max_time))
    width = int(request.args.get('width', 1600))
    max_points = int(request.args.get('max_points', 3000))
    
    try:
        # 加载信号数据
        data = vcd_parser.parse_signal_range(symbols, start_time, end_time, max_points)
        
        # 渲染为图片
        img_buf = wave_renderer.render_to_png(
            data,
            vcd_parser.max_time,
            vcd_parser.timescale,
            start_time,
            end_time
        )
        
        return send_file(img_buf, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': f'渲染失败: {str(e)}'}), 500

def main():
    parser = argparse.ArgumentParser(description='Web 波形查看器')
    parser.add_argument('--port', type=int, default=5000, help='Web 服务器端口')
    parser.add_argument('--host', default='0.0.0.0', help='Web 服务器地址')
    parser.add_argument('--wave-dir', default='.', help='波形文件目录')
    args = parser.parse_args()
    
    global wave_dir
    wave_dir = Path(args.wave_dir)
    
    if not wave_dir.exists():
        print(f"警告: 波形目录不存在: {wave_dir}")
        wave_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Web 波形查看器")
    print("=" * 60)
    print(f"服务器地址: http://{args.host}:{args.port}")
    print(f"波形目录: {wave_dir.absolute()}")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服务器\n")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
