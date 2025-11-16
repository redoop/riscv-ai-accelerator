#!/usr/bin/env python3
"""
波形渲染器 - 在后端生成波形图片
"""

import io
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

class WaveformRenderer:
    """波形渲染器 - 生成波形图片"""
    
    def __init__(self, width=1600, height=800, dpi=100):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.signal_height = 60  # 每个信号的高度（像素）
        
    def render_to_png(self, signal_data, max_time, timescale, start_time=0, end_time=None):
        """渲染波形到 PNG 图片"""
        
        if end_time is None:
            end_time = max_time
        
        num_signals = len(signal_data)
        if num_signals == 0:
            return self._create_empty_image()
        
        # 计算图片尺寸
        fig_height = max(4, num_signals * 0.8 + 1)
        fig_width = self.width / self.dpi
        
        # 创建图形
        fig, axes = plt.subplots(
            num_signals, 1,
            figsize=(fig_width, fig_height),
            facecolor='#1e1e1e',
            dpi=self.dpi
        )
        
        if num_signals == 1:
            axes = [axes]
        
        # 设置整体样式
        fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.08, hspace=0.3)
        
        # 渲染每个信号
        for idx, (symbol, data) in enumerate(signal_data.items()):
            ax = axes[idx]
            self._render_signal(ax, data, start_time, end_time, max_time, timescale)
        
        # 保存到字节流
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1e1e1e', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    
    def _render_signal(self, ax, signal_data, start_time, end_time, max_time, timescale):
        """渲染单个信号"""
        
        info = signal_data['info']
        values = signal_data['values']
        
        # 设置背景和边框
        ax.set_facecolor('#1e1e1e')
        for spine in ax.spines.values():
            spine.set_color('#3e3e42')
            spine.set_linewidth(0.5)
        
        # 设置标题（信号名称）
        ax.set_title(
            info['full_name'],
            color='#d4d4d4',
            fontsize=10,
            loc='left',
            pad=5
        )
        
        # 设置坐标轴
        ax.set_xlim(start_time, end_time)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        
        # X 轴样式
        ax.tick_params(axis='x', colors='#858585', labelsize=8)
        ax.set_xlabel(f'Time ({timescale})', color='#858585', fontsize=8)
        
        # 网格
        ax.grid(True, axis='x', color='#3e3e42', linestyle='--', linewidth=0.3, alpha=0.5)
        
        if len(values) == 0:
            ax.text(
                (start_time + end_time) / 2, 0.5,
                'No data',
                ha='center', va='center',
                color='#858585', fontsize=10
            )
            return
        
        # 根据信号宽度选择渲染方式
        if info['width'] == 1:
            self._render_digital_signal(ax, values, start_time, end_time)
        else:
            self._render_bus_signal(ax, values, start_time, end_time)
    
    def _render_digital_signal(self, ax, values, start_time, end_time):
        """渲染数字信号（单比特）"""
        
        times = []
        levels = []
        
        # 构建波形数据
        for i, v in enumerate(values):
            t = v['time']
            if t < start_time:
                continue
            if t > end_time:
                break
            
            val = v['value']
            level = 1.0 if val == '1' else 0.0 if val == '0' else 0.5
            
            # 添加转换点
            if times:
                times.append(t)
                levels.append(levels[-1])
            
            times.append(t)
            levels.append(level)
        
        # 延伸到结束
        if times:
            times.append(end_time)
            levels.append(levels[-1])
        
        # 绘制波形
        if times:
            ax.plot(times, levels, color='#4ec9b0', linewidth=1.5, drawstyle='steps-post')
            ax.fill_between(times, 0, levels, color='#4ec9b0', alpha=0.1, step='post')
    
    def _render_bus_signal(self, ax, values, start_time, end_time):
        """渲染总线信号（多比特）"""
        
        for i, v in enumerate(values):
            t1 = v['time']
            if t1 > end_time:
                break
            
            # 确定结束时间
            if i < len(values) - 1:
                t2 = min(values[i + 1]['time'], end_time)
            else:
                t2 = end_time
            
            if t2 < start_time:
                continue
            
            t1 = max(t1, start_time)
            
            # 绘制总线形状
            width = t2 - t1
            if width > 0:
                # 梯形
                polygon = patches.Polygon(
                    [
                        (t1, 0.2), (t1 + width * 0.05, 0.3),
                        (t2 - width * 0.05, 0.3), (t2, 0.2),
                        (t2, 0.8), (t2 - width * 0.05, 0.7),
                        (t1 + width * 0.05, 0.7), (t1, 0.8)
                    ],
                    closed=True,
                    edgecolor='#4ec9b0',
                    facecolor='#1e1e1e',
                    linewidth=1.5
                )
                ax.add_patch(polygon)
                
                # 显示值（十六进制）
                if width > (end_time - start_time) * 0.02:  # 只在足够宽时显示
                    try:
                        hex_val = hex(int(v['value'], 2))[2:].upper()
                        ax.text(
                            (t1 + t2) / 2, 0.5,
                            f'0x{hex_val}',
                            ha='center', va='center',
                            color='#d4d4d4',
                            fontsize=8,
                            family='monospace'
                        )
                    except:
                        # 如果转换失败，显示原始值
                        if len(v['value']) < 10:
                            ax.text(
                                (t1 + t2) / 2, 0.5,
                                v['value'],
                                ha='center', va='center',
                                color='#d4d4d4',
                                fontsize=7,
                                family='monospace'
                            )
    
    def _create_empty_image(self):
        """创建空白图片"""
        fig, ax = plt.subplots(figsize=(self.width/self.dpi, 4), facecolor='#1e1e1e', dpi=self.dpi)
        ax.set_facecolor('#1e1e1e')
        ax.text(
            0.5, 0.5,
            'No signals selected',
            ha='center', va='center',
            color='#858585',
            fontsize=14,
            transform=ax.transAxes
        )
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1e1e1e')
        buf.seek(0)
        plt.close(fig)
        
        return buf
