#!/usr/bin/env python3
"""
简单的 HTTP 服务器 - 用于查看静态波形 HTML 文件
"""

import http.server
import socketserver
import webbrowser
import argparse
import os
from pathlib import Path
import socket

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义 HTTP 请求处理器"""
    
    def end_headers(self):
        # 添加 CORS 头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def log_message(self, format, *args):
        # 自定义日志格式
        print(f"[{self.log_date_time_string()}] {format % args}")

def find_free_port(start_port=8000, max_attempts=10):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def list_html_files(directory='.'):
    """列出目录中的 HTML 文件"""
    html_files = []
    for file in Path(directory).glob('*.html'):
        size_kb = file.stat().st_size / 1024
        html_files.append({
            'name': file.name,
            'size': size_kb,
            'path': file
        })
    return sorted(html_files, key=lambda x: x['path'].stat().st_mtime, reverse=True)

def generate_index_page(html_files):
    """生成索引页面"""
    
    file_list_html = ""
    for i, file in enumerate(html_files, 1):
        file_list_html += f"""
        <tr>
            <td>{i}</td>
            <td><a href="{file['name']}">{file['name']}</a></td>
            <td>{file['size']:.2f} KB</td>
            <td>
                <a href="{file['name']}" class="btn">查看</a>
            </td>
        </tr>
        """
    
    if not file_list_html:
        file_list_html = """
        <tr>
            <td colspan="4" style="text-align: center; color: #858585;">
                未找到波形 HTML 文件<br>
                <small>运行 ./view_wave.sh 生成波形文件</small>
            </td>
        </tr>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>波形文件列表</title>
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
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: #2d2d30;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #3e3e42;
            text-align: center;
        }}
        
        .header h1 {{
            color: #4ec9b0;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #858585;
            font-size: 16px;
        }}
        
        .content {{
            background: #2d2d30;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #3e3e42;
        }}
        
        .content h2 {{
            color: #4ec9b0;
            margin-bottom: 20px;
            font-size: 24px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #3e3e42;
        }}
        
        th {{
            background: #252526;
            color: #858585;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
        }}
        
        td {{
            color: #d4d4d4;
            font-size: 14px;
        }}
        
        tr:hover {{
            background: #252526;
        }}
        
        a {{
            color: #4ec9b0;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .btn {{
            display: inline-block;
            padding: 6px 12px;
            background: #007acc;
            color: white;
            border-radius: 4px;
            font-size: 13px;
            text-decoration: none;
        }}
        
        .btn:hover {{
            background: #005a9e;
            text-decoration: none;
        }}
        
        .info-box {{
            background: #252526;
            padding: 20px;
            border-radius: 4px;
            margin-top: 30px;
            border-left: 4px solid #4ec9b0;
        }}
        
        .info-box h3 {{
            color: #4ec9b0;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        
        .info-box p {{
            color: #858585;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .info-box code {{
            background: #1e1e1e;
            padding: 2px 6px;
            border-radius: 3px;
            color: #4ec9b0;
            font-family: monospace;
        }}
        
        .footer {{
            text-align: center;
            color: #858585;
            margin-top: 30px;
            padding: 20px;
            font-size: 14px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 10px;
            background: #4ec9b0;
            color: #1e1e1e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌊 波形文件查看器 <span class="badge">HTTP 服务</span></h1>
            <p>简单的本地 HTTP 服务器，用于查看静态波形 HTML 文件</p>
        </div>
        
        <div class="content">
            <h2>可用的波形文件</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px;">#</th>
                        <th>文件名</th>
                        <th style="width: 120px;">大小</th>
                        <th style="width: 100px;">操作</th>
                    </tr>
                </thead>
                <tbody>
                    {file_list_html}
                </tbody>
            </table>
            
            <div class="info-box">
                <h3>💡 使用提示</h3>
                <p>1. 点击文件名或"查看"按钮打开波形页面</p>
                <p>2. 生成新的波形文件: <code>./view_wave.sh</code></p>
                <p>3. 刷新此页面查看新文件</p>
                <p>4. 按 Ctrl+C 停止服务器</p>
            </div>
        </div>
        
        <div class="footer">
            <p>本地 HTTP 服务器 | Python {os.sys.version.split()[0]}</p>
            <p style="margin-top: 5px; font-size: 12px;">
                提示: 此服务器仅用于本地开发，不要在生产环境使用
            </p>
        </div>
    </div>
    
    <script>
        // 自动刷新文件列表（每 30 秒）
        setTimeout(() => {{
            location.reload();
        }}, 30000);
    </script>
</body>
</html>
"""
    return html_content

def start_server(port=8000, directory='.', open_browser=True):
    """启动 HTTP 服务器"""
    
    # 切换到指定目录
    os.chdir(directory)
    
    # 查找可用端口
    if port is None:
        port = find_free_port()
        if port is None:
            print("❌ 错误: 无法找到可用端口")
            return 1
    
    # 生成索引页面
    html_files = list_html_files('.')
    index_content = generate_index_page(html_files)
    
    # 写入 index.html
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    # 创建服务器
    handler = CustomHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print("=" * 70)
            print("🌐 HTTP 服务器已启动")
            print("=" * 70)
            print(f"服务地址: http://localhost:{port}")
            print(f"服务目录: {Path.cwd()}")
            print(f"波形文件: {len(html_files)} 个")
            print("=" * 70)
            print()
            
            if html_files:
                print("可用文件:")
                for i, file in enumerate(html_files[:5], 1):
                    print(f"  {i}. {file['name']} ({file['size']:.2f} KB)")
                if len(html_files) > 5:
                    print(f"  ... 还有 {len(html_files) - 5} 个文件")
                print()
            
            print("访问方式:")
            print(f"  1. 浏览器打开: http://localhost:{port}")
            print(f"  2. 或直接访问文件: http://localhost:{port}/waveform_post_syn.html")
            print()
            print("按 Ctrl+C 停止服务器")
            print()
            
            # 自动打开浏览器
            if open_browser:
                url = f"http://localhost:{port}"
                print(f"正在打开浏览器: {url}")
                try:
                    webbrowser.open(url)
                except:
                    print("无法自动打开浏览器，请手动访问上述地址")
                print()
            
            # 启动服务器
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("服务器已停止")
        print("=" * 70)
        return 0
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 错误: 端口 {port} 已被占用")
            print(f"提示: 尝试使用其他端口: python3 serve_wave.py -p {port + 1}")
        else:
            print(f"❌ 错误: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='启动简单的 HTTP 服务器查看波形文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认端口 8000
  python3 serve_wave.py
  
  # 指定端口
  python3 serve_wave.py -p 8080
  
  # 指定目录
  python3 serve_wave.py -d /path/to/waves
  
  # 不自动打开浏览器
  python3 serve_wave.py --no-browser
  
访问:
  浏览器打开 http://localhost:8000
        """
    )
    
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8000,
        help='HTTP 服务器端口 (默认: 8000)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='.',
        help='服务目录 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='不自动打开浏览器'
    )
    
    args = parser.parse_args()
    
    return start_server(
        port=args.port,
        directory=args.directory,
        open_browser=not args.no_browser
    )

if __name__ == '__main__':
    exit(main())
