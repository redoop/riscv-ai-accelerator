"""
RISC-V AI Backend for macOS
提供RISC-V AI加速器的macOS仿真支持
"""

# 导入仿真后端
from .riscv_ai_backend_macos import *

# 版本信息
__version__ = "1.0.0-macos-simulator"
__author__ = "RISC-V AI Team"
__description__ = "RISC-V AI Accelerator macOS Simulator"

# 自动初始化标志
_auto_initialize = True

def set_auto_initialize(enabled: bool):
    """设置是否自动初始化"""
    global _auto_initialize
    _auto_initialize = enabled

# 尝试自动初始化
if _auto_initialize:
    try:
        initialize()
    except Exception as e:
        import warnings
        warnings.warn(f"自动初始化失败: {e}")
