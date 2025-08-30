"""
Setup script for PyTorch RISC-V AI backend
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension
import os
import sys

# Get PyTorch include directories
torch_include_dirs = [
    torch.utils.cpp_extension.include_paths()[0],
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'torch', 'csrc', 'api', 'include'),
]

# Get RISC-V AI library paths
riscv_ai_include = "../../lib"
riscv_ai_lib = "../../lib"
compiler_include = "../../compiler"

# Define extension module
ext_modules = [
    Pybind11Extension(
        "riscv_ai_backend",
        sources=[
            "riscv_ai_backend.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            riscv_ai_include,
            compiler_include,
        ] + torch_include_dirs,
        libraries=["tpu"],
        library_dirs=[riscv_ai_lib],
        language="c++",
        cxx_std=14,
        define_macros=[
            ("TORCH_EXTENSION_NAME", "riscv_ai_backend"),
            ("TORCH_API_INCLUDE_EXTENSION_H", None),
        ],
    ),
]

setup(
    name="riscv_ai_pytorch",
    version="0.1.0",
    description="PyTorch backend for RISC-V AI accelerator",
    author="RISC-V AI Team",
    author_email="team@riscv-ai.org",
    url="https://github.com/riscv-ai/accelerator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "onnx": [
            "onnx>=1.9.0",
            "onnx2torch>=1.5.0",
        ],
    },
    py_modules=[
        "model_optimizer",
        "runtime",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)