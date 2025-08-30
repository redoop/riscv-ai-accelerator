# Makefile for PyTorch RISC-V AI Backend

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++14 -fPIC -O3 -Wall -Wextra
LDFLAGS = -shared

# Python and PyTorch settings
PYTHON = python3
PYTHON_CONFIG = python3-config
TORCH_CXXFLAGS = $(shell $(PYTHON) -c "import torch; print(' '.join(torch.utils.cpp_extension.include_paths()))" | sed 's/^/-I/' | sed 's/ / -I/g')
TORCH_LDFLAGS = $(shell $(PYTHON) -c "import torch; print(' '.join(torch.utils.cpp_extension.library_paths()))" | sed 's/^/-L/' | sed 's/ / -L/g')

# pybind11 settings
PYBIND11_CXXFLAGS = $(shell $(PYTHON) -m pybind11 --includes)

# RISC-V AI library settings
RISCV_AI_INCLUDE = -I../../lib -I../../compiler
RISCV_AI_LIBS = -L../../lib -ltpu

# Source files
SOURCES = riscv_ai_backend.cpp
TARGET = riscv_ai_backend$(shell $(PYTHON_CONFIG) --extension-suffix)

# Build targets
.PHONY: all clean install test benchmark

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(PYBIND11_CXXFLAGS) $(TORCH_CXXFLAGS) $(RISCV_AI_INCLUDE) \
		$(SOURCES) -o $@ $(LDFLAGS) $(TORCH_LDFLAGS) $(RISCV_AI_LIBS) -ltorch -ltorch_cpu

# Alternative build using setuptools
build-setuptools:
	$(PYTHON) setup.py build_ext --inplace

# Install the package
install:
	$(PYTHON) setup.py install

# Install in development mode
install-dev:
	$(PYTHON) setup.py develop

# Run tests
test: $(TARGET)
	$(PYTHON) -m pytest test_pytorch_integration.py -v

# Run integration tests
test-integration: $(TARGET)
	$(PYTHON) test_pytorch_integration.py

# Run benchmarks
benchmark: $(TARGET)
	$(PYTHON) -c "from test_pytorch_integration import run_performance_tests; run_performance_tests()"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -f *.pyc
	rm -f *.so

# Format code
format:
	black *.py
	clang-format -i *.cpp *.h

# Lint code
lint:
	flake8 *.py
	cppcheck --enable=all --std=c++14 *.cpp

# Create distribution package
dist:
	$(PYTHON) setup.py sdist bdist_wheel

# Install dependencies
deps:
	pip install torch torchvision numpy pybind11 pytest pytest-cov black flake8

# Install optional dependencies
deps-optional:
	pip install onnx onnx2torch

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the extension module"
	@echo "  build-setuptools - Build using setuptools"
	@echo "  install          - Install the package"
	@echo "  install-dev      - Install in development mode"
	@echo "  test             - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  clean            - Clean build artifacts"
	@echo "  format           - Format code"
	@echo "  lint             - Lint code"
	@echo "  dist             - Create distribution package"
	@echo "  deps             - Install dependencies"
	@echo "  deps-optional    - Install optional dependencies"
	@echo "  help             - Show this help message"

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Profile build
profile: CXXFLAGS += -pg
profile: $(TARGET)

# Check if dependencies are available
check-deps:
	@$(PYTHON) -c "import torch; print('PyTorch version:', torch.__version__)"
	@$(PYTHON) -c "import pybind11; print('pybind11 version:', pybind11.__version__)"
	@$(PYTHON) -c "import numpy; print('NumPy version:', numpy.__version__)"

# Show build configuration
show-config:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "TORCH_CXXFLAGS: $(TORCH_CXXFLAGS)"
	@echo "PYBIND11_CXXFLAGS: $(PYBIND11_CXXFLAGS)"
	@echo "RISCV_AI_INCLUDE: $(RISCV_AI_INCLUDE)"
	@echo "RISCV_AI_LIBS: $(RISCV_AI_LIBS)"
	@echo "TARGET: $(TARGET)"