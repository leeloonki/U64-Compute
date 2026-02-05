# U64-Compute: High-Performance mod 2^64 Ring Arithmetic Library

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)


A NumPy-like C++ library designed for high-performance `uint64_t` operations with automatic SIMD optimization and optional CUDA acceleration. Built specifically for mod 2^64 ring arithmetic operations with transparent performance optimizations.

## Key Features

- **High Performance**: Automatic SIMD optimization (SSE2/AVX2/AVX-512/NEON) + OpenMP parallelization
- **NumPy-like API**: Familiar interface for NumPy users with C++ performance
- **Device Flexibility**: PyTorch-style device management (`.cpu()`, `.cuda()`, `.to()`)
- **Cross-Platform**: Support for x86-64 and ARM64 architectures (Linux/macOS)
- **Auto-Optimization**: Runtime selection of optimal algorithms based on data size and hardware
- **Optional CUDA**: GPU acceleration with automatic CPU fallback
- **Built-in Profiling**: Performance monitoring and benchmarking tools