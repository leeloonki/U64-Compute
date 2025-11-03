# U64-Compute: High-Performance mod 2^64 Ring Arithmetic Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/leeloonki/U64-Compute)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A NumPy-like C++ library designed for high-performance `uint64_t` operations with automatic SIMD optimization and optional CUDA acceleration. Built specifically for mod 2^64 ring arithmetic operations with transparent performance optimizations.

## 🚀 Key Features

- **🔥 High Performance**: Automatic SIMD optimization (SSE2/AVX2/AVX-512/NEON) + OpenMP parallelization
- **🎯 NumPy-like API**: Familiar interface for NumPy users with C++ performance
- **⚡ Device Flexibility**: PyTorch-style device management (`.cpu()`, `.cuda()`, `.to()`)
- **🌐 Cross-Platform**: Support for x86-64 and ARM64 architectures (Linux/macOS)
- **🔧 Auto-Optimization**: Runtime selection of optimal algorithms based on data size and hardware
- **🎮 Optional CUDA**: GPU acceleration with automatic CPU fallback
- **📊 Built-in Profiling**: Performance monitoring and benchmarking tools

## 📋 System Requirements

### Requirements
- **Compiler**: GCC 9.4.0+
- **C++ Standard**: C++17
- **Architecture**: x86-64 or ARM64

### Optional Requirements
- **CUDA**: NVIDIA CUDA Toolkit 12.x+ (for GPU acceleration)
- **OpenMP**: For parallel processing (auto-detected)

## 🛠️ Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/leeloonki/U64-Compute.git
cd U64-Compute

# Build the library
make

# Run tests
make test

# Install system-wide (optional)
sudo make install
```

### Build Options

```bash
make              # Default build with auto-optimization
make debug        # Debug build with symbols
make release      # Optimized release build
make libs         # Build only static/shared libraries
make test         # Build and run functional tests
make clean        # Clean build files
make install      # Install system-wide (requires sudo)
make uninstall    # Remove system-wide installation
make simdinfo     # Show detailed SIMD detection information
make simd-tool    # Build and run SIMD information tool
make sysinfo      # Show complete system information
make help         # Show all available targets
```

### Advanced Build Configuration

The Makefile automatically detects:
- **GCC compiler**: Auto-detects available GCC versions (gcc-15, gcc-14, etc.)
- **CUDA availability**: Enables GPU support if NVCC is found
- **Platform-specific optimizations**: macOS vs Linux build flags

## 🚀 Quick Start

### Basic Array Operations

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // Create arrays (NumPy-style)
    auto a = ops::np::arange(1, 11);  // [1, 2, 3, ..., 10]
    auto b = ops::np::ones({10});     // [1, 1, 1, ..., 1]
    
    // Basic operations with automatic SIMD + OpenMP optimization
    auto sum_result = a + b;          // Element-wise addition
    auto mul_result = a * 2;          // Scalar multiplication
    
    // Advanced operations
    auto matrix = ops::np::full({3, 4}, 42);
    auto result = ops::matmul(matrix, matrix.T());
    
    return 0;
}
```

### Device Management (PyTorch-style)

```cpp
// CPU operations (default)
auto cpu_array = ops::np::zeros({1000000});

// GPU operations (if CUDA available)
if (cuda::is_available()) {
    auto gpu_array = cpu_array.cuda();    // Transfer to GPU
    auto gpu_result = gpu_array * 2;      // GPU computation
    auto result = gpu_result.cpu();       // Transfer back to CPU
}

// Device-agnostic operations
auto array = ops::np::ones({1000});
auto gpu_array = array.to("cuda");       // String-based device selection
auto cpu_array = array.to("cpu");
```

### Performance Monitoring

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // Check system capabilities
    std::cout << "SIMD support: " << ops::PerformanceConfig::get_simd_info() << std::endl;
    std::cout << "OpenMP threads: " << ops::PerformanceConfig::get_num_threads() << std::endl;
    std::cout << "CUDA available: " << cuda::is_available() << std::endl;
    
    // Performance test
    auto large_array = ops::np::arange(0, 1000000);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = large_array * 2;
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Processing 1M elements took: " << duration.count() << " μs" << std::endl;
    
    return 0;
}
```

## 📖 Documentation

- **[API Reference](docs/API.md)** - Complete API documentation

## 🏗️ Architecture

### Core Components

```
U64-Compute/
├── Makefile                      # Build system
├── README.md                     # Project documentation
├── benchmark.cpp                 # Performance benchmarking tool
├── functional_test.cpp           # Usage examples and tests
├── docs/
│   └── API.md                    # Complete API documentation
├── include/
│   ├── uint64_compute.h          # Main API header
│   ├── uint64_compute_config.h   # Platform and compiler configuration
│   ├── simd_detect.h             # SIMD capability detection
│   ├── simd_dispatcher.h         # Runtime SIMD optimization
│   ├── simd_x86.h                # x86-64 SIMD interface
│   ├── simd_arm.h                # ARM64 NEON interface
│   ├── hardware_detector.h       # System hardware analysis
│   └── cuda_detect.h             # CUDA device management
├── src/
│   ├── uint64_compute_impl.cpp   # Core array operations
│   ├── uint64_array_impl.cpp     # UInt64Array class implementation
│   ├── simd_detect.cpp           # SIMD detection implementation
│   ├── simd_dispatcher.cpp       # SIMD dispatch logic
│   ├── simd_x86.cpp              # x86-64 SIMD implementations
│   ├── simd_arm.cpp              # ARM64 NEON implementations
│   ├── hardware_detector.cpp     # Hardware detection implementation
│   ├── cuda_detect.cpp           # CUDA detection implementation
│   └── uint64_compute_cuda_kernels.cu  # CUDA GPU kernels
├── bin/                          # Built libraries and executables
└── obj/                          # Object files (build artifacts)
```

  - functional_test = "这个库能工作吗？" ✅
  - benchmark = "这个库跑得有多快？" 🚀

### Performance Architecture

1. **Auto-Detection Layer**: Runtime detection of CPU features and optimal SIMD instructions
2. **SIMD Dispatch Layer**: Automatic selection between Scalar/SSE2/AVX2/AVX-512/NEON implementations
3. **Parallel Processing**: OpenMP integration for multi-core utilization
4. **Device Management**: Transparent CPU/GPU data movement and computation
5. **Memory Optimization**: SIMD-aligned memory allocation for maximum throughput

## 📊 Performance

### Benchmarks

**Element-wise Operations (1M uint64_t elements)**:

*Apple M2 (ARM64 NEON, 8 cores)*:
| Operation | Time (ms) | Throughput (M ops/sec) |
|-----------|-----------|------------------------|
| Addition | 3.13 | 319.55 |
| Multiplication | 0.87 | 1152.60 |

*Intel Xeon Platinum 8352Y (x86-64 AVX-512, 32 cores, 64 threads)*:
| Operation | Time (ms) | Throughput (M ops/sec) |
|-----------|-----------|------------------------|
| Addition | 20.23 | 49.43 |
| Multiplication | 9.97 | 100.27 |

**Matrix Multiplication**:

*Apple M2*:
| Matrix Size | Time (ms) | GFLOPS |
|-------------|-----------|--------|
| 256x256 | 3.09 | 10.84 |
| 512x512 | 30.70 | 8.74 |
| 1024x1024 | 547.10 | 3.93 |
| 2048x2048 | 19381.64 | 0.89 |

*Intel Xeon Platinum 8352Y*:
| Matrix Size | Time (ms) | GFLOPS |
|-------------|-----------|--------|
| 256x256 | 23.78 | 1.41 |
| 512x512 | 473.68 | 0.57 |
| 1024x1024 | 3153.19 | 0.68 |

**CUDA Performance** (NVIDIA RTX A6000):

*Multi-GPU Performance (1M elements)*:
| GPU Device | Time (ms) | Throughput (M ops/sec) |
|------------|-----------|------------------------|
| GPU 0 | 26.97 | 37.07 |
| GPU 1 | 24.55 | 40.73 |
| GPU 2 | 24.79 | 40.35 |

*CUDA Matrix Multiplication Performance*:
| Matrix Size | GPU Time (ms) | GPU GFLOPS | CPU Time (ms) | CPU GFLOPS | Speedup |
|-------------|---------------|------------|---------------|------------|---------|
| 256x256 | 0.89 | 37.55 | 23.78 | 1.41 | 26.61x |
| 512x512 | 2.03 | 132.51 | 473.68 | 0.57 | 233.82x |
| 1024x1024 | 8.60 | 249.78 | 3153.19 | 0.68 | 366.76x |

*Large Array Performance (10M elements, GPU 0)*:
| Operation | Time (ms) | Throughput (M ops/sec) |
|-----------|-----------|------------------------|
| Addition (with transfer) | 172.89 | 57.84 |

*Server: Intel Xeon Platinum 8352Y @ 2.20GHz, 3x NVIDIA RTX A6000 GPUs, data includes CPU↔GPU transfer overhead*

### SIMD Support Matrix

| Architecture | SIMD Instructions | Vector Width | Status |
|--------------|-------------------|--------------|--------|
| x86-64 | SSE2, SSE4.1, AVX, AVX2, AVX-512F | 128-512 bits | ✅ Supported |
| ARM64 | NEON, SVE | 128+ bits | ✅ Supported |

**Tested Configurations**:
- **Apple M2**: ARM64 NEON, 128-bit vectors, 8 cores
- **Intel Xeon Platinum 8352Y**: x86-64 AVX-512F, 512-bit vectors, 32 cores/64 threads

## 🔧 Configuration

### Runtime Configuration

```cpp
// OpenMP thread control
ops::PerformanceConfig::set_num_threads(8);

// Set parallel processing threshold
ops::PerformanceConfig::set_parallel_threshold(1024);

// CUDA device selection
cuda::set_device(0);  // Use GPU 0
```

### Build Configuration

The Makefile automatically detects:
- **GCC compiler**: Auto-detects available GCC versions (gcc-15, gcc-14, etc.)
- **CUDA availability**: Enables GPU support if NVCC is found
- **Platform-specific optimizations**: macOS vs Linux build flags

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo if you find it useful!** ⭐